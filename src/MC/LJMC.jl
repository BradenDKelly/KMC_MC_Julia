"""
Lennard-Jones NVT Metropolis Monte Carlo simulation (GPU-friendly SoA layout).
"""

using Random

"""
    LJParams

Parameters for Lennard-Jones Monte Carlo simulation.
"""
struct LJParams
    ϵ::Float64              # LJ energy parameter
    σ::Float64              # LJ size parameter
    rc::Float64             # Cutoff distance
    rc_sq::Float64          # Cutoff squared
    L::Float64              # Box length
    N::Int                  # Number of particles
    β::Float64              # Inverse temperature (1/(kB*T))
    max_disp::Float64       # Maximum displacement for trial moves
end

"""
    LJParams(ϵ, σ, rc, L, N, T, max_disp)

Construct LJ parameters with temperature `T` (kB*T in units of ϵ).
"""
function LJParams(ϵ::Float64, σ::Float64, rc::Float64, L::Float64, N::Int, T::Float64, max_disp::Float64)
    rc_sq = rc * rc
    β = 1.0 / T
    return LJParams(ϵ, σ, rc, rc_sq, L, N, β, max_disp)
end

"""
    LJState

State of the Lennard-Jones Monte Carlo simulation.
Uses SoA (Structure of Arrays) layout for GPU compatibility.
"""
mutable struct LJState{B <: AbstractBackend}
    backend::B                 # Backend (CPU, GPU, etc.)
    posx::Vector{Float64}      # X positions (SoA)
    posy::Vector{Float64}      # Y positions (SoA)
    posz::Vector{Float64}      # Z positions (SoA)
    cell_bins::CellBins        # Array-based cell binning
    rng::MersenneTwister       # Random number generator
    energy::Float64            # Current total energy
end

"""
    lj_potential(r_sq, params)

Compute Lennard-Jones potential: 4ϵ[(σ/r)^12 - (σ/r)^6] at squared distance `r_sq`.
Returns 0.0 if r >= rc.
"""
@inline function lj_potential(r_sq::Float64, params::LJParams)
    if r_sq >= params.rc_sq || r_sq <= 0.0
        return 0.0
    end
    @fastmath begin
        σ2_over_r2 = (params.σ * params.σ) / r_sq
        σ6_over_r6 = σ2_over_r2 * σ2_over_r2 * σ2_over_r2
        σ12_over_r12 = σ6_over_r6 * σ6_over_r6
        return 4.0 * params.ϵ * (σ12_over_r12 - σ6_over_r6)
    end
end

"""
    local_energy(backend, posx, posy, posz, particle_idx, cell_bins, params)

Compute the local energy (energy of interactions) for particle `particle_idx`.
Zero-allocation inner loop.
Dispatches on backend type.
"""
function local_energy(
    backend::AbstractBackend,
    posx::AbstractVector{<:Real},
    posy::AbstractVector{<:Real},
    posz::AbstractVector{<:Real},
    particle_idx::Int,
    cell_bins::CellBins,
    params::LJParams
)
    return _local_energy_cpu(posx, posy, posz, particle_idx, cell_bins, params)
end

"""
    _local_energy_cpu(posx, posy, posz, particle_idx, cell_bins, params)

CPU implementation of local_energy (internal).
GPU-compatible: no dictionaries, no IO, no allocations in hot paths.
"""
function _local_energy_cpu(
    posx::AbstractVector{<:Real},
    posy::AbstractVector{<:Real},
    posz::AbstractVector{<:Real},
    particle_idx::Int,
    cell_bins::CellBins,
    params::LJParams
)
    energy = 0.0
    L = cell_bins.L
    rc_sq = cell_bins.rc * cell_bins.rc
    
    # Get particle's cell
    cell_idx = cell_bins.particle_cell[particle_idx]
    
    # Get particle position
    px = posx[particle_idx]
    py = posy[particle_idx]
    pz = posz[particle_idx]
    
    # Iterate over neighbor cells
    for neighbor_cell_idx in iterate_neighbor_cells(cell_idx, cell_bins.ncell)
        # Iterate over particles in this neighbor cell
        for idx in iterate_particles_in_cell(cell_bins, neighbor_cell_idx)
            neighbor_idx = cell_bins.cell_particles[idx]
            
            if neighbor_idx != particle_idx
                # Compute distance vector components (no allocation)
                dr_x = posx[neighbor_idx] - px
                dr_y = posy[neighbor_idx] - py
                dr_z = posz[neighbor_idx] - pz
                
                # Apply minimum image convention (pure scalar ops)
                dr_x = dr_x - L * floor(dr_x / L + 0.5)
                dr_y = dr_y - L * floor(dr_y / L + 0.5)
                dr_z = dr_z - L * floor(dr_z / L + 0.5)
                
                r_sq = dr_x * dr_x + dr_y * dr_y + dr_z * dr_z
                
                if r_sq < rc_sq && r_sq > 0.0
                    energy += lj_potential(r_sq, params)
                end
            end
        end
    end
    return energy
end

"""
    total_energy(state, params)

Compute total energy of the system (O(N) operation for debugging).
"""
function total_energy(state::LJState, params::LJParams)
    energy = 0.0
    posx = state.posx
    posy = state.posy
    posz = state.posz
    N = params.N
    L = params.L
    
    # Compute energy by summing over all pairs
    @inbounds for i in 1:N
        for j in (i+1):N
            # Compute distance vector
            dr_x = posx[j] - posx[i]
            dr_y = posy[j] - posy[i]
            dr_z = posz[j] - posz[i]
            
            # Apply minimum image convention (pure scalar ops)
            dr_x = dr_x - L * floor(dr_x / L + 0.5)
            dr_y = dr_y - L * floor(dr_y / L + 0.5)
            dr_z = dr_z - L * floor(dr_z / L + 0.5)
            
            r_sq = dr_x * dr_x + dr_y * dr_y + dr_z * dr_z
            
            energy += lj_potential(r_sq, params)
        end
    end
    
    return energy
end

"""
    init_lj_state(params; posx=nothing, posy=nothing, posz=nothing, seed=42)

Initialize LJ simulation state. If positions are not provided, uses a simple cubic lattice.
Accepts SoA format positions.
"""
function init_lj_state(
    params::LJParams;
    backend::AbstractBackend = CPU,
    posx::Union{AbstractVector{<:Real}, Nothing} = nothing,
    posy::Union{AbstractVector{<:Real}, Nothing} = nothing,
    posz::Union{AbstractVector{<:Real}, Nothing} = nothing,
    seed::Int = 42
)
    N = params.N
    L = params.L
    
    if posx === nothing || posy === nothing || posz === nothing
        # Create simple cubic lattice
        n_per_side = ceil(Int, cbrt(N))
        spacing = L / n_per_side
        
        posx = zeros(Float64, N)
        posy = zeros(Float64, N)
        posz = zeros(Float64, N)
        
        idx = 1
        @inbounds for k in 1:n_per_side
            for j in 1:n_per_side
                for i in 1:n_per_side
                    if idx > N
                        break
                    end
                    posx[idx] = (i - 0.5) * spacing
                    posy[idx] = (j - 0.5) * spacing
                    posz[idx] = (k - 0.5) * spacing
                    idx += 1
                end
                if idx > N
                    break
                end
            end
            if idx > N
                break
            end
        end
    else
        # Make copies to avoid modifying the original
        posx = copy(posx)
        posy = copy(posy)
        posz = copy(posz)
    end
    
    # Wrap all positions to [0, L)
    wrap_soa!(posx, posy, posz, L)
    
    # Build cell bins
    cell_bins = CellBins(N, L, params.rc)
    build_cells!(backend, cell_bins, posx, posy, posz)
    
    # Initialize RNG
    rng = MersenneTwister(seed)
    
    # Compute initial energy
    temp_state = LJState(backend, posx, posy, posz, cell_bins, rng, 0.0)
    energy = total_energy(temp_state, params)
    
    return LJState(backend, posx, posy, posz, cell_bins, rng, energy)
end


"""
    mc_step!(backend, state, params)

Perform one Monte Carlo step (single-particle displacement move).
Returns (accepted, ΔE).
Zero-allocation inner loop.
Dispatches on backend type.
"""
function mc_step!(
    backend::AbstractBackend,
    state::LJState,
    params::LJParams
)
    return _mc_step_cpu!(state, params)
end

"""
    mc_step!(state, params)

Convenience method that uses state's backend.
"""
mc_step!(state::LJState, params::LJParams) = mc_step!(state.backend, state, params)

"""
    _mc_step_cpu!(state, params)

CPU implementation of mc_step! (internal).
GPU-compatible: no dictionaries, no IO, no allocations in hot paths.
"""
function _mc_step_cpu!(state::LJState, params::LJParams)
    N = params.N
    posx = state.posx
    posy = state.posy
    posz = state.posz
    L = params.L
    max_disp = params.max_disp
    
    # Select random particle
    particle_idx = rand(state.rng, 1:N)
    
    # Store old position
    @inbounds begin
        x_old = posx[particle_idx]
        y_old = posy[particle_idx]
        z_old = posz[particle_idx]
    end
    
    # Compute old local energy
    old_energy = local_energy(state.backend, posx, posy, posz, particle_idx, state.cell_bins, params)
    
    # Generate trial displacement
    dx = (2.0 * rand(state.rng) - 1.0) * max_disp
    dy = (2.0 * rand(state.rng) - 1.0) * max_disp
    dz = (2.0 * rand(state.rng) - 1.0) * max_disp
    
    # Apply trial move
    @inbounds begin
        posx[particle_idx] = x_old + dx
        posy[particle_idx] = y_old + dy
        posz[particle_idx] = z_old + dz
    end
    
    # Apply PBC (pure scalar ops)
    @inbounds begin
        x_new = posx[particle_idx]
        y_new = posy[particle_idx]
        z_new = posz[particle_idx]
        
        posx[particle_idx] = x_new - L * floor(x_new / L)
        posy[particle_idx] = y_new - L * floor(y_new / L)
        posz[particle_idx] = z_new - L * floor(z_new / L)
    end
    
    # Rebuild cell bins (needed when particle moves between cells)
    # For efficiency, we could update incrementally, but rebuild is simpler and still O(1) amortized
    build_cells!(state.backend, state.cell_bins, posx, posy, posz)
    
    # Compute new local energy
    new_energy = local_energy(state.backend, posx, posy, posz, particle_idx, state.cell_bins, params)
    
    # Compute energy change
    ΔE = new_energy - old_energy
    
    # Metropolis acceptance criterion
    accepted = false
    if ΔE <= 0.0 || rand(state.rng) < exp(-params.β * ΔE)
        accepted = true
        state.energy += ΔE
    else
        # Reject: restore old position
        @inbounds begin
            posx[particle_idx] = x_old
            posy[particle_idx] = y_old
            posz[particle_idx] = z_old
        end
        
        # Rebuild cell bins to restore state
        build_cells!(state.backend, state.cell_bins, posx, posy, posz)
    end
    
    return (accepted, ΔE)
end

"""
    run!(state, params; nsteps, sample_every=1)

Run Monte Carlo simulation for `nsteps` steps.
Returns (acceptance_rate, energies, [optional: pressure, g(r)]).
"""
function run!(
    state::LJState,
    params::LJParams;
    nsteps::Int,
    sample_every::Int = 1
)
    n_accepted = 0
    n_samples = div(nsteps, sample_every) + (nsteps % sample_every > 0 ? 1 : 0)
    energies = zeros(Float64, n_samples)
    
    sample_idx = 1
    for step in 1:nsteps
        accepted, ΔE = mc_step!(state, params)
        
        if accepted
            n_accepted += 1
        end
        
        # Sample observables
        if step % sample_every == 0
            energies[sample_idx] = state.energy
            sample_idx += 1
        end
    end
    
    acceptance_rate = n_accepted / nsteps
    return (acceptance_rate, energies)
end

"""
    ReplicaEnsemble

Ensemble of independent MC replicas for parallel execution.
"""
struct ReplicaEnsemble
    replicas::Vector{LJState}  # Vector of independent replica states
end

"""
    ReplicaEnsemble(params, R; base_seed=42)

Create an ensemble of `R` independent replicas, each with its own random seed.
Each replica is initialized independently with seed = base_seed + replica_id.
"""
function ReplicaEnsemble(params::LJParams, R::Int; base_seed::Int = 42)
    replicas = Vector{LJState}(undef, R)
    @inbounds for r in 1:R
        seed = base_seed + r
        replicas[r] = init_lj_state(params; seed=seed)
    end
    return ReplicaEnsemble(replicas)
end

"""
    ReplicaStats

Statistics for a single replica run.
"""
struct ReplicaStats
    replica_id::Int            # Replica identifier
    acceptance_rate::Float64   # Acceptance rate
    initial_energy::Float64    # Initial energy
    final_energy::Float64      # Final energy
    average_energy::Float64    # Average energy over samples
end

"""
    run_ensemble!(ensemble, params; nsteps, sample_every=1)

Run MC simulation for all replicas in parallel using threads.
Each replica runs independently and sequentially (correct Metropolis).
Returns Vector{ReplicaStats} with per-replica statistics.
"""
function run_ensemble!(
    ensemble::ReplicaEnsemble,
    params::LJParams;
    nsteps::Int,
    sample_every::Int = 1
)
    R = length(ensemble.replicas)
    stats = Vector{ReplicaStats}(undef, R)
    
    # Run replicas in parallel
    Base.Threads.@threads for r in 1:R
        state = ensemble.replicas[r]
        
        # Store initial energy
        initial_energy = state.energy
        
        # Run single-replica MC (sequential, correct Metropolis)
        acceptance_rate, energies = run!(state, params; nsteps=nsteps, sample_every=sample_every)
        
        # Compute statistics
        final_energy = state.energy
        average_energy = isempty(energies) ? final_energy : sum(energies) / length(energies)
        
        # Store statistics
        stats[r] = ReplicaStats(r, acceptance_rate, initial_energy, final_energy, average_energy)
    end
    
    return stats
end
