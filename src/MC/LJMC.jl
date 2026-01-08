"""
Lennard-Jones NVT Metropolis Monte Carlo simulation.
"""

using Random
using StaticArrays

"""
    LJParams

Parameters for Lennard-Jones Monte Carlo simulation.
"""
struct LJParams
    σ::Float64
    ϵ::Float64
    rc::Float64
    rc2::Float64
    β::Float64
    max_disp::Float64
end

"""
    LJState

State of the Lennard-Jones Monte Carlo simulation.
"""
mutable struct LJState
    N::Int
    L::Float64
    pos::Matrix{Float64}          # 3 x N, column = particle
    rng::Xoshiro                  # concrete RNG type
    cl::CellList                  # cell list
    scratch_dr::MVector{3,Float64} # reused displacement vector
    accepted::Int
    attempted::Int
end

"""
    lj_potential(r2, p::LJParams)

Compute Lennard-Jones potential: 4ϵ[(σ/r)^12 - (σ/r)^6] at squared distance `r2`.
Returns 0.0 if r2 >= rc2.
"""
@inline function lj_potential(r2::Float64, p::LJParams)
    if r2 >= p.rc2 || r2 <= 0.0
        return 0.0
    end
    σ2_over_r2 = (p.σ * p.σ) / r2
    σ6_over_r6 = σ2_over_r2 * σ2_over_r2 * σ2_over_r2
    σ12_over_r12 = σ6_over_r6 * σ6_over_r6
    return 4.0 * p.ϵ * (σ12_over_r12 - σ6_over_r6)
end

"""
    init_fcc(; N::Int=864, ρ::Float64=0.8, T::Float64=1.0, rc::Float64=2.5,
             max_disp::Float64=0.1, seed::Int=1234)

Initialize FCC lattice at density ρ and return (params::LJParams, st::LJState).
Requires N divisible by 4 and N/4 to be a perfect cube.
"""
function init_fcc(; N::Int=864, ρ::Float64=0.8, T::Float64=1.0, rc::Float64=2.5,
                  max_disp::Float64=0.1, seed::Int=1234)
    if N % 4 != 0
        throw(ArgumentError("N must be divisible by 4 for FCC lattice, got N=$N"))
    end
    
    n_uc = N ÷ 4  # number of unit cells
    nx = round(Int, cbrt(n_uc))
    if nx * nx * nx != n_uc
        throw(ArgumentError("N/4 must be a perfect cube for FCC lattice, got N=$N, N/4=$n_uc"))
    end
    
    # Compute box length from density
    V = N / ρ
    L = cbrt(V)
    
    # FCC lattice: 4 particles per unit cell at positions (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # Unit cell spacing
    a = L / nx
    
    # Allocate positions (3 x N)
    pos = zeros(Float64, 3, N)
    
    idx = 1
    @inbounds for k in 0:(nx-1)
        for j in 0:(nx-1)
            for i in 0:(nx-1)
                # Unit cell origin
                x0 = i * a
                y0 = j * a
                z0 = k * a
                
                # 4 particles in unit cell
                pos[1, idx] = x0
                pos[2, idx] = y0
                pos[3, idx] = z0
                idx += 1
                
                pos[1, idx] = x0 + 0.5 * a
                pos[2, idx] = y0 + 0.5 * a
                pos[3, idx] = z0
                idx += 1
                
                pos[1, idx] = x0 + 0.5 * a
                pos[2, idx] = y0
                pos[3, idx] = z0 + 0.5 * a
                idx += 1
                
                pos[1, idx] = x0
                pos[2, idx] = y0 + 0.5 * a
                pos[3, idx] = z0 + 0.5 * a
                idx += 1
            end
        end
    end
    
    # Wrap all positions to [0, L)
    scratch = MVector{3,Float64}(0.0, 0.0, 0.0)
    @inbounds for i in 1:N
        scratch[1] = pos[1, i]
        scratch[2] = pos[2, i]
        scratch[3] = pos[3, i]
        wrap!(scratch, L)
        pos[1, i] = scratch[1]
        pos[2, i] = scratch[2]
        pos[3, i] = scratch[3]
    end
    
    # Create parameters
    params = LJParams(1.0, 1.0, rc, rc*rc, 1.0/T, max_disp)
    
    # Create cell list
    cl = CellList(N, L, rc)
    
    # Initialize state
    rng = Xoshiro(seed)
    scratch_dr = MVector{3,Float64}(0.0, 0.0, 0.0)
    st = LJState(N, L, pos, rng, cl, scratch_dr, 0, 0)
    
    # Rebuild cell list
    rebuild_cells!(st)
    
    return (params, st)
end

"""
    local_energy(i::Int, st::LJState, p::LJParams)::Float64

Compute the local energy for particle i (sum over neighbors within rc).
Must be allocation-free. Uses st.scratch_dr.
Computes cell from actual position (does not rely on cell_of[i] being current).
"""
function local_energy(i::Int, st::LJState, p::LJParams)::Float64
    energy = 0.0
    N = st.N
    L = st.L
    rc2 = p.rc2
    ncell = st.cl.ncell
    pos = st.pos
    dr = st.scratch_dr
    
    # Get particle i position
    pix = pos[1, i]
    piy = pos[2, i]
    piz = pos[3, i]
    
    # Compute particle i's current cell from actual position
    x_wrapped = pix - L * floor(pix / L)
    y_wrapped = piy - L * floor(piy / L)
    z_wrapped = piz - L * floor(piz / L)
    i_cell_idx = get_cell(x_wrapped, y_wrapped, z_wrapped, L, ncell)
    
    # Convert to 3D cell indices
    k = ((i_cell_idx - 1) % ncell) + 1
    j = (((i_cell_idx - 1) ÷ ncell) % ncell) + 1
    i_cell = ((i_cell_idx - 1) ÷ (ncell * ncell)) + 1
    
    # Check all 27 neighboring cells (including self)
    @inbounds for di in -1:1
        for dj in -1:1
            for dk in -1:1
                cell_i = ((i_cell - 1 + di + ncell) % ncell) + 1
                cell_j = ((j - 1 + dj + ncell) % ncell) + 1
                cell_k = ((k - 1 + dk + ncell) % ncell) + 1
                
                neighbor_cell = cell_index(cell_i, cell_j, cell_k, ncell)
                
                # Iterate through particles in this cell (linked list)
                # Note: cell list may be slightly stale, but we verify distances
                pj = st.cl.head[neighbor_cell]
                while pj > 0
                    if pj != i
                        # Compute distance vector (no allocation)
                        dr[1] = pos[1, pj] - pix
                        dr[2] = pos[2, pj] - piy
                        dr[3] = pos[3, pj] - piz
                        
                        # Apply minimum image convention
                        minimum_image!(dr, L)
                        
                        r2 = dr[1]*dr[1] + dr[2]*dr[2] + dr[3]*dr[3]
                        
                        if r2 < rc2 && r2 > 0.0
                            energy += lj_potential(r2, p)
                        end
                    end
                    pj = st.cl.next[pj]
                end
            end
        end
    end
    
    return energy
end

"""
    mc_trial!(st::LJState, p::LJParams)::Bool

Perform one Monte Carlo trial move.
Does NOT modify cell list. Assumes cell list will be rebuilt periodically in sweep!.
Returns true if accepted, false if rejected.
Must be allocation-free.
"""
function mc_trial!(st::LJState, p::LJParams)::Bool
    N = st.N
    L = st.L
    pos = st.pos
    
    # Select random particle
    i = rand(st.rng, 1:N)
    
    # Store old position
    x_old = pos[1, i]
    y_old = pos[2, i]
    z_old = pos[3, i]
    
    # Compute old local energy
    Eold = local_energy(i, st, p)
    
    # Generate trial displacement (cube of size max_disp)
    dx = (rand(st.rng) - 0.5) * 2.0 * p.max_disp
    dy = (rand(st.rng) - 0.5) * 2.0 * p.max_disp
    dz = (rand(st.rng) - 0.5) * 2.0 * p.max_disp
    
    # Apply trial move
    @inbounds begin
        pos[1, i] = x_old + dx
        pos[2, i] = y_old + dy
        pos[3, i] = z_old + dz
    end
    
    # Wrap position
    dr = st.scratch_dr
    @inbounds begin
        dr[1] = pos[1, i]
        dr[2] = pos[2, i]
        dr[3] = pos[3, i]
    end
    wrap!(dr, L)
    @inbounds begin
        pos[1, i] = dr[1]
        pos[2, i] = dr[2]
        pos[3, i] = dr[3]
    end
    
    # Compute new local energy (cell list may be stale but positions are current)
    Enew = local_energy(i, st, p)
    
    # Metropolis acceptance criterion
    ΔE = Enew - Eold
    accepted = false
    if ΔE <= 0.0 || rand(st.rng) < exp(-p.β * ΔE)
        accepted = true
    else
        # Reject: restore old position (cell list unchanged, so still valid)
        @inbounds begin
            pos[1, i] = x_old
            pos[2, i] = y_old
            pos[3, i] = z_old
        end
    end
    
    return accepted
end

"""
    sweep!(st::LJState, p::LJParams; rebuild_every::Int=-1)::Float64

Perform N trial moves (one sweep).
Returns acceptance ratio for that sweep.
Rebuilds cell list at start and every rebuild_every trials (default: rebuild_every = st.N, i.e. once per sweep).
Keep allocation-free inside the per-trial loop.
"""
function sweep!(st::LJState, p::LJParams; rebuild_every::Int=-1)::Float64
    N = st.N
    n_accepted = 0
    
    # Default: rebuild once per sweep (rebuild_every = N)
    if rebuild_every == -1
        rebuild_every = N
    end
    
    # Rebuild cell list at start of sweep
    rebuild_cells!(st)
    
    @inbounds for trial in 1:N
        accepted = mc_trial!(st, p)
        if accepted
            n_accepted += 1
            st.accepted += 1
        end
        st.attempted += 1
        
        # Rebuild cells periodically if requested
        if trial % rebuild_every == 0 && trial < N
            rebuild_cells!(st)
        end
    end
    
    return Float64(n_accepted) / Float64(N)
end

"""
    total_energy(st, p)::Float64

Compute total energy of the system (O(N^2), can be slower/allocating).
Double-count-safe: only counts pairs i<j.
"""
function total_energy(st::LJState, p::LJParams)::Float64
    energy = 0.0
    N = st.N
    L = st.L
    rc2 = p.rc2
    pos = st.pos
    
    @inbounds for i in 1:N
        for j in (i+1):N
            # Compute distance vector
            dr_x = pos[1, j] - pos[1, i]
            dr_y = pos[2, j] - pos[2, i]
            dr_z = pos[3, j] - pos[3, i]
            
            # Apply minimum image convention
            L_half = L / 2.0
            if dr_x > L_half
                dr_x = dr_x - L
            elseif dr_x < -L_half
                dr_x = dr_x + L
            end
            if dr_y > L_half
                dr_y = dr_y - L
            elseif dr_y < -L_half
                dr_y = dr_y + L
            end
            if dr_z > L_half
                dr_z = dr_z - L
            elseif dr_z < -L_half
                dr_z = dr_z + L
            end
            
            r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
            
            if r2 < rc2 && r2 > 0.0
                energy += lj_potential(r2, p)
            end
        end
    end
    
    return energy
end

"""
    lj_force_magnitude_times_r(r2::Float64, p::LJParams)::Float64

Compute r * f(r) for Lennard-Jones force, where f(r) = -dU/dr.
Returns 4ε[12σ^12/r^12 - 6σ^6/r^6] = r * f(r) = r_ij · f_ij.
Returns 0.0 if r2 >= rc2.
"""
@inline function lj_force_magnitude_times_r(r2::Float64, p::LJParams)::Float64
    if r2 >= p.rc2 || r2 <= 0.0
        return 0.0
    end
    σ2_over_r2 = (p.σ * p.σ) / r2
    σ6_over_r6 = σ2_over_r2 * σ2_over_r2 * σ2_over_r2
    σ12_over_r12 = σ6_over_r6 * σ6_over_r6
    return 4.0 * p.ϵ * (12.0 * σ12_over_r12 - 6.0 * σ6_over_r6)
end

"""
    total_virial(st, p)::Float64

Compute total virial W = Σ r_ij · f_ij for pairs i<j.
Uses minimum image convention.
Double-count-safe: only counts pairs i<j.
"""
function total_virial(st::LJState, p::LJParams)::Float64
    virial = 0.0
    N = st.N
    L = st.L
    rc2 = p.rc2
    pos = st.pos
    
    @inbounds for i in 1:N
        for j in (i+1):N
            # Compute distance vector
            dr_x = pos[1, j] - pos[1, i]
            dr_y = pos[2, j] - pos[2, i]
            dr_z = pos[3, j] - pos[3, i]
            
            # Apply minimum image convention
            L_half = L / 2.0
            if dr_x > L_half
                dr_x = dr_x - L
            elseif dr_x < -L_half
                dr_x = dr_x + L
            end
            if dr_y > L_half
                dr_y = dr_y - L
            elseif dr_y < -L_half
                dr_y = dr_y + L
            end
            if dr_z > L_half
                dr_z = dr_z - L
            elseif dr_z < -L_half
                dr_z = dr_z + L
            end
            
            r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
            
            if r2 < rc2 && r2 > 0.0
                virial += lj_force_magnitude_times_r(r2, p)
            end
        end
    end
    
    return virial
end

"""
    pressure(st, p)::Float64

Compute pressure from virial equation: P = ρ*T + (1/(3V))*W
where W = Σ r_ij · f_ij is the virial.
"""
function pressure(st::LJState, p::LJParams)::Float64
    N = st.N
    L = st.L
    V = L * L * L
    ρ = N / V
    T = 1.0 / p.β
    W = total_virial(st, p)
    return ρ * T + W / (3.0 * V)
end

"""
    volume_trial!(st::LJState, p::LJParams; max_dlnV::Float64=0.01, Pext::Float64=1.0)::Bool

Perform one isotropic volume trial move for NPT Monte Carlo.
Returns true if accepted, false if rejected.
On reject, restores old positions and L.
"""
function volume_trial!(st::LJState, p::LJParams; max_dlnV::Float64=0.01, Pext::Float64=1.0)::Bool
    N = st.N
    L_old = st.L
    V_old = L_old * L_old * L_old
    lnV_old = log(V_old)
    pos = st.pos
    
    # Store old positions
    pos_old = copy(pos)
    
    # Compute old total energy
    U_old = total_energy(st, p)
    
    # Propose new volume: lnV' = lnV + (rand() - 0.5) * 2 * max_dlnV
    dlnV = (rand(st.rng) - 0.5) * 2.0 * max_dlnV
    lnV_new = lnV_old + dlnV
    V_new = exp(lnV_new)
    L_new = cbrt(V_new)
    scale = L_new / L_old
    
    # Scale all positions by L_new/L_old and wrap
    @inbounds for i in 1:N
        pos[1, i] = pos[1, i] * scale
        pos[2, i] = pos[2, i] * scale
        pos[3, i] = pos[3, i] * scale
    end
    
    # Wrap all positions to [0, L_new)
    scratch = st.scratch_dr
    @inbounds for i in 1:N
        scratch[1] = pos[1, i]
        scratch[2] = pos[2, i]
        scratch[3] = pos[3, i]
        wrap!(scratch, L_new)
        pos[1, i] = scratch[1]
        pos[2, i] = scratch[2]
        pos[3, i] = scratch[3]
    end
    
    # Update box length
    st.L = L_new
    
    # Update cell list for new box size (recreate with new L)
    st.cl = CellList(N, L_new, st.cl.rc)
    rebuild_cells!(st)
    
    # Compute new total energy
    U_new = total_energy(st, p)
    
    # Metropolis acceptance criterion for NPT:
    # acc = exp[-β(ΔU + Pext*(V' - V)) + N*ln(V'/V)]
    ΔU = U_new - U_old
    ΔV = V_new - V_old
    β = p.β
    log_acc = -β * (ΔU + Pext * ΔV) + N * (lnV_new - lnV_old)
    
    accepted = false
    if log_acc >= 0.0 || rand(st.rng) < exp(log_acc)
        accepted = true
    else
        # Reject: restore old positions and L
        copyto!(pos, pos_old)
        st.L = L_old
        st.cl = CellList(N, L_old, st.cl.rc)
        rebuild_cells!(st)
    end
    
    return accepted
end

"""
    run_npt!(st::LJState, p::LJParams; nsweeps::Int=1000, Pext::Float64=1.0,
             max_disp::Float64=-1.0, max_dlnV::Float64=0.01, vol_move_every::Int=10)

Run NPT Monte Carlo simulation.
Runs NVT sweeps and attempts one volume move every vol_move_every sweeps.
Returns (particle_acceptance, volume_acceptance, densities, energies).
"""
function run_npt!(
    st::LJState,
    p::LJParams;
    nsweeps::Int=1000,
    Pext::Float64=1.0,
    max_disp::Float64=-1.0,
    max_dlnV::Float64=0.01,
    vol_move_every::Int=10
)
    if max_disp < 0.0
        max_disp = p.max_disp
    end
    
    # Track acceptance
    particle_accepted = 0
    particle_attempted = 0
    volume_accepted = 0
    volume_attempted = 0
    
    # Track observables
    densities = Float64[]
    energies = Float64[]
    N = st.N
    
    for sweep_idx in 1:nsweeps
        # NVT sweep
        sweep_acc = sweep!(st, p)
        particle_accepted += Int(round(sweep_acc * N))
        particle_attempted += N
        
        # Volume move every vol_move_every sweeps
        if sweep_idx % vol_move_every == 0
            vol_acc = volume_trial!(st, p; max_dlnV=max_dlnV, Pext=Pext)
            if vol_acc
                volume_accepted += 1
            end
            volume_attempted += 1
            
            # Sample observables after volume move
            ρ = N / (st.L * st.L * st.L)
            E = total_energy(st, p)
            push!(densities, ρ)
            push!(energies, E)
        end
    end
    
    particle_acceptance = particle_attempted > 0 ? Float64(particle_accepted) / Float64(particle_attempted) : 0.0
    volume_acceptance = volume_attempted > 0 ? Float64(volume_accepted) / Float64(volume_attempted) : 0.0
    
    return (particle_acceptance, volume_acceptance, densities, energies)
end
