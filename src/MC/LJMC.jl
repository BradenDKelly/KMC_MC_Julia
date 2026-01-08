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
"""
function local_energy(i::Int, st::LJState, p::LJParams)::Float64
    energy = 0.0
    N = st.N
    L = st.L
    rc2 = p.rc2
    ncell = st.cl.ncell
    pos = st.pos
    dr = st.scratch_dr
    
    # Get particle i's cell
    cell_idx = st.cl.cell_of[i]
    k = ((cell_idx - 1) % ncell) + 1
    j = (((cell_idx - 1) ÷ ncell) % ncell) + 1
    i_cell = ((cell_idx - 1) ÷ (ncell * ncell)) + 1
    
    # Get particle i position
    pix = pos[1, i]
    piy = pos[2, i]
    piz = pos[3, i]
    
    # Check all 27 neighboring cells (including self)
    @inbounds for di in -1:1
        for dj in -1:1
            for dk in -1:1
                cell_i = ((i_cell - 1 + di + ncell) % ncell) + 1
                cell_j = ((j - 1 + dj + ncell) % ncell) + 1
                cell_k = ((k - 1 + dk + ncell) % ncell) + 1
                
                neighbor_cell = cell_index(cell_i, cell_j, cell_k, ncell)
                
                # Iterate through particles in this cell (linked list)
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
    
    # Rebuild cell list
    rebuild_cells!(st)
    
    # Compute new local energy
    Enew = local_energy(i, st, p)
    
    # Metropolis acceptance criterion
    ΔE = Enew - Eold
    accepted = false
    if ΔE <= 0.0 || rand(st.rng) < exp(-p.β * ΔE)
        accepted = true
    else
        # Reject: restore old position
        @inbounds begin
            pos[1, i] = x_old
            pos[2, i] = y_old
            pos[3, i] = z_old
        end
        
        # Rebuild cell list to restore state
        rebuild_cells!(st)
    end
    
    return accepted
end

"""
    sweep!(st::LJState, p::LJParams; rebuild_every::Int=1)::Float64

Perform N trial moves (one sweep).
Returns acceptance ratio for that sweep.
Keep allocation-free inside the per-trial loop.
"""
function sweep!(st::LJState, p::LJParams; rebuild_every::Int=1)::Float64
    N = st.N
    n_accepted = 0
    
    @inbounds for trial in 1:N
        accepted = mc_trial!(st, p)
        if accepted
            n_accepted += 1
            st.accepted += 1
        end
        st.attempted += 1
        
        # Rebuild cells periodically if requested (currently we rebuild every trial)
        # This is kept for API compatibility but has no effect with current implementation
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
