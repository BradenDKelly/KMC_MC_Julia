"""
Widom insertion method for excess chemical potential.
"""

"""
    widom_deltaU(st, p)::Float64

Compute energy change ΔU for inserting a test particle at a random position.
Must be allocation-free: reuses st.scratch_dr.
"""
function widom_deltaU(st::LJState, p::LJParams)::Float64
    N = st.N
    L = st.L
    rc2 = p.rc2
    ncell = st.cl.ncell
    pos = st.pos
    dr = st.scratch_dr
    
    # Generate random test position in [0, L)^3
    test_x = rand(st.rng) * L
    test_y = rand(st.rng) * L
    test_z = rand(st.rng) * L
    
    # Compute cell index for test position
    test_cell_idx = get_cell(test_x, test_y, test_z, L, ncell)
    
    # Convert to 3D cell indices
    k = ((test_cell_idx - 1) % ncell) + 1
    j = (((test_cell_idx - 1) ÷ ncell) % ncell) + 1
    i_cell = ((test_cell_idx - 1) ÷ (ncell * ncell)) + 1
    
    # Compute ΔU by summing over neighbors in 27 cells
    ΔU = 0.0
    
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
                    # Compute distance vector (no allocation)
                    dr[1] = pos[1, pj] - test_x
                    dr[2] = pos[2, pj] - test_y
                    dr[3] = pos[3, pj] - test_z
                    
                    # Apply minimum image convention
                    minimum_image!(dr, L)
                    
                    r2 = dr[1]*dr[1] + dr[2]*dr[2] + dr[3]*dr[3]
                    
                    if r2 < rc2 && r2 > 0.0
                        ΔU += lj_pair_u_from_r2(r2, p)
                    end
                    pj = st.cl.next[pj]
                end
            end
        end
    end
    
    return ΔU
end

"""
    rmin_to_particles(st, x)::Float64

Compute minimum distance from insertion point x to any particle.
Allocation-free: uses cell-scan only, reuses st.scratch_dr.
Must match the same scan geometry as widom_deltaU.
"""
function rmin_to_particles(st::LJState, x::Tuple{Float64, Float64, Float64})::Float64
    test_x, test_y, test_z = x
    L = st.L
    rc2 = st.cl.rc * st.cl.rc
    ncell = st.cl.ncell
    pos = st.pos
    dr = st.scratch_dr
    
    # Compute cell index for test position
    test_cell_idx = get_cell(test_x, test_y, test_z, L, ncell)
    
    # Convert to 3D cell indices
    k = ((test_cell_idx - 1) % ncell) + 1
    j = (((test_cell_idx - 1) ÷ ncell) % ncell) + 1
    i_cell = ((test_cell_idx - 1) ÷ (ncell * ncell)) + 1
    
    rmin_sq = Inf
    
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
                    # Compute distance vector (no allocation)
                    dr[1] = pos[1, pj] - test_x
                    dr[2] = pos[2, pj] - test_y
                    dr[3] = pos[3, pj] - test_z
                    
                    # Apply minimum image convention
                    minimum_image!(dr, L)
                    
                    r2 = dr[1]*dr[1] + dr[2]*dr[2] + dr[3]*dr[3]
                    
                    if r2 < rmin_sq
                        rmin_sq = r2
                    end
                    pj = st.cl.next[pj]
                end
            end
        end
    end
    
    return sqrt(rmin_sq)
end

"""
    widom_deltaU_explain(st, p, x::Vector{Float64}) -> (ΔU::Float64, nterms::Int, r_small::Vector{Float64}, u_small::Vector{Float64})

Debug version of widom_deltaU that returns additional diagnostics.
x is the insertion point [x, y, z].
Returns ΔU, number of terms, and up to 20 smallest r and corresponding u(r) values.
This function can allocate (for debugging only).
"""
function widom_deltaU_explain(st::LJState, p::LJParams, x::Vector{Float64})
    N = st.N
    L = st.L
    rc2 = p.rc2
    ncell = st.cl.ncell
    pos = st.pos
    dr = st.scratch_dr
    
    test_x = x[1]
    test_y = x[2]
    test_z = x[3]
    
    # Compute cell index for test position
    test_cell_idx = get_cell(test_x, test_y, test_z, L, ncell)
    
    # Convert to 3D cell indices
    k = ((test_cell_idx - 1) % ncell) + 1
    j = (((test_cell_idx - 1) ÷ ncell) % ncell) + 1
    i_cell = ((test_cell_idx - 1) ÷ (ncell * ncell)) + 1
    
    # Compute ΔU by summing over neighbors in 27 cells
    ΔU = 0.0
    nterms = 0
    
    # Track smallest r values (up to 20)
    r_small = Float64[]
    u_small = Float64[]
    rmin = Inf
    
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
                    # Compute distance vector (no allocation)
                    dr[1] = pos[1, pj] - test_x
                    dr[2] = pos[2, pj] - test_y
                    dr[3] = pos[3, pj] - test_z
                    
                    # Apply minimum image convention
                    minimum_image!(dr, L)
                    
                    r2 = dr[1]*dr[1] + dr[2]*dr[2] + dr[3]*dr[3]
                    
                    if r2 < rc2 && r2 > 0.0
                        r = sqrt(r2)
                        if r < rmin
                            rmin = r
                        end
                        
                        u_val = lj_pair_u_from_r2(r2, p)
                        ΔU += u_val
                        nterms += 1
                        
                        # Maintain top-k smallest r list (k=20)
                        if length(r_small) < 20
                            push!(r_small, r)
                            push!(u_small, u_val)
                        else
                            # Find maximum in current list
                            max_idx = 1
                            max_r = r_small[1]
                            for idx in 2:length(r_small)
                                if r_small[idx] > max_r
                                    max_r = r_small[idx]
                                    max_idx = idx
                                end
                            end
                            # Replace if current r is smaller
                            if r < max_r
                                r_small[max_idx] = r
                                u_small[max_idx] = u_val
                            end
                        end
                    end
                    pj = st.cl.next[pj]
                end
            end
        end
    end
    
    # Sort r_small and u_small by r (ascending)
    if length(r_small) > 1
        indices = collect(1:length(r_small))
        # Simple bubble sort (small list, so fine)
        for i in 1:(length(r_small)-1)
            for j in 1:(length(r_small)-i)
                if r_small[j] > r_small[j+1]
                    # Swap r
                    temp_r = r_small[j]
                    r_small[j] = r_small[j+1]
                    r_small[j+1] = temp_r
                    # Swap u
                    temp_u = u_small[j]
                    u_small[j] = u_small[j+1]
                    u_small[j+1] = temp_u
                end
            end
        end
    end
    
    return (ΔU, nterms, r_small, u_small, rmin)
end

"""
    WidomAccumulator

Numerically stable accumulator for exp(-βΔU) using Welford's online algorithm.
"""
mutable struct WidomAccumulator
    n::Int
    m::Float64  # running mean
    s::Float64  # running sum of squares of differences
end

"""
    WidomAccumulator()

Create a new Widom accumulator.
"""
function WidomAccumulator()
    return WidomAccumulator(0, 0.0, 0.0)
end

"""
    reset!(acc::WidomAccumulator)

Reset the accumulator to zero.
"""
function reset!(acc::WidomAccumulator)
    acc.n = 0
    acc.m = 0.0
    acc.s = 0.0
    return nothing
end

"""
    push!(acc::WidomAccumulator, β::Float64, ΔU::Float64)

Add an insertion result to the accumulator using Welford's algorithm.
"""
function Base.push!(acc::WidomAccumulator, β::Float64, ΔU::Float64)
    exp_val = exp(-β * ΔU)
    acc.n += 1
    delta = exp_val - acc.m
    acc.m += delta / acc.n
    delta2 = exp_val - acc.m
    acc.s += delta * delta2
    return acc
end

"""
    mu_ex(acc::WidomAccumulator, β::Float64)::Float64

Compute excess chemical potential: μ_ex = -T * ln(<exp(-βΔU)>)
"""
function mu_ex(acc::WidomAccumulator, β::Float64)::Float64
    if acc.n == 0
        return NaN
    end
    T = 1.0 / β
    return -T * log(acc.m)
end

"""
    widom_mu_ex!(acc::WidomAccumulator, st::LJState, p::LJParams; ninsert::Int=1000)::Float64

Perform ninsert Widom insertions and accumulate results.
Returns the excess chemical potential.
"""
function widom_mu_ex!(acc::WidomAccumulator, st::LJState, p::LJParams; ninsert::Int=1000)::Float64
    β = p.β
    for _ in 1:ninsert
        ΔU = widom_deltaU(st, p)
        push!(acc, β, ΔU)
    end
    return mu_ex(acc, β)
end

"""
    widom_mu_ex_cavity!(acc::WidomAccumulator, st::LJState, p::LJParams; ninsert::Int=1000, rmin_cut::Float64=0.85) -> (μ_ex::Float64, pbias::Float64)

Cavity-biased Widom insertion for variance reduction at high density.
Samples insertion points until rmin > rmin_cut, then computes μ_ex with bias correction.
Returns (μ_ex, pbias) where pbias = accepted_trials / total_trials.
"""
function widom_mu_ex_cavity!(acc::WidomAccumulator, st::LJState, p::LJParams; ninsert::Int=1000, rmin_cut::Float64=0.85)
    β = p.β
    L = st.L
    total_trials = 0
    accepted_trials = 0
    
    for k in 1:ninsert
        # Repeatedly sample until rmin > rmin_cut
        while true
            total_trials += 1
            test_x = rand(st.rng) * L
            test_y = rand(st.rng) * L
            test_z = rand(st.rng) * L
            
            rmin = rmin_to_particles(st, (test_x, test_y, test_z))
            
            if rmin > rmin_cut
                accepted_trials += 1
                # Compute ΔU at this point (need to compute manually since we have the point)
                ΔU = widom_deltaU_at_point(st, p, test_x, test_y, test_z)
                push!(acc, β, ΔU)
                break
            end
        end
    end
    
    pbias = Float64(accepted_trials) / Float64(total_trials)
    
    # Bias correction: <exp(-βΔU)>_unbiased = pbias * <exp(-βΔU)>_conditional
    # μ_ex = -(1/β)*log(pbias) - (1/β)*log(<exp(-βΔU)>_conditional)
    T = 1.0 / β
    μ_ex_conditional = mu_ex(acc, β)
    μ_ex_unbiased = -T * log(pbias) + μ_ex_conditional
    
    return (μ_ex_unbiased, pbias)
end

"""
    widom_deltaU_at_point(st, p, test_x, test_y, test_z)::Float64

Compute ΔU for insertion at a specific point (x, y, z).
Allocation-free: reuses st.scratch_dr.
"""
function widom_deltaU_at_point(st::LJState, p::LJParams, test_x::Float64, test_y::Float64, test_z::Float64)::Float64
    L = st.L
    rc2 = p.rc2
    ncell = st.cl.ncell
    pos = st.pos
    dr = st.scratch_dr
    
    # Compute cell index for test position
    test_cell_idx = get_cell(test_x, test_y, test_z, L, ncell)
    
    # Convert to 3D cell indices
    k = ((test_cell_idx - 1) % ncell) + 1
    j = (((test_cell_idx - 1) ÷ ncell) % ncell) + 1
    i_cell = ((test_cell_idx - 1) ÷ (ncell * ncell)) + 1
    
    # Compute ΔU by summing over neighbors in 27 cells
    ΔU = 0.0
    
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
                    # Compute distance vector (no allocation)
                    dr[1] = pos[1, pj] - test_x
                    dr[2] = pos[2, pj] - test_y
                    dr[3] = pos[3, pj] - test_z
                    
                    # Apply minimum image convention
                    minimum_image!(dr, L)
                    
                    r2 = dr[1]*dr[1] + dr[2]*dr[2] + dr[3]*dr[3]
                    
                    if r2 < rc2 && r2 > 0.0
                        ΔU += lj_pair_u_from_r2(r2, p)
                    end
                    pj = st.cl.next[pj]
                end
            end
        end
    end
    
    return ΔU
end
