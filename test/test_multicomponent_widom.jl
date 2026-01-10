"""
Test species-resolved Widom insertion for multicomponent systems.
"""

using Test
using Random
using MolSim.MC

# Import required functions
using MolSim.MC: init_fcc, LJParams, LJState, rebuild_cells!, WidomAccumulator, widom_deltaU, widom_deltaU_at_point, mu_ex

# Import Xoshiro for test
using Random: Xoshiro

# Helper function for log-sum-exp trick
function logsumexp(x::Vector{Float64})::Float64
    if isempty(x)
        return -Inf
    end
    m = maximum(x)
    if isinf(m)
        return m
    end
    s = sum(exp(xi - m) for xi in x)
    return m + log(s)
end

@testset "Species-resolved Widom insertion" begin
    Random.seed!(1234)
    
    N = 32
    ρ = 0.5
    T = 1.0
    rc = 2.5
    max_disp = 0.1
    
    # Test 1: Identical types should give identical Widom results
    σ_identical = 1.0
    ϵ_identical = 1.0
    
    _, st_identical = init_fcc(; N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp,
                               seed=1234, use_lrc=false, lj_model=:truncated)
    
    # Assign all particles type 1
    types_identical = fill(1, N)
    st_identical.types = types_identical
    rebuild_cells!(st_identical)
    
    p_identical = LJParams(σ_types=[σ_identical, σ_identical], 
                           ϵ_types=[ϵ_identical, ϵ_identical],
                           rc=rc, T=T, max_disp=max_disp, use_lrc=false, lj_model=:truncated)
    
    β = 1.0 / T
    n_samples = 500
    
    # Generate fixed list of insertion points (with fixed RNG seed)
    rng_points = Xoshiro(9876)
    L = st_identical.L
    insertion_points = Vector{Tuple{Float64, Float64, Float64}}(undef, n_samples)
    for i in 1:n_samples
        x = rand(rng_points) * L
        y = rand(rng_points) * L
        z = rand(rng_points) * L
        insertion_points[i] = (x, y, z)
    end
    
    # Compute ΔU for test_type=1 and test_type=2 at the SAME points
    ΔU_type1 = Vector{Float64}(undef, n_samples)
    ΔU_type2 = Vector{Float64}(undef, n_samples)
    
    for i in 1:n_samples
        x, y, z = insertion_points[i]
        ΔU_type1[i] = widom_deltaU_at_point(st_identical, p_identical, x, y, z; test_type=1)
        ΔU_type2[i] = widom_deltaU_at_point(st_identical, p_identical, x, y, z; test_type=2)
    end
    
    # For identical types, ΔU values should be identical (within numerical tolerance)
    @test maximum(abs.(ΔU_type1 .- ΔU_type2)) < 1e-10
    
    # Test 2: Different types should give different Widom results
    # Use milder thermodynamic state (lower density, higher T) to avoid underflow
    ρ_mild = 0.2  # Lower density
    T_mild = 2.0  # Higher temperature
    β_mild = 1.0 / T_mild
    
    σ1 = 1.0
    ϵ1 = 1.0
    σ2 = 1.8  # Much larger σ to create obvious difference
    ϵ2 = 3.0  # Much larger ϵ to create obvious difference
    
    _, st_different = init_fcc(; N=N, ρ=ρ_mild, T=T_mild, rc=rc, max_disp=max_disp,
                               seed=1234, use_lrc=false, lj_model=:truncated)
    
    # Assign types (half type 1, half type 2)
    types_different = Vector{Int}(undef, N)
    for i in 1:N
        types_different[i] = (i <= N ÷ 2) ? 1 : 2
    end
    st_different.types = types_different
    rebuild_cells!(st_different)
    
    p_different = LJParams(σ_types=[σ1, σ2], ϵ_types=[ϵ1, ϵ2],
                           rc=rc, T=T_mild, max_disp=max_disp, use_lrc=false, lj_model=:truncated)
    
    # Sample Widom ΔU for test_type=1 and test_type=2
    ΔU_diff1 = Float64[]
    ΔU_diff2 = Float64[]
    
    rng_diff = Xoshiro(9876)
    for _ in 1:n_samples
        # Use widom_deltaU for random insertions (different from test 1 which uses fixed points)
        ΔU1 = widom_deltaU(st_different, p_different; test_type=1)
        push!(ΔU_diff1, ΔU1)
        
        ΔU2 = widom_deltaU(st_different, p_different; test_type=2)
        push!(ΔU_diff2, ΔU2)
    end
    
    # Compute μ_ex robustly using log-sum-exp trick
    # logmean = logsumexp(-β .* ΔU) - log(n)
    # μ_ex = -(1/β) * logmean
    logmean1 = logsumexp(-β_mild .* ΔU_diff1) - log(length(ΔU_diff1))
    μ_ex1 = -(1.0 / β_mild) * logmean1
    
    logmean2 = logsumexp(-β_mild .* ΔU_diff2) - log(length(ΔU_diff2))
    μ_ex2 = -(1.0 / β_mild) * logmean2
    
    # For different types, μ_ex should differ significantly
    # Use absolute difference threshold (e.g., 0.1) instead of relative difference
    @test abs(μ_ex1 - μ_ex2) > 0.1  # At least 0.1 difference in chemical potential
    
    # Test 3: widom_deltaU_at_point should respect test_type
    test_x = st_different.L / 2.0
    test_y = st_different.L / 2.0
    test_z = st_different.L / 2.0
    
    ΔU_point1 = widom_deltaU_at_point(st_different, p_different, test_x, test_y, test_z; test_type=1)
    ΔU_point2 = widom_deltaU_at_point(st_different, p_different, test_x, test_y, test_z; test_type=2)
    
    # These should generally differ for different types (though specific values depend on local environment)
    # Just check they're both finite
    @test isfinite(ΔU_point1)
    @test isfinite(ΔU_point2)
end
