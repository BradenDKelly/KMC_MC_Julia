"""
Test that multicomponent LJ with identical types reduces to single-component behavior.
"""

using Test
using Random
using MolSim.MC

# Import required functions
using MolSim.MC: init_fcc, LJParams, LJState, rebuild_cells!, total_energy, total_virial, local_energy, widom_deltaU

# Import Xoshiro for test
using Random: Xoshiro

@testset "Multicomponent reduction to single-component" begin
    Random.seed!(1234)
    
    N = 32
    ρ = 0.5
    T = 1.0
    rc = 2.5
    max_disp = 0.1
    σ_global = 1.0
    ϵ_global = 1.0
    
    # Initialize FCC lattice
    p_single, st_single = init_fcc(; N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp, 
                                   seed=1234, use_lrc=false, lj_model=:truncated)
    
    # Create multicomponent params with identical types (should match single-component)
    p_multi = LJParams(σ_types=[σ_global, σ_global], ϵ_types=[ϵ_global, ϵ_global],
                       rc=rc, T=T, max_disp=max_disp, use_lrc=false, lj_model=:truncated)
    
    # Create state with random type assignments (but types are identical, so shouldn't matter)
    types = rand(Random.MersenneTwister(1234), 1:2, N)
    st_multi = LJState(st_single.N, st_single.L, copy(st_single.pos), types,
                       st_single.rng, st_single.cl, copy(st_single.scratch_dr),
                       st_single.accepted, st_single.attempted)
    
    # Ensure cell lists are consistent
    rebuild_cells!(st_multi)
    
    # Test 1: Mixing tables should match single-component
    @test p_multi.n_types == 2
    @test p_multi.σ_mix[1, 1] ≈ σ_global atol=1e-10
    @test p_multi.σ_mix[2, 2] ≈ σ_global atol=1e-10
    @test p_multi.σ_mix[1, 2] ≈ σ_global atol=1e-10  # (σ1 + σ2)/2 = (1+1)/2 = 1
    @test p_multi.ϵ_mix[1, 1] ≈ ϵ_global atol=1e-10
    @test p_multi.ϵ_mix[2, 2] ≈ ϵ_global atol=1e-10
    @test p_multi.ϵ_mix[1, 2] ≈ ϵ_global atol=1e-10  # sqrt(ϵ1*ϵ2) = sqrt(1*1) = 1
    
    # Test 2: Total energy should match
    E_single = total_energy(st_single, p_single)
    E_multi = total_energy(st_multi, p_multi)
    @test E_multi ≈ E_single rtol=1e-10
    
    # Test 3: Total virial should match
    W_single = total_virial(st_single, p_single)
    W_multi = total_virial(st_multi, p_multi)
    @test W_multi ≈ W_single rtol=1e-10
    
    # Test 4: Local energies for a few particles should match
    for i in [1, 5, 10, N]
        E_local_single = local_energy(i, st_single, p_single)
        E_local_multi = local_energy(i, st_multi, p_multi)
        @test E_local_multi ≈ E_local_single rtol=1e-10
    end
    
    # Test 5: Widom insertion should match
    # For single-component, test_type defaults to 1 (backward compatible)
    st_single_rng = copy(st_single.rng)
    st_multi_rng = copy(st_single.rng)
    st_single_test = LJState(st_single.N, st_single.L, copy(st_single.pos), st_single.types,
                             st_single_rng, st_single.cl, copy(st_single.scratch_dr),
                             st_single.accepted, st_single.attempted)
    st_multi_test = LJState(st_multi.N, st_multi.L, copy(st_multi.pos), st_multi.types,
                            st_multi_rng, st_multi.cl, copy(st_multi.scratch_dr),
                            st_multi.accepted, st_multi.attempted)
    
    # Use same RNG state for both
    st_single_test.rng = Xoshiro(5678)
    st_multi_test.rng = Xoshiro(5678)
    
    ΔU_widom_single = widom_deltaU(st_single_test, p_single; test_type=1)
    ΔU_widom_multi = widom_deltaU(st_multi_test, p_multi; test_type=1)
    @test ΔU_widom_multi ≈ ΔU_widom_single rtol=1e-10
    
    # Also test with test_type=2 (should match since types are identical)
    st_multi_test.rng = Xoshiro(5678)
    ΔU_widom_multi2 = widom_deltaU(st_multi_test, p_multi; test_type=2)
    @test ΔU_widom_multi2 ≈ ΔU_widom_single rtol=1e-10
    
    # Test 6: Create a single-component multicomponent params (n_types=1)
    # This should also match
    p_multi_1type = LJParams(σ_types=[σ_global], ϵ_types=[ϵ_global],
                             rc=rc, T=T, max_disp=max_disp, use_lrc=false, lj_model=:truncated)
    types_1type = fill(1, N)
    st_multi_1type = LJState(st_single.N, st_single.L, copy(st_single.pos), types_1type,
                             st_single.rng, st_single.cl, copy(st_single.scratch_dr),
                             st_single.accepted, st_single.attempted)
    rebuild_cells!(st_multi_1type)
    
    E_multi_1type = total_energy(st_multi_1type, p_multi_1type)
    @test E_multi_1type ≈ E_single rtol=1e-10
end
