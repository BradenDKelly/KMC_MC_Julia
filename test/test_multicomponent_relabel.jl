"""
Test that relabeling type IDs preserves physics (invariance under permutation).
"""

using Test
using Random
using MolSim.MC

# Import required functions
using MolSim.MC: init_fcc, LJParams, LJState, rebuild_cells!, total_energy, total_virial, local_energy, widom_deltaU

# Import Xoshiro for test
using Random: Xoshiro

@testset "Multicomponent relabel invariance" begin
    Random.seed!(1234)
    
    N = 32
    ρ = 0.5
    T = 1.0
    rc = 2.5
    max_disp = 0.1
    
    # Create 2-component system with distinct parameters
    σ1 = 1.0
    ϵ1 = 1.0
    σ2 = 1.5
    ϵ2 = 2.0
    
    # Initialize FCC lattice
    _, st_base = init_fcc(; N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp,
                          seed=1234, use_lrc=false, lj_model=:truncated)
    
    # Assign types randomly (approximately equal numbers)
    types1 = Vector{Int}(undef, N)
    for i in 1:N
        types1[i] = (i <= N ÷ 2) ? 1 : 2
    end
    # Shuffle to create random arrangement
    rng = Random.MersenneTwister(5678)
    types1 = shuffle(rng, types1)
    
    # Create original state and params
    p1 = LJParams(σ_types=[σ1, σ2], ϵ_types=[ϵ1, ϵ2],
                  rc=rc, T=T, max_disp=max_disp, use_lrc=false, lj_model=:truncated)
    st1 = LJState(st_base.N, st_base.L, copy(st_base.pos), types1,
                  Xoshiro(1234), st_base.cl, copy(st_base.scratch_dr),
                  0, 0)
    rebuild_cells!(st1)
    
    # Compute baseline observables
    E1 = total_energy(st1, p1)
    W1 = total_virial(st1, p1)
    E_local1_indices = [1, 5, 10, 15, N]
    E_local1 = [local_energy(i, st1, p1) for i in E_local1_indices]
    
    # Relabel: swap type 1 <-> 2
    # New params: swap σ_types and ϵ_types entries
    p2 = LJParams(σ_types=[σ2, σ1], ϵ_types=[ϵ2, ϵ1],
                  rc=rc, T=T, max_disp=max_disp, use_lrc=false, lj_model=:truncated)
    
    # Relabel types in state: 1 -> 2, 2 -> 1
    types2 = [t == 1 ? 2 : 1 for t in types1]
    
    st2 = LJState(st_base.N, st_base.L, copy(st_base.pos), types2,
                  Xoshiro(1234), st_base.cl, copy(st_base.scratch_dr),
                  0, 0)
    rebuild_cells!(st2)
    
    # Compute relabeled observables
    E2 = total_energy(st2, p2)
    W2 = total_virial(st2, p2)
    E_local2 = [local_energy(i, st2, p2) for i in E_local1_indices]
    
    # Assert invariance
    @test E2 ≈ E1 rtol=1e-10
    @test W2 ≈ W1 rtol=1e-10
    @test E_local2 ≈ E_local1 rtol=1e-10
    
    # Test Widom with relabeled system
    Random.seed!(9999)
    ΔU_widom1_type1 = widom_deltaU(st1, p1; test_type=1)
    Random.seed!(9999)
    ΔU_widom2_type2 = widom_deltaU(st2, p2; test_type=2)  # Type 2 in relabeled = Type 1 in original
    @test ΔU_widom2_type2 ≈ ΔU_widom1_type1 rtol=1e-10
end
