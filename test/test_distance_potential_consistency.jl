"""
Unit test for distance and potential consistency.
Verifies that distance computation and LJ potential are consistent:
- Distance computation returns r² (squared distance)
- Potential functions expect r²
- For two particles at (0,0,0) and (1,0,0), distance should be 1 (r²=1)
- LJ at r=1 should be finite and not astronomically large
"""

using Test
using MolSim
using Random

@testset "Distance and potential consistency" begin
    # Create a minimal test system: two particles at (0,0,0) and (1,0,0)
    N = 2
    L = 10.0  # Large box so minimum image doesn't affect
    rc = 2.5
    rc2 = rc * rc
    T = 1.0
    β = 1.0 / T
    
    # Create parameters
    p = MolSim.MC.LJParams(1.0, 1.0, rc, rc2, β, 0.1, false, 0.0, 0.0, :truncated, false, 0.0)
    
    # Create state with two particles at specific positions
    rng = Xoshiro(12345)
    
    pos = zeros(Float64, 3, N)
    pos[1, 1] = 0.0  # Particle 1 at (0, 0, 0)
    pos[2, 1] = 0.0
    pos[3, 1] = 0.0
    pos[1, 2] = 1.0  # Particle 2 at (1, 0, 0)
    pos[2, 2] = 0.0
    pos[3, 2] = 0.0
    
    types = [1, 1]  # Both type 1
    cl = MolSim.MC.CellList(N, L, rc)
    scratch_dr = MVector{3,Float64}(0.0, 0.0, 0.0)
    st = MolSim.MC.LJState(N, L, pos, types, rng, cl, scratch_dr, 0, 0)
    MolSim.MC.rebuild_cells!(st)
    
    # Compute distance vector manually
    dr = MVector{3,Float64}(0.0, 0.0, 0.0)
    dr[1] = pos[1, 2] - pos[1, 1]  # 1.0 - 0.0 = 1.0
    dr[2] = pos[2, 2] - pos[2, 1]  # 0.0 - 0.0 = 0.0
    dr[3] = pos[3, 2] - pos[3, 1]  # 0.0 - 0.0 = 0.0
    
    # Apply minimum image (should not change anything for L=10.0)
    MolSim.MC.minimum_image!(dr, L)
    
    # Compute r²
    r2 = dr[1]*dr[1] + dr[2]*dr[2] + dr[3]*dr[3]
    
    # Test 1: Distance should be 1 (so r² = 1)
    @test abs(r2 - 1.0) < 1e-12 "Distance squared should be 1.0 for particles at (0,0,0) and (1,0,0), got r²=$(r2)"
    
    # Test 2: Compute LJ potential using r²
    u_r2 = MolSim.MC.lj_pair_u_from_r2(r2, p)
    
    # For σ=1.0, ϵ=1.0, r=1.0:
    # u(r) = 4 * 1.0 * [(1.0/1.0)^12 - (1.0/1.0)^6]
    #      = 4 * [1 - 1] = 0.0
    # Actually wait, let me recalculate: u(r) = 4ε[(σ/r)^12 - (σ/r)^6]
    # At r=1, σ=1: u(1) = 4*1*[1^12 - 1^6] = 4*[1 - 1] = 0
    expected_u = 0.0
    
    @test abs(u_r2 - expected_u) < 1e-10 "LJ potential at r=1 should be 0.0 for σ=1.0, ϵ=1.0, got u=$(u_r2)"
    
    # Test 3: LJ should be finite (not Inf or NaN)
    @test isfinite(u_r2) "LJ potential should be finite, got $(u_r2)"
    @test !isnan(u_r2) "LJ potential should not be NaN, got $(u_r2)"
    
    # Test 4: For a different distance, verify consistency
    # At r = 2^(1/6) ≈ 1.1225 (minimum of LJ), u should be -ϵ = -1.0
    # Let's test at r = 1.1 (r² = 1.21)
    r2_test = 1.1 * 1.1  # 1.21
    u_test = MolSim.MC.lj_pair_u_from_r2(r2_test, p)
    
    # For σ=1.0, ϵ=1.0, r=1.1:
    # u(r) = 4 * 1.0 * [(1.0/1.1)^12 - (1.0/1.1)^6]
    # (1/1.1)^12 ≈ 0.3152, (1/1.1)^6 ≈ 0.5614
    # u ≈ 4 * [0.3152 - 0.5614] ≈ 4 * (-0.2462) ≈ -0.9848
    @test u_test < 0.0 "LJ at r=1.1 should be negative (attractive well)"
    @test isfinite(u_test) "LJ potential should be finite"
    
    # Test 5: Verify that passing r (not r²) would give wrong result
    # If someone accidentally passed r instead of r², u would be computed as u(r) where r is treated as r²
    # This would give a very wrong (much smaller) energy
    # We can't directly test this without breaking code, but we document it in comments
    
    # Test 6: Check total_energy gives same result as manual computation
    u_manual = MolSim.MC.lj_pair_u_from_r2(r2, p)
    u_total = MolSim.MC.total_energy(st, p)
    @test abs(u_total - u_manual) < 1e-10 "total_energy should match manual pair computation for N=2"
end
