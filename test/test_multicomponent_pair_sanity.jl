"""
Test analytic correctness of cross-interaction pair energies and forces.
"""

using Test
using MolSim.MC

# Import required functions
using MolSim.MC: LJParams, lj_pair_u_from_r2_mixed, lj_force_magnitude_times_r_mixed

@testset "Multicomponent pair interaction sanity" begin
    T = 1.0
    rc = 5.0  # Large cutoff to avoid truncation
    
    # Create 2-component system with distinct parameters
    σ1 = 1.0
    ϵ1 = 1.0
    σ2 = 1.5
    ϵ2 = 2.0
    
    # Use truncated model (no shift) for easier analytic comparison
    p = LJParams(σ_types=[σ1, σ2], ϵ_types=[ϵ1, ϵ2],
                 rc=rc, T=T, max_disp=0.1, use_lrc=false, lj_model=:truncated)
    
    # Test mixing tables
    σ11_expected = σ1  # (σ1 + σ1)/2 = σ1
    σ22_expected = σ2  # (σ2 + σ2)/2 = σ2
    σ12_expected = 0.5 * (σ1 + σ2)  # (σ1 + σ2)/2
    
    ϵ11_expected = ϵ1  # sqrt(ϵ1 * ϵ1) = ϵ1
    ϵ22_expected = ϵ2  # sqrt(ϵ2 * ϵ2) = ϵ2
    ϵ12_expected = sqrt(ϵ1 * ϵ2)  # sqrt(ϵ1 * ϵ2)
    
    @test p.σ_mix[1, 1] ≈ σ11_expected atol=1e-10
    @test p.σ_mix[2, 2] ≈ σ22_expected atol=1e-10
    @test p.σ_mix[1, 2] ≈ σ12_expected atol=1e-10
    @test p.σ_mix[2, 1] ≈ σ12_expected atol=1e-10  # Symmetry
    
    @test p.ϵ_mix[1, 1] ≈ ϵ11_expected atol=1e-10
    @test p.ϵ_mix[2, 2] ≈ ϵ22_expected atol=1e-10
    @test p.ϵ_mix[1, 2] ≈ ϵ12_expected atol=1e-10
    @test p.ϵ_mix[2, 1] ≈ ϵ12_expected atol=1e-10  # Symmetry
    
    # Test pair energies at specific r values
    r_values = [0.95 * σ1, 1.2 * σ1, 2.0 * σ1]  # In absolute units
    
    for r in r_values
        r2 = r * r
        
        # Test (1,1) pair
        u11 = lj_pair_u_from_r2_mixed(r2, 1, 1, p)
        σ11 = p.σ_mix[1, 1]
        ϵ11 = p.ϵ_mix[1, 1]
        u11_expected = 4.0 * ϵ11 * ((σ11/r)^12 - (σ11/r)^6)
        @test u11 ≈ u11_expected rtol=1e-10
        
        # Test (2,2) pair
        u22 = lj_pair_u_from_r2_mixed(r2, 2, 2, p)
        σ22 = p.σ_mix[2, 2]
        ϵ22 = p.ϵ_mix[2, 2]
        u22_expected = 4.0 * ϵ22 * ((σ22/r)^12 - (σ22/r)^6)
        @test u22 ≈ u22_expected rtol=1e-10
        
        # Test (1,2) pair (cross-interaction)
        u12 = lj_pair_u_from_r2_mixed(r2, 1, 2, p)
        σ12 = p.σ_mix[1, 2]
        ϵ12 = p.ϵ_mix[1, 2]
        u12_expected = 4.0 * ϵ12 * ((σ12/r)^12 - (σ12/r)^6)
        @test u12 ≈ u12_expected rtol=1e-10
        
        # Test symmetry: u12 == u21
        u21 = lj_pair_u_from_r2_mixed(r2, 2, 1, p)
        @test u21 ≈ u12 rtol=1e-10
        
        # Test forces (symmetry)
        f12 = lj_force_magnitude_times_r_mixed(r2, 1, 2, p)
        f21 = lj_force_magnitude_times_r_mixed(r2, 2, 1, p)
        @test f21 ≈ f12 rtol=1e-10
    end
    
    # Test at cutoff
    r_cut = rc - 0.01
    r2_cut = r_cut * r_cut
    u_cut = lj_pair_u_from_r2_mixed(r2_cut, 1, 2, p)
    
    if p.lj_model == :shifted
        # For shifted LJ, u(rc) = 0, so near cutoff should be near-zero
        @test abs(u_cut) < 1e-10
    else  # :truncated
        # For truncated LJ, u(rc) is negative (potential is attractive at cutoff)
        # Verify it's negative and matches analytic value
        σ12 = p.σ_mix[1, 2]
        ϵ12 = p.ϵ_mix[1, 2]
        u_cut_expected = 4.0 * ϵ12 * ((σ12/r_cut)^12 - (σ12/r_cut)^6)
        @test u_cut ≈ u_cut_expected rtol=1e-10
        @test u_cut < 0.0  # Should be negative for truncated LJ at cutoff
    end
    
    # Test beyond cutoff (should return 0)
    r_beyond = rc + 0.01
    r2_beyond = r_beyond * r_beyond
    u_beyond = lj_pair_u_from_r2_mixed(r2_beyond, 1, 2, p)
    @test u_beyond == 0.0
end
