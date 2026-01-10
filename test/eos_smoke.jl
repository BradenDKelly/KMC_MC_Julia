using Test
using MolSim

@testset "EOS smoke tests (low density only)" begin
    # Virial EOS is only valid at low density (ρ < 0.2)
    # Test at very low density (ρ = 0.01)
    T1 = 1.0
    ρ1 = 0.01
    
    u1 = MolSim.EOS.internal_energy(T1, ρ1)
    p1 = MolSim.EOS.pressure(T1, ρ1)
    
    @test isfinite(u1) && !isnan(u1)
    @test isfinite(p1) && !isnan(p1)
    @test u1 < 0.0  # Residual energy should be negative
    
    # Compressibility factor Z = p/(ρ*T) should be close to 1 at very low density
    Z1 = p1 / (ρ1 * T1)
    @test 0.8 <= Z1 <= 1.2  # Z should be near 1.0 for ideal gas limit
    
    # Test at low density (ρ = 0.05)
    T2 = 1.0
    ρ2 = 0.05
    
    u2 = MolSim.EOS.internal_energy(T2, ρ2)
    p2 = MolSim.EOS.pressure(T2, ρ2)
    
    @test isfinite(u2) && !isnan(u2)
    @test isfinite(p2) && !isnan(p2)
    @test u2 < 0.0  # Residual energy should be negative
    
    # Compressibility factor should be reasonable at low density
    Z2 = p2 / (ρ2 * T2)
    @test 0.8 <= Z2 <= 1.2  # Z should be near 1.0 for low density
    
    # Test at low density (ρ = 0.1)
    T3 = 1.0
    ρ3 = 0.1
    
    u3 = MolSim.EOS.internal_energy(T3, ρ3)
    p3 = MolSim.EOS.pressure(T3, ρ3)
    
    @test isfinite(u3) && !isnan(u3)
    @test isfinite(p3) && !isnan(p3)
    @test u3 < 0.0  # Residual energy should be negative
    
    # Pressure should be positive
    @test p3 > 0.0
    
    # Test at higher temperature, low density (T = 1.5, ρ = 0.05)
    T4 = 1.5
    ρ4 = 0.05
    
    u4 = MolSim.EOS.internal_energy(T4, ρ4)
    p4 = MolSim.EOS.pressure(T4, ρ4)
    
    @test isfinite(u4) && !isnan(u4)
    @test isfinite(p4) && !isnan(p4)
    @test p4 > 0.0
    
    # Test at higher temperature, low density (T = 1.5, ρ = 0.1)
    T5 = 1.5
    ρ5 = 0.1
    
    u5 = MolSim.EOS.internal_energy(T5, ρ5)
    p5 = MolSim.EOS.pressure(T5, ρ5)
    
    @test isfinite(u5) && !isnan(u5)
    @test isfinite(p5) && !isnan(p5)
    @test p5 > 0.0
end
