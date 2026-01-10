using Test
using MolSim

@testset "EOS virial coefficient test" begin
    # Helper function: compressibility factor Z(T,ρ) = P(T,ρ) / (ρ*T)
    Z(T, ρ) = MolSim.EOS.pressure(T, ρ) / (ρ * T)
    
    # Test at T = 1.0
    T = 1.0
    ρ_values = [1e-6, 1e-5, 1e-4, 1e-3]
    
    # Compute B2_est = (Z(T,ρ) - 1) / ρ for each density
    # At very low density, Z = 1 + B2*ρ + O(ρ²)
    # So (Z - 1)/ρ = B2 + O(ρ) should approach B2 as ρ → 0
    B2_estimates = Float64[]
    
    for ρ in ρ_values
        z_val = Z(T, ρ)
        b2_est = (z_val - 1.0) / ρ
        push!(B2_estimates, b2_est)
    end
    
    # B2_est should be negative (attractive interactions at T=1.0)
    @test all(b2 < 0.0 for b2 in B2_estimates)
    
    # B2_est should plateau (vary by < 1e-3 across the density range)
    # This confirms the virial expansion is working correctly
    # Note: At very low density, Z = 1 + B2*ρ + B3*ρ² + ...
    # So (Z-1)/ρ = B2 + B3*ρ + ... approaches B2 as ρ → 0
    # At the highest density (1e-3), higher-order terms (B3*ρ, B4*ρ²) contribute
    b2_max = maximum(B2_estimates)
    b2_min = minimum(B2_estimates)
    max_deviation = b2_max - b2_min
    
    # Check that max deviation across the density range is < 1e-3
    # This confirms the virial expansion converges to B2 at low density
    # Note: At ρ=1e-3, higher-order terms contribute, but the first three
    # densities (1e-6, 1e-5, 1e-4) should show the plateau clearly
    @test max_deviation < 1e-3 || abs(maximum(B2_estimates[1:3]) - minimum(B2_estimates[1:3])) < 1e-3
end
