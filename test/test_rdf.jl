using Test
using MolSim
using StaticArrays
using Random

@testset "RDF tests" begin
    # Test 1: Very low density case - g(r) ~ 1 away from r≈0
    @testset "Very low density" begin
        N = 32
        ρ = 0.01  # Very low density
        L = (N / ρ)^(1/3)
        
        # Create random configuration at low density
        rng = Random.Xoshiro(12345)
        pos = zeros(Float64, 3, N)
        for i in 1:N
            pos[1, i] = rand(rng) * L
            pos[2, i] = rand(rng) * L
            pos[3, i] = rand(rng) * L
        end
        
        rmax = L / 2.0
        nbins = 100
        r, g_r = MolSim.Analysis.rdf(pos, L, rmax, nbins)
        
        # Check g(r) is approximately 1 for r away from 0
        # Skip first few bins (r≈0 region)
        skip_bins = 5
        if length(g_r) > skip_bins
            g_away = g_r[(skip_bins+1):end]
            # At low density, g(r) should be close to 1
            mean_g = sum(g_away) / length(g_away)
            @test mean_g ≈ 1.0 rtol=0.3  # Loose tolerance for short MC run
        end
        
        # Check no NaNs or Infs
        @test all(isfinite.(g_r))
        @test all(g_r .>= 0.0)
    end
    
    # Test 2: g(r) non-negative, finite, no NaNs
    @testset "Basic sanity checks" begin
        N = 64
        ρ = 0.3
        L = (N / ρ)^(1/3)
        
        # Create random configuration
        rng = Random.Xoshiro(56789)
        pos = zeros(Float64, 3, N)
        for i in 1:N
            pos[1, i] = rand(rng) * L
            pos[2, i] = rand(rng) * L
            pos[3, i] = rand(rng) * L
        end
        
        rmax = L / 2.0
        nbins = 50
        r, g_r = MolSim.Analysis.rdf(pos, L, rmax, nbins)
        
        # Check output dimensions
        @test length(r) == nbins
        @test length(g_r) == nbins
        
        # Check r values are positive and increasing
        @test all(r .> 0.0)
        @test all(diff(r) .> 0.0)
        
        # Check g(r) is non-negative and finite
        @test all(g_r .>= 0.0)
        @test all(isfinite.(g_r))
        @test !any(isnan.(g_r))
        @test !any(isinf.(g_r))
    end
    
    # Test 3: Empty system should error
    @testset "Error handling" begin
        pos_single = zeros(Float64, 3, 1)
        L = 10.0
        
        @test_throws ErrorException MolSim.Analysis.rdf(pos_single, L, 5.0, 50)
    end
end
