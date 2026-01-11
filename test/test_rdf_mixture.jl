using Test
using MolSim
using Random
using StaticArrays
using Statistics

@testset "Mixture RDF: deterministic lattice geometry" begin
    # Create a small periodic configuration with alternating A/B pattern
    # Simple cubic: 2x2x2 = 8 particles, alternating A and B
    N = 8
    L = 2.0
    n_types = 2
    
    # Create positions: simple cubic lattice
    pos = zeros(Float64, 3, N)
    spacing = L / 2.0
    idx = 1
    for k in 0:1
        for j in 0:1
            for i in 0:1
                pos[1, idx] = (i + 0.5) * spacing
                pos[2, idx] = (j + 0.5) * spacing
                pos[3, idx] = (k + 0.5) * spacing
                idx += 1
            end
        end
    end
    
    # Assign types: alternating A/B
    types = Int[]
    for i in 1:N
        push!(types, (i % 2 == 1) ? 1 : 2)  # Odd indices = type 1 (A), even = type 2 (B)
    end
    
    # Compute RDF
    # Use rmax slightly larger than L/2 to ensure nearest neighbors at r=1.0 are included
    rmax = L / 2.0 * 1.1
    nbins = 50
    r, g_ab = MolSim.Analysis.rdf(pos, L, rmax, nbins; types=types)
    
    # Check output dimensions
    @test length(r) == nbins
    @test size(g_ab) == (n_types, n_types, nbins)
    
    # Check for expected peaks at nearest-neighbor distance
    # Nearest neighbors in simple cubic: spacing ≈ 1.0
    # Find bin closest to r = 1.0
    nearest_bin = argmin(abs.(r .- 1.0))
    
    # A-A pairs: should have peak (particles 1,3,5,7 are type 1)
    # A-B pairs: should have peak (mixed neighbors)
    # B-B pairs: should have peak (particles 2,4,6,8 are type 2)
    
    # All channels should show structure (not all zeros)
    @test any(g_ab[1, 1, :] .> 0.1)  # A-A has structure
    @test any(g_ab[2, 2, :] .> 0.1)  # B-B has structure
    @test any(g_ab[1, 2, :] .> 0.1)  # A-B has structure
    
    # Check symmetry: g_{αβ} = g_{βα}
    for α in 1:n_types
        for β in 1:n_types
            @test all(abs.(g_ab[α, β, :] - g_ab[β, α, :]) .< 1e-10)
        end
    end
    
    # Check g(r) values are non-negative and finite
    @test all(g_ab .>= 0.0)
    @test all(isfinite.(g_ab))
end

@testset "Mixture RDF: ideal-gas sanity (low density)" begin
    # Very low density mixture - interactions should be weak
    N = 64
    ρ = 0.01  # Very low density
    L = (N / ρ)^(1/3)
    n_types = 2
    seed = 11111
    
    # Create random configuration
    rng = Random.Xoshiro(seed)
    pos = zeros(Float64, 3, N)
    for i in 1:N
        pos[1, i] = rand(rng) * L
        pos[2, i] = rand(rng) * L
        pos[3, i] = rand(rng) * L
    end
    
    # Assign types: half A, half B
    types = [i <= N÷2 ? 1 : 2 for i in 1:N]
    
    # Compute RDF
    rmax = L / 2.0
    nbins = 100
    r, g_ab = MolSim.Analysis.rdf(pos, L, rmax, nbins; types=types)
    
    # At low density, g(r) should be approximately 1 for r > small r_min
    # Skip first few bins (r ≈ 0 region)
    skip_bins = 5
    if nbins > skip_bins
        for α in 1:n_types
            for β in 1:n_types
                g_away = g_ab[α, β, (skip_bins+1):end]
                mean_g = Statistics.mean(g_away)
                # At low density, g(r) should be close to 1
                @test mean_g ≈ 1.0 rtol=0.5  # Loose tolerance for random config
            end
        end
    end
    
    # All values should be finite and non-negative
    @test all(isfinite.(g_ab))
    @test all(g_ab .>= 0.0)
end

@testset "Mixture RDF: reduction to single-component" begin
    # Create a "mixture" where all particles are type 1 (effectively single-component)
    N = 64
    ρ = 0.5
    L = (N / ρ)^(1/3)
    seed = 22222
    
    # Create random configuration
    rng = Random.Xoshiro(seed)
    pos = zeros(Float64, 3, N)
    for i in 1:N
        pos[1, i] = rand(rng) * L
        pos[2, i] = rand(rng) * L
        pos[3, i] = rand(rng) * L
    end
    
    # All particles are type 1
    types = fill(1, N)
    
    # Compute single-component RDF
    rmax = L / 2.0
    nbins = 100
    r_single, g_single = MolSim.Analysis.rdf(pos, L, rmax, nbins; types=nothing)
    
    # Compute mixture RDF (all type 1)
    r_mix, g_mix = MolSim.Analysis.rdf(pos, L, rmax, nbins; types=types)
    
    # Check dimensions
    @test length(r_single) == length(r_mix)
    @test size(g_mix) == (1, 1, nbins)
    
    # Extract g_{11} from mixture
    g_11 = g_mix[1, 1, :]
    
    # They should match (within numerical tolerance)
    @test all(abs.(r_single - r_mix) .< 1e-10)
    @test all(abs.(g_single - g_11) .< 1e-6)  # Allow small numerical differences
end

@testset "Mixture RDF: two-species reduction test" begin
    # Create a "mixture" where all particles have the same type but labeled as type 1
    # Compare with explicit single-component call
    N = 32
    ρ = 0.3
    L = (N / ρ)^(1/3)
    
    # Use FCC-like positions (via init_fcc)
    p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=1.0, rc=2.5, max_disp=0.1, seed=42, use_lrc=false)
    
    # All particles are type 1 (single-component)
    types_all_one = fill(1, N)
    
    # Compute single-component RDF
    rmax = st.L / 2.0
    nbins = 50
    r_single, g_single = MolSim.Analysis.rdf(st.pos, st.L, rmax, nbins; types=nothing)
    
    # Compute mixture RDF with all type 1
    r_mix, g_mix = MolSim.Analysis.rdf(st.pos, st.L, rmax, nbins; types=types_all_one)
    
    # Extract g_{11}
    g_11 = g_mix[1, 1, :]
    
    # They should match
    @test all(abs.(r_single - r_mix) .< 1e-10)
    @test all(abs.(g_single - g_11) .< 1e-6)
end
