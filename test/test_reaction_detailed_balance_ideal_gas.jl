"""
Ideal gas test for reaction kernel (ε=0 to isolate reaction acceptance math).

This test disables all pair interactions (ε=0) so that ΔU=0 always.
The equilibrium should then be controlled purely by K/combinatorial/proposal terms.
If the reaction acceptance math is wrong, this test will fail even when energy is not a factor.
"""

using Test
using MolSim
using Random
using SpecialFunctions
using StaticArrays

@testset "Reaction detailed balance - ideal gas (ε=0)" begin
    # Use A⇌B reaction (ΔN=0) with identical species
    # Expected equilibrium: 50/50 mix (or controlled by logq difference if any)
    
    N_total = 100
    T = 2.0
    P = 5.0
    rc = 2.5
    max_dlnV = 0.01
    vol_move_every = 100
    
    # Create species with ε=0 (no interactions)
    σ_types = [1.0, 1.0]  # A and B
    ϵ_types = [0.0, 0.0]  # NO interactions
    logq = [log(0.002), log(0.002)]  # Same logq for both
    
    # Create parameters with ε=0
    p = MolSim.MC.LJParams(; σ_types=σ_types, ϵ_types=ϵ_types, rc=rc, T=T, 
                           max_disp=0.1, use_lrc=false, lj_model=:shifted,
                           apply_impulsive_correction=false)
    
    # Create reaction A⇌B with logK=0 (equilibrium should be 50/50)
    stoichiometry = [-1, 1]  # A -> B
    rxn = MolSim.MC.Reaction("A⇌B", stoichiometry, 0.0; logq=logq)
    
    # Initialize system: start with 80 A and 20 B (mixed state, not pure)
    L = (N_total / 0.75)^(1/3)  # Initial density guess 0.75
    rng = Xoshiro(12345)
    
    # Create initial positions (random placement)
    pos = zeros(Float64, 3, N_total)
    for i in 1:N_total
        pos[1, i] = rand(rng) * L
        pos[2, i] = rand(rng) * L
        pos[3, i] = rand(rng) * L
    end
    
    # Assign types: first 80 are A, next 20 are B
    types = vcat(fill(1, 80), fill(2, 20))  # 80 A, 20 B
    
    # Create state
    cl = MolSim.MC.CellList(N_total, L, rc)
    scratch_dr = MVector{3,Float64}(0.0, 0.0, 0.0)
    st = MolSim.MC.LJState(N_total, L, pos, types, rng, cl, scratch_dr, 0, 0)
    MolSim.MC.rebuild_cells!(st)
    
    # Run only reaction moves (no translation, no volume for this test)
    # Use high p_reaction to focus on reactions
    p_reaction = 1.0
    n_sweeps = 10000  # Enough to sample equilibrium
    
    # Track species counts
    N_A_samples = Int[]
    N_B_samples = Int[]
    
    for sweep in 1:n_sweeps
        # Only reaction moves (no translation, no volume)
        # We'll call reaction_trial! directly since we want only reactions
        if rand(st.rng) < p_reaction
            direction = rand(st.rng) < 0.5 ? :forward : :reverse
            accepted, feasible = MolSim.MC.reaction_trial!(st, p, rxn, direction;
                                                           use_alchemical=true, insertion_mode=:anchored)
        end
        
        # Sample every 10 sweeps
        if sweep % 10 == 0
            counts = MolSim.MC.count_species(st, 2)
            push!(N_A_samples, counts[1])
            push!(N_B_samples, counts[2])
        end
    end
    
    # Compute mean counts (discard first 20% as equilibration)
    discard = div(length(N_A_samples), 5)
    N_A_mean = sum(N_A_samples[(discard+1):end]) / (length(N_A_samples) - discard)
    N_B_mean = sum(N_B_samples[(discard+1):end]) / (length(N_B_samples) - discard)
    
    # For ε=0, logK=0, identical logq, equilibrium should be 50/50
    # Use loose tolerance (10-15% as specified)
    N_total_expected = N_total
    N_A_expected = N_total_expected / 2.0
    N_B_expected = N_total_expected / 2.0
    
    rel_error_A = abs(N_A_mean - N_A_expected) / N_A_expected
    rel_error_B = abs(N_B_mean - N_B_expected) / N_B_expected
    
    # Print diagnostics
    println("Ideal gas test results:")
    println("  N_A_mean = $(round(N_A_mean, digits=2)), expected = $(round(N_A_expected, digits=2)), rel_error = $(round(rel_error_A, digits=4))")
    println("  N_B_mean = $(round(N_B_mean, digits=2)), expected = $(round(N_B_expected, digits=2)), rel_error = $(round(rel_error_B, digits=4))")
    println("  Final species fractions: A = $(round(N_A_mean/N_total, digits=3)), B = $(round(N_B_mean/N_total, digits=3))")
    
    @test rel_error_A < 0.15 || @error "Ideal gas A⇌B: N_A mean should be ~50 with ε=0, logK=0, got $(round(N_A_mean, digits=1)) (rel_error=$(round(rel_error_A, digits=3)))"
    @test rel_error_B < 0.15 || @error "Ideal gas A⇌B: N_B mean should be ~50 with ε=0, logK=0, got $(round(N_B_mean, digits=1)) (rel_error=$(round(rel_error_B, digits=3)))"
    
    # Verify total is conserved
    @test abs(N_A_mean + N_B_mean - N_total) < 1.0 || @error "Total particle count should be conserved, got N_A=$(round(N_A_mean, digits=1)) + N_B=$(round(N_B_mean, digits=1)) = $(round(N_A_mean + N_B_mean, digits=1))"
end
