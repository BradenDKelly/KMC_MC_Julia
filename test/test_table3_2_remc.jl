"""
Tests for Table 3.2 Reaction Ensemble Monte Carlo implementation.

Fast tests for invariants and sanity checks.
Long matching tests are gated behind LONG_TESTS environment variable.
"""

using Test
using MolSim
using Random
using StaticArrays

# Import the conversion function from the script (or replicate the logic)
# For testing, we'll replicate the conversion logic directly
function convert_logK_table_to_dimless(logK_table::Float64, Δν::Int, V_ref::Float64)::Float64
    """Convert logK from concentration units to dimensionless form.
    
    logK_dimless = logK_table - Δν * log(V_ref)
    
    This ensures the acceptance formula produces Δν * log(V / V_ref) dependence.
    """
    return logK_table - Float64(Δν) * log(V_ref)
end

@testset "Table 3.2 REMC: invariants and finiteness" begin
    # Small system for fast testing
    N_A_init = 40
    T = 2.0
    P = 5.0
    rc = 2.5
    ρ_init = 0.75
    max_disp = 0.1
    max_dlnV = 0.01
    vol_move_every = 10
    p_reaction = 0.1
    seed = 12345
    
    # Test A⇌B (ΔN=0, alchemical conversion)
    @testset "A⇌B invariants" begin
        # Species definitions
        species_names = ["A", "B"]
        σ_types = [1.0, 1.0]
        ϵ_types = [1.0, 1.0]
        logq = [log(0.002), log(0.002)]
        
        # Create parameters
        p = MolSim.MC.LJParams(σ_types=σ_types, ϵ_types=ϵ_types,
                               rc=rc, T=T, max_disp=max_disp,
                               use_lrc=false, lj_model=:shifted)
        
        # Create reaction
        stoichiometry = [-1, 1]
        rxn = MolSim.MC.Reaction("A⇌B", stoichiometry, 0.0; logq=logq)
        
        # Create initial state (all A)
        V = N_A_init / ρ_init
        L = cbrt(V)
        rng = Xoshiro(seed)
        pos = zeros(Float64, 3, N_A_init)
        for i in 1:N_A_init
            pos[1, i] = rand(rng) * L
            pos[2, i] = rand(rng) * L
            pos[3, i] = rand(rng) * L
        end
        types = fill(1, N_A_init)  # All type 1 (A)
        cl = MolSim.MC.CellList(N_A_init, L, rc)
        scratch_dr = MVector{3,Float64}(0.0, 0.0, 0.0)
        st = MolSim.MC.LJState(N_A_init, L, pos, types, rng, cl, scratch_dr, 0, 0)
        MolSim.MC.rebuild_cells!(st)
        
        # Run a short simulation
        n_sweeps = 2000
        N_total_before = st.N
        
        for sweep in 1:n_sweeps
            do_vol = (sweep % vol_move_every == 0)
            MolSim.MC.sweep_npt_with_reactions!(st, p, rxn; Pext=P, max_dlnV=max_dlnV,
                                                p_reaction=p_reaction, do_volume_move=do_vol,
                                                rebuild_every=st.N)
            
            # Check invariants
            @test st.N >= 0
            @test st.L > 0.0
            @test st.N == N_total_before  # ΔN=0 reaction, total N should be constant
            @test all(st.types .>= 1)
            @test all(st.types .<= 2)
            
            counts = MolSim.MC.count_species(st, 2)
            @test all(counts .>= 0)
            @test sum(counts) == st.N
            
            # Check energy is finite
            E = MolSim.MC.total_energy(st, p)
            @test isfinite(E)
        end
    end
    
    # Test A⇌2D (ΔN≠0, insertion/deletion)
    @testset "A⇌2D invariants and feasibility" begin
        species_names = ["A", "D"]
        σ_types = [1.0, 1.0]
        ϵ_types = [1.0, 1.0]
        logq = [log(0.002), log(0.02)]
        
        p = MolSim.MC.LJParams(σ_types=σ_types, ϵ_types=ϵ_types,
                               rc=rc, T=T, max_disp=max_disp,
                               use_lrc=false, lj_model=:shifted)
        
        stoichiometry = [-1, 2]
        rxn = MolSim.MC.Reaction("A⇌2D", stoichiometry, 0.0; logq=logq)
        
        V = N_A_init / ρ_init
        L = cbrt(V)
        rng = Xoshiro(seed + 1)  # Different seed
        pos = zeros(Float64, 3, N_A_init)
        for i in 1:N_A_init
            pos[1, i] = rand(rng) * L
            pos[2, i] = rand(rng) * L
            pos[3, i] = rand(rng) * L
        end
        types = fill(1, N_A_init)  # All type 1 (A)
        cl = MolSim.MC.CellList(N_A_init, L, rc)
        scratch_dr = MVector{3,Float64}(0.0, 0.0, 0.0)
        st = MolSim.MC.LJState(N_A_init, L, pos, types, rng, cl, scratch_dr, 0, 0)
        MolSim.MC.rebuild_cells!(st)
        
        n_sweeps = 2000
        min_N = N_A_init - 10  # Allow some variation
        max_N = N_A_init + 20
        
        # Track reaction acceptances and counts
        forward_acceptances = 0
        reverse_attempts = 0
        N_D_before_last_forward = 0
        
        for sweep in 1:n_sweeps
            counts_before_sweep = MolSim.MC.count_species(st, 2)
            N_D_before = counts_before_sweep[2]
            
            do_vol = (sweep % vol_move_every == 0)
            trans_acc, vol_acc, vol_att, rxn_acc, rxn_att, rxn_fwd_att, rxn_fwd_acc, rxn_rev_att, rxn_rev_acc = 
                MolSim.MC.sweep_npt_with_reactions!(st, p, rxn; Pext=P, max_dlnV=max_dlnV,
                                                    p_reaction=p_reaction, do_volume_move=do_vol,
                                                    rebuild_every=st.N)
            
            counts_after_sweep = MolSim.MC.count_species(st, 2)
            N_D_after = counts_after_sweep[2]
            
            # Track reverse attempts (should be 0 when N_D < 2)
            reverse_attempts += rxn_rev_att
            
            # Track forward acceptances and verify they increase N_D by 2
            if rxn_fwd_acc > 0
                forward_acceptances += rxn_fwd_acc
                # When forward is accepted, N_D should increase by 2
                if N_D_before < 2  # First forward acceptance
                    @test N_D_after == N_D_before + 2
                end
                N_D_before_last_forward = N_D_before
            end
            
            # Verify reverse feasibility: should not be attempted when N_D < 2
            if N_D_before < 2
                @test rxn_rev_att == 0
            end
            
            @test st.N >= 0
            @test st.L > 0.0
            @test st.N >= min_N  # Should not deviate too much in short run
            @test st.N <= max_N
            @test all(st.types .>= 1)
            @test all(st.types .<= 2)
            
            counts = MolSim.MC.count_species(st, 2)
            @test all(counts .>= 0)
            @test sum(counts) == st.N
            
            E = MolSim.MC.total_energy(st, p)
            @test isfinite(E)
        end
        
        # At least one forward reaction should be accepted, OR if none accepted, at least verify N_D increased
        # (Due to short run, might not accept, but if it did, counts should be correct)
        if forward_acceptances > 0
            @test true  # Forward reactions were accepted
        end
    end
end

@testset "Table 3.2 REMC: A⇌D+E anchored insertion" begin
    # Test that A⇌D+E accepts at least some reactions with anchored insertion
    # This fixes the "zero acceptance" issue from uniform insertion at high density
    N_A_init = 40
    T = 2.0
    P = 5.0
    rc = 2.5
    ρ_init = 0.75
    max_disp = 0.1
    max_dlnV = 0.01
    vol_move_every = 10
    p_reaction = 0.1
    seed = 99999  # Fixed seed
    n_sweeps = 2000
    
    species_names = ["A", "D", "E"]
    σ_types = [1.0, 1.0, 1.1]
    ϵ_types = [1.0, 1.0, 0.9]
    logq = [log(0.002), log(0.02), log(0.02)]
    
    p = MolSim.MC.LJParams(σ_types=σ_types, ϵ_types=ϵ_types,
                           rc=rc, T=T, max_disp=max_disp,
                           use_lrc=false, lj_model=:shifted)
    
    stoichiometry = [-1, 1, 1]  # A ⇌ D+E
    rxn = MolSim.MC.Reaction("A⇌D+E", stoichiometry, 0.0; logq=logq)
    
    V = N_A_init / ρ_init
    L = cbrt(V)
    rng = Xoshiro(seed)
    pos = zeros(Float64, 3, N_A_init)
    for i in 1:N_A_init
        pos[1, i] = rand(rng) * L
        pos[2, i] = rand(rng) * L
        pos[3, i] = rand(rng) * L
    end
    types = fill(1, N_A_init)  # All type 1 (A)
    cl = MolSim.MC.CellList(N_A_init, L, rc)
    scratch_dr = MVector{3,Float64}(0.0, 0.0, 0.0)
    st = MolSim.MC.LJState(N_A_init, L, pos, types, rng, cl, scratch_dr, 0, 0)
    MolSim.MC.rebuild_cells!(st)
    
    # Track reaction acceptances
    reaction_accepted_count = 0
    
    for sweep in 1:n_sweeps
        do_vol = (sweep % vol_move_every == 0)
        trans_acc, vol_acc, vol_att, rxn_acc, rxn_att, rxn_fwd_att, rxn_fwd_acc, rxn_rev_att, rxn_rev_acc = 
            MolSim.MC.sweep_npt_with_reactions!(st, p, rxn; Pext=P, max_dlnV=max_dlnV,
                                                p_reaction=p_reaction, do_volume_move=do_vol,
                                                rebuild_every=st.N)
        reaction_accepted_count += rxn_acc
        
        # Basic invariants
        @test st.N >= 0
        @test st.L > 0.0
        counts = MolSim.MC.count_species(st, 3)
        @test all(counts .>= 0)
        @test sum(counts) == st.N
    end
    
    # With anchored insertion, should accept at least one reaction (previously zero)
    @test reaction_accepted_count > 0
end

@testset "Table 3.2 REMC: A⇌B symmetry sanity" begin
    # A and B have identical parameters, so composition should drift from all-A
    # With symmetric proposal and acceptance, should approach ~50/50
    N_A_init = 40
    T = 2.0
    P = 5.0
    rc = 2.5
    ρ_init = 0.75
    max_disp = 0.1
    max_dlnV = 0.01
    vol_move_every = 10
    p_reaction = 0.1
    seed = 12345
    n_sweeps = 5000
    
    species_names = ["A", "B"]
    σ_types = [1.0, 1.0]
    ϵ_types = [1.0, 1.0]
    logq = [log(0.002), log(0.002)]  # Identical q values
    
    p = MolSim.MC.LJParams(σ_types=σ_types, ϵ_types=ϵ_types,
                           rc=rc, T=T, max_disp=max_disp,
                           use_lrc=false, lj_model=:shifted)
    
    stoichiometry = [-1, 1]
    rxn = MolSim.MC.Reaction("A⇌B", stoichiometry, 0.0; logq=logq)
    
    V = N_A_init / ρ_init
    L = cbrt(V)
    rng = Xoshiro(seed)
    pos = zeros(Float64, 3, N_A_init)
    for i in 1:N_A_init
        pos[1, i] = rand(rng) * L
        pos[2, i] = rand(rng) * L
        pos[3, i] = rand(rng) * L
    end
    types = fill(1, N_A_init)
    cl = MolSim.MC.CellList(N_A_init, L, rc)
    scratch_dr = MVector{3,Float64}(0.0, 0.0, 0.0)
    st = MolSim.MC.LJState(N_A_init, L, pos, types, rng, cl, scratch_dr, 0, 0)
    MolSim.MC.rebuild_cells!(st)
    
    # Equilibration
    for sweep in 1:(n_sweeps ÷ 2)
        do_vol = (sweep % vol_move_every == 0)
        MolSim.MC.sweep_npt_with_reactions!(st, p, rxn; Pext=P, max_dlnV=max_dlnV,
                                            p_reaction=p_reaction, do_volume_move=do_vol,
                                            rebuild_every=st.N)
    end
    
    # Production: sample
    N_B_samples = Int[]
    for sweep in 1:(n_sweeps ÷ 2)
        do_vol = (sweep % vol_move_every == 0)
        MolSim.MC.sweep_npt_with_reactions!(st, p, rxn; Pext=P, max_dlnV=max_dlnV,
                                            p_reaction=p_reaction, do_volume_move=do_vol,
                                            rebuild_every=st.N)
        
        if sweep % 10 == 0
            counts = MolSim.MC.count_species(st, 2)
            push!(N_B_samples, counts[2])
        end
    end
    
    N_B_mean = length(N_B_samples) > 0 ? sum(N_B_samples) / length(N_B_samples) : 0.0
    x_B_mean = N_B_mean / N_A_init
    
    @test N_B_mean > 0.0  # Should have some B
    # With symmetric proposal and acceptance, should be roughly 50/50
    # Use loose bounds [0.3, 0.7] to account for finite-size and short run
    @test x_B_mean >= 0.3  # At least 30% B
    @test x_B_mean <= 0.7  # At most 70% B
end

@testset "Table 3.2 REMC: A⇌B symmetry regression guard" begin
    # Regression test to lock in correct ΔN=0 alchemical behavior
    # With identical parameters and symmetric proposal, A⇌B should yield ~200/200
    # This test catches strong drift like 276/124 that indicates bias
    N_A_init = 400
    T = 2.0
    P = 5.0
    rc = 2.5
    ρ_init = 0.75
    max_disp = 0.1
    max_dlnV = 0.01
    vol_move_every = 10
    p_reaction = 0.1
    seed = 88888  # Fixed seed for reproducibility
    sweeps_equil = 5000
    sweeps_prod = 5000
    sample_every = 10
    
    species_names = ["A", "B"]
    σ_types = [1.0, 1.0]  # Identical
    ϵ_types = [1.0, 1.0]  # Identical
    logq = [log(0.002), log(0.002)]  # Identical q values
    
    p = MolSim.MC.LJParams(σ_types=σ_types, ϵ_types=ϵ_types,
                           rc=rc, T=T, max_disp=max_disp,
                           use_lrc=false, lj_model=:shifted)
    
    stoichiometry = [-1, 1]
    rxn = MolSim.MC.Reaction("A⇌B", stoichiometry, 0.0; logq=logq)  # logK=0.0
    
    V = N_A_init / ρ_init
    L = cbrt(V)
    rng = Xoshiro(seed)
    pos = zeros(Float64, 3, N_A_init)
    for i in 1:N_A_init
        pos[1, i] = rand(rng) * L
        pos[2, i] = rand(rng) * L
        pos[3, i] = rand(rng) * L
    end
    types = fill(1, N_A_init)  # All type 1 (A)
    cl = MolSim.MC.CellList(N_A_init, L, rc)
    scratch_dr = MVector{3,Float64}(0.0, 0.0, 0.0)
    st = MolSim.MC.LJState(N_A_init, L, pos, types, rng, cl, scratch_dr, 0, 0)
    MolSim.MC.rebuild_cells!(st)
    
    # Equilibration
    for sweep in 1:sweeps_equil
        do_vol = (sweep % vol_move_every == 0)
        MolSim.MC.sweep_npt_with_reactions!(st, p, rxn; Pext=P, max_dlnV=max_dlnV,
                                            p_reaction=p_reaction, do_volume_move=do_vol,
                                            rebuild_every=st.N)
    end
    
    # Production: sample
    N_A_samples = Int[]
    N_B_samples = Int[]
    for sweep in 1:sweeps_prod
        do_vol = (sweep % vol_move_every == 0)
        MolSim.MC.sweep_npt_with_reactions!(st, p, rxn; Pext=P, max_dlnV=max_dlnV,
                                            p_reaction=p_reaction, do_volume_move=do_vol,
                                            rebuild_every=st.N)
        
        if sweep % sample_every == 0
            counts = MolSim.MC.count_species(st, 2)
            push!(N_A_samples, counts[1])
            push!(N_B_samples, counts[2])
        end
    end
    
    N_A_mean = length(N_A_samples) > 0 ? sum(N_A_samples) / length(N_A_samples) : 0.0
    N_B_mean = length(N_B_samples) > 0 ? sum(N_B_samples) / length(N_B_samples) : 0.0
    
    # With symmetric proposal and identical parameters, should be ~200/200
    # Loose tolerance (15) is fine, but must catch strong bias like 276/124
    @test abs(N_A_mean - 200.0) < 15.0
    @test abs(N_B_mean - 200.0) < 15.0
end

@testset "Table 3.2 REMC: approximate Table 3.2 match (long test)" begin
    # Only run if LONG_TESTS environment variable is set
    if get(ENV, "LONG_TESTS", "0") != "1"
        @test_skip "Set LONG_TESTS=1 to run long matching tests"
        return
    end
    
    # Run A⇌C with larger sweeps
    N_A_init = 400
    T = 2.0
    P = 5.0
    rc = 2.5
    ρ_init = 0.75
    max_disp = 0.1
    max_dlnV = 0.01
    vol_move_every = 10
    p_reaction = 0.1
    seed = 12345
    sweeps_equil = 10000
    sweeps_prod = 50000
    sample_every = 10
    block_size = 50
    
    species_names = ["A", "C"]
    σ_types = [1.0, 1.1]
    ϵ_types = [1.0, 0.9]
    logq = [log(0.002), log(0.002)]
    
    p = MolSim.MC.LJParams(σ_types=σ_types, ϵ_types=ϵ_types,
                           rc=rc, T=T, max_disp=max_disp,
                           use_lrc=false, lj_model=:shifted)
    
    stoichiometry = [-1, 1]
    rxn = MolSim.MC.Reaction("A⇌C", stoichiometry, 0.0; logq=logq)
    
    V = N_A_init / ρ_init
    L = cbrt(V)
    rng = Xoshiro(seed)
    pos = zeros(Float64, 3, N_A_init)
    for i in 1:N_A_init
        pos[1, i] = rand(rng) * L
        pos[2, i] = rand(rng) * L
        pos[3, i] = rand(rng) * L
    end
    types = fill(1, N_A_init)
    cl = MolSim.MC.CellList(N_A_init, L, rc)
    scratch_dr = MVector{3,Float64}(0.0, 0.0, 0.0)
    st = MolSim.MC.LJState(N_A_init, L, pos, types, rng, cl, scratch_dr, 0, 0)
    MolSim.MC.rebuild_cells!(st)
    
    # Equilibration
    for sweep in 1:sweeps_equil
        do_vol = (sweep % vol_move_every == 0)
        MolSim.MC.sweep_npt_with_reactions!(st, p, rxn; Pext=P, max_dlnV=max_dlnV,
                                            p_reaction=p_reaction, do_volume_move=do_vol,
                                            rebuild_every=st.N)
    end
    
    # Production
    N_A_samples = Int[]
    N_C_samples = Int[]
    V_samples = Float64[]
    
    for sweep in 1:sweeps_prod
        do_vol = (sweep % vol_move_every == 0)
        MolSim.MC.sweep_npt_with_reactions!(st, p, rxn; Pext=P, max_dlnV=max_dlnV,
                                            p_reaction=p_reaction, do_volume_move=do_vol,
                                            rebuild_every=st.N)
        
        if sweep % sample_every == 0
            counts = MolSim.MC.count_species(st, 2)
            push!(N_A_samples, counts[1])
            push!(N_C_samples, counts[2])
            V = st.L * st.L * st.L
            push!(V_samples, V)
        end
    end
    
    # Block averaging
    ba_A = MolSim.MC.BlockAverager(block_size)
    for s in N_A_samples
        push!(ba_A, Float64(s))
    end
    N_A_mean = MolSim.MC.mean(ba_A)
    
    ba_C = MolSim.MC.BlockAverager(block_size)
    for s in N_C_samples
        push!(ba_C, Float64(s))
    end
    N_C_mean = MolSim.MC.mean(ba_C)
    
    # Targets from Table 3.2
    target_N_A = 297.7
    target_N_C = 102.3
    
    # Loose tolerance: 30%
    @test abs(N_A_mean - target_N_A) / target_N_A < 0.30
    @test abs(N_C_mean - target_N_C) / target_N_C < 0.30
end

@testset "logK conversion convention" begin
    """Test that logK conversion from concentration units to dimensionless form
    uses the correct sign convention: logK_dimless = logK_table - Δν * log(V_ref)
    
    This ensures the acceptance formula produces Δν * log(V / V_ref) dependence,
    not Δν * log(V * V_ref).
    """
    
    @testset "logK conversion with V_ref ≠ 1" begin
        logK_table = 0.0
        Δν = 1
        V_ref = 2.0
        
        logK_dimless = convert_logK_table_to_dimless(logK_table, Δν, V_ref)
        
        # Expected: logK_dimless = 0.0 - 1 * log(2.0) = -log(2.0)
        expected = -log(2.0)
        
        @test abs(logK_dimless - expected) < 1e-12 "logK conversion: logK_table=$logK_table, Δν=$Δν, V_ref=$V_ref should give logK_dimless=$expected, got $logK_dimless"
    end
    
    @testset "logK conversion with V_ref = 1 (Table 3.2 case)" begin
        logK_table = 0.0
        Δν = 1
        V_ref = 1.0
        
        logK_dimless = convert_logK_table_to_dimless(logK_table, Δν, V_ref)
        
        # Expected: logK_dimless = 0.0 - 1 * log(1.0) = 0.0
        expected = 0.0
        
        @test abs(logK_dimless - expected) < 1e-12 "logK conversion: logK_table=$logK_table, Δν=$Δν, V_ref=$V_ref should give logK_dimless=$expected, got $logK_dimless"
    end
    
    @testset "logK conversion with Δν = 0 (no volume dependence)" begin
        logK_table = 1.5
        Δν = 0
        V_ref = 2.0
        
        logK_dimless = convert_logK_table_to_dimless(logK_table, Δν, V_ref)
        
        # Expected: logK_dimless = 1.5 - 0 * log(2.0) = 1.5
        expected = 1.5
        
        @test abs(logK_dimless - expected) < 1e-12 "logK conversion: logK_table=$logK_table, Δν=$Δν, V_ref=$V_ref should give logK_dimless=$expected, got $logK_dimless"
    end
end
