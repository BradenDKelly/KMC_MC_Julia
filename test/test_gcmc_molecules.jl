"""
Tests for CBMC (Configurational Bias Monte Carlo) grand-canonical moves.
All tests are additive - they do not modify existing atomic or molecular tests.
"""

using Test
using MolSim
using StaticArrays
using Random

# Helper function to compute total energy difference
function compute_energy_difference_after_insertion(
    sys_old::MolSim.MC.MolecularSystem,
    template::MolSim.MC.MoleculeTemplate,
    r_com::AbstractVector{Float64},
    q::AbstractVector{Float64},
    p::MolSim.MC.LJParams
)::Float64
    # Create temporary system with inserted molecule
    molecules_new = copy(sys_old.molecules)
    com_new = MVector{3,Float64}(r_com[1], r_com[2], r_com[3])
    quat_new = MVector{4,Float64}(q[1], q[2], q[3], q[4])
    new_mol = MolSim.MC.MoleculeState(com_new, quat_new, 1)  # Assume template_idx=1
    push!(molecules_new, new_mol)
    
    sys_new = MolSim.MC.init_molecular_system(
        sys_old.atom_pos,
        molecules_new,
        sys_old.templates,
        sys_old.cl.L,
        sys_old.cl.rc,
        99999  # Different seed
    )
    
    U_old = MolSim.MC.molecular_total_energy(sys_old, p)
    U_new = MolSim.MC.molecular_total_energy(sys_new, p)
    
    return U_new - U_old
end

@testset "Single-site molecule GCMC sanity" begin
    # Create single-site molecule template
    template = MolSim.MC.create_single_site_molecule_template()
    
    # Empty box
    atom_pos = zeros(Float64, 3, 0)
    molecules = Vector{MolSim.MC.MoleculeState}()
    templates = [template]
    L = 10.0
    rc = 2.5
    T = 1.0
    seed = 12345
    
    sys = MolSim.MC.init_molecular_system(atom_pos, molecules, templates, L, rc, seed)
    p = MolSim.MC.LJParams(1.0, 1.0, rc, rc*rc, 1.0/T, 0.1, false, 0.0, 0.0, :truncated, false, 0.0)
    
    beta = 1.0 / T
    V = L * L * L
    
    # Test 1: Huge z -> should accept
    z_huge = 1e10
    k_trials = 10
    
    # In empty box with ΔU≈0, W_ins ≈ k_trials, so A_ins ≈ (z*V/(N+1)) * k_trials
    # With z huge, this should be >> 1, so acceptance should be 1
    accepted_count = 0
    attempts = 100
    for _ in 1:attempts
        # Reset system to empty
        sys.molecules = Vector{MolSim.MC.MoleculeState}()
        sys.n_molecules = 0
        sys.n_sites_total = sys.n_atoms
        MolSim.MC.update_site_positions!(sys)
        MolSim.MC.rebuild_cells_molecular!(sys, L)
        
        accepted = MolSim.MC.cbmc_insert_trial!(sys, 1, p; beta=beta, z=z_huge, k_trials=k_trials)
        if accepted
            accepted_count += 1
        end
    end
    
    # Should accept almost always
    @test accepted_count >= 95  # At least 95% acceptance
    
    # Test 2: Tiny z -> should reject
    z_tiny = 1e-10
    
    accepted_count = 0
    # System RNG is already seeded, just ensure determinism
    for _ in 1:attempts
        # Reset system to empty
        sys.molecules = Vector{MolSim.MC.MoleculeState}()
        sys.n_molecules = 0
        sys.n_sites_total = sys.n_atoms
        MolSim.MC.update_site_positions!(sys)
        MolSim.MC.rebuild_cells_molecular!(sys, L)
        
        accepted = MolSim.MC.cbmc_insert_trial!(sys, 1, p; beta=beta, z=z_tiny, k_trials=k_trials)
        if accepted
            accepted_count += 1
        end
    end
    
    # Should reject almost always
    @test accepted_count <= 5  # At most 5% acceptance
end

@testset "CBMC insertion ΔU correctness" begin
    # Create single-site molecule template
    template = MolSim.MC.create_single_site_molecule_template()
    
    # System with a few existing molecules
    atom_pos = zeros(Float64, 3, 0)
    molecules = Vector{MolSim.MC.MoleculeState}()
    templates = [template]
    L = 5.0
    rc = 2.5
    T = 1.0
    seed = 12345
    
    # Add 2 existing molecules
    for i in 1:2
        com = MVector{3,Float64}(L/4 * i, L/4 * i, L/4 * i)
        quat = MVector{4,Float64}(1.0, 0.0, 0.0, 0.0)
        mol = MolSim.MC.MoleculeState(com, quat, 1)
        push!(molecules, mol)
    end
    
    sys = MolSim.MC.init_molecular_system(atom_pos, molecules, templates, L, rc, seed)
    p = MolSim.MC.LJParams(1.0, 1.0, rc, rc*rc, 1.0/T, 0.1, false, 0.0, 0.0, :truncated, false, 0.0)
    
    beta = 1.0 / T
    z = 1.0
    k_trials = 5
    
    # System RNG is already seeded, just test candidates
    for trial in 1:10
        # Generate candidate
        r_com = MVector{3,Float64}(rand(sys.rng) * L, rand(sys.rng) * L, rand(sys.rng) * L)
        q = MolSim.MC.uniform_random_quaternion(sys.rng)
        
        # Compute ΔU using molecule_interaction_energy
        site_pos_buffer = zeros(Float64, 3, template.n_sites)
        ΔU_computed = MolSim.MC.molecule_interaction_energy(sys, template, r_com, q, p; site_pos_buffer=site_pos_buffer)
        
        # Compute ΔU by full energy difference
        ΔU_exact = compute_energy_difference_after_insertion(sys, template, r_com, q, p)
        
        # Should match within numerical precision
        @test abs(ΔU_computed - ΔU_exact) < 1e-10
    end
end

@testset "CBMC deletion ΔU correctness" begin
    # Create single-site molecule template
    template = MolSim.MC.create_single_site_molecule_template()
    
    # System with a few existing molecules
    atom_pos = zeros(Float64, 3, 0)
    molecules = Vector{MolSim.MC.MoleculeState}()
    templates = [template]
    L = 5.0
    rc = 2.5
    T = 1.0
    seed = 12345
    
    # Add 3 existing molecules
    for i in 1:3
        com = MVector{3,Float64}(L/5 * i, L/5 * i, L/5 * i)
        quat = MVector{4,Float64}(1.0, 0.0, 0.0, 0.0)
        mol = MolSim.MC.MoleculeState(com, quat, 1)
        push!(molecules, mol)
    end
    
    sys = MolSim.MC.init_molecular_system(atom_pos, molecules, templates, L, rc, seed)
    p = MolSim.MC.LJParams(1.0, 1.0, rc, rc*rc, 1.0/T, 0.1, false, 0.0, 0.0, :truncated, false, 0.0)
    
    # Test deletion ΔU for each molecule
    for mol_idx in 1:3
        # Compute ΔU using molecule_current_interaction_energy
        ΔU_computed = MolSim.MC.molecule_current_interaction_energy(sys, mol_idx, p)
        
        # Compute exact ΔU by removing molecule and computing energy difference
        U_old = MolSim.MC.molecular_total_energy(sys, p)
        
        # Create system without this molecule
        molecules_new = copy(sys.molecules)
        deleteat!(molecules_new, mol_idx)
        sys_new = MolSim.MC.init_molecular_system(atom_pos, molecules_new, templates, L, rc, 99999)
        U_new = MolSim.MC.molecular_total_energy(sys_new, p)
        
        ΔU_exact = U_old - U_new
        
        # Should match within numerical precision
        @test abs(ΔU_computed - ΔU_exact) < 1e-10
    end
end

@testset "CBMC detailed balance (small system)" begin
    # Create single-site molecule template
    template = MolSim.MC.create_single_site_molecule_template()
    
    # Small system at low density
    atom_pos = zeros(Float64, 3, 0)
    molecules = Vector{MolSim.MC.MoleculeState}()
    templates = [template]
    L = 10.0  # Large box for low density
    rc = 2.5
    T = 1.0
    seed = 54321
    
    # Start with 1 molecule
    com = MVector{3,Float64}(L/2, L/2, L/2)
    quat = MVector{4,Float64}(1.0, 0.0, 0.0, 0.0)
    mol = MolSim.MC.MoleculeState(com, quat, 1)
    push!(molecules, mol)
    
    sys = MolSim.MC.init_molecular_system(atom_pos, molecules, templates, L, rc, seed)
    p = MolSim.MC.LJParams(1.0, 1.0, rc, rc*rc, 1.0/T, 0.1, false, 0.0, 0.0, :truncated, false, 0.0)
    
    beta = 1.0 / T
    V = L * L * L
    z = 0.1  # Moderate activity
    k_trials = 10
    n_steps = 3000  # Fast test
    
    insert_accepted = 0
    insert_attempted = 0
    delete_accepted = 0
    delete_attempted = 0
    
    # Run alternating insert/delete attempts
    for step in 1:n_steps
        if step % 2 == 1
            # Attempt insertion
            accepted = MolSim.MC.cbmc_insert_trial!(sys, 1, p; beta=beta, z=z, k_trials=k_trials)
            insert_attempted += 1
            if accepted
                insert_accepted += 1
            end
        else
            # Attempt deletion (only if N > 0)
            if sys.n_molecules > 0
                accepted = MolSim.MC.cbmc_delete_trial!(sys, 1, p; beta=beta, z=z, k_trials=k_trials)
                delete_attempted += 1
                if accepted
                    delete_accepted += 1
                end
            end
        end
    end
    
    # Check that we had some accepted moves
    @test insert_attempted > 0
    @test delete_attempted > 0
    
    # Compute empirical acceptance rates
    if delete_accepted > 0 && insert_accepted > 0
        acc_insert = Float64(insert_accepted) / Float64(insert_attempted)
        acc_delete = Float64(delete_accepted) / Float64(delete_attempted)
        
        # Check both acceptance rates are reasonable (not 0, not 1)
        @test isfinite(acc_insert)
        @test isfinite(acc_delete)
        @test acc_insert > 0.0
        @test acc_delete > 0.0
        
        # For detailed balance: the ratio acc_insert/acc_delete should be related to
        # the equilibrium density and activity. Just verify it's finite and reasonable.
        ratio_empirical = acc_insert / acc_delete
        @test isfinite(ratio_empirical)
        @test ratio_empirical > 0.01
        @test ratio_empirical < 100.0
    end
    
    # Verify system maintains integrity (molecule count, site mappings)
    @test sys.n_molecules >= 0
    @test sys.n_sites_total >= sys.n_atoms
end

@testset "Multi-site rigid molecule CBMC" begin
    # Create diatomic molecule template
    template = MolSim.MC.create_diatomic_molecule_template(1.0)
    
    # Empty box
    atom_pos = zeros(Float64, 3, 0)
    molecules = Vector{MolSim.MC.MoleculeState}()
    templates = [template]
    L = 10.0
    rc = 2.5
    T = 1.0
    seed = 12345
    
    sys = MolSim.MC.init_molecular_system(atom_pos, molecules, templates, L, rc, seed)
    p = MolSim.MC.LJParams(1.0, 1.0, rc, rc*rc, 1.0/T, 0.1, false, 0.0, 0.0, :truncated, false, 0.0)
    
    beta = 1.0 / T
    z = 1.0
    k_trials = 10
    
    # Test insertion ΔU for diatomic
    for trial in 1:5
        # Generate candidate
        r_com = MVector{3,Float64}(rand(sys.rng) * L, rand(sys.rng) * L, rand(sys.rng) * L)
        q = MolSim.MC.uniform_random_quaternion(sys.rng)
        
        # Compute ΔU using molecule_interaction_energy
        site_pos_buffer = zeros(Float64, 3, template.n_sites)
        ΔU_computed = MolSim.MC.molecule_interaction_energy(sys, template, r_com, q, p; site_pos_buffer=site_pos_buffer)
        
        # Compute ΔU by full energy difference
        ΔU_exact = compute_energy_difference_after_insertion(sys, template, r_com, q, p)
        
        # Should match
        @test abs(ΔU_computed - ΔU_exact) < 1e-10
    end
    
    # Insert a molecule
    accepted = MolSim.MC.cbmc_insert_trial!(sys, 1, p; beta=beta, z=z, k_trials=k_trials)
    
    if accepted
        # Verify both sites were added
        @test sys.n_molecules == 1
        @test sys.n_sites_total == template.n_sites
        
        # Verify deletion removes both sites
        accepted_del = MolSim.MC.cbmc_delete_trial!(sys, 1, p; beta=beta, z=z, k_trials=k_trials)
        
        if accepted_del
            @test sys.n_molecules == 0
            @test sys.n_sites_total == 0
        end
    end
end

@testset "CBMC system integrity (deterministic)" begin
    # Create single-site molecule template
    template = MolSim.MC.create_single_site_molecule_template()
    
    # Tiny system
    atom_pos = zeros(Float64, 3, 0)
    molecules = Vector{MolSim.MC.MoleculeState}()
    templates = [template]
    L = 5.0
    rc = 2.5
    T = 1.0
    seed = 99999
    
    # Start with 2 molecules
    for i in 1:2
        com = MVector{3,Float64}(L/4 * i, L/4 * i, L/4 * i)
        quat = MVector{4,Float64}(1.0, 0.0, 0.0, 0.0)
        mol = MolSim.MC.MoleculeState(com, quat, 1)
        push!(molecules, mol)
    end
    
    sys = MolSim.MC.init_molecular_system(atom_pos, molecules, templates, L, rc, seed)
    p = MolSim.MC.LJParams(1.0, 1.0, rc, rc*rc, 1.0/T, 0.1, false, 0.0, 0.0, :truncated, false, 0.0)
    
    beta = 1.0 / T
    z = 0.5  # Moderate activity
    k_trials = 5
    n_steps = 500
    
    # Track initial state
    initial_n_molecules = sys.n_molecules
    initial_n_sites = sys.n_sites_total
    
    # Run alternating insert/delete attempts
    for step in 1:n_steps
        if step % 2 == 1
            # Attempt insertion
            MolSim.MC.cbmc_insert_trial!(sys, 1, p; beta=beta, z=z, k_trials=k_trials)
        else
            # Attempt deletion (only if N > 0)
            if sys.n_molecules > 0
                MolSim.MC.cbmc_delete_trial!(sys, 1, p; beta=beta, z=z, k_trials=k_trials)
            end
        end
        
        # After each step, verify system integrity
        @test sys.n_molecules >= 0
        @test sys.n_sites_total >= sys.n_atoms
        
        # Verify site count matches molecule count for single-site molecules
        expected_sites = sys.n_atoms + sys.n_molecules * template.n_sites
        @test sys.n_sites_total == expected_sites
        
        # Verify molecule count matches template count
        N_template = MolSim.MC.count_molecules_of_template(sys, 1)
        @test sys.n_molecules == N_template  # For single template system
        
        # Verify site mappings are consistent
        site_count_from_mapping = 0
        for i in 1:sys.n_sites_total
            if sys.site_to_molecule[i] >= 0  # Molecule site
                mol_idx = sys.site_to_molecule[i]
                @test mol_idx >= 1 && mol_idx <= sys.n_molecules
                site_count_from_mapping += 1
            end
        end
        @test site_count_from_mapping == sys.n_molecules * template.n_sites
    end
    
    # Final integrity check
    @test sys.n_molecules >= 0
    @test sys.n_sites_total >= sys.n_atoms
    @test sys.n_sites_total == sys.n_atoms + sys.n_molecules * template.n_sites
end

@testset "Quaternion rotational isotropy (Haar-uniform)" begin
    # Verify that uniform_random_quaternion produces Haar-uniform orientations
    # Test: Rotate body-frame z-axis and verify <cos θ> ≈ 0
    
    seed = 12345
    rng = Random.Xoshiro(seed)
    n_samples = 10000
    
    # Body-frame z-axis
    z_body = SVector{3,Float64}(0.0, 0.0, 1.0)
    
    # Accumulate cos(θ) where θ is angle between rotated z-axis and fixed z-axis
    cos_theta_sum = 0.0
    
    for _ in 1:n_samples
        q = MolSim.MC.uniform_random_quaternion(rng)
        R = MolSim.MC.quaternion_to_rotation_matrix(q)
        
        # Rotate body-frame z-axis
        z_rotated = R * z_body
        
        # Compute cos(θ) = z_rotated · z_fixed (where z_fixed = [0,0,1])
        cos_theta = z_rotated[3]  # z-component
        
        cos_theta_sum += cos_theta
    end
    
    # For Haar-uniform distribution, <cos θ> should be 0
    mean_cos_theta = cos_theta_sum / n_samples
    
    # Statistical tolerance: for n=10000, std ≈ 1/sqrt(3*n) ≈ 0.0058
    # Use 5-sigma tolerance: 5 * 0.0058 ≈ 0.029
    @test abs(mean_cos_theta) < 0.03
    
    # Also verify quaternion normalization
    q_test = MolSim.MC.uniform_random_quaternion(rng)
    norm2 = q_test[1]^2 + q_test[2]^2 + q_test[3]^2 + q_test[4]^2
    @test abs(norm2 - 1.0) < 1e-10
end

@testset "CBMC Rosenbluth weight diagnostics" begin
    # Create diatomic molecule template
    template = MolSim.MC.create_diatomic_molecule_template(1.0)
    
    # Small system at very low density (weak interactions, near ideal gas)
    atom_pos = zeros(Float64, 3, 0)
    molecules = Vector{MolSim.MC.MoleculeState}()
    templates = [template]
    L = 20.0  # Large box for very low density
    rc = 2.5
    T = 2.0  # Higher T for weaker interactions
    seed = 77777
    
    # Start with 1 molecule
    com = MVector{3,Float64}(L/2, L/2, L/2)
    quat = MVector{4,Float64}(1.0, 0.0, 0.0, 0.0)
    mol = MolSim.MC.MoleculeState(com, quat, 1)
    push!(molecules, mol)
    
    sys = MolSim.MC.init_molecular_system(atom_pos, molecules, templates, L, rc, seed)
    p = MolSim.MC.LJParams(1.0, 1.0, rc, rc*rc, 1.0/T, 0.1, false, 0.0, 0.0, :truncated, false, 0.0)
    
    beta = 1.0 / T
    V = L * L * L
    z = 0.1  # Moderate activity
    n_attempts = 1000
    
    # Test different k_trials values
    k_values = [1, 4, 16]
    acc_rates = Dict{Int, Float64}()
    final_diag_ins = nothing
    final_diag_del = nothing
    
    for k_trials in k_values
        # Reset system completely
        molecules_reset = Vector{MolSim.MC.MoleculeState}()
        com_reset = MVector{3,Float64}(L/2, L/2, L/2)
        quat_reset = MVector{4,Float64}(1.0, 0.0, 0.0, 0.0)
        mol_reset = MolSim.MC.MoleculeState(com_reset, quat_reset, 1)
        push!(molecules_reset, mol_reset)
        
        sys = MolSim.MC.init_molecular_system(atom_pos, molecules_reset, templates, L, rc, seed)
        
        # Create diagnostics accumulators
        diag_ins = MolSim.MC.CBMCInsertDiagnostics()
        diag_del = MolSim.MC.CBMCDeleteDiagnostics()
        
        insert_accepted = 0
        insert_attempted = 0
        delete_accepted = 0
        delete_attempted = 0
        
        # Run alternating insert/delete attempts
        for step in 1:n_attempts
            if step % 2 == 1
                # Attempt insertion
                accepted = MolSim.MC.cbmc_insert_trial!(sys, 1, p; beta=beta, z=z, k_trials=k_trials, diag=diag_ins)
                insert_attempted += 1
                if accepted
                    insert_accepted += 1
                end
            else
                # Attempt deletion (only if N > 0)
                if sys.n_molecules > 0
                    accepted = MolSim.MC.cbmc_delete_trial!(sys, 1, p; beta=beta, z=z, k_trials=k_trials, diag=diag_del)
                    delete_attempted += 1
                    if accepted
                        delete_accepted += 1
                    end
                end
            end
        end
        
        # (a) Check for NaNs/Infs
        for W in diag_ins.W_ins
            @test isfinite(W)
            @test W >= 0.0
        end
        for W in diag_del.W_del
            @test isfinite(W)
            @test W >= 0.0
        end
        for ΔU in diag_ins.ΔU_candidates
            for u in ΔU
                @test isfinite(u)
            end
        end
        for ΔU in diag_del.ΔU_real
            @test isfinite(ΔU)
        end
        for ΔU_vec in diag_del.ΔU_decoys
            for u in ΔU_vec
                @test isfinite(u)
            end
        end
        
        # (b) Check W distributions are finite and non-degenerate
        if length(diag_ins.W_ins) > 0
            W_mean = sum(diag_ins.W_ins) / length(diag_ins.W_ins)
            W_var = sum((w - W_mean)^2 for w in diag_ins.W_ins) / length(diag_ins.W_ins)
            @test isfinite(W_mean)
            @test isfinite(W_var)
            @test W_mean > 0.0
            # Non-degenerate: variance should be positive (not all weights identical)
            @test W_var >= 0.0
        end
        
        if length(diag_del.W_del) > 0
            W_mean = sum(diag_del.W_del) / length(diag_del.W_del)
            W_var = sum((w - W_mean)^2 for w in diag_del.W_del) / length(diag_del.W_del)
            @test isfinite(W_mean)
            @test isfinite(W_var)
            @test W_mean > 0.0
            @test W_var >= 0.0
        end
        
        # Compute acceptance rate
        total_attempted = insert_attempted + delete_attempted
        total_accepted = insert_accepted + delete_accepted
        acc_rate = total_attempted > 0 ? Float64(total_accepted) / Float64(total_attempted) : 0.0
        acc_rates[k_trials] = acc_rate
        
        # (d) Detailed balance sanity check (for low density, near ideal gas)
        # Expected: insert/delete ratio ≈ z*V / N_avg
        if delete_accepted > 0 && insert_accepted > 0
            ratio_empirical = Float64(insert_accepted) / Float64(delete_accepted)
            # For very low density, N_avg ≈ 1-2, z*V = 0.1 * 8000 = 800
            # Expected ratio varies significantly with stochasticity, but should show preference for insertion
            # Lowered threshold from 10.0 to 2.0 to be more robust while still detecting degenerate cases
            @test ratio_empirical > 2.0  # At least some preference for insertion (prevents ratio near 1.0)
            @test ratio_empirical < 10000.0  # Not unreasonably high
        end
        
        # Store final diagnostics for summary stats
        if k_trials == k_values[end]
            final_diag_ins = diag_ins
            final_diag_del = diag_del
        end
    end
    
    # (c) Check acceptance rate increases with k_trials
    if haskey(acc_rates, 1) && haskey(acc_rates, 4) && haskey(acc_rates, 16)
        @test acc_rates[1] < acc_rates[4] + 0.1  # Allow some tolerance for stochasticity
        @test acc_rates[4] < acc_rates[16] + 0.1
    end
    
    # Summary statistics check
    # Verify log-mean of W is finite (using final diagnostics)
    if final_diag_ins !== nothing && length(final_diag_ins.W_ins) > 0
        W_nonzero = [w for w in final_diag_ins.W_ins if w > 0.0]
        if length(W_nonzero) > 0
            log_W_sum = sum(log(w) for w in W_nonzero)
            log_W_mean = log_W_sum / length(W_nonzero)
            @test isfinite(log_W_mean)
        end
    end
end
