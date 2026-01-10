"""
Pressure identity tests.
Use short MC runs (10^4-10^5 steps max) to verify pressure consistency.
"""

using Test
using MolSim

# Local helper functions for statistics
function mean(x::Vector{Float64})::Float64
    isempty(x) && return 0.0
    return sum(x) / length(x)
end

function std(x::Vector{Float64})::Float64
    isempty(x) && return 0.0
    length(x) == 1 && return 0.0
    m = mean(x)
    variance = sum((xi - m)^2 for xi in x) / (length(x) - 1)
    return sqrt(variance)
end

function stderr(x::Vector{Float64})::Float64
    isempty(x) && return 0.0
    return std(x) / sqrt(length(x))
end

"""
Helper function to compute thermodynamic pressure via FEP for a configuration.
Returns P_thermo estimate.
Uses the formula: P = N/(βV) + (1/β) * (ln⟨e^(-βΔU_+)⟩ - ln⟨e^(-βΔU_-)⟩) / (V_+ - V_-)
This ensures consistency with the Hamiltonian (respects lj_model).
"""
function compute_fep_pressure(st, p, T::Float64; epsV::Float64=1e-3, n_samples::Int=10)
    N = st.N
    L = st.L
    V = L * L * L
    
    # Helper to compute energy at given configuration
    # Uses the exact same logic as total_energy/lj_pair_u_from_r2 to ensure consistency
    function energy_at_configuration(pos::Matrix{Float64}, L_local::Float64, p_local)
        N_local = size(pos, 2)
        energy = 0.0
        rc2 = p_local.rc2
        L_half = L_local / 2.0
        
        # Use the same pair energy computation as total_energy (respects lj_model)
        for i in 1:N_local
            for j in (i+1):N_local
                dr_x = pos[1, j] - pos[1, i]
                dr_y = pos[2, j] - pos[2, i]
                dr_z = pos[3, j] - pos[3, i]
                
                # Apply minimum image convention
                if dr_x > L_half
                    dr_x = dr_x - L_local
                elseif dr_x < -L_half
                    dr_x = dr_x + L_local
                end
                if dr_y > L_half
                    dr_y = dr_y - L_local
                elseif dr_y < -L_half
                    dr_y = dr_y + L_local
                end
                if dr_z > L_half
                    dr_z = dr_z - L_local
                elseif dr_z < -L_half
                    dr_z = dr_z + L_local
                end
                
                r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
                
                if r2 < rc2 && r2 > 0.0
                    # Compute LJ potential using same logic as lj_pair_u_from_r2
                    # This ensures we respect lj_model (:shifted vs :truncated)
                    if r2 < 1e-14
                        energy += Inf
                        continue
                    end
                    σ2 = p_local.σ * p_local.σ
                    invr2 = σ2 / r2
                    invr6 = invr2 * invr2 * invr2
                    u_unshifted = 4.0 * p_local.ϵ * (invr6 * invr6 - invr6)
                    
                    if p_local.lj_model == :shifted
                        energy += u_unshifted - p_local.u_rc
                    else  # :truncated
                        energy += u_unshifted
                    end
                end
            end
        end
        
        # Add long-range correction if enabled (same as total_energy)
        if p_local.use_lrc
            energy += N_local * p_local.lrc_u_per_particle
        end
        
        return energy
    end
    
    β = 1.0 / T
    
    # Accumulate exp(-βΔU) for +epsV and -epsV separately
    exp_neg_beta_dU_plus = 0.0
    exp_neg_beta_dU_minus = 0.0
    
    for _ in 1:n_samples
        # Current configuration
        U_current = energy_at_configuration(st.pos, L, p)
        
        # Positive volume change: V' = V * (1 + epsV)
        V_new_pos = V * (1.0 + epsV)
        s_pos = (1.0 + epsV)^(1.0/3.0)
        L_new_pos = L * s_pos
        
        pos_plus = similar(st.pos)
        for j in 1:N
            pos_plus[1, j] = st.pos[1, j] * s_pos
            pos_plus[2, j] = st.pos[2, j] * s_pos
            pos_plus[3, j] = st.pos[3, j] * s_pos
        end
        
        # Wrap positions
        scratch = st.scratch_dr
        for j in 1:N
            scratch[1] = pos_plus[1, j]
            scratch[2] = pos_plus[2, j]
            scratch[3] = pos_plus[3, j]
            MolSim.MC.wrap!(scratch, L_new_pos)
            pos_plus[1, j] = scratch[1]
            pos_plus[2, j] = scratch[2]
            pos_plus[3, j] = scratch[3]
        end
        
        U_new_pos = energy_at_configuration(pos_plus, L_new_pos, p)
        ΔU_pos = U_new_pos - U_current
        exp_neg_beta_dU_plus += exp(-β * ΔU_pos)
        
        # Negative volume change: V' = V * (1 - epsV)
        V_new_neg = V * (1.0 - epsV)
        s_neg = (1.0 - epsV)^(1.0/3.0)
        L_new_neg = L * s_neg
        
        pos_minus = similar(st.pos)
        for j in 1:N
            pos_minus[1, j] = st.pos[1, j] * s_neg
            pos_minus[2, j] = st.pos[2, j] * s_neg
            pos_minus[3, j] = st.pos[3, j] * s_neg
        end
        
        for j in 1:N
            scratch[1] = pos_minus[1, j]
            scratch[2] = pos_minus[2, j]
            scratch[3] = pos_minus[3, j]
            MolSim.MC.wrap!(scratch, L_new_neg)
            pos_minus[1, j] = scratch[1]
            pos_minus[2, j] = scratch[2]
            pos_minus[3, j] = scratch[3]
        end
        
        U_new_neg = energy_at_configuration(pos_minus, L_new_neg, p)
        ΔU_neg = U_new_neg - U_current
        exp_neg_beta_dU_minus += exp(-β * ΔU_neg)
    end
    
    # Compute mean(exp(-βΔU)) for each direction
    mean_exp_plus = exp_neg_beta_dU_plus / n_samples
    mean_exp_minus = exp_neg_beta_dU_minus / n_samples
    
    # Compute free energy changes for +epsV (matching dev/long_nvt_run.jl approach)
    V_new_pos = V * (1.0 + epsV)
    ΔV_pos = V_new_pos - V  # = V * epsV
    ΔF_conf_pos = -T * log(max(mean_exp_plus, 1e-100))
    log_V_ratio_pos = log(V_new_pos / V)
    ΔF_ideal_pos = -N * T * log_V_ratio_pos
    ΔF_total_pos = ΔF_conf_pos + ΔF_ideal_pos
    P_fep_pos = -ΔF_total_pos / ΔV_pos
    
    # Compute free energy changes for -epsV
    V_new_neg = V * (1.0 - epsV)
    ΔV_neg = V_new_neg - V  # = -V * epsV
    ΔF_conf_neg = -T * log(max(mean_exp_minus, 1e-100))
    log_V_ratio_neg = log(V_new_neg / V)
    ΔF_ideal_neg = -N * T * log_V_ratio_neg
    ΔF_total_neg = ΔF_conf_neg + ΔF_ideal_neg
    P_fep_neg = -ΔF_total_neg / ΔV_neg
    
    # Centered estimate: average the two pressure estimates (same as dev/long_nvt_run.jl)
    P_thermo = 0.5 * (P_fep_pos + P_fep_neg)
    
    return P_thermo
end

@testset "Unshifted LJ: P_virial ≠ P_thermo" begin
    # Short MC run with truncated LJ
    N = 32
    ρ = 0.5
    T = 2.0
    rc = 2.5
    seed = 12345
    n_sweeps = 5000
    
    p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=seed, 
                                lj_model=:truncated, use_lrc=false)
    
    # Warmup
    for _ in 1:1000
        MolSim.MC.sweep!(st, p)
    end
    
    # Sample virial pressure
    P_virial_samples = Float64[]
    for _ in 1:n_sweeps
        MolSim.MC.sweep!(st, p)
        P_virial = MolSim.MC.pressure(st, p, T)
        push!(P_virial_samples, P_virial)
    end
    
    P_virial_mean = mean(P_virial_samples)
    P_virial_stderr = std(P_virial_samples) / sqrt(length(P_virial_samples))
    
    # Sample thermodynamic pressure (FEP) - use smaller sample for speed
    P_thermo_samples = Float64[]
    sample_every = 10
    for sweep_idx in 1:n_sweeps
        MolSim.MC.sweep!(st, p)
        if sweep_idx % sample_every == 0
            P_thermo = compute_fep_pressure(st, p, T; n_samples=5)
            push!(P_thermo_samples, P_thermo)
        end
    end
    
    P_thermo_mean = mean(P_thermo_samples)
    P_thermo_stderr = std(P_thermo_samples) / sqrt(length(P_thermo_samples))
    
    # Difference should be statistically significant
    diff = abs(P_virial_mean - P_thermo_mean)
    combined_error = sqrt(P_virial_stderr^2 + P_thermo_stderr^2)
    
    # For truncated LJ, difference should be O(ρ²) and significant
    expected_scale = ρ * ρ
    @test diff > 2.0 * combined_error
    @test diff > 0.01 * expected_scale
end

@testset "Shifted LJ without impulse: P_virial ≠ P_thermo" begin
    # Short MC run with shifted LJ, WITHOUT impulsive correction
    # Expected: mismatch because force is discontinuous at rc
    N = 32
    ρ = 0.5
    T = 2.0
    rc = 2.5
    seed = 54321
    n_sweeps = 5000
    
    println("Shifted LJ test (WITHOUT impulsive correction):")
    
    p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=seed,
                                lj_model=:shifted, use_lrc=false, apply_impulsive_correction=false)
    
    # Verify model flags are correct
    @test p.lj_model == :shifted
    @test p.apply_impulsive_correction == false
    
    # Warmup
    for _ in 1:1000
        MolSim.MC.sweep!(st, p)
    end
    
    # Sample virial pressure
    P_virial_samples = Float64[]
    for _ in 1:n_sweeps
        MolSim.MC.sweep!(st, p)
        P_virial = MolSim.MC.pressure(st, p, T)
        push!(P_virial_samples, P_virial)
    end
    
    P_virial_mean = mean(P_virial_samples)
    P_virial_stderr = std(P_virial_samples) / sqrt(length(P_virial_samples))
    
    # Sample thermodynamic pressure (FEP)
    P_thermo_samples = Float64[]
    sample_every = 10
    for sweep_idx in 1:n_sweeps
        MolSim.MC.sweep!(st, p)
        if sweep_idx % sample_every == 0
            P_thermo = compute_fep_pressure(st, p, T; n_samples=5)
            push!(P_thermo_samples, P_thermo)
        end
    end
    
    P_thermo_mean = mean(P_thermo_samples)
    P_thermo_stderr = std(P_thermo_samples) / sqrt(length(P_thermo_samples))
    
    # Compute excess pressures
    P_id = ρ * T
    P_virial_ex = P_virial_mean - P_id
    P_thermo_ex = P_thermo_mean - P_id
    ΔP_ex = P_virial_ex - P_thermo_ex
    
    println("  P_virial_mean = ", P_virial_mean)
    println("  P_thermo_mean = ", P_thermo_mean)
    println("  ΔP_ex = ", ΔP_ex)
    
    # Expect statistically significant mismatch
    diff = abs(P_virial_mean - P_thermo_mean)
    combined_error = sqrt(P_virial_stderr^2 + P_thermo_stderr^2)
    z_score = diff / combined_error
    
    # Assert mismatch: either z_score > 3 OR |ΔP_ex| > 0.02
    @test (z_score > 3.0) || (abs(ΔP_ex) > 0.02)
end

@testset "Shifted LJ with impulse: P_virial ≈ P_thermo" begin
    # Short MC run with shifted LJ, WITH impulsive correction
    # Expected: agreement because impulsive correction accounts for force discontinuity
    N = 32
    ρ = 0.5
    T = 2.0
    rc = 2.5
    seed = 54322  # Different seed
    n_sweeps = 5000
    
    println("Shifted LJ test (WITH impulsive correction):")
    
    p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=seed,
                                lj_model=:shifted, use_lrc=false, apply_impulsive_correction=true)
    
    # Verify model flags are correct
    @test p.lj_model == :shifted
    @test p.apply_impulsive_correction == true
    
    # Warmup
    for _ in 1:1000
        MolSim.MC.sweep!(st, p)
    end
    
    # Sample virial pressure (includes impulsive correction)
    P_virial_samples = Float64[]
    for _ in 1:n_sweeps
        MolSim.MC.sweep!(st, p)
        P_virial = MolSim.MC.pressure(st, p, T)
        push!(P_virial_samples, P_virial)
    end
    
    P_virial_mean = mean(P_virial_samples)
    P_virial_stderr = std(P_virial_samples) / sqrt(length(P_virial_samples))
    
    # Sample thermodynamic pressure (FEP)
    # Note: FEP computes thermodynamic pressure directly from energy changes
    # The impulsive correction is only needed for virial estimator
    # So we do NOT apply correction to FEP - it should already be correct
    P_thermo_samples = Float64[]
    sample_every = 10
    for sweep_idx in 1:n_sweeps
        MolSim.MC.sweep!(st, p)
        if sweep_idx % sample_every == 0
            P_thermo = compute_fep_pressure(st, p, T; n_samples=10)  # Increase samples for better statistics
            push!(P_thermo_samples, P_thermo)
        end
    end
    
    P_thermo_mean = mean(P_thermo_samples)
    P_thermo_stderr = std(P_thermo_samples) / sqrt(length(P_thermo_samples))
    
    # Compute excess pressures
    P_id = ρ * T
    P_virial_ex = P_virial_mean - P_id
    P_thermo_ex = P_thermo_mean - P_id
    ΔP_ex = P_virial_ex - P_thermo_ex
    
    println("  P_virial_mean = ", P_virial_mean)
    println("  P_thermo_mean = ", P_thermo_mean)
    println("  ΔP_ex = ", ΔP_ex)
    
    # Regression check: verify correction changes the result
    # The mismatch with correction should be smaller than without correction
    # We know from the "without impulse" test that ΔP_ex ≈ -0.076 without correction
    # With correction, it should be closer to zero
    println("  Expected: |ΔP_ex| should be smaller with correction than without (~0.076)")
    println("  Actual |ΔP_ex| with correction = ", abs(ΔP_ex))
    
    # Regression: correction should reduce mismatch magnitude significantly
    # Allow some tolerance since we're comparing different runs
    @test abs(ΔP_ex) < 0.05  # Should be much smaller than ~0.076 without correction
    
    # For shifted LJ with correction, they should agree within ~2-3σ
    diff = abs(P_virial_mean - P_thermo_mean)
    combined_error = sqrt(P_virial_stderr^2 + P_thermo_stderr^2)
    z_score = diff / combined_error
    
    @test z_score < 3.0
end
