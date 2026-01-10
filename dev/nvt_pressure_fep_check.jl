"""
NVT Pressure Check via Free-Energy Perturbation (FEP)

Uses virtual volume changes to compute pressure via:
  ΔF = -kT * log(⟨exp(-βΔU)⟩)
  P = -ΔF / ΔV

This provides an independent check of the virial pressure estimator.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

# Parameters
N = 864
T = 2.0
rho = 0.2
rc = 2.5
max_disp = 0.1
use_lrc = false
warmup_sweeps = 5000
prod_sweeps = 50000
sample_every = 50
epsV_list = [+1e-3, -1e-3, +2e-3, -2e-3]  # relative ΔV/V
seed = 12345

println("=" ^ 80)
println("NVT Pressure Check via Free-Energy Perturbation")
println("=" ^ 80)
println()
println("Parameters:")
println("  N = $N")
println("  T = $T")
println("  rho = $rho")
println("  rc = $rc")
println("  warmup_sweeps = $warmup_sweeps")
println("  prod_sweeps = $prod_sweeps")
println("  sample_every = $sample_every")
println("  epsV_list = $epsV_list")
println()

# Helper function: compute energy for given positions and box length
# This does NOT modify the simulation state
# Explicit guard: rc must be exactly as specified in p.rc (no clamping)
function energy_at_configuration(pos::Matrix{Float64}, L::Float64, p::MolSim.MC.LJParams)::Float64
    energy = 0.0
    N_local = size(pos, 2)
    rc2 = p.rc2  # Use exactly rc from params (no min(rc, L/2) or other clamping)
    rc_actual = p.rc  # Actual cutoff radius
    L_half = L / 2.0
    
    # Guard: rc should be <= L/2 for minimum image convention to work correctly
    if rc_actual > L_half
        error("energy_at_configuration: rc = $rc_actual > L/2 = $L_half. Minimum image convention may fail.")
    end
    
    @inbounds for i in 1:N_local
        for j in (i+1):N_local
            # Compute distance vector
            dr_x = pos[1, j] - pos[1, i]
            dr_y = pos[2, j] - pos[2, i]
            dr_z = pos[3, j] - pos[3, i]
            
            # Apply minimum image convention
            if dr_x > L_half
                dr_x = dr_x - L
            elseif dr_x < -L_half
                dr_x = dr_x + L
            end
            if dr_y > L_half
                dr_y = dr_y - L
            elseif dr_y < -L_half
                dr_y = dr_y + L
            end
            if dr_z > L_half
                dr_z = dr_z - L
            elseif dr_z < -L_half
                dr_z = dr_z + L
            end
            
            r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
            
            # Use exactly rc2 from params (no clamping)
            if r2 < rc2 && r2 > 0.0
                energy += MolSim.MC.lj_pair_u_from_r2(r2, p)
            end
        end
    end
    
    # Add long-range correction if enabled
    if p.use_lrc
        energy += N_local * p.lrc_u_per_particle
    end
    
    return energy
end

# Initialize simulation
println("Initializing simulation...")
p, st = MolSim.MC.init_fcc(N=N, ρ=rho, T=T, rc=rc, max_disp=max_disp, seed=seed, use_lrc=use_lrc)
T_actual = 1.0 / p.β
β = p.β
L_initial = st.L
L_half_initial = L_initial / 2.0
println("  Actual T = $T_actual")
println("  Initial L = $L_initial")
println("  Initial L/2 = $L_half_initial")
println("  rc = $(p.rc)")
println("  rc < L/2? $(p.rc < L_half_initial)")
if p.rc >= L_half_initial
    println("  WARNING: rc >= L/2, minimum image convention may be incorrect!")
end
println()

# Warmup
println("Warmup: $warmup_sweeps sweeps...")
for i in 1:warmup_sweeps
    MolSim.MC.sweep!(st, p)
    if i % 1000 == 0
        println("  Completed $i / $warmup_sweeps sweeps")
    end
end
println("  Warmup complete")
println()

# Production: sample and accumulate FEP data
println("Production: $prod_sweeps sweeps...")
println("  Sampling every $sample_every sweeps")
println()

# Accumulators for FEP: one for each epsV
exp_neg_beta_dU = Dict{Float64, Vector{Float64}}()
for epsV in epsV_list
    exp_neg_beta_dU[epsV] = Float64[]
end

# Virial pressure accumulator
pressure_ba = MolSim.MC.BlockAverager(200)

# Temporary storage for scaled positions (reused, allocated once)
pos_scaled = similar(st.pos)

let
    n_samples = 0
    
    for sweep_idx in 1:prod_sweeps
        MolSim.MC.sweep!(st, p)
        
        if sweep_idx % sample_every == 0
            n_samples += 1
        
        # Get current configuration (do not modify st)
        V_current = st.L * st.L * st.L
        L_current = st.L
        U_current = MolSim.MC.total_energy(st, p)
        
        # Sample virial pressure
        P_virial = MolSim.MC.pressure(st, p, T_actual)
        push!(pressure_ba, P_virial)
        
        # For each volume perturbation
        for epsV in epsV_list
            # Compute new volume and scale factor
            V_new = V_current * (1.0 + epsV)
            s = (1.0 + epsV)^(1.0/3.0)  # scale factor for positions
            L_new = L_current * s
            
            # Scale positions: r' = s * r
            @inbounds for i in 1:N
                pos_scaled[1, i] = st.pos[1, i] * s
                pos_scaled[2, i] = st.pos[2, i] * s
                pos_scaled[3, i] = st.pos[3, i] * s
            end
            
            # Compute energy in scaled configuration
            # Note: rc is in absolute units, so it's the same
            U_new = energy_at_configuration(pos_scaled, L_new, p)
            
            # Compute ΔU = U' - U
            ΔU = U_new - U_current
            
            # Accumulate exp(-βΔU)
            push!(exp_neg_beta_dU[epsV], exp(-β * ΔU))
        end
        
            if n_samples % 200 == 0
                println("  Completed $sweep_idx / $prod_sweeps sweeps ($n_samples samples)")
            end
        end
    end
    
    global n_samples_total = n_samples
end

println("  Production complete ($n_samples_total samples)")
println()

# Compute FEP pressure estimates
println("=" ^ 80)
println("Results")
println("=" ^ 80)
println()

P_virial_mean = MolSim.MC.mean(pressure_ba)
P_virial_stderr = MolSim.MC.stderr(pressure_ba)

println("Virial pressure:")
println("  P_virial = $P_virial_mean ± $P_virial_stderr")
println()

println("FEP pressure estimates:")
println()

# Store results for CSV
results_data = []

for epsV in epsV_list
    exp_values = exp_neg_beta_dU[epsV]
    if isempty(exp_values)
        error("No samples for epsV = $epsV")
    end
    
    # Compute mean(exp(-βΔU))
    mean_exp = sum(exp_values) / length(exp_values)
    
    # Compute configurational free energy change: ΔF_conf = -T * log(mean_exp)
    if mean_exp > 0.0
        ΔF_conf = -T_actual * log(mean_exp)
    else
        error("mean_exp <= 0 for epsV = $epsV (mean_exp = $mean_exp)")
    end
    
    # Compute V' and ΔV
    V_current = st.L * st.L * st.L  # Use current V
    V_new = V_current * (1.0 + epsV)
    ΔV = V_new - V_current  # = V_current * epsV
    
    if abs(ΔV) < 1e-10
        error("ΔV too small for epsV = $epsV")
    end
    
    # Compute ideal gas/Jacobian term: -N*T*log(V'/V)
    log_V_ratio = log(V_new / V_current)
    ΔF_ideal = -N * T_actual * log_V_ratio
    
    # Total free energy change
    ΔF_total = ΔF_conf + ΔF_ideal
    
    # Compute pressures
    P_ex_fep = -ΔF_conf / ΔV  # Excess (configurational) pressure
    P_total_fep = -ΔF_total / ΔV  # Total pressure (including ideal gas)
    
    # Ideal gas pressure
    rho_local = N / V_current
    P_id = rho_local * T_actual
    
    # Check: P_id + P_ex_fep should ≈ P_virial
    P_check = P_id + P_ex_fep
    diff_check = P_check - P_virial_mean
    
    diff_total = P_total_fep - P_virial_mean
    
    epsV_str = epsV >= 0 ? "+$(round(epsV, digits=4))" : "$(round(epsV, digits=4))"
    println("  epsV = $(lpad(epsV_str, 8)):")
    println("    mean(exp(-βΔU)) = $mean_exp")
    println("    ΔF_conf = $ΔF_conf")
    println("    ΔF_ideal = $ΔF_ideal (Jacobian term)")
    println("    ΔF_total = $ΔF_total")
    println("    ΔV = $ΔV")
    println("    P_ex_fep = $P_ex_fep (excess configurational)")
    println("    P_id = $P_id (ideal gas)")
    println("    P_check = P_id + P_ex_fep = $P_check")
    println("    P_virial = $P_virial_mean")
    println("    diff(P_check - P_virial) = $diff_check")
    println("    P_total_fep = $P_total_fep (with Jacobian)")
    println("    diff(P_total_fep - P_virial) = $diff_total")
    println()
    
    push!(results_data, (epsV=epsV, P_ex_fep=P_ex_fep, P_total_fep=P_total_fep, P_virial_mean=P_virial_mean, diff=diff_total))
end

# Compute centered estimates to remove finite-epsV bias
println("Centered estimates (removing finite-epsV bias):")
println("  P_c(epsV) = 0.5 * (P(+epsV) + P(-epsV))")
println()

P_centered = Dict{Float64, Float64}()
for i in 1:2
    epsV_abs = abs(epsV_list[2*i-1])  # 1e-3 or 2e-3
    epsV_pos = epsV_list[2*i-1]  # +1e-3, +2e-3
    epsV_neg = epsV_list[2*i]    # -1e-3, -2e-3
    
    # Find indices in results_data
    idx_pos = findfirst(r -> r.epsV == epsV_pos, results_data)
    idx_neg = findfirst(r -> r.epsV == epsV_neg, results_data)
    
    P_pos = results_data[idx_pos].P_total_fep
    P_neg = results_data[idx_neg].P_total_fep
    
    # Centered estimate
    P_c = 0.5 * (P_pos + P_neg)
    P_centered[epsV_abs] = P_c
    
    diff_sym = P_pos - P_neg
    rel_diff = abs(diff_sym) / abs(P_virial_mean)
    
    println("  epsV = ±$(epsV_abs):")
    println("    P_total_fep(+epsV) = $P_pos")
    println("    P_total_fep(-epsV) = $P_neg")
    println("    P_c(±$(epsV_abs)) = $P_c")
    println("    diff(P(+epsV) - P(-epsV)) = $diff_sym (rel = $(round(rel_diff*100, digits=4))%)")
    println()
end

# Check: P_id + P_ex_fep ≈ P_virial
println("Check: P_id + P_ex_fep should ≈ P_virial")
rho_current = N / (st.L * st.L * st.L)
P_id_check = rho_current * T_actual
println("  P_id = $P_id_check")
println("  P_ex_fep (averaged over epsV):")
P_ex_avg = sum([r.P_ex_fep for r in results_data]) / length(results_data)
println("    P_ex_fep_avg = $P_ex_avg")
P_total_check = P_id_check + P_ex_avg
println("  P_id + P_ex_fep_avg = $P_total_check")
println("  P_virial = $P_virial_mean")
println("  diff = $(P_total_check - P_virial_mean)")
println()

# Richardson extrapolation using centered values
if length(epsV_list) >= 4 && haskey(P_centered, 1e-3) && haskey(P_centered, 2e-3)
    epsV_small = 1e-3
    epsV_large = 2e-3
    
    P_c_small = P_centered[epsV_small]
    P_c_large = P_centered[epsV_large]
    
    # Richardson extrapolation to epsV -> 0
    # Formula: P0 ≈ P_c(eps) + (P_c(eps) - P_c(2*eps))/3
    # This assumes error scales as O(eps^2)
    P0 = P_c_small + (P_c_small - P_c_large) / 3.0
    
    diff_P0 = P0 - P_virial_mean
    rel_diff_P0 = abs(diff_P0) / abs(P_virial_mean) * 100.0
    
    println("Richardson extrapolation (to epsV -> 0):")
    println("  Using centered values:")
    println("    P_c(epsV=1e-3) = $P_c_small")
    println("    P_c(epsV=2e-3) = $P_c_large")
    println("  Extrapolation formula: P0 = P_c(eps) + (P_c(eps) - P_c(2*eps))/3")
    println("    P0 = $P0")
    println("    P_virial = $P_virial_mean")
    println("    diff(P0 - P_virial) = $diff_P0")
    println("    rel_diff = $(round(rel_diff_P0, digits=4))%")
    
    if abs(rel_diff_P0) <= 2.0
        println("  ✓ PASS: |rel_diff| <= 2%")
    else
        println("  ⚠ WARNING: |rel_diff| > 2%")
    end
    println()
end

# Write CSV
results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)
csv_file = joinpath(results_dir, "nvt_pressure_fep_check.csv")

open(csv_file, "w") do io
    println(io, "epsV,P_ex_fep,P_total_fep,P_virial_mean,diff")
    for r in results_data
        println(io, "$(r.epsV),$(r.P_ex_fep),$(r.P_total_fep),$(r.P_virial_mean),$(r.diff)")
    end
end

println("Wrote results to: $csv_file")
println()

println("=" ^ 80)
