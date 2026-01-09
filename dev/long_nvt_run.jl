"""
Long NVT simulation for validation workflow.
Measures P, U, Z, Widom μ_ex, acceptance, and throughput.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

# Parameters (easy to edit)
N = 864
T = 2.0
rho = 0.2
rc = 2.5
max_disp = 0.1
use_lrc = false   # validation path vs Kolafa PLJ
seed = 12345
warmup_sweeps = 5_000
prod_sweeps = 50_000
sample_every = 50          # sampling cadence
widom_every = 200         # widom cadence (less frequent)
widom_ninsert = 200       # number of insertions per Widom sample
block_size_samples = 50   # for block stderr on sampled observables

# Ensure results directory exists
results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)

println("=" ^ 80)
println("Long NVT Simulation")
println("=" ^ 80)
println()
println("Parameters:")
println("  N = $N")
println("  T = $T")
println("  ρ = $rho")
println("  rc = $rc")
println("  max_disp = $max_disp")
println("  use_lrc = $use_lrc")
println("  seed = $seed")
println("  warmup_sweeps = $warmup_sweeps")
println("  prod_sweeps = $prod_sweeps")
println("  sample_every = $sample_every")
println("  widom_every = $widom_every")
println("  widom_ninsert = $widom_ninsert")
println("  block_size_samples = $block_size_samples")
println("  FEP thermodynamic pressure: enabled (epsV = ±1e-3)")
println()

# Helper function: compute energy for given positions and box length (for FEP pressure)
function energy_at_configuration(pos::Matrix{Float64}, L::Float64, p::MolSim.MC.LJParams)::Float64
    energy = 0.0
    N_local = size(pos, 2)
    rc2 = p.rc2
    rc_actual = p.rc
    L_half = L / 2.0
    
    if rc_actual > L_half
        error("energy_at_configuration: rc = $rc_actual > L/2 = $L_half. Minimum image convention may fail.")
    end
    
    @inbounds for i in 1:N_local
        for j in (i+1):N_local
            dr_x = pos[1, j] - pos[1, i]
            dr_y = pos[2, j] - pos[2, i]
            dr_z = pos[3, j] - pos[3, i]
            
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
            
            if r2 < rc2 && r2 > 0.0
                energy += MolSim.MC.lj_pair_u_from_r2(r2, p)
            end
        end
    end
    
    if p.use_lrc
        energy += N_local * p.lrc_u_per_particle
    end
    
    return energy
end

# Initialize
println("Initializing simulation...")
p, st = MolSim.MC.init_fcc(N=N, ρ=rho, T=T, rc=rc, max_disp=max_disp, seed=seed, use_lrc=use_lrc)
T_actual = 1.0 / p.β
β = p.β
println("  Actual T = $T_actual")
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

# Production setup
println("Production: $prod_sweeps sweeps...")
pressure_ba = MolSim.MC.BlockAverager(block_size_samples)
pressure_thermo_ba = MolSim.MC.BlockAverager(block_size_samples)
energy_ba = MolSim.MC.BlockAverager(block_size_samples)
z_ba = MolSim.MC.BlockAverager(block_size_samples)
mu_ex_ba = MolSim.MC.BlockAverager(block_size_samples)

widom_acc = MolSim.MC.WidomAccumulator()

# FEP pressure parameters
epsV_fep = 1e-3
epsV_fep_neg = -1e-3
pos_scaled = similar(st.pos)  # Temporary storage for scaled positions

# Timeseries data
timeseries_data = []

# Acceptance tracking
let
    total_acc = 0.0
    total_acc_count = 0
    
    # Timing
    production_start = time_ns()
    
    # Production loop
    for sweep_idx in 1:prod_sweeps
        acc = MolSim.MC.sweep!(st, p)
        total_acc += acc
        total_acc_count += 1
        
        if sweep_idx % sample_every == 0
        # Sample P (virial), U, Z
        P = MolSim.MC.pressure(st, p, T_actual)
        E_total = MolSim.MC.total_energy(st, p)
        U = E_total / N  # energy per particle
        Z = P / (rho * T_actual)
        
        push!(pressure_ba, P)
        push!(energy_ba, U)
        push!(z_ba, Z)
        
        # FEP thermodynamic pressure (centered estimate using ±epsV)
        V_current = st.L * st.L * st.L
        L_current = st.L
        U_current = E_total
        
        # Try +epsV
        V_new_pos = V_current * (1.0 + epsV_fep)
        s_pos = (1.0 + epsV_fep)^(1.0/3.0)
        L_new_pos = L_current * s_pos
        @inbounds for i in 1:N
            pos_scaled[1, i] = st.pos[1, i] * s_pos
            pos_scaled[2, i] = st.pos[2, i] * s_pos
            pos_scaled[3, i] = st.pos[3, i] * s_pos
        end
        U_new_pos = energy_at_configuration(pos_scaled, L_new_pos, p)
        ΔU_pos = U_new_pos - U_current
        ΔV_pos = V_new_pos - V_current
        mean_exp_pos = exp(-β * ΔU_pos)
        ΔF_conf_pos = -T_actual * log(mean_exp_pos)
        log_V_ratio_pos = log(V_new_pos / V_current)
        ΔF_ideal_pos = -N * T_actual * log_V_ratio_pos
        ΔF_total_pos = ΔF_conf_pos + ΔF_ideal_pos
        P_fep_pos = -ΔF_total_pos / ΔV_pos
        
        # Try -epsV
        V_new_neg = V_current * (1.0 + epsV_fep_neg)
        s_neg = (1.0 + epsV_fep_neg)^(1.0/3.0)
        L_new_neg = L_current * s_neg
        @inbounds for i in 1:N
            pos_scaled[1, i] = st.pos[1, i] * s_neg
            pos_scaled[2, i] = st.pos[2, i] * s_neg
            pos_scaled[3, i] = st.pos[3, i] * s_neg
        end
        U_new_neg = energy_at_configuration(pos_scaled, L_new_neg, p)
        ΔU_neg = U_new_neg - U_current
        ΔV_neg = V_new_neg - V_current
        mean_exp_neg = exp(-β * ΔU_neg)
        ΔF_conf_neg = -T_actual * log(mean_exp_neg)
        log_V_ratio_neg = log(V_new_neg / V_current)
        ΔF_ideal_neg = -N * T_actual * log_V_ratio_neg
        ΔF_total_neg = ΔF_conf_neg + ΔF_ideal_neg
        P_fep_neg = -ΔF_total_neg / ΔV_neg
        
        # Centered estimate
        P_thermo = 0.5 * (P_fep_pos + P_fep_neg)
        push!(pressure_thermo_ba, P_thermo)
        
        # Widom insertion if scheduled
        mu_ex_value = NaN
        if sweep_idx % widom_every == 0
            MolSim.MC.reset!(widom_acc)
            mu_ex_value = MolSim.MC.widom_mu_ex!(widom_acc, st, p; ninsert=widom_ninsert)
            push!(mu_ex_ba, mu_ex_value)
        end
        
        # Record timeseries
        current_acc = total_acc / total_acc_count
        push!(timeseries_data, (
            sweep_index = sweep_idx,
            P = P,
            U = U,
            Z = Z,
            mu_ex = mu_ex_value,
            acc_so_far = current_acc
        ))
        end
        
        if sweep_idx % 5000 == 0
            println("  Completed $sweep_idx / $prod_sweeps sweeps")
        end
    end

    production_end = time_ns()
    global runtime_ns = production_end - production_start
    global runtime_sec = runtime_ns / 1e9
    
    # Statistics
    global P_mean = MolSim.MC.mean(pressure_ba)
    global P_stderr = MolSim.MC.stderr(pressure_ba)
    global P_thermo_mean = MolSim.MC.mean(pressure_thermo_ba)
    global P_thermo_stderr = length(pressure_thermo_ba.block_means) >= 2 ? MolSim.MC.stderr(pressure_thermo_ba) : NaN
    global U_mean = MolSim.MC.mean(energy_ba)
    global U_stderr = MolSim.MC.stderr(energy_ba)
    global Z_mean = MolSim.MC.mean(z_ba)
    global Z_stderr = MolSim.MC.stderr(z_ba)
    global mu_ex_mean = (length(mu_ex_ba.block_means) > 0 || length(mu_ex_ba.current_block) > 0) ? MolSim.MC.mean(mu_ex_ba) : NaN
    global mu_ex_stderr = length(mu_ex_ba.block_means) >= 2 ? MolSim.MC.stderr(mu_ex_ba) : NaN
    
    global acc_move = total_acc / total_acc_count
    total_attempted_moves = prod_sweeps * N
    global moves_per_sec = total_attempted_moves / runtime_sec
    global sweeps_per_sec = prod_sweeps / runtime_sec
end

# Print summary
println()
println("=" ^ 80)
println("Results Summary")
println("=" ^ 80)
println()
println("Observables:")
println("  P_virial_mean = $P_mean ± $P_stderr (virial estimator)")
if !isnan(P_thermo_stderr)
    println("  P_thermo_mean = $P_thermo_mean ± $P_thermo_stderr (FEP thermodynamic)")
else
    println("  P_thermo_mean = $P_thermo_mean (FEP thermodynamic, stderr unavailable)")
end
println("  U_mean = $U_mean ± $U_stderr")
println("  Z_mean = $Z_mean ± $Z_stderr")
if !isnan(mu_ex_mean)
    println("  μ_ex_mean = $mu_ex_mean ± $mu_ex_stderr")
else
    println("  μ_ex_mean = NaN (no Widom samples)")
end
println()
println("Acceptance:")
println("  acc_move = $acc_move")
println()
println("Performance:")
println("  Runtime = $(round(runtime_sec, digits=2)) s")
println("  Moves/sec = $(round(moves_per_sec, digits=0))")
println("  Sweeps/sec = $(round(sweeps_per_sec, digits=2))")
println()

# Write summary CSV
summary_file = joinpath(results_dir, "nvt_summary.csv")
open(summary_file, "w") do io
    # Header
    println(io, "N,T,rho,rc,max_disp,use_lrc,seed,warmup_sweeps,prod_sweeps,sample_every," *
                "P_mean,P_stderr,P_thermo_mean,P_thermo_stderr,U_mean,U_stderr,Z_mean,Z_stderr," *
                "mu_ex_mean,mu_ex_stderr," *
                "acc_move,moves_per_sec,sweeps_per_sec,runtime_sec")
    
    # Data row
    mu_ex_mean_str = isnan(mu_ex_mean) ? "NaN" : string(mu_ex_mean)
    mu_ex_stderr_str = isnan(mu_ex_stderr) ? "NaN" : string(mu_ex_stderr)
    P_thermo_stderr_str = isnan(P_thermo_stderr) ? "NaN" : string(P_thermo_stderr)
    
    println(io, "$N,$T,$rho,$rc,$max_disp,$use_lrc,$seed,$warmup_sweeps,$prod_sweeps,$sample_every," *
                "$P_mean,$P_stderr,$P_thermo_mean,$P_thermo_stderr_str,$U_mean,$U_stderr,$Z_mean,$Z_stderr," *
                "$mu_ex_mean_str,$mu_ex_stderr_str," *
                "$acc_move,$moves_per_sec,$sweeps_per_sec,$runtime_sec")
end
println("Wrote summary to: $summary_file")

# Write timeseries CSV
timeseries_file = joinpath(results_dir, "nvt_timeseries.csv")
open(timeseries_file, "w") do io
    # Header
    println(io, "sweep_index,P,U,Z,mu_ex,acc_so_far")
    
    # Data rows
    for row in timeseries_data
        mu_ex_str = isnan(row.mu_ex) ? "" : string(row.mu_ex)
        println(io, "$(row.sweep_index),$(row.P),$(row.U),$(row.Z),$mu_ex_str,$(row.acc_so_far)")
    end
end
println("Wrote timeseries to: $timeseries_file")
println()

println("=" ^ 80)
