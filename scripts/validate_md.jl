"""
Validation script comparing MD (NVT and NPT) with MC and eKMC results.

Usage (from repo root):
  julia scripts/validate_md.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "src", "MolSim.jl"))
using .MolSim
using .MolSim.MC
using .MolSim.EOS
using Statistics
using Random

const N = 400
const T = 1.0
const ρ_nvt = 0.8
const rc = 2.5
const max_disp = 0.1
const max_dlnV = 0.01
const max_dV = 0.1

# MD parameters
const dt_md = 0.001
const thermostat_every = 10
const vol_move_every_md = N  # MC volume moves every N steps
const max_dlnV_md = 0.01  # Maximum volume change for MD NPT

# Simulation lengths
const burnin_sweeps_mc = 20000  # Increased from 5000 to allow MC to reach same equilibrium as MD/eKMC
const prod_sweeps_mc = 20000
const burnin_steps_md = 100000
const prod_steps_md = 100000
const burnin_events_ekmc = 10800000
const prod_events_ekmc = 5400000

const sample_every_mc = 10
const sample_every_md = 10
const sample_every_ekmc = max(1, (prod_events_ekmc ÷ (prod_sweeps_mc ÷ sample_every_mc)) ÷ 2)

# RDF parameters
const rdf_nbins = 200
const rdf_snapshot_every = 100  # Collect snapshot every N sweeps/steps/events for RDF

println("=================================================================================")
println("MD vs MC vs eKMC Validation")
println("=================================================================================")
println("Parameters:")
println("  N = $N")
println("  T = $T")
println("  ρ_NVT (target) = $ρ_nvt")
println("  rc = $rc")
println("=================================================================================")

# Initialize parameters
p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=max_disp,
                use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)

# Initialize states from FCC lattice
seed_mc = 12345
seed_md = 23456
seed_ekmc = 34567

println("\n=================================================================================")
println("Test 1a: NVT MC Simulation")
println("=================================================================================")
_, st_nvt_mc = MC.init_simple(N=N, ρ=ρ_nvt, T=T, rc=rc, max_disp=max_disp,
                               seed=seed_mc, use_lrc=false, lj_model=:shifted,
                               apply_impulsive_correction=false)
println("Initial NVT MC state:")
println("  ρ = $ρ_nvt")
println("  L = $(st_nvt_mc.L)")

println("NVT MC burn-in ($burnin_sweeps_mc sweeps)...")
# Initialize adaptive tuning for NVT (only during equilibration)
acc_params_nvt = MC.AdaptiveAcceptanceParams(; max_disp=max_disp, max_dlnV=max_dlnV,
                                               target_acceptance=0.45, adjust_every=50)
for sweep in 1:burnin_sweeps_mc
    MC.sweep!(st_nvt_mc, p; rebuild_every=1, acc_params=acc_params_nvt)
    if sweep % 50 == 0 && sweep > 0
        MC.adjust_parameters!(acc_params_nvt)
    end
end
# Extract final tuned parameters for production
final_max_disp_nvt = acc_params_nvt.max_disp
println("  Final tuned max_disp = $(round(final_max_disp_nvt, digits=6))")

println("NVT MC production ($prod_sweeps_mc sweeps)...")
# Collect data using a function (no adaptive tuning in production)
function run_nvt_mc_production(st, p, T, N, prod_sweeps, sample_every, rdf_snapshot_every, max_disp_final)
    pressures = Float64[]
    densities = Float64[]
    energies = Float64[]
    snapshots = []
    
    for sweep in 1:prod_sweeps
        MC.sweep!(st, p; rebuild_every=1, max_disp_override=max_disp_final)
        if sweep % sample_every == 0
            P = MC.pressure(st, p, T)
            ρ_inst = N / (st.L * st.L * st.L)
            U = MC.total_energy(st, p)
            push!(pressures, P)
            push!(densities, ρ_inst)
            push!(energies, U)
        end
        # Collect snapshots for RDF
        if sweep % rdf_snapshot_every == 0
            push!(snapshots, copy(st.pos))
        end
    end
    
    ρ_mean = Statistics.mean(densities)
    ρ_stderr = Statistics.std(densities) / sqrt(length(densities))
    P_mean = Statistics.mean(pressures)
    P_stderr = Statistics.std(pressures) / sqrt(length(pressures))
    U_mean = Statistics.mean(energies) / N
    U_stderr = Statistics.std(energies) / (sqrt(length(energies)) * N)
    
    return (snapshots, ρ_mean, ρ_stderr, P_mean, P_stderr, U_mean, U_stderr)
end

nvt_mc_snapshots, ρ_nvt_mc_mean, ρ_nvt_mc_stderr, P_nvt_mc_mean, P_nvt_mc_stderr, U_nvt_mc_mean, U_nvt_mc_stderr = 
    run_nvt_mc_production(st_nvt_mc, p, T, N, prod_sweeps_mc, sample_every_mc, rdf_snapshot_every, final_max_disp_nvt)

println("NVT MC Results:")
println("  <ρ> = $(round(ρ_nvt_mc_mean, digits=6)) ± $(round(ρ_nvt_mc_stderr, digits=6))")
println("  <P> = $(round(P_nvt_mc_mean, digits=6)) ± $(round(P_nvt_mc_stderr, digits=6))")
println("  <U/N> = $(round(U_nvt_mc_mean, digits=6)) ± $(round(U_nvt_mc_stderr, digits=6))")

println("\n=================================================================================")
println("Test 1b: NVT MD Simulation")
println("=================================================================================")
_, st_nvt_mc_init = MC.init_simple(N=N, ρ=ρ_nvt, T=T, rc=rc, max_disp=max_disp,
                                    seed=seed_md, use_lrc=false, lj_model=:shifted,
                                    apply_impulsive_correction=false)
st_nvt_md = MC.init_md_state(st_nvt_mc_init.pos, T, Random.Xoshiro(seed_md))
st_nvt_md.L = st_nvt_mc_init.L

println("Initial NVT MD state:")
println("  ρ = $ρ_nvt")
println("  L = $(st_nvt_md.L)")

println("NVT MD burn-in ($burnin_steps_md steps)...")
MC.run_md_nvt!(st_nvt_md, p; nsteps=burnin_steps_md, dt=dt_md, T=T, thermostat_every=thermostat_every)

println("NVT MD production ($prod_steps_md steps)...")
# Collect snapshots and compute statistics using a function
function run_nvt_md_production(st, p, T, prod_steps, dt, thermostat_every, rdf_snapshot_every)
    snapshots = []
    MC.compute_forces!(st, p)
    MC.velocity_rescale!(st, T)
    
    U_sum = 0.0
    U_sq_sum = 0.0
    P_sum = 0.0
    P_sq_sum = 0.0
    count = 0
    
    for step in 1:prod_steps
        MC.velocity_verlet_step!(st, p, dt)
        
        if step % thermostat_every == 0
            MC.velocity_rescale!(st, T)
            U = MC._md_total_energy(st, p)
            P = MC.pressure(st, p, T)
            
            U_sum += U
            U_sq_sum += U * U
            P_sum += P
            P_sq_sum += P * P
            count += 1
        end
        
        # Collect snapshots for RDF
        if step % rdf_snapshot_every == 0
            push!(snapshots, copy(st.pos))
        end
    end
    
    U_avg = U_sum / count
    U_var = (U_sq_sum / count) - (U_avg * U_avg)
    U_std = sqrt(max(0.0, U_var))
    P_avg = P_sum / count
    P_var = (P_sq_sum / count) - (P_avg * P_avg)
    P_std = sqrt(max(0.0, P_var))
    
    return (snapshots, U_avg, U_std, P_avg, P_std)
end

nvt_md_snapshots, U_md_nvt, U_std_md_nvt, P_md_nvt, P_std_md_nvt = 
    run_nvt_md_production(st_nvt_md, p, T, prod_steps_md, dt_md, thermostat_every, rdf_snapshot_every)
ρ_md_nvt = ρ_nvt  # Fixed in NVT

println("NVT MD Results:")
println("  <ρ> = $(round(ρ_md_nvt, digits=6))")
println("  <P> = $(round(P_md_nvt, digits=6)) ± $(round(P_std_md_nvt, digits=6))")
println("  <U/N> = $(round(U_md_nvt / N, digits=6)) ± $(round(U_std_md_nvt / N, digits=6))")

println("\n=================================================================================")
println("Test 1c: NVT eKMC Simulation")
println("=================================================================================")
_, st_nvt_ekmc_init = MC.init_simple(N=N, ρ=ρ_nvt, T=T, rc=rc, max_disp=max_disp,
                                      seed=seed_ekmc, use_lrc=false, lj_model=:shifted,
                                      apply_impulsive_correction=false)
ekst_nvt = MC.init_ekmc_state(st_nvt_ekmc_init, p)
ekst_nvt.debug_check_every = 10000

acc_mu_nvt = MC.ChemicalPotentialAccumulator()
# Collect snapshots manually for RDF during production
nvt_ekmc_snapshots = []

# Burn-in
println("Burning in eKMC...")
for event in 1:burnin_events_ekmc
    MC.ekmc_step!(ekst_nvt, p, acc_mu_nvt)
end
println("Burning in eKMC... done")

# Production with snapshot collection
println("Running eKMC production...")
nvt_ekmc_obs = MC.NVTObservables()
MC.reset!(acc_mu_nvt)

for event in 1:prod_events_ekmc
    # Sample before move if needed
    do_sample = (event % sample_every_ekmc == 0)
    if do_sample
        # Use optimized functions that use stored pair_u matrix
        U = MC.total_energy_from_pair_u(ekst_nvt, p)
        P = MC.pressure_from_pair_u(ekst_nvt, p, T)
    end
    
    # Perform KMC step
    dt = MC.ekmc_step!(ekst_nvt, p, acc_mu_nvt)
    
    # Accumulate observables
    if do_sample && dt > 0.0
        MC.push!(nvt_ekmc_obs.U_per_particle, U / N, dt)
        MC.push!(nvt_ekmc_obs.pressure, P, dt)
    end
    
    # Collect snapshots for RDF
    if event % rdf_snapshot_every == 0
        push!(nvt_ekmc_snapshots, copy(ekst_nvt.pos))
    end
end

nvt_ekmc_total_time = acc_mu_nvt.t_total
obs_ekmc = nvt_ekmc_obs
obs_ekmc = nvt_ekmc_obs

ρ_nvt_ekmc_mean = ρ_nvt  # Density is fixed in NVT
ρ_nvt_ekmc_stderr = 0.0
P_nvt_ekmc_mean = MC.mean(obs_ekmc.pressure)
P_nvt_ekmc_stderr = MC.stderr(obs_ekmc.pressure)
U_nvt_ekmc_mean = MC.mean(obs_ekmc.U_per_particle)
U_nvt_ekmc_stderr = MC.stderr(obs_ekmc.U_per_particle)

println("NVT eKMC Results:")
println("  <ρ> = $(round(ρ_nvt_ekmc_mean, digits=6)) ± $(round(ρ_nvt_ekmc_stderr, digits=6))")
println("  <P> = $(round(P_nvt_ekmc_mean, digits=6)) ± $(round(P_nvt_ekmc_stderr, digits=6))")
println("  <U/N> = $(round(U_nvt_ekmc_mean, digits=6)) ± $(round(U_nvt_ekmc_stderr, digits=6))")

println("\n=================================================================================")
println("Test 2a: NPT MC at P = <P>_NVT_MC = $(round(P_nvt_mc_mean, digits=6))")
println("=================================================================================")
_, st_npt_mc = MC.init_simple(N=N, ρ=ρ_nvt, T=T, rc=rc, max_disp=max_disp,
                               seed=seed_mc+100, use_lrc=false, lj_model=:shifted,
                               apply_impulsive_correction=false)

println("NPT MC burn-in ($burnin_sweeps_mc sweeps)...")
# Initialize adaptive tuning for NPT (only during equilibration)
acc_params_npt = MC.AdaptiveAcceptanceParams(; max_disp=max_disp, max_dlnV=max_dlnV,
                                                target_acceptance=0.45, adjust_every=50)
for sweep in 1:burnin_sweeps_mc
    MC.sweep!(st_npt_mc, p; rebuild_every=1, acc_params=acc_params_npt)
    if sweep % 10 == 0
        current_max_dlnV = acc_params_npt.max_dlnV
        vol_acc = MC.volume_trial!(st_npt_mc, p; max_dlnV=current_max_dlnV, Pext=P_nvt_mc_mean)
        if vol_acc
            acc_params_npt.volume_accepted += 1
        end
        acc_params_npt.volume_attempted += 1
    end
    if sweep % 50 == 0 && sweep > 0
        MC.adjust_parameters!(acc_params_npt)
    end
end
# Extract final tuned parameters for production
final_max_disp_npt = acc_params_npt.max_disp
final_max_dlnV_npt = acc_params_npt.max_dlnV
println("  Final tuned max_disp = $(round(final_max_disp_npt, digits=6))")
println("  Final tuned max_dlnV = $(round(final_max_dlnV_npt, digits=6))")

println("NPT MC production ($prod_sweeps_mc sweeps)...")
# Collect timeseries and track acceptance using a function to avoid globals (no adaptive tuning in production)
function run_npt_mc_production(st, p, T, N, prod_sweeps, sample_every, Pext, max_disp_final, max_dlnV_final)
    sweeps = Float64[]
    densities_ts = Float64[]
    energies_ts = Float64[]
    pressures_ts = Float64[]
    particle_accepted = 0
    particle_attempted = 0
    volume_accepted = 0
    volume_attempted = 0
    
    for sweep in 1:prod_sweeps
        acc = MC.sweep!(st, p; rebuild_every=1, max_disp_override=max_disp_final)
        particle_accepted += Int(round(acc * N))
        particle_attempted += N
        
        if sweep % 10 == 0
            vol_acc = MC.volume_trial!(st, p; max_dlnV=max_dlnV_final, Pext=Pext)
            if vol_acc
                volume_accepted += 1
            end
            volume_attempted += 1
            
            # Sample observables
            if sweep % sample_every == 0
                ρ_inst = N / (st.L^3)
                U_inst = MC.total_energy(st, p)
                P_inst = MC.pressure(st, p, T)
                push!(sweeps, Float64(sweep))
                push!(densities_ts, ρ_inst)
                push!(energies_ts, U_inst)
                push!(pressures_ts, P_inst)
            end
        end
    end
    
    particle_acceptance = particle_attempted > 0 ? Float64(particle_accepted) / Float64(particle_attempted) : 0.0
    volume_acceptance = volume_attempted > 0 ? Float64(volume_accepted) / Float64(volume_attempted) : 0.0
    
    return (sweeps, densities_ts, energies_ts, pressures_ts, particle_acceptance, volume_acceptance)
end

npt_mc_sweeps, npt_mc_densities_ts, npt_mc_energies_ts, npt_mc_pressures_ts, mc_particle_acceptance, mc_volume_acceptance = 
    run_npt_mc_production(st_npt_mc, p, T, N, prod_sweeps_mc, sample_every_mc, P_nvt_mc_mean, final_max_disp_npt, final_max_dlnV_npt)

mc_densities = npt_mc_densities_ts
mc_energies = npt_mc_energies_ts

ρ_npt_mc = Statistics.mean(mc_densities)
ρ_npt_mc_std = Statistics.std(mc_densities) / sqrt(length(mc_densities))
U_npt_mc = Statistics.mean(mc_energies)
U_npt_mc_std = Statistics.std(mc_energies) / sqrt(length(mc_energies))

println("NPT MC Results:")
println("  <ρ> = $(round(ρ_npt_mc, digits=6)) ± $(round(ρ_npt_mc_std, digits=6))")
println("  <U/N> = $(round(U_npt_mc / N, digits=6)) ± $(round(U_npt_mc_std / N, digits=6))")
println("  Particle acceptance: $(round(mc_particle_acceptance, digits=4))")
println("  Volume acceptance: $(round(mc_volume_acceptance, digits=3))")

println("\n=================================================================================")
println("Test 2b: NPT MD at P = <P>_NVT_MD = $(round(P_md_nvt, digits=6))")
println("=================================================================================")
_, st_npt_mc_init = MC.init_simple(N=N, ρ=ρ_nvt, T=T, rc=rc, max_disp=max_disp,
                                    seed=seed_md+100, use_lrc=false, lj_model=:shifted,
                                    apply_impulsive_correction=false)
st_npt_md = MC.init_md_state(st_npt_mc_init.pos, T, Random.Xoshiro(seed_md+100))
st_npt_md.L = st_npt_mc_init.L

println("NPT MD burn-in ($burnin_steps_md steps)...")
MC.run_md_npt!(st_npt_md, p; nsteps=burnin_steps_md, dt=dt_md, T=T, P=P_md_nvt,
               thermostat_every=thermostat_every, vol_move_every=vol_move_every_md, max_dlnV=max_dlnV_md)

println("NPT MD production ($prod_steps_md steps)...")
# Initialize forces before production
try
    MC.compute_forces!(st_npt_md, p)
    println("  Forces initialized successfully")
catch e
    println("  ERROR initializing forces: $e")
    rethrow(e)
end

# Collect timeseries using a function to avoid globals
function run_npt_md_production(st, p, T, prod_steps, dt, thermostat_every, vol_move_every, Pext, max_dlnV)
    steps = Float64[]
    densities_ts = Float64[]
    energies_ts = Float64[]
    pressures_ts = Float64[]
    
    U_sum = 0.0
    U_sq_sum = 0.0
    P_sum = 0.0
    P_sq_sum = 0.0
    ρ_sum = 0.0
    ρ_sq_sum = 0.0
    count = 0
    
    for step in 1:prod_steps
        try
            MC.velocity_verlet_step!(st, p, dt)
        catch e
            println("Error in velocity_verlet_step! at step $step: $e")
            rethrow(e)
        end
        
        # Apply thermostat
        if step % thermostat_every == 0
            MC.velocity_rescale!(st, T)
        end
        
        # Apply MC volume move
        if step % vol_move_every == 0
            MC.volume_trial_md!(st, p, T, Pext, max_dlnV, st.rng)
            # Forces are recomputed inside volume_trial_md! if accepted
        end
        
        # Sample observables (always after thermostat step, forces should be current)
        if step % thermostat_every == 0
            try
                # Ensure forces are current before sampling
                # If volume move was applied this step, forces were already recomputed
                # Otherwise, forces are current from velocity_verlet_step!
                if step % vol_move_every != 0
                    MC.compute_forces!(st, p)
                end
                
                U = MC._md_total_energy(st, p)
                P_inst = MC.pressure(st, p, T)
                ρ_inst = st.N / (st.L^3)
                
                # Debug first few samples
                if step <= 50
                    println("  Step $step: U=$(U), P=$(P_inst), ρ=$(ρ_inst), L=$(st.L)")
                end
                
                if isfinite(U) && isfinite(P_inst) && isfinite(ρ_inst) && ρ_inst > 0.0 && st.L > 0.0
                    U_sum += U
                    U_sq_sum += U * U
                    P_sum += P_inst
                    P_sq_sum += P_inst * P_inst
                    ρ_sum += ρ_inst
                    ρ_sq_sum += ρ_inst * ρ_inst
                    count += 1
                    
                    # Store for timeseries
                    push!(steps, Float64(step))
                    push!(densities_ts, ρ_inst)
                    push!(energies_ts, U)
                    push!(pressures_ts, P_inst)
                else
                    # Debug: print why sampling failed
                    if step <= 50 || (step % 1000 == 0 && count == 0)
                        println("  Warning: Skipping sample at step $step: U=$(U), P=$(P_inst), ρ=$(ρ_inst), L=$(st.L)")
                        println("    isfinite(U)=$(isfinite(U)), isfinite(P)=$(isfinite(P_inst)), isfinite(ρ)=$(isfinite(ρ_inst))")
                    end
                end
            catch e
                if step <= 50 || (step % 1000 == 0 && count == 0)
                    println("Error sampling at step $step: $e")
                    println("  State: L=$(st.L), N=$(st.N)")
                    println("  Stacktrace: ", sprint(showerror, e, catch_backtrace()))
                end
                # Continue anyway
            end
        end
    end
    
    # Debug output
    println("  NPT MD sampling: count = $count out of $(prod_steps ÷ thermostat_every) expected samples")
    
    # Compute averages
    if count > 0
        U_avg = U_sum / count
        U_var = (U_sq_sum / count) - (U_avg * U_avg)
        U_std = sqrt(max(0.0, U_var))
        
        P_avg = P_sum / count
        P_var = (P_sq_sum / count) - (P_avg * P_avg)
        P_std = sqrt(max(0.0, P_var))
        
        ρ_avg = ρ_sum / count
        ρ_var = (ρ_sq_sum / count) - (ρ_avg * ρ_avg)
        ρ_std = sqrt(max(0.0, ρ_var))
    else
        U_avg = 0.0
        U_std = 0.0
        P_avg = NaN
        P_std = NaN
        ρ_avg = NaN
        ρ_std = NaN
    end
    
    return (steps, densities_ts, energies_ts, pressures_ts, U_avg, U_std, P_avg, P_std, ρ_avg, ρ_std)
end

npt_md_steps, npt_md_densities_ts, npt_md_energies_ts, npt_md_pressures_ts, U_md_npt, U_std_md_npt, P_md_npt, P_std_md_npt, ρ_md_npt, ρ_std_md_npt = 
    run_npt_md_production(st_npt_md, p, T, prod_steps_md, dt_md, thermostat_every, vol_move_every_md, P_md_nvt, max_dlnV_md)

println("NPT MD Results:")
println("  <ρ> = $(round(ρ_md_npt, digits=6)) ± $(round(ρ_std_md_npt, digits=6))")
println("  <P> = $(round(P_md_npt, digits=6)) ± $(round(P_std_md_npt, digits=6))")
println("  <U/N> = $(round(U_md_npt / N, digits=6)) ± $(round(U_std_md_npt / N, digits=6))")

println("\n=================================================================================")
println("Test 2c: NPT eKMC at P = <P>_NVT_eKMC = $(round(P_nvt_ekmc_mean, digits=6))")
println("=================================================================================")
_, st_npt_ekmc_init = MC.init_simple(N=N, ρ=ρ_nvt, T=T, rc=rc, max_disp=max_disp,
                                      seed=seed_ekmc+100, use_lrc=false, lj_model=:shifted,
                                      apply_impulsive_correction=false)
ekst_npt = MC.init_ekmc_state(st_npt_ekmc_init, p)
ekst_npt.debug_check_every = 10000

acc_mu_npt_burnin = MC.ChemicalPotentialAccumulator()
println("NPT eKMC burn-in ($burnin_events_ekmc events)...")
MC.run_ekmc_npt!(ekst_npt, p, acc_mu_npt_burnin; nsteps=burnin_events_ekmc, Pext=P_nvt_ekmc_mean,
                 max_dV=max_dV, vol_move_every=100, sample_every=1000, collect_timeseries=false)

acc_mu_npt_prod = MC.ChemicalPotentialAccumulator()
println("NPT eKMC production ($prod_events_ekmc events)...")
ekmc_obs, ekmc_total_time, ekmc_timeseries = MC.run_ekmc_npt!(
    ekst_npt, p, acc_mu_npt_prod; nsteps=prod_events_ekmc, Pext=P_nvt_ekmc_mean, max_dV=max_dV,
    vol_move_every=100, sample_every=sample_every_ekmc, collect_timeseries=true
)

ρ_npt_ekmc_mean = MC.mean(ekmc_obs.rho)
ρ_npt_ekmc_stderr = MC.stderr(ekmc_obs.rho)
U_npt_ekmc_mean = MC.mean(ekmc_obs.U_per_particle)
U_npt_ekmc_stderr = MC.stderr(ekmc_obs.U_per_particle)
P_npt_ekmc_mean = MC.mean(ekmc_obs.pressure)
P_npt_ekmc_stderr = MC.stderr(ekmc_obs.pressure)

println("NPT eKMC Results:")
println("  <ρ> = $(round(ρ_npt_ekmc_mean, digits=6)) ± $(round(ρ_npt_ekmc_stderr, digits=6))")
println("  <U/N> = $(round(U_npt_ekmc_mean, digits=6)) ± $(round(U_npt_ekmc_stderr, digits=6))")
println("  <P> = $(round(P_npt_ekmc_mean, digits=6)) ± $(round(P_npt_ekmc_stderr, digits=6))")
println("  Volume moves: $(ekmc_obs.vol_moves)")
println("  Total time: $(round(ekmc_total_time, digits=4))")

println("\n=================================================================================")
println("Consistency Checks")
println("=================================================================================")

println("Check 1: NVT MC vs MD vs eKMC consistency")
println("  Density:")
println("    MC:    $(round(ρ_nvt_mc_mean, digits=6)) ± $(round(ρ_nvt_mc_stderr, digits=6))")
println("    MD:    $(round(ρ_md_nvt, digits=6))")
println("    eKMC:  $(round(ρ_nvt_ekmc_mean, digits=6)) ± $(round(ρ_nvt_ekmc_stderr, digits=6))")
println("  Energy:")
println("    MC:    $(round(U_nvt_mc_mean, digits=6)) ± $(round(U_nvt_mc_stderr, digits=6))")
println("    MD:    $(round(U_md_nvt / N, digits=6)) ± $(round(U_std_md_nvt / N, digits=6))")
println("    eKMC:  $(round(U_nvt_ekmc_mean, digits=6)) ± $(round(U_nvt_ekmc_stderr, digits=6))")
println("  Pressure:")
println("    MC:    $(round(P_nvt_mc_mean, digits=6)) ± $(round(P_nvt_mc_stderr, digits=6))")
println("    MD:    $(round(P_md_nvt, digits=6)) ± $(round(P_std_md_nvt, digits=6))")
println("    eKMC:  $(round(P_nvt_ekmc_mean, digits=6)) ± $(round(P_nvt_ekmc_stderr, digits=6))")

println("\nCheck 2: NPT MC vs MD vs eKMC consistency")
println("  Density:")
println("    MC:    $(round(ρ_npt_mc, digits=6)) ± $(round(ρ_npt_mc_std, digits=6))")
println("    MD:    $(round(ρ_md_npt, digits=6)) ± $(round(ρ_std_md_npt, digits=6))")
println("    eKMC:  $(round(ρ_npt_ekmc_mean, digits=6)) ± $(round(ρ_npt_ekmc_stderr, digits=6))")
println("  Energy:")
println("    MC:    $(round(U_npt_mc / N, digits=6)) ± $(round(U_npt_mc_std / N, digits=6))")
println("    MD:    $(round(U_md_npt / N, digits=6)) ± $(round(U_std_md_npt / N, digits=6))")
println("    eKMC:  $(round(U_npt_ekmc_mean, digits=6)) ± $(round(U_npt_ekmc_stderr, digits=6))")

println("\n=================================================================================")
println("Summary")
println("=================================================================================")
println("MD implementation complete. Compare results above.")

# --- Plot NPT time series ---
try
    using Plots
    gr()  # Use GR backend
    
    # NPT MC plots
    if length(npt_mc_sweeps) > 0
        p1 = plot(npt_mc_sweeps, npt_mc_energies_ts ./ N, label="U/N", xlabel="Sweep", ylabel="U/N", 
                 title="NPT MC: Energy per particle", linewidth=2)
        vline!(p1, [burnin_sweeps_mc], linestyle=:dash, color=:gray, label="Burn-in end")
        
        p2 = plot(npt_mc_sweeps, npt_mc_pressures_ts, label="P", xlabel="Sweep", ylabel="P", 
                 title="NPT MC: Pressure", linewidth=2)
        vline!(p2, [burnin_sweeps_mc], linestyle=:dash, color=:gray, label="Burn-in end")
        
        p3 = plot(npt_mc_sweeps, npt_mc_densities_ts, label="ρ", xlabel="Sweep", ylabel="ρ", 
                 title="NPT MC: Density", linewidth=2)
        vline!(p3, [burnin_sweeps_mc], linestyle=:dash, color=:gray, label="Burn-in end")
        
        plot_npt_mc = plot(p1, p2, p3, layout=(3,1), size=(800,1200))
        savefig(plot_npt_mc, "npt_mc_timeseries.png")
        println("\nSaved: npt_mc_timeseries.png")
    end
    
    # NPT MD plots
    if length(npt_md_steps) > 0
        p1 = plot(npt_md_steps, npt_md_energies_ts ./ N, label="U/N", xlabel="Step", ylabel="U/N", 
                 title="NPT MD: Energy per particle", linewidth=2)
        vline!(p1, [burnin_steps_md], linestyle=:dash, color=:gray, label="Burn-in end")
        
        p2 = plot(npt_md_steps, npt_md_pressures_ts, label="P", xlabel="Step", ylabel="P", 
                 title="NPT MD: Pressure", linewidth=2)
        vline!(p2, [burnin_steps_md], linestyle=:dash, color=:gray, label="Burn-in end")
        
        p3 = plot(npt_md_steps, npt_md_densities_ts, label="ρ", xlabel="Step", ylabel="ρ", 
                 title="NPT MD: Density", linewidth=2)
        vline!(p3, [burnin_steps_md], linestyle=:dash, color=:gray, label="Burn-in end")
        
        plot_npt_md = plot(p1, p2, p3, layout=(3,1), size=(800,1200))
        savefig(plot_npt_md, "npt_md_timeseries.png")
        println("Saved: npt_md_timeseries.png")
    end
    
    # NPT eKMC plots
    if ekmc_timeseries !== nothing && haskey(ekmc_timeseries, :event_idx) && length(ekmc_timeseries[:event_idx]) > 0
        events = Float64.(ekmc_timeseries[:event_idx])
        U_ekmc = ekmc_timeseries[:U_per_particle]
        P_ekmc = ekmc_timeseries[:pressure]
        ρ_ekmc = ekmc_timeseries[:rho]
        
        p1 = plot(events, U_ekmc, label="U/N", xlabel="Event", ylabel="U/N", 
                 title="NPT eKMC: Energy per particle", linewidth=2)
        vline!(p1, [burnin_events_ekmc], linestyle=:dash, color=:gray, label="Burn-in end")
        
        p2 = plot(events, P_ekmc, label="P", xlabel="Event", ylabel="P", 
                 title="NPT eKMC: Pressure", linewidth=2)
        vline!(p2, [burnin_events_ekmc], linestyle=:dash, color=:gray, label="Burn-in end")
        
        p3 = plot(events, ρ_ekmc, label="ρ", xlabel="Event", ylabel="ρ", 
                 title="NPT eKMC: Density", linewidth=2)
        vline!(p3, [burnin_events_ekmc], linestyle=:dash, color=:gray, label="Burn-in end")
        
        plot_npt_ekmc = plot(p1, p2, p3, layout=(3,1), size=(800,1200))
        savefig(plot_npt_ekmc, "npt_ekmc_timeseries.png")
        println("Saved: npt_ekmc_timeseries.png")
    end
    
catch e
    println("\nWarning: Could not create NPT plots (Plots.jl may not be available): $e")
end

# --- Compute and plot RDF comparison ---
println("\n=================================================================================")
println("RDF Comparison")
println("=================================================================================")

try
    using Plots
    gr()
    
    # Compute RDF from snapshots
    rmax_rdf = st_nvt_mc.L / 2.0
    
    println("Computing RDFs from snapshots...")
    println("  MC snapshots: $(length(nvt_mc_snapshots))")
    println("  MD snapshots: $(length(nvt_md_snapshots))")
    println("  eKMC snapshots: $(length(nvt_ekmc_snapshots))")
    
    # MC RDF: average over snapshots
    if length(nvt_mc_snapshots) > 0
        r_mc, g_r_mc = Analysis.rdf(nvt_mc_snapshots[1], st_nvt_mc.L, rmax_rdf, rdf_nbins)
        g_r_mc_sum = copy(g_r_mc)
        for i in 2:length(nvt_mc_snapshots)
            _, g_r_i = Analysis.rdf(nvt_mc_snapshots[i], st_nvt_mc.L, rmax_rdf, rdf_nbins)
            g_r_mc_sum .+= g_r_i
        end
        g_r_mc_avg = g_r_mc_sum ./ length(nvt_mc_snapshots)
        println("  MC RDF computed from $(length(nvt_mc_snapshots)) snapshots")
    else
        r_mc = zeros(Float64, rdf_nbins)
        g_r_mc_avg = zeros(Float64, rdf_nbins)
    end
    
    # MD RDF: average over snapshots
    if length(nvt_md_snapshots) > 0
        r_md, g_r_md = Analysis.rdf(nvt_md_snapshots[1], st_nvt_md.L, rmax_rdf, rdf_nbins)
        g_r_md_sum = copy(g_r_md)
        for i in 2:length(nvt_md_snapshots)
            _, g_r_i = Analysis.rdf(nvt_md_snapshots[i], st_nvt_md.L, rmax_rdf, rdf_nbins)
            g_r_md_sum .+= g_r_i
        end
        g_r_md_avg = g_r_md_sum ./ length(nvt_md_snapshots)
        println("  MD RDF computed from $(length(nvt_md_snapshots)) snapshots")
    else
        r_md = zeros(Float64, rdf_nbins)
        g_r_md_avg = zeros(Float64, rdf_nbins)
    end
    
    # eKMC RDF: average over snapshots
    if length(nvt_ekmc_snapshots) > 0
        r_ekmc, g_r_ekmc = Analysis.rdf(nvt_ekmc_snapshots[1], ekst_nvt.L, rmax_rdf, rdf_nbins)
        g_r_ekmc_sum = copy(g_r_ekmc)
        for i in 2:length(nvt_ekmc_snapshots)
            _, g_r_i = Analysis.rdf(nvt_ekmc_snapshots[i], ekst_nvt.L, rmax_rdf, rdf_nbins)
            g_r_ekmc_sum .+= g_r_i
        end
        g_r_ekmc_avg = g_r_ekmc_sum ./ length(nvt_ekmc_snapshots)
        println("  eKMC RDF computed from $(length(nvt_ekmc_snapshots)) snapshots")
    else
        r_ekmc = zeros(Float64, rdf_nbins)
        g_r_ekmc_avg = zeros(Float64, rdf_nbins)
    end
    
    # Plot RDF comparison
    p_rdf = plot(xlabel="r", ylabel="g(r)", title="Radial Distribution Function Comparison (NVT, T=$T, ρ=$ρ_nvt)", 
                 legend=:topright, grid=true, linewidth=2)
    
    if length(nvt_mc_snapshots) > 0
        plot!(p_rdf, r_mc, g_r_mc_avg, label="MC", color=:blue, linewidth=2)
    end
    if length(nvt_md_snapshots) > 0
        plot!(p_rdf, r_md, g_r_md_avg, label="MD", color=:red, linewidth=2, linestyle=:dash)
    end
    if length(nvt_ekmc_snapshots) > 0
        plot!(p_rdf, r_ekmc, g_r_ekmc_avg, label="eKMC", color=:green, linewidth=2, linestyle=:dot)
    end
    
    savefig(p_rdf, "rdf_comparison_nvt.png")
    println("\nSaved: rdf_comparison_nvt.png")
    
catch e
    println("\nWarning: Could not create RDF comparison plot: $e")
    if isa(e, ErrorException)
        println("  Error: ", e)
    end
end
