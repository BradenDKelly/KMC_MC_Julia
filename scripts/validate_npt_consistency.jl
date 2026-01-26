"""
Validate NPT ensemble consistency:
1. Compare NVT MC vs NVT eKMC (should match)
2. Compare NPT MC vs NPT eKMC (should match)
3. Verify thermodynamic consistency: NVT at density ρ should have same average density
   when run in NPT at pressure P = <P>_NVT (for both MC and eKMC)

Usage (from repo root):
  julia scripts/validate_npt_consistency.jl
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

# Force-reload MC validation code
Base.include(MolSim.MC, joinpath(@__DIR__, "..", "src", "MC", "validation", "CompareNVT.jl"))

# ============================================================================
# Simulation parameters
# ============================================================================

# State point
const N = 108
const T = 1.0
const ρ_nvt = 0.8  # Target density for NVT
const rc = 2.5
const max_disp = 0.1
const max_dlnV = 0.01
const max_dV = 0.1  # Maximum volume change for eKMC (absolute, not relative)

# Run parameters - longer runs for proper thermodynamic consistency
const burnin_sweeps_nvt = 5000
const prod_sweeps_nvt = 20000
const burnin_sweeps_npt = 5000
const prod_sweeps_npt = 20000
const burnin_events_ekmc = burnin_sweeps_nvt * N * 10
const prod_events_ekmc = prod_sweeps_nvt * N * 10
const vol_move_every_mc = 10
const vol_move_every_ekmc = 10
const sample_every_nvt = 10
# eKMC should sample more frequently than MC since it has many more events
# MC samples every 10 sweeps = 10*N moves = 1080 moves
# eKMC has 10x more events, so sample 10x more frequently = every 108 events
# This gives ~10x more samples for better statistics
const sample_every_ekmc = max(1, (sample_every_nvt * N) ÷ 10)
const dlnV_virtual = 1e-4

# RNG seeds
const seed_nvt_mc = 11111
const seed_nvt_ekmc = 22222
const seed_npt_mc = 33333
const seed_npt_ekmc = 44444

println("=" ^ 80)
println("NPT Consistency Validation")
println("=" ^ 80)
println()
println("Test 1: NVT MC vs eKMC comparison")
println("Test 2: NPT MC vs eKMC comparison")
println("Test 3: NVT-NPT thermodynamic consistency (MC)")
println("Test 4: NVT-NPT thermodynamic consistency (eKMC)")
println()
println("Parameters:")
println("  N = $N")
println("  T = $T")
println("  ρ_NVT (target) = $ρ_nvt")
println("  rc = $rc")
println()

# ============================================================================
# Initialize parameters
# ============================================================================

# Initialize NVT MC state (this also creates the parameters)
p, st_nvt_mc = MC.init_fcc(N=N, ρ=ρ_nvt, T=T, rc=rc, max_disp=max_disp,
                           seed=seed_nvt_mc, use_lrc=false, lj_model=:shifted,
                           apply_impulsive_correction=false)

# ============================================================================
# Test 1: NVT MC simulation
# ============================================================================

println("=" ^ 80)
println("Test 1a: NVT MC Simulation")
println("=" ^ 80)
println()

L_nvt = st_nvt_mc.L
println("Initial NVT MC state:")
println("  ρ = $ρ_nvt")
println("  L = $(round(L_nvt, digits=6))")
println()

# Burn-in
println("NVT MC burn-in ($burnin_sweeps_nvt sweeps)...")
for sweep in 1:burnin_sweeps_nvt
    MC.sweep!(st_nvt_mc, p; rebuild_every=1)
end

# Production: sample pressure
println("NVT MC production ($prod_sweeps_nvt sweeps)...")
nvt_mc_pressures = Float64[]
nvt_mc_densities = Float64[]
nvt_mc_energies = Float64[]

for sweep in 1:prod_sweeps_nvt
    MC.sweep!(st_nvt_mc, p; rebuild_every=1)
    if sweep % sample_every_nvt == 0
        P = MC.pressure(st_nvt_mc, p, T)
        ρ_inst = N / (st_nvt_mc.L * st_nvt_mc.L * st_nvt_mc.L)
        U = MC.total_energy(st_nvt_mc, p)
        push!(nvt_mc_pressures, P)
        push!(nvt_mc_densities, ρ_inst)
        push!(nvt_mc_energies, U)
    end
end

ρ_nvt_mc_mean = Statistics.mean(nvt_mc_densities)
ρ_nvt_mc_stderr = Statistics.std(nvt_mc_densities) / sqrt(length(nvt_mc_densities))
P_nvt_mc_mean = Statistics.mean(nvt_mc_pressures)
P_nvt_mc_stderr = Statistics.std(nvt_mc_pressures) / sqrt(length(nvt_mc_pressures))
U_nvt_mc_mean = Statistics.mean(nvt_mc_energies) / N
U_nvt_mc_stderr = Statistics.std(nvt_mc_energies) / (sqrt(length(nvt_mc_energies)) * N)

println("NVT MC Results:")
println("  <ρ> = $(round(ρ_nvt_mc_mean, digits=6)) ± $(round(ρ_nvt_mc_stderr, digits=6))")
println("  <P> = $(round(P_nvt_mc_mean, digits=6)) ± $(round(P_nvt_mc_stderr, digits=6))")
println("  <U/N> = $(round(U_nvt_mc_mean, digits=6)) ± $(round(U_nvt_mc_stderr, digits=6))")
println()

# ============================================================================
# Test 2: NVT eKMC simulation
# ============================================================================

println("=" ^ 80)
println("Test 1b: NVT eKMC Simulation")
println("=" ^ 80)
println()

# Initialize eKMC state from a fresh FCC lattice (same as MC started from)
# Both should start from the same initial configuration
_, st_nvt_ekmc_init = MC.init_fcc(N=N, ρ=ρ_nvt, T=T, rc=rc, max_disp=max_disp,
                                  seed=seed_nvt_ekmc, use_lrc=false, lj_model=:shifted,
                                  apply_impulsive_correction=false)
ekst_nvt = MC.init_ekmc_state(st_nvt_ekmc_init, p)
# Enable periodic phi consistency checks to catch drift (every 1000 events)
ekst_nvt.debug_check_every = 1000

# Run eKMC NVT
acc_mu_nvt = MC.ChemicalPotentialAccumulator()
nvt_ekmc_obs, nvt_ekmc_total_time, _ = MC.run_ekmc(
    p, ekst_nvt, acc_mu_nvt;
    burnin_events=burnin_events_ekmc,
    prod_events=prod_events_ekmc,
    sample_every=sample_every_ekmc,
    dlnV_virtual=dlnV_virtual,
    collect_timeseries=false
)

# For NVT, density is constant (fixed volume), so use the target density
# The box size doesn't change in NVT
ρ_nvt_ekmc_mean = ρ_nvt
ρ_nvt_ekmc_stderr = 0.0  # Density is fixed in NVT

P_nvt_ekmc_mean = MC.mean(nvt_ekmc_obs.pressure)
P_nvt_ekmc_stderr = MC.stderr(nvt_ekmc_obs.pressure)
U_nvt_ekmc_mean = MC.mean(nvt_ekmc_obs.U_per_particle)
U_nvt_ekmc_stderr = MC.stderr(nvt_ekmc_obs.U_per_particle)

println("NVT eKMC Results:")
println("  <ρ> = $(round(ρ_nvt_ekmc_mean, digits=6)) ± $(round(ρ_nvt_ekmc_stderr, digits=6))")
println("  <P> = $(round(P_nvt_ekmc_mean, digits=6)) ± $(round(P_nvt_ekmc_stderr, digits=6))")
println("  <U/N> = $(round(U_nvt_ekmc_mean, digits=6)) ± $(round(U_nvt_ekmc_stderr, digits=6))")
println("  μ_ex = $(round(nvt_ekmc_obs.mu_ex, digits=6)) ± $(round(nvt_ekmc_obs.mu_ex_err, digits=6))")
println()

# Use MC NVT pressure for NPT MC, and eKMC NVT pressure for NPT eKMC
Pext_mc = P_nvt_mc_mean
Pext_ekmc = P_nvt_ekmc_mean

# ============================================================================
# Test 3: NPT MC at P = <P>_NVT_MC
# ============================================================================

println("=" ^ 80)
println("Test 2a: NPT MC at P = <P>_NVT_MC = $(round(Pext_mc, digits=6))")
println("=" ^ 80)
println()

# Initialize NPT MC state (start at NVT density, use same parameters)
_, st_npt_mc = MC.init_fcc(N=N, ρ=ρ_nvt, T=T, rc=rc, max_disp=max_disp,
                          seed=seed_npt_mc, use_lrc=false, lj_model=:shifted,
                          apply_impulsive_correction=false)

# Burn-in
println("NPT MC burn-in ($burnin_sweeps_npt sweeps)...")
for sweep in 1:burnin_sweeps_npt
    MC.sweep!(st_npt_mc, p; rebuild_every=1)
    if sweep % vol_move_every_mc == 0
        MC.volume_trial!(st_npt_mc, p; max_dlnV=max_dlnV, Pext=Pext_mc)
    end
end

# Production
println("NPT MC production ($prod_sweeps_npt sweeps)...")
mc_particle_acc, mc_volume_acc, mc_densities, mc_energies = MC.run_npt!(
    st_npt_mc, p;
    nsweeps=prod_sweeps_npt, Pext=Pext_mc, max_disp=max_disp,
    max_dlnV=max_dlnV, vol_move_every=vol_move_every_mc
)

ρ_npt_mc_mean = Statistics.mean(mc_densities)
ρ_npt_mc_stderr = Statistics.std(mc_densities) / sqrt(length(mc_densities))
U_npt_mc_mean = Statistics.mean(mc_energies) / N
U_npt_mc_stderr = Statistics.std(mc_energies) / (sqrt(length(mc_energies)) * N)

println("NPT MC Results:")
println("  <ρ> = $(round(ρ_npt_mc_mean, digits=6)) ± $(round(ρ_npt_mc_stderr, digits=6))")
println("  <U/N> = $(round(U_npt_mc_mean, digits=6)) ± $(round(U_npt_mc_stderr, digits=6))")
println("  Particle acceptance: $(round(mc_particle_acc, digits=4))")
println("  Volume acceptance: $(round(mc_volume_acc, digits=4))")
println()

# ============================================================================
# Test 4: NPT eKMC at P = <P>_NVT_eKMC
# ============================================================================

println("=" ^ 80)
println("Test 2b: NPT eKMC at P = <P>_NVT_eKMC = $(round(Pext_ekmc, digits=6))")
println("=" ^ 80)
println()

# Initialize eKMC state from a fresh FCC lattice (same as MC started from)
_, st_npt_ekmc_init = MC.init_fcc(N=N, ρ=ρ_nvt, T=T, rc=rc, max_disp=max_disp,
                                  seed=seed_npt_ekmc, use_lrc=false, lj_model=:shifted,
                                  apply_impulsive_correction=false)
ekst_npt = MC.init_ekmc_state(st_npt_ekmc_init, p)
ekst_npt.debug_check_every = 10000  # Enable periodic phi consistency checks

# Burn-in
println("NPT eKMC burn-in ($burnin_events_ekmc events)...")
acc_mu_burnin = MC.ChemicalPotentialAccumulator()
MC.run_ekmc!(ekst_npt, p, acc_mu_burnin; nsteps=burnin_events_ekmc)

# Production
println("NPT eKMC production ($prod_events_ekmc events)...")
acc_mu_prod = MC.ChemicalPotentialAccumulator()
ekmc_obs, ekmc_total_time, _ = MC.run_ekmc_npt!(
    ekst_npt, p, acc_mu_prod;
    nsteps=prod_events_ekmc, Pext=Pext_ekmc, max_dV=max_dV,
    vol_move_every=vol_move_every_ekmc, sample_every=sample_every_ekmc,
    collect_timeseries=false
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
println("  μ_ex = $(round(ekmc_obs.mu_ex, digits=6)) ± $(round(ekmc_obs.mu_ex_err, digits=6))")
println()

# ============================================================================
# Consistency Checks
# ============================================================================

println("=" ^ 80)
println("Consistency Checks")
println("=" ^ 80)
println()

# Check 1: NVT MC vs eKMC
Δρ_nvt = ρ_nvt_mc_mean - ρ_nvt_ekmc_mean
σ_Δρ_nvt = sqrt(ρ_nvt_mc_stderr^2 + ρ_nvt_ekmc_stderr^2)
n_sigma_nvt = abs(Δρ_nvt) / max(σ_Δρ_nvt, 1e-10)

ΔU_nvt = U_nvt_mc_mean - U_nvt_ekmc_mean
σ_ΔU_nvt = sqrt(U_nvt_mc_stderr^2 + U_nvt_ekmc_stderr^2)
n_sigma_U_nvt = abs(ΔU_nvt) / max(σ_ΔU_nvt, 1e-10)

ΔP_nvt = P_nvt_mc_mean - P_nvt_ekmc_mean
σ_ΔP_nvt = sqrt(P_nvt_mc_stderr^2 + P_nvt_ekmc_stderr^2)
n_sigma_P_nvt = abs(ΔP_nvt) / max(σ_ΔP_nvt, 1e-10)

println("Check 1: NVT MC vs eKMC consistency")
println("  Density:")
println("    MC:    $(round(ρ_nvt_mc_mean, digits=6)) ± $(round(ρ_nvt_mc_stderr, digits=6))")
println("    eKMC:  $(round(ρ_nvt_ekmc_mean, digits=6)) ± $(round(ρ_nvt_ekmc_stderr, digits=6))")
println("    Δ = $(round(Δρ_nvt, digits=6)), |Δ|/σ = $(round(n_sigma_nvt, digits=2))")
println("  Energy:")
println("    MC:    $(round(U_nvt_mc_mean, digits=6)) ± $(round(U_nvt_mc_stderr, digits=6))")
println("    eKMC:  $(round(U_nvt_ekmc_mean, digits=6)) ± $(round(U_nvt_ekmc_stderr, digits=6))")
println("    Δ = $(round(ΔU_nvt, digits=6)), |Δ|/σ = $(round(n_sigma_U_nvt, digits=2))")
println("  Pressure:")
println("    MC:    $(round(P_nvt_mc_mean, digits=6)) ± $(round(P_nvt_mc_stderr, digits=6))")
println("    eKMC:  $(round(P_nvt_ekmc_mean, digits=6)) ± $(round(P_nvt_ekmc_stderr, digits=6))")
println("    Δ = $(round(ΔP_nvt, digits=6)), |Δ|/σ = $(round(n_sigma_P_nvt, digits=2))")
if n_sigma_nvt < 3.0 && n_sigma_U_nvt < 3.0 && n_sigma_P_nvt < 3.0
    println("  ✓ PASS (all within 3σ)")
else
    println("  ✗ FAIL (some outside 3σ)")
end
println()

# Check 2: NPT MC vs eKMC
Δρ_npt = ρ_npt_mc_mean - ρ_npt_ekmc_mean
σ_Δρ_npt = sqrt(ρ_npt_mc_stderr^2 + ρ_npt_ekmc_stderr^2)
n_sigma_npt = abs(Δρ_npt) / max(σ_Δρ_npt, 1e-10)

ΔU_npt = U_npt_mc_mean - U_npt_ekmc_mean
σ_ΔU_npt = sqrt(U_npt_mc_stderr^2 + U_npt_ekmc_stderr^2)
n_sigma_U_npt = abs(ΔU_npt) / max(σ_ΔU_npt, 1e-10)

println("Check 2: NPT MC vs eKMC consistency")
println("  Density:")
println("    MC:    $(round(ρ_npt_mc_mean, digits=6)) ± $(round(ρ_npt_mc_stderr, digits=6))")
println("    eKMC:  $(round(ρ_npt_ekmc_mean, digits=6)) ± $(round(ρ_npt_ekmc_stderr, digits=6))")
println("    Δ = $(round(Δρ_npt, digits=6)), |Δ|/σ = $(round(n_sigma_npt, digits=2))")
println("  Energy:")
println("    MC:    $(round(U_npt_mc_mean, digits=6)) ± $(round(U_npt_mc_stderr, digits=6))")
println("    eKMC:  $(round(U_npt_ekmc_mean, digits=6)) ± $(round(U_npt_ekmc_stderr, digits=6))")
println("    Δ = $(round(ΔU_npt, digits=6)), |Δ|/σ = $(round(n_sigma_U_npt, digits=2))")
if n_sigma_npt < 3.0 && n_sigma_U_npt < 3.0
    println("  ✓ PASS (all within 3σ)")
else
    println("  ✗ FAIL (some outside 3σ)")
end
println()

# Check 3: NVT-NPT thermodynamic consistency (MC)
Δρ_nvt_npt_mc = ρ_nvt_mc_mean - ρ_npt_mc_mean
σ_Δρ_nvt_npt_mc = sqrt(ρ_nvt_mc_stderr^2 + ρ_npt_mc_stderr^2)
n_sigma_nvt_npt_mc = abs(Δρ_nvt_npt_mc) / max(σ_Δρ_nvt_npt_mc, 1e-10)

println("Check 3: NVT-NPT thermodynamic consistency (MC)")
println("  NVT <ρ>:   $(round(ρ_nvt_mc_mean, digits=6)) ± $(round(ρ_nvt_mc_stderr, digits=6))")
println("  NPT <ρ>:   $(round(ρ_npt_mc_mean, digits=6)) ± $(round(ρ_npt_mc_stderr, digits=6))")
println("  (at P = $(round(Pext_mc, digits=6)))")
println("  Δ = $(round(Δρ_nvt_npt_mc, digits=6)), |Δ|/σ = $(round(n_sigma_nvt_npt_mc, digits=2))")
if n_sigma_nvt_npt_mc < 3.0
    println("  ✓ PASS (within 3σ)")
else
    println("  ✗ FAIL (outside 3σ)")
end
println()

# Check 4: NVT-NPT thermodynamic consistency (eKMC)
Δρ_nvt_npt_ekmc = ρ_nvt_ekmc_mean - ρ_npt_ekmc_mean
σ_Δρ_nvt_npt_ekmc = sqrt(ρ_nvt_ekmc_stderr^2 + ρ_npt_ekmc_stderr^2)
n_sigma_nvt_npt_ekmc = abs(Δρ_nvt_npt_ekmc) / max(σ_Δρ_nvt_npt_ekmc, 1e-10)

println("Check 4: NVT-NPT thermodynamic consistency (eKMC)")
println("  NVT <ρ>:   $(round(ρ_nvt_ekmc_mean, digits=6)) ± $(round(ρ_nvt_ekmc_stderr, digits=6))")
println("  NPT <ρ>:   $(round(ρ_npt_ekmc_mean, digits=6)) ± $(round(ρ_npt_ekmc_stderr, digits=6))")
println("  (at P = $(round(Pext_ekmc, digits=6)))")
println("  Δ = $(round(Δρ_nvt_npt_ekmc, digits=6)), |Δ|/σ = $(round(n_sigma_nvt_npt_ekmc, digits=2))")
if n_sigma_nvt_npt_ekmc < 3.0
    println("  ✓ PASS (within 3σ)")
else
    println("  ✗ FAIL (outside 3σ)")
end
println()

# Check 5: NPT eKMC pressure matches Pext_ekmc
ΔP_ekmc = P_npt_ekmc_mean - Pext_ekmc
n_sigma_P = abs(ΔP_ekmc) / max(P_npt_ekmc_stderr, 1e-10)

println("Check 5: NPT eKMC pressure matches Pext (from eKMC NVT)")
println("  <P>_eKMC:  $(round(P_npt_ekmc_mean, digits=6)) ± $(round(P_npt_ekmc_stderr, digits=6))")
println("  Pext:      $(round(Pext_ekmc, digits=6))")
println("  Δ = $(round(ΔP_ekmc, digits=6)), |Δ|/σ = $(round(n_sigma_P, digits=2))")
if n_sigma_P < 3.0
    println("  ✓ PASS (within 3σ)")
else
    println("  ✗ FAIL (outside 3σ)")
end
println()

# Summary
println("=" ^ 80)
println("Summary")
println("=" ^ 80)
all_passed = (n_sigma_nvt < 3.0) && (n_sigma_U_nvt < 3.0) && (n_sigma_P_nvt < 3.0) &&
             (n_sigma_npt < 3.0) && (n_sigma_U_npt < 3.0) &&
             (n_sigma_nvt_npt_mc < 3.0) && (n_sigma_nvt_npt_ekmc < 3.0) && (n_sigma_P < 3.0)

if all_passed
    println("✓ All consistency checks PASSED")
else
    println("✗ Some consistency checks FAILED")
end
println("=" ^ 80)
