"""
Validate NPT eKMC against NPT MC using Tan et al. method.

Compares:
- Density (ρ)
- Energy per particle (U/N)
- Pressure (P)
- Chemical potential (μ_ex)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "src", "MolSim.jl"))
using .MolSim
using .MolSim.MC
using .MolSim.EOS
using Statistics

# Simulation parameters
const N = 108
const T = 1.0
const Pext = 0.5  # External pressure
const rc = 2.5
const max_disp = 0.1
const max_dlnV = 0.01
const max_dV = 0.1  # Maximum volume change for eKMC (absolute, not relative)

# Run parameters
const burnin_sweeps_mc = 500
const prod_sweeps_mc = 1000
const burnin_events_ekmc = burnin_sweeps_mc * N * 10
const prod_events_ekmc = prod_sweeps_mc * N * 10
const vol_move_every_mc = 10
const vol_move_every_ekmc = 10
const sample_every_ekmc = max(1, (prod_events_ekmc ÷ 100) ÷ 10)

# RNG seeds
const seed_mc = 12345
const seed_ekmc = 67890

println("=" ^ 80)
println("NPT Validation: eKMC vs MC")
println("=" ^ 80)
println()
println("Parameters:")
println("  N = $N")
println("  T = $T")
println("  Pext = $Pext")
println("  rc = $rc")
println("  max_dlnV = $max_dlnV (MC)")
println("  max_dV = $max_dV (eKMC)")
println()
println("Run parameters:")
println("  MC: $burnin_sweeps_mc burnin + $prod_sweeps_mc production sweeps")
println("  eKMC: $burnin_events_ekmc burnin + $prod_events_ekmc production events")
println("  Volume move every: $vol_move_every_mc (MC), $vol_move_every_ekmc (eKMC)")
println("  Sample every: $sample_every_ekmc (eKMC)")
println()

# Initialize parameters
p_mc = MC.LJParams(
    σ=1.0, ϵ=1.0, rc=rc, rc2=rc*rc,
    β=1.0/T, max_disp=max_disp,
    use_lrc=true, lrc_u_per_particle=0.0, lrc_p=0.0,
    lj_model=:truncated, apply_impulsive_correction=false, u_rc=0.0,
    n_types=1, σ_types=Float64[], ϵ_types=Float64[],
    σ_mix=zeros(Float64, 0, 0), ϵ_mix=zeros(Float64, 0, 0)
)
MC.compute_lrc!(p_mc, N, 1.0)  # Initial density guess

p_ekmc = p_mc  # Same parameters

# Initialize states
println("Initializing states...")
st_mc = MC.init_lj_state(N, 1.0, seed_mc)  # Initial density = 1.0
ekst = MC.init_ekmc_state(st_mc, p_ekmc)

# Burn-in
println("Burn-in...")
for _ in 1:burnin_sweeps_mc
    MC.sweep!(st_mc, p_mc)
    if _ % vol_move_every_mc == 0
        MC.volume_trial!(st_mc, p_mc; max_dlnV=max_dlnV, Pext=Pext)
    end
end

acc_mu_burnin = MC.ChemicalPotentialAccumulator()
MC.run_ekmc!(ekst, p_ekmc, acc_mu_burnin; nsteps=burnin_events_ekmc)

println("  MC final density: $(N / (st_mc.L * st_mc.L * st_mc.L))")
println("  eKMC final density: $(N / (ekst.L * ekst.L * ekst.L))")
println()

# Production
println("Production...")
acc_mu_prod = MC.ChemicalPotentialAccumulator()

# MC production
mc_particle_acc, mc_volume_acc, mc_densities, mc_energies = MC.run_npt!(
    st_mc, p_mc;
    nsweeps=prod_sweeps_mc, Pext=Pext, max_disp=max_disp,
    max_dlnV=max_dlnV, vol_move_every=vol_move_every_mc
)

# eKMC production
ekmc_obs, ekmc_total_time, ekmc_timeseries = MC.run_ekmc_npt!(
    ekst, p_ekmc, acc_mu_prod;
    nsteps=prod_events_ekmc, Pext=Pext, max_dV=max_dV,
    vol_move_every=vol_move_every_ekmc, sample_every=sample_every_ekmc,
    collect_timeseries=false
)

println("  MC particle acceptance: $(round(mc_particle_acc, digits=4))")
println("  MC volume acceptance: $(round(mc_volume_acc, digits=4))")
println("  eKMC volume moves: $(ekmc_obs.vol_moves)")
println("  eKMC total time: $(round(ekmc_total_time, digits=4))")
println()

# Compute statistics
println("=" ^ 80)
println("Results")
println("=" ^ 80)
println()

# MC statistics
mc_rho_mean = Statistics.mean(mc_densities)
mc_rho_stderr = Statistics.std(mc_densities) / sqrt(length(mc_densities))
mc_U_mean = Statistics.mean(mc_energies) / N
mc_U_stderr = Statistics.std(mc_energies) / (sqrt(length(mc_energies)) * N)

# eKMC statistics
ekmc_rho_mean = MC.mean(ekmc_obs.rho)
ekmc_rho_stderr = MC.stderr(ekmc_obs.rho)
ekmc_U_mean = MC.mean(ekmc_obs.U_per_particle)
ekmc_U_stderr = MC.stderr(ekmc_obs.U_per_particle)
ekmc_P_mean = MC.mean(ekmc_obs.pressure)
ekmc_P_stderr = MC.stderr(ekmc_obs.pressure)

println("Density (ρ):")
println("  MC:    $(round(mc_rho_mean, digits=6)) ± $(round(mc_rho_stderr, digits=6))")
println("  eKMC:  $(round(ekmc_rho_mean, digits=6)) ± $(round(ekmc_rho_stderr, digits=6))")
println("  Δ:     $(round(ekmc_rho_mean - mc_rho_mean, digits=6))")
println()

println("Energy per particle (U/N):")
println("  MC:    $(round(mc_U_mean, digits=6)) ± $(round(mc_U_stderr, digits=6))")
println("  eKMC:  $(round(ekmc_U_mean, digits=6)) ± $(round(ekmc_U_stderr, digits=6))")
println("  Δ:     $(round(ekmc_U_mean - mc_U_mean, digits=6))")
println()

println("Pressure (P):")
println("  eKMC:  $(round(ekmc_P_mean, digits=6)) ± $(round(ekmc_P_stderr, digits=6))")
println("  Pext:  $Pext")
println("  Δ:     $(round(ekmc_P_mean - Pext, digits=6))")
println()

println("Chemical potential (μ_ex):")
println("  eKMC:  $(round(ekmc_obs.mu_ex, digits=6)) ± $(round(ekmc_obs.mu_ex_err, digits=6))")
println()

println("=" ^ 80)
