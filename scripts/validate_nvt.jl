"""
Validate NVT eKMC vs Metropolis MC and compare pressure to Nezbeda EOS.

Usage (from repo root):
  julia scripts/validate_nvt.jl

Adjust parameters below as needed.
"""

using Pkg

# Activate project and ensure dependencies are available
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "src", "MolSim.jl"))

using .MolSim
using .MolSim.MC
using .MolSim.EOS

# Force-reload MC validation code to avoid stale precompile artifacts
Base.include(MolSim.MC, joinpath(@__DIR__, "..", "src", "MC", "eKMC.jl"))
Base.include(MolSim.MC, joinpath(@__DIR__, "..", "src", "MC", "validation", "CompareNVT.jl"))

# --- Simulation parameters (edit as needed) ---
const N = 864
const ρ = 0.8 #0.8645
const T = 1.2 #0.81
const L = cbrt(N / ρ)
const rc = L / 2.0
const use_lrc = false
const lj_model = :shifted

const burnin_sweeps_mc = 10000
const prod_sweeps_mc = 10000
# Match number of particle translations: 1 sweep ≈ N translations
const burnin_events_ekmc = burnin_sweeps_mc * N 
const prod_events_ekmc = prod_sweeps_mc * N 
const sample_every_mc = 10
# Match number of samples between MC and eKMC
const n_samples_mc = prod_sweeps_mc ÷ sample_every_mc
const sample_every_ekmc = max(1, prod_events_ekmc ÷ n_samples_mc)
const dlnV_virtual = 1e-4
const debug_check_every_ekmc = 1000
const widom_every_mc = 1000
const widom_ninsert_mc = 200

println("Starting MC + eKMC validation run...")

# --- Optional MC sanity check (bruteforce vs cell list) ---
const mc_sanity_check = true
const mc_sanity_sweeps = 1
const mc_sanity_every = 1

if mc_sanity_check
    p_check, st_check = MC.init_fcc(; N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1,
                                    seed=1234, use_lrc=use_lrc, lj_model=lj_model)
    println("MC sanity check (bruteforce vs cell list):")
    for sweep_idx in 1:mc_sanity_sweeps
        MC.sweep!(st_check, p_check; rebuild_every=1)
        if sweep_idx % mc_sanity_every == 0
            E_cell = MC.total_energy(st_check, p_check)
            E_brute = MC.energy_bruteforce(st_check, p_check)
            P_cell = MC.pressure(st_check, p_check, T)
            P_brute = MC.pressure_bruteforce(st_check, p_check, T)
            println("  sweep $(sweep_idx): ΔE = $(E_cell - E_brute), ΔP = $(P_cell - P_brute)")
        end
    end
end

# --- Run comparison ---
result = MC.compare_nvt(
    N=N, ρ=ρ, T=T, rc=rc,
    use_lrc=use_lrc,
    lj_model=lj_model,
    init_lattice=:auto,
    burnin_sweeps_mc=burnin_sweeps_mc,
    prod_sweeps_mc=prod_sweeps_mc,
    burnin_events_ekmc=burnin_events_ekmc,
    prod_events_ekmc=prod_events_ekmc,
    sample_every_mc=sample_every_mc,
    sample_every_ekmc=sample_every_ekmc,
    dlnV_virtual=dlnV_virtual,
    debug_check_every_ekmc=debug_check_every_ekmc,
    widom_every_mc=widom_every_mc,
    widom_ninsert_mc=widom_ninsert_mc,
    profile=true
)

println("Finished MC + eKMC validation run.")

MC.print_comparison(result)

println()
println("NKEOS (Python reference port):")
P_nkeos = MolSim.EOS.pressure_nkeos(T, ρ)
U_nkeos = MolSim.EOS.internal_energy_nkeos(T, ρ)
mu_nkeos = MolSim.EOS.nkeos_alj_res(T, ρ)
mu_lrc = result.mu_lrc
println("  P_nkeos = ", P_nkeos)
println("  U/N     = ", U_nkeos)
println("  μ_ex    = ", mu_nkeos)
println("  μ_LRC   = ", mu_lrc)
println("  μ_ex+μ_LRC = ", mu_nkeos + mu_lrc)
println("  ΔP (eKMC - NKEOS) = ", result.ekmc_P - P_nkeos)
println("  ΔP (MC   - NKEOS) = ", result.mc_P - P_nkeos)
println("  ΔU (eKMC - NKEOS) = ", result.ekmc_U_per_particle - U_nkeos)
println("  ΔU (MC   - NKEOS) = ", result.mc_U_per_particle - U_nkeos)
println("  Δμ_ex (eKMC - NKEOS) = ", result.ekmc_mu_ex - mu_nkeos)
println("  Δμ_ex (MC   - NKEOS) = ", result.mc_mu_ex - mu_nkeos)
println("  Δ(μ_ex+μ_LRC) eKMC = ", result.ekmc_mu_ex_lrc - mu_nkeos)
println("  Δ(μ_ex+μ_LRC) MC   = ", result.mc_mu_ex_lrc - mu_nkeos)

println()
println("NKEOS_Author (wrapper):")
P_nkeos_a = MolSim.EOS.pressure_nkeos_author(T, ρ)
U_nkeos_a = MolSim.EOS.internal_energy_nkeos_author(T, ρ)
mu_nkeos_a = MolSim.EOS.alj_res_author(T, ρ)
println("  P_nkeos_author = ", P_nkeos_a)
println("  U/N            = ", U_nkeos_a)
println("  μ_ex           = ", mu_nkeos_a)
println("  ΔP (eKMC - NKEOS_Author) = ", result.ekmc_P - P_nkeos_a)
println("  ΔP (MC   - NKEOS_Author) = ", result.mc_P - P_nkeos_a)
println("  ΔU (eKMC - NKEOS_Author) = ", result.ekmc_U_per_particle - U_nkeos_a)
println("  ΔU (MC   - NKEOS_Author) = ", result.mc_U_per_particle - U_nkeos_a)
println("  Δμ_ex (eKMC - NKEOS_Author) = ", result.ekmc_mu_ex - mu_nkeos_a)
println("  Δμ_ex (MC   - NKEOS_Author) = ", result.mc_mu_ex - mu_nkeos_a)

# --- Optional: write CSV summary ---
MC.write_comparison_csv(result, "nvt_validation.csv")
println()
println("Wrote: nvt_validation.csv")
