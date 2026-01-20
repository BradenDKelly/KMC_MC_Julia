"""
Validate NVT eKMC vs Metropolis MC and compare with EOS.

Compares MC and eKMC results with:
- TholAllen EOS (cut-and-shifted at r_cut = 2.5σ) - PRIMARY COMPARISON
- NKEOS (full LJ) - for reference only

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
# Cut-and-shifted LJ at r_cut = 2.5σ (matching TholAllen EOS)
const N = 864
const ρ = 0.75 #0.8645
const T = 1.0 #0.81
const L = cbrt(N / ρ)
const rc = 2.5  # Fixed cutoff matching TholAllen cut-and-shifted EOS
const use_lrc = false  # No LRC for cut-and-shifted potential
const lj_model = :shifted  # Cut-and-shifted LJ

const burnin_sweeps_mc = 2000
const prod_sweeps_mc = 2000
# Match number of particle translations: 1 sweep ≈ N translations
const burnin_events_ekmc = burnin_sweeps_mc * N 
const prod_events_ekmc = prod_sweeps_mc * N
const sample_every_mc = 10
# Match number of samples between MC and eKMC
const n_samples_mc = prod_sweeps_mc ÷ sample_every_mc
# Sample eKMC 10x more frequently than MC for better time-weighted averaging
const sample_every_ekmc = max(1, (prod_events_ekmc ÷ n_samples_mc))
const dlnV_virtual = 1e-4
const debug_check_every_ekmc = 1000
const widom_every_mc = 100
const widom_ninsert_mc = 2000

println("Starting MC + eKMC validation run...")

# --- Optional MC sanity check (bruteforce vs cell list) ---
const mc_sanity_check = false
const mc_sanity_sweeps = 1
const mc_sanity_every = 1

if mc_sanity_check
    p_check, st_check = MC.init_fcc(; N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1,
                                    seed=612354, use_lrc=use_lrc, lj_model=lj_model)
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
comparison_result = MC.compare_nvt(
    N=N, ρ=ρ, T=T, rc=rc,
    use_lrc=use_lrc,
    lj_model=lj_model,
    init_lattice=:fcc,
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
    profile=true,
    collect_timeseries=true
)

# Unpack results (compare_nvt returns tuple when collect_timeseries=true)
if isa(comparison_result, Tuple)
    if length(comparison_result) == 5
        result, timeseries_mc, timeseries_ekmc, rdf_mc, rdf_ekmc = comparison_result
    elseif length(comparison_result) == 3
        result, timeseries_mc, timeseries_ekmc = comparison_result
        rdf_mc = nothing
        rdf_ekmc = nothing
    else
        result = comparison_result[1]
        timeseries_mc = length(comparison_result) > 1 ? comparison_result[2] : nothing
        timeseries_ekmc = length(comparison_result) > 2 ? comparison_result[3] : nothing
        rdf_mc = length(comparison_result) > 3 ? comparison_result[4] : nothing
        rdf_ekmc = length(comparison_result) > 4 ? comparison_result[5] : nothing
    end
else
    result = comparison_result
    timeseries_mc = nothing
    timeseries_ekmc = nothing
    rdf_mc = nothing
    rdf_ekmc = nothing
end

println("Finished MC + eKMC validation run.")

MC.print_comparison(result)

println()
println("NKEOS (Python reference port):")
P_nkeos = MolSim.EOS.pressure_nkeos(T, ρ)
U_nkeos = MolSim.EOS.internal_energy_nkeos(T, ρ)
mu_res_nkeos = MolSim.EOS.chemical_potential_nkeos(T, ρ)
mu_lrc = result.mu_lrc
println("  P_nkeos = ", P_nkeos)
println("  U/N     = ", U_nkeos)
println("  μ_res   = ", mu_res_nkeos)
println("  μ_LRC   = ", mu_lrc)
println("  μ_res+μ_LRC = ", mu_res_nkeos + mu_lrc)
println("  ΔP (eKMC - NKEOS) = ", result.ekmc_P - P_nkeos)
println("  ΔP (MC   - NKEOS) = ", result.mc_P - P_nkeos)
println("  ΔU (eKMC - NKEOS) = ", result.ekmc_U_per_particle - U_nkeos)
println("  ΔU (MC   - NKEOS) = ", result.mc_U_per_particle - U_nkeos)
println("  Δμ_ex (eKMC - NKEOS μ_res) = ", result.ekmc_mu_ex - mu_res_nkeos)
println("  Δμ_ex (MC   - NKEOS μ_res) = ", result.mc_mu_ex - mu_res_nkeos)
println("  Δ(μ_ex+μ_LRC) eKMC vs (μ_res+μ_LRC) = ", result.ekmc_mu_ex_lrc - (mu_res_nkeos + mu_lrc))
println("  Δ(μ_ex+μ_LRC) MC   vs (μ_res+μ_LRC) = ", result.mc_mu_ex_lrc - (mu_res_nkeos + mu_lrc))

println()
println("NKEOS_Author (wrapper):")
P_nkeos_a = MolSim.EOS.pressure_nkeos_author(T, ρ)
U_nkeos_a = MolSim.EOS.internal_energy_nkeos_author(T, ρ)
mu_res_nkeos_a = MolSim.EOS.chemical_potential_nkeos_author(T, ρ)
println("  P_nkeos_author = ", P_nkeos_a)
println("  U/N            = ", U_nkeos_a)
println("  μ_res          = ", mu_res_nkeos_a)
println("  ΔP (eKMC - NKEOS_Author) = ", result.ekmc_P - P_nkeos_a)
println("  ΔP (MC   - NKEOS_Author) = ", result.mc_P - P_nkeos_a)
println("  ΔU (eKMC - NKEOS_Author) = ", result.ekmc_U_per_particle - U_nkeos_a)
println("  ΔU (MC   - NKEOS_Author) = ", result.mc_U_per_particle - U_nkeos_a)
println("  Δμ_ex (eKMC - NKEOS_Author μ_res) = ", result.ekmc_mu_ex - mu_res_nkeos_a)
println("  Δμ_ex (MC   - NKEOS_Author μ_res) = ", result.mc_mu_ex - mu_res_nkeos_a)

println()
println("=" ^ 70)
println("TholAllen EOS (cut-and-shifted at r_cut = 2.5σ):")
println("=" ^ 70)
P_thol = MolSim.EOS.pressure_thol_cutshift(T, ρ)
U_thol = MolSim.EOS.internal_energy_thol_cutshift(T, ρ)
mu_res_thol = MolSim.EOS.chemical_potential_residual_thol_cutshift(T, ρ)
println("  P_thol (cutshift) = ", P_thol)
println("  U/N (cutshift)    = ", U_thol)
println("  μ_res (cutshift)  = ", mu_res_thol)
println()
println("Comparison with MC/eKMC (cut-and-shifted, rc=2.5, no LRC):")
println("  ΔP (eKMC - TholAllen) = ", result.ekmc_P - P_thol)
println("  ΔP (MC   - TholAllen) = ", result.mc_P - P_thol)
U_thol_config = U_thol - 1.5 * T  # EOS total U - ideal gas kinetic energy (3/2)T
println("  U_thol (total)    = ", U_thol)
println("  U_thol (config)   = ", U_thol_config, "  [total - (3/2)T]")
println("  ΔU_config (eKMC - TholAllen config) = ", result.ekmc_U_per_particle - U_thol_config)
println("  ΔU_config (MC   - TholAllen config) = ", result.mc_U_per_particle - U_thol_config)
println("  ΔU_total (eKMC - TholAllen total) = ", result.ekmc_U_per_particle - U_thol)
println("  ΔU_total (MC   - TholAllen total) = ", result.mc_U_per_particle - U_thol)
println("  Δμ_ex (eKMC - TholAllen μ_res) = ", result.ekmc_mu_ex - mu_res_thol)
println("  Δμ_ex (MC   - TholAllen μ_res) = ", result.mc_mu_ex - mu_res_thol)

# --- Optional: write CSV summary ---
MC.write_comparison_csv(result, "nvt_validation.csv")
println()
println("Wrote: nvt_validation.csv")

# --- Plot RDF comparison ---
if rdf_mc !== nothing && rdf_ekmc !== nothing
    MC.plot_rdf_comparison(rdf_mc, rdf_ekmc, "rdf_comparison.png")
end

# --- Plot time series ---
if timeseries_mc !== nothing || timeseries_ekmc !== nothing
    try
        using Plots
        gr()  # Use GR backend
        
        # MC plots
        if timeseries_mc !== nothing && (length(timeseries_mc.sweeps_burnin) > 0 || length(timeseries_mc.sweeps_prod) > 0)
            # Combine burn-in and production (handle empty arrays)
            sweeps_burnin = length(timeseries_mc.sweeps_burnin) > 0 ? timeseries_mc.sweeps_burnin : Float64[]
            sweeps_prod = length(timeseries_mc.sweeps_prod) > 0 ? timeseries_mc.sweeps_prod .+ burnin_sweeps_mc : Float64[]
            U_burnin = length(timeseries_mc.U_burnin) > 0 ? timeseries_mc.U_burnin : Float64[]
            U_prod = length(timeseries_mc.U_prod) > 0 ? timeseries_mc.U_prod : Float64[]
            P_burnin = length(timeseries_mc.P_burnin) > 0 ? timeseries_mc.P_burnin : Float64[]
            P_prod = length(timeseries_mc.P_prod) > 0 ? timeseries_mc.P_prod : Float64[]
            mu_burnin = length(timeseries_mc.mu_burnin) > 0 ? timeseries_mc.mu_burnin : Float64[]
            mu_prod = length(timeseries_mc.mu_prod) > 0 ? timeseries_mc.mu_prod : Float64[]
            
            # Combine burn-in and production
            sweeps_mc = [sweeps_burnin; sweeps_prod]
            U_mc_all = [U_burnin; U_prod]
            P_mc_all = [P_burnin; P_prod]
            mu_mc_all = [mu_burnin; mu_prod]
            
            # Ensure mu array matches length of other arrays (pad with NaN if needed)
            while length(mu_mc_all) < length(sweeps_mc)
                push!(mu_mc_all, NaN)
            end
            if length(mu_mc_all) > length(sweeps_mc)
                mu_mc_all = mu_mc_all[1:length(sweeps_mc)]
            end
            
            # Filter out NaN values for mu
            mu_valid_idx = .!isnan.(mu_mc_all)
            if sum(mu_valid_idx) > 0
                sweeps_mc_mu = sweeps_mc[mu_valid_idx]
                mu_mc_valid = mu_mc_all[mu_valid_idx]
            else
                sweeps_mc_mu = Float64[]
                mu_mc_valid = Float64[]
            end
            
            p1 = plot(sweeps_mc, U_mc_all, label="U/N", xlabel="Sweep", ylabel="U/N", 
                     title="MC: Energy per particle", linewidth=2)
            vline!(p1, [burnin_sweeps_mc], linestyle=:dash, color=:gray, label="Burn-in end")
            
            p2 = plot(sweeps_mc, P_mc_all, label="P", xlabel="Sweep", ylabel="P", 
                     title="MC: Pressure", linewidth=2)
            vline!(p2, [burnin_sweeps_mc], linestyle=:dash, color=:gray, label="Burn-in end")
            
            if length(sweeps_mc_mu) > 0
                p3 = plot(sweeps_mc_mu, mu_mc_valid, label="μ_ex", xlabel="Sweep", ylabel="μ_ex", 
                         title="MC: Chemical potential", linewidth=2)
                vline!(p3, [burnin_sweeps_mc], linestyle=:dash, color=:gray, label="Burn-in end")
            else
                p3 = plot([], [], label="μ_ex", xlabel="Sweep", ylabel="μ_ex", 
                         title="MC: Chemical potential (no data)", linewidth=2)
            end
            
            plot_mc = plot(p1, p2, p3, layout=(3,1), size=(800,1200))
            savefig(plot_mc, "mc_timeseries.png")
            println("Saved: mc_timeseries.png")
        end
        
        # eKMC plots
        if timeseries_ekmc !== nothing && (length(timeseries_ekmc.events_burnin) > 0 || length(timeseries_ekmc.events_prod) > 0)
            # Combine burn-in and production (handle empty arrays)
            events_burnin = length(timeseries_ekmc.events_burnin) > 0 ? timeseries_ekmc.events_burnin : Float64[]
            events_prod = length(timeseries_ekmc.events_prod) > 0 ? timeseries_ekmc.events_prod .+ burnin_events_ekmc : Float64[]
            U_burnin = length(timeseries_ekmc.U_burnin) > 0 ? timeseries_ekmc.U_burnin : Float64[]
            U_prod = length(timeseries_ekmc.U_prod) > 0 ? timeseries_ekmc.U_prod : Float64[]
            P_burnin = length(timeseries_ekmc.P_burnin) > 0 ? timeseries_ekmc.P_burnin : Float64[]
            P_prod = length(timeseries_ekmc.P_prod) > 0 ? timeseries_ekmc.P_prod : Float64[]
            mu_burnin = length(timeseries_ekmc.mu_burnin) > 0 ? timeseries_ekmc.mu_burnin : Float64[]
            mu_prod = length(timeseries_ekmc.mu_prod) > 0 ? timeseries_ekmc.mu_prod : Float64[]
            
            events_ekmc = [events_burnin; events_prod]
            U_ekmc_all = [U_burnin; U_prod]
            P_ekmc_all = [P_burnin; P_prod]
            mu_ekmc_all = [mu_burnin; mu_prod]
            
            # Ensure mu array matches length of other arrays (pad with NaN if needed)
            while length(mu_ekmc_all) < length(events_ekmc)
                push!(mu_ekmc_all, NaN)
            end
            if length(mu_ekmc_all) > length(events_ekmc)
                mu_ekmc_all = mu_ekmc_all[1:length(events_ekmc)]
            end
            
            # Filter out NaN values for mu
            mu_valid_idx = .!isnan.(mu_ekmc_all)
            if sum(mu_valid_idx) > 0
                events_ekmc_mu = events_ekmc[mu_valid_idx]
                mu_ekmc_valid = mu_ekmc_all[mu_valid_idx]
            else
                events_ekmc_mu = Float64[]
                mu_ekmc_valid = Float64[]
            end
            
            p1 = plot(events_ekmc, U_ekmc_all, label="U/N", xlabel="Event", ylabel="U/N", 
                     title="eKMC: Energy per particle", linewidth=2)
            vline!(p1, [burnin_events_ekmc], linestyle=:dash, color=:gray, label="Burn-in end")
            
            p2 = plot(events_ekmc, P_ekmc_all, label="P", xlabel="Event", ylabel="P", 
                     title="eKMC: Pressure", linewidth=2)
            vline!(p2, [burnin_events_ekmc], linestyle=:dash, color=:gray, label="Burn-in end")
            
            if length(events_ekmc_mu) > 0
                p3 = plot(events_ekmc_mu, mu_ekmc_valid, label="μ_ex", xlabel="Event", ylabel="μ_ex", 
                         title="eKMC: Chemical potential", linewidth=2)
                vline!(p3, [burnin_events_ekmc], linestyle=:dash, color=:gray, label="Burn-in end")
            else
                p3 = plot([], [], label="μ_ex", xlabel="Event", ylabel="μ_ex", 
                         title="eKMC: Chemical potential (no data)", linewidth=2)
            end
            
            plot_ekmc = plot(p1, p2, p3, layout=(3,1), size=(800,1200))
            savefig(plot_ekmc, "ekmc_timeseries.png")
            println("Saved: ekmc_timeseries.png")
        end
    catch e
        println("Warning: Could not create plots. Error: $e")
        println("Install Plots.jl with: using Pkg; Pkg.add(\"Plots\")")
    end
end
