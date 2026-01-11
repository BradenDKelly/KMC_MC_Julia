#!/usr/bin/env julia
"""
Script to run LJ NVT MC simulations at multiple densities, compute observables,
compare with EOS, and generate RDF plots.

Usage: julia --project dev/multidensity_analysis.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using Random

# ============================================================================
# Parameters
# ============================================================================
const N = 500
const T = 1.0
const ρ_list = [0.4, 0.6, 0.8]
const rc = 2.5
const max_disp = 0.1
const seed_base = 1234

# MC simulation parameters
const warmup_sweeps = 2000
const prod_sweeps = 20000
const sample_every = 10
const block_size = 50

# Widom parameters
const widom_every = 200
const widom_ninsert = 200

# RDF parameters
const rdf_nbins = 200
const rdf_rmax = nothing  # Will be set to L/2

# Output directory
const output_dir = joinpath(@__DIR__, "out")

# ============================================================================
# Helper functions
# ============================================================================

"""
Write RDF to CSV file.
"""
function write_rdf_csv(path::String, r::Vector{Float64}, g_r::Vector{Float64})
    open(path, "w") do io
        println(io, "r,g_r")
        for i in eachindex(r)
            println(io, "$(r[i]),$(g_r[i])")
        end
    end
    return nothing
end

"""
Write simulation results summary to CSV.
"""
function write_summary_csv(path::String, results::Vector{Dict{String, Any}})
    open(path, "w") do io
        # Header
        println(io, "rho,U_mean,U_stderr,P_mean,P_stderr,mu_ex_mean,mu_ex_stderr,Z_MC,U_EOS,Z_EOS")
        
        # Data rows
        for r in results
            ρ = r["rho"]
            U_mean = r["U_mean"]
            U_stderr = r["U_stderr"]
            P_mean = r["P_mean"]
            P_stderr = r["P_stderr"]
            mu_ex_mean = r["mu_ex_mean"]
            mu_ex_stderr = r["mu_ex_stderr"]
            Z_MC = r["Z_MC"]
            U_EOS = r["U_EOS"]
            Z_EOS = r["Z_EOS"]
            
            println(io, "$ρ,$U_mean,$U_stderr,$P_mean,$P_stderr,$mu_ex_mean,$mu_ex_stderr,$Z_MC,$U_EOS,$Z_EOS")
        end
    end
    return nothing
end

# ============================================================================
# Main script
# ============================================================================

# Create output directory
if !isdir(output_dir)
    mkdir(output_dir)
end

println("=" ^ 80)
println("LJ NVT MC: Multi-density Analysis")
println("=" ^ 80)
println()
println("Parameters:")
println("  N = $N")
println("  T = $T")
println("  ρ = $ρ_list")
println("  rc = $rc")
println("  warmup = $warmup_sweeps sweeps")
println("  production = $prod_sweeps sweeps")
println("  sample_every = $sample_every sweeps")
println()

# Storage for results
results = Vector{Dict{String, Any}}()

# Run simulations for each density
for (idx, ρ) in enumerate(ρ_list)
    println("=" ^ 80)
    println("Density $idx/$(length(ρ_list)): ρ = $ρ")
    println("=" ^ 80)
    println()
    
    # Seed for reproducibility
    seed = seed_base + idx * 1000
    Random.seed!(seed)
    
    # Initialize system
    p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp, 
                                seed=seed, use_lrc=false)
    L = st.L
    rmax_actual = rdf_rmax === nothing ? L / 2.0 : rdf_rmax
    
    println("  Box size L = $L")
    println("  RDF rmax = $rmax_actual")
    println()
    
    # Block averagers
    U_ba = MolSim.MC.BlockAverager(block_size)
    P_ba = MolSim.MC.BlockAverager(block_size)
    
    # Widom accumulator
    widom_acc = MolSim.MC.WidomAccumulator()
    MolSim.MC.reset!(widom_acc)
    
    # Warmup
    println("  Warmup: $warmup_sweeps sweeps...")
    for _ in 1:warmup_sweeps
        MolSim.MC.sweep!(st, p; rebuild_every=1)
    end
    
    # Production
    println("  Production: $prod_sweeps sweeps...")
    sample_count = 0
    widom_count = 0
    
    for sweep in 1:prod_sweeps
        MolSim.MC.sweep!(st, p; rebuild_every=1)
        
        # Sample observables
        if sweep % sample_every == 0
            sample_count += 1
            
            # Energy per particle
            U_total = MolSim.MC.total_energy(st, p)
            U_per_particle = U_total / N
            push!(U_ba, U_per_particle)
            
            # Pressure
            P = MolSim.MC.pressure(st, p, T)
            push!(P_ba, P)
        end
        
        # Widom insertion
        if sweep % widom_every == 0
            widom_count += 1
            MolSim.MC.widom_mu_ex!(widom_acc, st, p; ninsert=widom_ninsert)
        end
    end
    
    # Compute statistics
    U_mean = MolSim.MC.mean(U_ba)
    U_stderr = MolSim.MC.stderr(U_ba)
    P_mean = MolSim.MC.mean(P_ba)
    P_stderr = MolSim.MC.stderr(P_ba)
    
    # Compressibility factor
    Z_MC = P_mean / (ρ * T)
    
    # Chemical potential
    if widom_acc.n > 0
        β = p.β
        mu_ex_mean = MolSim.MC.mu_ex(widom_acc, β)
    else
        mu_ex_mean = NaN
    end
    mu_ex_stderr = NaN  # Widom doesn't provide error easily (would need multiple runs)
    
    # EOS predictions
    P_EOS = NaN
    Z_EOS = NaN
    U_EOS = NaN
    
    try
        P_EOS = MolSim.EOS.pressure_kolafa_sklogwiki(T, ρ)
        Z_EOS = P_EOS / (ρ * T)
    catch e
        println("    Warning: Could not compute EOS pressure: $e")
    end
    
    # Try to get internal energy from virial EOS if available
    try
        U_EOS = MolSim.EOS.internal_energy(T, ρ)
    catch e
        # Keep NaN if not available
    end
    
    println()
    println("  Results:")
    println("    U/N = $U_mean ± $U_stderr")
    println("    P = $P_mean ± $P_stderr")
    println("    Z_MC = $Z_MC")
    if !isnan(mu_ex_mean)
        println("    μ_ex = $mu_ex_mean")
    end
    if !isnan(Z_EOS)
        println("    Z_EOS = $Z_EOS")
    end
    if !isnan(U_EOS)
        println("    U_EOS/N = $U_EOS")
    end
    println()
    
    # Compute RDF on final configuration
    println("  Computing RDF...")
    r, g_r = MolSim.Analysis.rdf(st.pos, L, rmax_actual, rdf_nbins)
    rdf_path = joinpath(output_dir, "rdf_rho_$(ρ).csv")
    write_rdf_csv(rdf_path, r, g_r)
    println("    RDF saved to $(basename(rdf_path))")
    println()
    
    # Store results
    push!(results, Dict(
        "rho" => ρ,
        "U_mean" => U_mean,
        "U_stderr" => U_stderr,
        "P_mean" => P_mean,
        "P_stderr" => P_stderr,
        "mu_ex_mean" => mu_ex_mean,
        "mu_ex_stderr" => mu_ex_stderr,
        "Z_MC" => Z_MC,
        "U_EOS" => U_EOS,
        "Z_EOS" => Z_EOS
    ))
end

# Write summary
summary_path = joinpath(output_dir, "multidensity_summary.csv")
write_summary_csv(summary_path, results)

println("=" ^ 80)
println("Summary")
println("=" ^ 80)
println()
println("ρ        U/N (MC)     U/N (EOS)    P (MC)       Z (MC)       Z (EOS)      μ_ex")
println("-" ^ 80)
for r in results
    ρ = r["rho"]
    U_MC = r["U_mean"]
    U_EOS = r["U_EOS"]
    P_MC = r["P_mean"]
    Z_MC = r["Z_MC"]
    Z_EOS = r["Z_EOS"]
    mu_ex = r["mu_ex_mean"]
    
    U_EOS_str = isnan(U_EOS) ? "N/A" : string(round(U_EOS, digits=4))
    Z_EOS_str = isnan(Z_EOS) ? "N/A" : string(round(Z_EOS, digits=4))
    mu_ex_str = isnan(mu_ex) ? "N/A" : string(round(mu_ex, digits=4))
    
    U_MC_str = string(round(U_MC, digits=4))
    P_MC_str = string(round(P_MC, digits=4))
    Z_MC_str = string(round(Z_MC, digits=4))
    
    println(lpad(string(round(ρ, digits=2)), 8) * "  " * 
            lpad(U_MC_str, 12) * "  " * 
            lpad(U_EOS_str, 12) * "  " * 
            lpad(P_MC_str, 12) * "  " * 
            lpad(Z_MC_str, 12) * "  " * 
            lpad(Z_EOS_str, 12) * "  " * 
            lpad(mu_ex_str, 12))
end
println("-" ^ 80)
println()
println("Output files written to $(output_dir):")
println("  multidensity_summary.csv - Summary of all densities")
for ρ in ρ_list
    println("  rdf_rho_$(ρ).csv - RDF for ρ = $ρ")
end
println()

# ============================================================================
# Plotting (optional, requires Plots.jl)
# ============================================================================

try
    using Pkg
    Pkg.add("Plots")
    using Plots
    
    println("=" ^ 80)
    println("Generating RDF plots...")
    println("=" ^ 80)
    println()
    
    # Plot RDF for all densities
    p_rdf = plot(xlabel="r", ylabel="g(r)", title="Radial Distribution Function (T = $T)", 
                 legend=:topright, grid=true)
    
    colors = [:blue, :red, :green]
    for (idx, ρ) in enumerate(ρ_list)
        rdf_path = joinpath(output_dir, "rdf_rho_$(ρ).csv")
        if isfile(rdf_path)
            # Read RDF data
            data = readlines(rdf_path)
            r_vals = Float64[]
            g_vals = Float64[]
            for line in data[2:end]  # Skip header
                parts = split(line, ',')
                if length(parts) == 2
                    push!(r_vals, parse(Float64, parts[1]))
                    push!(g_vals, parse(Float64, parts[2]))
                end
            end
            
            plot!(p_rdf, r_vals, g_vals, label="ρ = $ρ", color=colors[idx], linewidth=2)
        end
    end
    
    # Save plot
    plot_path = joinpath(output_dir, "rdf_plot.png")
    savefig(p_rdf, plot_path)
    println("  RDF plot saved to $(basename(plot_path))")
    println()
    
catch e
    println("  Plotting skipped (Plots.jl not available or error: $e)")
    println("  Install Plots.jl to generate plots: using Pkg; Pkg.add(\"Plots\")")
    println()
end

println("Done!")
