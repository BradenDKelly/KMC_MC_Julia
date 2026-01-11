#!/usr/bin/env julia
"""
Script to compare different Kolafa-Nezbeda EOS implementations with the authoritative
Nezbeda implementation (when available).

Usage: julia --project dev/compare_eos_nezbeda.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using Printf

# Test state points
T_values = [1.0, 1.5, 2.0]
rho_values = [0.1, 0.3, 0.5, 0.7, 0.8]

println("=" ^ 80)
println("EOS Comparison: Kolafa-Nezbeda Implementations")
println("=" ^ 80)
println()
println("Comparing:")
println("  1. Kolafa-SklogWiki (current implementation)")
println("  2. Nezbeda-Author (placeholder - needs NKEOS module)")
println()
println("State points: T = $T_values, ρ = $rho_values")
println()

println("T        ρ        Z_SklogWiki    Z_NezbedaAuthor    Diff")
println("-" ^ 80)

for T in T_values
    for rho in rho_values
        # SklogWiki implementation
        try
            P_sklog = MolSim.EOS.pressure_kolafa_sklogwiki(T, rho)
            Z_sklog = P_sklog / (rho * T)
            
            # Nezbeda Author implementation (placeholder)
            try
                P_nez = MolSim.EOS.pressure_nezbeda_author(T, rho)
                Z_nez = P_nez / (rho * T)
                diff = abs(Z_sklog - Z_nez)
                
                println(@sprintf("%6.2f  %6.2f  %12.6f  %18.6f  %10.6f", 
                                 T, rho, Z_sklog, Z_nez, diff))
            catch e
                println(@sprintf("%6.2f  %6.2f  %12.6f  %18s  %10s", 
                                 T, rho, Z_sklog, "N/A (placeholder)", "N/A"))
            end
        catch e
            println(@sprintf("%6.2f  %6.2f  %12s  %18s  %10s", 
                             T, rho, "ERROR", "N/A", "N/A"))
        end
    end
end

println()
println("Note: The Nezbeda-Author implementation is currently a placeholder.")
println("      It calls the SklogWiki implementation until the actual NKEOS")
println("      module functions (PLJ, ALJres) are ported from Python.")
println()
