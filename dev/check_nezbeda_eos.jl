#!/usr/bin/env julia
"""
Quick check to compare Nezbeda-Author EOS with SklogWiki implementation.

Usage: julia --project dev/check_nezbeda_eos.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using Printf

println("Comparing Nezbeda-Author EOS with SklogWiki implementation")
println("=" ^ 70)
println()

test_points = [
    (1.0, 0.1),
    (1.0, 0.3),
    (1.0, 0.5),
    (1.5, 0.3),
    (2.0, 0.2),
]

println("T        ρ        Z_SklogWiki    Z_NezbedaAuthor    Diff        Rel_Diff(%)")
println("-" ^ 70)

for (T, rho) in test_points
    P_sklog = MolSim.EOS.pressure_kolafa_sklogwiki(T, rho)
    Z_sklog = P_sklog / (rho * T)
    
    P_nez = MolSim.EOS.pressure_nezbeda_author(T, rho)
    Z_nez = P_nez / (rho * T)
    
    diff = abs(Z_sklog - Z_nez)
    rel_diff = diff / abs(Z_sklog) * 100.0
    
    println(@sprintf("%6.2f  %6.2f  %12.6f  %18.6f  %10.6e  %10.4f", 
                     T, rho, Z_sklog, Z_nez, diff, rel_diff))
end

println()
println("Note: Nezbeda-Author currently uses the SklogWiki implementation.")
println("      They should match exactly (diff should be 0.0).")
println()
