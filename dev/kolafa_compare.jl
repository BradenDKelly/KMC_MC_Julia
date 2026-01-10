"""
Compare Kolafa SklogWiki (TRUSTED) vs current Kolafa implementation (UNTRUSTED).
Diagnostic script to show differences between implementations.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

include(joinpath(@__DIR__, "..", "src", "EOS", "LJKolafaSklogwiki1994.jl"))

println("Kolafa EOS Comparison (Diagnostic)")
println("=" ^ 70)
println("TRUSTED: pressure_kolafa_sklogwiki (SklogWiki PLJ oracle)")
println("UNTRUSTED: pressure_kolafa (current implementation - known to be wrong)")
println("=" ^ 70)
println()

test_points = [
    (T=1.0, ρ=0.05),
    (T=1.0, ρ=0.1),
    (T=2.0, ρ=0.2),
]

for pt in test_points
    T = pt.T
    ρ = pt.ρ
    
    Z_trusted = pressure_kolafa_sklogwiki(T, ρ) / (ρ * T)
    
    # Check if current implementation exists
    if isdefined(MolSim.EOS, :pressure_kolafa)
        Z_untrusted = MolSim.EOS.pressure_kolafa(T, ρ) / (ρ * T)
        diff = abs(Z_trusted - Z_untrusted)
        println("T=$T, ρ=$ρ:")
        println("  Z_trusted   = $Z_trusted  (SklogWiki PLJ)")
        println("  Z_untrusted = $Z_untrusted  (current - BROKEN)")
        println("  |diff|      = $diff")
    else
        println("T=$T, ρ=$ρ:")
        println("  Z_trusted   = $Z_trusted  (SklogWiki PLJ)")
        println("  Z_untrusted = (not available)")
    end
    println()
end
