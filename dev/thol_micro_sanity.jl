"""
Micro sanity check: verify Z approaches 1 at low density (ideal gas limit).
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

println("Thol EOS low-density sanity check (ideal gas limit)")
println("=" ^ 60)

for (T, ρ) in ((2.0, 1e-4), (2.0, 1e-3), (2.0, 1e-2))
    P = MolSim.EOS.pressure_thol(T, ρ)
    Z = P / (ρ * T)
    println("T=$T, ρ=$ρ: Z=$Z")
    
    # Check if Z is close to 1
    if ρ <= 1e-3
        if abs(Z - 1.0) > 0.01
            println("  ✗ WARNING: Z should be ≈ 1.0 at low density, but got $Z")
        else
            println("  ✓ OK: Z ≈ 1.0 (ideal gas limit)")
        end
    else
        println("  (ρ = $ρ is moderate density, Z may deviate from 1)")
    end
end
