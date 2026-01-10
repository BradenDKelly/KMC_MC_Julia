"""
Sanity check for Johnson 1993 EOS derivative implementation.
Compares analytical Z to numerical Z from central difference.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

include(joinpath(@__DIR__, "..", "src", "EOS", "LJJohnson1993.jl"))

println("=" ^ 60)
println("Johnson 1993 EOS Derivative Sanity Check")
println("=" ^ 60)
println()

# Test point 1: (T,ρ) = (1.0, 0.01)
T = 1.0
ρ = 0.01
h = 1e-6

println("Test point 1: T = $T, ρ = $ρ")
println("Using central difference with h = $h")
println()

# Numerical Z from central difference
# Note: get_alphar_johnson returns alphar (not alphar/T)
# So d(alphar)/dρ = (alphar(ρ+h) - alphar(ρ-h))/(2h)
# Then Z = 1 + ρ*(d(alphar)/dρ)/T
alphar_plus = get_alphar_johnson(T, ρ + h)
alphar_minus = get_alphar_johnson(T, ρ - h)
dalphar_drho_numeric = (alphar_plus - alphar_minus) / (2.0 * h)
Z_numeric = 1.0 + ρ * dalphar_drho_numeric / T

println("Numerical derivative:")
println("  alphar(ρ+h) = $alphar_plus")
println("  alphar(ρ-h) = $alphar_minus")
println("  d(alphar)/dρ (numerical) = $dalphar_drho_numeric")
println("  Z (numerical) = 1 + ρ*(d(alphar)/dρ)/T = $Z_numeric")
println()

# Analytical Z
P_analytical = MolSim.EOS.pressure_johnson(T, ρ)
Z_analytical = P_analytical / (ρ * T)

println("Analytical Z:")
println("  P = $P_analytical")
println("  Z = $Z_analytical")
println()

# Compare
diff = abs(Z_analytical - Z_numeric)
println("Comparison:")
println("  |Z_analytical - Z_numeric| = $diff")
if diff < 1e-6
    println("  ✓ PASS: Difference < 1e-6")
else
    println("  ✗ FAIL: Difference >= 1e-6")
end
println()

# Test point 2: ρ = 1e-6 (ideal gas limit)
T2 = 1.0
ρ2 = 1e-6

println("Test point 2: T = $T2, ρ = $ρ2 (ideal gas limit)")
P2 = MolSim.EOS.pressure_johnson(T2, ρ2)
Z2 = P2 / (ρ2 * T2)

println("  Z = $Z2")
println("  |Z - 1| = $(abs(Z2 - 1.0))")
if abs(Z2 - 1.0) < 1e-6
    println("  ✓ PASS: |Z - 1| < 1e-6")
else
    println("  ✗ FAIL: |Z - 1| >= 1e-6")
end
println()

# Summary
all_pass = (diff < 1e-6) && (abs(Z2 - 1.0) < 1e-6)
if all_pass
    println("=" ^ 60)
    println("✓ ALL TESTS PASS")
    println("=" ^ 60)
else
    println("=" ^ 60)
    println("✗ SOME TESTS FAILED")
    println("=" ^ 60)
end
