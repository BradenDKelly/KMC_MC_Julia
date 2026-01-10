"""
Quick check: compute Z at (T=1.0, ρ=0.3) using teqp-style formulas to verify our implementation.
"""

println("Checking Thol EOS formula at T=1.0, ρ=0.3")
println("=" ^ 60)

T = 1.0
ρ = 0.3

# Thol reduced variables
T_ref = 1.32
ρ_ref = 0.31
δ = ρ / ρ_ref
τ = T_ref / T

println("δ = $δ, τ = $τ")
println()

# Check if maybe the issue is that we need to divide α^r by T?
# Or maybe the formula is Z = 1 + δ * (∂α^r/∂δ) / T?

println("Standard formula: Z = 1 + δ * (∂α^r/∂δ)")
println("  This assumes α^r = a^r/(kT) already normalized")
println()

# Check MC result
println("MC result at (T=1.0, ρ=0.3): Z ≈ -0.15")
println("  This is negative but O(1), suggesting liquid-like behavior")
println()

println("Thol EOS at (T=1.0, ρ=0.3): Z ≈ -23.28")
println("  This is wildly negative, suggesting either:")
println("  1) Bug in get_alphar_thol (α^r too negative)")
println("  2) Bug in derivative formula")
println("  3) Wrong coefficients")
println("  4) Formula itself is wrong")
println()

println("Since numeric derivative matches analytical derivative,")
println("the derivative formula is likely correct.")
println("The issue must be in get_alphar_thol or coefficients.")
