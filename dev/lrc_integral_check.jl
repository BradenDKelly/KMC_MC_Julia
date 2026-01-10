"""
Diagnostic script to verify closed-form pressure tail correction equals numerical integral.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Parameters
ρ = 0.2
rc = 2.5
rmax = 50.0  # Upper limit for numerical integral

println("=" ^ 80)
println("LRC Integral Verification")
println("=" ^ 80)
println()
println("Parameters:")
println("  ρ = $ρ")
println("  rc = $rc")
println("  rmax = $rmax (upper limit for numerical integral)")
println()

# 1) Closed-form pressure tail correction
# P_tail = (16πρ²/3) * [2/(3*rc^9) - 1/rc^3]
rc3 = rc * rc * rc
rc9 = rc3 * rc3 * rc3
inv_rc3 = 1.0 / rc3
inv_rc9 = 1.0 / rc9
ρ2 = ρ * ρ

p_tail_closed = (16.0 * π * ρ2 / 3.0) * (2.0 * inv_rc9 / 3.0 - inv_rc3)

println("1) Closed-form pressure tail correction:")
println("   P_tail = (16πρ²/3) * [2/(3*rc^9) - 1/rc^3]")
println("   p_tail_closed = $p_tail_closed")
println()

# 2) Numerical integral
# The tail correction comes from the virial contribution:
# P_tail = -(2πρ²/3) * ∫_{rc}^{∞} r³ * dU/dr dr
# (negative sign because virial is r·f = r * (-dU/dr) = -r * dU/dr)
# 
# For LJ: U(r) = 4*(r^-12 - r^-6) in reduced units (σ=ε=1)
# dU/dr = 4*(-12*r^-13 + 6*r^-7) = -48*r^-13 + 24*r^-7
# 
# So: P_tail = -(2πρ²/3) * ∫_{rc}^{∞} r³ * (-48*r^-13 + 24*r^-7) dr
#            = -(2πρ²/3) * ∫_{rc}^{∞} (-48*r^-10 + 24*r^-4) dr
#            = (2πρ²/3) * ∫_{rc}^{∞} (48*r^-10 - 24*r^-4) dr

function integrand(r::Float64)::Float64
    # For virial tail correction: r·f = -r * dU/dr
    # dU/dr = -48*r^-13 + 24*r^-7
    # So: r·f = -r * (-48*r^-13 + 24*r^-7) = 48*r^-12 - 24*r^-6
    # But we need r² * (r·f) = r² * (48*r^-12 - 24*r^-6) = 48*r^-10 - 24*r^-4
    # Actually wait - let me check the standard formula...
    # Standard: P_tail = (2πρ²/3) * ∫_{rc}^{∞} r² * (r·f) * g(r) dr
    # With r·f = -r * dU/dr = r * (48*r^-13 - 24*r^-7) = 48*r^-12 - 24*r^-6
    # So integrand = r² * (48*r^-12 - 24*r^-6) = 48*r^-10 - 24*r^-4
    if r <= 0.0
        return 0.0
    end
    r4 = r * r * r * r
    r10 = r4 * r4 * r * r
    return 48.0 / r10 - 24.0 / r4
end

# Numerical integration using Simpson's rule
function integrate_simpson(f, a, b, n::Int)::Float64
    h = (b - a) / n
    result = f(a) + f(b)
    
    for i in 1:(n-1)
        x = a + i * h
        if i % 2 == 0
            result += 2.0 * f(x)
        else
            result += 4.0 * f(x)
        end
    end
    
    return result * h / 3.0
end

# Use adaptive integration: start with coarse grid, refine if needed
function integrate_adaptive(f, a, b, tol=1e-12)::Float64
    # Start with 10000 points for better initial accuracy
    n = 10000
    I1 = integrate_simpson(f, a, b, n)
    I2 = integrate_simpson(f, a, b, 2*n)
    
    error_est = abs(I2 - I1) / 15.0  # Error estimate from Simpson's rule
    
    if error_est < tol
        return I2
    else
        # Refine further
        I4 = integrate_simpson(f, a, b, 4*n)
        error_est2 = abs(I4 - I2) / 15.0
        if error_est2 < tol
            return I4
        else
            # Use even finer grid
            I8 = integrate_simpson(f, a, b, 8*n)
            error_est3 = abs(I8 - I4) / 15.0
            if error_est3 < tol
                return I8
            else
                # One more refinement
                I16 = integrate_simpson(f, a, b, 16*n)
                return I16
            end
        end
    end
end

println("2) Numerical integral:")
println("   P_tail = (2πρ²/3) * ∫_{rc}^{∞} r² * (r·f) dr")
println("   where r·f = -r*dU/dr = 48*r^-12 - 24*r^-6")
println("   and dU/dr = -48*r^-13 + 24*r^-7 for LJ U(r) = 4*(r^-12 - r^-6)")
println("   So integrand = r²*(r·f) = 48*r^-10 - 24*r^-4")
println("   Computing integral from rc=$rc to rmax=$rmax numerically,")
println("   then adding analytic contribution from rmax to infinity...")

# Compute the integral from rc to rmax with tight tolerance
integral_rc_to_rmax = integrate_adaptive(integrand, rc, rmax, 1e-12)

# Add the analytic contribution from rmax to infinity
# Antiderivative: ∫ (48*r^-10 - 24*r^-4) dr = -16/(3*r^9) + 8/r^3
# Evaluating from rmax to ∞:
# = [0 - 0] - [-16/(3*rmax^9) + 8/rmax^3]
# = 16/(3*rmax^9) - 8/rmax^3
rmax3 = rmax * rmax * rmax
rmax9 = rmax3 * rmax3 * rmax3
integral_rmax_to_inf = 16.0 / (3.0 * rmax9) - 8.0 / rmax3  # This is negative

integral_value = integral_rc_to_rmax + integral_rmax_to_inf

# Apply the prefactor
p_tail_num = (2.0 * π * ρ2 / 3.0) * integral_value

println("   Integral from rc to rmax = $integral_rc_to_rmax")
println("   Integral from rmax to ∞ (analytic) = $integral_rmax_to_inf")
println("   Total integral value = $integral_value")
println("   p_tail_num = (2πρ²/3) * integral = $p_tail_num")
println()

# Compute relative error
rel_error = abs(p_tail_closed - p_tail_num) / abs(p_tail_closed)

println("=" ^ 80)
println("Comparison")
println("=" ^ 80)
println()
println("  p_tail_closed = $p_tail_closed")
println("  p_tail_num    = $p_tail_num")
println("  difference    = $(p_tail_closed - p_tail_num)")
println("  relative error = $rel_error")
println()

# Check acceptance criterion
tolerance = 1e-8
if rel_error < tolerance
    println("  ✓ PASS: Relative error ($rel_error) < tolerance ($tolerance)")
else
    println("  ✗ FAIL: Relative error ($rel_error) >= tolerance ($tolerance)")
end

println()
println("=" ^ 80)
println()

# Also verify by computing the analytic integral directly
println("Verification by analytic integration:")
println("  ∫_{rc}^{∞} (48*r^-10 - 24*r^-4) dr")
println("  = [48*r^-9/(-9) - 24*r^-3/(-3)]_{rc}^{∞}")
println("  = [-48/(9*r^9) + 24/(3*r^3)]_{rc}^{∞}")
println("  = [-16/(3*r^9) + 8/r^3]_{rc}^{∞}")
println("  = [0 - 0] - [-16/(3*rc^9) + 8/rc^3]")
println("  = 16/(3*rc^9) - 8/rc^3")
println()
println("  P_tail = (2πρ²/3) * [16/(3*rc^9) - 8/rc^3]")
println("         = (16πρ²/3) * [2/(3*rc^9) - 1/rc^3]")
println()

# For the integrand (48*r^-10 - 24*r^-4):
integral_analytic = 16.0 * inv_rc9 / 3.0 - 8.0 * inv_rc3
p_tail_from_analytic = (2.0 * π * ρ2 / 3.0) * integral_analytic

println("  Analytic integral = $integral_analytic")
println("  P_tail from analytic = $p_tail_from_analytic")
println("  p_tail_closed = $p_tail_closed")
println("  |difference| = $(abs(p_tail_closed - p_tail_from_analytic))")
if abs(p_tail_closed - p_tail_from_analytic) < 1e-12
    println("  ✓ Closed-form matches analytic integration")
else
    println("  ✗ Closed-form does not match analytic integration")
end
println()
