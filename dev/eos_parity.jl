"""
Debug script to verify EOS implementations by checking Z computation.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

include(joinpath(@__DIR__, "..", "src", "EOS", "LJThol2016.jl"))
include(joinpath(@__DIR__, "..", "src", "EOS", "LJKolafaNezbeda1994.jl"))

# Test points
test_points = [
    (T=1.0, ρ=0.05),
    (T=1.0, ρ=0.3),
    (T=2.0, ρ=0.2),
]

println("=" ^ 80)
println("EOS Parity Check")
println("=" ^ 80)
println()

# Thol constants
const THOL_TREF = 1.32
const THOL_RHOREF = 0.31

for (point_idx, pt) in enumerate(test_points)
    T = pt.T
    ρ = pt.ρ
    
    println("-" ^ 80)
    println("Point $point_idx: T = $T, ρ = $ρ")
    println("-" ^ 80)
    
    # Compute reduced variables for Thol
    δ = ρ / THOL_RHOREF
    τ = THOL_TREF / T
    
    println("\nThol EOS:")
    println("  δ = ρ/ρ_ref = $δ")
    println("  τ = T_ref/T = $τ")
    
    # Current implementation Z from pressure
    P_thol_press = MolSim.EOS.pressure_thol(T, ρ)
    Z_thol_press = P_thol_press / (ρ * T)
    
    println("\n  From pressure_thol:")
    println("    P = $P_thol_press")
    println("    Z = P/(ρ*T) = $Z_thol_press")
    
    # Independent computation via numeric derivative
    h = 1e-6
    δ_plus = δ + h
    δ_minus = δ - h
    
    # Convert back to ρ for alphar call
    ρ_plus = δ_plus * THOL_RHOREF
    ρ_minus = δ_minus * THOL_RHOREF
    
    αr_plus = get_alphar_thol(T, ρ_plus)
    αr_minus = get_alphar_thol(T, ρ_minus)
    
    # Numeric derivative: ∂αr/∂δ
    # Note: αr is computed as function of ρ, so we need to convert
    # If αr = f(δ) where δ = ρ/ρ_ref, then ∂αr/∂ρ = (1/ρ_ref) * ∂αr/∂δ
    # But we want ∂αr/∂δ, so we use: ∂αr/∂δ = (αr(δ+h) - αr(δ-h))/(2h)
    αrδ_numeric = (αr_plus - αr_minus) / (2.0 * h)
    
    # Z = 1 + δ * (∂αr/∂δ)
    Z_thol_indep = 1.0 + δ * αrδ_numeric
    
    println("\n  Independent computation (via numeric derivative):")
    println("    αr(δ+h) = $αr_plus")
    println("    αr(δ-h) = $αr_minus")
    println("    ∂αr/∂δ (numeric) = $αrδ_numeric")
    println("    Z = 1 + δ*(∂αr/∂δ) = $Z_thol_indep")
    
    diff_thol = abs(Z_thol_indep - Z_thol_press)
    println("\n  Comparison:")
    println("    |Z_indep - Z_press| = $diff_thol")
    
    # Check if Z is sane (within reasonable range for LJ fluid)
    # For normal liquid states, Z should be roughly between -1 and 10
    Z_sane = (abs(Z_thol_indep) < 100.0) && (Z_thol_indep > -50.0)
    
    if Z_sane
        println("    ✓ Z_indep is within plausible range")
        if diff_thol > 1e-6
            println("    ✗ Z_press differs from Z_indep - bug likely in pressure_thol derivative")
        else
            println("    ✓ Z_press matches Z_indep")
        end
    else
        println("    ✗ Z_indep is outside plausible range")
        println("      αr = $(get_alphar_thol(T, ρ))")
        println("      This suggests get_alphar_thol or coefficients may be incorrect")
    end
    println()
    
    # Kolafa EOS
    println("Kolafa EOS:")
    
    # Current implementation Z from pressure
    P_kolafa_press = MolSim.EOS.pressure_kolafa(T, ρ)
    Z_kolafa_press = P_kolafa_press / (ρ * T)
    
    println("\n  From pressure_kolafa:")
    println("    P = $P_kolafa_press")
    println("    Z = P/(ρ*T) = $Z_kolafa_press")
    
    # Independent computation via numeric derivative w.r.t. ρ
    ρ_plus_k = ρ + h
    ρ_minus_k = ρ - h
    
    αr_plus_k = get_alphar_kolafa(T, ρ_plus_k)
    αr_minus_k = get_alphar_kolafa(T, ρ_minus_k)
    
    # Numeric derivative: ∂αr/∂ρ
    αrρ_numeric = (αr_plus_k - αr_minus_k) / (2.0 * h)
    
    # Z = 1 + ρ * (∂αr/∂ρ)
    Z_kolafa_indep = 1.0 + ρ * αrρ_numeric
    
    println("\n  Independent computation (via numeric derivative):")
    println("    αr(ρ+h) = $αr_plus_k")
    println("    αr(ρ-h) = $αr_minus_k")
    println("    ∂αr/∂ρ (numeric) = $αrρ_numeric")
    println("    Z = 1 + ρ*(∂αr/∂ρ) = $Z_kolafa_indep")
    
    diff_kolafa = abs(Z_kolafa_indep - Z_kolafa_press)
    println("\n  Comparison:")
    println("    |Z_indep - Z_press| = $diff_kolafa")
    
    # Check if Z is sane (within reasonable range for LJ fluid)
    Z_sane = (abs(Z_kolafa_indep) < 100.0) && (Z_kolafa_indep > -50.0)
    
    if Z_sane
        println("    ✓ Z_indep is within plausible range")
        if diff_kolafa > 1e-6
            println("    ✗ Z_press differs from Z_indep - bug likely in pressure_kolafa derivative")
        else
            println("    ✓ Z_press matches Z_indep")
        end
    else
        println("    ✗ Z_indep is outside plausible range")
        println("      αr = $(get_alphar_kolafa(T, ρ))")
        println("      This suggests get_alphar_kolafa or coefficients may be incorrect")
    end
    println()
end

println("=" ^ 80)
