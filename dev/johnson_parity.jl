"""
Debug script to verify Johnson 1993 EOS implementation matches teqp.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

# Access the Johnson module - it's included in MolSim.EOS
# We'll need to check the actual structure

# Test point
T = 1.0
ρ = 0.01

println("=" ^ 60)
println("Johnson 1993 EOS Debug")
println("=" ^ 60)
println("T = $T")
println("ρ = $ρ")
println()

# Step 1: Check F = exp(-gamma*ρ^2)
gamma = 3.0
F = exp(-gamma * ρ * ρ)
println("F = exp(-gamma*ρ^2) = $F")
println()

# Load the Johnson module directly to access internal functions
include(joinpath(@__DIR__, "..", "src", "EOS", "LJJohnson1993.jl"))

# Step 2: Check G_i recursion - our implementation
println("--- Our get_Gi implementation ---")
# Access via the included module - check if it's a module or just functions
G1_our = get_Gi(1, T, ρ)
G2_our = get_Gi(2, T, ρ)
println("G1 (our) = $G1_our")
println("G2 (our) = $G2_our")
println()

# Step 2: Literal teqp translation
println("--- Literal teqp translation ---")
# G1 = (1 - F)/(2*gamma)
G1_teqp = (1.0 - F) / (2.0 * gamma)
# G_i recursion: G_i = -(F*ρ^(2(i-1)) - 2*(i-1)*G_{i-1})/(2*gamma)
G2_teqp = -(F * ρ^(2*(2-1)) - 2.0 * (2-1) * G1_teqp) / (2.0 * gamma)
println("G1 (teqp) = $G1_teqp")
println("G2 (teqp) = $G2_teqp")
println()

println("--- Comparison ---")
diff_G1 = abs(G1_our - G1_teqp)
diff_G2 = abs(G2_our - G2_teqp)
println("|G1_our - G1_teqp| = $diff_G1")
println("|G2_our - G2_teqp| = $diff_G2")
if diff_G1 < 1e-12 && diff_G2 < 1e-12
    println("✓ G_i matches teqp")
else
    println("✗ G_i mismatch - need to fix get_Gi")
end
println()

# Step 3: Check get_ai and get_bi - what's actually used in alphar
println("--- Our get_ai/get_bi (as used in alphar) ---")
println("For i=1 in alphar loop:")
idx_a1 = 1 + 1  # X[2]
idx_b1 = 1 + 9  # X[10]
idx_c1 = 1 + 17 # X[18]
a1_alphar = get_ai(idx_a1, T)
b1_alphar = idx_b1 <= length(X) ? get_ai(idx_b1, T) * (1.0/T) : 0.0
c1_alphar = idx_c1 <= length(X) ? get_ai(idx_c1, T) * (1.0/T)^2 : 0.0
println("  idx_a=$idx_a1, a1 = X[$idx_a1] = $a1_alphar")
println("  idx_b=$idx_b1, b1 = X[$idx_b1]/T = $b1_alphar")
println("  idx_c=$idx_c1, c1 = X[$idx_c1]/T² = $c1_alphar")
println()

println("For i=2 in alphar loop:")
idx_a2 = 2 + 1  # X[3]
idx_b2 = 2 + 9  # X[11]
idx_c2 = 2 + 17 # X[19]
a2_alphar = get_ai(idx_a2, T)
b2_alphar = idx_b2 <= length(X) ? get_ai(idx_b2, T) * (1.0/T) : 0.0
c2_alphar = idx_c2 <= length(X) ? get_ai(idx_c2, T) * (1.0/T)^2 : 0.0
println("  idx_a=$idx_a2, a2 = X[$idx_a2] = $a2_alphar")
println("  idx_b=$idx_b2, b2 = X[$idx_b2]/T = $b2_alphar")
println("  idx_c=$idx_c2, c2 = X[$idx_c2]/T² = $c2_alphar")
println()

# Check X array directly
println("--- X array (first 10 elements) ---")
for i in 1:min(10, length(X))
    println("X[$i] = $(X[i])")
end
println()

# Step 4: Check alphar calculation - break down terms
println("--- Our alphar calculation (detailed) ---")
T_inv = 1.0 / T
T_inv2 = T_inv * T_inv

function compute_alphar_breakdown(T_val, ρ_val)
    T_inv_local = 1.0 / T_val
    T_inv2_local = T_inv_local * T_inv_local
    
    αr_poly_local = 0.0
    ρ_pow_local = ρ_val
    for i in 1:8
        idx_a = i + 1
        idx_b = i + 9
        idx_c = i + 17
        
        ai = get_ai(idx_a, T_val)
        bi = idx_b <= length(X) ? get_ai(idx_b, T_val) * T_inv_local : 0.0
        ci = idx_c <= length(X) ? get_ai(idx_c, T_val) * T_inv2_local : 0.0
        
        term = (ai + bi + ci) * ρ_pow_local / Float64(i)
        αr_poly_local += term
        println("  i=$i: a=$ai, b=$bi, c=$ci, term=$(term), cumulative=$(αr_poly_local)")
        ρ_pow_local *= ρ_val
    end
    println("  Polynomial part: $αr_poly_local")
    println()
    
    αr_gau_local = 0.0
    for i in 26:32
        if i <= length(X)
            G_idx = i - 25
            G_val = get_Gi(G_idx, T_val, ρ_val)
            term = get_ai(i, T_val) * G_val
            αr_gau_local += term
            println("  i=$i (G_$(G_idx)): X[$i]=$(get_ai(i, T_val)), G=$(G_val), term=$(term), cumulative=$(αr_gau_local)")
        end
    end
    println("  Gaussian part: $αr_gau_local")
    println()
    
    return αr_poly_local + αr_gau_local
end

alphar_our = compute_alphar_breakdown(T, ρ)
println("Total alphar (our) = $alphar_our")
println()

# Step 5: Check Z calculation and derivative
println("--- Our Z calculation ---")
# Check derivative manually
h = 1e-8
alphar_plus = get_alphar_johnson(T, ρ + h)
alphar_minus = get_alphar_johnson(T, ρ - h)
dalphar_drho = (alphar_plus - alphar_minus) / (2.0 * h)
println("alphar at ρ=$ρ = $alphar_our")
println("d(alphar)/dρ (numerical) = $dalphar_drho")
Z_from_derivative = 1.0 + ρ * dalphar_drho
println("Z = 1 + ρ * d(alphar)/dρ = $Z_from_derivative")
println()

P_our = pressure_johnson(T, ρ)
Z_our = P_our / (ρ * T)
println("P (our) = $P_our")
println("Z (our) = $Z_our")
println()

# Check if Z is sane
if abs(Z_our) < 100.0 && Z_our > 0.0
    println("✓ Z is sane (~O(1))")
else
    println("✗ Z is not sane (value = $Z_our)")
end
