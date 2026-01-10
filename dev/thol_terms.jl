"""
Instrument get_alphar_thol to debug term-by-term computation.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

include(joinpath(@__DIR__, "..", "src", "EOS", "LJThol2016.jl"))

# Thol constants
const THOL_TREF = 1.32
const THOL_RHOREF = 0.31

# Import constants from Thol module
const THOL_POW_d = (4, 1, 1, 2, 2, 3, 1, 1, 3, 2, 2, 5)
const THOL_POW_l = (0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 2, 1)
const THOL_POW_n = (
     0.52080730e-2,  0.21862520e+1, -0.21610160e+1,  0.14527000e+1,
    -0.20417920e+1,  0.18695286e+0, -0.62086250e+0, -0.56883900e+0,
    -0.80055922e+0,  0.10901431e+0, -0.49745610e+0, -0.90988445e-1,
)
const THOL_POW_t = (1.000, 0.320, 0.505, 0.672, 0.843, 0.898, 1.205, 1.786, 2.770, 1.786, 2.590, 1.294)

const THOL_GAU_beta = (0.625, 0.638, 3.91, 0.156, 0.157, 0.153, 1.16, 1.73, 383.0, 0.112, 0.119)
const THOL_GAU_d    = (1, 1, 2, 3, 3, 2, 1, 2, 3, 1, 1)
const THOL_GAU_eps  = (0.2053, 0.409, 0.6, 1.203, 1.829, 1.397, 1.39, 0.539, 0.934, 2.369, 2.43)
const THOL_GAU_eta  = (2.067, 1.522, 8.82, 1.722, 0.679, 1.883, 3.925, 2.461, 28.2, 0.753, 0.82)
const THOL_GAU_gamma= (0.71, 0.86, 1.94, 1.48, 1.49, 1.945, 3.02, 1.11, 1.17, 1.33, 0.24)
const THOL_GAU_n    = (
    -0.14667177e+1,  0.18914690e+1, -0.13837010e+0, -0.38696450e+0,
     0.12657020e+0,  0.60578100e+0,  0.11791890e+1, -0.47732679e+0,
    -0.99218575e+1, -0.57479320e+0,  0.37729230e-2,
)
const THOL_GAU_t    = (2.830, 2.548, 4.650, 1.385, 1.460, 1.351, 0.660, 1.496, 1.830, 1.616, 4.970)

# Test points
test_points = [
    (T=1.0, ρ=0.05),
    (T=1.0, ρ=0.1),
    (T=1.0, ρ=0.2),
    (T=1.0, ρ=0.3),  # This is where Z goes to -23
]

println("=" ^ 80)
println("Thol EOS Term-by-Term Analysis")
println("=" ^ 80)
println()

for (pt_idx, pt) in enumerate(test_points)
    T = pt.T
    ρ = pt.ρ
    
    println("-" ^ 80)
    println("Point $pt_idx: T = $T, ρ = $ρ")
    println("-" ^ 80)
    
    # Reduced variables
    δ = ρ / THOL_RHOREF
    τ = THOL_TREF / T
    
    println("\nReduced variables:")
    println("  δ = ρ/ρ_ref = $δ")
    println("  τ = T_ref/T = $τ")
    println()
    
    # Power terms: sum n_i * δ^d_i * τ^t_i * (1 + δ^l_i) for l_i > 0
    sum_power = 0.0
    power_terms = Float64[]
    max_power_term_mag = 0.0
    max_power_term_idx = 0
    
    for i in 1:length(THOL_POW_n)
        d_i = THOL_POW_d[i]
        l_i = THOL_POW_l[i]
        n_i = THOL_POW_n[i]
        t_i = THOL_POW_t[i]
        
        δ_power = δ^d_i
        τ_power = τ^t_i
        
        if l_i == 0
            term_val = n_i * δ_power * τ_power
        else
            # Terms with l_i > 0: multiply by (1 + δ^l_i)
            term_val = n_i * δ_power * τ_power * (1.0 + δ^l_i)
        end
        
        sum_power += term_val
        push!(power_terms, term_val)
        
        term_mag = abs(term_val)
        if term_mag > max_power_term_mag
            max_power_term_mag = term_mag
            max_power_term_idx = i
        end
    end
    
    println("Power terms (12 terms):")
    println("  sum_power = $sum_power")
    println("  max(|term|) = $max_power_term_mag (term index $max_power_term_idx)")
    println()
    
    # Gaussian terms
    sum_gauss = 0.0
    gauss_terms = Float64[]
    gauss_exp_args = Float64[]
    max_gauss_term_mag = 0.0
    max_gauss_term_idx = 0
    max_exp_arg = -Inf
    min_exp_arg = Inf
    
    for i in 1:length(THOL_GAU_n)
        d_i = THOL_GAU_d[i]
        t_i = THOL_GAU_t[i]
        n_i = THOL_GAU_n[i]
        beta_i = THOL_GAU_beta[i]
        eps_i = THOL_GAU_eps[i]
        eta_i = THOL_GAU_eta[i]
        gamma_i = THOL_GAU_gamma[i]
        
        δ_power = δ^d_i
        τ_power = τ^t_i
        
        # Gaussian exponential argument
        δ_diff = δ - eps_i
        τ_diff = τ - gamma_i
        exp_arg = -eta_i * δ_diff * δ_diff - beta_i * τ_diff * τ_diff
        
        push!(gauss_exp_args, exp_arg)
        
        if exp_arg > max_exp_arg
            max_exp_arg = exp_arg
        end
        if exp_arg < min_exp_arg
            min_exp_arg = exp_arg
        end
        
        # HARD ASSERT: exp_arg must be <= 0
        if exp_arg > 0.0
            println("  ✗ ERROR: Gaussian term $i has positive exp_arg = $exp_arg")
            println("    Parameters: n=$n_i, d=$d_i, t=$t_i, beta=$beta_i, eps=$eps_i, eta=$eta_i, gamma=$gamma_i")
            println("    δ_diff = $δ_diff, τ_diff = $τ_diff")
            println("    -eta*(δ-eps)^2 = $(-eta_i * δ_diff * δ_diff)")
            println("    -beta*(τ-gamma)^2 = $(-beta_i * τ_diff * τ_diff)")
            error("Gaussian exp_arg must be <= 0")
        end
        
        exp_val = exp(exp_arg)
        term_val = n_i * δ_power * τ_power * exp_val
        
        sum_gauss += term_val
        push!(gauss_terms, term_val)
        
        term_mag = abs(term_val)
        if term_mag > max_gauss_term_mag
            max_gauss_term_mag = term_mag
            max_gauss_term_idx = i
        end
    end
    
    println("Gaussian terms (11 terms):")
    println("  sum_gauss = $sum_gauss")
    println("  max(|term|) = $max_gauss_term_mag (term index $max_gauss_term_idx)")
    println("  exp_arg range: min = $min_exp_arg, max = $max_exp_arg")
    
    # HARD ASSERT: max_exp_arg should be <= 0
    if max_exp_arg > 0.0
        error("max_exp_arg = $max_exp_arg must be <= 0")
    end
    
    # Check for astronomically large terms at ρ <= 0.2
    if ρ <= 0.2 && max_gauss_term_mag > 1000.0
        println("  ✗ WARNING: Very large gaussian term at index $max_gauss_term_idx")
        i = max_gauss_term_idx
        println("    Parameters: n=$(THOL_GAU_n[i]), d=$(THOL_GAU_d[i]), t=$(THOL_GAU_t[i])")
        println("    beta=$(THOL_GAU_beta[i]), eps=$(THOL_GAU_eps[i]), eta=$(THOL_GAU_eta[i]), gamma=$(THOL_GAU_gamma[i])")
        println("    δ_power = $(δ^THOL_GAU_d[i]), τ_power = $(τ^THOL_GAU_t[i])")
        println("    exp_arg = $(gauss_exp_args[i]), exp_val = $(exp(gauss_exp_args[i]))")
        println("    term_val = $(gauss_terms[i])")
    end
    
    # Print detailed breakdown of all gaussian terms for debugging
    if ρ >= 0.2
        println("  Detailed gaussian terms breakdown:")
        for i in 1:length(THOL_GAU_n)
            δ_diff = δ - THOL_GAU_eps[i]
            τ_diff = τ - THOL_GAU_gamma[i]
            exp_arg = -THOL_GAU_eta[i] * δ_diff * δ_diff - THOL_GAU_beta[i] * τ_diff * τ_diff
            term_val = gauss_terms[i]
            if abs(term_val) > 0.1 || abs(exp_arg) < 1.0
                println("    Term $i: n=$(THOL_GAU_n[i]), d=$(THOL_GAU_d[i]), t=$(THOL_GAU_t[i])")
                println("      eta=$(THOL_GAU_eta[i]), beta=$(THOL_GAU_beta[i]), eps=$(THOL_GAU_eps[i]), gamma=$(THOL_GAU_gamma[i])")
                println("      δ_diff=$(δ_diff), τ_diff=$(τ_diff), exp_arg=$(exp_arg)")
                println("      term_val=$(term_val)")
            end
        end
    end
    
    println()
    
    # Total alpha_r
    alphar = sum_power + sum_gauss
    
    println("Total:")
    println("  α^r = sum_power + sum_gauss = $alphar")
    
    # Compare with our implementation
    alphar_impl = get_alphar_thol(T, ρ)
    println("  α^r (from get_alphar_thol) = $alphar_impl")
    
    diff = abs(alphar - alphar_impl)
    if diff > 1e-10
        println("  ✗ ERROR: Mismatch between term-by-term and implementation!")
        error("Difference = $diff")
    else
        println("  ✓ Matches implementation")
    end
    
    println()
    
    # Compute Z for reference
    P = MolSim.EOS.pressure_thol(T, ρ)
    Z = P / (ρ * T)
    println("  Z = P/(ρ*T) = $Z")
    println()
end

println("=" ^ 80)
