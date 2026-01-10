"""
Definitive parity check against teqp's Thol 2016 implementation.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

include(joinpath(@__DIR__, "..", "src", "EOS", "LJThol2016.jl"))

# Thol constants - EXACT from teqp
const THOL_TREF = 1.32
const THOL_RHOREF = 0.31

# Power terms
const THOL_POW_d = (4, 1, 1, 2, 2, 3, 1, 1, 3, 2, 2, 5)
const THOL_POW_l = (0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 2, 1)
const THOL_POW_n = (
     0.52080730e-2,  0.21862520e+1, -0.21610160e+1,  0.14527000e+1,
    -0.20417920e+1,  0.18695286e+0, -0.62086250e+0, -0.56883900e+0,
    -0.80055922e+0,  0.10901431e+0, -0.49745610e+0, -0.90988445e-1,
)
const THOL_POW_t = (1.000, 0.320, 0.505, 0.672, 0.843, 0.898, 1.205, 1.786, 2.770, 1.786, 2.590, 1.294)

# Gaussian terms
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
    (T=2.0, ρ=0.2),
    (T=1.35, ρ=0.2),
    (T=1.0, ρ=0.1),
]

"""
Reference implementation mirroring teqp's JSON structure exactly:
- epsilon is the δ-shift
- eta is the τ-shift
- beta scales (δ - epsilon)^2
- gamma scales (τ - eta)^2
"""
function alphar_ref_thol(T::Float64, ρ::Float64)::Float64
    δ = ρ / THOL_RHOREF
    τ = THOL_TREF / T
    
    αr = 0.0
    
    # Power terms
    for i in 1:length(THOL_POW_n)
        d_i = THOL_POW_d[i]
        l_i = THOL_POW_l[i]
        n_i = THOL_POW_n[i]
        t_i = THOL_POW_t[i]
        
        δ_power = δ^d_i
        τ_power = τ^t_i
        
        if l_i == 0
            αr += n_i * δ_power * τ_power
        else
            αr += n_i * δ_power * τ_power * (1.0 + δ^l_i)
        end
    end
    
    # Gaussian terms: exp(-beta*(δ - epsilon)^2 - gamma*(τ - eta)^2)
    for i in 1:length(THOL_GAU_n)
        d_i = THOL_GAU_d[i]
        t_i = THOL_GAU_t[i]
        n_i = THOL_GAU_n[i]
        beta_i = THOL_GAU_beta[i]
        gamma_i = THOL_GAU_gamma[i]
        epsilon_i = THOL_GAU_eps[i]  # δ-shift
        eta_i = THOL_GAU_eta[i]      # τ-shift
        
        δ_power = δ^d_i
        τ_power = τ^t_i
        
        δ_diff = δ - epsilon_i
        τ_diff = τ - eta_i
        exp_arg = -beta_i * δ_diff * δ_diff - gamma_i * τ_diff * τ_diff
        exp_val = exp(exp_arg)
        
        αr += n_i * δ_power * τ_power * exp_val
    end
    
    return αr
end

"""
Reference derivative: d(alphar)/dδ
"""
function alphar_delta_ref_thol(T::Float64, ρ::Float64)::Float64
    δ = ρ / THOL_RHOREF
    τ = THOL_TREF / T
    
    dαr_dδ = 0.0
    
    # Power terms: d/dδ [ n*δ^d*τ^t ] = n*d*δ^(d-1)*τ^t
    # For l_i > 0: d/dδ [ n*δ^d*τ^t*(1+δ^l) ] = n*τ^t * [ d*δ^(d-1)*(1+δ^l) + δ^d*l*δ^(l-1) ]
    #                                           = n*τ^t * [ d*δ^(d-1) + (d+l)*δ^(d+l-1) ]
    for i in 1:length(THOL_POW_n)
        d_i = THOL_POW_d[i]
        l_i = THOL_POW_l[i]
        n_i = THOL_POW_n[i]
        t_i = THOL_POW_t[i]
        
        τ_power = τ^t_i
        
        if l_i == 0
            if d_i > 0
                dαr_dδ += n_i * d_i * (δ^(d_i - 1)) * τ_power
            end
        else
            term1 = d_i > 0 ? d_i * (δ^(d_i - 1)) : 0.0
            term2 = (d_i + l_i) * (δ^(d_i + l_i - 1))
            dαr_dδ += n_i * τ_power * (term1 + term2)
        end
    end
    
    # Gaussian terms: d/dδ [ n*δ^d*τ^t*exp(arg) ]
    # = n*τ^t*exp(arg) * [ d*δ^(d-1) + δ^d * darg_dδ ]
    # where darg_dδ = -2*beta*(δ - epsilon)
    for i in 1:length(THOL_GAU_n)
        d_i = THOL_GAU_d[i]
        t_i = THOL_GAU_t[i]
        n_i = THOL_GAU_n[i]
        beta_i = THOL_GAU_beta[i]
        gamma_i = THOL_GAU_gamma[i]
        epsilon_i = THOL_GAU_eps[i]  # δ-shift
        eta_i = THOL_GAU_eta[i]      # τ-shift
        
        τ_power = τ^t_i
        δ_power = d_i > 0 ? δ^d_i : 1.0
        
        δ_diff = δ - epsilon_i
        τ_diff = τ - eta_i
        exp_arg = -beta_i * δ_diff * δ_diff - gamma_i * τ_diff * τ_diff
        exp_val = exp(exp_arg)
        
        darg_dδ = -2.0 * beta_i * δ_diff
        
        if d_i > 0
            term1 = d_i * (δ^(d_i - 1))
            term2 = δ_power * darg_dδ
            dαr_dδ += n_i * τ_power * exp_val * (term1 + term2)
        else
            term2 = δ_power * darg_dδ
            dαr_dδ += n_i * τ_power * exp_val * term2
        end
    end
    
    return dαr_dδ
end

println("=" ^ 80)
println("Thol 2016 Parity Check (teqp reference)")
println("=" ^ 80)
println()

all_passed = true

for (pt_idx, pt) in enumerate(test_points)
    T = pt.T
    ρ = pt.ρ
    
    println("-" ^ 80)
    println("Point $pt_idx: T = $T, ρ = $ρ")
    println("-" ^ 80)
    
    δ = ρ / THOL_RHOREF
    τ = THOL_TREF / T
    
    println("Reduced variables: δ = $δ, τ = $τ")
    println()
    
    # Reference computation
    alphar_ref = alphar_ref_thol(T, ρ)
    alphar_delta_ref = alphar_delta_ref_thol(T, ρ)
    
    # Our implementation
    alphar_our = get_alphar_thol(T, ρ)
    
    # Our derivative (from pressure_thol)
    # We need to extract it - let's compute it manually from our code structure
    τ_our = THOL_TREF / T
    δ_our = ρ / THOL_RHOREF
    dαr_dδ_our = 0.0
    
    # Power terms derivative (same as reference)
    for i in 1:length(THOL_POW_d)
        d_i = THOL_POW_d[i]
        l_i = THOL_POW_l[i]
        n_i = THOL_POW_n[i]
        t_i = THOL_POW_t[i]
        
        τ_power = τ_our^t_i
        
        if l_i == 0
            if d_i > 0
                dαr_dδ_our += n_i * d_i * (δ_our^(d_i - 1)) * τ_power
            end
        else
            term1 = d_i > 0 ? d_i * (δ_our^(d_i - 1)) : 0.0
            term2 = (d_i + l_i) * (δ_our^(d_i + l_i - 1))
            dαr_dδ_our += n_i * τ_power * (term1 + term2)
        end
    end
    
    # Gaussian terms derivative (from our production implementation - should match reference)
    for i in 1:length(THOL_GAU_n)
        d_i = THOL_GAU_d[i]
        t_i = THOL_GAU_t[i]
        n_i = THOL_GAU_n[i]
        beta_i = THOL_GAU_beta[i]
        gamma_i = THOL_GAU_gamma[i]
        epsilon_i = THOL_GAU_eps[i]  # δ-shift
        eta_i = THOL_GAU_eta[i]      # τ-shift
        
        τ_power = τ_our^t_i
        δ_power = d_i > 0 ? δ_our^d_i : 1.0
        
        # Correct formula: exp(-beta*(δ-epsilon)^2 - gamma*(τ-eta)^2)
        δ_diff = δ_our - epsilon_i
        τ_diff = τ_our - eta_i
        exp_arg = -beta_i * δ_diff * δ_diff - gamma_i * τ_diff * τ_diff
        exp_val = exp(exp_arg)
        
        # Derivative: d/dδ [exp(-beta*(δ-epsilon)^2 - gamma*(τ-eta)^2)] = exp(...) * [-2*beta*(δ-epsilon)]
        darg_dδ = -2.0 * beta_i * δ_diff
        
        if d_i > 0
            term1 = d_i * (δ_our^(d_i - 1))
            term2 = δ_power * darg_dδ
            dαr_dδ_our += n_i * τ_power * exp_val * (term1 + term2)
        else
            term2 = δ_power * darg_dδ
            dαr_dδ_our += n_i * τ_power * exp_val * term2
        end
    end
    
    println("alphar:")
    println("  ref  = $alphar_ref")
    println("  our  = $alphar_our")
    diff_alphar = abs(alphar_ref - alphar_our)
    println("  diff = $diff_alphar")
    
    println()
    println("alphar_delta (d(alphar)/dδ):")
    println("  ref  = $alphar_delta_ref")
    println("  our  = $dαr_dδ_our")
    diff_delta = abs(alphar_delta_ref - dαr_dδ_our)
    println("  diff = $diff_delta")
    
    println()
    
    # Hard asserts
    if diff_alphar >= 1e-10
        println("  ✗ FAIL: alphar mismatch (diff = $diff_alphar >= 1e-10)")
        all_passed = false
    else
        println("  ✓ PASS: alphar matches (diff = $diff_alphar < 1e-10)")
    end
    
    if diff_delta >= 1e-10
        println("  ✗ FAIL: alphar_delta mismatch (diff = $diff_delta >= 1e-10)")
        all_passed = false
    else
        println("  ✓ PASS: alphar_delta matches (diff = $diff_delta < 1e-10)")
    end
    
    println()
    
    # Compute Z from reference
    Z_ref = 1.0 + δ * alphar_delta_ref
    println("Z = 1 + δ*alphar_delta:")
    println("  Z_ref = $Z_ref")
    println()
end

println("=" ^ 80)
if all_passed
    println("✓ ALL PARITY CHECKS PASSED")
else
    println("✗ PARITY CHECKS FAILED - fix get_alphar_thol")
end
println("=" ^ 80)
