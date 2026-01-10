"""
Thol et al. (2016) equation of state for full Lennard-Jones fluid.

Reference: Thol, M., Rutkai, G., Köster, A., Lustig, R., Span, R., Vrabec, J.
"Equation of state for the Lennard-Jones fluid", Journal of Physical Chemistry Reference Data,
45:2, 023101 (2016).

This implementation ports exact coefficients from teqp's Thol2016 implementation.
The pressure is computed from residual Helmholtz energy: Z = 1 + δ * (∂α^r/∂δ)
where α^r = a^r/(kT), δ = ρ/ρ_ref, and τ = T_ref/T.

Uses exact coefficients from teqp source:
- THOL_TREF, THOL_RHOREF (reducing parameters)
- ResidualHelmholtzPower terms (12 terms)
- ResidualHelmholtzGaussian terms (11 terms)

All quantities in reduced units (ε=σ=1, k_B=1).
"""

# Exact coefficients from teqp Thol2016 implementation
const THOL_TREF  = 1.32
const THOL_RHOREF = 0.31

# ResidualHelmholtzPower (12 terms)
const THOL_POW_d = (4, 1, 1, 2, 2, 3, 1, 1, 3, 2, 2, 5)
const THOL_POW_l = (0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 2, 1)
const THOL_POW_n = (
     0.52080730e-2,  0.21862520e+1, -0.21610160e+1,  0.14527000e+1,
    -0.20417920e+1,  0.18695286e+0, -0.62086250e+0, -0.56883900e+0,
    -0.80055922e+0,  0.10901431e+0, -0.49745610e+0, -0.90988445e-1,
)
const THOL_POW_t = (1.000, 0.320, 0.505, 0.672, 0.843, 0.898, 1.205, 1.786, 2.770, 1.786, 2.590, 1.294)

# ResidualHelmholtzGaussian (11 terms)
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

"""
    get_alphar_thol(T::Float64, ρ::Float64)::Float64

Compute residual Helmholtz energy per particle divided by kT.
From Thol 2016: α^r = a^r/(kT)

Uses exact coefficients from teqp:
- Power terms: α^r_pow = sum n_i * δ^d_i * τ^t_i * (1 + δ^l_i) for l_i > 0
- Gaussian terms: α^r_gau = sum n_i * δ^d_i * τ^t_i * exp(-beta_i*(δ-epsilon_i)^2 - gamma_i*(τ-eta_i)^2)
where δ = ρ/ρ_ref, τ = T_ref/T
"""
function get_alphar_thol(T::Float64, ρ::Float64)::Float64
    if T <= 0.0 || ρ < 0.0
        return NaN
    end
    
    # Reduced variables
    τ = THOL_TREF / T
    δ = ρ / THOL_RHOREF
    
    αr = 0.0
    
    # Power terms: sum n_i * δ^d_i * τ^t_i * (1 + δ^l_i) for l_i > 0
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
            # Terms with l_i > 0: multiply by (1 + δ^l_i)
            αr += n_i * δ_power * τ_power * (1.0 + δ^l_i)
        end
    end
    
    # Gaussian terms: sum n_i * δ^d_i * τ^t_i * exp(-beta_i*(δ-epsilon_i)^2 - gamma_i*(τ-eta_i)^2)
    # Correct parameter mapping per teqp:
    #   epsilon is the δ-shift (THOL_GAU_eps)
    #   eta is the τ-shift (THOL_GAU_eta)
    #   beta scales (δ - epsilon)^2 (THOL_GAU_beta)
    #   gamma scales (τ - eta)^2 (THOL_GAU_gamma)
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
        
        # Gaussian exponential: exp(-beta*(δ-epsilon)^2 - gamma*(τ-eta)^2)
        δ_diff = δ - epsilon_i
        τ_diff = τ - eta_i
        exp_arg = -beta_i * δ_diff * δ_diff - gamma_i * τ_diff * τ_diff
        exp_val = exp(exp_arg)
        
        αr += n_i * δ_power * τ_power * exp_val
    end
    
    return αr
end

"""
    pressure_thol(T::Float64, ρ::Float64)::Float64

Compute pressure for full Lennard-Jones fluid using Thol et al. (2016) EOS.
Returns pressure in reduced units.

Pressure is computed from: P/(ρT) = 1 + δ * (∂α^r/∂δ)
where α^r = a^r/(kT), δ = ρ/ρ_ref, τ = T_ref/T.

Analytical derivative computation for type stability and performance.
"""
function pressure_thol(T::Float64, ρ::Float64)::Float64
    if T <= 0.0 || ρ < 0.0
        return NaN
    end
    
    # Reduced variables
    τ = THOL_TREF / T
    δ = ρ / THOL_RHOREF
    
    # Derivative of residual Helmholtz energy with respect to δ
    dαr_dδ = 0.0
    
    # Power terms: ∂/∂δ [n_i * δ^d_i * τ^t_i * (1 + δ^l_i)]
    # For l_i = 0: d/dδ (n_i * δ^d_i * τ^t_i) = n_i * d_i * δ^(d_i-1) * τ^t_i
    # For l_i > 0: d/dδ (n_i * δ^d_i * τ^t_i * (1 + δ^l_i))
    #            = n_i * τ^t_i * [d_i * δ^(d_i-1) * (1 + δ^l_i) + δ^d_i * l_i * δ^(l_i-1)]
    #            = n_i * τ^t_i * [d_i * δ^(d_i-1) + (d_i + l_i) * δ^(d_i + l_i - 1)]
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
            # Term with l_i > 0
            term1 = d_i > 0 ? d_i * (δ^(d_i - 1)) : 0.0
            term2 = (d_i + l_i) * (δ^(d_i + l_i - 1))
            dαr_dδ += n_i * τ_power * (term1 + term2)
        end
    end
    
    # Gaussian terms: ∂/∂δ [n_i * δ^d_i * τ^t_i * exp(-beta_i*(δ-epsilon_i)^2 - gamma_i*(τ-eta_i)^2)]
    # = n_i * τ^t_i * exp(...) * [d_i * δ^(d_i-1) + δ^d_i * (-2*beta_i*(δ-epsilon_i))]
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
        
        # Gaussian exponential: exp(-beta*(δ-epsilon)^2 - gamma*(τ-eta)^2)
        δ_diff = δ - epsilon_i
        τ_diff = τ - eta_i
        exp_arg = -beta_i * δ_diff * δ_diff - gamma_i * τ_diff * τ_diff
        exp_val = exp(exp_arg)
        
        # Derivative: d/dδ [exp(-beta*(δ-epsilon)^2 - gamma*(τ-eta)^2)] = exp(...) * [-2*beta*(δ-epsilon)]
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
    
    # Compressibility factor: Z = 1 + δ * (∂α^r/∂δ)
    Z = 1.0 + δ * dαr_dδ
    
    return ρ * T * Z
end
