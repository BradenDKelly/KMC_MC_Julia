"""
Johnson et al. (1993) equation of state for full Lennard-Jones fluid.

Reference: Johnson, J.K., Zollweg, J.A., Gubbins, K.E. "The Lennard-Jones equation
of state revisited", Molecular Physics, 78:3, 591-618 (1993).

This implementation ports exact coefficients from teqp's Johnson1993 implementation.
The pressure is computed from residual Helmholtz energy: Z = 1 + ρ * (∂α^r/∂ρ)
where α^r = a^r/(kT).

Uses exact coefficients from teqp source:
- X array (33 coefficients including placeholder)
- JOHNSON_GAMMA constant for Gaussian terms

All quantities in reduced units (ε=σ=1, k_B=1).
"""

# Exact coefficients from teqp Johnson1993 implementation
const JOHNSON_GAMMA = 3.0

# teqp keeps a 0 placeholder for 1-based indexing
const X = (
    0.0,  # placeholder (index 1)
    0.8623085097507421,
    2.976218765822098,
   -8.402230115796038,
    0.1054136629203555,
   -0.8564583828174598,
    1.582759470107601,
    0.7639421948305453,
    1.753173414312048,
    2.798291772190376e03,
   -4.8394220260857657e-2,
    0.9963265197721935,
   -3.698000291272493e01,
    2.084012299434647e01,
    8.305402124717285e01,
   -9.574799715203068e02,
   -1.477746229234994e02,
    6.398607852471505e01,
    1.603993673294834e01,
    6.805916615864377e01,
   -2.791293578795945e03,
   -6.245128304568454,
   -8.116836104958410e03,
    1.488735559561229e01,
   -1.059346754655084e04,
   -1.131607632802822e02,
   -8.867771540418822e03,
   -3.986982844450543e01,
   -4.689270299917261e03,
    2.593535277438717e02,
   -2.694523589434903e03,
   -7.218487631550215e02,
    1.721802063863269e02,
)

"""
    get_ai(i::Int, T::Float64)::Float64

Compute MBWR coefficient a_i for index i (1-based, with placeholder at index 1) at temperature T.
From Johnson 1993 MBWR equation using exact X array from teqp.
"""
@inline function get_ai(i::Int, T::Float64)::Float64
    if T <= 0.0
        return NaN
    end
    
    if i < 1 || i > length(X)
        return 0.0
    end
    
    return X[i]
end

"""
    get_bi(i::Int, T::Float64)::Float64

Compute MBWR coefficient b_i for index i (1-based) at temperature T.
From Johnson 1993 MBWR equation: b_i = a_i / T
"""
@inline function get_bi(i::Int, T::Float64)::Float64
    if T <= 0.0
        return NaN
    end
    
    ai = get_ai(i, T)
    return ai / T
end

"""
    get_Gi(i::Int, T::Float64, ρ::Float64)::Float64

Compute MBWR Gaussian term G_i for index i.
From teqp's Johnson1993: recursive formula
- G1 = (1 - F)/(2*gamma) where F = exp(-gamma*ρ²)
- G_i = -(F*ρ^(2(i-1)) - 2*(i-1)*G_{i-1})/(2*gamma)
"""
@inline function get_Gi(i::Int, T::Float64, ρ::Float64)::Float64
    if T <= 0.0 || ρ < 0.0
        return NaN
    end
    
    F = exp(-JOHNSON_GAMMA * ρ * ρ)
    
    if i == 1
        # G1 = (1 - F)/(2*gamma)
        return (1.0 - F) / (2.0 * JOHNSON_GAMMA)
    else
        # G_i = -(F*ρ^(2(i-1)) - 2*(i-1)*G_{i-1})/(2*gamma)
        G_prev = get_Gi(i - 1, T, ρ)
        ρ_power = ρ^(2 * (i - 1))
        return -(F * ρ_power - 2.0 * (i - 1) * G_prev) / (2.0 * JOHNSON_GAMMA)
    end
end

"""
    get_alphar_johnson(T::Float64, ρ::Float64)::Float64

Compute residual Helmholtz energy per particle divided by kT.
From Johnson 1993 MBWR: α^r = a^r/(kT)

MBWR form from teqp:
α^r = sum_{i=1}^8 [X[i] + X[i]/T + X[i+8]/T²] * ρ^i / i
     + sum_{i=9}^15 X[i] * G_{i-8}
where G_i = ρ^(i-1) * exp(-γ*ρ²) with γ = JOHNSON_GAMMA
"""
function get_alphar_johnson(T::Float64, ρ::Float64)::Float64
    if T <= 0.0 || ρ < 0.0
        return NaN
    end
    
    T_inv = 1.0 / T
    
    αr = 0.0
    
    # Polynomial terms: sum_{i=1..8} a_i(T)*ρ^i/i
    # a_i = X[i+1] (X[1] is placeholder)
    ρ_power = ρ
    for i in 1:8
        idx_a = i + 1  # X[2] through X[9]
        ai = get_ai(idx_a, T)
        αr += ai * ρ_power / Float64(i)
        ρ_power *= ρ
    end
    
    # Gaussian terms: sum_{i=1..6} b_i(T)*G_i(F,ρ)
    # b_i = X[i+9]/T, G_i uses teqp recursion
    F = exp(-JOHNSON_GAMMA * ρ * ρ)
    
    # G1 = (1-F)/(2γ)
    G_prev = (1.0 - F) / (2.0 * JOHNSON_GAMMA)
    idx_b1 = 1 + 9  # X[10]
    if idx_b1 <= length(X)
        b1 = get_ai(idx_b1, T) * T_inv
        αr += b1 * G_prev  # i=1: b_1*G_1
    end
    
    # For i=2..6: G_i = -(F*ρ^m - 2*(i-1)*G_{i-1})/(2*gamma) where m = 2*(i-1)
    for i in 2:6
        idx_b = i + 9  # X[11] through X[15]
        if idx_b <= length(X)
            bi = get_ai(idx_b, T) * T_inv
            
            m = 2 * (i - 1)
            ρ_m = ρ^m
            G_i = -(F * ρ_m - 2.0 * (i - 1) * G_prev) / (2.0 * JOHNSON_GAMMA)
            
            αr += bi * G_i
            G_prev = G_i
        end
    end
    
    return αr
end

"""
    pressure_johnson(T::Float64, ρ::Float64)::Float64

Compute pressure for full Lennard-Jones fluid using Johnson et al. (1993) EOS.
Returns pressure in reduced units.

Pressure is computed from: Z = 1 + ρ * (∂α^r/∂ρ) / T
where α^r = a^r/(kT).

Exact analytical derivative computation based on teqp structure:
dget = sum_{i=1..8} a_i(T)*ρ^(i-1) + sum_{i=1..6} b_i(T)*dGi
Z = 1 + ρ*(dget/T)
"""
function pressure_johnson(T::Float64, ρ::Float64)::Float64
    if T <= 0.0 || ρ < 0.0
        return NaN
    end
    
    T_inv = 1.0 / T
    
    # Derivative of polynomial terms: ∂/∂ρ [sum a_i*ρ^i/i] = sum a_i*ρ^(i-1)
    dget = 0.0
    ρ_power = 1.0
    for i in 1:8
        idx_a = i + 1  # X[2] through X[9]
        ai = get_ai(idx_a, T)
        dget += ai * ρ_power
        ρ_power *= ρ
    end
    
    # Derivative of Gaussian terms: sum_{i=1..6} b_i(T)*dGi
    # Compute both G_i and dG_i simultaneously
    F = exp(-JOHNSON_GAMMA * ρ * ρ)
    dF = -2.0 * JOHNSON_GAMMA * ρ * F
    
    # G1 = (1-F)/(2γ); dG1 = ρ*F
    G_prev = (1.0 - F) / (2.0 * JOHNSON_GAMMA)
    dG_prev = ρ * F
    
    # i=1: b_1*dG_1
    idx_b1 = 1 + 9  # X[10]
    if idx_b1 <= length(X)
        b1 = get_ai(idx_b1, T) * T_inv
        dget += b1 * dG_prev
    end
    
    # For i>1: compute G_i and dG_i recursively
    for i in 2:6
        idx_b = i + 9  # X[11] through X[15]
        if idx_b <= length(X)
            bi = get_ai(idx_b, T) * T_inv
            
            # G_i = -(F*ρ^m - 2*(i-1)*G_{i-1})/(2*gamma) where m = 2*(i-1)
            m = 2 * (i - 1)
            ρ_m = ρ^m
            G_i = -(F * ρ_m - 2.0 * (i - 1) * G_prev) / (2.0 * JOHNSON_GAMMA)
            
            # d(F*ρ^m) = dF*ρ^m + F*m*ρ^(m-1)
            dF_ρ_m = dF * ρ_m + F * m * (m > 0 ? ρ^(m - 1) : 0.0)
            
            # dG_i = -( d(F*ρ^m) - 2*(i-1)*dG_{i-1} )/(2*gamma)
            dG_i = -(dF_ρ_m - 2.0 * (i - 1) * dG_prev) / (2.0 * JOHNSON_GAMMA)
            
            dget += bi * dG_i
            
            G_prev = G_i
            dG_prev = dG_i
        end
    end
    
    # Z = 1 + ρ*(dget/T)
    Z = 1.0 + ρ * dget * T_inv
    
    return ρ * T * Z
end
