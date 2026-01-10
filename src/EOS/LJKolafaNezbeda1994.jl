"""
Kolafa-Nezbeda (1994) equation of state for full Lennard-Jones fluid.

WARNING: This implementation does not match the SklogWiki PLJ oracle.
Do not use for validation. Use pressure_kolafa_sklogwiki from LJKolafaSklogwiki1994.jl instead.

Reference: Kolafa, J., Nezbeda, I. "The Lennard-Jones fluid: an accurate analytical
and theoretically-based equation of state", Fluid Phase Equilibria, 100, 1-34 (1994).

This implementation ports exact coefficients from teqp's KolafaNezbeda1994 implementation.
The pressure is computed from the residual Helmholtz energy derivative:
Z = 1 + ρ * (∂α^r/∂ρ) where α^r = a^r/(kT) is the reduced residual Helmholtz energy.

Uses exact coefficients from teqp source:
- C_DHBH, C_LN_DHBH for hard-sphere diameter
- C_DELTAB2_HBH for second virial coefficient difference
- C_CIJ for correction terms
- KOLAFA_GAMMA constant

All quantities in reduced units (ε=σ=1, k_B=1).
"""

# Exact coefficients from teqp KolafaNezbeda1994 implementation
# dhBH: tuples (i, C_i) for sum C_i * T^(i/2) plus c_ln_dhBH*log(T)
const C_DHBH = (
    (-2,  0.011117524),
    (-1, -0.076383859),
    ( 0,  1.080142248),
    ( 1,  0.000693129),
)
const C_LN_DHBH = -0.063920968

# DeltaB2hBH: tuples (i, C_i) for sum C_i * T^(i/2)
const C_DELTAB2_HBH = (
    (-7, -0.58544978),
    (-6,  0.43102052),
    (-5,  0.87361369),
    (-4, -4.13749995),
    (-3,  2.90616279),
    (-2, -7.02181962),
    ( 0,  0.02459877),
)

# Cij: tuples (i, j, Cij) for sum Cij*T^(i/2)*rho^j
const C_CIJ = (
    ( 0, 2,   2.01546797),
    ( 0, 3, -28.17881636),
    ( 0, 4,  28.28313847),
    ( 0, 5, -10.42402873),
    (-1, 2, -19.58371655),
    (-1, 3,  75.62340289),
    (-1, 4, -120.70586598),
    (-1, 5,  93.92740328),
    (-1, 6, -27.37737354),
    (-2, 2,  29.34470520),
    (-2, 3, -112.3535693),
    (-2, 4,  170.64908980),
    (-2, 5, -123.06669187),
    (-2, 6,  34.42288969),
    (-4, 2, -13.37031968),
    (-4, 3,  65.38059570),
    (-4, 4, -115.09233113),
    (-4, 5,  88.91973082),
    (-4, 6, -25.62099890),
)

const KOLAFA_GAMMA = 1.92907278

"""
    get_dhBH(T::Float64)::Float64

Compute hard-sphere diameter function from Barker-Henderson theory.
From Kolafa-Nezbeda 1994, this is used to compute the hard-sphere contribution.
Uses exact coefficients from teqp: d_hBH = sum_i C_i * T^(i/2) + C_LN_DHBH * log(T)
"""
@inline function get_dhBH(T::Float64)::Float64
    if T <= 0.0
        return 0.0
    end
    T_sqrt = sqrt(T)
    
    d = 0.0
    # Sum C_i * T^(i/2) terms
    for (i, c_i) in C_DHBH
        T_power = T^(i / 2.0)
        d += c_i * T_power
    end
    # Add log term
    d += C_LN_DHBH * log(T)
    
    return d
end

"""
    get_ln_dhBH(T::Float64)::Float64

Compute ln(d_hBH) for hard-sphere contribution.
"""
@inline function get_ln_dhBH(T::Float64)::Float64
    d = get_dhBH(T)
    return log(d)
end

"""
    get_DeltaB2hBH(T::Float64)::Float64

Compute ΔB2 = B2 - B2^HS where B2^HS is hard-sphere second virial coefficient.
From Kolafa-Nezbeda 1994.
Uses exact coefficients from teqp: ΔB2 = sum_i C_i * T^(i/2)
"""
@inline function get_DeltaB2hBH(T::Float64)::Float64
    if T <= 0.0
        return 0.0
    end
    
    delta_B2 = 0.0
    # Sum C_i * T^(i/2) terms
    for (i, c_i) in C_DELTAB2_HBH
        T_power = T^(i / 2.0)
        delta_B2 += c_i * T_power
    end
    
    return delta_B2
end

"""
    get_ahs(η::Float64)::Float64

Compute hard-sphere residual Helmholtz energy per particle / (kT).
From Carnahan-Starling equation.
"""
@inline function get_ahs(η::Float64)::Float64
    if η <= 0.0
        return 0.0
    end
    one_minus_eta = 1.0 - η
    eta2 = η * η
    eta3 = eta2 * η
    eta4 = eta3 * η
    
    # Carnahan-Starling equation
    a_hs = η * (4.0 - 3.0 * η) / (one_minus_eta * one_minus_eta)
    return a_hs
end

"""
    get_zhs(η::Float64)::Float64

Compute hard-sphere compressibility factor.
From Carnahan-Starling equation: Z_HS = (1 + η + η² - η³) / (1 - η)³
"""
@inline function get_zhs(η::Float64)::Float64
    if η <= 0.0
        return 1.0
    end
    one_minus_eta = 1.0 - η
    one_minus_eta3 = one_minus_eta * one_minus_eta * one_minus_eta
    eta2 = η * η
    eta3 = eta2 * η
    
    z_hs = (1.0 + η + eta2 - eta3) / one_minus_eta3
    return z_hs
end

"""
    get_alphar_kolafa(T::Float64, ρ::Float64)::Float64

Compute residual Helmholtz energy per particle divided by kT.
From Kolafa-Nezbeda 1994: α^r = a^r/(kT)
"""
function get_alphar_kolafa(T::Float64, ρ::Float64)::Float64
    if T <= 0.0 || ρ < 0.0
        return NaN
    end
    
    d = get_dhBH(T)
    η = (π / 6.0) * ρ * d * d * d  # Packing fraction
    
    # Hard-sphere contribution
    a_hs = get_ahs(η)
    
    # Attractive contribution
    Delta_B2 = get_DeltaB2hBH(T)
    a_att = -2.0 * π * ρ * Delta_B2
    
    # Correction terms from Kolafa-Nezbeda 1994
    # Using exact coefficient array C_CIJ from teqp
    # α^r_corr = sum Cij * T^(i/2) * ρ^j
    a_corr = 0.0
    
    for (i, j, cij) in C_CIJ
        T_power = T^(i / 2.0)
        ρ_power = ρ^j
        a_corr += cij * T_power * ρ_power
    end
    
    αr = a_hs + a_att + a_corr
    return αr
end

"""
    pressure_kolafa(T::Float64, ρ::Float64)::Float64

Compute pressure for full Lennard-Jones fluid using Kolafa-Nezbeda (1994) EOS.
Returns pressure in reduced units.

Pressure is computed from: Z = 1 + ρ * (∂α^r/∂ρ)
where α^r = a^r/(kT) is the reduced residual Helmholtz energy.

Analytical derivative computation for type stability and performance.
"""
function pressure_kolafa(T::Float64, ρ::Float64)::Float64
    if T <= 0.0 || ρ < 0.0
        return NaN
    end
    
    d = get_dhBH(T)
    d3 = d * d * d
    η = (π / 6.0) * ρ * d3  # Packing fraction
    
    # Derivative of hard-sphere contribution
    if η > 0.99  # Avoid numerical issues
        η = 0.99
    end
    one_minus_eta = 1.0 - η
    one_minus_eta2 = one_minus_eta * one_minus_eta
    one_minus_eta3 = one_minus_eta2 * one_minus_eta
    eta2 = η * η
    
    # Z_hs = d(ρ*α_hs)/dρ * (1/ρ) = (1 + η + η² - η³) / (1 - η)³
    Z_hs = get_zhs(η)
    
    # Derivative of attractive contribution
    Delta_B2 = get_DeltaB2hBH(T)
    Z_att = -2.0 * π * Delta_B2 * ρ
    
    # Derivative of correction terms
    # ∂/∂ρ (sum Cij * T^(i/2) * ρ^j) = sum j * Cij * T^(i/2) * ρ^(j-1)
    # Z contribution: ρ * ∂α^r_corr/∂ρ = sum j * Cij * T^(i/2) * ρ^j
    Z_corr = 0.0
    
    for (i, j, cij) in C_CIJ
        if j > 0  # Only terms with j > 0 contribute to derivative
            T_power = T^(i / 2.0)
            ρ_power = ρ^j
            Z_corr += j * cij * T_power * ρ_power
        end
    end
    
    # Compressibility factor: Z = 1 + ρ * (∂α^r/∂ρ)
    Z = 1.0 + (Z_hs - 1.0) + Z_att + Z_corr
    
    return ρ * T * Z
end
