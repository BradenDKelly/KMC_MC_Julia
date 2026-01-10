"""
Kolafa-Nezbeda (1994) equation of state - exact Fortran PLJ implementation.

Reference: Kolafa, J., Nezbeda, I. "The Lennard-Jones fluid: an accurate analytical
and theoretically-based equation of state", Fluid Phase Equilibria, 100, 1-34 (1994).

This implementation ports the exact Fortran PLJ function structure from the
SklogWiki reference implementation.

All quantities in reduced units (ε=σ=1, k_B=1).
"""

# Constants and coefficients (same as existing Kolafa implementation)
# dhBH: tuples (i, C_i) for sum C_i * T^(i/2) plus c_ln_dhBH*log(T)
const C_DHBH_SKW = (
    (-2,  0.011117524),
    (-1, -0.076383859),
    ( 0,  1.080142248),
    ( 1,  0.000693129),
)
const C_LN_DHBH_SKW = -0.063920968

# DeltaB2hBH: tuples (i, C_i) for sum C_i * T^(i/2)
const C_DELTAB2_HBH_SKW = (
    (-7, -0.58544978),
    (-6,  0.43102052),
    (-5,  0.87361369),
    (-4, -4.13749995),
    (-3,  2.90616279),
    (-2, -7.02181962),
    ( 0,  0.02459877),
)

# Cij: tuples (i, j, Cij) for sum Cij*T^(i/2)*rho^j
const C_CIJ_SKW = (
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

const GAMMA_BH_SKW = 1.92907278

"""
    zHS(η::Float64)::Float64

Hard-sphere compressibility factor from Carnahan-Starling equation.
Z_HS = (1 + η + η² - η³) / (1 - η)³
"""
@inline function zHS(η::Float64)::Float64
    if η <= 0.0
        return 1.0
    end
    one_minus_eta = 1.0 - η
    one_minus_eta3 = one_minus_eta * one_minus_eta * one_minus_eta
    eta2 = η * η
    eta3 = eta2 * η
    
    return (1.0 + η + eta2 - eta3) / one_minus_eta3
end

"""
    dC(T::Float64)::Float64

Hard-sphere diameter function from Barker-Henderson theory.
d_hBH = sum_i C_i * T^(i/2) + C_LN_DHBH * log(T)
"""
@inline function dC(T::Float64)::Float64
    if T <= 0.0
        return 0.0
    end
    
    d = 0.0
    # Sum C_i * T^(i/2) terms
    for (i, c_i) in C_DHBH_SKW
        T_power = T^(i / 2.0)
        d += c_i * T_power
    end
    # Add log term
    d += C_LN_DHBH_SKW * log(T)
    
    return d
end

"""
    BC(T::Float64)::Float64

Second virial coefficient difference: ΔB2 = B2 - B2^HS
BC = sum_i C_i * T^(i/2)
"""
@inline function BC(T::Float64)::Float64
    if T <= 0.0
        return 0.0
    end
    
    delta_B2 = 0.0
    # Sum C_i * T^(i/2) terms
    for (i, c_i) in C_DELTAB2_HBH_SKW
        T_power = T^(i / 2.0)
        delta_B2 += c_i * T_power
    end
    
    return delta_B2
end

"""
    gammaBH(T::Float64)::Float64

Gamma parameter for Barker-Henderson theory (constant).
"""
@inline function gammaBH(T::Float64)::Float64
    return GAMMA_BH_SKW
end

"""
    pressure_kolafa_sklogwiki(T::Float64, rho::Float64)::Float64

Compute pressure for full Lennard-Jones fluid using Kolafa-Nezbeda (1994) EOS.
Exact Fortran PLJ implementation structure.

PLJ = (( zHS(eta)
         + BC(T)/exp(gammaBH(T)*rho^2) * rho*(1 - 2*gammaBH(T)*rho^2)
       )*T
       + sum
      ) * rho

where:
  eta = pi/6 * rho * dC(T)^3
  sum = Σ Cij * T^(i/2) * rho^j  (polynomial correction terms)
"""
function pressure_kolafa_sklogwiki(T::Float64, rho::Float64)::Float64
    if T <= 0.0 || rho < 0.0
        return NaN
    end
    
    # Compute hard-sphere diameter
    d = dC(T)
    d3 = d * d * d
    
    # Packing fraction
    eta = (π / 6.0) * rho * d3
    
    # Hard-sphere compressibility factor
    z_hs = zHS(eta)
    
    # BC term: BC(T)/exp(gammaBH(T)*rho^2) * rho*(1 - 2*gammaBH(T)*rho^2)
    gamma_bh = gammaBH(T)
    rho2 = rho * rho
    exp_arg = gamma_bh * rho2
    exp_val = exp(exp_arg)
    BC_val = BC(T)
    BC_term = (BC_val / exp_val) * rho * (1.0 - 2.0 * gamma_bh * rho2)
    
    # Sum: polynomial correction terms Σ Cij * T^(i/2) * rho^j
    sum_poly = 0.0
    for (i, j, cij) in C_CIJ_SKW
        T_power = T^(i / 2.0)
        rho_power = rho^j
        sum_poly += cij * T_power * rho_power
    end
    
    # PLJ formula: ((zHS + BC_term)*T + sum_poly) * rho
    PLJ = ((z_hs + BC_term) * T + sum_poly) * rho
    
    return PLJ
end
