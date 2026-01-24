"""
NKEOS_Author: Authoritative NKEOS wrapper (ported from provided Python).

This is a thin wrapper around the reduced-unit NKEOS functions with the same
interface and unit conversions as the Python script. It exposes both reduced
and SI-unit convenience functions.
"""

const NKEOS_AUTHOR_KB = 1.3806488e-23
const NKEOS_AUTHOR_NA = 6.02214129e23
const NKEOS_AUTHOR_R = NKEOS_AUTHOR_KB * NKEOS_AUTHOR_NA
const NKEOS_AUTHOR_SIG = 0.3405e-9
const NKEOS_AUTHOR_EPS = 119.8

"""
    alj_res_reduced(T, ρ)

Residual Helmholtz free energy in reduced units.
"""
alj_res_reduced(T::Float64, ρ::Float64)::Float64 = nkeos_alj_res(T, ρ)

"""
    pressure_reduced(T, ρ)

Pressure in reduced units.
"""
pressure_reduced(T::Float64, ρ::Float64)::Float64 = pressure_nkeos(T, ρ)

"""
    internal_energy_reduced(T, ρ)

Internal energy per particle in reduced units.
"""
internal_energy_reduced(T::Float64, ρ::Float64)::Float64 = internal_energy_nkeos(T, ρ)

"""
    pressure_nkeos_author(T, ρ)

Compatibility wrapper for reduced units (same as pressure_reduced).
"""
pressure_nkeos_author(T::Float64, ρ::Float64)::Float64 = pressure_reduced(T, ρ)

"""
    internal_energy_nkeos_author(T, ρ)

Compatibility wrapper for reduced units (same as internal_energy_reduced).
"""
internal_energy_nkeos_author(T::Float64, ρ::Float64)::Float64 = internal_energy_reduced(T, ρ)

"""
    alj_res_author(T, ρ)

Compatibility wrapper for reduced units (same as alj_res_reduced).
"""
alj_res_author(T::Float64, ρ::Float64)::Float64 = alj_res_reduced(T, ρ)

"""
    chemical_potential_nkeos_author(T, ρ)

Residual chemical potential in reduced units (same as chemical_potential_nkeos).
"""
chemical_potential_nkeos_author(T::Float64, ρ::Float64)::Float64 = chemical_potential_nkeos(T, ρ)

"""
    getPress_SI(T, ρ)

Pressure in Pa. T [K], ρ [kg/m^3] in LJ reduced units with σ, ε as above.
"""
function getPress_SI(T::Float64, ρ::Float64)::Float64
    Tr = T / NKEOS_AUTHOR_EPS
    rho_r = ρ * NKEOS_AUTHOR_SIG^3
    return pressure_nkeos(Tr, rho_r) * NKEOS_AUTHOR_EPS * NKEOS_AUTHOR_KB / (NKEOS_AUTHOR_SIG^3)
end

"""
    getAres_SI(V, N, T)

Residual Helmholtz energy in J using SI inputs.
"""
function getAres_SI(V::Float64, N::Int, T::Float64)::Float64
    rho = N / V
    Tr = T / NKEOS_AUTHOR_EPS
    rho_r = rho * NKEOS_AUTHOR_SIG^3
    return N * alj_res_reduced(Tr, rho_r) * NKEOS_AUTHOR_EPS * NKEOS_AUTHOR_KB
end

"""
    getResChemPot_SI(V, N, T)

Residual chemical potential in J/mol using SI inputs.
"""
function getResChemPot_SI(V::Float64, N::Int, T::Float64)::Float64
    Tr = T / NKEOS_AUTHOR_EPS
    rho_r = (N / V) * NKEOS_AUTHOR_SIG^3
    return (alj_res_reduced(Tr, rho_r) + pressure_nkeos(Tr, rho_r) / rho_r - Tr) *
           NKEOS_AUTHOR_EPS * NKEOS_AUTHOR_R
end
