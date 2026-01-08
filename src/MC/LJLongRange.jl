"""
Long-range (tail) corrections for truncated Lennard-Jones potential.

For LJ 12-6 potential u(r) = 4ε[(σ/r)^12 - (σ/r)^6] in reduced units (ε=σ=1),
assuming g(r)=1 for r>rc (mean-field approximation), the tail corrections are:

Energy per particle: U_tail/N = (8πρ/3) * [1/(3*rc^9) - 1/rc^3]
Pressure: P_tail = (16πρ²/3) * [2/(3*rc^9) - 1/rc^3]

Reference: Standard LJ tail correction derivation (see Allen & Tildesley, Frenkel & Smit).
"""

"""
    compute_lrc_energy_per_particle(ρ::Float64, rc::Float64)::Float64

Compute long-range correction for energy per particle.
U_tail/N = (8πρ/3) * [1/(3*rc^9) - 1/rc^3]
Allocation-free.
"""
@inline function compute_lrc_energy_per_particle(ρ::Float64, rc::Float64)::Float64
    rc3 = rc * rc * rc
    rc9 = rc3 * rc3 * rc3
    inv_rc3 = 1.0 / rc3
    inv_rc9 = 1.0 / rc9
    return (8.0 * π * ρ / 3.0) * (inv_rc9 / 3.0 - inv_rc3)
end

"""
    compute_lrc_pressure(ρ::Float64, rc::Float64)::Float64

Compute long-range correction for pressure.
P_tail = (16πρ²/3) * [2/(3*rc^9) - 1/rc^3]
Allocation-free.
"""
@inline function compute_lrc_pressure(ρ::Float64, rc::Float64)::Float64
    rc3 = rc * rc * rc
    rc9 = rc3 * rc3 * rc3
    inv_rc3 = 1.0 / rc3
    inv_rc9 = 1.0 / rc9
    ρ2 = ρ * ρ
    return (16.0 * π * ρ2 / 3.0) * (2.0 * inv_rc9 / 3.0 - inv_rc3)
end
