"""
Thol et al. EOS for Lennard-Jones (Allen/Tildesley Python port).

Based on Python code from "Computer Simulation of Liquids", 2nd ed. (2017)
by M.P. Allen and D.J. Tildesley.

References:
- M Thol, G Rutkai, A Koester, R Lustig, R Span, J Vrabec, 
  J Phys Chem Ref Data 45, 023101 (2016) - Full LJ
- M Thol, G Rutkai, R Span, J Vrabec, R Lustig, 
  Int J Thermophys 36, 25 (2015) - Cut-and-shifted LJ

The routines use fitting functions described in the above papers.
All quantities in reduced units (ε=σ=k_B=1).

SUMMARY:
- `a_res_full`: Full Lennard-Jones potential (NO cutoff, infinite range)
- `a_res_cutshift`: Cut-and-shifted LJ at r_cut = 2.5σ (shifted so u(r_cut) = 0)

Functions available:
- Full LJ: `pressure_thol_full`, `internal_energy_thol_full`, 
           `chemical_potential_thol_full`, `chemical_potential_residual_thol_full`
- Cut-shifted: `pressure_thol_cutshift`, `internal_energy_thol_cutshift`,
               `chemical_potential_thol_cutshift`, `chemical_potential_residual_thol_cutshift`
"""

"""
    power_term(tau, delta, n, t, d)

Power basis function: n * (tau^t) * (delta^d)
Returns 3x3 matrix of scaled derivatives:
- a[i,j] where i is tau-derivative order (multiplied by tau^i)
- j is delta-derivative order (multiplied by delta^j)
"""
function power_term(tau::Float64, delta::Float64, n::Float64, t::Float64, d::Float64)
    base = n * (tau^t) * (delta^d)
    f = fill(base, 3, 3)
    # Tau derivatives (Python: f[1,:] = f[1,:] * t, f[2,:] = f[2,:] * t*(t-1))
    # Julia: row 2 = first tau derivative, row 3 = second tau derivative
    f[2, :] .= f[2, :] .* t
    f[3, :] .= f[3, :] .* t .* (t - 1.0)
    # Delta derivatives (Python: f[:,1] = f[:,1] * d, f[:,2] = f[:,2] * d*(d-1))
    # Julia: col 2 = first delta derivative, col 3 = second delta derivative
    f[:, 2] .= f[:, 2] .* d
    f[:, 3] .= f[:, 3] .* d .* (d - 1.0)
    return f
end

"""
    expon_term(tau, delta, n, t, d, l)

Exponential basis function: n * (tau^t) * (delta^d) * exp(-delta^l)
Returns 3x3 matrix of scaled derivatives.
"""
function expon_term(tau::Float64, delta::Float64, n::Float64, t::Float64, d::Float64, l::Float64)
    exp_factor = exp(-delta^l)
    f = fill(n * (tau^t) * (delta^d) * exp_factor, 3, 3)
    f[2, :] .= f[2, :] .* t
    f[3, :] .= f[3, :] .* t .* (t - 1.0)
    d_minus_l_delta = d - l * delta^l
    f[:, 2] .= f[:, 2] .* d_minus_l_delta
    f[:, 3] .= f[:, 3] .* ((d_minus_l_delta * (d - 1.0 - l * delta^l)) - (l^2 * delta^l))
    return f
end

"""
    gauss_term(tau, delta, n, t, d, beta, gamma, eta, epsilon)

Gaussian basis function: n * (tau^t) * exp(-beta*(tau-gamma)^2) * (delta^d) * exp(-eta*(delta-epsilon)^2)
Returns 3x3 matrix of scaled derivatives.
"""
function gauss_term(tau::Float64, delta::Float64, n::Float64, t::Float64, d::Float64,
                    beta::Float64, gamma::Float64, eta::Float64, epsilon::Float64)
    exp_tau = exp(-beta * (tau - gamma)^2)
    exp_delta = exp(-eta * (delta - epsilon)^2)
    f = fill(n * (tau^t) * exp_tau * (delta^d) * exp_delta, 3, 3)
    
    tau_deriv = t - 2.0 * beta * tau * (tau - gamma)
    f[2, :] .= f[2, :] .* tau_deriv
    f[3, :] .= f[3, :] .* (tau_deriv^2 - t - 2 * beta * tau^2)
    
    delta_deriv = d - 2.0 * eta * delta * (delta - epsilon)
    f[:, 2] .= f[:, 2] .* delta_deriv
    f[:, 3] .= f[:, 3] .* (delta_deriv^2 - d - 2 * eta * delta^2)
    
    return f
end

"""
    a_res_full(temp, rho)

Reduced residual free energy and scaled derivatives for the full Lennard-Jones potential.
Returns 3x3 matrix a[i,j] where:
- a[0,0] = residual Helmholtz free energy per particle (a_res)
- a[0,1] = delta-derivative (scaled) - used for pressure
- a[1,0] = tau-derivative (scaled) - used for energy
- etc.

Reference: Thol et al. (2016), Table 2
"""
function a_res_full(temp::Float64, rho::Float64)
    temp_crit = 1.32
    rho_crit = 0.31
    
    tau = temp_crit / temp
    delta = rho / rho_crit
    
    a = zeros(3, 3)
    
    # Power terms (Table 2, Thol 2016)
    cp_n = [0.005208073, 2.186252, -2.161016, 1.452700, -2.041792, 0.18695286]
    cp_t = [1.000, 0.320, 0.505, 0.672, 0.843, 0.898]
    cp_d = [4.0, 1.0, 1.0, 2.0, 2.0, 3.0]
    
    for i in 1:length(cp_n)
        a .+= power_term(tau, delta, cp_n[i], cp_t[i], cp_d[i])
    end
    
    # Exponential terms
    ce_n = [-0.090988445, -0.49745610, 0.10901431, -0.80055922, -0.56883900, -0.62086250]
    ce_t = [1.294, 2.590, 1.786, 2.770, 1.786, 1.205]
    ce_d = [5.0, 2.0, 2.0, 3.0, 1.0, 1.0]
    ce_l = [1.0, 2.0, 1.0, 2.0, 2.0, 1.0]
    
    for i in 1:length(ce_n)
        a .+= expon_term(tau, delta, ce_n[i], ce_t[i], ce_d[i], ce_l[i])
    end
    
    # Gaussian terms
    cg_n = [-1.4667177, 1.8914690, -0.13837010, -0.38696450, 0.12657020, 0.6057810,
            1.1791890, -0.47732679, -9.9218575, -0.57479320, 0.0037729230]
    cg_t = [2.830, 2.548, 4.650, 1.385, 1.460, 1.351, 0.660, 1.496, 1.830, 1.616, 4.970]
    cg_d = [1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0]
    cg_eta = [2.067, 1.522, 8.82, 1.722, 0.679, 1.883, 3.925, 2.461, 28.2, 0.753, 0.82]
    cg_beta = [0.625, 0.638, 3.91, 0.156, 0.157, 0.153, 1.16, 1.73, 383.0, 0.112, 0.119]
    cg_gamma = [0.71, 0.86, 1.94, 1.48, 1.49, 1.945, 3.02, 1.11, 1.17, 1.33, 0.24]
    cg_epsilon = [0.2053, 0.409, 0.6, 1.203, 1.829, 1.397, 1.39, 0.539, 0.934, 2.369, 2.43]
    
    for i in 1:length(cg_n)
        a .+= gauss_term(tau, delta, cg_n[i], cg_t[i], cg_d[i], 
                         cg_beta[i], cg_gamma[i], cg_eta[i], cg_epsilon[i])
    end
    
    return a
end

"""
    a_res_cutshift(temp, rho)

Reduced residual free energy and scaled derivatives for Lennard-Jones potential 
cut-and-shifted at r_cut = 2.5σ.

Reference: Thol et al. (2015), Table 1
"""
function a_res_cutshift(temp::Float64, rho::Float64)
    temp_crit = 1.086
    rho_crit = 0.319
    
    tau = temp_crit / temp
    delta = rho / rho_crit
    
    a = zeros(3, 3)
    
    # Power terms (Table 1, Thol 2015)
    cp_n = [0.015606084, 1.7917527, -1.9613228, 1.3045604, -1.8117673, 0.15483997]
    cp_t = [1.000, 0.304, 0.583, 0.662, 0.870, 0.870]
    cp_d = [4.0, 1.0, 1.0, 2.0, 2.0, 3.0]
    
    for i in 1:length(cp_n)
        a .+= power_term(tau, delta, cp_n[i], cp_t[i], cp_d[i])
    end
    
    # Exponential terms
    ce_n = [-0.094885204, -0.20092412, 0.11639644, -0.50607364, -0.58422807, -0.47510982]
    ce_t = [1.250, 3.000, 1.700, 2.400, 1.960, 1.286]
    ce_d = [5.0, 2.0, 2.0, 3.0, 1.0, 1.0]
    ce_l = [1.0, 2.0, 1.0, 2.0, 2.0, 1.0]
    
    for i in 1:length(ce_n)
        a .+= expon_term(tau, delta, ce_n[i], ce_t[i], ce_d[i], ce_l[i])
    end
    
    # Gaussian terms
    cg_n = [0.0094333106, 0.30444628, -0.0010820946, -0.099693391, 0.0091193522,
            0.12970543, 0.023036030, -0.082671073, -2.2497821]
    cg_t = [3.600, 2.080, 5.240, 0.960, 1.360, 1.655, 0.900, 0.860, 3.950]
    cg_d = [1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 2.0, 3.0]
    cg_eta = [4.70, 1.92, 2.70, 1.49, 0.65, 1.73, 3.70, 1.90, 13.2]
    cg_beta = [20.0, 0.77, 0.5, 0.8, 0.4, 0.43, 8.0, 3.3, 114.0]
    cg_gamma = [1.0, 0.5, 0.8, 1.5, 0.7, 1.6, 1.3, 0.6, 1.3]
    cg_epsilon = [0.55, 0.7, 2.0, 1.14, 1.2, 1.31, 1.14, 0.53, 0.96]
    
    for i in 1:length(cg_n)
        a .+= gauss_term(tau, delta, cg_n[i], cg_t[i], cg_d[i],
                         cg_beta[i], cg_gamma[i], cg_eta[i], cg_epsilon[i])
    end
    
    return a
end

"""
    pressure_thol_full(temp, rho)

Pressure for full Lennard-Jones potential.
P = ρ * T * (1 + a[0,1])
where a[0,1] is the scaled delta-derivative of residual Helmholtz energy.
"""
function pressure_thol_full(temp::Float64, rho::Float64)::Float64
    a = a_res_full(temp, rho)
    # Python a[0,1] -> Julia a[1,2] (row 1 = base row, col 2 = first delta derivative)
    return rho * temp * (1.0 + a[1, 2])
end

"""
    internal_energy_thol_full(temp, rho)

Internal energy per particle for full Lennard-Jones potential.
E/N = T * (1.5 + a[1,0])
where a[1,0] is the scaled tau-derivative of residual Helmholtz energy.
"""
function internal_energy_thol_full(temp::Float64, rho::Float64)::Float64
    a = a_res_full(temp, rho)
    # Python a[1,0] -> Julia a[2,1] (row 2 = first tau derivative, col 1 = base)
    return temp * (1.5 + a[2, 1])
end

"""
    chemical_potential_thol_full(temp, rho)

Full chemical potential for full Lennard-Jones potential.
μ = T * (ln(ρ) + a[0,0] + a[0,1])
Note: This is the full chemical potential, not residual.
For residual: μ_res = T * (a[0,0] + a[0,1])
"""
function chemical_potential_thol_full(temp::Float64, rho::Float64)::Float64
    a = a_res_full(temp, rho)
    # Python a[0,0] + a[0,1] -> Julia a[1,1] + a[1,2]
    return temp * (log(rho) + a[1, 1] + a[1, 2])
end

"""
    chemical_potential_residual_thol_full(temp, rho)

Residual chemical potential for full Lennard-Jones potential.
μ_res = T * (a[0,0] + a[0,1])
"""
function chemical_potential_residual_thol_full(temp::Float64, rho::Float64)::Float64
    a = a_res_full(temp, rho)
    return temp * (a[1, 1] + a[1, 2])
end

"""
    pressure_thol_cutshift(temp, rho)

Pressure for cut-and-shifted Lennard-Jones potential (r_cut = 2.5σ).
"""
function pressure_thol_cutshift(temp::Float64, rho::Float64)::Float64
    a = a_res_cutshift(temp, rho)
    return rho * temp * (1.0 + a[1, 2])
end

"""
    internal_energy_thol_cutshift(temp, rho)

Internal energy per particle for cut-and-shifted Lennard-Jones potential (r_cut = 2.5σ).
"""
function internal_energy_thol_cutshift(temp::Float64, rho::Float64)::Float64
    a = a_res_cutshift(temp, rho)
    return temp * (1.5 + a[2, 1])
end

"""
    chemical_potential_thol_cutshift(temp, rho)

Full chemical potential for cut-and-shifted Lennard-Jones potential (r_cut = 2.5σ).
μ = T * (ln(ρ) + a[0,0] + a[0,1])
"""
function chemical_potential_thol_cutshift(temp::Float64, rho::Float64)::Float64
    a = a_res_cutshift(temp, rho)
    return temp * (log(rho) + a[1, 1] + a[1, 2])
end

"""
    chemical_potential_residual_thol_cutshift(temp, rho)

Residual chemical potential for cut-and-shifted Lennard-Jones potential (r_cut = 2.5σ).
μ_res = T * (a[0,0] + a[0,1])
"""
function chemical_potential_residual_thol_cutshift(temp::Float64, rho::Float64)::Float64
    a = a_res_cutshift(temp, rho)
    return temp * (a[1, 1] + a[1, 2])
end
