"""
Observable computation utilities (energy, virial, pressure).
"""

"""
    lj_pair_u_from_r2(r2, p)::Float64

Compute LJ pair energy from r2, using LJ model specified in p.
For :truncated: u(r) = 4ε[(σ/r)^12 - (σ/r)^6]
For :shifted: u_shift(r) = u(r) - u(rc)
Returns 0.0 if r2 >= rc2.
"""
@inline function lj_pair_u_from_r2(r2::Float64, p::LJParams)::Float64
    if r2 >= p.rc2 || r2 <= 0.0
        return 0.0
    end
    # Guard against very small distances (must not allocate)
    if r2 < 1e-14
        return Inf
    end
    σ2 = p.σ * p.σ
    invr2 = σ2 / r2
    invr6 = invr2 * invr2 * invr2
    u_unshifted = 4.0 * p.ϵ * (invr6 * invr6 - invr6)
    
    if p.lj_model == :shifted
        return u_unshifted - p.u_rc
    else  # :truncated (default)
        return u_unshifted
    end
end

"""
    total_energy(st, p)::Float64

Compute total energy of the system (O(N^2), can be slower/allocating).
Double-count-safe: only counts pairs i<j.
"""
function total_energy(st, p)::Float64
    energy = 0.0
    N = st.N
    L = st.L
    rc2 = p.rc2
    pos = st.pos
    
    @inbounds for i in 1:N
        for j in (i+1):N
            # Compute distance vector
            dr_x = pos[1, j] - pos[1, i]
            dr_y = pos[2, j] - pos[2, i]
            dr_z = pos[3, j] - pos[3, i]
            
            # Apply minimum image convention
            L_half = L / 2.0
            if dr_x > L_half
                dr_x = dr_x - L
            elseif dr_x < -L_half
                dr_x = dr_x + L
            end
            if dr_y > L_half
                dr_y = dr_y - L
            elseif dr_y < -L_half
                dr_y = dr_y + L
            end
            if dr_z > L_half
                dr_z = dr_z - L
            elseif dr_z < -L_half
                dr_z = dr_z + L
            end
            
            r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
            
            if r2 < rc2 && r2 > 0.0
                energy += lj_pair_u_from_r2(r2, p)
            end
        end
    end
    
    # Add long-range correction if enabled
    if p.use_lrc
        energy += st.N * p.lrc_u_per_particle
    end
    
    return energy
end

"""
    lj_force_magnitude_times_r(r2::Float64, p::LJParams)::Float64

Compute r * f(r) for Lennard-Jones force, where f(r) = -dU/dr.
For LJ: U(r) = 4ε[(σ/r)^12 - (σ/r)^6]
For shifted LJ: U_shift(r) = U(r) - U(rc), so f_shift(r) = -dU_shift/dr = -dU/dr = f(r)
Thus the force (and virial) is the same for both models.
Force: f(r) = -dU/dr = 4ε[12σ^12/r^13 - 6σ^6/r^7]
r * f(r) = 4ε[12σ^12/r^12 - 6σ^6/r^6] = r_ij · f_ij (virial contribution)
Returns 0.0 if r2 >= rc2.
Note: This function does not depend on lj_model since force is independent of constant shift.
"""
@inline function lj_force_magnitude_times_r(r2::Float64, p::LJParams)::Float64
    if r2 >= p.rc2 || r2 <= 0.0
        return 0.0
    end
    # Use invr2 for numerical stability
    invr2 = 1.0 / r2
    σ2 = p.σ * p.σ
    σ2_invr2 = σ2 * invr2
    σ6_invr6 = σ2_invr2 * σ2_invr2 * σ2_invr2
    σ12_invr12 = σ6_invr6 * σ6_invr6
    # f_over_r = 4ε[12σ^12/r^12 - 6σ^6/r^6] / r
    # But we want r * f(r), so multiply by r^2 to get r^2 * f_over_r = r * f(r)
    # Actually: r * f(r) = 4ε[12σ^12/r^12 - 6σ^6/r^6]
    # Since σ^12/r^12 = (σ^2/r^2)^6 = (σ^2*invr2)^6 = σ12_invr12
    # Note: Force is the same for shifted and truncated LJ (shift is constant)
    return 4.0 * p.ϵ * (12.0 * σ12_invr12 - 6.0 * σ6_invr6)
end

"""
    total_virial(st, p)::Float64

Compute total virial W = Σ r_ij · f_ij for pairs i<j.
Uses minimum image convention.
Double-count-safe: only counts pairs i<j.
"""
function total_virial(st, p)::Float64
    virial = 0.0
    N = st.N
    L = st.L
    rc2 = p.rc2
    pos = st.pos
    
    @inbounds for i in 1:N
        for j in (i+1):N
            # Compute distance vector
            dr_x = pos[1, j] - pos[1, i]
            dr_y = pos[2, j] - pos[2, i]
            dr_z = pos[3, j] - pos[3, i]
            
            # Apply minimum image convention
            L_half = L / 2.0
            if dr_x > L_half
                dr_x = dr_x - L
            elseif dr_x < -L_half
                dr_x = dr_x + L
            end
            if dr_y > L_half
                dr_y = dr_y - L
            elseif dr_y < -L_half
                dr_y = dr_y + L
            end
            if dr_z > L_half
                dr_z = dr_z - L
            elseif dr_z < -L_half
                dr_z = dr_z + L
            end
            
            r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
            
            if r2 < rc2 && r2 > 0.0
                virial += lj_force_magnitude_times_r(r2, p)
            end
        end
    end
    
    return virial
end

"""
    compute_g_rc(st, p; bin_width::Float64=0.02)::Float64

Estimate radial distribution function g(rc) by counting pairs near rc.
Reporting-only helper for impulsive correction.
"""
function compute_g_rc(st, p; bin_width::Float64=0.02)::Float64
    N = st.N
    L = st.L
    rc = p.rc
    rc2 = p.rc2
    
    # Bin around rc: [rc - bin_width, rc + bin_width]
    r_min = rc - bin_width
    r_max = rc + bin_width
    r_min2 = r_min * r_min
    r_max2 = r_max * r_max
    
    # Count pairs in this shell
    pair_count = 0
    L_half = L / 2.0
    pos = st.pos
    
    @inbounds for i in 1:N
        for j in (i+1):N
            dr_x = pos[1, j] - pos[1, i]
            dr_y = pos[2, j] - pos[2, i]
            dr_z = pos[3, j] - pos[3, i]
            
            # Minimum image
            if dr_x > L_half
                dr_x = dr_x - L
            elseif dr_x < -L_half
                dr_x = dr_x + L
            end
            if dr_y > L_half
                dr_y = dr_y - L
            elseif dr_y < -L_half
                dr_y = dr_y + L
            end
            if dr_z > L_half
                dr_z = dr_z - L
            elseif dr_z < -L_half
                dr_z = dr_z + L
            end
            
            r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
            
            if r2 >= r_min2 && r2 < r_max2 && r2 > 0.0
                pair_count += 1
            end
        end
    end
    
    # Volume of shell: (4π/3) * (r_max^3 - r_min^3)
    V_shell = (4.0 * π / 3.0) * (r_max^3 - r_min^3)
    
    # Ideal gas pair density at this distance
    ρ = N / (L * L * L)
    n_ideal = 0.5 * N * ρ * V_shell  # 0.5 to avoid double counting
    
    if n_ideal > 0.0
        return Float64(pair_count) / n_ideal
    else
        return 1.0  # Default to ideal gas if no pairs expected
    end
end

"""
    pressure(st, p, T::Float64)::Float64

Compute pressure from virial equation: P = ρ*T + (1/(3V))*W
where W = Σ r_ij · f_ij is the virial.

For both truncated and shifted LJ, if impulsive correction is enabled (reporting only),
adds ΔP_imp = -(2π/3) * ρ² * rc³ * u(rc) * g(rc).
Note: Even for shifted LJ, the force is discontinuous at rc, so impulsive correction
may be needed for thermodynamic consistency with FEP pressure.
"""
function pressure(st, p, T::Float64)::Float64
    N = st.N
    L = st.L
    V = L * L * L
    ρ = N / V
    W = total_virial(st, p)
    P_sampled = ρ * T + W / (3.0 * V)
    
    # Add long-range correction if enabled
    if p.use_lrc
        P_sampled += p.lrc_p
    end
    
    # Add impulsive correction if enabled (reporting only)
    # Note: This correction is reporting-only and does not affect the Hamiltonian
    # Applies to both truncated and shifted LJ because the force is discontinuous at rc
    # (even though shifted LJ has continuous potential, the force derivative is still discontinuous)
    if p.apply_impulsive_correction
        g_rc = compute_g_rc(st, p)
        rc = p.rc
        # For both truncated and shifted, u(rc) is the unshifted value at rc
        # Compute u_unshifted(rc) = u_rc (which is stored in params)
        u_rc_unshifted = p.u_rc
        ΔP_imp = -(2.0 * π / 3.0) * ρ * ρ * rc * rc * rc * u_rc_unshifted * g_rc
        P_sampled += ΔP_imp
    end
    
    return P_sampled
end

"""
    pressure_virial_corrected(st, p, T::Float64)::Float64

Compute virial pressure with impulsive correction explicitly labeled.
Useful for reporting when correction is applied.
Returns (P_virial, P_corrected) tuple.
"""
function pressure_virial_corrected(st, p, T::Float64)::Tuple{Float64, Float64}
    N = st.N
    L = st.L
    V = L * L * L
    ρ = N / V
    W = total_virial(st, p)
    P_virial = ρ * T + W / (3.0 * V)
    
    if p.use_lrc
        P_virial += p.lrc_p
    end
    
    if p.apply_impulsive_correction && p.lj_model == :truncated
        g_rc = compute_g_rc(st, p)
        rc = p.rc
        # For truncated LJ, u(rc) is the unshifted value at rc
        u_rc_unshifted = p.u_rc
        ΔP_imp = -(2.0 * π / 3.0) * ρ * ρ * rc * rc * rc * u_rc_unshifted * g_rc
        P_corrected = P_virial + ΔP_imp
        return (P_virial, P_corrected)
    else
        return (P_virial, P_virial)
    end
end
