"""
Observable computation utilities (energy, virial, pressure).
"""

"""
    lj_pair_u_from_r2(r2, p)::Float64

Compute LJ pair energy u(r) = 4ε[(σ/r)^12 - (σ/r)^6] from r2.
Given r2 = dx*dx + dy*dy + dz*dz, with σ and ϵ from params:
  invr2 = (σ*σ) / r2
  invr6 = invr2 * invr2 * invr2
  u = 4*ϵ*(invr6*invr6 - invr6)
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
    return 4.0 * p.ϵ * (invr6 * invr6 - invr6)
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
    
    return energy
end

"""
    lj_force_magnitude_times_r(r2::Float64, p::LJParams)::Float64

Compute r * f(r) for Lennard-Jones force, where f(r) = -dU/dr.
For LJ: U(r) = 4ε[(σ/r)^12 - (σ/r)^6]
Force: f(r) = -dU/dr = 4ε[12σ^12/r^13 - 6σ^6/r^7]
r * f(r) = 4ε[12σ^12/r^12 - 6σ^6/r^6] = r_ij · f_ij (virial contribution)
Returns 0.0 if r2 >= rc2.
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
    pressure(st, p, T::Float64)::Float64

Compute pressure from virial equation: P = ρ*T + (1/(3V))*W
where W = Σ r_ij · f_ij is the virial.
"""
function pressure(st, p, T::Float64)::Float64
    N = st.N
    L = st.L
    V = L * L * L
    ρ = N / V
    W = total_virial(st, p)
    return ρ * T + W / (3.0 * V)
end
