"""
Simple Molecular Dynamics (MD) with thermostat and barostat for Lennard-Jones systems.
Uses Velocity Verlet integrator, velocity rescaling thermostat, and Berendsen barostat.
All in reduced units (ε=σ=k_B=1, m=1).
"""

using Random
using StaticArrays

# Overlap cap constants (matches Observables.jl)
const OVERLAP_R_REDUCED = 0.75
const OVERLAP_U_REDUCED = 100

"""
    MDState

State of the MD simulation.
"""
mutable struct MDState
    N::Int
    L::Float64
    pos::Matrix{Float64}      # 3 x N, column = particle
    vel::Matrix{Float64}      # 3 x N, column = particle
    force::Matrix{Float64}    # 3 x N, column = particle
    types::Vector{Int}        # type[i] = type ID of particle i (1-based)
    rng::Xoshiro
end

"""
    init_md_state(pos, T, rng)

Initialize MD state from positions, assigning velocities from Maxwell-Boltzmann distribution at temperature T.
"""
function init_md_state(pos::Matrix{Float64}, T::Float64, rng::Xoshiro)
    N = size(pos, 2)
    L = 0.0  # Will be set by caller
    vel = zeros(Float64, 3, N)
    force = zeros(Float64, 3, N)
    types = ones(Int, N)  # Single-component default
    
    # Initialize velocities from Maxwell-Boltzmann distribution
    # v ~ sqrt(T) * N(0,1) in reduced units (m=1, k_B=1)
    σ_v = sqrt(T)
    for i in 1:N
        for d in 1:3
            vel[d, i] = σ_v * randn(rng)
        end
    end
    
    # Remove center-of-mass velocity
    vcm = zeros(Float64, 3)
    for i in 1:N
        for d in 1:3
            vcm[d] += vel[d, i]
        end
    end
    vcm ./= N
    for i in 1:N
        for d in 1:3
            vel[d, i] -= vcm[d]
        end
    end
    
    return MDState(N, L, pos, vel, force, types, rng)
end

"""
    lj_force(r2, dr, p)::SVector{3,Float64}

Compute Lennard-Jones force vector from squared distance r2 and displacement vector dr.
Force = -∇u(r) = -du/dr * (dr/r)
For LJ: u(r) = 4ε[(σ/r)^12 - (σ/r)^6] - u_rc (if shifted)
du/dr = 4ε * [12*(σ/r)^12/r - 6*(σ/r)^6/r] = 4ε/r * [12*(σ/r)^12 - 6*(σ/r)^6]
Returns zero force if r2 >= rc2 or if r < overlap cap.
"""
@inline function lj_force(r2::Float64, dr::SVector{3,Float64}, p::LJParams)::SVector{3,Float64}
    if r2 >= p.rc2 || r2 <= 0.0
        return SVector(0.0, 0.0, 0.0)
    end
    
    # Overlap cap: if r < r_cap, return zero force
    rcap2 = (OVERLAP_R_REDUCED * p.σ)^2
    if r2 < rcap2
        return SVector(0.0, 0.0, 0.0)
    end
    
    r = sqrt(r2)
    invr = 1.0 / r
    σ_over_r = p.σ * invr
    σ6_over_r6 = σ_over_r^6
    σ12_over_r12 = σ6_over_r6 * σ6_over_r6
    
    # du/dr = 4ε/r * [12*(σ/r)^12 - 6*(σ/r)^6]
    dudr = 4.0 * p.ϵ * invr * (12.0 * σ12_over_r12 - 6.0 * σ6_over_r6)
    
    # Force = -du/dr * (dr/r) = -dudr * dr * invr
    f = -dudr * invr
    return SVector(f * dr[1], f * dr[2], f * dr[3])
end

"""
    compute_forces!(st, p)

Compute all pair forces using minimum image convention (PBC).
"""
function compute_forces!(st::MDState, p::LJParams)
    N = st.N
    L = st.L
    L_half = 0.5 * L
    
    # Zero forces
    @inbounds for i in 1:N
        for d in 1:3
            st.force[d, i] = 0.0
        end
    end
    
    # Compute pair forces
    @inbounds for i in 1:N
        for j in (i+1):N
            # Minimum image convention
            dx = st.pos[1, i] - st.pos[1, j]
            dy = st.pos[2, i] - st.pos[2, j]
            dz = st.pos[3, i] - st.pos[3, j]
            
            # Apply PBC
            if dx > L_half
                dx -= L
            elseif dx < -L_half
                dx += L
            end
            if dy > L_half
                dy -= L
            elseif dy < -L_half
                dy += L
            end
            if dz > L_half
                dz -= L
            elseif dz < -L_half
                dz += L
            end
            
            r2 = dx*dx + dy*dy + dz*dz
            
            if r2 < p.rc2 && r2 > 0.0
                dr = SVector(dx, dy, dz)
                f = lj_force(r2, dr, p)
                
                # Newton's third law
                st.force[1, i] += f[1]
                st.force[2, i] += f[2]
                st.force[3, i] += f[3]
                st.force[1, j] -= f[1]
                st.force[2, j] -= f[2]
                st.force[3, j] -= f[3]
            end
        end
    end
end

"""
    kinetic_energy(st)::Float64

Compute total kinetic energy: K = (1/2) * sum(m_i * v_i^2)
In reduced units, m=1 for all particles.
"""
function kinetic_energy(st::MDState)::Float64
    K = 0.0
    @inbounds for i in 1:st.N
        v2 = st.vel[1, i]^2 + st.vel[2, i]^2 + st.vel[3, i]^2
        K += 0.5 * v2
    end
    return K
end

"""
    instantaneous_temperature(st)::Float64

Compute instantaneous temperature from kinetic energy.
T = 2*K / (3*N - 3)  (subtract 3 for COM motion removed)
"""
function instantaneous_temperature(st::MDState)::Float64
    K = kinetic_energy(st)
    dof = 3 * st.N - 3  # 3N - 3 (COM motion removed)
    return 2.0 * K / dof
end

"""
    virial(st, p)::Float64

Compute virial for pressure calculation: W = sum_i r_i · F_i
"""
function virial(st::MDState, p::LJParams)::Float64
    N = st.N
    L = st.L
    L_half = 0.5 * L
    W = 0.0
    
    @inbounds for i in 1:N
        for j in (i+1):N
            # Minimum image convention
            dx = st.pos[1, i] - st.pos[1, j]
            dy = st.pos[2, i] - st.pos[2, j]
            dz = st.pos[3, i] - st.pos[3, j]
            
            if dx > L_half
                dx -= L
            elseif dx < -L_half
                dx += L
            end
            if dy > L_half
                dy -= L
            elseif dy < -L_half
                dy += L
            end
            if dz > L_half
                dz -= L
            elseif dz < -L_half
                dz += L
            end
            
            r2 = dx*dx + dy*dy + dz*dz
            
            if r2 < p.rc2 && r2 > 0.0
                rcap2 = (OVERLAP_R_REDUCED * p.σ)^2
                if r2 >= rcap2
                    r = sqrt(r2)
                    invr = 1.0 / r
                    σ_over_r = p.σ * invr
                    σ6_over_r6 = σ_over_r^6
                    σ12_over_r12 = σ6_over_r6 * σ6_over_r6
                    dudr = 4.0 * p.ϵ * invr * (12.0 * σ12_over_r12 - 6.0 * σ6_over_r6)
                    f_mag = dudr
                    W += f_mag * r
                end
            end
        end
    end
    
    return W
end

"""
    _md_total_energy(st, p)::Float64

Compute total potential energy for MDState (compatible with total_energy interface).
"""
function _md_total_energy(st::MDState, p::LJParams)::Float64
    energy = 0.0
    N = st.N
    L = st.L
    L_half = 0.5 * L
    rc2 = p.rc2
    
    @inbounds for i in 1:N
        for j in (i+1):N
            # Minimum image convention
            dx = st.pos[1, i] - st.pos[1, j]
            dy = st.pos[2, i] - st.pos[2, j]
            dz = st.pos[3, i] - st.pos[3, j]
            
            if dx > L_half
                dx -= L
            elseif dx < -L_half
                dx += L
            end
            if dy > L_half
                dy -= L
            elseif dy < -L_half
                dy += L
            end
            if dz > L_half
                dz -= L
            elseif dz < -L_half
                dz += L
            end
            
            r2 = dx*dx + dy*dy + dz*dz
            
            if r2 < rc2 && r2 > 0.0
                rcap2 = (OVERLAP_R_REDUCED * p.σ)^2
                if r2 >= rcap2
                    # Use lj_pair_u_from_r2 logic
                    σ2 = p.σ * p.σ
                    invr2 = σ2 / r2
                    invr6 = invr2 * invr2 * invr2
                    u_unshifted = 4.0 * p.ϵ * (invr6 * invr6 - invr6)
                    
                    if p.lj_model == :shifted
                        energy += u_unshifted - p.u_rc
                    else
                        energy += u_unshifted
                    end
                else
                    # Overlap cap
                    energy += OVERLAP_U_REDUCED * p.ϵ
                end
            end
        end
    end
    
    return energy
end

"""
    pressure(st, p, T)::Float64

Compute pressure from virial equation: P = ρ*T + (1/(3*V)) * W
where W is the virial.
"""
function pressure(st::MDState, p::LJParams, T::Float64)::Float64
    N = st.N
    V = st.L^3
    ρ = N / V
    W = virial(st, p)
    P = ρ * T + W / (3.0 * V)
    return P
end

"""
    velocity_verlet_step!(st, p, dt)

Perform one Velocity Verlet integration step.
"""
function velocity_verlet_step!(st::MDState, p::LJParams, dt::Float64)
    N = st.N
    L = st.L
    
    # Update positions: r(t+dt) = r(t) + v(t)*dt + 0.5*F(t)*dt^2
    @inbounds for i in 1:N
        for d in 1:3
            st.pos[d, i] += st.vel[d, i] * dt + 0.5 * st.force[d, i] * dt * dt
            # Apply PBC
            if st.pos[d, i] >= L
                st.pos[d, i] -= L
            elseif st.pos[d, i] < 0.0
                st.pos[d, i] += L
            end
        end
    end
    
    # Store old forces
    force_old = copy(st.force)
    
    # Compute new forces
    compute_forces!(st, p)
    
    # Update velocities: v(t+dt) = v(t) + 0.5*(F(t) + F(t+dt))*dt
    @inbounds for i in 1:N
        for d in 1:3
            st.vel[d, i] += 0.5 * (force_old[d, i] + st.force[d, i]) * dt
        end
    end
end

"""
    velocity_rescale!(st, T_target)

Rescale velocities to match target temperature.
"""
function velocity_rescale!(st::MDState, T_target::Float64)
    T_inst = instantaneous_temperature(st)
    if T_inst > 0.0
        scale = sqrt(T_target / T_inst)
        @inbounds for i in 1:st.N
            for d in 1:3
                st.vel[d, i] *= scale
            end
        end
    end
end

"""
    berendsen_barostat!(st, p, T, P_target, dt, tau_p)

Apply Berendsen barostat to scale volume and positions.
Volume scaling: dV/dt = -κ * (P - P_target) / τ_p
where κ is the isothermal compressibility (approximated as 1/(ρ*T) for ideal gas).
"""
function berendsen_barostat!(st::MDState, p::LJParams, T::Float64, P_target::Float64, dt::Float64, tau_p::Float64)
    P_inst = pressure(st, p, T)
    μ = 1.0 - (dt / tau_p) * (P_inst - P_target) / (p.σ^3 * T)  # Compressibility approximation
    μ = max(0.95, min(1.05, μ))  # Limit scaling to ±5% per step
    
    # Scale box length and positions
    L_new = st.L * μ
    scale = μ
    
    @inbounds for i in 1:st.N
        for d in 1:3
            st.pos[d, i] *= scale
        end
    end
    
    st.L = L_new
end

"""
    run_md_nvt!(st, p; nsteps, dt, T, thermostat_every)

Run NVT MD simulation with velocity rescaling thermostat.
"""
function run_md_nvt!(st::MDState, p::LJParams; nsteps::Int=10000, dt::Float64=0.001,
                    T::Float64=1.0, thermostat_every::Int=10)
    # Initialize forces
    compute_forces!(st, p)
    
    # Initialize velocities at target temperature
    velocity_rescale!(st, T)
    
    # Accumulators
    U_sum = 0.0
    U_sq_sum = 0.0
    P_sum = 0.0
    P_sq_sum = 0.0
    count = 0
    
    for step in 1:nsteps
        # Velocity Verlet step
        velocity_verlet_step!(st, p, dt)
        
        # Thermostat every N steps
        if step % thermostat_every == 0
            velocity_rescale!(st, T)
        end
        
        # Sample observables
        if step % thermostat_every == 0
            U = _md_total_energy(st, p)
            P = pressure(st, p, T)
            
            U_sum += U
            U_sq_sum += U * U
            P_sum += P
            P_sq_sum += P * P
            count += 1
        end
    end
    
    # Compute averages
    U_avg = U_sum / count
    U_var = (U_sq_sum / count) - (U_avg * U_avg)
    U_std = sqrt(max(0.0, U_var))
    
    P_avg = P_sum / count
    P_var = (P_sq_sum / count) - (P_avg * P_avg)
    P_std = sqrt(max(0.0, P_var))
    
    ρ = st.N / (st.L^3)
    
    return (U_avg, U_std, P_avg, P_std, ρ)
end

"""
    run_md_npt!(st, p; nsteps, dt, T, P, thermostat_every, barostat_every, tau_p)

Run NPT MD simulation with velocity rescaling thermostat and Berendsen barostat.
"""
function run_md_npt!(st::MDState, p::LJParams; nsteps::Int=10000, dt::Float64=0.001,
                     T::Float64=1.0, P::Float64=1.0, thermostat_every::Int=10,
                     barostat_every::Int=10, tau_p::Float64=1.0)
    # Initialize forces
    compute_forces!(st, p)
    
    # Initialize velocities at target temperature
    velocity_rescale!(st, T)
    
    # Accumulators
    U_sum = 0.0
    U_sq_sum = 0.0
    P_sum = 0.0
    P_sq_sum = 0.0
    ρ_sum = 0.0
    ρ_sq_sum = 0.0
    count = 0
    
    for step in 1:nsteps
        # Velocity Verlet step
        velocity_verlet_step!(st, p, dt)
        
        # Thermostat every N steps
        if step % thermostat_every == 0
            velocity_rescale!(st, T)
        end
        
        # Barostat every M steps
        if step % barostat_every == 0
            berendsen_barostat!(st, p, T, P, dt, tau_p)
            # Recompute forces after volume change
            compute_forces!(st, p)
        end
        
        # Sample observables
        if step % thermostat_every == 0
            U = _md_total_energy(st, p)
            P_inst = pressure(st, p, T)
            ρ_inst = st.N / (st.L^3)
            
            U_sum += U
            U_sq_sum += U * U
            P_sum += P_inst
            P_sq_sum += P_inst * P_inst
            ρ_sum += ρ_inst
            ρ_sq_sum += ρ_inst * ρ_inst
            count += 1
        end
    end
    
    # Compute averages
    U_avg = U_sum / count
    U_var = (U_sq_sum / count) - (U_avg * U_avg)
    U_std = sqrt(max(0.0, U_var))
    
    P_avg = P_sum / count
    P_var = (P_sq_sum / count) - (P_avg * P_avg)
    P_std = sqrt(max(0.0, P_var))
    
    ρ_avg = ρ_sum / count
    ρ_var = (ρ_sq_sum / count) - (ρ_avg * ρ_avg)
    ρ_std = sqrt(max(0.0, ρ_var))
    
    return (U_avg, U_std, P_avg, P_std, ρ_avg, ρ_std)
end
