"""
Equilibrium Kinetic Monte Carlo (eKMC) for single-component Lennard-Jones.

Algorithm (Tan et al., CEJ 2020):
- u_i = sum_{j≠i} φ(r_ij)
- v_i = exp(β u_i)
- R = sum_i v_i
- Δt = ln(1/ξ) / R
- Select i with probability v_i / R
- Relocate i uniformly in the box (all moves accepted)
- Recompute u_i, v_i, R

Overlap handling (as specified):
- Compute r_cap such that U/ε = 100 in reduced units (σ=1).
- If r < r_cap, set pair energy to 100 (reduced), do not evaluate LJ.
"""

using Random
using StaticArrays
using Base.Threads

# Note: OVERLAP_U_REDUCED and OVERLAP_R_REDUCED are defined in Observables.jl
# (included before this file in MolSim.jl), and used by lj_pair_u_from_r2()
# No need to redefine them here.

mutable struct eKMCState
    N::Int
    L::Float64
    pos::Matrix{Float64}          # 3 x N
    types::Vector{Int}            # type IDs (single-component: all 1)
    rng::Xoshiro
    scratch_dr::MVector{3,Float64}
    pair_u::Matrix{Float64}       # symmetric pair energies u_ij
    phi::Vector{Float64}          # u_i
    m::Vector{Float64}            # v_i
    R::Float64                    # total rate
    thread_sums::Vector{Float64}  # per-thread scratch for reductions
    debug_check_every::Int        # diagnostic cadence (0 = off)
end

mutable struct ChemicalPotentialAccumulator
    t_total::Float64
    S::Float64                    # sum (R/N) * dt
    S_sq::Float64                 # sum ((R/N)^2) * dt for variance
    t_total_sq::Float64           # sum (dt^2) for effective sample size
    count::Int
end

ChemicalPotentialAccumulator() = ChemicalPotentialAccumulator(0.0, 0.0, 0.0, 0.0, 0)

function reset!(acc::ChemicalPotentialAccumulator)
    acc.t_total = 0.0
    acc.S = 0.0
    acc.S_sq = 0.0
    acc.t_total_sq = 0.0
    acc.count = 0
    return nothing
end

# NPT-specific chemical potential accumulator (Equation A.132)
# μ₁^(res) = kT ln [⟨(exp(βφ_N)) / V⟩ ⟨V⟩]
# where exp(βφ_N) ≈ R/N (average mobility)
mutable struct NPTChemicalPotentialAccumulator
    t_total::Float64
    S_over_V::Float64             # sum ((R/N) / V) * dt = ⟨(exp(βφ_N)) / V⟩
    S_over_V_sq::Float64          # sum (((R/N) / V)^2) * dt for variance
    V_avg::Float64                # sum V * dt = ⟨V⟩
    V_avg_sq::Float64             # sum (V^2) * dt for variance
    t_total_sq::Float64           # sum (dt^2) for effective sample size
    count::Int
end

NPTChemicalPotentialAccumulator() = NPTChemicalPotentialAccumulator(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

function reset!(acc::NPTChemicalPotentialAccumulator)
    acc.t_total = 0.0
    acc.S_over_V = 0.0
    acc.S_over_V_sq = 0.0
    acc.V_avg = 0.0
    acc.V_avg_sq = 0.0
    acc.t_total_sq = 0.0
    acc.count = 0
    return nothing
end

function Base.push!(acc::NPTChemicalPotentialAccumulator, R_over_N::Float64, V::Float64, dt::Float64)
    R_over_V = R_over_N / V
    acc.t_total += dt
    acc.S_over_V += R_over_V * dt
    acc.S_over_V_sq += (R_over_V * R_over_V) * dt
    acc.V_avg += V * dt
    acc.V_avg_sq += (V * V) * dt
    acc.t_total_sq += dt * dt
    acc.count += 1
    return acc
end

function mu_ex_npt(acc::NPTChemicalPotentialAccumulator, T::Float64)::Float64
    if acc.t_total <= 0.0
        return NaN
    end
    # Equation A.132: μ₁^(res) = kT ln [⟨(exp(βφ_N)) / V⟩ ⟨V⟩]
    # where exp(βφ_N) ≈ R/N
    avg_S_over_V = acc.S_over_V / acc.t_total  # ⟨(R/N) / V⟩
    avg_V = acc.V_avg / acc.t_total             # ⟨V⟩
    
    if avg_S_over_V <= 0.0 || avg_V <= 0.0
        return NaN
    end
    return T * log(avg_S_over_V * avg_V)
end

function mu_ex_npt_stderr(acc::NPTChemicalPotentialAccumulator, T::Float64)::Float64
    if acc.t_total <= 0.0 || acc.count < 2
        return NaN
    end
    
    # Compute averages
    avg_S_over_V = acc.S_over_V / acc.t_total
    avg_V = acc.V_avg / acc.t_total
    
    if avg_S_over_V <= 0.0 || avg_V <= 0.0
        return NaN
    end
    
    # Compute variances
    var_S_over_V = (acc.S_over_V_sq / acc.t_total) - (avg_S_over_V * avg_S_over_V)
    var_V = (acc.V_avg_sq / acc.t_total) - (avg_V * avg_V)
    var_S_over_V = max(var_S_over_V, 0.0)
    var_V = max(var_V, 0.0)
    
    # Effective number of samples
    n_eff = (acc.t_total * acc.t_total) / max(acc.t_total_sq, eps(Float64))
    if n_eff <= 1.0
        return NaN
    end
    
    # Standard errors
    stderr_S_over_V = sqrt(var_S_over_V / n_eff)
    stderr_V = sqrt(var_V / n_eff)
    
    # Error propagation: μ = T * ln(avg_S_over_V * avg_V)
    # δμ = T * sqrt((δ(avg_S_over_V) / avg_S_over_V)^2 + (δ(avg_V) / avg_V)^2)
    rel_err_S_over_V = stderr_S_over_V / avg_S_over_V
    rel_err_V = stderr_V / avg_V
    rel_err_total = sqrt(rel_err_S_over_V * rel_err_S_over_V + rel_err_V * rel_err_V)
    
    return T * rel_err_total
end

@inline function _minimum_image!(dr::MVector{3,Float64}, L::Float64)
    L_half = L / 2.0
    for k in 1:3
        if dr[k] > L_half
            dr[k] -= L
        elseif dr[k] < -L_half
            dr[k] += L
        end
    end
    return nothing
end

function compute_phi_i(i::Int, st::eKMCState, p::LJParams)::Float64
    energy = 0.0
    @inbounds for j in 1:st.N
        if j != i
            energy += st.pair_u[i, j]
        end
    end
    return energy
end

function compute_all_m_R!(st::eKMCState, p::LJParams)
    N = st.N
    β = p.β
    st.R = 0.0
    @inbounds for i in 1:N
        phi_i = compute_phi_i(i, st, p)
        st.phi[i] = phi_i
        m_i = exp(β * phi_i)
        st.m[i] = m_i
        st.R += m_i
    end
    return nothing
end

"""
    total_energy_from_pair_u(st::eKMCState, p::LJParams)::Float64

Compute total energy from stored pair_u matrix (O(N^2) sum, but much faster than recomputing distances).
Each pair is stored twice in the symmetric matrix, so we divide by 2.
"""
function total_energy_from_pair_u(st::eKMCState, p::LJParams)::Float64
    energy = 0.5 * sum(st.pair_u)
    if p.use_lrc
        energy += st.N * p.lrc_u_per_particle
    end
    return energy
end

"""
    total_virial_from_positions(st::eKMCState, p::LJParams)::Float64

Compute virial from positions, but only for pairs with non-zero energy in pair_u.
This is more efficient than full O(N^2) calculation when many pairs are beyond cutoff.
"""
function total_virial_from_positions(st::eKMCState, p::LJParams)::Float64
    virial = 0.0
    N = st.N
    L = st.L
    L_half = 0.5 * L
    rc2 = p.rc2
    pos = st.pos
    dr = st.scratch_dr
    
    @inbounds for i in 1:N
        for j in (i+1):N
            # Only compute virial for pairs that have non-zero energy (within cutoff)
            if st.pair_u[i, j] == 0.0
                continue
            end
            
            # Compute distance vector
            dr[1] = pos[1, j] - pos[1, i]
            dr[2] = pos[2, j] - pos[2, i]
            dr[3] = pos[3, j] - pos[3, i]
            _minimum_image!(dr, L)
            r2 = dr[1]*dr[1] + dr[2]*dr[2] + dr[3]*dr[3]
            
            if r2 < rc2 && r2 > 0.0
                virial += lj_force_magnitude_times_r(r2, p)
            end
        end
    end
    
    return virial
end

"""
    pressure_from_pair_u(st::eKMCState, p::LJParams, T::Float64)::Float64

Compute pressure using stored pair_u matrix for energy and efficient virial calculation.
"""
function pressure_from_pair_u(st::eKMCState, p::LJParams, T::Float64)::Float64
    N = st.N
    L = st.L
    V = L * L * L
    ρ = N / V
    W = total_virial_from_positions(st, p)
    P_sampled = ρ * T + W / (3.0 * V)
    
    # Add long-range correction if enabled
    if p.use_lrc
        P_sampled += p.lrc_p
    end
    
    # Note: Impulsive correction is not included here for eKMC (typically not used)
    # If needed, it can be added similar to the regular pressure function
    
    return P_sampled
end

function update_phi_after_move!(i::Int, oldx::Float64, oldy::Float64, oldz::Float64,
                                st::eKMCState, p::LJParams)
    N = st.N
    L = st.L
    rc2 = p.rc2
    pos = st.pos
    dr = st.scratch_dr
    β = p.β

    # New position
    pix = pos[1, i]
    piy = pos[2, i]
    piz = pos[3, i]

    # Reset phi for i; we'll rebuild it from updated pairs
    st.phi[i] = 0.0
    # Update pair energies against all other particles
    @inbounds for j in 1:N
        if j == i
            continue
        end

        # Old pair energy (stored)
        u_old = st.pair_u[i, j]

        # New pair energy
        dr[1] = pos[1, j] - pix
        dr[2] = pos[2, j] - piy
        dr[3] = pos[3, j] - piz
        _minimum_image!(dr, L)
        r2 = dr[1]*dr[1] + dr[2]*dr[2] + dr[3]*dr[3]
        u_new = (r2 < rc2 && r2 > 0.0) ? lj_pair_u_from_r2(r2, p) : 0.0

        # Update stored pair matrix (symmetric)
        st.pair_u[i, j] = u_new
        st.pair_u[j, i] = u_new

        # Update phi for j using delta
        delta = u_new - u_old
        st.phi[j] += delta
        # Accumulate phi_i directly from new pair value
        st.phi[i] += u_new
    end

    # Recompute mobilities and total rate from updated phi
    R = 0.0
    @inbounds for j in 1:N
        st.m[j] = exp(β * st.phi[j])
        R += st.m[j]
    end
    st.R = R

    if st.R <= 0.0 || !isfinite(st.R)
        error("eKMC: Total rate R is non-positive or non-finite after update: R = $(st.R)")
    end

    return nothing
end

function sample_particle_by_mobility(st::eKMCState)::Int
    N = st.N
    R = st.R
    if R <= 0.0 || !isfinite(R)
        error("eKMC: Total rate R must be positive and finite, got R = $R")
    end
    r = rand(st.rng) * R
    cumsum = 0.0
    @inbounds for i in 1:N
        cumsum += st.m[i]
        if cumsum > r
            return i
        end
    end
    return N
end

function ekmc_step!(st::eKMCState, p::LJParams, acc::ChemicalPotentialAccumulator)::Float64
    N = st.N
    L = st.L
    R = st.R
    if R <= 0.0 || !isfinite(R)
        error("eKMC: Total rate R must be positive and finite, got R = $R")
    end
    u = rand(st.rng)
    u = max(eps(Float64), min(u, 1.0 - eps(Float64)))
    dt = log(1/u) / R
    if dt <= 0.0 || !isfinite(dt)
        error("eKMC: dt must be positive and finite, got dt = $dt (u = $u, R = $R)")
    end

    acc.t_total += dt
    R_over_N = R / N
    acc.S += R_over_N * dt
    acc.S_sq += (R_over_N * R_over_N) * dt
    acc.t_total_sq += dt * dt
    acc.count += 1

    i = sample_particle_by_mobility(st)

    oldx = st.pos[1, i]
    oldy = st.pos[2, i]
    oldz = st.pos[3, i]

    st.pos[1, i] = rand(st.rng) * L
    st.pos[2, i] = rand(st.rng) * L
    st.pos[3, i] = rand(st.rng) * L

    update_phi_after_move!(i, oldx, oldy, oldz, st, p)

    if st.debug_check_every > 0 && (acc.count % st.debug_check_every == 0)
        # Energy consistency: pair matrix vs full energy
        E_pair = 0.5 * sum(st.pair_u)
        if p.use_lrc
            E_pair += st.N * p.lrc_u_per_particle
        end
        E_full = total_energy(st, p)
        if !isfinite(E_pair) || !isfinite(E_full) || abs(E_pair - E_full) > 1e-6
            println("eKMC debug: energy mismatch, pair=$(E_pair), full=$(E_full), Δ=$(E_pair - E_full)")
        end

        # Recompute phi and R from scratch for consistency
        max_phi_err = 0.0
        R_re = 0.0
        @inbounds for k in 1:st.N
            phi_k = 0.0
            for j in 1:st.N
                if j != k
                    phi_k += st.pair_u[k, j]
                end
            end
            max_phi_err = max(max_phi_err, abs(phi_k - st.phi[k]))
            R_re += exp(p.β * phi_k)
        end
        ΔR = st.R - R_re
        rel_ΔR = isfinite(R_re) && R_re != 0.0 ? abs(ΔR) / abs(R_re) : NaN
        if max_phi_err > 1e-6 || !isfinite(R_re) || (isfinite(rel_ΔR) && rel_ΔR > 1e-6)
            println("eKMC debug: phi_err=$(max_phi_err), R=$(st.R), R_re=$(R_re), rel_ΔR=$(rel_ΔR)")
        end
    end
    return dt
end

function mu_ex(acc::ChemicalPotentialAccumulator, T::Float64)::Float64
    if acc.t_total <= 0.0
        return NaN
    end
    ratio = acc.S / acc.t_total
    if ratio <= 0.0
        return NaN
    end
    return T * log(ratio)
end

function mu_ex_stderr(acc::ChemicalPotentialAccumulator, T::Float64)::Float64
    if acc.t_total <= 0.0 || acc.count < 2
        return NaN
    end
    # Compute variance of (R/N) using time-weighted statistics
    μ_RN = acc.S / acc.t_total
    var_RN = (acc.S_sq / acc.t_total) - (μ_RN * μ_RN)
    var_RN = max(var_RN, 0.0)  # Ensure non-negative
    
    # Effective number of samples (same formula as TimeWeightedObservable)
    n_eff = (acc.t_total * acc.t_total) / max(acc.t_total_sq, eps(Float64))
    if n_eff <= 1.0
        return NaN
    end
    
    # Standard error of (R/N) average
    stderr_RN = sqrt(var_RN / n_eff)
    
    # Error propagation: μ = T * log(S/t_total) = T * log(μ_RN)
    # δμ = T * (δμ_RN) / μ_RN
    if μ_RN <= 0.0
        return NaN
    end
    return T * stderr_RN / μ_RN
end

function init_ekmc_state(st::LJState, p::LJParams)::eKMCState
    N = st.N
    ekst = eKMCState(
        N, st.L, copy(st.pos), fill(1, N),
        st.rng,
        MVector{3,Float64}(0.0, 0.0, 0.0),
        zeros(Float64, N, N),
        zeros(Float64, N),
        zeros(Float64, N),
        0.0,
        zeros(Float64, nthreads()),
        0
    )

    # Initialize pair energies and phi
    @inbounds for i in 1:N
        for j in (i+1):N
            # Compute pair energy once
            drx = ekst.pos[1, j] - ekst.pos[1, i]
            dry = ekst.pos[2, j] - ekst.pos[2, i]
            drz = ekst.pos[3, j] - ekst.pos[3, i]
            dr = ekst.scratch_dr
            dr[1] = drx
            dr[2] = dry
            dr[3] = drz
            _minimum_image!(dr, ekst.L)
            r2 = dr[1]*dr[1] + dr[2]*dr[2] + dr[3]*dr[3]
            u = (r2 < p.rc2 && r2 > 0.0) ? lj_pair_u_from_r2(r2, p) : 0.0
            ekst.pair_u[i, j] = u
            ekst.pair_u[j, i] = u
            ekst.phi[i] += u
            ekst.phi[j] += u
        end
    end

    # Initialize mobilities and total rate
    compute_all_m_R!(ekst, p)
    return ekst
end

function run_ekmc!(st::eKMCState, p::LJParams, acc::ChemicalPotentialAccumulator;
                   nsteps::Int=1000)::Float64
    total_time = 0.0
    for _ in 1:nsteps
        dt = ekmc_step!(st, p, acc)
        total_time += dt
    end
    return total_time
end

# ============================================================================
# NPT Ensemble (Tan et al. method)
# ============================================================================

"""
    rebuild_pair_phi_R_after_volume!(st::eKMCState, p::LJParams)

Rebuild pair energies, phi, mobilities, and total rate after a volume change.
All positions have been scaled and wrapped, but pair_u matrix is stale.
"""
function rebuild_pair_phi_R_after_volume!(st::eKMCState, p::LJParams)
    N = st.N
    L = st.L
    rc2 = p.rc2
    pos = st.pos
    dr = st.scratch_dr
    β = p.β

    # Reset pair energies and phi
    fill!(st.pair_u, 0.0)
    fill!(st.phi, 0.0)

    # Recompute all pair energies
    @inbounds for i in 1:N
        for j in (i+1):N
            dr[1] = pos[1, j] - pos[1, i]
            dr[2] = pos[2, j] - pos[2, i]
            dr[3] = pos[3, j] - pos[3, i]
            _minimum_image!(dr, L)
            r2 = dr[1]*dr[1] + dr[2]*dr[2] + dr[3]*dr[3]
            u = (r2 < rc2 && r2 > 0.0) ? lj_pair_u_from_r2(r2, p) : 0.0
            st.pair_u[i, j] = u
            st.pair_u[j, i] = u
            st.phi[i] += u
            st.phi[j] += u
        end
    end

    # Recompute mobilities and total rate
    st.R = 0.0
    @inbounds for i in 1:N
        st.m[i] = exp(β * st.phi[i])
        st.R += st.m[i]
    end

    if st.R <= 0.0 || !isfinite(st.R)
        error("eKMC NPT: Total rate R is non-positive or non-finite after volume change: R = $(st.R)")
    end

    return nothing
end

"""
    ekmc_volume_move_tan!(st::eKMCState, p::LJParams, Pext::Float64, max_dV::Float64)::Float64

Perform a volume change move using Tan et al. auxiliary pressure method.
This is rejection-free: the volume change is deterministic based on pressure comparison.

Algorithm (Tan et al., CEJ 2017):
- Compute current pressure p from virial
- Define auxiliary pressure: p_aux = n * (p + Pext) where n is random
- If p > p_aux: increase volume: V' = V + n * ΔV
- If p < p_aux: decrease volume: V' = V - n * ΔV
- All moves are accepted (rejection-free)

Returns the residence time dt for this volume move.
"""
function ekmc_volume_move_tan!(st::eKMCState, p::LJParams, Pext::Float64, max_dV::Float64)
    N = st.N
    L_old = st.L
    V_old = L_old * L_old * L_old
    T = 1.0 / p.β

    # Store rate BEFORE volume move (for residence time calculation)
    R_before = st.R
    if R_before <= 0.0 || !isfinite(R_before)
        error("eKMC NPT: Total rate R must be positive and finite before volume move: R = $R_before")
    end

    # Compute current pressure from virial using optimized function (uses stored pair_u matrix)
    # Note: This is called BEFORE volume change, so pair_u is still valid
    p_current = pressure_from_pair_u(st, p, T)

    # Tan et al. auxiliary pressure method
    # p_aux = n * (p + p*), where n is random, p* is specified pressure, p is current pressure
    n = rand(st.rng)  # Random number in [0, 1)
    p_aux = n * (p_current + Pext)

    # Determine volume change direction and magnitude
    if p_current > p_aux
        # Pressure too high, increase volume: V' = V + n * ΔV
        dV = n * max_dV
        V_new = V_old + dV
    else
        # Pressure too low, decrease volume: V' = V - n * ΔV
        dV = -n * max_dV
        V_new = V_old + dV
    end

    # Ensure volume is positive
    if V_new <= 0.0
        V_new = max(V_old * 0.5, 1e-10)  # Safety: don't collapse to zero
    end

    L_new = cbrt(V_new)
    scale = L_new / L_old

    # Scale all positions
    @inbounds for i in 1:N
        st.pos[1, i] *= scale
        st.pos[2, i] *= scale
        st.pos[3, i] *= scale
    end

    # Wrap all positions to [0, L_new)
    scratch = st.scratch_dr
    @inbounds for i in 1:N
        scratch[1] = st.pos[1, i]
        scratch[2] = st.pos[2, i]
        scratch[3] = st.pos[3, i]
        wrap!(scratch, L_new)
        st.pos[1, i] = scratch[1]
        st.pos[2, i] = scratch[2]
        st.pos[3, i] = scratch[3]
    end

    # Update box length
    st.L = L_new

    # Rebuild pair energies, phi, mobilities, and total rate
    rebuild_pair_phi_R_after_volume!(st, p)

    # Volume moves are instantaneous - they don't have a separate residence time
    # The configuration after the volume move will be weighted by subsequent particle moves
    return nothing
end

"""
    TimeWeightedAccumulator

Accumulator for time-weighted averages in NPT ensemble.
"""
mutable struct TimeWeightedAccumulator
    weighted_sum::Float64
    weighted_sum_sq::Float64
    total_time::Float64
    total_time_sq::Float64
    count::Int
end

TimeWeightedAccumulator() = TimeWeightedAccumulator(0.0, 0.0, 0.0, 0.0, 0)

function Base.push!(acc::TimeWeightedAccumulator, x::Float64, dt::Float64)
    acc.weighted_sum += x * dt
    acc.weighted_sum_sq += x * x * dt
    acc.total_time += dt
    acc.total_time_sq += dt * dt
    acc.count += 1
    return acc
end

function mean(acc::TimeWeightedAccumulator)::Float64
    return acc.total_time > 0.0 ? acc.weighted_sum / acc.total_time : NaN
end

function stderr(acc::TimeWeightedAccumulator)::Float64
    if acc.total_time <= 0.0 || acc.count < 2
        return NaN
    end
    μ = acc.weighted_sum / acc.total_time
    var_pop = acc.weighted_sum_sq / acc.total_time - μ * μ
    var_pop = max(var_pop, 0.0)
    n_eff = (acc.total_time * acc.total_time) / max(acc.total_time_sq, eps(Float64))
    return n_eff > 1.0 ? sqrt(var_pop / n_eff) : NaN
end

"""
    NPTObservables

Observables accumulated during NPT eKMC simulation.
"""
mutable struct NPTObservables
    rho::TimeWeightedAccumulator
    U_per_particle::TimeWeightedAccumulator
    pressure::TimeWeightedAccumulator
    mu_ex::Float64
    mu_ex_err::Float64
    vol_moves::Int
end

NPTObservables() = NPTObservables(
    TimeWeightedAccumulator(),
    TimeWeightedAccumulator(),
    TimeWeightedAccumulator(),
    NaN, NaN, 0
)

"""
    run_ekmc_npt!(st::eKMCState, p::LJParams, acc_mu::ChemicalPotentialAccumulator;
                  nsteps::Int=10000, Pext::Float64=1.0, max_dV::Float64=0.1,
                  vol_move_every::Int=10, sample_every::Int=10,
                  collect_timeseries::Bool=false)

Run NPT eKMC simulation using Tan et al. method.

Alternates between:
1. Particle displacement moves (NVT eKMC steps)
2. Volume change moves (Tan et al. auxiliary pressure method)

Parameters:
- nsteps: Total number of particle moves
- Pext: External pressure
- max_dV: Maximum volume change per move
- vol_move_every: Perform volume move every N particle moves
- sample_every: Sample observables every N events (particle or volume)
- collect_timeseries: If true, return timeseries data

Returns: (obs::NPTObservables, total_time::Float64, timeseries::Union{Nothing, Dict})
"""
function run_ekmc_npt!(st::eKMCState, p::LJParams, acc_mu::ChemicalPotentialAccumulator;
                       nsteps::Int=10000, Pext::Float64=1.0, max_dV::Float64=0.1,
                       vol_move_every::Int=10, sample_every::Int=10,
                       collect_timeseries::Bool=false)
    T = 1.0 / p.β
    N = st.N

    # Reset chemical potential accumulator
    reset!(acc_mu)

    # Initialize observables
    obs = NPTObservables()
    acc_mu_prod = NPTChemicalPotentialAccumulator()  # Use NPT-specific accumulator
    total_time = 0.0

    # Timeseries storage
    timeseries = collect_timeseries ? Dict(
        :event_idx => Int[],
        :time => Float64[],
        :rho => Float64[],
        :U_per_particle => Float64[],
        :pressure => Float64[],
        :L => Float64[]
    ) : nothing

    event_idx = 0
    particle_move_count = 0  # Track particle moves for volume move frequency

    for step_idx in 1:nsteps
        # Every step: compute dt from current state
        R_current = st.R
        if R_current <= 0.0 || !isfinite(R_current)
            error("eKMC NPT: Total rate R must be positive and finite: R = $R_current")
        end
        
        u = rand(st.rng)
        u = max(eps(Float64), min(u, 1.0 - eps(Float64)))
        dt = log(1/u) / R_current
        
        # Check if this is a sampling step
        do_sample = (event_idx % sample_every == 0)
        
        # Only accumulate dt and observables when sampling
        if do_sample && dt > 0.0
            # Sample observables from current state using optimized functions
            # (use stored pair_u matrix instead of recomputing distances)
            U = total_energy_from_pair_u(st, p)
            P = pressure_from_pair_u(st, p, T)
            ρ_inst = N / (st.L * st.L * st.L)
            V = st.L * st.L * st.L
            
            # Weight observables by dt from this sampled state
            push!(obs.U_per_particle, U / N, dt)
            push!(obs.pressure, P, dt)
            push!(obs.rho, ρ_inst, dt)
            
            # For NPT, accumulate (R/N)/V and V separately (Equation A.132)
            R_over_N = R_current / N
            push!(acc_mu_prod, R_over_N, V, dt)
            
            if collect_timeseries
                push!(timeseries[:event_idx], event_idx)
                push!(timeseries[:time], total_time + dt)
                push!(timeseries[:rho], ρ_inst)
                push!(timeseries[:U_per_particle], U / N)
                push!(timeseries[:pressure], P)
                push!(timeseries[:L], st.L)
            end
        end
        
        # Always update accumulator and perform move (even if not sampling)
        acc_mu.t_total += dt
        R_over_N = R_current / N
        acc_mu.S += R_over_N * dt
        acc_mu.S_sq += (R_over_N * R_over_N) * dt
        acc_mu.t_total_sq += dt * dt
        acc_mu.count += 1
        
        # Perform the move to next state
        i = sample_particle_by_mobility(st)
        oldx = st.pos[1, i]
        oldy = st.pos[2, i]
        oldz = st.pos[3, i]
        st.pos[1, i] = rand(st.rng) * st.L
        st.pos[2, i] = rand(st.rng) * st.L
        st.pos[3, i] = rand(st.rng) * st.L
        update_phi_after_move!(i, oldx, oldy, oldz, st, p)
        
        total_time += dt
        event_idx += 1
        particle_move_count += 1

        # Volume change move every vol_move_every * N particle moves
        # (matching MC frequency: vol_move_every sweeps, where 1 sweep = N moves)
        if particle_move_count % (vol_move_every * N) == 0
            # Store rate BEFORE volume move (for accumulator update)
            R_before_vol = st.R
            
            do_sample_vol = (event_idx % sample_every == 0)
            if do_sample_vol
                U_vol = total_energy_from_pair_u(st, p)
                P_vol = pressure_from_pair_u(st, p, T)
                ρ_vol = N / (st.L * st.L * st.L)
            end

            # Perform volume move (deterministic, instantaneous - no separate residence time)
            # Volume moves are instantaneous transitions - they don't contribute separate dt
            # The configuration AFTER the volume move will be weighted by subsequent particle moves' dt
            ekmc_volume_move_tan!(st, p, Pext, max_dV)
            obs.vol_moves += 1
            # Volume moves don't increment event_idx or add to time - they're instantaneous
        end
    end

    # Compute chemical potential using NPT formula (Equation A.132)
    obs.mu_ex = mu_ex_npt(acc_mu_prod, T)
    obs.mu_ex_err = mu_ex_npt_stderr(acc_mu_prod, T)

    return (obs, total_time, timeseries)
end
