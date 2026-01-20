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
    dt = -log(u) / R
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
