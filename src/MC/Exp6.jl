"""
EXP-6 (Buckingham) pair potential support.

This file adds EXP-6 functionality via new types and methods only.
Existing LJ codepaths are untouched.
"""

using StaticArrays

abstract type AbstractForceField end
struct Exp6FF <: AbstractForceField end

struct Exp6Params
    ff::Exp6FF
    rc::Float64
    rc2::Float64
    β::Float64
    max_disp::Float64
    cutoff_model::Symbol  # :truncated or :shifted
    use_lrc::Bool
    lrc_u_per_particle::Float64
    lrc_p::Float64
    n_types::Int
    rm_types::Vector{Float64}
    ϵ_types::Vector{Float64}
    α_types::Vector{Float64}
    rm_mix::Matrix{Float64}
    ϵ_mix::Matrix{Float64}
    α_mix::Matrix{Float64}
    A_mix::Matrix{Float64}
    B_mix::Matrix{Float64}
    k_mix::Matrix{Float64}
    exp_alpha_mix::Matrix{Float64}
    EA_mix::Matrix{Float64}
    EB_mix::Matrix{Float64}
    rm6_mix::Matrix{Float64}
    u_rc_mix::Matrix{Float64}
    u_tail_mix::Matrix{Float64}
    p_tail_mix::Matrix{Float64}
    r_guard_mix::Matrix{Float64}
    rmin_factor::Float64
end

const _KB_J_PER_K = 1.380649e-23
const _ANGSTROM_M = 1.0e-10
const _GPA_PA = 1.0e9

"""
    exp6_to_reduced(rm_A, ϵ_K, α; rm_ref_A, ϵ_ref_K, T_K, P_GPa)

Convert Exp-6 parameters and statepoint from physical units (Å, K, GPa) to reduced
units using reference species (rm_ref_A, ϵ_ref_K).
"""
function exp6_to_reduced(rm_A::Vector{Float64}, ϵ_K::Vector{Float64}, α::Vector{Float64};
                         rm_ref_A::Float64, ϵ_ref_K::Float64,
                         T_K::Float64, P_GPa::Float64)
    rm_star = rm_A ./ rm_ref_A
    ϵ_star = ϵ_K ./ ϵ_ref_K
    α_star = copy(α)
    T_star = T_K / ϵ_ref_K
    ϵ_ref_J = ϵ_ref_K * _KB_J_PER_K
    rm_ref_m = rm_ref_A * _ANGSTROM_M
    P_star = (P_GPa * _GPA_PA) * (rm_ref_m^3) / ϵ_ref_J
    return (rm_star, ϵ_star, α_star, T_star, P_star)
end

@inline function _exp6_f_ratio(x::Float64, α::Float64)::Float64
    # f(x) = A * exp(α(1-x)) - B * x^-6
    A = 6.0 / (α - 6.0)
    B = α / (α - 6.0)
    return A * exp(α * (1.0 - x)) - B * (x ^ -6.0)
end

function _exp6_small_root_ratio(α::Float64; x_min::Float64=1e-4, nscan::Int=512)::Float64
    # Find the smaller root x in (0, 1) of f(x)=0, if it exists.
    # We scan from x=1 downwards to detect sign changes.
    x_prev = 1.0
    f_prev = _exp6_f_ratio(x_prev, α)
    found_positive = false
    root_interval = nothing
    step = (1.0 - x_min) / nscan
    for k in 1:nscan
        x = 1.0 - k * step
        if x <= x_min
            x = x_min
        end
        f = _exp6_f_ratio(x, α)
        if !found_positive
            if f_prev <= 0.0 && f > 0.0
                # Passed the larger root (negative -> positive)
                found_positive = true
            end
        else
            if f_prev >= 0.0 && f < 0.0
                # Passed the smaller root (positive -> negative)
                root_interval = (x, x_prev)
                break
            end
        end
        x_prev = x
        f_prev = f
    end
    if root_interval === nothing
        return 0.0
    end
    a, b = root_interval
    fa = _exp6_f_ratio(a, α)
    fb = _exp6_f_ratio(b, α)
    if fa == 0.0
        return a
    elseif fb == 0.0
        return b
    end
    for _ in 1:64
        m = 0.5 * (a + b)
        fm = _exp6_f_ratio(m, α)
        if fm == 0.0
            return m
        elseif fa * fm < 0.0
            b = m
            fb = fm
        else
            a = m
            fa = fm
        end
    end
    return 0.5 * (a + b)
end

@inline function _exp6_tail_integrals(rc::Float64, rm::Float64, ϵ::Float64, α::Float64)::Tuple{Float64, Float64}
    if rc <= 0.0
        return (0.0, 0.0)
    end
    A = 6.0 / (α - 6.0)
    B = α / (α - 6.0)
    k = α / rm
    expfac = exp(α * (1.0 - rc / rm))

    # Energy integral: ∫_{rc}^∞ r^2 u(r) dr
    term_exp_u = expfac * (rc * rc / k + 2.0 * rc / (k * k) + 2.0 / (k * k * k))
    term_pow_u = rm^6 / (3.0 * rc^3)
    u_int = ϵ * (A * term_exp_u - B * term_pow_u)

    # Pressure integral: ∫_{rc}^∞ r^3 * (-du/dr) dr
    term_exp_p = expfac * (rc^3 + 3.0 * rc * rc / k + 6.0 * rc / (k * k) + 6.0 / (k * k * k))
    term_pow_p = 2.0 * rm^6 / (rc^3)
    p_int = ϵ * (A * term_exp_p - B * term_pow_p)

    return (u_int, p_int)
end

function Exp6Params(; rm_types::Vector{Float64}, ϵ_types::Vector{Float64}, α_types::Vector{Float64},
                    rc::Float64, T::Float64, max_disp::Float64=0.1,
                    cutoff_model::Symbol=:truncated, mixing_rule::Symbol=:lb,
                    rmin_factor::Float64=0.2, use_lrc::Bool=false)
    n_types = length(rm_types)
    @assert length(ϵ_types) == n_types "rm_types and ϵ_types must have same length"
    @assert length(α_types) == n_types "rm_types and α_types must have same length"
    @assert n_types >= 1 "Must have at least 1 type"
    if cutoff_model != :truncated && cutoff_model != :shifted
        throw(ArgumentError("cutoff_model must be :truncated or :shifted, got :$cutoff_model"))
    end
    if mixing_rule != :lb && mixing_rule != :none
        throw(ArgumentError("mixing_rule must be :lb or :none, got :$mixing_rule"))
    end
    rc2 = rc * rc
    β = 1.0 / T
    
    rm_mix = zeros(Float64, n_types, n_types)
    ϵ_mix = zeros(Float64, n_types, n_types)
    α_mix = zeros(Float64, n_types, n_types)
    
    if mixing_rule == :lb
        for i in 1:n_types
            for j in 1:n_types
                rm_mix[i, j] = 0.5 * (rm_types[i] + rm_types[j])
                ϵ_mix[i, j] = sqrt(ϵ_types[i] * ϵ_types[j])
                # α_ij uses arithmetic mean (species α_i, α_j)
                α_mix[i, j] = 0.5 * (α_types[i] + α_types[j])
            end
        end
    else
        throw(ArgumentError("mixing_rule=:none requires explicit pair tables (not implemented)"))
    end
    
    u_rc_mix = zeros(Float64, n_types, n_types)
    A_mix = zeros(Float64, n_types, n_types)
    B_mix = zeros(Float64, n_types, n_types)
    k_mix = zeros(Float64, n_types, n_types)
    exp_alpha_mix = zeros(Float64, n_types, n_types)
    EA_mix = zeros(Float64, n_types, n_types)
    EB_mix = zeros(Float64, n_types, n_types)
    rm6_mix = zeros(Float64, n_types, n_types)
    u_tail_mix = zeros(Float64, n_types, n_types)
    p_tail_mix = zeros(Float64, n_types, n_types)
    r_guard_mix = zeros(Float64, n_types, n_types)
    if cutoff_model == :shifted
        for i in 1:n_types
            for j in 1:n_types
                u_rc_mix[i, j] = exp6_potential(rc2, rm_mix[i, j], ϵ_mix[i, j], α_mix[i, j], rmin_factor)
            end
        end
    end
    for i in 1:n_types
        for j in 1:n_types
            rm_ij = rm_mix[i, j]
            α_ij = α_mix[i, j]
            A_ij = 6.0 / (α_ij - 6.0)
            B_ij = α_ij / (α_ij - 6.0)
            A_mix[i, j] = A_ij
            B_mix[i, j] = B_ij
            k_mix[i, j] = α_ij / rm_ij
            exp_alpha_mix[i, j] = exp(α_ij)
            EA_mix[i, j] = ϵ_mix[i, j] * A_ij
            EB_mix[i, j] = ϵ_mix[i, j] * B_ij
            rm6_mix[i, j] = rm_ij^6
            u_tail_mix[i, j], p_tail_mix[i, j] = _exp6_tail_integrals(rc, rm_ij, ϵ_mix[i, j], α_ij)
            # Guard against unphysical small-r root
            x_root = _exp6_small_root_ratio(α_ij)
            r_root = x_root > 0.0 ? x_root * rm_ij : 0.0
            r_guard_mix[i, j] = max(rmin_factor * rm_ij, r_root)
        end
    end
    
    return Exp6Params(Exp6FF(), rc, rc2, β, max_disp, cutoff_model,
                      use_lrc, 0.0, 0.0, n_types,
                      rm_types, ϵ_types, α_types,
                      rm_mix, ϵ_mix, α_mix, A_mix, B_mix, k_mix, exp_alpha_mix, EA_mix, EB_mix, rm6_mix,
                      u_rc_mix, u_tail_mix, p_tail_mix, r_guard_mix, rmin_factor)
end

"""
    exp6_potential(r2, rm, ϵ, α, rmin_factor)

Canonical EXP-6 energy (Buckingham). For very short distances, clamp r to rmin to
avoid numerical overflow. This is a numerical guard, not a physical modification.
"""
@inline function exp6_potential(r2::Float64, rm::Float64, ϵ::Float64, α::Float64, rmin_factor::Float64)::Float64
    if r2 <= 0.0
        return Inf
    end
    r = sqrt(r2)
    rmin = rmin_factor * rm
    if r < rmin
        r = rmin
    end
    A = 6.0 / (α - 6.0)
    B = α / (α - 6.0)
    exp_term = exp(α * (1.0 - r / rm))
    ratio = rm / r
    ratio2 = ratio * ratio
    ratio6 = ratio2 * ratio2 * ratio2
    return ϵ * (A * exp_term - B * ratio6)
end

@inline function pair_energy(::Exp6FF, r2::Float64, rm::Float64, ϵ::Float64, α::Float64, rmin_factor::Float64)::Float64
    return exp6_potential(r2, rm, ϵ, α, rmin_factor)
end

@inline function exp6_pair_u_from_r2(r2::Float64, i::Int, j::Int, p::Exp6Params)::Float64
    if r2 >= p.rc2 || r2 <= 0.0
        return 0.0
    end
    @inbounds rm = p.rm_mix[i, j]
    @inbounds EA = p.EA_mix[i, j]
    @inbounds EB = p.EB_mix[i, j]
    @inbounds k = p.k_mix[i, j]
    @inbounds exp_alpha = p.exp_alpha_mix[i, j]
    @inbounds rm6 = p.rm6_mix[i, j]
    r = sqrt(r2)
    @inbounds r_guard = p.r_guard_mix[i, j]
    if r < r_guard
        r = r_guard
    end
    exp_term = exp_alpha * Base.FastMath.exp_fast(-k * r)
    invr = 1.0 / r
    invr2 = invr * invr
    invr6 = invr2 * invr2 * invr2
    u = EA * exp_term - EB * (rm6 * invr6)
    if p.cutoff_model == :shifted
        @inbounds u -= p.u_rc_mix[i, j]
    end
    return u
end

@inline function exp6_force_magnitude_times_r(r2::Float64, rm::Float64, ϵ::Float64, α::Float64, rmin_factor::Float64)::Float64
    if r2 >= 0.0
        if r2 >= 1e-14
            r = sqrt(r2)
            rmin = rmin_factor * rm
            if r < rmin
                r = rmin
            end
            A = 6.0 / (α - 6.0)
            B = α / (α - 6.0)
            exp_term = exp(α * (1.0 - r / rm))
            ratio = rm / r
            ratio2 = ratio * ratio
            ratio6 = ratio2 * ratio2 * ratio2
            return ϵ * (A * α * (r / rm) * exp_term - 6.0 * B * ratio6)
        else
            return Inf
        end
    end
    return 0.0
end

@inline function pair_force_times_r(::Exp6FF, r2::Float64, rm::Float64, ϵ::Float64, α::Float64, rmin_factor::Float64)::Float64
    return exp6_force_magnitude_times_r(r2, rm, ϵ, α, rmin_factor)
end

@inline function exp6_force_magnitude_times_r_mixed(r2::Float64, i::Int, j::Int, p::Exp6Params)::Float64
    if r2 >= p.rc2 || r2 <= 0.0
        return 0.0
    end
    @inbounds rm = p.rm_mix[i, j]
    @inbounds EA = p.EA_mix[i, j]
    @inbounds EB = p.EB_mix[i, j]
    @inbounds k = p.k_mix[i, j]
    @inbounds exp_alpha = p.exp_alpha_mix[i, j]
    @inbounds rm6 = p.rm6_mix[i, j]
    r = sqrt(r2)
    @inbounds r_guard = p.r_guard_mix[i, j]
    if r < r_guard
        r = r_guard
    end
    exp_term = exp_alpha * Base.FastMath.exp_fast(-k * r)
    invr = 1.0 / r
    invr2 = invr * invr
    invr6 = invr2 * invr2 * invr2
    return EA * (k * r * exp_term) - 6.0 * EB * (rm6 * invr6)
end

@inline function _counts_from_types(types::Vector{Int}, n_types::Int)::Vector{Int}
    counts = zeros(Int, n_types)
    @inbounds for t in types
        if 1 <= t <= n_types
            counts[t] += 1
        end
    end
    return counts
end

@inline function exp6_lrc_energy_per_particle(counts::Vector{Int}, L::Float64, p::Exp6Params)::Float64
    N = sum(counts)
    if N == 0
        return 0.0
    end
    V = L * L * L
    ρ = N / V
    mix_sum = 0.0
    inv_N2 = 1.0 / (Float64(N) * Float64(N))
    @inbounds for i in 1:p.n_types
        for j in 1:p.n_types
            mix_sum += (Float64(counts[i]) * Float64(counts[j]) * inv_N2) * p.u_tail_mix[i, j]
        end
    end
    return 2.0 * π * ρ * mix_sum
end

@inline function exp6_lrc_energy_total(counts::Vector{Int}, L::Float64, p::Exp6Params)::Float64
    N = sum(counts)
    return Float64(N) * exp6_lrc_energy_per_particle(counts, L, p)
end

@inline function exp6_lrc_pressure(counts::Vector{Int}, L::Float64, p::Exp6Params)::Float64
    N = sum(counts)
    if N == 0
        return 0.0
    end
    V = L * L * L
    ρ = N / V
    mix_sum = 0.0
    inv_N2 = 1.0 / (Float64(N) * Float64(N))
    @inbounds for i in 1:p.n_types
        for j in 1:p.n_types
            mix_sum += (Float64(counts[i]) * Float64(counts[j]) * inv_N2) * p.p_tail_mix[i, j]
        end
    end
    return (2.0 * π / 3.0) * ρ * ρ * mix_sum
end

@inline function exp6_lrc_mu(counts::Vector{Int}, L::Float64, p::Exp6Params, test_type::Int)::Float64
    N = sum(counts)
    if N == 0
        return 0.0
    end
    V = L * L * L
    ρ = N / V
    mix_sum = 0.0
    inv_N = 1.0 / Float64(N)
    @inbounds for j in 1:p.n_types
        mix_sum += (Float64(counts[j]) * inv_N) * p.u_tail_mix[test_type, j]
    end
    return 2.0 * π * ρ * mix_sum
end

function total_energy(st::LJState, p::Exp6Params)::Float64
    energy = 0.0
    N = st.N
    L = st.L
    L_half = 0.5 * L
    rc2 = p.rc2
    pos = st.pos
    types = st.types
    
    if Base.Threads.nthreads() > 1 && N > 128
        sums = zeros(Float64, Base.Threads.maxthreadid())
        Base.Threads.@threads for i in 1:N
            tid = Base.Threads.threadid()
            local_sum = 0.0
            @inbounds type_i = types[i]
            for j in (i+1):N
                @inbounds type_j = types[j]
                @inbounds dr_x = pos[1, j] - pos[1, i]
                @inbounds dr_y = pos[2, j] - pos[2, i]
                @inbounds dr_z = pos[3, j] - pos[3, i]
                if dr_x > L_half
                    dr_x -= L
                elseif dr_x < -L_half
                    dr_x += L
                end
                if dr_y > L_half
                    dr_y -= L
                elseif dr_y < -L_half
                    dr_y += L
                end
                if dr_z > L_half
                    dr_z -= L
                elseif dr_z < -L_half
                    dr_z += L
                end
                r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
                if r2 < rc2 && r2 > 0.0
                    local_sum += exp6_pair_u_from_r2(r2, type_i, type_j, p)
                end
            end
            sums[tid] += local_sum
        end
        energy = sum(sums)
    else
        @inbounds for i in 1:N
            type_i = types[i]
            for j in (i+1):N
                type_j = types[j]
                dr_x = pos[1, j] - pos[1, i]
                dr_y = pos[2, j] - pos[2, i]
                dr_z = pos[3, j] - pos[3, i]
                if dr_x > L_half
                    dr_x -= L
                elseif dr_x < -L_half
                    dr_x += L
                end
                if dr_y > L_half
                    dr_y -= L
                elseif dr_y < -L_half
                    dr_y += L
                end
                if dr_z > L_half
                    dr_z -= L
                elseif dr_z < -L_half
                    dr_z += L
                end
                r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
                if r2 < rc2 && r2 > 0.0
                    energy += exp6_pair_u_from_r2(r2, type_i, type_j, p)
                end
            end
        end
    end
    if p.use_lrc
        counts = _counts_from_types(st.types, p.n_types)
        energy += exp6_lrc_energy_total(counts, L, p)
    end
    return energy
end

function total_virial(st::LJState, p::Exp6Params)::Float64
    virial = 0.0
    N = st.N
    L = st.L
    L_half = 0.5 * L
    rc2 = p.rc2
    pos = st.pos
    types = st.types
    
    if Base.Threads.nthreads() > 1 && N > 128
        sums = zeros(Float64, Base.Threads.maxthreadid())
        Base.Threads.@threads for i in 1:N
            tid = Base.Threads.threadid()
            local_sum = 0.0
            @inbounds type_i = types[i]
            for j in (i+1):N
                @inbounds type_j = types[j]
                @inbounds dr_x = pos[1, j] - pos[1, i]
                @inbounds dr_y = pos[2, j] - pos[2, i]
                @inbounds dr_z = pos[3, j] - pos[3, i]
                if dr_x > L_half
                    dr_x -= L
                elseif dr_x < -L_half
                    dr_x += L
                end
                if dr_y > L_half
                    dr_y -= L
                elseif dr_y < -L_half
                    dr_y += L
                end
                if dr_z > L_half
                    dr_z -= L
                elseif dr_z < -L_half
                    dr_z += L
                end
                r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
                if r2 < rc2 && r2 > 0.0
                    local_sum += exp6_force_magnitude_times_r_mixed(r2, type_i, type_j, p)
                end
            end
            sums[tid] += local_sum
        end
        virial = sum(sums)
    else
        @inbounds for i in 1:N
            type_i = types[i]
            for j in (i+1):N
                type_j = types[j]
                dr_x = pos[1, j] - pos[1, i]
                dr_y = pos[2, j] - pos[2, i]
                dr_z = pos[3, j] - pos[3, i]
                if dr_x > L_half
                    dr_x -= L
                elseif dr_x < -L_half
                    dr_x += L
                end
                if dr_y > L_half
                    dr_y -= L
                elseif dr_y < -L_half
                    dr_y += L
                end
                if dr_z > L_half
                    dr_z -= L
                elseif dr_z < -L_half
                    dr_z += L
                end
                r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
                if r2 < rc2 && r2 > 0.0
                    virial += exp6_force_magnitude_times_r_mixed(r2, type_i, type_j, p)
                end
            end
        end
    end
    
    return virial
end

function pressure(st::LJState, p::Exp6Params, T::Float64)::Float64
    N = st.N
    L = st.L
    V = L * L * L
    ρ = N / V
    W = total_virial(st, p)
    P = ρ * T + W / (3.0 * V)
    if p.use_lrc
        counts = _counts_from_types(st.types, p.n_types)
        P += exp6_lrc_pressure(counts, L, p)
    end
    return P
end

function local_energy(i::Int, st::LJState, p::Exp6Params)::Float64
    energy = 0.0
    L = st.L
    L_half = 0.5 * L
    rc2 = p.rc2
    ncell = st.cl.ncell
    pos = st.pos
    types = st.types
    rm_mix = p.rm_mix
    EA_mix = p.EA_mix
    EB_mix = p.EB_mix
    k_mix = p.k_mix
    exp_alpha_mix = p.exp_alpha_mix
    rm6_mix = p.rm6_mix
    r_guard_mix = p.r_guard_mix
    cutoff_model = p.cutoff_model
    u_rc_mix = p.u_rc_mix
    
    pix = pos[1, i]
    piy = pos[2, i]
    piz = pos[3, i]
    type_i = types[i]
    
    x_wrapped = pix - L * floor(pix / L)
    y_wrapped = piy - L * floor(piy / L)
    z_wrapped = piz - L * floor(piz / L)
    i_cell_idx = get_cell(x_wrapped, y_wrapped, z_wrapped, L, ncell)
    
    k = ((i_cell_idx - 1) % ncell) + 1
    j = (((i_cell_idx - 1) ÷ ncell) % ncell) + 1
    i_cell = ((i_cell_idx - 1) ÷ (ncell * ncell)) + 1
    
    @inbounds for di in -1:1
        for dj in -1:1
            for dk in -1:1
                cell_i = ((i_cell - 1 + di + ncell) % ncell) + 1
                cell_j = ((j - 1 + dj + ncell) % ncell) + 1
                cell_k = ((k - 1 + dk + ncell) % ncell) + 1
                neighbor_cell = cell_index(cell_i, cell_j, cell_k, ncell)
                @inbounds pj = st.cl.head[neighbor_cell]
                while pj > 0
                    if pj != i
                        @inbounds dx = pos[1, pj] - pix
                        @inbounds dy = pos[2, pj] - piy
                        @inbounds dz = pos[3, pj] - piz
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
                            @inbounds type_j = types[pj]
                            r = sqrt(r2)
                            @inbounds r_guard = r_guard_mix[type_i, type_j]
                            if r < r_guard
                                r = r_guard
                            end
                            @inbounds exp_term = exp_alpha_mix[type_i, type_j] * Base.FastMath.exp_fast(-k_mix[type_i, type_j] * r)
                            invr = 1.0 / r
                            invr2 = invr * invr
                            invr6 = invr2 * invr2 * invr2
                            @inbounds u = EA_mix[type_i, type_j] * exp_term - EB_mix[type_i, type_j] * (rm6_mix[type_i, type_j] * invr6)
                            if cutoff_model == :shifted
                                @inbounds u -= u_rc_mix[type_i, type_j]
                            end
                            energy += u
                        end
                    end
                    @inbounds pj = st.cl.next[pj]
                end
            end
        end
    end
    
    return energy
end

function mc_trial!(st::LJState, p::Exp6Params)::Bool
    N = st.N
    L = st.L
    pos = st.pos
    
    i = rand(st.rng, 1:N)
    x_old = pos[1, i]
    y_old = pos[2, i]
    z_old = pos[3, i]
    
    Eold = local_energy(i, st, p)
    
    dx = (rand(st.rng) - 0.5) * 2.0 * p.max_disp
    dy = (rand(st.rng) - 0.5) * 2.0 * p.max_disp
    dz = (rand(st.rng) - 0.5) * 2.0 * p.max_disp
    
    @inbounds begin
        pos[1, i] = x_old + dx
        pos[2, i] = y_old + dy
        pos[3, i] = z_old + dz
    end
    
    dr = st.scratch_dr
    @inbounds begin
        dr[1] = pos[1, i]
        dr[2] = pos[2, i]
        dr[3] = pos[3, i]
    end
    wrap!(dr, L)
    @inbounds begin
        pos[1, i] = dr[1]
        pos[2, i] = dr[2]
        pos[3, i] = dr[3]
    end
    
    Enew = local_energy(i, st, p)
    ΔE = Enew - Eold
    accepted = false
    if ΔE <= 0.0 || rand(st.rng) < exp(-p.β * ΔE)
        accepted = true
    else
        @inbounds begin
            pos[1, i] = x_old
            pos[2, i] = y_old
            pos[3, i] = z_old
        end
    end
    
    return accepted
end

function sweep!(st::LJState, p::Exp6Params; rebuild_every::Int=-1)::Float64
    N = st.N
    n_accepted = 0
    if rebuild_every == -1
        rebuild_every = N
    end
    rebuild_cells!(st)
    @inbounds for trial in 1:N
        accepted = mc_trial!(st, p)
        if accepted
            n_accepted += 1
            st.accepted += 1
        end
        st.attempted += 1
        if trial % rebuild_every == 0 && trial < N
            rebuild_cells!(st)
        end
    end
    return Float64(n_accepted) / Float64(N)
end

function volume_trial!(st::LJState, p::Exp6Params; max_dlnV::Float64=0.01, Pext::Float64=1.0)::Bool
    N = st.N
    L_old = st.L
    V_old = L_old * L_old * L_old
    lnV_old = log(V_old)
    pos = st.pos
    
    pos_old = copy(pos)
    U_old = total_energy(st, p)
    
    dlnV = (rand(st.rng) - 0.5) * 2.0 * max_dlnV
    lnV_new = lnV_old + dlnV
    V_new = exp(lnV_new)
    L_new = cbrt(V_new)
    scale = L_new / L_old
    
    @inbounds for i in 1:N
        pos[1, i] = pos[1, i] * scale
        pos[2, i] = pos[2, i] * scale
        pos[3, i] = pos[3, i] * scale
    end
    
    scratch = st.scratch_dr
    @inbounds for i in 1:N
        scratch[1] = pos[1, i]
        scratch[2] = pos[2, i]
        scratch[3] = pos[3, i]
        wrap!(scratch, L_new)
        pos[1, i] = scratch[1]
        pos[2, i] = scratch[2]
        pos[3, i] = scratch[3]
    end
    
    st.L = L_new
    st.cl = CellList(N, L_new, st.cl.rc)
    rebuild_cells!(st)
    
    U_new = total_energy(st, p)
    ΔU = U_new - U_old
    ΔV = V_new - V_old
    log_acc = -p.β * (ΔU + Pext * ΔV) + N * (lnV_new - lnV_old)
    
    accepted = false
    if log_acc >= 0.0 || rand(st.rng) < exp(log_acc)
        accepted = true
    else
        copyto!(pos, pos_old)
        st.L = L_old
        st.cl = CellList(N, L_old, st.cl.rc)
        rebuild_cells!(st)
    end
    
    return accepted
end

function init_fcc_exp6(; N::Int=864, ρ::Float64=0.8, T::Float64=1.0, rc::Float64=2.5,
                      max_disp::Float64=0.1, seed::Int=1234,
                      cutoff_model::Symbol=:truncated, mixing_rule::Symbol=:lb,
                      types::Union{Vector{Int}, Nothing}=nothing,
                      rm_types::Vector{Float64}, ϵ_types::Vector{Float64}, α_types::Vector{Float64},
                      rmin_factor::Float64=0.2, use_lrc::Bool=false)
    if N <= 0
        throw(ArgumentError("N must be >= 1, got N=$N"))
    end
    n_uc = ceil(Int, N / 4)
    nx = ceil(Int, cbrt(n_uc))
    V = N / ρ
    L = cbrt(V)
    a = L / nx
    pos = zeros(Float64, 3, N)
    idx = 1
    @inbounds for k in 0:(nx-1)
        for j in 0:(nx-1)
            for i in 0:(nx-1)
                if idx > N
                    break
                end
                x0 = i * a
                y0 = j * a
                z0 = k * a
                pos[1, idx] = x0; pos[2, idx] = y0; pos[3, idx] = z0; idx += 1
                if idx > N
                    break
                end
                pos[1, idx] = x0 + 0.5 * a; pos[2, idx] = y0 + 0.5 * a; pos[3, idx] = z0; idx += 1
                if idx > N
                    break
                end
                pos[1, idx] = x0 + 0.5 * a; pos[2, idx] = y0; pos[3, idx] = z0 + 0.5 * a; idx += 1
                if idx > N
                    break
                end
                pos[1, idx] = x0; pos[2, idx] = y0 + 0.5 * a; pos[3, idx] = z0 + 0.5 * a; idx += 1
            end
            if idx > N
                break
            end
        end
        if idx > N
            break
        end
    end
    scratch = MVector{3,Float64}(0.0, 0.0, 0.0)
    @inbounds for i in 1:N
        scratch[1] = pos[1, i]
        scratch[2] = pos[2, i]
        scratch[3] = pos[3, i]
        wrap!(scratch, L)
        pos[1, i] = scratch[1]
        pos[2, i] = scratch[2]
        pos[3, i] = scratch[3]
    end
    
    if types === nothing
        types = ones(Int, N)
    else
        @assert length(types) == N "types length must match N"
    end
    cl = CellList(N, L, rc)
    rng = Xoshiro(seed)
    scratch_dr = MVector{3,Float64}(0.0, 0.0, 0.0)
    st = LJState(N, L, pos, copy(types), rng, cl, scratch_dr, 0, 0)
    rebuild_cells!(st)
    
    p = Exp6Params(rm_types=rm_types, ϵ_types=ϵ_types, α_types=α_types,
                   rc=rc, T=T, max_disp=max_disp, cutoff_model=cutoff_model,
                   mixing_rule=mixing_rule, rmin_factor=rmin_factor, use_lrc=use_lrc)
    
    return (p, st)
end

@inline function _build_cell_list(pos::Matrix{Float64}, N::Int, L::Float64, rc::Float64)
    ncell = max(1, floor(Int, L / rc))
    ncell_total = ncell * ncell * ncell
    head = zeros(Int, ncell_total)
    next = zeros(Int, N)
    cell_of = zeros(Int, N)
    @inbounds for i in 1:N
        x, y, z = pos[1, i], pos[2, i], pos[3, i]
        x_wrapped = x - L * floor(x / L)
        y_wrapped = y - L * floor(y / L)
        z_wrapped = z - L * floor(z / L)
        cell_idx = get_cell(x_wrapped, y_wrapped, z_wrapped, L, ncell)
        cell_of[i] = cell_idx
        next[i] = head[cell_idx]
        head[cell_idx] = i
    end
    return (head, next, cell_of, ncell)
end

function _energy_contrib_indices(indices::Vector{Int}, pos::Matrix{Float64}, types::Vector{Int},
                                 N::Int, L::Float64, p::Exp6Params)::Float64
    if isempty(indices)
        return 0.0
    end
    head, next, cell_of, ncell = _build_cell_list(pos, N, L, p.rc)
    rc2 = p.rc2
    energy = 0.0
    L_half = 0.5 * L
    
    @inbounds for idx in indices
        @inbounds pix = pos[1, idx]
        @inbounds piy = pos[2, idx]
        @inbounds piz = pos[3, idx]
        @inbounds type_i = types[idx]
        
        @inbounds i_cell_idx = cell_of[idx]
        k = ((i_cell_idx - 1) % ncell) + 1
        j = (((i_cell_idx - 1) ÷ ncell) % ncell) + 1
        i_cell = ((i_cell_idx - 1) ÷ (ncell * ncell)) + 1
        
        for di in -1:1
            for dj in -1:1
                for dk in -1:1
                    cell_i = ((i_cell - 1 + di + ncell) % ncell) + 1
                    cell_j = ((j - 1 + dj + ncell) % ncell) + 1
                    cell_k = ((k - 1 + dk + ncell) % ncell) + 1
                    neighbor_cell = cell_index(cell_i, cell_j, cell_k, ncell)
                    @inbounds pj = head[neighbor_cell]
                    while pj > 0
                        if pj != idx
                            @inbounds dx = pos[1, pj] - pix
                            @inbounds dy = pos[2, pj] - piy
                            @inbounds dz = pos[3, pj] - piz
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
                                @inbounds energy += exp6_pair_u_from_r2(r2, type_i, types[pj], p)
                            end
                        end
                        @inbounds pj = next[pj]
                    end
                end
            end
        end
    end
    
    # Correct double counting for pairs within indices
    if length(indices) > 1
        @inbounds for a in 1:(length(indices) - 1)
            i = indices[a]
            xi = pos[1, i]
            yi = pos[2, i]
            zi = pos[3, i]
            type_i = types[i]
            for b in (a + 1):length(indices)
                j = indices[b]
                energy -= _pair_energy_from_positions(xi, yi, zi,
                                                      pos[1, j], pos[2, j], pos[3, j],
                                                      L, L_half, p, type_i, types[j])
            end
        end
    end
    
    return energy
end

@inline function _pair_energy_from_positions(
    xi::Float64, yi::Float64, zi::Float64,
    xj::Float64, yj::Float64, zj::Float64,
    L::Float64, L_half::Float64,
    p::Exp6Params, type_i::Int, type_j::Int
)::Float64
    dx = xj - xi
    dy = yj - yi
    dz = zj - zi
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
    r2 = dx * dx + dy * dy + dz * dz
    return exp6_pair_u_from_r2(r2, type_i, type_j, p)
end
