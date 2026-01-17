"""
Reaction Ensemble Monte Carlo (RxMC) for NVT and NPT simulations.
Implements chemical reaction moves for multicomponent LJ systems.

v1: Single-site atomic species only (no molecules/CBMC).
Supports both NVT and NPT ensembles.
"""

using Random
using StaticArrays
using SpecialFunctions  # For loggamma

# Global debug counters for debug_logacc (track per direction)
const _debug_logacc_counters = Dict{Symbol, Int}()

# Global debug counter for DEBUG_REACTION (track A -> D + F attempts)
const _debug_reaction_counter = Ref(0)

# Global debug counter for REACT_DEBUG (track decomposed acceptance math)
const _react_debug_counter = Ref(0)
const _equil_accept_counter = Ref(Dict{Symbol, Int}(:forward => 0, :reverse => 0))
const _equil_debug_accept_counter = Ref(Dict{Symbol, Int}(:forward => 0, :reverse => 0))
const _exp6_audit_counter = Ref(0)

# Detailed balance audit breakdown (for A⇌D+F DB audit)
struct AcceptanceBreakdown
    direction::Symbol  # :forward or :reverse
    N_before::Vector{Int}  # Species counts before move
    N_after::Vector{Int}  # Species counts after move
    V::Float64  # Volume
    ΔU::Float64  # Energy change
    β::Float64  # Inverse temperature (1/T)
    log_combo::Float64  # Combinatorial term (fact_term)
    logq_term::Float64  # Ideal-gas partition function term
    vol_term::Float64  # Volume term
    logK_term::Float64  # Equilibrium constant term
    logGamma_term::Float64  # Standard-state chemistry term (logGamma)
    log_prop_ratio::Float64  # Proposal density ratio
    logα_total::Float64  # Total acceptance ratio (log_acc)
    log_pi_ratio::Float64  # Target measure ratio (log π(y) - log π(x))
    log_g_ratio::Float64  # Proposal density ratio (log g(y→x) - log g(x→y))
    residual::Float64  # logα_used - logα_theory, where logα_theory = log_pi_ratio + log_g_ratio (should be ≈ 0)
    logα_used::Float64  # Exact scalar passed to Metropolis (log_acc, before any min(0, ...) truncation)
    logq_terms::Vector{Tuple{String, Float64}}  # Labeled ν_i * log(q_i) terms
    insertion_mode::Symbol  # Insertion mode used for this move
    proposal_mode::Symbol  # Proposal mode used for this move
end

# DB_AUDIT print guard for logGamma-based reactions
const _db_audit_gamma_printed = Ref(Dict{Tuple{String, Symbol}, Bool}())

# Reaction instrumentation counters (for A⇌D+F specifically)
mutable struct ReactionCounters
    # Forward counters
    forward_attempted::Int
    forward_pass_geom::Int  # Passed feasibility check
    forward_pass_energy::Int  # Passed energy check (ΔU finite, not NaN/Inf)
    forward_accepted::Int  # Metropolis accepted
    forward_committed::Int  # State updated successfully
    
    # Reverse counters
    reverse_attempted::Int
    reverse_pass_geom::Int
    reverse_pass_energy::Int
    reverse_accepted::Int
    reverse_committed::Int
    
    # Rejection reasons (both directions combined)
    reject_overlap::Int  # Insertion failed due to overlap
    reject_invalid_selection::Int  # Could not select required particles
    reject_metropolis::Int  # Rejected by Metropolis criterion
    reject_nan_or_inf::Int  # ΔU or log_acc was NaN/Inf
    
    # Logging guards (to prevent spam)
    logged_mode_forward::Bool  # Has logged insertion_mode for forward direction
    logged_mode_reverse::Bool  # Has logged insertion_mode for reverse direction
end

function ReactionCounters()
    return ReactionCounters(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, false)
end

"""
    Reaction

Data structure representing a chemical reaction.

Fields:
- label: Human-readable reaction label (e.g., "2A ⇌ B")
- stoichiometry: Vector where stoichiometry[i] is the stoichiometric coefficient for species i
  Negative values are reactants, positive values are products
- logK: Equilibrium constant log(K) at the temperature of interest (deprecated, kept for compatibility)
  For Table 3.2 benchmarks, set logK=0.0 and use logq instead
- logq: Optional vector of log(q/λ³) for each species (ideal-gas partition function factors)
  If provided, reaction acceptance includes Σ ν_i * log(q_i/λ³) term
  If nothing, ideal-gas factors are not included (default for backward compatibility)
"""
struct Reaction
    label::String
    stoichiometry::Vector{Int}  # stoichiometry[species_id] = coefficient (negative=reactant, positive=product)
    logK::Float64               # log of equilibrium constant (deprecated for Table 3.2, use logq instead)
    logq::Union{Vector{Float64}, Nothing}  # log(q/λ³) per species, or nothing
    logGamma::Union{Float64, Nothing}  # standard-state chemistry term (logGamma)
end

"""
    Reaction(label, stoichiometry, logK; logq=nothing)

Convenience constructor for Reaction.
"""
function Reaction(label::String, stoichiometry::Vector{Int}, logK::Float64;
                  logq::Union{Vector{Float64}, Nothing}=nothing,
                  logGamma::Union{Float64, Nothing}=nothing)
    return Reaction(label, stoichiometry, logK, logq, logGamma)
end

"""
    count_species(st::LJState, n_species::Int)::Vector{Int}

Count particles of each species in the current state.
Returns counts[species_id] = number of particles of that species.
"""
function count_species(st::LJState, n_species::Int)::Vector{Int}
    counts = zeros(Int, n_species)
    for i in 1:st.N
        type_id = st.types[i]
        if 1 <= type_id <= n_species
            counts[type_id] += 1
        end
    end
    return counts
end

"""
    insert_particle!(st::LJState, pos_new::SVector{3,Float64}, type_new::Int)

Insert a new particle at position pos_new with type type_new.
Resizes pos matrix and types vector, and updates st.N.
Does NOT rebuild cell list (caller's responsibility).
"""
function insert_particle!(st::LJState, pos_new::SVector{3,Float64}, type_new::Int)
    st.N += 1
    # Resize pos matrix
    if size(st.pos, 2) < st.N
        # Need to grow - allocate new matrix
        pos_new_matrix = zeros(Float64, 3, st.N)
        pos_new_matrix[:, 1:st.N-1] = st.pos
        st.pos = pos_new_matrix
    else
        # Just extend the view
        st.pos = st.pos[:, 1:st.N]
    end
    st.pos[1, st.N] = pos_new[1]
    st.pos[2, st.N] = pos_new[2]
    st.pos[3, st.N] = pos_new[3]
    push!(st.types, type_new)
    return nothing
end

"""
    delete_particle!(st::LJState, idx::Int)

Delete a particle at index `idx` from the system.
To maintain contiguous arrays, the last particle is moved to `idx`.
Modifies st.pos, st.types, and st.N.
Does NOT rebuild cell list (caller's responsibility).
"""
function delete_particle!(st::LJState, idx::Int)
    if st.N == 0
        return nothing
    end
    
    # Move last particle to the position of the deleted particle
    if idx != st.N
        st.pos[1, idx] = st.pos[1, st.N]
        st.pos[2, idx] = st.pos[2, st.N]
        st.pos[3, idx] = st.pos[3, st.N]
        st.types[idx] = st.types[st.N]
    end
    
    # Shrink arrays
    st.pos = st.pos[:, 1:st.N-1]
    pop!(st.types)
    st.N -= 1
    return nothing
end

"""
    _pair_energy_from_positions(xi, yi, zi, xj, yj, zj, L, L_half, p, type_i, type_j)

Compute pair energy between two positions with minimum-image PBC.
Uses mixed parameters if multicomponent.
"""
@inline function _pair_energy_from_positions(
    xi::Float64, yi::Float64, zi::Float64,
    xj::Float64, yj::Float64, zj::Float64,
    L::Float64, L_half::Float64,
    p::LJParams, type_i::Int, type_j::Int
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
    if p.n_types > 1
        return lj_pair_u_from_r2_mixed(r2, type_i, type_j, p)
    else
        return lj_potential(r2, p)
    end
end

"""
    _energy_contrib_indices(indices, pos, types, N, L, p)::Float64

Compute the total pair energy contributions involving the given indices.
Counts each pair exactly once (pairs where both indices are in the set
are included once).
"""
function _energy_contrib_indices(
    indices::Vector{Int},
    pos::Matrix{Float64},
    types::Vector{Int},
    N::Int,
    L::Float64,
    p::LJParams
)::Float64
    if isempty(indices)
        return 0.0
    end
    L_half = 0.5 * L
    energy = 0.0
    @inbounds for idx in indices
        xi = pos[1, idx]
        yi = pos[2, idx]
        zi = pos[3, idx]
        type_i = types[idx]
        for j in 1:N
            if j != idx
                energy += _pair_energy_from_positions(
                    xi, yi, zi,
                    pos[1, j], pos[2, j], pos[3, j],
                    L, L_half, p, type_i, types[j]
                )
            end
        end
    end
    # Correct double-counting for pairs within indices
    if length(indices) > 1
        @inbounds for a in 1:(length(indices) - 1)
            i = indices[a]
            xi = pos[1, i]
            yi = pos[2, i]
            zi = pos[3, i]
            type_i = types[i]
            for b in (a + 1):length(indices)
                j = indices[b]
                energy -= _pair_energy_from_positions(
                    xi, yi, zi,
                    pos[1, j], pos[2, j], pos[3, j],
                    L, L_half, p, type_i, types[j]
                )
            end
        end
    end
    return energy
end

"""
    _energy_contrib_single(i, pos, types, N, L, p, type_override)::Float64

Compute the total pair energy of a single particle i with all others,
optionally overriding its type.
"""
function _energy_contrib_single(
    i::Int,
    pos::Matrix{Float64},
    types::Vector{Int},
    N::Int,
    L::Float64,
    p::LJParams,
    type_override::Int
)::Float64
    L_half = 0.5 * L
    xi = pos[1, i]
    yi = pos[2, i]
    zi = pos[3, i]
    energy = 0.0
    @inbounds for j in 1:N
        if j != i
            energy += _pair_energy_from_positions(
                xi, yi, zi,
                pos[1, j], pos[2, j], pos[3, j],
                L, L_half, p, type_override, types[j]
            )
        end
    end
    return energy
end

"""
    compute_com_from_indices(pos::Matrix{Float64}, indices::Vector{Int}, L::Float64)::SVector{3,Float64}

Compute center of mass (COM) of particles at given indices, using minimum image convention.
Returns COM wrapped into [0, L).
"""
function compute_com_from_indices(pos::Matrix{Float64}, indices::Vector{Int}, L::Float64)::SVector{3,Float64}
    if isempty(indices)
        return SVector{3,Float64}(0.0, 0.0, 0.0)
    end
    
    # Compute COM using minimum image convention (unwrap first particle, then relative positions)
    if length(indices) == 1
        idx = indices[1]
        x = pos[1, idx] - L * floor(pos[1, idx] / L)
        y = pos[2, idx] - L * floor(pos[2, idx] / L)
        z = pos[3, idx] - L * floor(pos[3, idx] / L)
        return SVector{3,Float64}(x, y, z)
    end
    
    # For multiple particles: use first as reference, compute relative positions
    ref_idx = indices[1]
    ref_x = pos[1, ref_idx]
    ref_y = pos[2, ref_idx]
    ref_z = pos[3, ref_idx]
    
    com_x = 0.0
    com_y = 0.0
    com_z = 0.0
    
    for idx in indices
        dx = pos[1, idx] - ref_x
        dy = pos[2, idx] - ref_y
        dz = pos[3, idx] - ref_z
        
        # Apply minimum image convention
        dx = dx - L * round(dx / L)
        dy = dy - L * round(dy / L)
        dz = dz - L * round(dz / L)
        
        com_x += dx
        com_y += dy
        com_z += dz
    end
    
    com_x = com_x / length(indices) + ref_x
    com_y = com_y / length(indices) + ref_y
    com_z = com_z / length(indices) + ref_z
    
    # Wrap COM into [0, L)
    com_x = com_x - L * floor(com_x / L)
    com_y = com_y - L * floor(com_y / L)
    com_z = com_z - L * floor(com_z / L)
    
    return SVector{3,Float64}(com_x, com_y, com_z)
end

"""
    propose_positions_com_kernel!(rng, out_positions, com, Δ, box) -> log_q

Propose insertion positions using a symmetric kernel centered at COM.
For the first product, places at COM (with kernel width Δ).
For additional products, places at COM + small symmetric displacement.

Returns log_q (log proposal density) for Metropolis-Hastings correction.

This is a symmetric proposal: p(δ) = p(-δ), so the proposal ratio cancels
in detailed balance for symmetric moves.
"""
function propose_positions_com_kernel!(rng::Xoshiro, out_positions::Vector{SVector{3,Float64}}, 
                                       com::SVector{3,Float64}, Δ::Float64, box::Float64)::Float64
    n_insert = length(out_positions)
    if n_insert == 0
        return 0.0
    end
    
    # First product: place at COM (kernel centered at COM)
    # Uniform cube kernel: p(r) = 1/(8*Δ^3) for |r - COM| < Δ in each dimension
    out_positions[1] = SVector{3,Float64}(
        com[1] + (rand(rng) - 0.5) * 2.0 * Δ,
        com[2] + (rand(rng) - 0.5) * 2.0 * Δ,
        com[3] + (rand(rng) - 0.5) * 2.0 * Δ
    )
    
    # Wrap into box
    out_positions[1] = SVector{3,Float64}(
        out_positions[1][1] - box * floor(out_positions[1][1] / box),
        out_positions[1][2] - box * floor(out_positions[1][2] / box),
        out_positions[1][3] - box * floor(out_positions[1][3] / box)
    )
    
    # Additional products: small symmetric displacements from COM
    for i in 2:n_insert
        out_positions[i] = SVector{3,Float64}(
            com[1] + (rand(rng) - 0.5) * 2.0 * Δ,
            com[2] + (rand(rng) - 0.5) * 2.0 * Δ,
            com[3] + (rand(rng) - 0.5) * 2.0 * Δ
        )
        # Wrap
        out_positions[i] = SVector{3,Float64}(
            out_positions[i][1] - box * floor(out_positions[i][1] / box),
            out_positions[i][2] - box * floor(out_positions[i][2] / box),
            out_positions[i][3] - box * floor(out_positions[i][3] / box)
        )
    end
    
    # Log proposal density: -n_insert * log(8*Δ^3) for uniform cube kernel
    return -Float64(n_insert) * log(8.0 * Δ * Δ * Δ)
end

"""
    alchemical_flip_trial!(st, p, reaction, direction) -> accepted::Bool

Perform an alchemical identity flip trial for ΔN=0 reactions (e.g., A⇌B, A⇌C).

DIRECTION-ENFORCED PROPOSAL:
- Pick one particle uniformly at random from all N_total particles (probability 1/N)
- Forward direction: only allow reactant→product flips (reject if particle is not reactant)
- Reverse direction: only allow product→reactant flips (reject if particle is not product)
- This makes reverse infeasible when N_product==0, which is correct

ACCEPTANCE FORMULA (minimal, no bias corrections):
  ln_acc = -β*ΔU + (logq_new - logq_old) + logK0

where:
- β = 1/(kT)
- ΔU = energy change from type flip
- logq_new = log(q/λ³) for new type
- logq_old = log(q/λ³) for old type
- logK0 = reaction.logK (equilibrium constant term, typically 0 for Table 3.2)

NOTE: ΔN=0 alchemical identity flips require no additional bias correction;
acceptance is pure Metropolis with ΔU + Δlogq (+logK0). This is consistent with
Braden thesis / alchemical REMC practice. No volume terms (ΔN=0), no factorial
combinatorial factors (no insertion/deletion), no proposal density ratios
(symmetric proposal).

Returns true if accepted, false if rejected.
"""
function alchemical_flip_trial!(st::LJState, p, reaction::Reaction, direction::Symbol)::Bool
    stoichiometry = reaction.stoichiometry
    n_species = length(stoichiometry)
    
    # Identify reactant and product species (must be exactly one of each for 1:1 flip)
    reactant_id = 0
    product_id = 0
    for (species_id, ν) in enumerate(stoichiometry)
        if ν < 0
            if reactant_id != 0
                return false  # Multiple reactants not supported for 1:1 flip
            end
            reactant_id = species_id
            if abs(ν) != 1
                return false  # Only 1:1 conversions supported
            end
        elseif ν > 0
            if product_id != 0
                return false  # Multiple products not supported for 1:1 flip
            end
            product_id = species_id
            if ν != 1
                return false  # Only 1:1 conversions supported
            end
        end
    end
    
    if reactant_id == 0 || product_id == 0
        return false  # Must have exactly one reactant and one product
    end
    
    # Pick one particle uniformly from ALL particles
    if st.N == 0
        return false
    end
    particle_idx = rand(st.rng, 1:st.N)
    old_type = st.types[particle_idx]
    
    # Enforce direction: forward = reactant->product, reverse = product->reactant
    if direction == :forward
        # Forward: only allow reactant->product flips
        if old_type != reactant_id
            return false  # Not a reactant, reject
        end
        new_type = product_id
    else  # :reverse
        # Reverse: only allow product->reactant flips
        if old_type != product_id
            return false  # Not a product, reject
        end
        new_type = reactant_id
    end
    
    # Compute energy change from local interactions only
    N = st.N
    L = st.L
    counts_before = count_species(st, n_species)
    U_before = _energy_contrib_single(particle_idx, st.pos, st.types, N, L, p, old_type)
    
    # Perform the type flip
    st.types[particle_idx] = new_type
    
    # Compute energy after flip (positions unchanged, only type changed)
    U_after = _energy_contrib_single(particle_idx, st.pos, st.types, N, L, p, new_type)
    ΔU = U_after - U_before

    if p.use_lrc && (p isa Exp6Params)
        counts_after = count_species(st, n_species)
        ΔU += exp6_lrc_energy_total(counts_after, L, p) - exp6_lrc_energy_total(counts_before, L, p)
    end
    
    # MINIMAL ACCEPTANCE: ln_acc = -β*ΔU + (logq_new - logq_old) + logK0 (+logGamma if provided)
    log_acc = -p.β * ΔU
    
    # Add logq difference
    if reaction.logq !== nothing
        if 1 <= old_type <= length(reaction.logq) && 1 <= new_type <= length(reaction.logq)
            log_acc += reaction.logq[new_type] - reaction.logq[old_type]
        end
    end
    
    # Add logK0 term (reaction.logK)
    log_acc += reaction.logK
    
    # Add logGamma term if provided (directional)
    if reaction.logGamma !== nothing
        log_acc += (direction == :forward ? reaction.logGamma : -reaction.logGamma)
    end
    
    # Metropolis acceptance
    accepted = false
    if log_acc >= 0.0 || rand(st.rng) < exp(log_acc)
        accepted = true
    else
        # Reject: restore old type
        st.types[particle_idx] = old_type
    end
    
    return accepted
end

"""
    reaction_trial!(st, p, reaction, direction; proposal_mode, com_kernel_Δ) -> accepted::Bool

Perform a reaction ensemble Monte Carlo trial move.

For Table 3.2 (Braden thesis), uses the following acceptance criterion:
For ΔN≠0 reactions with anchored insertion:
  ln_acc = -β * ΔU + (n_uniform_reverse - n_uniform_forward) * log(V) + Σ ν_i * log(q_i/λ³) + Σ [log(N_i!) - log((N_i + ν_i)!)]
where:
- β = 1/(kT)
- ΔU = energy change
- n_uniform_forward = number of uniform insertions in forward direction
- n_uniform_reverse = number of uniform insertions in reverse direction
- V = volume
- ν_i = stoichiometric coefficient (negative for reactants, positive for products)
- N_i = count of species i before reaction
- q_i/λ³ = ideal-gas partition function per volume for species i
Note: The volume term uses n_uniform_inserts (not ΔN) because anchored insertions have no 1/V proposal density.

For ΔN=0 reactions (e.g., A⇌B, A⇌C), uses alchemical conversion via alchemical_flip_trial!:
- Pick a particle uniformly from all particles and flip its type
- Acceptance: ln_acc = -β*ΔU + (logq_new - logq_old) + logK0
- No volume terms, no combinatorial factors, no bias corrections

For ΔN≠0 reactions:
- Uses anchored product insertion: min(n_p, n_r) products are inserted at deleted reactant COM positions
  (anchored), remaining products are inserted uniformly. Reverse moves mirror this.
- Volume term in acceptance: (n_uniform_inserts_reverse - n_uniform_inserts_forward) * log(V)
  where n_uniform_inserts counts only uniform insertions (anchored insertions have no 1/V factor).

Parameters:
- proposal_mode: (deprecated, kept for backward compatibility, ignored)
- com_kernel_Δ: (deprecated, kept for backward compatibility, ignored)

Returns: (accepted::Bool, feasible::Bool)
- feasible: true if the reaction was feasible (enough particles), false if infeasible
- accepted: true if the reaction was committed (state updated AND invariants passed), false otherwise
If feasible=false, the state is unchanged. If accepted=false, state is restored.
"""
function reaction_trial!(st::LJState, p, reaction::Reaction, direction::Symbol;
                         proposal_mode::Symbol=:uniform, com_kernel_Δ::Float64=0.05,
                         debug_reactions::Bool=false,
                         use_alchemical::Bool=true,
                         insertion_mode::Symbol=:anchored,
                         debug_logacc::Bool=false,
                         reaction_separation::Float64=1.0,
                         counters::Union{ReactionCounters, Nothing}=nothing,
                         db_audit_callback::Union{Function, Nothing}=nothing)::Tuple{Bool, Bool}
    # Thin wrapper: call internal function and return (committed, feasible)
    _met, committed, feasible, _invfail = _reaction_trial4!(st, p, reaction, direction; 
                                                             proposal_mode=proposal_mode, 
                                                             com_kernel_Δ=com_kernel_Δ,
                                                             debug_reactions=debug_reactions,
                                                             use_alchemical=use_alchemical,
                                                             insertion_mode=insertion_mode,
                                                             debug_logacc=debug_logacc,
                                                             reaction_separation=reaction_separation,
                                                             counters=counters,
                                                             db_audit_callback=db_audit_callback)
    return (committed, feasible)
end

"""
    _reaction_trial4!(st::LJState, p::LJParams, reaction::Reaction, direction::Symbol;
                      proposal_mode::Symbol=:uniform, com_kernel_Δ::Float64=0.05,
                      debug_reactions::Bool=false)::Tuple{Bool, Bool, Bool, Bool}

Internal function that returns detailed diagnostics:
Returns: (metropolis_accepted::Bool, committed::Bool, feasible::Bool, invariant_failed::Bool)
- feasible: true if the reaction was feasible (enough particles), false if infeasible
- metropolis_accepted: true if accepted by Metropolis-Hastings probability (before invariant check)
- committed: true if state update was applied AND invariants passed (only true if state actually changed)
- invariant_failed: true if state update was attempted but invariants failed (state was rolled back)
If feasible=false, the state is unchanged. If invariant_failed=true, state is restored.
"""
function _reaction_trial4!(st::LJState, p, reaction::Reaction, direction::Symbol;
                           proposal_mode::Symbol=:uniform, com_kernel_Δ::Float64=0.05,
                           debug_reactions::Bool=false,
                           use_alchemical::Bool=true,
                           insertion_mode::Symbol=:anchored,
                           debug_logacc::Bool=false,
                           reaction_separation::Float64=1.0,
                           counters::Union{ReactionCounters, Nothing}=nothing,
                           db_audit_callback::Union{Function, Nothing}=nothing)::Tuple{Bool, Bool, Bool, Bool}
    # Note: proposal_mode and com_kernel_Δ are deprecated (kept for backward compatibility)
    # All ΔN≠0 reactions now use anchored insertion
    
    # Validate insertion_mode
    allowed_modes = [:anchored, :standard_remc, :uniform_only]
    if !(insertion_mode in allowed_modes)
        error("Invalid insertion_mode: $insertion_mode. Must be one of: $allowed_modes")
    end
    
    # Debug logging: log insertion_mode once per direction (guarded by REACT_DEBUG, NOT DB_AUDIT)
    # DB_AUDIT mode prints insertion_mode in the audit breakdown instead
    # Note: counters is only created for A⇌D+F, so we check reaction.label first
    react_debug_enabled = get(ENV, "REACT_DEBUG", "0") == "1"
    db_audit_enabled = get(ENV, "DB_AUDIT", "0") == "1"
    if react_debug_enabled && !db_audit_enabled && reaction.label == "A⇌D+F"
        # Log once per direction using counters guard (counters should exist for A⇌D+F)
        if counters !== nothing
            if direction == :forward && !counters.logged_mode_forward
                println("REACTION_INSERTION_MODE_USED: insertion_mode=$insertion_mode, direction=$direction, reaction=$(reaction.label), proposal_mode=$proposal_mode")
                counters.logged_mode_forward = true
            elseif direction == :reverse && !counters.logged_mode_reverse
                println("REACTION_INSERTION_MODE_USED: insertion_mode=$insertion_mode, direction=$direction, reaction=$(reaction.label), proposal_mode=$proposal_mode")
                counters.logged_mode_reverse = true
            end
        end
    end
    stoichiometry = reaction.stoichiometry
    n_species = length(stoichiometry)
    L = st.L
    V = L * L * L
    
    # Count current species
    counts_before = count_species(st, n_species)
    
    # Determine which species are reactants and products for this direction
    Δn_forward = sum(stoichiometry)  # Change in total particle number for forward reaction
    
    # Handle ΔN=0 reactions
    if Δn_forward == 0
        # Count reactant/product species for this direction
        n_reactant_species = 0
        n_product_species = 0
        for ν in stoichiometry
            if ν < 0
                n_reactant_species += 1
            elseif ν > 0
                n_product_species += 1
            end
        end
        # Only allow alchemical flip for 1↔1 reactions; otherwise fall through to general path
        if use_alchemical && n_reactant_species == 1 && n_product_species == 1
            # Use alchemical flip (current optimized path)
            accepted = alchemical_flip_trial!(st, p, reaction, direction)
            return (accepted, accepted, true, false)  # (metropolis_accepted, committed, feasible, invariant_failed)
        elseif !use_alchemical && n_reactant_species == 1 && n_product_species == 1
            # Use uniform delete+insert (Smith-Tríska path)
            # Compute νeff for direction
            νeff = direction == :forward ? stoichiometry : [-ν for ν in stoichiometry]
            
            # Check feasibility
            for (species_id, ν) in enumerate(νeff)
                if ν < 0  # Consumed in this direction
                    if counts_before[species_id] < -ν
                        return (false, false, false, false)  # Infeasible
                    end
                end
            end
            
            # Store state
            N_before = st.N
            pos_old = copy(st.pos)
            types_old = copy(st.types)
            
            # Delete one reactant (ΔN=0 means one reactant, one product)
            species_to_delete = 0
            species_to_insert = 0
            for (species_id, ν) in enumerate(νeff)
                if ν < 0
                    species_to_delete = species_id
                elseif ν > 0
                    species_to_insert = species_id
                end
            end
            
            # Find particles of the species to delete
            species_indices = Int[]
            for i in 1:st.N
                if st.types[i] == species_to_delete
                    push!(species_indices, i)
                end
            end
            
            if length(species_indices) == 0
                st.N = N_before
                st.pos = pos_old
                st.types = types_old
                return (false, false, false, false)  # Should not happen after feasibility check
            end
            
            # Randomly select one particle to delete
            particle_to_delete = rand(st.rng, species_indices)
            
            # Energy contribution from deleted particle (before deletion)
            old_contrib = _energy_contrib_indices([particle_to_delete], pos_old, types_old, N_before, L, p)
            
            # Delete the particle
            delete_particle!(st, particle_to_delete)
            
            # Insert one product uniformly
            pos_insert = SVector{3,Float64}(rand(st.rng) * L, rand(st.rng) * L, rand(st.rng) * L)
            insert_particle!(st, pos_insert, species_to_insert)
            inserted_index = st.N
            
            # Compute energy change from local contributions
            new_contrib = _energy_contrib_indices([inserted_index], st.pos, st.types, st.N, L, p)
            ΔU = new_contrib - old_contrib
            
            # Count species after
            counts_after = count_species(st, n_species)
            
            # Smith-Tríska acceptance: log_acc = -βΔU + ΔN*log(V) + Σνeff[i]*logq[i] + Σ[log(N_i!) - log((N_i+νeff[i])!)] + logK
            ΔN = sum(νeff)  # Should be 0 for ΔN=0 reactions, but compute for formula
            log_acc = -p.β * ΔU
            log_acc += Float64(ΔN) * log(V)  # Should be 0, but included for formula correctness
            
            # logq terms
            if reaction.logq !== nothing
                for (species_id, ν) in enumerate(νeff)
                    if ν != 0 && 1 <= species_id <= length(reaction.logq)
                        log_acc += Float64(ν) * reaction.logq[species_id]
                    end
                end
            end
            
            # Combinatorial terms
            for (species_id, ν) in enumerate(νeff)
                if ν != 0
                    N_i_before = counts_before[species_id]
                    N_i_after = counts_after[species_id]
                    if N_i_before >= 0
                        log_acc += loggamma(Float64(N_i_before) + 1.0)
                    end
                    if N_i_after >= 0
                        log_acc -= loggamma(Float64(N_i_after) + 1.0)
                    end
                end
            end
            
            # logK term
            log_acc += reaction.logK
            
            # logGamma term (directional)
            if reaction.logGamma !== nothing
                log_acc += (direction == :forward ? reaction.logGamma : -reaction.logGamma)
            end
            
            # Metropolis acceptance
            metropolis_accepted = false
            if log_acc >= 0.0 || rand(st.rng) < exp(log_acc)
                metropolis_accepted = true
            else
                # Reject: restore state
                st.N = N_before
                st.pos = pos_old
                st.types = types_old
                return (false, false, true, false)
            end
            
            # Deterministic sanity check after commit
            counts_after_check = count_species(st, n_species)
            if st.N != N_before + ΔN
                error("ΔN=0 uniform delete+insert: ΔN mismatch. Expected N_after=$(N_before + ΔN), got $(st.N), direction=$direction, νeff=$νeff")
            end
            for (species_id, ν) in enumerate(νeff)
                if ν != 0
                    expected_count = counts_before[species_id] + ν
                    actual_count = counts_after_check[species_id]
                    if actual_count != expected_count
                        error("ΔN=0 uniform delete+insert: Species $species_id count mismatch. Expected $expected_count, got $actual_count, direction=$direction, νeff=$νeff, counts_before=$counts_before, counts_after=$counts_after_check")
                    end
                end
            end
            
            # Rebuild cell list after accepted move
            st.cl = CellList(st.N, L, p.rc)
            rebuild_cells!(st)
            return (metropolis_accepted, metropolis_accepted, true, false)
        end
    end
    
    # UNIFIED STOICHIOMETRY: νeff is the effective stoichiometry for this direction
    # Forward: νeff = stoichiometry (ν_i < 0 consumed, ν_i > 0 produced)
    # Reverse: νeff = -stoichiometry (flip signs: what was produced is now consumed, what was consumed is now produced)
    νeff = direction == :forward ? stoichiometry : [-ν for ν in stoichiometry]
    
    # Check feasibility for ΔN≠0 reactions using νeff
    # Feasibility: for each species i where νeff[i] < 0 (being consumed), need N_i >= -νeff[i]
    feasible = true
    for (species_id, ν) in enumerate(νeff)
        if ν < 0  # This species is consumed in this direction
            if counts_before[species_id] < -ν
                # Increment attempted counter even for infeasible moves
                if counters !== nothing && reaction.label == "A⇌D+F"
                    if direction == :forward
                        counters.forward_attempted += 1
                    else
                        counters.reverse_attempted += 1
                    end
                end
                return (false, false, false, false)  # (metropolis_accepted=false, committed=false, feasible=false, invariant_failed=false)
            end
        end
    end
    
    # Increment attempted counter for feasible moves
    if counters !== nothing && reaction.label == "A⇌D+F"
        if direction == :forward
            counters.forward_attempted += 1
        else
            counters.reverse_attempted += 1
        end
    end
    
    # Increment pass_geom counter (feasibility check passed)
    if counters !== nothing && reaction.label == "A⇌D+F"
        if direction == :forward
            counters.forward_pass_geom += 1
        else
            counters.reverse_pass_geom += 1
        end
    end
    
    # Debug logging: track feasible proposals per direction
    if debug_logacc && feasible
        if !haskey(_debug_logacc_counters, direction)
            _debug_logacc_counters[direction] = 0
        end
        if _debug_logacc_counters[direction] < 50
            _debug_logacc_counters[direction] += 1
        end
    end
    
    # Assert feasibility before state modification (fail fast if bypassed)
    for i in 1:n_species
        if νeff[i] < 0
            @assert counts_before[i] >= -νeff[i] "Feasibility check bypassed! Species $i: counts_before=$(counts_before[i]), νeff=$(νeff[i]), counts_before=$counts_before, νeff=$νeff"
        end
    end
    
    # Store old state for rejection
    N_before = st.N
    pos_old = copy(st.pos)
    types_old = copy(st.types)
    
    # ΔN ≠ 0: use insertion/deletion with anchored product insertion
    # UNIFIED LOGIC using νeff: delete where νeff < 0, insert where νeff > 0
    particles_to_delete = Int[]
    deleted_positions = Vector{SVector{3,Float64}}()
    
    # Delete particles for species where νeff < 0 (consumed in this direction)
    for (species_id, ν) in enumerate(νeff)
        if ν < 0
            n_delete = -ν
            # Find all particles of this species
            species_indices = Int[]
            for i in 1:st.N
                if st.types[i] == species_id
                    push!(species_indices, i)
                end
            end
            # Randomly select n_delete particles to delete
            if length(species_indices) < n_delete
                # This should not happen if feasibility check passed, but handle gracefully
                if counters !== nothing && reaction.label == "A⇌D+F"
                    counters.reject_invalid_selection += 1
                end
                st.N = N_before
                st.pos = pos_old
                st.types = types_old
                return (false, false, false, false)  # (metropolis_accepted=false, committed=false, feasible=false, invariant_failed=false)
            end
            selected = Int[]
            available = copy(species_indices)
            for _ in 1:n_delete
                idx_choice = rand(st.rng, 1:length(available))
                push!(selected, available[idx_choice])
                deleteat!(available, idx_choice)
            end
            append!(particles_to_delete, selected)
        end
    end
    
    # Store positions before deletion (need to do this before sorting indices)
    for idx in particles_to_delete
        push!(deleted_positions, SVector{3,Float64}(pos_old[1, idx], pos_old[2, idx], pos_old[3, idx]))
    end
    
    # Energy contributions from deleted particles (before deletion)
    old_contrib = _energy_contrib_indices(particles_to_delete, pos_old, types_old, N_before, L, p)
    
    # Delete in reverse sorted order to avoid index shifts
    sort!(particles_to_delete, rev=true)
    for idx in particles_to_delete
        delete_particle!(st, idx)
    end
    
    # Count total particles deleted and to insert
    n_r = length(deleted_positions)  # Number deleted (reactants in this direction)
    n_p = 0  # Number to insert (products in this direction)
    for (species_id, ν) in enumerate(νeff)
        if ν > 0
            n_p += ν
        end
    end
    
    # Build list of positions for insertion
    # Smith–Tríska REMC: uniform-box insertion for extra particles.
    # For A⇌D+F specifically (MANDATORY):
    #   Forward (A→D+F): D at deleted A position (deterministic), F uniformly in box (1/V)
    #   Reverse (D+F→A): A at deleted D position (deterministic)
    positions_to_insert = Vector{SVector{3,Float64}}()
    
    if reaction.label == "A⇌D+F" && n_p == 2 && n_r == 1 && direction == :forward
        # Forward (A→D+F): D at deleted A position (deterministic), F uniform in box
        r0 = deleted_positions[1]  # Deleted A position
        # D placed at r0 (deterministic)
        push!(positions_to_insert, SVector{3,Float64}(r0[1], r0[2], r0[3]))
        # F placed uniformly in box (density 1/V)
        push!(positions_to_insert, SVector{3,Float64}(rand(st.rng) * L, rand(st.rng) * L, rand(st.rng) * L))
    elseif reaction.label == "A⇌D+F" && n_p == 1 && n_r == 2 && direction == :reverse
        # Reverse (D+F→A): A at deleted D position (deterministic)
        # D is deleted first (species_id=2), so deleted_positions[1] is D's position
        rD = deleted_positions[1]  # Deleted D position
        # A placed at rD (deterministic)
        push!(positions_to_insert, SVector{3,Float64}(rD[1], rD[2], rD[3]))
    else
        if insertion_mode == :anchored
            # Anchored insertion: place up to min(n_p, n_r) products at deleted positions,
            # remaining products uniformly in the box.
            n_anchor = min(n_p, n_r)
            # Anchor using the first deleted positions (order is deterministic)
            for i in 1:n_anchor
                r0 = deleted_positions[i]
                push!(positions_to_insert, SVector{3,Float64}(r0[1], r0[2], r0[3]))
            end
            # Remaining products inserted uniformly
            n_uniform = n_p - n_anchor
            for _ in 1:n_uniform
                push!(positions_to_insert, SVector{3,Float64}(rand(st.rng) * L, rand(st.rng) * L, rand(st.rng) * L))
            end
        else
            # Standard REMC / uniform-only: uniform insertion for ALL products.
            for (species_id, ν) in enumerate(νeff)
                if ν > 0
                    for _ in 1:ν
                        push!(positions_to_insert, SVector{3,Float64}(rand(st.rng) * L, rand(st.rng) * L, rand(st.rng) * L))
                    end
                end
            end
        end
    end
    
    # DEBUG_REACTION: Track A -> D + F reactions
    debug_reaction_enabled = get(ENV, "DEBUG_REACTION", "false") == "true"
    debug_reaction_data = nothing
    if debug_reaction_enabled && reaction.label == "A⇌D+F" && direction == :forward && feasible && _debug_reaction_counter[] < 50 && n_p >= 2
        _debug_reaction_counter[] += 1
        
        # Identify D and F positions (for A -> D + F, νeff = [-1, 1, 1] for [A, D, F])
        # positions_to_insert[1] is D (species 2), positions_to_insert[2] is F (species 3)
        pos_D = positions_to_insert[1]
        pos_F = positions_to_insert[2]
        
        # Compute rmin_D and rmin_F to existing particles (after deletion, before insertion)
        # Loop over all particles directly (no cell list needed for simple min distance)
        dr_temp = MVector{3,Float64}(0.0, 0.0, 0.0)
        rmin_D_sq = Inf
        rmin_F_sq = Inf
        for i in 1:st.N
            # Distance from D
            dr_temp[1] = st.pos[1, i] - pos_D[1]
            dr_temp[2] = st.pos[2, i] - pos_D[2]
            dr_temp[3] = st.pos[3, i] - pos_D[3]
            minimum_image!(dr_temp, L)
            r2 = dr_temp[1]*dr_temp[1] + dr_temp[2]*dr_temp[2] + dr_temp[3]*dr_temp[3]
            if r2 < rmin_D_sq
                rmin_D_sq = r2
            end
            
            # Distance from F
            dr_temp[1] = st.pos[1, i] - pos_F[1]
            dr_temp[2] = st.pos[2, i] - pos_F[2]
            dr_temp[3] = st.pos[3, i] - pos_F[3]
            minimum_image!(dr_temp, L)
            r2 = dr_temp[1]*dr_temp[1] + dr_temp[2]*dr_temp[2] + dr_temp[3]*dr_temp[3]
            if r2 < rmin_F_sq
                rmin_F_sq = r2
            end
        end
        rmin_D = sqrt(rmin_D_sq)
        rmin_F = sqrt(rmin_F_sq)
        
        # Compute r_DF (distance between D and F)
        dr_DF = MVector{3,Float64}(pos_F[1] - pos_D[1], pos_F[2] - pos_D[2], pos_F[3] - pos_D[3])
        minimum_image!(dr_DF, L)
        r_DF = sqrt(dr_DF[1]*dr_DF[1] + dr_DF[2]*dr_DF[2] + dr_DF[3]*dr_DF[3])
        
        # Check if positions are aliased (same reference)
        pos_D_aliased = (pos_D === pos_F)
        
        # Store debug data (will print after computing acceptance terms)
        debug_reaction_data = (pos_D, pos_F, r_DF, rmin_D, rmin_F, pos_D_aliased)
    end
    
    # Insert products in stoichiometric order (using νeff)
    inserted_indices = Int[]
    insert_idx = 1
    for (species_id, ν) in enumerate(νeff)
        if ν > 0
            for _ in 1:ν
                insert_particle!(st, positions_to_insert[insert_idx], species_id)
                push!(inserted_indices, st.N)
                insert_idx += 1
            end
        end
    end
    
    # Compute energy change from local contributions
    new_contrib = _energy_contrib_indices(inserted_indices, st.pos, st.types, st.N, L, p)
    ΔU = new_contrib - old_contrib
    
    # Count species after reaction
    counts_after = count_species(st, n_species)

    # Exp-6 tail corrections (composition-dependent)
    if p.use_lrc
        if p isa Exp6Params
            lrc_before = exp6_lrc_energy_total(counts_before, L, p)
            lrc_after = exp6_lrc_energy_total(counts_after, L, p)
            ΔU += (lrc_after - lrc_before)
        else
            ΔU += Float64(st.N - N_before) * p.lrc_u_per_particle
        end
    end

    # Check for NaN/Inf in energy
    energy_valid = isfinite(ΔU)
    if counters !== nothing && reaction.label == "A⇌D+F"
        if direction == :forward
            if energy_valid
                counters.forward_pass_energy += 1
            else
                counters.reject_nan_or_inf += 1
            end
        else
            if energy_valid
                counters.reverse_pass_energy += 1
            else
                counters.reject_nan_or_inf += 1
            end
        end
    end
    
    # Smith–Tríska acceptance formula (strict)
    # logα = -β * ΔU + Σ ν_i * log(q_i) + log(∏ N_i! / (N_i + ν_i)!) + (Σ ν_i) * log(V) + logGamma
    ΔN = sum(νeff)
    ΔU_term = -p.β * ΔU
    log_acc = ΔU_term
    
    # Volume term: (sum ν_i) * log(V)
    vol_term = Float64(ΔN) * log(V)  # ΔN = sum(νeff) = Δν
    log_acc += vol_term
    
    # Ideal-gas partition function terms: Σ ν_i * log(q_i)
    logq_term = 0.0
    logq_terms = Vector{Tuple{String, Float64}}()
    if reaction.logq !== nothing
        for (species_id, ν) in enumerate(νeff)
            if ν != 0 && 1 <= species_id <= length(reaction.logq)
                term = Float64(ν) * reaction.logq[species_id]
                logq_term += term
                push!(logq_terms, ("nu_logq_species_$species_id", term))
            end
        end
    end
    log_acc += logq_term
    
    # Combinatorial factor: Σ [log(N_i!) - log((N_i + ν_i)!)]
    # Using loggamma: log(n!) = loggamma(n+1)
    # N_i are counts BEFORE the move (use counts_before, not counts_after)
    fact_term = 0.0
    for (species_id, ν) in enumerate(νeff)
        if ν != 0
            N_i_before = counts_before[species_id]
            # log(N_i!) - log((N_i + ν_i)!)
            if N_i_before >= 0
                fact_term += loggamma(Float64(N_i_before) + 1.0)
            end
            if (N_i_before + ν) >= 0
                fact_term -= loggamma(Float64(N_i_before + ν) + 1.0)
            end
        end
    end
    log_acc += fact_term
    
    # Standard-state chemistry term (logGamma, directional)
    logGamma_term = reaction.logGamma === nothing ? 0.0 :
        (direction == :forward ? reaction.logGamma : -reaction.logGamma)
    log_acc += logGamma_term
    
    # No proposal density corrections, no logK term for Smith–Tríska REMC
    logK_term = 0.0
    log_prop_ratio = 0.0
    
    # Check for NaN/Inf in log_acc
    log_acc_valid = isfinite(log_acc)
    if !log_acc_valid && counters !== nothing && reaction.label == "A⇌D+F"
        counters.reject_nan_or_inf += 1
    end
    
    # REACT_DEBUG: Print decomposed acceptance math for first ~200 attempts
    react_debug_enabled = get(ENV, "REACT_DEBUG", "0") == "1"
    if react_debug_enabled && _react_debug_counter[] < 200
        _react_debug_counter[] += 1
        # Decompose acceptance into terms
        log_boltz = -p.β * ΔU  # Boltzmann term
        # vol_term, logq_term, fact_term already computed above
        logα_total = log_acc
        
        # A⇌D+F specific term print (first 50 attempts only)
        if reaction.label == "A⇌D+F" && _react_debug_counter[] <= 50
            println("REACT_DEBUG_TERM[$(_react_debug_counter[])]: ΔU=$(ΔU) log_boltz=$(log_boltz) log_combo=$(fact_term) logq=$(logq_term) vol=$(vol_term) logα=$(logα_total)")
            
            # Non-energy acceptance bias check
            nonenergy_forward = fact_term + logq_term + vol_term
            V = st.L^3
            ln_V = log(V)
            Δν = ΔN  # ΔN = sum(νeff) for this direction
            println("REACT_DEBUG_NONENERGY[$(_react_debug_counter[])]: dir=$direction, nonenergy=$nonenergy_forward, V=$V, ln_V=$ln_V, Δν=$Δν, vol_term=$vol_term")
        end
        
        # Equilibration acceptance-term logging for A⇌D+F (only for accepted moves)
        equil_log_accept_enabled = get(ENV, "EQUIL_LOG_ACCEPT", "0") == "1"
        if equil_log_accept_enabled && reaction.label == "A⇌D+F" && metropolis_accepted && energy_valid && log_acc_valid
            # This will be checked after metropolis_accepted is determined
            # We'll print after the acceptance decision
        end
        
        # Determine rejection reason (if rejected)
        rejection_reason = "none"
        if !energy_valid || !log_acc_valid
            rejection_reason = "nan_or_inf"
        end
        
        # Format counts
        counts_str = "($(join(counts_before, ",")))"
        counts_after_str = "($(join(counts_after, ",")))"
        
        accepted_status_preview = "pending"
        rejection_reason_final = rejection_reason
        if !energy_valid || !log_acc_valid
            accepted_status_preview = "rejected_nan_inf"
            rejection_reason_final = "nan_or_inf"
        end
        
        println("REACT_DEBUG[$(_react_debug_counter[])]: direction=$direction, current=$counts_str, proposed=$counts_after_str, ΔU=$(round(ΔU, digits=6)), log_boltz=$(round(log_boltz, digits=6)), log_combo=$(round(fact_term, digits=6)), logq_term=$(round(logq_term, digits=6)), vol_term=$(round(vol_term, digits=6)), logα_total=$(round(logα_total, digits=6)), rejected=$rejection_reason_final, status=$accepted_status_preview")
        
        # Detailed balance antisymmetry check for A⇌D+F (first 50 attempts only)
        if reaction.label == "A⇌D+F" && _react_debug_counter[] <= 50 && feasible && energy_valid && log_acc_valid
            logR_forward = logα_total
            
            # For detailed balance, logR_reverse should be the negation of logR_forward (term-by-term)
            # Negate each forward term to get the reverse term
            log_boltz_rev = -log_boltz
            log_combo_rev = -fact_term  # fact_term is the combinatorial term
            logq_term_rev = -logq_term
            vol_term_rev = -vol_term
            
            logR_reverse_check = log_boltz_rev + log_combo_rev + logq_term_rev + vol_term_rev
            
            # Check detailed balance: logR_forward + logR_reverse_check should be ≈ 0
            db_sum = logR_forward + logR_reverse_check
            db_tolerance = 1e-10
            
            println("REACT_DEBUG_DB[$(_react_debug_counter[])]: sum=$(round(db_sum, digits=10))")
            
            if abs(db_sum) > db_tolerance
                # Print term mismatches (each term + its negation should sum to 0)
                boltz_mismatch = log_boltz + log_boltz_rev
                combo_mismatch = fact_term + log_combo_rev
                q_mismatch = logq_term + logq_term_rev
                vol_mismatch = vol_term + vol_term_rev
                
                println("REACT_DEBUG_DB[$(_react_debug_counter[])]: term_mismatches boltz=$(round(boltz_mismatch, digits=10)) combo=$(round(combo_mismatch, digits=10)) q=$(round(q_mismatch, digits=10)) vol=$(round(vol_mismatch, digits=10))")
            end
            
            # Paired move detailed balance check: compute actual reverse acceptance from current state
            # This checks that logα_forward(state→state') + logα_reverse(state'→state) ≈ 0
            # where state' is the current state after the forward move
            # Note: This runs before Metropolis decision, so we compute for all feasible forward moves
            if direction == :forward
                # Compute what the reverse acceptance would be from the current state (state')
                # Reverse move: D+F → A
                # Need to compute reverse acceptance using counts_after as counts_before for reverse
                counts_before_reverse = counts_after  # After forward = before reverse
                νeff_reverse = [-ν for ν in νeff]
                
                # Check reverse feasibility
                reverse_feasible = true
                for (species_id, ν) in enumerate(νeff_reverse)
                    if ν < 0 && counts_before_reverse[species_id] < -ν
                        reverse_feasible = false
                        break
                    end
                end
                
                if reverse_feasible
                    # Compute reverse energy change (negate forward)
                    ΔU_reverse = -ΔU
                    log_boltz_reverse = -p.β * ΔU_reverse  # = p.β * ΔU = -log_boltz
                    
                    # Reverse volume term: use Δν * log(V) (symmetric with forward)
                    ΔN_reverse = sum(νeff_reverse)  # = -ΔN
                    vol_term_reverse = Float64(ΔN_reverse) * log(V)
                    
                    # Reverse logq term
                    logq_term_reverse = 0.0
                    if reaction.logq !== nothing
                        for (species_id, ν) in enumerate(νeff_reverse)
                            if ν != 0 && 1 <= species_id <= length(reaction.logq)
                                logq_term_reverse += Float64(ν) * reaction.logq[species_id]
                            end
                        end
                    end
                    
                    # Reverse combinatorial term (use counts_before_reverse)
                    fact_term_reverse = 0.0
                    for (species_id, ν) in enumerate(νeff_reverse)
                        if ν != 0
                            N_i_before_reverse = counts_before_reverse[species_id]
                            if N_i_before_reverse >= 0
                                fact_term_reverse += loggamma(Float64(N_i_before_reverse) + 1.0)
                            end
                            if (N_i_before_reverse + ν) >= 0
                                fact_term_reverse -= loggamma(Float64(N_i_before_reverse + ν) + 1.0)
                            end
                        end
                    end
                    
                    logα_reverse_paired = log_boltz_reverse + vol_term_reverse + logq_term_reverse + fact_term_reverse
                    
                    db_paired_sum = logR_forward + logα_reverse_paired
                    
                    println("REACT_DEBUG_PAIRED[$(_react_debug_counter[])]: logα_forward=$(round(logR_forward, digits=10)), logα_reverse_paired=$(round(logα_reverse_paired, digits=10)), sum=$(round(db_paired_sum, digits=10))")
                    
                    if abs(db_paired_sum) > db_tolerance
                        println("REACT_DEBUG_PAIRED[$(_react_debug_counter[])]: VIOLATION! Forward and reverse acceptance do not sum to zero.")
                    end
                end
            end
        end
    end
    
    # DEBUG_REACTION: Print debug info for A -> D + F
    if debug_reaction_data !== nothing
        pos_D, pos_F, r_DF, rmin_D, rmin_F, pos_D_aliased = debug_reaction_data
        # Compute acceptance (before MH decision, so we can print it)
        accepted_preview = (log(rand(st.rng)) < min(0.0, log_acc))
        beta_val = p.β
        println("DEBUG_REACTION: counts_before=$counts_before, counts_after=$counts_after, vol_term=$(round(vol_term, digits=6)), fact_term=$(round(fact_term, digits=6)), logq_term=$(round(logq_term, digits=6)), ΔU_term=$(round(ΔU_term, digits=6)), log_acc=$(round(log_acc, digits=6)), ΔU=$(round(ΔU, digits=6)), beta=$beta_val, pos_D=($(round(pos_D[1], digits=6)), $(round(pos_D[2], digits=6)), $(round(pos_D[3], digits=6))), pos_F=($(round(pos_F[1], digits=6)), $(round(pos_F[2], digits=6)), $(round(pos_F[3], digits=6))), r_DF=$(round(r_DF, digits=6)), rmin_D=$(round(rmin_D, digits=6)), rmin_F=$(round(rmin_F, digits=6)), pos_aliased=$pos_D_aliased, accepted=$accepted_preview")
        # Note: actual acceptance decision happens below
    end
    
    # Debug logging: print first 50 feasible proposals per direction
    if debug_logacc && feasible && _debug_logacc_counters[direction] <= 50
        accepted = (log(rand(st.rng)) < min(0.0, log_acc))
        println("DEBUG_LOGACC: direction=$direction, νeff=$νeff, counts_before=$counts_before, ΔU_term=$(round(ΔU_term, digits=6)), vol_term=$(round(vol_term, digits=6)), logq_term=$(round(logq_term, digits=6)), fact_term=$(round(fact_term, digits=6)), log_acc=$(round(log_acc, digits=6)), accepted=$accepted")
        if !accepted
            # If we rejected, restore state now (normally done below)
            st.N = N_before
            st.pos = pos_old
            st.types = types_old
            return (false, false, true, false)
        end
    end
    
    # Smith–Tríska acceptance decision (BEFORE invariant check)
    # Acceptance: log(rand()) < min(0, logα)
    metropolis_accepted = false
    u_metropolis = 1.0  # Default value (will be set below)
    if !energy_valid || !log_acc_valid
        # Energy or log_acc is NaN/Inf, reject immediately
        # Note: already counted in reject_nan_or_inf above
        st.N = N_before
        st.pos = pos_old
        st.types = types_old
        return (false, false, true, false)  # (metropolis_accepted=false, committed=false, feasible=true, invariant_failed=false)
    else
        u_metropolis = rand(st.rng)
        log_u = log(u_metropolis)
        if log_u < min(0.0, log_acc)
            metropolis_accepted = true
        end
    end
    
    # EXP-6 audit (first 20 attempted reactions)
    if get(ENV, "EXP6_AUDIT", "0") == "1" && (p isa Exp6Params) && _exp6_audit_counter[] < 20
        _exp6_audit_counter[] += 1
        println("EXP6_AUDIT[$(_exp6_audit_counter[])]: direction=$direction counts_before=$counts_before ΔU=$(round(ΔU, digits=6)) " *
                "(-βΔU)=$(round(ΔU_term, digits=6)) logq=$(round(logq_term, digits=6)) " *
                "logcombo=$(round(fact_term, digits=6)) vol=$(round(vol_term, digits=6)) " *
                "logGamma=$(round(logGamma_term, digits=6)) logK=$(round(logK_term, digits=6)) logα=$(round(log_acc, digits=6)) " *
                "log(u)=$(round(log_u, digits=6)) accepted=$metropolis_accepted")
    end
    
    if metropolis_accepted
        # Count as accepted
        if counters !== nothing && reaction.label == "A⇌D+F"
            if direction == :forward
                counters.forward_accepted += 1
            else
                counters.reverse_accepted += 1
            end
        end
        # Equilibration acceptance-term logging for A⇌D+F (only for accepted moves, first 10 per direction)
        equil_log_accept_enabled = get(ENV, "EQUIL_LOG_ACCEPT", "0") == "1"
        if equil_log_accept_enabled && reaction.label == "A⇌D+F" && energy_valid && log_acc_valid
            if _equil_accept_counter[][direction] < 10
                _equil_accept_counter[][direction] += 1
                # Compute terms (already computed above)
                log_boltz = -p.β * ΔU
                # vol_term, logq_term, fact_term already computed above
                println("EQUIL_ACCEPT_TERM[$(_equil_accept_counter[][direction])][$direction]: ΔU=$(round(ΔU, digits=6)), -βΔU=$(round(log_boltz, digits=6)), log_combo=$(round(fact_term, digits=6)), logq_term=$(round(logq_term, digits=6)), vol_term=$(round(vol_term, digits=6)), logα_total=$(round(log_acc, digits=6))")
            end
        end
        
        # Equilibration debug acceptance breakdown (first 5 accepted moves per direction, all reactions)
        equil_debug_accept_enabled = get(ENV, "EQUIL_DEBUG_ACCEPT", "0") == "1"
        if equil_debug_accept_enabled && energy_valid && log_acc_valid
            if _equil_debug_accept_counter[][direction] < 5
                _equil_debug_accept_counter[][direction] += 1
                # Compute terms (already computed above)
                log_boltz = -p.β * ΔU
                V = st.L^3
                Δν = ΔN  # ΔN = sum(νeff) for this direction
                
                # Use the actual Uniform(0,1) draw from Metropolis acceptance decision
                log_u = log(u_metropolis)  # log of the actual Uniform(0,1) draw used in Metropolis acceptance
                
                # Format counts_before for log_combo display
                counts_str = join(counts_before, ",")
                
                println("EQUIL_DEBUG_ACCEPT[$(_equil_debug_accept_counter[][direction])][$direction]:")
                println("  ΔU=$(round(ΔU, digits=6))")
                println("  -βΔU=$(round(log_boltz, digits=6))")
                println("  log_combo=$(round(fact_term, digits=6)) (N_before=[$counts_str])")
                println("  logq_term=$(round(logq_term, digits=6))")
                println("  vol_term=$(round(vol_term, digits=6)) (V=$(round(V, digits=4)), Δν=$Δν)")
                # logK_term/log_prop_ratio are zero for Smith–Tríska REMC
                println("  logα_total=$(round(log_acc, digits=6))")
                println("  log(u)=$(round(log_u, digits=6)) (u=$(round(u_metropolis, digits=6)))")
            end
        end
        
        # REACT_DEBUG: Update status after MH acceptance
        react_debug_enabled = get(ENV, "REACT_DEBUG", "0") == "1"
        if react_debug_enabled && _react_debug_counter[] <= 200 && _react_debug_counter[] > 0
            println("REACT_DEBUG[$(_react_debug_counter[])]: ACCEPTED by Metropolis (logα=$(round(log_acc, digits=6)))")
        end
    else
        # MH rejected: restore old state immediately
        if counters !== nothing && reaction.label == "A⇌D+F"
            counters.reject_metropolis += 1
        end
        # REACT_DEBUG: Update status after MH rejection
        react_debug_enabled = get(ENV, "REACT_DEBUG", "0") == "1"
        if react_debug_enabled && _react_debug_counter[] <= 200 && _react_debug_counter[] > 0
            println("REACT_DEBUG[$(_react_debug_counter[])]: REJECTED by Metropolis (logα=$(round(log_acc, digits=6)))")
        end
        st.N = N_before
        st.pos = pos_old
        st.types = types_old
        return (false, false, true, false)  # (metropolis_accepted=false, committed=false, feasible=true, invariant_failed=false)
    end
    
    # Deterministic sanity checks after commit
    counts_after_check = count_species(st, n_species)
    N_total_after = st.N
    expected_ΔN = sum(νeff)
    actual_ΔN = N_total_after - N_before
    
    invariant_failed = false
    if actual_ΔN != expected_ΔN
        invariant_failed = true
    else
        for (species_id, ν) in enumerate(νeff)
            if ν != 0
                expected_count = counts_before[species_id] + ν
                actual_count = counts_after_check[species_id]
                if actual_count != expected_count
                    invariant_failed = true
                    break
                end
            end
        end
    end
    
    if invariant_failed
        # Invariant failed - restore state and return
        st.N = N_before
        st.pos = pos_old
        st.types = types_old
        return (metropolis_accepted, false, true, true)  # (metropolis_accepted, committed=false, feasible=true, invariant_failed=true)
    end
    
    # Rebuild cell list after accepted move
    st.cl = CellList(st.N, L, p.rc)
    rebuild_cells!(st)
    
    # State update is valid and committed - increment committed counter
    if counters !== nothing && reaction.label == "A⇌D+F"
        if direction == :forward
            counters.forward_committed += 1
        else
            counters.reverse_committed += 1
        end
    end
    
    # DB audit callback: if enabled and this is A⇌D+F, compute and call callback with breakdown
    # Only invoke for accepted and committed moves
    db_audit_enabled = get(ENV, "DB_AUDIT", "0") == "1"
    if db_audit_enabled && db_audit_callback !== nothing && reaction.label == "A⇌D+F" && metropolis_accepted && energy_valid && log_acc_valid
        # DB_AUDIT sanity check: ensure all required variables are finite (not NaN/Inf)
        if !isfinite(ΔU_term) || !isfinite(logq_term) || !isfinite(vol_term) || !isfinite(fact_term) || !isfinite(log_acc)
            error("DB_AUDIT_FAIL: Non-finite values detected. ΔU_term=$(ΔU_term), logq_term=$(logq_term), vol_term=$(vol_term), fact_term=$(fact_term), log_acc=$(log_acc)")
        end
        
        # Compute breakdown components
        # log_pi_ratio = log π(y) - log π(x) = ΔU_term + logq_term + vol_term + fact_term
        log_pi_ratio = ΔU_term + logq_term + vol_term + fact_term
        # log_g_ratio = 0.0 for Smith–Tríska REMC (no proposal-density corrections)
        log_g_ratio = 0.0
        # logα_theory = log_pi_ratio + log_g_ratio (theoretical acceptance from detailed balance)
        logα_theory = log_pi_ratio + log_g_ratio
        # logα_used = log_acc (the actual value used in Metropolis, before any min(0, ...) truncation)
        logα_used = log_acc
        # residual = logα_used - logα_theory (should be ≈ 0 for detailed balance)
        residual = logα_used - logα_theory
        
        # Assertion 1: consistency check (logα_theory should equal logα_used within tolerance)
        # This checks that the breakdown components sum correctly
        theory_diff = abs(logα_theory - logα_used)
        if theory_diff >= 1e-10
            error("DB_AUDIT_FAIL: |logα_theory - logα_used| = $(theory_diff) >= 1e-10. logα_theory=$(logα_theory), logα_used=$(logα_used), log_pi_ratio=$(log_pi_ratio), log_g_ratio=$(log_g_ratio), ΔU_term=$(ΔU_term), logq_term=$(logq_term), vol_term=$(vol_term), fact_term=$(fact_term))")
        end
        
        # Create breakdown struct
        breakdown = AcceptanceBreakdown(
            direction,
            copy(counts_before),
            copy(counts_after_check),
            V,
            ΔU,
            p.β,  # β = 1/T
            fact_term,  # log_combo
            logq_term,
            vol_term,
            logK_term,
            logGamma_term,
            log_prop_ratio,
            log_acc,  # logα_total
            log_pi_ratio,
            log_g_ratio,
            residual,
            logα_used,
            logq_terms,
            insertion_mode,  # Insertion mode used
            proposal_mode   # Proposal mode used
        )
        
        # Call callback
        db_audit_callback(breakdown)
    end
    
    # DB_AUDIT for logGamma reactions: print first accepted forward/reverse
    if db_audit_enabled && reaction.logGamma !== nothing && metropolis_accepted && energy_valid && log_acc_valid
        key = (reaction.label, direction)
        printed = get(_db_audit_gamma_printed[], key, false)
        if !printed
            _db_audit_gamma_printed[][key] = true
            log_alpha_sum = ΔU_term + vol_term + fact_term + logGamma_term
            println("DB_AUDIT_GAMMA: reaction=$(reaction.label) direction=$direction")
            println("  ΔU=$(round(ΔU, digits=6)) β=$(round(p.β, digits=6)) -βΔU=$(round(ΔU_term, digits=6))")
            println("  logGamma=$(round(logGamma_term, digits=6)) logcombo=$(round(fact_term, digits=6)) vol_term=$(round(vol_term, digits=6))")
            println("  logα=$(round(log_acc, digits=6)) sum_terms=$(round(log_alpha_sum, digits=6))")
            if abs(log_alpha_sum - log_acc) >= 1e-10
                error("DB_AUDIT_FAIL_GAMMA: |logα - sum_terms| >= 1e-10")
            end
        end
    end
    
    # State update is valid and committed
    return (true, true, true, false)  # (metropolis_accepted=true, committed=true, feasible=true, invariant_failed=false)
end

"""
    sweep_with_reactions!(st::LJState, p::LJParams, reaction::Union{Reaction, Nothing};
                          p_reaction::Float64=0.0, rebuild_every::Int=-1,
                          proposal_mode::Symbol=:uniform, com_kernel_Δ::Float64=0.05)

Perform one sweep of MC moves (N particle translation moves) plus optional reaction moves.
If reaction is not nothing and p_reaction > 0, attempts a reaction move with probability p_reaction.

Parameters:
- proposal_mode: :uniform or :com_insert (default: :uniform)
- com_kernel_Δ: Kernel width for COM insertion (default: 0.05, relative to box length)
- use_alchemical: enable alchemical flips for 1↔1 reactions (default: true)
- insertion_mode: :anchored, :standard_remc, or :uniform_only (default: :anchored)
- debug_logacc: enable reaction log acceptance debug (default: false)

Returns: (translation_acceptance_rate, reaction_accepted, reaction_attempted, 
          reaction_forward_attempted, reaction_forward_accepted,
          reaction_reverse_attempted, reaction_reverse_accepted)
"""
function sweep_with_reactions!(st::LJState, p, reaction::Union{Reaction, Nothing};
                               p_reaction::Float64=0.0, rebuild_every::Int=-1,
                               proposal_mode::Symbol=:uniform, com_kernel_Δ::Float64=0.05,
                               use_alchemical::Bool=true, insertion_mode::Symbol=:anchored,
                               debug_logacc::Bool=false)
    # Perform regular translation sweep
    translation_acceptance = sweep!(st, p; rebuild_every=rebuild_every)
    
    # Attempt reaction move if enabled
    reaction_accepted = 0
    reaction_attempted = 0
    reaction_forward_attempted = 0
    reaction_forward_accepted = 0
    reaction_reverse_attempted = 0
    reaction_reverse_accepted = 0
    if reaction !== nothing && p_reaction > 0.0 && st.N > 0
        if rand(st.rng) < p_reaction
            # Choose direction randomly (50/50 forward/reverse)
            direction = rand(st.rng) < 0.5 ? :forward : :reverse
            accepted, feasible = reaction_trial!(st, p, reaction, direction; 
                                                 proposal_mode=proposal_mode, com_kernel_Δ=com_kernel_Δ,
                                                 use_alchemical=use_alchemical, insertion_mode=insertion_mode,
                                                 debug_logacc=debug_logacc)
            if feasible
                # Only count as attempted if feasible
                reaction_attempted = 1
                if direction == :forward
                    reaction_forward_attempted = 1
                else
                    reaction_reverse_attempted = 1
                end
                if accepted
                    reaction_accepted = 1
                    if direction == :forward
                        reaction_forward_accepted = 1
                    else
                        reaction_reverse_accepted = 1
                    end
                end
            end
            # If not feasible, don't count as attempted (was skipped)
        end
    end
    
    return (translation_acceptance, reaction_accepted, reaction_attempted, 
            reaction_forward_attempted, reaction_forward_accepted,
            reaction_reverse_attempted, reaction_reverse_accepted)
end

"""
    sweep_npt_with_reactions!(st::LJState, p::LJParams, reaction::Union{Reaction, Nothing};
                               Pext::Float64=1.0, max_dlnV::Float64=0.01,
                               p_reaction::Float64=0.0, do_volume_move::Bool=false,
                               rebuild_every::Int=-1,
                               proposal_mode::Symbol=:uniform, com_kernel_Δ::Float64=0.05)

Perform one NPT sweep: N translation moves, optional volume move, optional reaction move.
Volume moves should be attempted every vol_move_every sweeps (caller tracks sweep count).
Reaction moves are attempted with probability p_reaction per sweep.

Parameters:
- proposal_mode: :uniform or :com_insert (default: :uniform)
- com_kernel_Δ: Kernel width for COM insertion (default: 0.05, relative to box length)

Returns: (translation_acceptance_rate, volume_accepted, volume_attempted, 
          reaction_accepted, reaction_attempted,
          reaction_forward_attempted, reaction_forward_accepted,
          reaction_reverse_attempted, reaction_reverse_accepted,
          forward_committed, reverse_committed, immediate_undo,
          counts_before_reaction, counts_after_reaction)
"""
function sweep_npt_with_reactions!(st::LJState, p, reaction::Union{Reaction, Nothing};
                                    Pext::Float64=1.0, max_dlnV::Float64=0.01,
                                    p_reaction::Float64=0.0, do_volume_move::Bool=false,
                                    rebuild_every::Int=-1,
                                    proposal_mode::Symbol=:uniform, com_kernel_Δ::Float64=0.05,
                                    use_alchemical::Bool=true, insertion_mode::Symbol=:anchored,
                                    debug_logacc::Bool=false,
                                    db_audit_callback::Union{Function, Nothing}=nothing)
    # Perform regular translation sweep
    translation_acceptance = sweep!(st, p; rebuild_every=rebuild_every)
    
    # Attempt volume move if requested
    volume_accepted = 0
    volume_attempted = 0
    if do_volume_move
        volume_attempted = 1
        vol_acc = volume_trial!(st, p; max_dlnV=max_dlnV, Pext=Pext)
        if vol_acc
            volume_accepted = 1
        end
    end
    
    # Track counts before reaction attempt for instrumentation
    counts_before_reaction = Int[]
    counts_after_reaction = Int[]
    forward_committed = 0
    reverse_committed = 0
    immediate_undo = 0
    
    # Track state before reaction attempt (for immediate undo detection)
    counts_before_forward_in_sweep = Int[]  # Store counts before any forward commit in this sweep
    
    # Attempt reaction move if enabled
    reaction_accepted = 0
    reaction_attempted = 0
    reaction_forward_attempted = 0
    reaction_forward_accepted = 0
    reaction_reverse_attempted = 0
    reaction_reverse_accepted = 0
    reaction_counters = nothing  # Initialize counters (will be set if A⇌D+F)
    if reaction !== nothing && p_reaction > 0.0 && st.N > 0
        if rand(st.rng) < p_reaction
            # Track counts before reaction attempt
            n_species = length(reaction.stoichiometry)
            counts_before_reaction = count_species(st, n_species)
            
            # Create counters for A⇌D+F if this is that reaction
            reaction_counters = nothing
            if reaction.label == "A⇌D+F"
                reaction_counters = ReactionCounters()
            end
            
            # Choose direction randomly (50/50 forward/reverse)
            direction = rand(st.rng) < 0.5 ? :forward : :reverse
            accepted, feasible = reaction_trial!(st, p, reaction, direction; 
                                                 proposal_mode=proposal_mode, com_kernel_Δ=com_kernel_Δ,
                                                 use_alchemical=use_alchemical, insertion_mode=insertion_mode,
                                                 debug_logacc=debug_logacc,
                                                 counters=reaction_counters,
                                                 db_audit_callback=db_audit_callback)
            
            # Track counts after reaction attempt
            counts_after_reaction = count_species(st, n_species)
            
            if feasible
                # Only count as attempted if feasible
                reaction_attempted = 1
                if direction == :forward
                    reaction_forward_attempted = 1
                else
                    reaction_reverse_attempted = 1
                end
                if accepted
                    reaction_accepted = 1
                    if direction == :forward
                        reaction_forward_accepted = 1
                        forward_committed = 1
                        # Store counts before this forward commit to detect immediate undo
                        counts_before_forward_in_sweep = copy(counts_before_reaction)
                    else
                        reaction_reverse_accepted = 1
                        reverse_committed = 1
                        
                        # Check for immediate undo: if we just did a reverse and counts match pre-forward state
                        # (This can only happen if logic changes to allow multiple reactions per sweep)
                        if length(counts_before_forward_in_sweep) > 0 && counts_after_reaction == counts_before_forward_in_sweep
                            immediate_undo = 1
                        end
                    end
                    
                    # Assert that counts changed by νeff (validation)
                    # Compute νeff for this direction
                    νeff_check = direction == :forward ? reaction.stoichiometry : [-ν for ν in reaction.stoichiometry]
                    for (species_id, ν) in enumerate(νeff_check)
                        if ν != 0
                            expected_change = ν
                            actual_change = counts_after_reaction[species_id] - counts_before_reaction[species_id]
                            if actual_change != expected_change
                                error("Reaction committed but counts didn't change correctly. Direction=$direction, Species $species_id: expected_change=$expected_change, actual_change=$actual_change, counts_before=$(counts_before_reaction[species_id]), counts_after=$(counts_after_reaction[species_id]), νeff=$νeff_check")
                            end
                        end
                    end
                end
            end
            # If not feasible, don't count as attempted (was skipped)
        end
    end
    
    return (translation_acceptance, volume_accepted, volume_attempted, 
            reaction_accepted, reaction_attempted,
            reaction_forward_attempted, reaction_forward_accepted,
            reaction_reverse_attempted, reaction_reverse_accepted,
            forward_committed, reverse_committed, immediate_undo,
            counts_before_reaction, counts_after_reaction,
            reaction_counters)  # Add counters to return tuple (nil unless A⇌D+F)
end

# Import needed functions (all MC modules are in same namespace when included)
# These will be available when this file is included in MolSim.jl
