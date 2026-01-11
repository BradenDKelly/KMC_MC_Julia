"""
Reaction Ensemble Monte Carlo (RxMC) for NVT and NPT simulations.
Implements chemical reaction moves for multicomponent LJ systems.

v1: Single-site atomic species only (no molecules/CBMC).
Supports both NVT and NPT ensembles.
"""

using Random
using StaticArrays
using SpecialFunctions  # For loggamma

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
end

"""
    Reaction(label, stoichiometry, logK; logq=nothing)

Convenience constructor for Reaction.
"""
function Reaction(label::String, stoichiometry::Vector{Int}, logK::Float64; logq::Union{Vector{Float64}, Nothing}=nothing)
    return Reaction(label, stoichiometry, logK, logq)
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
    reaction_trial!(st::LJState, p::LJParams, reaction::Reaction, direction::Symbol)::Bool

Perform a reaction ensemble Monte Carlo trial move.

For Table 3.2 (Braden thesis), uses the following acceptance criterion:
ln_acc = -β * ΔU + ΔN * log(V) + Σ ν_i * log(q_i/λ³) + Σ [log(N_i!) - log((N_i + ν_i)!)]
where:
- β = 1/(kT)
- ΔU = energy change
- ΔN = sum_i ν_i (change in total particle number)
- V = volume
- ν_i = stoichiometric coefficient (negative for reactants, positive for products)
- N_i = count of species i before reaction
- q_i/λ³ = ideal-gas partition function per volume for species i

For ΔN=0 reactions (e.g., A⇌B, A⇌C), uses alchemical conversion:
- Pick a particle of reactant species uniformly
- Change its type to product species
- Compute ΔU using local energy change

For ΔN≠0 reactions, uses insertion/deletion:
- Delete reactant particles
- Insert product particles at random positions

Returns: true if accepted, false if rejected.
"""
function reaction_trial!(st::LJState, p::LJParams, reaction::Reaction, direction::Symbol)::Bool
    stoichiometry = reaction.stoichiometry
    n_species = length(stoichiometry)
    L = st.L
    V = L * L * L
    
    # Count current species
    counts_before = count_species(st, n_species)
    
    # Determine which species are reactants and products for this direction
    Δn = sum(stoichiometry)  # Change in total particle number for forward reaction
    
    # Check feasibility
    if direction == :forward
        # Forward: negative coefficients are consumed, positive are produced
        for (species_id, ν) in enumerate(stoichiometry)
            if ν < 0  # Reactant
                if counts_before[species_id] < -ν
                    return false  # Not enough reactants
                end
            end
        end
    else  # :reverse
        # Reverse: positive coefficients are consumed, negative are produced
        Δn = -Δn  # Flip sign for reverse
        for (species_id, ν) in enumerate(stoichiometry)
            if ν > 0  # Was product, now reactant in reverse
                if counts_before[species_id] < ν
                    return false  # Not enough to reverse
                end
            end
        end
    end
    
    # Compute energy before reaction
    U_before = total_energy(st, p)
    
    # Store old state for rejection
    N_before = st.N
    pos_old = copy(st.pos)
    types_old = copy(st.types)
    
    # Handle alchemical conversion for ΔN=0 reactions
    if Δn == 0
        # Alchemical conversion: change type of particles
        if direction == :forward
            # Find reactant species (negative ν)
            reactant_species = Int[]
            product_species = Int[]
            for (species_id, ν) in enumerate(stoichiometry)
                if ν < 0
                    push!(reactant_species, species_id)
                elseif ν > 0
                    push!(product_species, species_id)
                end
            end
            
            if length(reactant_species) != 1 || length(product_species) != 1
                # Complex alchemical conversion not implemented
                return false
            end
            
            reactant_id = reactant_species[1]
            product_id = product_species[1]
            n_convert = -stoichiometry[reactant_id]  # Should equal stoichiometry[product_id]
            
            # Find all particles of reactant species
            reactant_indices = Int[]
            for i in 1:st.N
                if st.types[i] == reactant_id
                    push!(reactant_indices, i)
                end
            end
            
            if length(reactant_indices) < n_convert
                return false
            end
            
            # Randomly select particles to convert
            selected = Int[]
            available = copy(reactant_indices)
            for _ in 1:n_convert
                idx_choice = rand(st.rng, 1:length(available))
                push!(selected, available[idx_choice])
                deleteat!(available, idx_choice)
            end
            
            # Convert types
            for idx in selected
                st.types[idx] = product_id
            end
        else  # :reverse
            # Reverse: convert products back to reactants
            reactant_species = Int[]
            product_species = Int[]
            for (species_id, ν) in enumerate(stoichiometry)
                if ν > 0  # Was product, now reactant
                    push!(product_species, species_id)
                elseif ν < 0  # Was reactant, now product
                    push!(reactant_species, species_id)
                end
            end
            
            if length(reactant_species) != 1 || length(product_species) != 1
                return false
            end
            
            product_id = product_species[1]
            reactant_id = reactant_species[1]
            n_convert = stoichiometry[product_id]  # Number to convert back
            
            product_indices = Int[]
            for i in 1:st.N
                if st.types[i] == product_id
                    push!(product_indices, i)
                end
            end
            
            if length(product_indices) < n_convert
                return false
            end
            
            selected = Int[]
            available = copy(product_indices)
            for _ in 1:n_convert
                idx_choice = rand(st.rng, 1:length(available))
                push!(selected, available[idx_choice])
                deleteat!(available, idx_choice)
            end
            
            for idx in selected
                st.types[idx] = reactant_id
            end
        end
        
        # Rebuild cell list (types changed but positions didn't)
        st.cl = CellList(st.N, L, p.rc)
        rebuild_cells!(st)
    else
        # ΔN ≠ 0: use insertion/deletion
        if direction == :forward
            # Delete reactants (negative stoichiometry)
            particles_to_delete = Int[]
            for (species_id, ν) in enumerate(stoichiometry)
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
                        st.N = N_before
                        st.pos = pos_old
                        st.types = types_old
                        return false
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
            # Delete in reverse sorted order to avoid index shifts
            sort!(particles_to_delete, rev=true)
            for idx in particles_to_delete
                delete_particle!(st, idx)
            end
            
            # Insert products (positive stoichiometry)
            for (species_id, ν) in enumerate(stoichiometry)
                if ν > 0
                    n_insert = ν
                    for _ in 1:n_insert
                        x_new = rand(st.rng) * L
                        y_new = rand(st.rng) * L
                        z_new = rand(st.rng) * L
                        pos_new = SVector{3,Float64}(x_new, y_new, z_new)
                        insert_particle!(st, pos_new, species_id)
                    end
                end
            end
        else  # :reverse
            # Delete products (positive stoichiometry, but we're reversing)
            particles_to_delete = Int[]
            for (species_id, ν) in enumerate(stoichiometry)
                if ν > 0  # Was product, now reactant in reverse
                    n_delete = ν
                    species_indices = Int[]
                    for i in 1:st.N
                        if st.types[i] == species_id
                            push!(species_indices, i)
                        end
                    end
                    if length(species_indices) < n_delete
                        st.N = N_before
                        st.pos = pos_old
                        st.types = types_old
                        return false
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
            sort!(particles_to_delete, rev=true)
            for idx in particles_to_delete
                delete_particle!(st, idx)
            end
            
            # Insert reactants (negative stoichiometry, but we're reversing)
            for (species_id, ν) in enumerate(stoichiometry)
                if ν < 0  # Was reactant, now product in reverse
                    n_insert = -ν
                    for _ in 1:n_insert
                        x_new = rand(st.rng) * L
                        y_new = rand(st.rng) * L
                        z_new = rand(st.rng) * L
                        pos_new = SVector{3,Float64}(x_new, y_new, z_new)
                        insert_particle!(st, pos_new, species_id)
                    end
                end
            end
        end
        
        # Rebuild cell list for new configuration
        st.cl = CellList(st.N, L, p.rc)
        rebuild_cells!(st)
    end
    
    # Compute energy after reaction
    U_after = total_energy(st, p)
    ΔU = U_after - U_before
    
    # Count species after reaction
    counts_after = count_species(st, n_species)
    
    # Compute acceptance ratio using loggamma for factorial calculations
    # ln_acc = -β * ΔU + ΔN * log(V) + Σ ν_i * log(q_i/λ³) + Σ [log(N_i!) - log((N_i + ν_i)!)]
    log_acc = -p.β * ΔU
    
    # Volume term: ΔN * log(V)
    if direction == :forward
        log_acc += Float64(Δn) * log(V)
    else
        log_acc -= Float64(Δn) * log(V)  # Reverse: Δn is flipped
    end
    
    # Ideal-gas partition function terms: Σ ν_i * log(q_i/λ³)
    if reaction.logq !== nothing
        for (species_id, ν) in enumerate(stoichiometry)
            if ν != 0 && 1 <= species_id <= length(reaction.logq)
                if direction == :forward
                    log_acc += Float64(ν) * reaction.logq[species_id]
                else
                    log_acc -= Float64(ν) * reaction.logq[species_id]  # Reverse: sign flipped
                end
            end
        end
    end
    
    # Combinatorial factor: Σ [log(N_i!) - log((N_i + ν_i)!)]
    # Using loggamma: log(n!) = loggamma(n+1)
    for (species_id, ν) in enumerate(stoichiometry)
        if ν != 0
            N_i_before = counts_before[species_id]
            N_i_after = counts_after[species_id]
            
            if direction == :forward
                # Forward: log(N_i!) - log((N_i + ν_i)!)
                if N_i_before >= 0
                    log_acc += loggamma(Float64(N_i_before) + 1.0)
                end
                if N_i_after >= 0
                    log_acc -= loggamma(Float64(N_i_after) + 1.0)
                end
            else  # :reverse
                # Reverse: sign is flipped, so log((N_i - ν_i)!) - log(N_i!)
                # But N_i_before is after the forward direction, so we need to think carefully
                # Actually, for reverse: we're going from counts_after back to counts_before
                # So it's log(N_i_after!) - log(N_i_before!)
                if N_i_after >= 0
                    log_acc += loggamma(Float64(N_i_after) + 1.0)
                end
                if N_i_before >= 0
                    log_acc -= loggamma(Float64(N_i_before) + 1.0)
                end
            end
        end
    end
    
    # Metropolis acceptance
    accepted = false
    if log_acc >= 0.0 || rand(st.rng) < exp(log_acc)
        accepted = true
    else
        # Reject: restore old state
        st.N = N_before
        st.pos = pos_old
        st.types = types_old
        st.cl = CellList(N_before, L, p.rc)
        rebuild_cells!(st)
    end
    
    return accepted
end

"""
    sweep_with_reactions!(st::LJState, p::LJParams, reaction::Union{Reaction, Nothing};
                          p_reaction::Float64=0.0, rebuild_every::Int=-1)

Perform one sweep of MC moves (N particle translation moves) plus optional reaction moves.
If reaction is not nothing and p_reaction > 0, attempts a reaction move with probability p_reaction.

Returns: (translation_acceptance_rate, reaction_accepted, reaction_attempted)
"""
function sweep_with_reactions!(st::LJState, p::LJParams, reaction::Union{Reaction, Nothing};
                               p_reaction::Float64=0.0, rebuild_every::Int=-1)
    # Perform regular translation sweep
    translation_acceptance = sweep!(st, p; rebuild_every=rebuild_every)
    
    # Attempt reaction move if enabled
    reaction_accepted = 0
    reaction_attempted = 0
    if reaction !== nothing && p_reaction > 0.0 && st.N > 0
        if rand(st.rng) < p_reaction
            reaction_attempted = 1
            # Choose direction randomly (50/50 forward/reverse)
            direction = rand(st.rng) < 0.5 ? :forward : :reverse
            accepted = reaction_trial!(st, p, reaction, direction)
            if accepted
                reaction_accepted = 1
            end
        end
    end
    
    return (translation_acceptance, reaction_accepted, reaction_attempted)
end

"""
    sweep_npt_with_reactions!(st::LJState, p::LJParams, reaction::Union{Reaction, Nothing};
                               Pext::Float64=1.0, max_dlnV::Float64=0.01,
                               p_reaction::Float64=0.0, do_volume_move::Bool=false,
                               rebuild_every::Int=-1)

Perform one NPT sweep: N translation moves, optional volume move, optional reaction move.
Volume moves should be attempted every vol_move_every sweeps (caller tracks sweep count).
Reaction moves are attempted with probability p_reaction per sweep.

Returns: (translation_acceptance_rate, volume_accepted, volume_attempted, reaction_accepted, reaction_attempted)
"""
function sweep_npt_with_reactions!(st::LJState, p::LJParams, reaction::Union{Reaction, Nothing};
                                    Pext::Float64=1.0, max_dlnV::Float64=0.01,
                                    p_reaction::Float64=0.0, do_volume_move::Bool=false,
                                    rebuild_every::Int=-1)
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
    
    # Attempt reaction move if enabled
    reaction_accepted = 0
    reaction_attempted = 0
    if reaction !== nothing && p_reaction > 0.0 && st.N > 0
        if rand(st.rng) < p_reaction
            reaction_attempted = 1
            # Choose direction randomly (50/50 forward/reverse)
            direction = rand(st.rng) < 0.5 ? :forward : :reverse
            accepted = reaction_trial!(st, p, reaction, direction)
            if accepted
                reaction_accepted = 1
            end
        end
    end
    
    return (translation_acceptance, volume_accepted, volume_attempted, reaction_accepted, reaction_attempted)
end

# Import needed functions (all MC modules are in same namespace when included)
# These will be available when this file is included in MolSim.jl
