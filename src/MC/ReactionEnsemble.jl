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
        L_half = L / 2.0
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
        
        com_x += dx
        com_y += dy
        com_z += dz
    end
    
    n = Float64(length(indices))
    com_x = (com_x / n) + ref_x
    com_y = (com_y / n) + ref_y
    com_z = (com_z / n) + ref_z
    
    # Wrap COM into [0, L)
    com_x = com_x - L * floor(com_x / L)
    com_y = com_y - L * floor(com_y / L)
    com_z = com_z - L * floor(com_z / L)
    
    return SVector{3,Float64}(com_x, com_y, com_z)
end

"""
    propose_positions_com_kernel!(rng, out_positions::Vector{SVector{3,Float64}}, 
                                   com::SVector{3,Float64}, Δ::Float64, L::Float64)::Float64

Propose positions for n_insert particles using a COM-centered kernel.
First particle is placed at COM (with uniform kernel of width Δ).
Additional particles are placed at COM + small random displacement (uniform in cube [-Δ, Δ]^3).
All positions are wrapped into [0, L).

Returns: log_q (log proposal density).
For uniform kernel in cube [-Δ, Δ]^3: density = 1/(8*Δ^3) per particle (after first).
For first particle: density = 1/(8*Δ^3) as well.

Total log_q = n_insert * log(1/(8*Δ^3)) = -n_insert * log(8*Δ^3)
"""
function propose_positions_com_kernel!(rng, out_positions::Vector{SVector{3,Float64}}, 
                                       com::SVector{3,Float64}, Δ::Float64, L::Float64)::Float64
    n_insert = length(out_positions)
    @assert n_insert > 0 "Must insert at least one particle"
    
    # First particle: at COM with kernel (uniform in cube [-Δ, Δ]^3)
    dx1 = (rand(rng) - 0.5) * 2.0 * Δ
    dy1 = (rand(rng) - 0.5) * 2.0 * Δ
    dz1 = (rand(rng) - 0.5) * 2.0 * Δ
    
    x1 = com[1] + dx1
    y1 = com[2] + dy1
    z1 = com[3] + dz1
    
    # Wrap into [0, L)
    x1 = x1 - L * floor(x1 / L)
    y1 = y1 - L * floor(y1 / L)
    z1 = z1 - L * floor(z1 / L)
    
    out_positions[1] = SVector{3,Float64}(x1, y1, z1)
    
    # Additional particles: COM + small displacement
    for i in 2:n_insert
        dx = (rand(rng) - 0.5) * 2.0 * Δ
        dy = (rand(rng) - 0.5) * 2.0 * Δ
        dz = (rand(rng) - 0.5) * 2.0 * Δ
        
        x = com[1] + dx
        y = com[2] + dy
        z = com[3] + dz
        
        # Wrap into [0, L)
        x = x - L * floor(x / L)
        y = y - L * floor(y / L)
        z = z - L * floor(z / L)
        
        out_positions[i] = SVector{3,Float64}(x, y, z)
    end
    
    # Proposal density: uniform in cube [-Δ, Δ]^3, so density = 1/(8*Δ^3) per particle
    # log_q = n_insert * log(1/(8*Δ^3)) = -n_insert * log(8*Δ^3)
    log_q = -Float64(n_insert) * log(8.0 * Δ * Δ * Δ)
    
    return log_q
end

"""
    reaction_trial!(st::LJState, p::LJParams, reaction::Reaction, direction::Symbol;
                    proposal_mode::Symbol=:uniform, com_kernel_Δ::Float64=0.05)::Bool

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

For ΔN≠0 reactions:
- proposal_mode = :uniform: Insert product particles at uniformly random positions
- proposal_mode = :com_insert: Insert product particles at COM of deleted reactants (with kernel)

Parameters:
- proposal_mode: :uniform or :com_insert (default: :uniform)
- com_kernel_Δ: Kernel width for COM insertion (default: 0.05, relative to box length)

Returns: true if accepted, false if rejected.
"""
function reaction_trial!(st::LJState, p::LJParams, reaction::Reaction, direction::Symbol;
                         proposal_mode::Symbol=:uniform, com_kernel_Δ::Float64=0.05)::Bool
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
    
    # Proposal densities for MH correction (only used for ΔN≠0)
    log_q_forward = 0.0
    log_q_reverse = 0.0
    
    # Handle alchemical conversion for ΔN=0 reactions
    if Δn == 0
        # SYMMETRIC PROPOSAL: Pick one particle uniformly from all N_total particles
        # Then flip its type (reactant→product or product→reactant)
        # This ensures detailed balance: same probability of selecting any particle regardless of direction
        
        # Identify reactant and product species
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
        
        if n_convert != 1
            # For now, only support 1:1 conversions (A⇌B, A⇌C)
            # Multi-particle conversions would need more complex symmetric proposal
            return false
        end
        
        # SYMMETRIC PROPOSAL: Pick one particle uniformly from ALL particles
        # Then check its type and flip accordingly
        particle_idx = rand(st.rng, 1:st.N)
        current_type = st.types[particle_idx]
        
        # Determine what to convert to based on current type
        if current_type == reactant_id
            # This particle is reactant, convert to product (forward move)
            st.types[particle_idx] = product_id
        elseif current_type == product_id
            # This particle is product, convert to reactant (reverse move)
            st.types[particle_idx] = reactant_id
        else
            # Particle is neither reactant nor product (shouldn't happen for simple reactions)
            return false
        end
        
        # Rebuild cell list (types changed but positions didn't)
        st.cl = CellList(st.N, L, p.rc)
        rebuild_cells!(st)
    else
        # ΔN ≠ 0: use insertion/deletion
        if direction == :forward
            # Delete reactants (negative stoichiometry)
            particles_to_delete = Int[]
            deleted_positions = Vector{SVector{3,Float64}}()
            
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
            
            # Store positions before deletion (need to do this before sorting indices)
            for idx in particles_to_delete
                push!(deleted_positions, SVector{3,Float64}(st.pos[1, idx], st.pos[2, idx], st.pos[3, idx]))
            end
            
            # Delete in reverse sorted order to avoid index shifts
            sort!(particles_to_delete, rev=true)
            for idx in particles_to_delete
                delete_particle!(st, idx)
            end
            
            # Compute COM of deleted particles (for COM insertion mode)
            com_deleted = compute_com_from_indices(pos_old, particles_to_delete, L)
            
            # Compute total number of products to insert
            total_n_insert = 0
            for (species_id, ν) in enumerate(stoichiometry)
                if ν > 0
                    total_n_insert += ν
                end
            end
            
            # Insert products (positive stoichiometry)
            if proposal_mode == :com_insert && total_n_insert > 0
                # COM-centered kernel insertion
                # Use relative kernel width (default 0.05 means 5% of box length)
                Δ_absolute = com_kernel_Δ * L
                proposed_positions = Vector{SVector{3,Float64}}(undef, total_n_insert)
                log_q_forward = propose_positions_com_kernel!(st.rng, proposed_positions, com_deleted, Δ_absolute, L)
                
                pos_idx = 1
                for (species_id, ν) in enumerate(stoichiometry)
                    if ν > 0
                        for _ in 1:ν
                            insert_particle!(st, proposed_positions[pos_idx], species_id)
                            pos_idx += 1
                        end
                    end
                end
            else
                # Uniform random insertion
                log_q_forward = -Float64(total_n_insert) * log(V)  # log(1/V^n) = -n*log(V)
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
            end
            
            # For reverse direction proposal density (if we were to reverse this move)
            # Reverse move would delete the products we just inserted, insert reactants
            total_n_delete_reverse = 0
            for (species_id, ν) in enumerate(stoichiometry)
                if ν > 0  # These are products, would be deleted in reverse
                    total_n_delete_reverse += ν
                end
            end
            total_n_insert_reverse = 0
            for (species_id, ν) in enumerate(stoichiometry)
                if ν < 0  # These are reactants, would be inserted in reverse
                    total_n_insert_reverse += -ν
                end
            end
            
            if proposal_mode == :com_insert
                # COM insertion: reverse would insert reactants at COM of deleted products
                # The proposal density is the same formula: -n_insert_reverse * log(8*Δ^3)
                Δ_absolute = com_kernel_Δ * L
                log_q_reverse = -Float64(total_n_insert_reverse) * log(8.0 * Δ_absolute * Δ_absolute * Δ_absolute)
            else
                # Uniform: reverse proposal density is -n_insert_reverse*log(V)
                log_q_reverse = -Float64(total_n_insert_reverse) * log(V)
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
            
            # Store positions before deletion
            deleted_positions = Vector{SVector{3,Float64}}()
            for idx in particles_to_delete
                push!(deleted_positions, SVector{3,Float64}(st.pos[1, idx], st.pos[2, idx], st.pos[3, idx]))
            end
            
            sort!(particles_to_delete, rev=true)
            for idx in particles_to_delete
                delete_particle!(st, idx)
            end
            
            # Compute COM of deleted products
            com_deleted = compute_com_from_indices(pos_old, particles_to_delete, L)
            
            # Compute total number of reactants to insert
            total_n_insert = 0
            for (species_id, ν) in enumerate(stoichiometry)
                if ν < 0  # Was reactant, now product in reverse
                    total_n_insert += -ν
                end
            end
            
            # Insert reactants (negative stoichiometry, but we're reversing)
            if proposal_mode == :com_insert && total_n_insert > 0
                # COM-centered kernel insertion
                Δ_absolute = com_kernel_Δ * L
                proposed_positions = Vector{SVector{3,Float64}}(undef, total_n_insert)
                log_q_reverse = propose_positions_com_kernel!(st.rng, proposed_positions, com_deleted, Δ_absolute, L)
                
                pos_idx = 1
                for (species_id, ν) in enumerate(stoichiometry)
                    if ν < 0  # Was reactant, now product in reverse
                        n_insert = -ν
                        for _ in 1:n_insert
                            insert_particle!(st, proposed_positions[pos_idx], species_id)
                            pos_idx += 1
                        end
                    end
                end
            else
                # Uniform random insertion
                log_q_reverse = -Float64(total_n_insert) * log(V)
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
            
            # For forward direction proposal density (if we were to forward this move)
            # Forward move would delete reactants, insert products
            total_n_delete_forward = 0
            for (species_id, ν) in enumerate(stoichiometry)
                if ν < 0  # These are reactants, would be deleted in forward
                    total_n_delete_forward += -ν
                end
            end
            total_n_insert_forward = 0
            for (species_id, ν) in enumerate(stoichiometry)
                if ν > 0  # These are products, would be inserted in forward
                    total_n_insert_forward += ν
                end
            end
            
            if proposal_mode == :com_insert
                # COM insertion: forward would insert products at COM of deleted reactants
                Δ_absolute = com_kernel_Δ * L
                log_q_forward = -Float64(total_n_insert_forward) * log(8.0 * Δ_absolute * Δ_absolute * Δ_absolute)
            else
                # Uniform: forward proposal density is -n_insert_forward*log(V)
                log_q_forward = -Float64(total_n_insert_forward) * log(V)
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
    
    # Compute acceptance ratio
    # For ΔN=0 (alchemical): ln_acc = -βΔU + (logq_product - logq_reactant) + logK0
    # For ΔN≠0 (insertion/deletion): includes volume and combinatorial terms
    log_acc = -p.β * ΔU
    
    if Δn == 0
        # ΔN=0 alchemical conversion: simplified acceptance
        # No volume terms (ΔN=0), no combinatorial factors (no insertion/deletion)
        # Only energy change and logq difference
        
        # Identify reactant and product species
        reactant_id = 0
        product_id = 0
        for (species_id, ν) in enumerate(stoichiometry)
            if ν < 0
                reactant_id = species_id
            elseif ν > 0
                product_id = species_id
            end
        end
        
        # Add logq difference: logq_product - logq_reactant
        if reaction.logq !== nothing && reactant_id > 0 && product_id > 0
            if 1 <= reactant_id <= length(reaction.logq) && 1 <= product_id <= length(reaction.logq)
                # Determine which direction we went based on counts
                if counts_after[product_id] > counts_before[product_id]
                    # Forward: reactant→product
                    log_acc += reaction.logq[product_id] - reaction.logq[reactant_id]
                else
                    # Reverse: product→reactant
                    log_acc += reaction.logq[reactant_id] - reaction.logq[product_id]
                end
            end
        end
        
        # Note: logK0 is typically 0.0 for Table 3.2 reactions
        # For A⇌B with identical species and logK=0, this gives ln_acc = -βΔU
        # If ΔU ≈ 0 (identical LJ params), then ln_acc ≈ 0, so acceptance ≈ 1
    else
        # ΔN≠0: use full insertion/deletion acceptance formula
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
    end
    
    # Metropolis-Hastings correction for biased insertion (proposal density ratio)
    # Only applies to ΔN≠0 reactions with non-uniform insertion
    if Δn != 0
        # Add log(q_reverse / q_forward) = log_q_reverse - log_q_forward
        log_acc += log_q_reverse - log_q_forward
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
                          p_reaction::Float64=0.0, rebuild_every::Int=-1,
                          proposal_mode::Symbol=:uniform, com_kernel_Δ::Float64=0.05)

Perform one sweep of MC moves (N particle translation moves) plus optional reaction moves.
If reaction is not nothing and p_reaction > 0, attempts a reaction move with probability p_reaction.

Parameters:
- proposal_mode: :uniform or :com_insert (default: :uniform)
- com_kernel_Δ: Kernel width for COM insertion (default: 0.05, relative to box length)

Returns: (translation_acceptance_rate, reaction_accepted, reaction_attempted)
"""
function sweep_with_reactions!(st::LJState, p::LJParams, reaction::Union{Reaction, Nothing};
                               p_reaction::Float64=0.0, rebuild_every::Int=-1,
                               proposal_mode::Symbol=:uniform, com_kernel_Δ::Float64=0.05)
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
            reaction_attempted = 1
            # Choose direction randomly (50/50 forward/reverse)
            direction = rand(st.rng) < 0.5 ? :forward : :reverse
            if direction == :forward
                reaction_forward_attempted = 1
            else
                reaction_reverse_attempted = 1
            end
            accepted = reaction_trial!(st, p, reaction, direction; proposal_mode=proposal_mode, com_kernel_Δ=com_kernel_Δ)
            if accepted
                reaction_accepted = 1
                if direction == :forward
                    reaction_forward_accepted = 1
                else
                    reaction_reverse_accepted = 1
                end
            end
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

Returns: (translation_acceptance_rate, volume_accepted, volume_attempted, reaction_accepted, reaction_attempted)
"""
function sweep_npt_with_reactions!(st::LJState, p::LJParams, reaction::Union{Reaction, Nothing};
                                    Pext::Float64=1.0, max_dlnV::Float64=0.01,
                                    p_reaction::Float64=0.0, do_volume_move::Bool=false,
                                    rebuild_every::Int=-1,
                                    proposal_mode::Symbol=:uniform, com_kernel_Δ::Float64=0.05)
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
    reaction_forward_attempted = 0
    reaction_forward_accepted = 0
    reaction_reverse_attempted = 0
    reaction_reverse_accepted = 0
    if reaction !== nothing && p_reaction > 0.0 && st.N > 0
        if rand(st.rng) < p_reaction
            reaction_attempted = 1
            # Choose direction randomly (50/50 forward/reverse)
            direction = rand(st.rng) < 0.5 ? :forward : :reverse
            if direction == :forward
                reaction_forward_attempted = 1
            else
                reaction_reverse_attempted = 1
            end
            accepted = reaction_trial!(st, p, reaction, direction; proposal_mode=proposal_mode, com_kernel_Δ=com_kernel_Δ)
            if accepted
                reaction_accepted = 1
                if direction == :forward
                    reaction_forward_accepted = 1
                else
                    reaction_reverse_accepted = 1
                end
            end
        end
    end
    
    return (translation_acceptance, volume_accepted, volume_attempted, 
            reaction_accepted, reaction_attempted,
            reaction_forward_attempted, reaction_forward_accepted,
            reaction_reverse_attempted, reaction_reverse_accepted)
end

# Import needed functions (all MC modules are in same namespace when included)
# These will be available when this file is included in MolSim.jl
