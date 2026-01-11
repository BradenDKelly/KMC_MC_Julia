"""
Radial Distribution Function (RDF) computation utilities.
Supports both single-component and multicomponent (species-resolved) RDF.
"""

"""
    rdf(pos, L, rmax, nbins; types=nothing) -> (r, g_r) or (r, g_ab)

Compute radial distribution function g(r) for a periodic cubic box.

For single-component (types=nothing):
  Returns (r, g_r) where g_r is a vector.

For multicomponent (types provided):
  Returns (r, g_ab) where g_ab is a matrix of size (n_types, n_types, nbins).
  g_ab[α, β, :] gives g_{αβ}(r) for species α, β.

Inputs:
- `pos`: Matrix{Float64} of size (3, N) where N is number of particles, columns are particles
- `L`: Box length (cubic box)
- `rmax`: Maximum distance to compute RDF (typically L/2)
- `nbins`: Number of bins for histogram
- `types`: Optional Vector{Int} of particle types (1-indexed). If provided, computes species-resolved RDF.

Outputs:
- `r`: Vector{Float64} of bin centers
- `g_r` or `g_ab`: Single-component g(r) vector or multicomponent matrix

Normalization:
- Single-component: g(r) = (2*n_pairs) / ((N-1) * ρ * V_shell)
- Multicomponent: g_{αβ}(r) normalized by ideal-gas pair density for that species pair.
  For α ≠ β: normalized by ρ_β * N_α * V_shell
  For α = β: normalized by ρ_α * (N_α - 1) * V_shell / 2 (to avoid double counting)

Uses minimum-image convention for periodic boundary conditions.
"""
function rdf(pos::Matrix{Float64}, L::Float64, rmax::Float64, nbins::Int; types::Union{Vector{Int}, Nothing}=nothing)
    N = size(pos, 2)
    if N < 2
        error("RDF requires at least 2 particles, got N=$N")
    end
    
    # Bin width
    dr = rmax / nbins
    
    # Number density
    ρ = N / (L * L * L)
    
    # Half box length for minimum image
    L_half = L / 2.0
    
    if types === nothing
        # Single-component RDF
        return _rdf_single_component(pos, L, L_half, rmax, nbins, dr, ρ, N)
    else
        # Multicomponent RDF
        if length(types) != N
            error("types length $(length(types)) must match N=$N")
        end
        n_types = maximum(types)
        @assert minimum(types) >= 1 "type IDs must be >= 1"
        return _rdf_multicomponent(pos, L, L_half, rmax, nbins, dr, ρ, N, types, n_types)
    end
end

"""
    _rdf_single_component(...)

Single-component RDF computation.
"""
function _rdf_single_component(pos::Matrix{Float64}, L::Float64, L_half::Float64, rmax::Float64, 
                                nbins::Int, dr::Float64, ρ::Float64, N::Int)
    # Histogram of pair distances
    hist = zeros(Int, nbins)
    
    # Count pairs
    @inbounds for i in 1:N
        for j in (i+1):N
            # Compute distance vector
            dr_x = pos[1, j] - pos[1, i]
            dr_y = pos[2, j] - pos[2, i]
            dr_z = pos[3, j] - pos[3, i]
            
            # Apply minimum image convention
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
            
            # Squared distance
            r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
            r = sqrt(r2)
            
            # Bin index
            if r < rmax && r > 0.0
                bin_idx = floor(Int, r / dr) + 1
                if bin_idx >= 1 && bin_idx <= nbins
                    hist[bin_idx] += 1
                end
            end
        end
    end
    
    # Compute bin centers and g(r)
    r_bins = zeros(Float64, nbins)
    g_r = zeros(Float64, nbins)
    
    for i in 1:nbins
        # Bin center
        r_lower = (i - 1) * dr
        r_upper = i * dr
        r_bins[i] = 0.5 * (r_lower + r_upper)
        
        # Shell volume: V_shell = 4π/3 * (r_upper^3 - r_lower^3)
        V_shell = (4.0 * π / 3.0) * (r_upper^3 - r_lower^3)
        
        # Number of pairs in this shell
        n_pairs = hist[i]
        
        # Normalize: g(r) = (2*n_pairs) / ((N-1) * ρ * V_shell)
        if V_shell > 0.0 && N > 1
            g_r[i] = (2.0 * n_pairs) / ((N - 1) * ρ * V_shell)
        else
            g_r[i] = 0.0
        end
    end
    
    return (r_bins, g_r)
end

"""
    _rdf_multicomponent(...)

Multicomponent RDF computation: g_{αβ}(r) for all species pairs.
"""
function _rdf_multicomponent(pos::Matrix{Float64}, L::Float64, L_half::Float64, rmax::Float64,
                             nbins::Int, dr::Float64, ρ::Float64, N::Int, types::Vector{Int}, n_types::Int)
    # Histogram: hist[α, β, bin] for species α, β
    # We store only α <= β to avoid redundancy (symmetric for identical species)
    # But for output, we'll compute full matrix for convenience
    hist = zeros(Int, n_types, n_types, nbins)
    
    # Count species-specific pairs
    @inbounds for i in 1:N
        type_i = types[i]
        for j in (i+1):N
            type_j = types[j]
            
            # Compute distance vector
            dr_x = pos[1, j] - pos[1, i]
            dr_y = pos[2, j] - pos[2, i]
            dr_z = pos[3, j] - pos[3, i]
            
            # Apply minimum image convention
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
            
            # Squared distance
            r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
            r = sqrt(r2)
            
                # Bin index
                if r < rmax && r > 0.0
                    bin_idx = floor(Int, r / dr) + 1
                    if bin_idx >= 1 && bin_idx <= nbins
                        # Count pair (i,j) once, store in symmetric matrix
                        hist[type_i, type_j, bin_idx] += 1
                        if type_i != type_j
                            hist[type_j, type_i, bin_idx] += 1  # Symmetric only for different species
                        end
                    end
                end
        end
    end
    
    # Compute bin centers
    r_bins = zeros(Float64, nbins)
    for i in 1:nbins
        r_lower = (i - 1) * dr
        r_upper = i * dr
        r_bins[i] = 0.5 * (r_lower + r_upper)
    end
    
    # Count particles per species
    N_species = zeros(Int, n_types)
    @inbounds for i in 1:N
        N_species[types[i]] += 1
    end
    
    # Compute g_{αβ}(r) for all species pairs
    g_ab = zeros(Float64, n_types, n_types, nbins)
    
    for α in 1:n_types
        N_α = N_species[α]
        ρ_α = N_α / (L * L * L)  # Number density of species α
        
        for β in 1:n_types
            N_β = N_species[β]
            ρ_β = N_β / (L * L * L)
            
            for bin_idx in 1:nbins
                # Shell volume
                r_lower = (bin_idx - 1) * dr
                r_upper = bin_idx * dr
                V_shell = (4.0 * π / 3.0) * (r_upper^3 - r_lower^3)
                
                # Normalization
                if α == β
                    # Same species: each pair (i<j) counted once in hist[α,α]
                    # Expected pairs: (N_α * (N_α - 1) / 2) * (V_shell / V)
                    # Normalization: g_{αα}(r) = (2 * n_pairs) / ((N_α - 1) * ρ_α * V_shell)
                    # where n_pairs = hist[α, α, bin_idx] (no division by 2)
                    n_pairs = Float64(hist[α, β, bin_idx])
                    if N_α > 1 && V_shell > 0.0
                        g_ab[α, β, bin_idx] = (2.0 * n_pairs) / (Float64(N_α - 1) * ρ_α * V_shell)
                    else
                        g_ab[α, β, bin_idx] = 0.0
                    end
                else
                    # Different species: each pair (i<j) counted in BOTH hist[α,β] and hist[β,α]
                    # So we need to combine them: total pairs = (hist[α,β] + hist[β,α]) / 2
                    # Or equivalently, just use hist[α,β] directly since hist[α,β] == hist[β,α]
                    # Expected pairs: N_α * N_β * (V_shell / V)
                    # Normalization: g_{αβ}(r) = n_pairs / (N_α * ρ_β * V_shell)
                    # where n_pairs = hist[α, β, bin_idx] (already contains all pairs)
                    n_pairs = Float64(hist[α, β, bin_idx])
                    if N_α > 0 && N_β > 0 && V_shell > 0.0
                        g_ab[α, β, bin_idx] = n_pairs / (Float64(N_α) * ρ_β * V_shell)
                    else
                        g_ab[α, β, bin_idx] = 0.0
                    end
                end
            end
        end
    end
    
    return (r_bins, g_ab)
end
