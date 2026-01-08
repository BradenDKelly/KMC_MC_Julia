"""
Array-based cell binning for efficient neighbor finding (GPU-friendly).
"""

"""
    CellBins

Array-based cell binning data structure for efficient neighbor finding.
GPU-friendly: uses arrays instead of linked lists.
"""
struct CellBins
    cell_counts::Vector{Int}      # cell_counts[c] = number of particles in cell c
    cell_offsets::Vector{Int}      # cell_offsets[c] = start index in cell_particles for cell c (prefix sums)
    cell_particles::Vector{Int}    # cell_particles[i] = particle index, packed by cell
    particle_cell::Vector{Int}    # particle_cell[i] = cell containing particle i
    ncell::Int                    # number of cells per dimension
    ncell_total::Int              # total number of cells (ncell^3)
    L::Float64                    # box length
    rc::Float64                   # cutoff distance
end

"""
    CellBins(N, L, rc)

Construct a cell binning structure for `N` particles in a box of length `L` with cutoff `rc`.
"""
function CellBins(N::Int, L::Float64, rc::Float64)
    ncell = max(1, floor(Int, L / rc))
    ncell_total = ncell * ncell * ncell
    cell_counts = zeros(Int, ncell_total)
    cell_offsets = zeros(Int, ncell_total + 1)
    cell_particles = zeros(Int, N)
    particle_cell = zeros(Int, N)
    return CellBins(cell_counts, cell_offsets, cell_particles, particle_cell, ncell, ncell_total, L, rc)
end

"""
    cell_index(i, j, k, ncell)

Convert 3D cell indices (i, j, k) to linear cell index.
"""
@inline function cell_index(i::Int, j::Int, k::Int, ncell::Int)
    return (i - 1) * ncell * ncell + (j - 1) * ncell + k
end

"""
    get_cell(x, y, z, L, ncell)

Get the cell index for a position (x, y, z).
"""
@inline function get_cell(x::Float64, y::Float64, z::Float64, L::Float64, ncell::Int)
    # Clamp to [0, L) then convert to cell indices [1, ncell]
    i = clamp(floor(Int, x / L * ncell) + 1, 1, ncell)
    j = clamp(floor(Int, y / L * ncell) + 1, 1, ncell)
    k = clamp(floor(Int, z / L * ncell) + 1, 1, ncell)
    return cell_index(i, j, k, ncell)
end

"""
    build_cells!(backend, cell_bins, posx, posy, posz)

Build cell binning from current particle positions (SoA format).
Zero-allocation: reuses existing arrays.
Dispatches on backend type.
"""
function build_cells!(
    backend::AbstractBackend,
    cell_bins::CellBins,
    posx::AbstractVector{<:Real},
    posy::AbstractVector{<:Real},
    posz::AbstractVector{<:Real}
)
    return _build_cells_cpu!(cell_bins, posx, posy, posz)
end

"""
    _build_cells_cpu!(cell_bins, posx, posy, posz)

CPU implementation of build_cells! (internal).
GPU-compatible: no dictionaries, no IO, no allocations in hot paths.
"""
function _build_cells_cpu!(
    cell_bins::CellBins,
    posx::AbstractVector{<:Real},
    posy::AbstractVector{<:Real},
    posz::AbstractVector{<:Real}
)
    N = length(posx)
    L = cell_bins.L
    ncell = cell_bins.ncell
    ncell_total = cell_bins.ncell_total
    
    # Clear cell counts
    fill!(cell_bins.cell_counts, 0)
    
    # Count particles per cell
    @inbounds for i in 1:N
        x, y, z = posx[i], posy[i], posz[i]
        # Wrap coordinates to [0, L)
        x_wrapped = x - L * floor(x / L)
        y_wrapped = y - L * floor(y / L)
        z_wrapped = z - L * floor(z / L)
        
        cell_idx = get_cell(x_wrapped, y_wrapped, z_wrapped, L, ncell)
        cell_bins.particle_cell[i] = cell_idx
        cell_bins.cell_counts[cell_idx] += 1
    end
    
    # Build prefix sums (cell_offsets)
    cell_bins.cell_offsets[1] = 1
    @inbounds for c in 1:ncell_total
        cell_bins.cell_offsets[c + 1] = cell_bins.cell_offsets[c] + cell_bins.cell_counts[c]
    end
    
    # Reset counts (will use as insertion indices)
    fill!(cell_bins.cell_counts, 0)
    
    # Pack particles into cell_particles array
    @inbounds for i in 1:N
        cell_idx = cell_bins.particle_cell[i]
        offset = cell_bins.cell_offsets[cell_idx]
        idx = offset + cell_bins.cell_counts[cell_idx]
        cell_bins.cell_particles[idx] = i
        cell_bins.cell_counts[cell_idx] += 1
    end
    
    return nothing
end

"""
    iterate_neighbor_cells(cell_idx, ncell)

Iterate over the 27 neighbor cells (including self) of a given cell.
Returns an iterator that yields neighbor cell indices.
"""
function iterate_neighbor_cells(cell_idx::Int, ncell::Int)
    # Convert linear index to 3D (i, j, k)
    k = ((cell_idx - 1) % ncell) + 1
    j = (((cell_idx - 1) ÷ ncell) % ncell) + 1
    i = ((cell_idx - 1) ÷ (ncell * ncell)) + 1
    
    return NeighborCellIterator(i, j, k, ncell)
end

"""
    NeighborCellIterator

Iterator that yields neighbor cell indices.
"""
struct NeighborCellIterator
    i::Int
    j::Int
    k::Int
    ncell::Int
end

function Base.iterate(iter::NeighborCellIterator)
    di = -1
    dj = -1
    dk = -1
    return _iterate_neighbor_cells(iter, di, dj, dk)
end

function Base.iterate(iter::NeighborCellIterator, state)
    di, dj, dk = state
    return _iterate_neighbor_cells(iter, di, dj, dk)
end

function _iterate_neighbor_cells(iter::NeighborCellIterator, di::Int, dj::Int, dk::Int)
    i, j, k, ncell = iter.i, iter.j, iter.k, iter.ncell
    
    # Increment indices
    dk += 1
    if dk > 1
        dk = -1
        dj += 1
        if dj > 1
            dj = -1
            di += 1
            if di > 1
                return nothing  # Done
            end
        end
    end
    
    # Compute neighbor cell indices with periodic wrapping
    cell_i = ((i - 1 + di + ncell) % ncell) + 1
    cell_j = ((j - 1 + dj + ncell) % ncell) + 1
    cell_k = ((k - 1 + dk + ncell) % ncell) + 1
    
    cell_idx = cell_index(cell_i, cell_j, cell_k, ncell)
    return (cell_idx, (di, dj, dk))
end

"""
    iterate_particles_in_cell(cell_bins, cell_idx)

Iterate over particle indices in a given cell.
Returns an iterator that yields particle indices.
"""
function iterate_particles_in_cell(cell_bins::CellBins, cell_idx::Int)
    start_idx = cell_bins.cell_offsets[cell_idx]
    end_idx = cell_bins.cell_offsets[cell_idx + 1] - 1
    return start_idx:end_idx
end

"""
    for_each_neighbor(cell_bins, posx, posy, posz, particle_idx, f)

Call function `f(neighbor_idx, dr_x, dr_y, dr_z, r_sq)` for each neighbor of particle `particle_idx`.
Uses array-based cell binning for efficient neighbor finding.
Zero-allocation inner loop.
"""
function for_each_neighbor(
    cell_bins::CellBins,
    posx::AbstractVector{<:Real},
    posy::AbstractVector{<:Real},
    posz::AbstractVector{<:Real},
    particle_idx::Int,
    f::Function
)
    L = cell_bins.L
    rc = cell_bins.rc
    rc_sq = rc * rc
    
    # Get particle's cell
    cell_idx = cell_bins.particle_cell[particle_idx]
    
    # Get particle position
    px = posx[particle_idx]
    py = posy[particle_idx]
    pz = posz[particle_idx]
    
    # Iterate over neighbor cells
    for neighbor_cell_idx in iterate_neighbor_cells(cell_idx, cell_bins.ncell)
        # Iterate over particles in this neighbor cell
        for idx in iterate_particles_in_cell(cell_bins, neighbor_cell_idx)
            neighbor_idx = cell_bins.cell_particles[idx]
            
            if neighbor_idx != particle_idx
                # Compute distance vector components (no allocation)
                dr_x = posx[neighbor_idx] - px
                dr_y = posy[neighbor_idx] - py
                dr_z = posz[neighbor_idx] - pz
                
                # Apply minimum image convention (pure scalar ops)
                dr_x = dr_x - L * floor(dr_x / L + 0.5)
                dr_y = dr_y - L * floor(dr_y / L + 0.5)
                dr_z = dr_z - L * floor(dr_z / L + 0.5)
                
                r_sq = dr_x * dr_x + dr_y * dr_y + dr_z * dr_z
                
                if r_sq < rc_sq && r_sq > 0.0
                    f(neighbor_idx, dr_x, dr_y, dr_z, r_sq)
                end
            end
        end
    end
    return nothing
end
