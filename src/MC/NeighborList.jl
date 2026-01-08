"""
Cell-linked list for efficient neighbor finding.
"""

"""
    CellList

Cell-linked list data structure for efficient neighbor finding.
Uses head/next linked list structure.
"""
struct CellList
    head::Vector{Int}        # head[cell] = first particle in cell (0 if empty)
    next::Vector{Int}        # next[i] = next particle in same cell (0 if none)
    cell_of::Vector{Int}     # cell_of[i] = cell containing particle i
    ncell::Int               # number of cells per dimension
    ncell_total::Int         # total number of cells (ncell^3)
    L::Float64               # box length
    rc::Float64              # cutoff distance
end

"""
    CellList(N, L, rc)

Construct a cell-linked list for `N` particles in a box of length `L` with cutoff `rc`.
"""
function CellList(N::Int, L::Float64, rc::Float64)
    ncell = max(1, floor(Int, L / rc))
    ncell_total = ncell * ncell * ncell
    head = zeros(Int, ncell_total)
    next = zeros(Int, N)
    cell_of = zeros(Int, N)
    return CellList(head, next, cell_of, ncell, ncell_total, L, rc)
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
    i = clamp(floor(Int, x / L * ncell) + 1, 1, ncell)
    j = clamp(floor(Int, y / L * ncell) + 1, 1, ncell)
    k = clamp(floor(Int, z / L * ncell) + 1, 1, ncell)
    return cell_index(i, j, k, ncell)
end

"""
    rebuild_cells!(st)

Rebuild the cell-linked list from current particle positions.
Zero-allocation: reuses existing arrays.
"""
function rebuild_cells!(st)
    # Clear all head pointers
    fill!(st.cl.head, 0)
    fill!(st.cl.next, 0)
    
    N = st.N
    L = st.L
    ncell = st.cl.ncell
    pos = st.pos
    
    # Assign particles to cells
    @inbounds for i in 1:N
        x, y, z = pos[1, i], pos[2, i], pos[3, i]
        # Wrap coordinates to [0, L)
        x_wrapped = x - L * floor(x / L)
        y_wrapped = y - L * floor(y / L)
        z_wrapped = z - L * floor(z / L)
        
        cell_idx = get_cell(x_wrapped, y_wrapped, z_wrapped, L, ncell)
        st.cl.cell_of[i] = cell_idx
        st.cl.next[i] = st.cl.head[cell_idx]
        st.cl.head[cell_idx] = i
    end
    
    return nothing
end
