"""
Periodic boundary conditions utilities.
"""

"""
    wrap!(positions, L)

Wrap all particle positions to be within [0, L) in each dimension.
Modifies `positions` in-place.
"""
function wrap!(positions::AbstractMatrix, L::Real)
    @inbounds for i in 1:size(positions, 2)
        for d in 1:size(positions, 1)
            x = positions[d, i]
            positions[d, i] = x - L * floor(x / L)
        end
    end
    return nothing
end

"""
    wrap(x, L)

Wrap a single coordinate `x` to be within [0, L).
"""
wrap(x::Real, L::Real) = x - L * floor(x / L)

"""
    minimum_image(dr, L)

Apply minimum image convention to a displacement vector `dr` with box length `L`.
Returns the wrapped displacement vector.
"""
function minimum_image(dr::AbstractVector, L::Real)
    @inbounds for i in eachindex(dr)
        dr_i = dr[i]
        if dr_i > L / 2
            dr[i] = dr_i - L
        elseif dr_i < -L / 2
            dr[i] = dr_i + L
        end
    end
    return dr
end

"""
    minimum_image!(dr, L)

Apply minimum image convention to a displacement vector `dr` with box length `L`.
Modifies `dr` in-place.
"""
function minimum_image!(dr::AbstractVector, L::Real)
    @inbounds for i in eachindex(dr)
        dr_i = dr[i]
        if dr_i > L / 2
            dr[i] = dr_i - L
        elseif dr_i < -L / 2
            dr[i] = dr_i + L
        end
    end
    return nothing
end

"""
    minimum_image(dr_sq, L)

Compute the squared distance using minimum image convention.
Returns the squared distance after applying minimum image.
"""
function minimum_image_distance_sq(dr::AbstractVector, L::Real)
    dr_sq = 0.0
    @inbounds for i in eachindex(dr)
        dr_i = dr[i]
        if dr_i > L / 2
            dr_i = dr_i - L
        elseif dr_i < -L / 2
            dr_i = dr_i + L
        end
        dr_sq += dr_i * dr_i
    end
    return dr_sq
end

"""
    wrap_soa!(posx, posy, posz, L)

Wrap all particle positions to be within [0, L) in each dimension.
Modifies positions in-place (SoA format).
"""
function wrap_soa!(
    posx::AbstractVector{<:Real},
    posy::AbstractVector{<:Real},
    posz::AbstractVector{<:Real},
    L::Real
)
    N = length(posx)
    @inbounds for i in 1:N
        posx[i] = posx[i] - L * floor(posx[i] / L)
        posy[i] = posy[i] - L * floor(posy[i] / L)
        posz[i] = posz[i] - L * floor(posz[i] / L)
    end
    return nothing
end
