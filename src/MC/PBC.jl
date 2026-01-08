"""
Periodic boundary conditions utilities.
"""

using StaticArrays

"""
    wrap!(x::MVector{3,Float64}, L::Float64)

Wrap a 3D position vector to be within [0, L) in each dimension.
Modifies `x` in-place.
"""
function wrap!(x::MVector{3,Float64}, L::Float64)
    @inbounds begin
        x[1] = x[1] - L * floor(x[1] / L)
        x[2] = x[2] - L * floor(x[2] / L)
        x[3] = x[3] - L * floor(x[3] / L)
    end
    return nothing
end

"""
    minimum_image!(dr::MVector{3,Float64}, L::Float64)

Apply minimum image convention to a displacement vector `dr` with box length `L`.
Modifies `dr` in-place. Components will be in [-L/2, L/2].
"""
function minimum_image!(dr::MVector{3,Float64}, L::Float64)
    L_half = L / 2.0
    @inbounds begin
        if dr[1] > L_half
            dr[1] = dr[1] - L
        elseif dr[1] < -L_half
            dr[1] = dr[1] + L
        end
        if dr[2] > L_half
            dr[2] = dr[2] - L
        elseif dr[2] < -L_half
            dr[2] = dr[2] + L
        end
        if dr[3] > L_half
            dr[3] = dr[3] - L
        elseif dr[3] < -L_half
            dr[3] = dr[3] + L
        end
    end
    return nothing
end
