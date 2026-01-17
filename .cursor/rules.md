# Julia performance rules (must follow)

- No global variables. Everything inside modules/functions.
- Hot loops must be allocation-free: 0 allocations per step/sweep is the goal.
- Use concrete types everywhere (no AbstractVector/Any fields in structs).
- Prefer mutating functions with ! and preallocated buffers.
- For 3D vectors use StaticArrays (SVector/MVector), not Vector{Float64}.
- No comprehensions, broadcasting, map, filter, collect in hot paths.
- No temporary arrays in energy/force loops; use scalars.
- Any container for events/particles must be Vector{T} with concrete T.
- Every new feature must include:
  - @btime benchmark
  - @allocated check
  - @code_warntype check for key kernels
- Rebuild neighbor lists/event tables on schedule; incremental updates must be mutation-only.
