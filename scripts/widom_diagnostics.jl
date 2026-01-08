using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using StaticArrays

# Initialize NVT LJ system
N = 256
ρ = 0.8
T = 1.0
rc = 2.5
max_disp = 0.1

p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp, seed=42)

# Warmup
for i in 1:50
    MolSim.MC.sweep!(st, p)
end

# Diagnostic: compute rmin and ΔU for random insertions
n = 200
L = st.L
pos = st.pos
scratch_dr = MVector{3,Float64}(0.0, 0.0, 0.0)

println("rmin, ΔU")
println("-" ^ 30)

counts = Dict(0.8 => 0, 0.6 => 0, 0.4 => 0, 0.3 => 0, 0.2 => 0)
rmin_values = Float64[]

for i in 1:n
    # Random insertion point
    test_x = rand(st.rng) * L
    test_y = rand(st.rng) * L
    test_z = rand(st.rng) * L
    
    # Compute minimum distance to any particle
    rmin_sq = Inf
    @inbounds for j in 1:N
        dr_x = pos[1, j] - test_x
        dr_y = pos[2, j] - test_y
        dr_z = pos[3, j] - test_z
        
        # Minimum image
        L_half = L / 2.0
        if dr_x > L_half
            dr_x = dr_x - L
        elseif dr_x < -L_half
            dr_x = dr_x + L
        end
        if dr_y > L_half
            dr_y = dr_y - L
        elseif dr_y < -L_half
            dr_y = dr_y + L
        end
        if dr_z > L_half
            dr_z = dr_z - L
        elseif dr_z < -L_half
            dr_z = dr_z + L
        end
        
        r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
        if r2 < rmin_sq
            rmin_sq = r2
        end
    end
    
    rmin = sqrt(rmin_sq)
    push!(rmin_values, rmin)
    
    # Count thresholds
    if rmin < 0.8
        counts[0.8] += 1
    end
    if rmin < 0.6
        counts[0.6] += 1
    end
    if rmin < 0.4
        counts[0.4] += 1
    end
    if rmin < 0.3
        counts[0.3] += 1
    end
    if rmin < 0.2
        counts[0.2] += 1
    end
    
    # Compute ΔU
    ΔU = MolSim.MC.widom_deltaU(st, p)
    
    println("$(round(rmin, digits=6)), $(round(ΔU, digits=6))")
end

println("\nCounts of rmin below thresholds:")
println("  rmin < 0.8: $(counts[0.8])")
println("  rmin < 0.6: $(counts[0.6])")
println("  rmin < 0.4: $(counts[0.4])")
println("  rmin < 0.3: $(counts[0.3])")
println("  rmin < 0.2: $(counts[0.2])")
