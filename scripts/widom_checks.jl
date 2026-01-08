using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using StaticArrays

# Small NVT system
N = 256
ρ = 0.8
T = 1.0
rc = 2.5
max_disp = 0.1

p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp, seed=42)

# Check allocation
allocated = @allocated MolSim.MC.widom_deltaU(st, p)
println("@allocated widom_deltaU = $allocated")

# Sample a few ΔU values and compute rmin statistics
println("\nSample ΔU values and rmin:")
n_samples = 50
L = st.L
pos = st.pos
rmin_values = Float64[]

for i in 1:n_samples
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
    
    # Compute ΔU
    ΔU = MolSim.MC.widom_deltaU(st, p)
    if i <= 5
        println("  ΔU[$i] = $(round(ΔU, digits=6)), rmin = $(round(rmin, digits=6))")
    end
end

# rmin statistics
rmin_min = minimum(rmin_values)
rmin_mean = sum(rmin_values) / length(rmin_values)
println("\nrmin statistics (over $n_samples samples):")
println("  min(rmin) = $(round(rmin_min, digits=6))")
println("  mean(rmin) = $(round(rmin_mean, digits=6))")

# Sanity check: u(r) for r=0.7
r_test = 0.7
r2_test = r_test * r_test
# Access the helper function via reflection or compute directly
# Since it's internal, we'll compute it directly for the sanity check
σ2 = p.σ * p.σ
invr2 = σ2 / r2_test
invr6 = invr2 * invr2 * invr2
u_test = 4.0 * p.ϵ * (invr6 * invr6 - invr6)
println("\nSanity check: u(r) for r=$(r_test):")
println("  u($(r_test)) = $(round(u_test, digits=6))")

# Compute μ_ex from insertions (plain Widom)
n_plain = 200
println("\nComputing μ_ex from $n_plain insertions (plain Widom)...")
widom_acc = MolSim.MC.WidomAccumulator()
μ_ex = MolSim.MC.widom_mu_ex!(widom_acc, st, p; ninsert=n_plain)
println("μ_ex (plain) = $(round(μ_ex, digits=6))")

# Compute μ_ex using cavity-biased insertion
n_cavity = 50
max_total_trials = 20000
println("\nComputing μ_ex using cavity-biased insertion...")
for rmin_cut in [0.85, 0.9]
    widom_acc_cavity = MolSim.MC.WidomAccumulator()
    β = p.β
    local L = st.L
    total_trials = 0
    accepted_trials = 0
    capped = false
    
    for k in 1:n_cavity
        # Repeatedly sample until rmin > rmin_cut or cap hit
        while true
            total_trials += 1
            if total_trials > max_total_trials
                capped = true
                break
            end
            
            test_x = rand(st.rng) * L
            test_y = rand(st.rng) * L
            test_z = rand(st.rng) * L
            
            rmin = MolSim.MC.rmin_to_particles(st, (test_x, test_y, test_z))
            
            if rmin > rmin_cut
                accepted_trials += 1
                # Compute ΔU at this point
                ΔU = MolSim.MC.widom_deltaU_at_point(st, p, test_x, test_y, test_z)
                Base.push!(widom_acc_cavity, β, ΔU)
                break
            end
        end
        
        if capped
            break
        end
    end
    
    pbias = Float64(accepted_trials) / Float64(total_trials)
    
    if capped
        println("  cavity run capped (rmin_cut=$(rmin_cut)): pbias = $(round(pbias, digits=6))")
    else
        # Bias correction: <exp(-βΔU)>_unbiased = pbias * <exp(-βΔU)>_conditional
        # μ_ex = -(1/β)*log(pbias) - (1/β)*log(<exp(-βΔU)>_conditional)
        local T = 1.0 / β
        μ_ex_conditional = MolSim.MC.mu_ex(widom_acc_cavity, β)
        μ_ex_cavity = -T * log(pbias) + μ_ex_conditional
        println("  μ_ex (cavity, rmin_cut=$(rmin_cut)) = $(round(μ_ex_cavity, digits=6)), pbias = $(round(pbias, digits=6))")
    end
end
