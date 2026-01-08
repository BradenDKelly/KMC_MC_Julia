using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

function main()
    # Settings
    N = 864
    T = 1.0
    ρ = 0.8
    rc = 2.5
    use_lrc = true
    seed = 12345
    warmup_sweeps = 200
    prod_sweeps = 2000
    
    # Initialize
    p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=seed, use_lrc=use_lrc)
    T_val = 1.0 / p.β
    
    println("LJ NVT Regression Run")
    println("N=$N, T=$T, ρ=$ρ, rc=$rc, LRC=$use_lrc, seed=$seed")
    
    # Warmup
    println("Warmup: $warmup_sweeps sweeps")
    for i in 1:warmup_sweeps
        MolSim.MC.sweep!(st, p)
    end
    
    # Production
    println("Production: $prod_sweeps sweeps")
    sample_every = 10
    total_acc = 0.0
    energy_samples = Float64[]
    pressure_samples = Float64[]
    
    for sweep_idx in 1:prod_sweeps
        acc = MolSim.MC.sweep!(st, p)
        total_acc += acc
        
        if sweep_idx % sample_every == 0
            E_total = MolSim.MC.total_energy(st, p)
            u_per_particle = E_total / st.N
            P = MolSim.MC.pressure(st, p, T_val)
            push!(energy_samples, u_per_particle)
            push!(pressure_samples, P)
        end
    end
    
    avg_acc = total_acc / prod_sweeps
    u_mean = sum(energy_samples) / length(energy_samples)
    p_mean = sum(pressure_samples) / length(pressure_samples)
    
    # Print summary
    println("\nSummary:")
    println("  Acceptance ratio: $(round(avg_acc, digits=4))")
    if p.use_lrc
        u_sampled = u_mean - p.lrc_u_per_particle
        p_sampled = p_mean - p.lrc_p
        println("  Energy per particle (sampled): $(round(u_sampled, digits=6))")
        println("  Energy per particle (corrected): $(round(u_mean, digits=6))")
        println("  Pressure (sampled): $(round(p_sampled, digits=6))")
        println("  Pressure (corrected): $(round(p_mean, digits=6))")
    else
        println("  Energy per particle: $(round(u_mean, digits=6))")
        println("  Pressure: $(round(p_mean, digits=6))")
    end
end

main()
