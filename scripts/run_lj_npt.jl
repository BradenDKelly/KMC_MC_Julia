using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

# Simulation parameters
N = 864
T = 1.0
rc = 2.5
max_disp = 0.1
max_dlnV = 0.01
vol_move_every = 10
Pext_values = [0.5, 1.0]
sample_every = 10
block_size = 20

# Initial density
ρ_init = 0.8

println("==========================================")
println("LJ NPT Monte Carlo Simulation")
println("==========================================")
println("N = $N")
println("T = $T")
println("Initial density = $ρ_init")
println("Volume move every $vol_move_every sweeps")
println()

for Pext in Pext_values
    println("------------------------------------------")
    println("Pext = $Pext")
    println("------------------------------------------")
    
    # Initialize
    p, st = MolSim.MC.init_fcc(N=N, ρ=ρ_init, T=T, rc=rc, max_disp=max_disp, seed=42)
    
    # Warmup
    println("Warmup: 100 sweeps")
    for i in 1:100
        MolSim.MC.sweep!(st, p)
        if i % vol_move_every == 0
            MolSim.MC.volume_trial!(st, p; max_dlnV=max_dlnV, Pext=Pext)
        end
    end
    
    # Production with block averaging
    println("Production: 1000 sweeps")
    density_ba = MolSim.MC.BlockAverager(block_size)
    energy_ba = MolSim.MC.BlockAverager(block_size)
    pressure_ba = MolSim.MC.BlockAverager(block_size)
    
    particle_accepted = 0
    particle_attempted = 0
    volume_accepted = 0
    volume_attempted = 0
    
    for sweep_idx in 1:1000
        acc = MolSim.MC.sweep!(st, p)
        particle_accepted += Int(round(acc * N))
        particle_attempted += N
        
        if sweep_idx % vol_move_every == 0
            vol_acc = MolSim.MC.volume_trial!(st, p; max_dlnV=max_dlnV, Pext=Pext)
            if vol_acc
                volume_accepted += 1
            end
            volume_attempted += 1
        end
        
        if sweep_idx % sample_every == 0
            ρ = N / (st.L * st.L * st.L)
            E_total = MolSim.MC.total_energy(st, p)
            u_per_particle = E_total / N
            P = MolSim.MC.pressure(st, p, T)
            push!(density_ba, ρ)
            push!(energy_ba, u_per_particle)
            push!(pressure_ba, P)
        end
    end
    
    particle_acceptance = Float64(particle_accepted) / Float64(particle_attempted)
    volume_acceptance = volume_attempted > 0 ? Float64(volume_accepted) / Float64(volume_attempted) : 0.0
    
    ρ_mean = MolSim.MC.mean(density_ba)
    ρ_se = MolSim.MC.stderr(density_ba)
    u_mean = MolSim.MC.mean(energy_ba)
    u_se = MolSim.MC.stderr(energy_ba)
    p_mean = MolSim.MC.mean(pressure_ba)
    p_se = MolSim.MC.stderr(pressure_ba)
    
    println("\nResults:")
    println("  Particle acceptance: $(round(particle_acceptance, digits=4))")
    println("  Volume acceptance: $(round(volume_acceptance, digits=4))")
    println("  Mean density: $(round(ρ_mean, digits=4)) ± $(round(ρ_se, digits=4))")
    println("  Mean energy per particle: $(round(u_mean, digits=6)) ± $(round(u_se, digits=6))")
    println("  Mean pressure: $(round(p_mean, digits=6)) ± $(round(p_se, digits=6))")
    println()
end

println("==========================================")
