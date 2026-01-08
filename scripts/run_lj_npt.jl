using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using Statistics

# Simulation parameters
N = 864
T = 1.0
rc = 2.5
max_disp = 0.1
max_dlnV = 0.01
vol_move_every = 10
Pext_values = [0.5, 1.0]

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
    
    # Production
    println("Production: 1000 sweeps")
    particle_acc, volume_acc, densities, energies = MolSim.MC.run_npt!(
        st, p;
        nsweeps=1000,
        Pext=Pext,
        max_disp=max_disp,
        max_dlnV=max_dlnV,
        vol_move_every=vol_move_every
    )
    
    # Statistics
    ρ_mean = mean(densities)
    ρ_std = std(densities)
    E_mean = mean(energies) / N
    E_std = std(energies) / N
    
    # Compute pressure from final state
    P = MolSim.MC.pressure(st, p)
    
    println("\nResults:")
    println("  Particle acceptance: $(round(particle_acc, digits=4))")
    println("  Volume acceptance: $(round(volume_acc, digits=4))")
    println("  Mean density: $(round(ρ_mean, digits=4)) ± $(round(ρ_std, digits=4))")
    println("  Mean energy per particle: $(round(E_mean, digits=6)) ± $(round(E_std, digits=6))")
    println("  Pressure (from final state): $(round(P, digits=6))")
    println()
end

println("==========================================")
