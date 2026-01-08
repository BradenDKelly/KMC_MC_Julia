using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

p, st = MolSim.MC.init_fcc(N=864, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=1234)
T = 1.0 / p.β

# Warmup sweeps
println("Warmup: 100 sweeps")
for i in 1:100
    MolSim.MC.sweep!(st, p)
end

# Production sweeps with block averaging
println("Production: 500 sweeps")
sample_every = 5
block_size = 20
energy_ba = MolSim.MC.BlockAverager(block_size)
pressure_ba = MolSim.MC.BlockAverager(block_size)
mu_ex_ba = MolSim.MC.BlockAverager(block_size)
widom_acc = MolSim.MC.WidomAccumulator()
total_acc = 0.0

for sweep_idx in 1:500
    acc = MolSim.MC.sweep!(st, p)
    global total_acc
    total_acc += acc
    
    if sweep_idx % sample_every == 0
        # Sample energy and pressure
        E_total = MolSim.MC.total_energy(st, p)
        u_per_particle = E_total / st.N
        P = MolSim.MC.pressure(st, p, T)
        push!(energy_ba, u_per_particle)
        push!(pressure_ba, P)
        
        # Widom insertions
        MolSim.MC.reset!(widom_acc)
        MolSim.MC.widom_mu_ex!(widom_acc, st, p; ninsert=50)
        μ_ex = MolSim.MC.mu_ex(widom_acc, p.β)
        push!(mu_ex_ba, μ_ex)
    end
end

avg_acc = total_acc / 500.0
u_mean = MolSim.MC.mean(energy_ba)
u_se = MolSim.MC.stderr(energy_ba)
p_mean = MolSim.MC.mean(pressure_ba)
p_se = MolSim.MC.stderr(pressure_ba)
mu_ex_mean = MolSim.MC.mean(mu_ex_ba)
mu_ex_se = MolSim.MC.stderr(mu_ex_ba)

println("\nResults:")
println("  Acceptance ratio: $(round(avg_acc, digits=4))")
println("  Energy per particle: $(round(u_mean, digits=6)) ± $(round(u_se, digits=6))")
println("  Pressure: $(round(p_mean, digits=6)) ± $(round(p_se, digits=6))")
println("  Excess chemical potential: $(round(mu_ex_mean, digits=6)) ± $(round(mu_ex_se, digits=6))")
