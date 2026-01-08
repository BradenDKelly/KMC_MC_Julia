using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

p, st = MolSim.MC.init_fcc(N=864, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=1234)

# Warmup sweeps
println("Warmup: 100 sweeps")
for i in 1:100
    MolSim.MC.sweep!(st, p)
end

# Production sweeps
println("Production: 100 sweeps")
total_acc = 0.0
for i in 1:100
    global total_acc
    acc = MolSim.MC.sweep!(st, p)
    total_acc += acc
end

avg_acc = total_acc / 100.0
E_total = MolSim.MC.total_energy(st, p)
E_per_particle = E_total / st.N
P = MolSim.MC.pressure(st, p)

println("\nResults:")
println("  Acceptance ratio: $(round(avg_acc, digits=4))")
println("  Energy per particle: $(round(E_per_particle, digits=6))")
println("  Pressure: $(round(P, digits=6))")