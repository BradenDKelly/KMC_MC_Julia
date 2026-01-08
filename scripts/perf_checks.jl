using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using BenchmarkTools

p, st = MolSim.MC.init_fcc(N=864, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=1234)

# Allocated bytes check
allocated_trial = @allocated MolSim.MC.mc_trial!(st, p)
println("@allocated mc_trial! = $allocated_trial")

allocated_sweep = @allocated MolSim.MC.sweep!(st, p)
println("@allocated sweep! = $allocated_sweep")

allocated_widom = @allocated MolSim.MC.widom_deltaU(st, p)
println("@allocated widom_deltaU = $allocated_widom")

# Code warntype check
println("\n@code_warntype mc_trial!:")
using InteractiveUtils
code_warntype(MolSim.MC.mc_trial!, (typeof(st), typeof(p)))

# Benchmark
println("\nBenchmark:")
@btime MolSim.MC.sweep!($st, $p)
