using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using BenchmarkTools

p, st = MolSim.MC.init_fcc(N=108, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42)

println("LJ single sweep benchmark (no allocations expected):")
@btime MolSim.MC.sweep!($st, $p; rebuild_every=$st.N)
