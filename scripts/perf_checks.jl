using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using BenchmarkTools

# Test with LRC enabled
p, st = MolSim.MC.init_fcc(N=864, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=1234, use_lrc=true)
T = 1.0 / p.β

# Allocated bytes check - wrap in local functions to avoid global scope effects
function measure_trial_alloc()
    Base.@allocated MolSim.MC.mc_trial!(st, p)
end

function measure_sweep_alloc()
    Base.@allocated MolSim.MC.sweep!(st, p)
end

function measure_widom_alloc()
    Base.@allocated MolSim.MC.widom_deltaU(st, p)
end

function measure_total_energy_alloc()
    Base.@allocated MolSim.MC.total_energy(st, p)
end

function measure_pressure_alloc()
    Base.@allocated MolSim.MC.pressure(st, p, T)
end

allocated_trial = measure_trial_alloc()
println("@allocated mc_trial! = $allocated_trial")

allocated_sweep = measure_sweep_alloc()
println("@allocated sweep! = $allocated_sweep")

allocated_widom = measure_widom_alloc()
println("@allocated widom_deltaU = $allocated_widom")

allocated_total_energy = measure_total_energy_alloc()
println("@allocated total_energy (with LRC) = $allocated_total_energy")

allocated_pressure = measure_pressure_alloc()
println("@allocated pressure (with LRC) = $allocated_pressure")

# Compare with LRC disabled to verify LRC addition doesn't add allocations
p_no_lrc, st_no_lrc = MolSim.MC.init_fcc(N=864, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=1234, use_lrc=false)
T_no_lrc = 1.0 / p_no_lrc.β

function measure_total_energy_alloc_no_lrc()
    Base.@allocated MolSim.MC.total_energy(st_no_lrc, p_no_lrc)
end

function measure_pressure_alloc_no_lrc()
    Base.@allocated MolSim.MC.pressure(st_no_lrc, p_no_lrc, T_no_lrc)
end

allocated_total_energy_no_lrc = measure_total_energy_alloc_no_lrc()
allocated_pressure_no_lrc = measure_pressure_alloc_no_lrc()
println("@allocated total_energy (no LRC) = $allocated_total_energy_no_lrc")
println("@allocated pressure (no LRC) = $allocated_pressure_no_lrc")

# Code warntype check
println("\n@code_warntype mc_trial!:")
using InteractiveUtils
code_warntype(MolSim.MC.mc_trial!, (typeof(st), typeof(p)))

# Benchmark with BenchmarkTools
println("\nBenchmark:")
@btime MolSim.MC.sweep!($st, $p)
@btime MolSim.MC.total_energy($st, $p)
@btime MolSim.MC.pressure($st, $p, $T)