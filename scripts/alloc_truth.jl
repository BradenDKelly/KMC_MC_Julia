using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

# Initialize with LRC enabled
p_lrc, st_lrc = MolSim.MC.init_fcc(N=864, ρ=0.8, T=1.0, rc=2.5, use_lrc=true)
T = 1.0 / p_lrc.β

# Warm up
MolSim.MC.total_energy(st_lrc, p_lrc)
MolSim.MC.pressure(st_lrc, p_lrc, T)

# Measure allocations with LRC
alloc_total_energy_lrc = @allocated MolSim.MC.total_energy(st_lrc, p_lrc)
alloc_pressure_lrc = @allocated MolSim.MC.pressure(st_lrc, p_lrc, T)

println("@allocated total_energy (with LRC) = $alloc_total_energy_lrc")
println("@allocated pressure (with LRC) = $alloc_pressure_lrc")

# Initialize with LRC disabled
p_no_lrc, st_no_lrc = MolSim.MC.init_fcc(N=864, ρ=0.8, T=1.0, rc=2.5, use_lrc=false)
T_no_lrc = 1.0 / p_no_lrc.β

# Warm up
MolSim.MC.total_energy(st_no_lrc, p_no_lrc)
MolSim.MC.pressure(st_no_lrc, p_no_lrc, T_no_lrc)

# Measure allocations without LRC
alloc_total_energy_no_lrc = @allocated MolSim.MC.total_energy(st_no_lrc, p_no_lrc)
alloc_pressure_no_lrc = @allocated MolSim.MC.pressure(st_no_lrc, p_no_lrc, T_no_lrc)

println("@allocated total_energy (no LRC) = $alloc_total_energy_no_lrc")
println("@allocated pressure (no LRC) = $alloc_pressure_no_lrc")
