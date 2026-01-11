#!/usr/bin/env julia
"""
NVT and NPT comparison script for LJ fluid.
Runs NVT simulation, then NPT at the average pressure from NVT.
Computes heat capacity, RDF, energy, pressure, and compares with EOS.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using Random

# ============================================================================
# Parameters
# ============================================================================
const N = 500
const ρ_nvt = 0.7
const T = 1.0
const rc = 2.5
const max_disp = 0.1
const max_dlnV = 0.01
const vol_move_every = 10
const seed_nvt = 12345
const seed_npt = 12346

const warmup_sweeps = 1000
const prod_sweeps = 3000
const sample_every = 10
const block_size = 50

# RDF parameters
const rmax = nothing  # Will be set to L/2
const nbins_rdf = 200

# Output directory
const output_dir = joinpath(@__DIR__, "out")
mkpath(output_dir)

# ============================================================================
# Helper functions
# ============================================================================

"""
Write RDF to CSV file.
"""
function write_rdf_csv(path::String, r::Vector{Float64}, g_r::Vector{Float64})
    open(path, "w") do io
        println(io, "r,g_r")
        for i in eachindex(r)
            println(io, "$(r[i]),$(g_r[i])")
        end
    end
    return nothing
end

# ============================================================================
# NVT Simulation
# ============================================================================

println("=" ^ 80)
println("NVT Monte Carlo Simulation")
println("=" ^ 80)
println("\nParameters:")
println("  N = $N")
println("  ρ = $ρ_nvt")
println("  T = $T")
println("  rc = $rc")
println("  warmup_sweeps = $warmup_sweeps")
println("  prod_sweeps = $prod_sweeps")
println()

# Initialize NVT
p_nvt, st_nvt = MolSim.MC.init_fcc(N=N, ρ=ρ_nvt, T=T, rc=rc, max_disp=max_disp, 
                                   seed=seed_nvt, use_lrc=false, lj_model=:truncated)
L_nvt = st_nvt.L
println("  Box length L = $L_nvt")

# Warmup
println("\nWarmup: $warmup_sweeps sweeps...")
for _ in 1:warmup_sweeps
    MolSim.MC.sweep!(st_nvt, p_nvt; rebuild_every=1)
end

# Production: sample energy, pressure, and compute C_V
println("Production: $prod_sweeps sweeps...")
U_samples = Float64[]
P_samples = Float64[]

for sweep in 1:prod_sweeps
    MolSim.MC.sweep!(st_nvt, p_nvt; rebuild_every=1)
    if sweep % sample_every == 0
        U = MolSim.MC.total_energy(st_nvt, p_nvt)
        P = MolSim.MC.pressure(st_nvt, p_nvt, T)
        push!(U_samples, U)
        push!(P_samples, P)
    end
end

# Compute statistics using block averaging
energy_ba = MolSim.MC.BlockAverager(block_size)
pressure_ba = MolSim.MC.BlockAverager(block_size)
for u in U_samples
    push!(energy_ba, u / Float64(N))  # Per-particle energy
end
for p_val in P_samples
    push!(pressure_ba, p_val)
end

U_mean = MolSim.MC.mean(energy_ba)
U_se = MolSim.MC.stderr(energy_ba)
P_mean = MolSim.MC.mean(pressure_ba)
P_se = MolSim.MC.stderr(pressure_ba)

# Compute C_V from energy fluctuations
C_V, U_mean_total = MolSim.MC.heat_capacity_CV(U_samples, T, block_size)
C_V_per_particle = C_V / Float64(N)

# Compute final RDF
rmax_actual = isnothing(rmax) ? L_nvt / 2.0 : rmax
println("\nComputing RDF for final NVT configuration...")
r_rdf, g_rdf = MolSim.Analysis.rdf(st_nvt.pos, L_nvt, rmax_actual, nbins_rdf)
write_rdf_csv(joinpath(output_dir, "rdf_nvt_rho_0.7.csv"), r_rdf, g_rdf)

# EOS comparison
println("\nEOS Comparison (NVT):")
Z_mc = P_mean / (ρ_nvt * T)
println("  MC: Z = $Z_mc")
try
    P_eos = MolSim.EOS.pressure_kolafa_sklogwiki(T, ρ_nvt)
    Z_eos = P_eos / (ρ_nvt * T)
    println("  EOS (Kolafa-Sklogwiki): Z = $Z_eos")
    println("  Difference: ΔZ = $(Z_mc - Z_eos)")
catch e
    println("  EOS evaluation failed: $e")
end

println("\nNVT Results:")
println("  Energy per particle: $(round(U_mean, digits=6)) ± $(round(U_se, digits=6))")
println("  Pressure: $(round(P_mean, digits=6)) ± $(round(P_se, digits=6))")
println("  Compressibility factor Z: $(round(Z_mc, digits=6))")
println("  Heat capacity C_V/N: $(round(C_V_per_particle, digits=6))")
println()

# Store average pressure for NPT
P_target = P_mean

# ============================================================================
# NPT Simulation
# ============================================================================

println("=" ^ 80)
println("NPT Monte Carlo Simulation")
println("=" ^ 80)
println("\nParameters:")
println("  N = $N")
println("  T = $T")
println("  P_target = $P_target")
println("  rc = $rc")
println("  Initial density (from NVT): $ρ_nvt")
println("  warmup_sweeps = $warmup_sweeps")
println("  prod_sweeps = $prod_sweeps")
println()

# Initialize NPT (start at NVT density)
p_npt, st_npt = MolSim.MC.init_fcc(N=N, ρ=ρ_nvt, T=T, rc=rc, max_disp=max_disp,
                                   seed=seed_npt, use_lrc=false, lj_model=:truncated)

# Warmup
println("Warmup: $warmup_sweeps sweeps...")
for sweep in 1:warmup_sweeps
    MolSim.MC.sweep!(st_npt, p_npt; rebuild_every=1)
    if sweep % vol_move_every == 0
        MolSim.MC.volume_trial!(st_npt, p_npt; max_dlnV=max_dlnV, Pext=P_target)
    end
end

# Production: sample density, energy, pressure
println("Production: $prod_sweeps sweeps...")
ρ_samples = Float64[]
U_samples_npt = Float64[]
P_samples_npt = Float64[]

for sweep in 1:prod_sweeps
    MolSim.MC.sweep!(st_npt, p_npt; rebuild_every=1)
    if sweep % vol_move_every == 0
        MolSim.MC.volume_trial!(st_npt, p_npt; max_dlnV=max_dlnV, Pext=P_target)
    end
    if sweep % sample_every == 0
        V = st_npt.L * st_npt.L * st_npt.L
        ρ = Float64(N) / V
        U = MolSim.MC.total_energy(st_npt, p_npt)
        P = MolSim.MC.pressure(st_npt, p_npt, T)
        push!(ρ_samples, ρ)
        push!(U_samples_npt, U / Float64(N))  # Per-particle energy
        push!(P_samples_npt, P)
    end
end

# Compute statistics using block averaging
density_ba = MolSim.MC.BlockAverager(block_size)
energy_ba_npt = MolSim.MC.BlockAverager(block_size)
pressure_ba_npt = MolSim.MC.BlockAverager(block_size)
for ρ_val in ρ_samples
    push!(density_ba, ρ_val)
end
for u in U_samples_npt
    push!(energy_ba_npt, u)
end
for p_val in P_samples_npt
    push!(pressure_ba_npt, p_val)
end

ρ_mean_npt = MolSim.MC.mean(density_ba)
ρ_se_npt = MolSim.MC.stderr(density_ba)
U_mean_npt = MolSim.MC.mean(energy_ba_npt)
U_se_npt = MolSim.MC.stderr(energy_ba_npt)
P_mean_npt = MolSim.MC.mean(pressure_ba_npt)
P_se_npt = MolSim.MC.stderr(pressure_ba_npt)

# Compute κ_T from volume fluctuations
V_samples = [Float64(N) / ρ for ρ in ρ_samples]
κ_T, mean_V = MolSim.MC.compressibility_kappaT(V_samples, T, block_size)

# Compute final RDF for NPT
L_npt_final = st_npt.L
rmax_npt = L_npt_final / 2.0
println("\nComputing RDF for final NPT configuration...")
r_rdf_npt, g_rdf_npt = MolSim.Analysis.rdf(st_npt.pos, L_npt_final, rmax_npt, nbins_rdf)
write_rdf_csv(joinpath(output_dir, "rdf_npt_rho_0.7.csv"), r_rdf_npt, g_rdf_npt)

# EOS comparison for NPT
println("\nEOS Comparison (NPT):")
Z_mc_npt = P_mean_npt / (ρ_mean_npt * T)
println("  MC: Z = $Z_mc_npt (at ρ = $ρ_mean_npt)")
try
    P_eos_npt = MolSim.EOS.pressure_kolafa_sklogwiki(T, ρ_mean_npt)
    Z_eos_npt = P_eos_npt / (ρ_mean_npt * T)
    println("  EOS (Kolafa-Sklogwiki): Z = $Z_eos_npt")
    println("  Difference: ΔZ = $(Z_mc_npt - Z_eos_npt)")
catch e
    println("  EOS evaluation failed: $e")
end

println("\nNPT Results:")
println("  Average density: $(round(ρ_mean_npt, digits=6)) ± $(round(ρ_se_npt, digits=6))")
println("  Target density (NVT): $ρ_nvt")
println("  Difference: Δρ = $(round(ρ_mean_npt - ρ_nvt, digits=6))")
println("  Z-score: $(round((ρ_mean_npt - ρ_nvt) / ρ_se_npt, digits=2))")
println("  Energy per particle: $(round(U_mean_npt, digits=6)) ± $(round(U_se_npt, digits=6))")
println("  Pressure: $(round(P_mean_npt, digits=6)) ± $(round(P_se_npt, digits=6))")
println("  Compressibility factor Z: $(round(Z_mc_npt, digits=6))")
println("  Isothermal compressibility κ_T: $(round(κ_T, digits=6))")
println()

# ============================================================================
# Summary
# ============================================================================

println("=" ^ 80)
println("Summary")
println("=" ^ 80)
println("\nNVT (ρ = $ρ_nvt):")
println("  U/N = $(round(U_mean, digits=6)) ± $(round(U_se, digits=6))")
println("  P = $(round(P_mean, digits=6)) ± $(round(P_se, digits=6))")
println("  Z = $(round(Z_mc, digits=6))")
println("  C_V/N = $(round(C_V_per_particle, digits=6))")

println("\nNPT (P = $(round(P_target, digits=6))):")
println("  ρ = $(round(ρ_mean_npt, digits=6)) ± $(round(ρ_se_npt, digits=6))")
println("  U/N = $(round(U_mean_npt, digits=6)) ± $(round(U_se_npt, digits=6))")
println("  P = $(round(P_mean_npt, digits=6)) ± $(round(P_se_npt, digits=6))")
println("  Z = $(round(Z_mc_npt, digits=6))")
println("  κ_T = $(round(κ_T, digits=6))")

println("\nNVT-NPT Consistency:")
println("  ρ_NVT (target) = $ρ_nvt")
println("  ρ_NPT (measured) = $(round(ρ_mean_npt, digits=6)) ± $(round(ρ_se_npt, digits=6))")
println("  Difference = $(round(ρ_mean_npt - ρ_nvt, digits=6))")
z_score = (ρ_mean_npt - ρ_nvt) / ρ_se_npt
println("  Z-score = $(round(z_score, digits=2))")

println("\nOutput files written to $output_dir:")
println("  rdf_nvt_rho_0.7.csv - NVT radial distribution function")
println("  rdf_npt_rho_0.7.csv - NPT radial distribution function")
println()
