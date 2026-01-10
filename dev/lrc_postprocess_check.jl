"""
Diagnostic script to verify LRC as deterministic post-processing on identical configurations.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

# Load Kolafa SklogWiki implementation
include(joinpath(@__DIR__, "..", "src", "EOS", "LJKolafaSklogwiki1994.jl"))

# Simulation parameters
T = 2.0
ρ = 0.2
N = 864
rc = 2.5
max_disp = 0.1
warmup_sweeps = 100
production_sweeps = 500
sample_every = 5
block_size = 20
seed = 12345

println("=" ^ 80)
println("LRC Post-Processing Check")
println("=" ^ 80)
println()
println("Simulation settings:")
println("  T = $T")
println("  ρ = $ρ")
println("  N = $N")
println("  rc = $rc")
println("  Warmup sweeps = $warmup_sweeps")
println("  Production sweeps = $production_sweeps")
println("  Sample every = $sample_every sweeps")
println("  Block size = $block_size")
println("  use_lrc = false (LRC will be applied as post-processing)")
println()

# Run ONE simulation with use_lrc=false
println("Running simulation with use_lrc=false...")
p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp, seed=seed, use_lrc=false)

# Warmup
for _ in 1:warmup_sweeps
    MolSim.MC.sweep!(st, p)
end

# Production: record raw pressure values (p_raw)
println("Production run: recording raw pressure values...")
p_raw_values = Float64[]

for sweep_idx in 1:production_sweeps
    MolSim.MC.sweep!(st, p)
    
    if sweep_idx % sample_every == 0
        # Get raw pressure (without LRC, since use_lrc=false)
        P_raw = MolSim.MC.pressure(st, p, T)
        push!(p_raw_values, P_raw)
    end
end

println("Recorded $(length(p_raw_values)) pressure samples")
println()

# Compute deterministic LRC correction once
# Use the same formula as in the codebase (from LJLongRange.jl)
# Formula: P_tail = (16πρ²/3) * [2/(3*rc^9) - 1/rc^3]
rc3 = rc * rc * rc
rc9 = rc3 * rc3 * rc3
inv_rc3 = 1.0 / rc3
inv_rc9 = 1.0 / rc9
ρ2 = ρ * ρ
p_lrc = (16.0 * π * ρ2 / 3.0) * (2.0 * inv_rc9 / 3.0 - inv_rc3)

println("LRC correction (deterministic, computed once):")
println("  p_lrc = $p_lrc")
println()

# Post-process: add LRC to each raw pressure value
p_corr_values = [p_raw + p_lrc for p_raw in p_raw_values]

# Compute Z values
Z_raw_values = [p_raw / (ρ * T) for p_raw in p_raw_values]
Z_corr_values = [p_corr / (ρ * T) for p_corr in p_corr_values]

# Block average for statistics
Z_raw_ba = MolSim.MC.BlockAverager(block_size)
Z_corr_ba = MolSim.MC.BlockAverager(block_size)

for z_raw in Z_raw_values
    push!(Z_raw_ba, z_raw)
end

for z_corr in Z_corr_values
    push!(Z_corr_ba, z_corr)
end

# Compute statistics
Z_raw_mean = MolSim.MC.mean(Z_raw_ba)
Z_raw_se = MolSim.MC.stderr(Z_raw_ba)
Z_corr_mean = MolSim.MC.mean(Z_corr_ba)
Z_corr_se = MolSim.MC.stderr(Z_corr_ba)

# EOS prediction
P_eos = pressure_kolafa_sklogwiki(T, ρ)
Z_eos = P_eos / (ρ * T)

# Compute deltas
Δ_raw = Z_raw_mean - Z_eos
Δ_corr = Z_corr_mean - Z_eos

println("=" ^ 80)
println("Results")
println("=" ^ 80)
println()

println("Raw compressibility factor (no LRC):")
println("  Z_raw = $Z_raw_mean ± $Z_raw_se")
println()

println("Corrected compressibility factor (raw + LRC post-processed):")
println("  Z_corr = $Z_corr_mean ± $Z_corr_se")
println()

println("EOS (Kolafa SklogWiki):")
println("  Z_eos = $Z_eos")
println()

println("Discrepancies:")
println("  Δ_raw  = Z_raw  - Z_eos = $Z_raw_mean - $Z_eos = $Δ_raw")
println("  Δ_corr = Z_corr - Z_eos = $Z_corr_mean - $Z_eos = $Δ_corr")
println()

println("Verification:")
println("  Expected: Z_corr = Z_raw + (p_lrc / (ρ*T))")
Z_corr_expected = Z_raw_mean + (p_lrc / (ρ * T))
Z_corr_diff = abs(Z_corr_mean - Z_corr_expected)
println("  Z_raw + p_lrc/(ρ*T) = $Z_raw_mean + ($p_lrc / ($ρ * $T)) = $Z_corr_expected")
println("  Z_corr (from data)  = $Z_corr_mean")
println("  |difference|        = $Z_corr_diff")
if Z_corr_diff < 1e-10
    println("  ✓ PASS: Post-processing correctly adds LRC correction")
else
    println("  ✗ FAIL: Post-processing does not match expected relationship")
end
println()

println("=" ^ 80)
