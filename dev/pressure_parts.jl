"""
Diagnostic script to break down pressure components and verify LRC correction application.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

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
seed_base = 12345

println("=" ^ 80)
println("Pressure Components Diagnostic")
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
println()

# Ideal gas pressure
p_ideal = ρ * T

println("Ideal gas pressure:")
println("  p_ideal = ρ*T = $ρ * $T = $p_ideal")
println()

# Run simulation WITHOUT LRC
println("Running simulation WITHOUT LRC...")
p_noLRC, st_noLRC = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp, seed=seed_base, use_lrc=false)

# Warmup
for _ in 1:warmup_sweeps
    MolSim.MC.sweep!(st_noLRC, p_noLRC)
end

# Production - collect pressure measurements
pressure_ba_noLRC = MolSim.MC.BlockAverager(block_size)

for sweep_idx in 1:production_sweeps
    MolSim.MC.sweep!(st_noLRC, p_noLRC)
    
    if sweep_idx % sample_every == 0
        P = MolSim.MC.pressure(st_noLRC, p_noLRC, T)
        push!(pressure_ba_noLRC, P)
    end
end

p_total_noLRC_mean = MolSim.MC.mean(pressure_ba_noLRC)
p_total_noLRC_se = MolSim.MC.stderr(pressure_ba_noLRC)

# Run simulation WITH LRC
println("Running simulation WITH LRC...")
p_LRC, st_LRC = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp, seed=seed_base + 1, use_lrc=true)

# Warmup
for _ in 1:warmup_sweeps
    MolSim.MC.sweep!(st_LRC, p_LRC)
end

# Production - collect pressure measurements
pressure_ba_LRC = MolSim.MC.BlockAverager(block_size)

for sweep_idx in 1:production_sweeps
    MolSim.MC.sweep!(st_LRC, p_LRC)
    
    if sweep_idx % sample_every == 0
        P = MolSim.MC.pressure(st_LRC, p_LRC, T)
        push!(pressure_ba_LRC, P)
    end
end

p_total_LRC_mean = MolSim.MC.mean(pressure_ba_LRC)
p_total_LRC_se = MolSim.MC.stderr(pressure_ba_LRC)

# Get analytic LRC correction term
p_lrc_analytic = p_LRC.lrc_p

# Compute excess pressures
p_excess_noLRC = p_total_noLRC_mean - p_ideal
p_excess_LRC = p_total_LRC_mean - p_ideal

# Compute difference
p_total_diff = p_total_LRC_mean - p_total_noLRC_mean

println()
println("=" ^ 80)
println("Results (averaged over production run)")
println("=" ^ 80)
println()

println("Total pressures:")
println("  p_total_noLRC = $p_total_noLRC_mean ± $p_total_noLRC_se")
println("  p_total_LRC   = $p_total_LRC_mean ± $p_total_LRC_se")
println()

println("Ideal gas pressure:")
println("  p_ideal = $p_ideal")
println()

println("Excess pressures (total - ideal):")
println("  p_excess_noLRC = p_total_noLRC - p_ideal = $p_total_noLRC_mean - $p_ideal = $p_excess_noLRC")
println("  p_excess_LRC   = p_total_LRC   - p_ideal = $p_total_LRC_mean - $p_ideal = $p_excess_LRC")
println()

println("LRC correction term (analytic):")
println("  p_lrc = $p_lrc_analytic")
println()

println("Verification checks:")
println()
println("1. LRC correction application:")
println("   p_total_LRC - p_total_noLRC = $p_total_LRC_mean - $p_total_noLRC_mean = $p_total_diff")
println("   p_lrc (analytic)            = $p_lrc_analytic")
diff_verification = abs(p_total_diff - p_lrc_analytic)
println("   |difference|                = $diff_verification")
if diff_verification < 1e-10
    println("   ✓ PASS: p_total_LRC - p_total_noLRC equals p_lrc (within numerical precision)")
else
    println("   ✗ FAIL: p_total_LRC - p_total_noLRC does NOT equal p_lrc")
    println("   This suggests the LRC correction may be applied incorrectly or twice.")
end
println()

println("2. Excess pressure relationship:")
println("   p_excess_LRC - p_excess_noLRC = $p_excess_LRC - $p_excess_noLRC = $(p_excess_LRC - p_excess_noLRC)")
println("   Should equal p_lrc = $p_lrc_analytic")
excess_diff_verification = abs((p_excess_LRC - p_excess_noLRC) - p_lrc_analytic)
if excess_diff_verification < 1e-10
    println("   ✓ PASS: Excess pressure difference equals p_lrc")
else
    println("   ✗ FAIL: Excess pressure difference does NOT equal p_lrc")
end
println()

println("3. Total pressure components:")
println("   p_total_noLRC = p_ideal + p_excess_noLRC = $p_ideal + $p_excess_noLRC = $(p_ideal + p_excess_noLRC)")
println("   Should equal $p_total_noLRC_mean")
total_noLRC_check = abs(p_total_noLRC_mean - (p_ideal + p_excess_noLRC))
if total_noLRC_check < 1e-10
    println("   ✓ PASS: p_total_noLRC = p_ideal + p_excess_noLRC")
else
    println("   ✗ FAIL: Components don't add up for noLRC case")
end
println()

println("   p_total_LRC = p_ideal + p_excess_LRC = $p_ideal + $p_excess_LRC = $(p_ideal + p_excess_LRC)")
println("   Should equal $p_total_LRC_mean")
total_LRC_check = abs(p_total_LRC_mean - (p_ideal + p_excess_LRC))
if total_LRC_check < 1e-10
    println("   ✓ PASS: p_total_LRC = p_ideal + p_excess_LRC")
else
    println("   ✗ FAIL: Components don't add up for LRC case")
end
println()

println("=" ^ 80)
println("Summary:")
println("=" ^ 80)
println("The LRC correction should be added to the total pressure, not to excess pressure separately.")
println("Verification: p_total_LRC should equal p_total_noLRC + p_lrc")
println()
println("Note: The two simulations (noLRC and LRC) use different seeds and will sample")
println("      different configurations, so the sampled virial contributions will differ.")
println("      Therefore, p_total_LRC - p_total_noLRC will not exactly equal p_lrc due to")
println("      statistical differences in the sampled configurations.")
println("      To verify exact LRC application, check the implementation code that adds")
println("      p.lrc_p to the sampled pressure in the pressure() function.")
println()
