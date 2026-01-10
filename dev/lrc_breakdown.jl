"""
Instrumentation script to break down pressure into raw (sampled) and LRC components.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

# Load Kolafa SklogWiki implementation
include(joinpath(@__DIR__, "..", "src", "EOS", "LJKolafaSklogwiki1994.jl"))

# Simulation parameters
N = 864
rc = 2.5
max_disp = 0.1
T = 2.0
warmup_sweeps = 100
production_sweeps = 500
sample_every = 5
block_size = 20
seed = 12345

println("=" ^ 80)
println("LRC Breakdown Instrumentation")
println("=" ^ 80)
println()
println("Simulation settings:")
println("  N = $N")
println("  rc = $rc")
println("  T = $T")
println("  Warmup sweeps = $warmup_sweeps")
println("  Production sweeps = $production_sweeps")
println("  Sample every = $sample_every sweeps")
println("  Block size = $block_size")
println()

# State points
state_points = [
    (ρ=0.2, label="ρ=0.2 (moderate density)"),
    (ρ=0.05, label="ρ=0.05 (low density, LRC helps)"),
]

results = []

for (point_idx, state_pt) in enumerate(state_points)
    ρ = state_pt.ρ
    
    println("-" ^ 80)
    println("State point $point_idx: T=$T, $(state_pt.label)")
    println("-" ^ 80)
    
    # Initialize with LRC enabled to get the lrc_p value
    p_lrc, st_lrc = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp, seed=seed + point_idx, use_lrc=true)
    
    # Warmup
    for _ in 1:warmup_sweeps
        MolSim.MC.sweep!(st_lrc, p_lrc)
    end
    
    # Production run - collect raw pressure measurements (without LRC added)
    # We'll compute raw pressure manually: P = ρ*T + W/(3V)
    pressure_ba_raw = MolSim.MC.BlockAverager(block_size)
    pressure_ba_total = MolSim.MC.BlockAverager(block_size)
    
    for sweep_idx in 1:production_sweeps
        MolSim.MC.sweep!(st_lrc, p_lrc)
        
        if sweep_idx % sample_every == 0
            # Compute raw pressure (sampled virial, no LRC)
            N_particles = st_lrc.N
            L = st_lrc.L
            V = L * L * L
            ρ_actual = N_particles / V
            W = MolSim.MC.total_virial(st_lrc, p_lrc)
            P_raw = ρ_actual * T + W / (3.0 * V)
            
            # Total pressure (with LRC added)
            P_total = MolSim.MC.pressure(st_lrc, p_lrc, T)
            
            push!(pressure_ba_raw, P_raw)
            push!(pressure_ba_total, P_total)
        end
    end
    
    # Compute statistics
    p_raw_mean = MolSim.MC.mean(pressure_ba_raw)
    p_raw_se = MolSim.MC.stderr(pressure_ba_raw)
    p_total_mean = MolSim.MC.mean(pressure_ba_total)
    p_total_se = MolSim.MC.stderr(pressure_ba_total)
    
    # LRC term (analytic, stored in params)
    p_lrc_analytic = p_lrc.lrc_p
    
    # Compute Z values
    Z_raw = p_raw_mean / (ρ * T)
    Z_lrc = p_lrc_analytic / (ρ * T)
    Z_total = p_total_mean / (ρ * T)
    
    # EOS prediction
    P_eos = pressure_kolafa_sklogwiki(T, ρ)
    Z_eos = P_eos / (ρ * T)
    
    # Print results
    println()
    println("Raw pressure (sampled virial, no LRC):")
    println("  p_raw = $p_raw_mean ± $p_raw_se")
    println("  Z_raw = $Z_raw")
    println()
    println("LRC correction (analytic):")
    println("  p_lrc = $p_lrc_analytic")
    println("  Z_lrc = $Z_lrc")
    println()
    println("Total pressure (raw + LRC):")
    println("  p_total = $p_total_mean ± $p_total_se")
    println("  Z_total = $Z_total")
    println()
    println("Verification: p_total ≈ p_raw + p_lrc?")
    p_total_expected = p_raw_mean + p_lrc_analytic
    diff_check = abs(p_total_mean - p_total_expected)
    println("  p_raw + p_lrc = $p_total_expected")
    println("  |p_total - (p_raw + p_lrc)| = $diff_check")
    if diff_check < 1e-10
        println("  ✓ Match (within numerical precision)")
    else
        println("  ⚠ Mismatch (may indicate sampling or implementation issue)")
    end
    println()
    println("EOS (Kolafa SklogWiki):")
    println("  Z_eos = $Z_eos")
    println()
    println("Discrepancies:")
    println("  Δ(Z_raw - Z_eos) = $(Z_raw - Z_eos)")
    println("  Δ(Z_total - Z_eos) = $(Z_total - Z_eos)")
    println()
    
    # Store for density scaling check
    push!(results, (ρ=ρ, p_lrc=p_lrc_analytic))
end

# Assert LRC pressure scaling ~ ρ^2
println("=" ^ 80)
println("LRC Pressure Scaling Check")
println("=" ^ 80)
println()

if length(results) >= 2
    ρ_high = results[1].ρ
    ρ_low = results[2].ρ
    p_lrc_high = results[1].p_lrc
    p_lrc_low = results[2].p_lrc
    
    # Expected ratio if scaling is ~ ρ^2
    expected_ratio = (ρ_high / ρ_low)^2
    actual_ratio = p_lrc_high / p_lrc_low
    
    println("Density ratio: ρ_high / ρ_low = $ρ_high / $ρ_low = $(ρ_high / ρ_low)")
    println("Expected p_lrc ratio (if ~ ρ^2): ($ρ_high / $ρ_low)^2 = $expected_ratio")
    println("Actual p_lrc ratio: $p_lrc_high / $p_lrc_low = $actual_ratio")
    println()
    
    # Check if ratio matches expected (within 5% tolerance for loose check)
    ratio_diff = abs(actual_ratio - expected_ratio) / expected_ratio
    tolerance = 0.05  # 5% tolerance
    
    if ratio_diff < tolerance
        println("✓ LRC pressure scales ~ ρ^2 (ratio difference: $(round(ratio_diff * 100, digits=2))% < $(tolerance * 100)%)")
    else
        println("✗ LRC pressure does NOT scale ~ ρ^2 (ratio difference: $(round(ratio_diff * 100, digits=2))% >= $(tolerance * 100)%)")
        println("  This suggests the LRC formula is wrong or using incorrect density.")
    end
    
    # Assertion (in script, not test)
    @assert ratio_diff < tolerance "LRC pressure scaling check failed: actual_ratio=$actual_ratio, expected_ratio=$expected_ratio, difference=$(round(ratio_diff * 100, digits=2))%"
    
    println()
    println("Assertion passed: LRC pressure scaling is approximately ρ^2")
else
    println("⚠ Cannot check scaling: need at least 2 state points")
end

println("=" ^ 80)
