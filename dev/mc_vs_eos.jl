"""
Validation script comparing Monte Carlo simulations with EOS predictions.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

# Load Kolafa SklogWiki implementation
include(joinpath(@__DIR__, "..", "src", "EOS", "LJKolafaSklogwiki1994.jl"))

# State points to test
T_values = [1.0, 1.35, 2.0]
ρ_values = [0.05, 0.1, 0.2, 0.3, 0.5]

# Simulation parameters
N = 864  # Number of particles
rc = 2.5  # Cutoff distance
max_disp = 0.1  # Maximum displacement
use_lrc = true  # Enable long-range corrections
seed_base = 1234  # Base seed (will vary per statepoint)

# Simulation settings
warmup_sweeps = 200
production_sweeps = 1000
sample_every = 5
block_size = 20

println("=" ^ 80)
println("Monte Carlo vs EOS Validation")
println("=" ^ 80)
println()
println("Validation target: Z_noLRC (truncated LJ MC) vs Kolafa PLJ oracle (full LJ EOS)")
println("Z_LRC is shown for diagnostic purposes only.")
println()
println("Note: Kolafa PLJ matches truncated virial pressure; the analytic tail correction")
println("      (LRC) is correct but represents a different pressure model (truncated+LRC")
println("      vs full LJ).")
println()
println("Simulation settings:")
println("  N = $N")
println("  rc = $rc")
println("  LRC = $use_lrc (used for comparison, but validation is vs noLRC)")
println("  Warmup sweeps = $warmup_sweeps")
println("  Production sweeps = $production_sweeps")
println("  Sample every = $sample_every sweeps")
println("  Block size = $block_size")
println()

# Prepare results table
results = []

# Long-run settings for T=1.0, ρ=0.5
long_run_T = 1.0
long_run_ρ = 0.5
long_run_warmup = 2000
long_run_production = 20000
long_run_sample_every = 10
long_run_block_size = 50

# Function to run simulation at a state point
function run_simulation(T, ρ, warmup_sw, prod_sw, samp_every, blk_size, seed_val)
    # Initialize simulation
    p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp, seed=seed_val, use_lrc=use_lrc)
    
    # Track acceptance during warmup
    warmup_acc_total = 0.0
    warmup_attempts = 0
    
    # Warmup
    for _ in 1:warmup_sw
        acc = MolSim.MC.sweep!(st, p)
        warmup_acc_total += acc
        warmup_attempts += 1
    end
    
    # Production with block averaging
    pressure_ba = MolSim.MC.BlockAverager(blk_size)
    prod_acc_total = 0.0
    prod_attempts = 0
    
    for sweep_idx in 1:prod_sw
        acc = MolSim.MC.sweep!(st, p)
        prod_acc_total += acc
        prod_attempts += 1
        
        if sweep_idx % samp_every == 0
            P = MolSim.MC.pressure(st, p, T)
            push!(pressure_ba, P)
        end
    end
    
    # Compute acceptance ratio
    acceptance_ratio = prod_acc_total / prod_attempts
    
    # Compute MC statistics
    p_mean = MolSim.MC.mean(pressure_ba)
    p_se = MolSim.MC.stderr(pressure_ba)
    Z_MC = p_mean / (ρ * T)
    Z_MC_se = p_se / (ρ * T)
    
    return (acceptance_ratio=acceptance_ratio, Z_MC=Z_MC, Z_MC_se=Z_MC_se, p_mean=p_mean, p_se=p_se)
end

# Run simulations for each state point
for (T_idx, T) in enumerate(T_values)
    for (ρ_idx, ρ) in enumerate(ρ_values)
        println("Running (T=$T, ρ=$ρ)...")
        
        seed = seed_base + T_idx * 1000 + ρ_idx * 100
        
        # Check if this is the long-run state point
        is_long_run_point = (T == long_run_T) && (ρ == long_run_ρ)
        
        # Run with LRC = false
        # Temporarily override use_lrc for this simulation
        p_noLRC, st_noLRC = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp, seed=seed, use_lrc=false)
        warmup_acc_total_noLRC = 0.0
        warmup_attempts_noLRC = 0
        for _ in 1:warmup_sweeps
            acc = MolSim.MC.sweep!(st_noLRC, p_noLRC)
            warmup_acc_total_noLRC += acc
            warmup_attempts_noLRC += 1
        end
        pressure_ba_noLRC = MolSim.MC.BlockAverager(block_size)
        prod_acc_total_noLRC = 0.0
        prod_attempts_noLRC = 0
        for sweep_idx in 1:production_sweeps
            acc = MolSim.MC.sweep!(st_noLRC, p_noLRC)
            prod_acc_total_noLRC += acc
            prod_attempts_noLRC += 1
            if sweep_idx % sample_every == 0
                P = MolSim.MC.pressure(st_noLRC, p_noLRC, T)
                push!(pressure_ba_noLRC, P)
            end
        end
        acceptance_noLRC = prod_acc_total_noLRC / prod_attempts_noLRC
        p_mean_noLRC = MolSim.MC.mean(pressure_ba_noLRC)
        p_se_noLRC = MolSim.MC.stderr(pressure_ba_noLRC)
        Z_MC_noLRC = p_mean_noLRC / (ρ * T)
        Z_MC_noLRC_se = p_se_noLRC / (ρ * T)
        
        # Run with LRC = true (default behavior)
        p_LRC, st_LRC = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp, seed=seed + 1, use_lrc=true)
        warmup_acc_total_LRC = 0.0
        warmup_attempts_LRC = 0
        for _ in 1:warmup_sweeps
            acc = MolSim.MC.sweep!(st_LRC, p_LRC)
            warmup_acc_total_LRC += acc
            warmup_attempts_LRC += 1
        end
        pressure_ba_LRC = MolSim.MC.BlockAverager(block_size)
        prod_acc_total_LRC = 0.0
        prod_attempts_LRC = 0
        for sweep_idx in 1:production_sweeps
            acc = MolSim.MC.sweep!(st_LRC, p_LRC)
            prod_acc_total_LRC += acc
            prod_attempts_LRC += 1
            if sweep_idx % sample_every == 0
                P = MolSim.MC.pressure(st_LRC, p_LRC, T)
                push!(pressure_ba_LRC, P)
            end
        end
        acceptance_LRC = prod_acc_total_LRC / prod_attempts_LRC
        p_mean_LRC = MolSim.MC.mean(pressure_ba_LRC)
        p_se_LRC = MolSim.MC.stderr(pressure_ba_LRC)
        Z_MC_LRC = p_mean_LRC / (ρ * T)
        Z_MC_LRC_se = p_se_LRC / (ρ * T)
        
        println("  Acceptance ratio (noLRC): $(round(acceptance_noLRC, digits=4))")
        println("  Acceptance ratio (LRC):   $(round(acceptance_LRC, digits=4))")
        println("  Z_MC (noLRC) = $Z_MC_noLRC ± $Z_MC_noLRC_se")
        println("  Z_MC (LRC)   = $Z_MC_LRC ± $Z_MC_LRC_se")
        
        # Optional long-run for T=1.0, ρ=0.5 (only for LRC=true case)
        mc_long_result = nothing
        if is_long_run_point
            println("  Running long-run mode (LRC=true, warmup=$long_run_warmup, production=$long_run_production)...")
            seed_long = seed + 10000  # Different seed for long run
            # Run long-run with LRC=true
            p_long, st_long = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp, seed=seed_long, use_lrc=true)
            warmup_acc_total_long = 0.0
            warmup_attempts_long = 0
            for _ in 1:long_run_warmup
                acc = MolSim.MC.sweep!(st_long, p_long)
                warmup_acc_total_long += acc
                warmup_attempts_long += 1
            end
            pressure_ba_long = MolSim.MC.BlockAverager(long_run_block_size)
            prod_acc_total_long = 0.0
            prod_attempts_long = 0
            for sweep_idx in 1:long_run_production
                acc = MolSim.MC.sweep!(st_long, p_long)
                prod_acc_total_long += acc
                prod_attempts_long += 1
                if sweep_idx % long_run_sample_every == 0
                    P = MolSim.MC.pressure(st_long, p_long, T)
                    push!(pressure_ba_long, P)
                end
            end
            acceptance_long = prod_acc_total_long / prod_attempts_long
            p_mean_long = MolSim.MC.mean(pressure_ba_long)
            p_se_long = MolSim.MC.stderr(pressure_ba_long)
            Z_MC_long = p_mean_long / (ρ * T)
            Z_MC_long_se = p_se_long / (ρ * T)
            mc_long_result = (acceptance_ratio=acceptance_long, Z_MC=Z_MC_long, Z_MC_se=Z_MC_long_se, p_mean=p_mean_long, p_se=p_se_long)
            println("  Long-run acceptance ratio: $(round(acceptance_long, digits=4))")
            println("  Long-run Z_MC (LRC) = $Z_MC_long ± $Z_MC_long_se")
        end
        
        # Compute EOS prediction (Kolafa SklogWiki PLJ oracle)
        P_eos = pressure_kolafa_sklogwiki(T, ρ)
        Z_eos = P_eos / (ρ * T)
        
        # Compute deltas
        delta_noLRC_eos = Z_MC_noLRC - Z_eos
        delta_LRC_eos = Z_MC_LRC - Z_eos
        diff_LRC_noLRC = Z_MC_LRC - Z_MC_noLRC
        
        # Store results
        push!(results, (
            T = T,
            ρ = ρ,
            acceptance = acceptance_LRC,  # Use LRC acceptance for display (or average? use LRC as it's the default)
            Z_MC_noLRC = Z_MC_noLRC,
            Z_MC_noLRC_se = Z_MC_noLRC_se,
            Z_MC_LRC = Z_MC_LRC,
            Z_MC_LRC_se = Z_MC_LRC_se,
            Z_MC_long = mc_long_result !== nothing ? mc_long_result.Z_MC : nothing,
            Z_MC_long_se = mc_long_result !== nothing ? mc_long_result.Z_MC_se : nothing,
            Z_eos = Z_eos,
            delta_noLRC_eos = delta_noLRC_eos,
            delta_LRC_eos = delta_LRC_eos,
            diff_LRC_noLRC = diff_LRC_noLRC
        ))
        
        println("  Z_eos = $Z_eos")
        println()
    end
end

# Print results table
println("=" ^ 80)
println("Results Table")
println("=" ^ 80)
println()
println(lpad("T", 6), " | ", lpad("ρ", 6), " | ", 
        lpad("Accept", 7), " | ",
        lpad("Z_noLRC ± err", 18), " | ",
        lpad("Z_LRC ± err", 18), " | ",
        lpad("Z_eos", 10), " | ",
        lpad("Δ(noLRC-EOS)", 14), " | ",
        lpad("Δ(LRC-EOS)", 12), " | ",
        lpad("(Z_LRC-Z_noLRC)", 16))
println("-" ^ 120)

for r in results
    Z_noLRC_str = string(round(r.Z_MC_noLRC, digits=4), " ± ", round(r.Z_MC_noLRC_se, digits=4))
    Z_LRC_str = string(round(r.Z_MC_LRC, digits=4), " ± ", round(r.Z_MC_LRC_se, digits=4))
    
    println(lpad(r.T, 6), " | ", lpad(r.ρ, 6), " | ",
            lpad(round(r.acceptance, digits=4), 7), " | ",
            lpad(Z_noLRC_str, 18), " | ",
            lpad(Z_LRC_str, 18), " | ",
            lpad(round(r.Z_eos, digits=4), 10), " | ",
            lpad(round(r.delta_noLRC_eos, digits=4), 14), " | ",
            lpad(round(r.delta_LRC_eos, digits=4), 12), " | ",
            lpad(round(r.diff_LRC_noLRC, digits=4), 16))
end

println("=" ^ 120)
println()
println("Validation: Z_noLRC (truncated LJ MC) is compared against Z_eos (Kolafa PLJ oracle).")
println("            Kolafa PLJ matches truncated virial pressure; the analytic tail correction")
println("            (LRC) is correct but represents a different pressure model.")
println("            Z_LRC is shown as a diagnostic column only.")
println()
println("Note: Long-run mode (warmup=2000, production=20000) applied to T=$(long_run_T), ρ=$(long_run_ρ)")
println("Note: Z_eos uses Kolafa-Nezbeda 1994 (SklogWiki PLJ oracle)")
println()

# Cutoff radius analysis: verify that increasing rc reduces MC–EOS discrepancy
println("=" ^ 80)
println("Cutoff Radius Analysis (LRC enabled)")
println("=" ^ 80)
println()
println("State points: (T,ρ) = (2.0,0.2), (1.35,0.2), (1.0,0.1)")
println("rc values: 2.5, 3.0, 3.5")
println("Expected: |Δ(MC–EOS)| decreases as rc increases")
println()

# Reduced state points for rc analysis
rc_analysis_points = [
    (T=2.0, ρ=0.2),
    (T=1.35, ρ=0.2),
    (T=1.0, ρ=0.1),
]

rc_values = [2.5, 3.0, 3.5]

# Function to run simulation with specified rc
function run_simulation_rc(T, ρ, rc_val, warmup_sw, prod_sw, samp_every, blk_size, seed_val)
    # Initialize simulation with specified rc and LRC enabled
    p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc_val, max_disp=max_disp, seed=seed_val, use_lrc=true)
    
    # Warmup
    for _ in 1:warmup_sw
        MolSim.MC.sweep!(st, p)
    end
    
    # Production with block averaging
    pressure_ba = MolSim.MC.BlockAverager(blk_size)
    prod_acc_total = 0.0
    prod_attempts = 0
    
    for sweep_idx in 1:prod_sw
        acc = MolSim.MC.sweep!(st, p)
        prod_acc_total += acc
        prod_attempts += 1
        
        if sweep_idx % samp_every == 0
            P = MolSim.MC.pressure(st, p, T)
            push!(pressure_ba, P)
        end
    end
    
    # Compute acceptance ratio
    acceptance_ratio = prod_acc_total / prod_attempts
    
    # Compute MC statistics
    p_mean = MolSim.MC.mean(pressure_ba)
    p_se = MolSim.MC.stderr(pressure_ba)
    Z_MC = p_mean / (ρ * T)
    Z_MC_se = p_se / (ρ * T)
    
    return (acceptance_ratio=acceptance_ratio, Z_MC=Z_MC, Z_MC_se=Z_MC_se)
end

# Run rc analysis for each state point
for (point_idx, pt) in enumerate(rc_analysis_points)
    T_rc = pt.T
    ρ_rc = pt.ρ
    
    println("-" ^ 80)
    println("State point: T=$T_rc, ρ=$ρ_rc")
    println("-" ^ 80)
    println()
    
    # Compute EOS prediction once per state point
    P_eos_rc = pressure_kolafa_sklogwiki(T_rc, ρ_rc)
    Z_eos_rc = P_eos_rc / (ρ_rc * T_rc)
    
    println(lpad("rc", 6), " | ", lpad("Z_MC ± err", 18), " | ", 
            lpad("Z_eos", 10), " | ", lpad("Δ(MC–EOS)", 12), " | ", lpad("Accept", 7))
    println("-" ^ 80)
    
    rc_results = []
    
    for rc_val in rc_values
        seed_rc = seed_base + 50000 + point_idx * 1000 + Int(rc_val * 10)
        
        result_rc = run_simulation_rc(T_rc, ρ_rc, rc_val, warmup_sweeps, production_sweeps, 
                                      sample_every, block_size, seed_rc)
        
        delta_MC_eos_rc = result_rc.Z_MC - Z_eos_rc
        
        Z_MC_str_rc = string(round(result_rc.Z_MC, digits=4), " ± ", round(result_rc.Z_MC_se, digits=4))
        
        println(lpad(rc_val, 6), " | ", lpad(Z_MC_str_rc, 18), " | ",
                lpad(round(Z_eos_rc, digits=4), 10), " | ",
                lpad(round(delta_MC_eos_rc, digits=4), 12), " | ",
                lpad(round(result_rc.acceptance_ratio, digits=4), 7))
        
        push!(rc_results, (rc=rc_val, delta=abs(delta_MC_eos_rc)))
    end
    
    # Verify that |Δ| decreases as rc increases
    println()
    if length(rc_results) >= 2
        deltas = [r.delta for r in rc_results]
        is_decreasing = true
        for i in 2:length(deltas)
            if deltas[i] >= deltas[i-1]
                is_decreasing = false
                break
            end
        end
        
        if is_decreasing
            println("✓ |Δ(MC–EOS)| decreases as rc increases")
        else
            println("⚠ |Δ(MC–EOS)| does NOT consistently decrease with rc")
            println("  Deltas: $(round.(deltas, digits=4))")
        end
    end
    
    println()
end

println("=" ^ 80)