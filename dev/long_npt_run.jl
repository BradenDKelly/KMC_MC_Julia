"""
Long NPT simulation for validation workflow.
Reads NVT summary to obtain P_target, runs NPT, and measures ρ, P, U.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

# Parameters (same as NVT except where noted)
N = 864
T = 2.0
rc = 2.5
max_disp = 0.1
max_dlnV = 0.01  # from existing NPT scripts
vol_move_every = 10  # from existing NPT scripts
use_lrc = false
seed_nvt = 12345
seed = seed_nvt + 1  # different seed from NVT

warmup_sweeps = 50_000
prod_sweeps = 200_000
sample_every = 50
block_size_samples = 200

# Helper function for robust CSV parsing
function parse_csv_summary(filepath::String, required_fields::Vector{String})
    if !isfile(filepath)
        error("File not found: $filepath")
    end
    
    # Read file as text
    lines = String[]
    open(filepath, "r") do io
        for line in eachline(io)
            stripped = strip(line)
            if !isempty(stripped)
                push!(lines, stripped)
            end
        end
    end
    
    if length(lines) < 2
        error("File $filepath has fewer than 2 non-empty lines (need header + data)")
    end
    
    header_line = lines[1]
    data_line = lines[2]
    
    # Split on comma
    header_fields = split(header_line, ',')
    data_fields = split(data_line, ',')
    
    if length(header_fields) != length(data_fields)
        error("Header and data line have different number of fields in $filepath")
    end
    
    # Build Dict mapping column name => index
    col_dict = Dict{String, Int}()
    for (idx, col_name) in enumerate(header_fields)
        col_dict[strip(col_name)] = idx
    end
    
    # Check for required fields
    missing_fields = String[]
    for field in required_fields
        if !haskey(col_dict, field)
            push!(missing_fields, field)
        end
    end
    
    if !isempty(missing_fields)
        available = sort(collect(keys(col_dict)))
        error("Required fields missing in $filepath:\n  Missing: $(join(missing_fields, ", "))\n  Available: $(join(available, ", "))")
    end
    
    # Parse fields
    function get_value(field_name::String, default_type::Type{T}) where T
        idx = col_dict[field_name]
        val_str = strip(data_fields[idx])
        if T == Bool
            if val_str == "true"
                return true
            elseif val_str == "false"
                return false
            else
                error("Cannot parse '$val_str' as Bool for field '$field_name'")
            end
        else
            return parse(T, val_str)
        end
    end
    
    return (get_value, col_dict)
end

# Read NVT summary to get P_target
results_dir = joinpath(@__DIR__, "results")
nvt_summary_file = joinpath(@__DIR__, "results", "nvt_summary.csv")

if !isfile(nvt_summary_file)
    error("NVT summary file not found: $nvt_summary_file. Run dev/long_nvt_run.jl first.")
end

# Parse NVT summary (check for P_thermo_mean first)
# Read all lines first to avoid double-reading
all_lines = String[]
open(nvt_summary_file, "r") do io
    for line in eachline(io)
        stripped = strip(line)
        if !isempty(stripped)
            push!(all_lines, stripped)
        end
    end
end

if length(all_lines) < 2
    error("NVT summary file has fewer than 2 lines")
end

header_line = all_lines[1]
data_line = all_lines[2]
header_fields = split(header_line, ',')
data_fields = split(data_line, ',')

# Build column dictionary
col_dict = Dict{String, Int}()
for (idx, col_name) in enumerate(header_fields)
    col_dict[strip(col_name)] = idx
end

# Parse required fields (reuse the helper for consistency)
required_fields = ["N", "T", "rho", "rc", "max_disp", "use_lrc", "seed"]
get_val, _ = parse_csv_summary(nvt_summary_file, required_fields)

# Check if P_thermo_mean exists and get value
if haskey(col_dict, "P_thermo_mean")
    P_target_idx = col_dict["P_thermo_mean"]
    P_target = parse(Float64, strip(data_fields[P_target_idx]))
    P_target_source = "P_thermo_mean (FEP thermodynamic)"
elseif haskey(col_dict, "P_mean")
    P_target_idx = col_dict["P_mean"]
    P_target = parse(Float64, strip(data_fields[P_target_idx]))
    P_target_source = "P_mean (virial)"
else
    available = sort(collect(keys(col_dict)))
    error("Neither P_thermo_mean nor P_mean found in NVT summary. Available fields: $(join(available, ", "))")
end
# Get other parameters (already parsed above)
nvt_N = get_val("N", Int)
nvt_T = get_val("T", Float64)
nvt_rho = get_val("rho", Float64)
nvt_rc = get_val("rc", Float64)
nvt_max_disp = get_val("max_disp", Float64)
nvt_use_lrc = get_val("use_lrc", Bool)
nvt_seed = get_val("seed", Int)

rho_init = nvt_rho  # use same initial density as NVT

println("=" ^ 80)
println("Long NPT Simulation")
println("=" ^ 80)
println()
println("Parameters:")
println("  N = $N (from NVT: $nvt_N)")
println("  T = $T (from NVT: $nvt_T)")
println("  P_target = $P_target (from NVT: $P_target_source)")
println("  rho_init = $rho_init (initial density, from NVT)")
println("  rc = $rc (from NVT: $nvt_rc)")
println("  max_disp = $max_disp (from NVT: $nvt_max_disp)")
println("  max_dlnV = $max_dlnV")
println("  vol_move_every = $vol_move_every")
println("  use_lrc = $use_lrc (from NVT: $nvt_use_lrc)")
println("  seed = $seed (NVT seed was: $nvt_seed)")
println("  warmup_sweeps = $warmup_sweeps")
println("  prod_sweeps = $prod_sweeps")
println("  sample_every = $sample_every")
println("  block_size_samples = $block_size_samples")
println()

# Initialize with same initial density as NVT
println("Initializing simulation...")
p, st = MolSim.MC.init_fcc(N=N, ρ=rho_init, T=T, rc=rc, max_disp=max_disp, seed=seed, use_lrc=use_lrc)
T_actual = 1.0 / p.β
println("  Actual T = $T_actual")
println()

# Warmup with auto-tuning of max_dlnV
println("Warmup: $warmup_sweeps sweeps (with volume move auto-tuning)...")
let
    warmup_volume_accepted = 0
    warmup_volume_attempted = 0
    current_max_dlnV = max_dlnV
    total_warmup_volume_accepted = 0
    total_warmup_volume_attempted = 0
    
    for i in 1:warmup_sweeps
        MolSim.MC.sweep!(st, p)
        if i % vol_move_every == 0
            vol_acc = MolSim.MC.volume_trial!(st, p; max_dlnV=current_max_dlnV, Pext=P_target)
            if vol_acc
                warmup_volume_accepted += 1
                total_warmup_volume_accepted += 1
            end
            warmup_volume_attempted += 1
            total_warmup_volume_attempted += 1
            
            # Auto-tune max_dlnV every 2000 sweeps during warmup
            if i % 2000 == 0 && warmup_volume_attempted > 0
                acc_vol_warmup = Float64(warmup_volume_accepted) / Float64(warmup_volume_attempted)
                
                if acc_vol_warmup > 0.55
                    current_max_dlnV *= 1.25
                    println("  Sweep $i: acc_vol = $(round(acc_vol_warmup, digits=3)) > 0.55, increasing max_dlnV to $(round(current_max_dlnV, digits=5))")
                elseif acc_vol_warmup < 0.15
                    current_max_dlnV *= 0.80
                    println("  Sweep $i: acc_vol = $(round(acc_vol_warmup, digits=3)) < 0.15, decreasing max_dlnV to $(round(current_max_dlnV, digits=5))")
                end
                
                # Reset counters for next tuning window
                warmup_volume_accepted = 0
                warmup_volume_attempted = 0
            end
        end
        if i % 10000 == 0
            println("  Completed $i / $warmup_sweeps sweeps")
        end
    end
    
    # Final warmup volume acceptance (overall)
    final_warmup_acc_vol = total_warmup_volume_attempted > 0 ? Float64(total_warmup_volume_accepted) / Float64(total_warmup_volume_attempted) : 0.0
    global final_max_dlnV = current_max_dlnV
    global final_warmup_acc_vol = final_warmup_acc_vol
end

println("  Warmup complete")
println("  Final max_dlnV = $(round(final_max_dlnV, digits=5))")
println("  Final warmup acc_vol = $(round(final_warmup_acc_vol, digits=3))")
println()

# Production setup
println("Production: $prod_sweeps sweeps...")
density_ba = MolSim.MC.BlockAverager(block_size_samples)
pressure_ba = MolSim.MC.BlockAverager(block_size_samples)
energy_ba = MolSim.MC.BlockAverager(block_size_samples)

# Timeseries data
timeseries_data = []

# Acceptance tracking
let
    particle_accepted = 0
    particle_attempted = 0
    volume_accepted = 0
    volume_attempted = 0
    
    # Timing
    production_start = time_ns()
    
    # Production loop
    for sweep_idx in 1:prod_sweeps
        # NVT sweep
        acc = MolSim.MC.sweep!(st, p)
        particle_accepted += Int(round(acc * N))
        particle_attempted += N
        
        # Volume move (use tuned max_dlnV from warmup)
        if sweep_idx % vol_move_every == 0
            vol_acc = MolSim.MC.volume_trial!(st, p; max_dlnV=final_max_dlnV, Pext=P_target)
            if vol_acc
                volume_accepted += 1
            end
            volume_attempted += 1
        end
        
        # Sample observables
        if sweep_idx % sample_every == 0
            V = st.L * st.L * st.L
            rho_inst = N / V
            P = MolSim.MC.pressure(st, p, T_actual)
            E_total = MolSim.MC.total_energy(st, p)
            U = E_total / N  # energy per particle
            
            push!(density_ba, rho_inst)
            push!(pressure_ba, P)
            push!(energy_ba, U)
            
            # Record timeseries
            acc_move_so_far = Float64(particle_accepted) / Float64(particle_attempted)
            acc_vol_so_far = volume_attempted > 0 ? Float64(volume_accepted) / Float64(volume_attempted) : 0.0
            
            push!(timeseries_data, (
                sweep_index = sweep_idx,
                rho = rho_inst,
                P = P,
                U = U,
                acc_move_so_far = acc_move_so_far,
                acc_vol_so_far = acc_vol_so_far
            ))
        end
        
        if sweep_idx % 10000 == 0
            println("  Completed $sweep_idx / $prod_sweeps sweeps")
        end
    end
    
    production_end = time_ns()
    global runtime_ns = production_end - production_start
    global runtime_sec = runtime_ns / 1e9
    
    # Statistics
    global rho_mean = MolSim.MC.mean(density_ba)
    global rho_stderr = MolSim.MC.stderr(density_ba)
    global P_mean = MolSim.MC.mean(pressure_ba)
    global P_stderr = MolSim.MC.stderr(pressure_ba)
    global U_mean = MolSim.MC.mean(energy_ba)
    global U_stderr = MolSim.MC.stderr(energy_ba)
    
    global acc_move = Float64(particle_accepted) / Float64(particle_attempted)
    global acc_vol = volume_attempted > 0 ? Float64(volume_accepted) / Float64(volume_attempted) : 0.0
    
    # Total moves: particle moves + volume moves
    total_particle_moves = particle_attempted
    total_volume_moves = volume_attempted
    total_moves = total_particle_moves + total_volume_moves
    global moves_per_sec = total_moves / runtime_sec
    global sweeps_per_sec = prod_sweeps / runtime_sec
end

# Print summary
println()
println("=" ^ 80)
println("Results Summary")
println("=" ^ 80)
println()
println("Observables:")
println("  ρ_mean = $rho_mean ± $rho_stderr")
println("  P_mean = $P_mean ± $P_stderr")
println("  U_mean = $U_mean ± $U_stderr")
println()
println("Acceptance:")
println("  acc_move = $acc_move")
println("  acc_vol = $acc_vol")
println()
println("Performance:")
println("  Runtime = $(round(runtime_sec, digits=2)) s")
println("  Moves/sec = $(round(moves_per_sec, digits=0))")
println("  Sweeps/sec = $(round(sweeps_per_sec, digits=2))")
println()

# Write summary CSV
summary_file = joinpath(@__DIR__, "results", "npt_summary.csv")
open(summary_file, "w") do io
    # Header
    println(io, "N,T,P_target,rc,max_disp,use_lrc,seed,warmup_sweeps,prod_sweeps,sample_every," *
                "rho_mean,rho_stderr," *
                "P_mean,P_stderr,U_mean,U_stderr," *
                "acc_move,acc_vol," *
                "moves_per_sec,sweeps_per_sec,runtime_sec")
    
    # Data row
    println(io, "$N,$T,$P_target,$rc,$max_disp,$use_lrc,$seed,$warmup_sweeps,$prod_sweeps,$sample_every," *
                "$rho_mean,$rho_stderr," *
                "$P_mean,$P_stderr,$U_mean,$U_stderr," *
                "$acc_move,$acc_vol," *
                "$moves_per_sec,$sweeps_per_sec,$runtime_sec")
end
println("Wrote summary to: $summary_file")

# Write timeseries CSV
timeseries_file = joinpath(@__DIR__, "results", "npt_timeseries.csv")
open(timeseries_file, "w") do io
    # Header
    println(io, "sweep_index,rho,P,U,acc_move_so_far,acc_vol_so_far")
    
    # Data rows
    for row in timeseries_data
        println(io, "$(row.sweep_index),$(row.rho),$(row.P),$(row.U),$(row.acc_move_so_far),$(row.acc_vol_so_far)")
    end
end
println("Wrote timeseries to: $timeseries_file")
println()

println("=" ^ 80)
