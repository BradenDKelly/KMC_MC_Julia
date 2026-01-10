"""
Debug script to inspect NPT volume move acceptance formula.
Logs detailed information about volume move attempts to verify correctness.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

# Parameters (matching NVT run)
N = 864
T = 2.0
rc = 2.5
max_disp = 0.1
max_dlnV = 0.04768  # Use final tuned value from recent run, or initial 0.01
vol_move_every = 10
use_lrc = false
seed = 12346

# Read P_target from NVT summary
results_dir = joinpath(@__DIR__, "results")
nvt_summary_file = joinpath(@__DIR__, "results", "nvt_summary.csv")

if !isfile(nvt_summary_file)
    error("NVT summary file not found: $nvt_summary_file. Run dev/long_nvt_run.jl first.")
end

# Helper function for CSV parsing
function parse_csv_summary(filepath::String, required_fields::Vector{String})
    if !isfile(filepath)
        error("File not found: $filepath")
    end
    
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
        error("File $filepath has fewer than 2 non-empty lines")
    end
    
    header_line = lines[1]
    data_line = lines[2]
    
    header_fields = split(header_line, ',')
    data_fields = split(data_line, ',')
    
    if length(header_fields) != length(data_fields)
        error("Header and data line have different number of fields")
    end
    
    col_dict = Dict{String, Int}()
    for (idx, col_name) in enumerate(header_fields)
        col_dict[strip(col_name)] = idx
    end
    
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
    
    function get_value(field_name::String, default_type::Type{T}) where T
        idx = col_dict[field_name]
        val_str = strip(data_fields[idx])
        if T == Bool
            return val_str == "true" || val_str == "True"
        else
            return parse(T, val_str)
        end
    end
    
    return (get_value, col_dict)
end

get_val, _ = parse_csv_summary(nvt_summary_file, ["P_mean", "rho"])
P_target = get_val("P_mean", Float64)
rho_init = get_val("rho", Float64)

println("=" ^ 80)
println("NPT Volume Move Debug")
println("=" ^ 80)
println()
println("Parameters:")
println("  N = $N")
println("  T = $T")
println("  P_target = $P_target")
println("  rho_init = $rho_init")
println("  rc = $rc")
println("  max_dlnV = $max_dlnV")
println("  vol_move_every = $vol_move_every")
println()
println("Volume move acceptance formula (from code inspection):")
println("  log_acc = -β * (ΔU + Pext * ΔV) + N * (lnV_new - lnV_old)")
println("  accept_prob = exp(log_acc) if log_acc < 0, else 1.0")
println("  accept if: log_acc >= 0 OR rand() < exp(log_acc)")
println()
println("Breaking down the terms:")
println("  -β * (ΔU + Pext * ΔV): energy + pressure work term (negative for favorable moves)")
println("  +N * (lnV_new - lnV_old): Jacobian term (dimensionless, already includes β implicitly if needed)")
println()
println("Questions to verify:")
println("  1. Is Jacobian sign correct? Should be +N*ln(V'/V) or -N*ln(V'/V)?")
println("  2. Is β multiplied by Jacobian? (Should NOT be; Jacobian is dimensionless)")
println("  3. Are Pext units consistent with U and V? (All in reduced LJ units)")
println()

# Initialize
p, st = MolSim.MC.init_fcc(N=N, ρ=rho_init, T=T, rc=rc, max_disp=max_disp, seed=seed, use_lrc=use_lrc)
T_actual = 1.0 / p.β
β = p.β

# Short warmup
println("Short warmup: 100 sweeps...")
for i in 1:100
    MolSim.MC.sweep!(st, p)
    if i % vol_move_every == 0
        MolSim.MC.volume_trial!(st, p; max_dlnV=max_dlnV, Pext=P_target)
    end
end
println("  Warmup complete")
println()

# Debug logging: instrument volume_trial! by recreating the logic
# We'll manually call the steps and log everything
println("Logging first 5000 volume move attempts...")
println()

# Ensure results directory exists
mkpath(results_dir)
debug_file = joinpath(results_dir, "npt_vol_debug.csv")

# Open CSV for writing
open(debug_file, "w") do io
    # Header
    println(io, "attempt_id,accepted,V_old,V_new,ln_V_ratio,DeltaU,Pext_DeltaV,jac_term_N_lnV,exponent_total,accept_prob")
    
    attempt_id = 0
    volume_attempts = 0
    
    while volume_attempts < 5000
        # Do some sweeps
        for _ in 1:vol_move_every
            MolSim.MC.sweep!(st, p)
        end
        
        # Now perform a volume move with full logging
        attempt_id += 1
        volume_attempts += 1
        
        # Capture state before move
        N_debug = st.N
        L_old_debug = st.L
        V_old_debug = L_old_debug * L_old_debug * L_old_debug
        lnV_old_debug = log(V_old_debug)
        
        # Store old positions
        pos_old_debug = copy(st.pos)
        
        # Compute old energy
        U_old_debug = MolSim.MC.total_energy(st, p)
        
        # Propose new volume (replicate the exact logic from volume_trial!)
        dlnV_debug = (rand(st.rng) - 0.5) * 2.0 * max_dlnV
        lnV_new_debug = lnV_old_debug + dlnV_debug
        V_new_debug = exp(lnV_new_debug)
        L_new_debug = cbrt(V_new_debug)
        scale_debug = L_new_debug / L_old_debug
        
        # Scale positions
        @inbounds for i in 1:N_debug
            st.pos[1, i] = st.pos[1, i] * scale_debug
            st.pos[2, i] = st.pos[2, i] * scale_debug
            st.pos[3, i] = st.pos[3, i] * scale_debug
        end
        
        # Wrap positions
        scratch_debug = st.scratch_dr
        @inbounds for i in 1:N_debug
            scratch_debug[1] = st.pos[1, i]
            scratch_debug[2] = st.pos[2, i]
            scratch_debug[3] = st.pos[3, i]
            MolSim.MC.wrap!(scratch_debug, L_new_debug)
            st.pos[1, i] = scratch_debug[1]
            st.pos[2, i] = scratch_debug[2]
            st.pos[3, i] = scratch_debug[3]
        end
        
        # Update box
        st.L = L_new_debug
        st.cl = MolSim.MC.CellList(N_debug, L_new_debug, st.cl.rc)
        MolSim.MC.rebuild_cells!(st)
        
        # Compute new energy
        U_new_debug = MolSim.MC.total_energy(st, p)
        
        # Compute acceptance terms (EXACTLY as in volume_trial!)
        ΔU_debug = U_new_debug - U_old_debug
        ΔV_debug = V_new_debug - V_old_debug
        lnV_ratio = lnV_new_debug - lnV_old_debug
        
        # Terms in the acceptance formula
        Pext_DeltaV = P_target * ΔV_debug
        jac_term = N_debug * lnV_ratio
        
        # Exact formula from code: log_acc = -β * (ΔU + Pext * ΔV) + N * (lnV_new - lnV_old)
        exponent_total = -β * (ΔU_debug + Pext_DeltaV) + jac_term
        accept_prob = exponent_total >= 0.0 ? 1.0 : exp(exponent_total)
        
        # Metropolis decision
        accepted_debug = false
        rand_val = rand(st.rng)
        if exponent_total >= 0.0 || rand_val < accept_prob
            accepted_debug = true
        else
            # Reject: restore
            copyto!(st.pos, pos_old_debug)
            st.L = L_old_debug
            st.cl = MolSim.MC.CellList(N_debug, L_old_debug, st.cl.rc)
            MolSim.MC.rebuild_cells!(st)
        end
        
        # Write to CSV
        accepted_int = accepted_debug ? 1 : 0
        println(io, "$attempt_id,$accepted_int,$V_old_debug,$V_new_debug,$lnV_ratio,$ΔU_debug,$Pext_DeltaV,$jac_term,$exponent_total,$accept_prob")
        
        if volume_attempts % 500 == 0
            println("  Logged $volume_attempts / 5000 volume attempts...")
        end
    end
end

println("  Completed logging 5000 volume attempts")
println("Wrote debug log to: $debug_file")
println()

# Print summary statistics
println("=" ^ 80)
println("Summary")
println("=" ^ 80)
println()
println("Formula verified from code (line 398 of src/MC/LJMC.jl):")
println("  log_acc = -β * (ΔU + Pext * ΔV) + N * (lnV_new - lnV_old)")
println()
println("Breakdown:")
println("  Term 1: -β * (ΔU + Pext * ΔV)  (energy + pressure work)")
println("  Term 2: +N * (lnV_new - lnV_old)  (Jacobian term)")
println()
println("Questions:")
println("  1. Jacobian sign: +N*ln(V'/V) (should be positive when V' > V)")
println("  2. β factor: β multiplies (ΔU + Pext*ΔV) only, NOT the Jacobian")
println("  3. Units: All in reduced LJ units (σ=ε=kB=1)")
println()
println("Expected NPT acceptance formula (standard):")
println("  acc = min(1, exp(-β(ΔU + Pext*ΔV) + (N+1)*ln(V'/V)))")
println("  OR sometimes: acc = min(1, exp(-β(ΔU + Pext*ΔV) + N*ln(V'/V)))")
println()
println("Current implementation uses: +N*ln(V'/V) (not N+1)")
println()

println("=" ^ 80)
