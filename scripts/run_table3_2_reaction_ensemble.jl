"""
Table 3.2 Reaction Ensemble Monte Carlo benchmark runner.

Reproduces Table 3.2 from Kelly Braden's thesis using NPT Reaction Ensemble MC.

Usage:
    julia --project scripts/run_table3_2_reaction_ensemble.jl [options]

Options:
    --seed <int>              RNG seed (default: 12345)
    --equil <int>             Equilibration sweeps (default: 5000)
    --prod <int>              Production sweeps (default: 10000)
    --stride <int>            Sampling stride (default: 10)
    --blocks <int>            Block size for averaging (default: 50)
    --p_reaction <float>      Reaction move probability (default: 0.1)
    --rho_init <float>        Initial density guess (default: 0.75)
    --outdir <path>           Output directory (default: results)
    --logfile <path>          Combined log file (default: outdir/table3_2_run.log)
    --reaction <name>         Run single reaction (e.g., A⇌B)
    --all                     Run all reactions (default)
    --smoke                   Smoke test mode (A⇌B, 50+100 sweeps)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using Random
using StaticArrays
using JSON
using Dates

# ============================================================================
# Table 3.1: Species definitions
# ============================================================================
const SPECIES_DATA = Dict(
    "A" => (σ=1.0, ε=1.0, q=0.002),
    "B" => (σ=1.0, ε=1.0, q=0.002),
    "C" => (σ=1.1, ε=0.9, q=0.002),
    "D" => (σ=1.0, ε=1.0, q=0.02),
    "E" => (σ=1.1, ε=0.9, q=0.02),
    "F" => (σ=1.0, ε=1.0, q=0.02),
)

# ============================================================================
# Table 3.2: Reaction definitions and target values
# ============================================================================
const REACTIONS = Dict(
    "A⇌B" => Dict(
        "stoichiometry" => Dict("A" => -1, "B" => 1),
        "target" => Dict("N_A" => 200.0, "N_B" => 200.0, "ρ_total" => 0.766),
    ),
    "A⇌C" => Dict(
        "stoichiometry" => Dict("A" => -1, "C" => 1),
        "target" => Dict("N_A" => 297.7, "N_C" => 102.3, "ρ_total" => 0.718),
    ),
    "A⇌2D" => Dict(
        "stoichiometry" => Dict("A" => -1, "D" => 2),
        "target" => Dict("N_A" => 372.2, "N_D" => 55.7, "ρ_total" => 0.766),
    ),
    "A⇌2E" => Dict(
        "stoichiometry" => Dict("A" => -1, "E" => 2),
        "target" => Dict("N_A" => 390.5, "N_E" => 19.0, "ρ_total" => 0.757),
    ),
    "A⇌D+F" => Dict(
        "stoichiometry" => Dict("A" => -1, "D" => 1, "F" => 1),
        "target" => Dict("N_A" => 345.1, "N_D" => 54.9, "N_F" => 54.9, "ρ_total" => 0.766),
    ),
    "A⇌D+E" => Dict(
        "stoichiometry" => Dict("A" => -1, "D" => 1, "E" => 1),
        "target" => Dict("N_A" => 367.4, "N_D" => 32.6, "N_E" => 32.6, "ρ_total" => 0.751),
    ),
)

# ============================================================================
# Statepoint constants
# ============================================================================
const T_TARGET = 2.0
const P_TARGET = 5.0
const RC = 2.5
const N_A_INIT = 400
const RHO_INIT_GUESS = 0.75

# ============================================================================
# Simple logging utility
# ============================================================================
mutable struct SimpleLogger
    logfile::Union{IO, Nothing}
end

function log_line(logger::SimpleLogger, s::AbstractString)
    println(s)
    if logger.logfile !== nothing
        println(logger.logfile, s)
        flush(logger.logfile)
    end
end

# Add timestamp to all log outputs
function log_print(logger::SimpleLogger, msg::String)
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
    log_line(logger, "[$timestamp] $msg")
end

# Backward compatibility (now just an alias)
log_print_with_timestamp(logger::SimpleLogger, msg::String) = log_print(logger, msg)

# ============================================================================
# Helper functions
# ============================================================================

function create_species_vectors(reaction_name::String)
    """Create σ_types, ϵ_types, and logq vectors for a reaction."""
    rxn_dict = REACTIONS[reaction_name]["stoichiometry"]
    species_names = sort(collect(keys(rxn_dict)))  # Deterministic order
    n_species = length(species_names)
    
    σ_types = Vector{Float64}(undef, n_species)
    ϵ_types = Vector{Float64}(undef, n_species)
    logq = Vector{Float64}(undef, n_species)
    
    for (idx, name) in enumerate(species_names)
        data = SPECIES_DATA[name]
        σ_types[idx] = data.σ
        ϵ_types[idx] = data.ε
        logq[idx] = log(data.q)  # log(q/λ³)
    end
    
    return (species_names, σ_types, ϵ_types, logq)
end

function create_reaction_object(reaction_name::String, species_names::Vector{String}, logq::Vector{Float64})
    """Create Reaction object from reaction name and species order."""
    rxn_dict = REACTIONS[reaction_name]["stoichiometry"]
    n_species = length(species_names)
    stoichiometry = zeros(Int, n_species)
    
    for (idx, name) in enumerate(species_names)
        if haskey(rxn_dict, name)
            stoichiometry[idx] = rxn_dict[name]
        end
    end
    
    # logK = 0.0 for Table 3.2 (equilibrium determined by logq only)
    return MolSim.MC.Reaction(reaction_name, stoichiometry, 0.0; logq=logq)
end

function create_initial_state(species_names::Vector{String}, N_A_init::Int, ρ_init::Float64,
                               T::Float64, rc::Float64, max_disp::Float64, seed::Int)
    """Create initial state with N_A_init particles of type A (first species)."""
    # Find index of A
    A_idx = findfirst(x -> x == "A", species_names)
    if A_idx === nothing
        throw(ArgumentError("Species A not found in species_names"))
    end
    
    # Create types vector: all A initially
    types = fill(A_idx, N_A_init)
    
    # Compute box length from density
    V = N_A_init / ρ_init
    L = cbrt(V)
    
    # Create random positions in box [0, L)
    rng = Xoshiro(seed)
    pos = zeros(Float64, 3, N_A_init)
    for i in 1:N_A_init
        pos[1, i] = rand(rng) * L
        pos[2, i] = rand(rng) * L
        pos[3, i] = rand(rng) * L
    end
    
    # Create cell list
    cl = MolSim.MC.CellList(N_A_init, L, rc)
    
    # Create scratch vector
    scratch_dr = MVector{3,Float64}(0.0, 0.0, 0.0)
    
    # Create state
    st = MolSim.MC.LJState(N_A_init, L, pos, types, rng, cl, scratch_dr, 0, 0)
    
    # Rebuild cell list
    MolSim.MC.rebuild_cells!(st)
    
    return st
end

function print_statepoint_echo(logger::SimpleLogger, reaction_name::String, species_names::Vector{String},
                                σ_types::Vector{Float64}, ϵ_types::Vector{Float64},
                                logq::Vector{Float64}, stoichiometry::Vector{Int},
                                T::Float64, P::Float64, rc::Float64,
                                N_init::Dict{String,Int}, ρ_init::Float64,
                                seed::Int, max_disp::Float64, max_dlnV::Float64,
                                vol_move_every::Int, p_reaction::Float64,
                                sweeps_equil::Int, sweeps_prod::Int,
                                sample_every::Int, block_size::Int)
    """Print structured statepoint echo."""
    log_print(logger, "=" ^ 80)
    log_print(logger, "STATEPOINT ECHO")
    log_print(logger, "=" ^ 80)
    log_print(logger, "Reaction: $reaction_name")
    stoich_parts = String[]
    for (idx, name) in enumerate(species_names)
        nu = stoichiometry[idx]
        if nu < 0
            push!(stoich_parts, "$(-nu)$name")
        elseif nu > 0
            push!(stoich_parts, "+$nu$name")
        end
    end
    log_print(logger, "Stoichiometry: " * join(stoich_parts, " "))
    log_print(logger, "Ensemble: NPT Reaction Ensemble MC (REMC)")
    log_print(logger, "Seed: $seed")
    log_print(logger, "T = $T")
    log_print(logger, "P = $P")
    log_print(logger, "rc = $rc")
    log_print(logger, "LJ model: shifted (cut-and-shifted)")
    log_print(logger, "LRC: false")
    log_print(logger, "Mixing rule: Lorentz-Berthelot (σ_ab = (σ_a + σ_b)/2, ε_ab = sqrt(ε_a * ε_b))")
    log_print(logger, "")
    log_print(logger, "Species parameters:")
    for (idx, name) in enumerate(species_names)
        log_print(logger, "  $name: σ=$(round(σ_types[idx], digits=4)), ε=$(round(ϵ_types[idx], digits=4)), q=$(round(exp(logq[idx]), digits=6)), logq=$(round(logq[idx], digits=6))")
    end
    log_print(logger, "")
    log_print(logger, "Initial composition:")
    for name in species_names
        n_init = get(N_init, name, 0)
        log_print(logger, "  N_$name = $n_init")
    end
    log_print(logger, "Initial density guess: $(round(ρ_init, digits=4))")
    log_print(logger, "")
    log_print(logger, "Move parameters:")
    log_print(logger, "  Translation: max_disp = $(round(max_disp, digits=4))")
    log_print(logger, "  Volume: max_dlnV = $(round(max_dlnV, digits=6)), every $vol_move_every sweeps")
    log_print(logger, "  Reaction: p_reaction = $(round(p_reaction, digits=4)) per sweep")
    log_print(logger, "")
    log_print(logger, "Run lengths:")
    log_print(logger, "  Equilibration: $sweeps_equil sweeps")
    log_print(logger, "  Production: $sweeps_prod sweeps")
    log_print(logger, "  Sample every: $sample_every")
    log_print(logger, "  Block size: $block_size")
    log_print(logger, "=" ^ 80)
    log_print(logger, "")
end

function run_reaction_benchmark(reaction_name::String, logger::SimpleLogger;
                                seed::Int=12345,
                                sweeps_equil::Int=5000,
                                sweeps_prod::Int=10000,
                                output_dir::String="results",
                                max_disp::Float64=0.1,
                                max_dlnV::Float64=0.01,
                                vol_move_every::Int=10,
                                p_reaction::Float64=0.1,
                                sample_every::Int=10,
                                block_size::Int=50,
                                rho_init::Float64=0.75,
                                proposal_mode::Symbol=:com_insert,
                                com_kernel_Δ::Float64=0.05)
    """Run a single reaction benchmark. Returns results dict and timing dict."""
    
    timing = Dict{String, Float64}()
    
    # Create species vectors
    t_start = time()
    species_names, σ_types, ϵ_types, logq = create_species_vectors(reaction_name)
    n_species = length(species_names)
    
    # Create stoichiometry vector
    rxn_dict = REACTIONS[reaction_name]["stoichiometry"]
    stoichiometry = zeros(Int, n_species)
    for (idx, name) in enumerate(species_names)
        if haskey(rxn_dict, name)
            stoichiometry[idx] = rxn_dict[name]
        end
    end
    
    # Initial composition (always start with N_A = 400, others = 0)
    N_init = Dict{String,Int}()
    for name in species_names
        N_init[name] = (name == "A") ? N_A_INIT : 0
    end
    
    # Create multicomponent parameters (shifted LJ)
    p = MolSim.MC.LJParams(σ_types=σ_types, ϵ_types=ϵ_types,
                           rc=RC, T=T_TARGET, max_disp=max_disp,
                           use_lrc=false, lj_model=:shifted)
    
    # Create reaction
    rxn = create_reaction_object(reaction_name, species_names, logq)
    
    # Print statepoint echo
    print_statepoint_echo(logger, reaction_name, species_names, σ_types, ϵ_types, logq,
                          stoichiometry, T_TARGET, P_TARGET, RC,
                          N_init, rho_init, seed, max_disp, max_dlnV,
                          vol_move_every, p_reaction, sweeps_equil, sweeps_prod,
                          sample_every, block_size)
    
    # Create initial state (always start with exactly 400 A particles, zero products)
    t_init_start = time()
    st = create_initial_state(species_names, N_A_INIT, rho_init,
                               T_TARGET, RC, max_disp, seed)
    timing["initialization"] = time() - t_init_start
    
    # Equilibration
    log_print(logger, "Equilibration: $sweeps_equil sweeps...")
    t_equil_start = time()
    for sweep in 1:sweeps_equil
        do_vol = (sweep % vol_move_every == 0)
        MolSim.MC.sweep_npt_with_reactions!(st, p, rxn; Pext=P_TARGET, max_dlnV=max_dlnV,
                                            p_reaction=p_reaction, do_volume_move=do_vol,
                                            rebuild_every=st.N,
                                            proposal_mode=proposal_mode, com_kernel_Δ=com_kernel_Δ)
    end
    timing["equilibration"] = time() - t_equil_start
    log_print(logger, "Equilibration completed in $(round(timing["equilibration"], digits=2)) seconds")
    
    # Production: accumulate data
    log_print(logger, "Production: $sweeps_prod sweeps...") 
    N_species_samples = Vector{Vector{Int}}()
    V_samples = Float64[]
    
    translation_accepted_total = 0
    translation_attempted_total = 0
    volume_accepted_total = 0
    volume_attempted_total = 0
    reaction_accepted_total = 0
    reaction_attempted_total = 0
    reaction_forward_attempted_total = 0
    reaction_forward_accepted_total = 0
    reaction_reverse_attempted_total = 0
    reaction_reverse_accepted_total = 0
    
    t_prod_start = time()
    for sweep in 1:sweeps_prod
        do_vol = (sweep % vol_move_every == 0)
        trans_acc, vol_acc, vol_att, rxn_acc, rxn_att, rxn_fwd_att, rxn_fwd_acc, rxn_rev_att, rxn_rev_acc = 
            MolSim.MC.sweep_npt_with_reactions!(st, p, rxn; Pext=P_TARGET, max_dlnV=max_dlnV,
                                                p_reaction=p_reaction, do_volume_move=do_vol,
                                                rebuild_every=st.N)
        
        translation_attempted_total += st.N
        translation_accepted_total += Int(round(trans_acc * st.N))
        volume_accepted_total += vol_acc
        volume_attempted_total += vol_att
        reaction_accepted_total += rxn_acc
        reaction_attempted_total += rxn_att
        reaction_forward_attempted_total += rxn_fwd_att
        reaction_forward_accepted_total += rxn_fwd_acc
        reaction_reverse_attempted_total += rxn_rev_att
        reaction_reverse_accepted_total += rxn_rev_acc
        
        if sweep % sample_every == 0
            counts = MolSim.MC.count_species(st, n_species)
            push!(N_species_samples, counts)
            V = st.L * st.L * st.L
            push!(V_samples, V)
        end
    end
    timing["production"] = time() - t_prod_start
    timing["total"] = time() - t_start
    log_print(logger, "Production completed in $(round(timing["production"], digits=2)) seconds") 
    
    # Compute block-averaged means and uncertainties
    n_samples = length(N_species_samples)
    n_blocks = n_samples ÷ block_size
    if n_blocks < 1
        n_blocks = 1
    end
    
    # Block averages for each species
    N_species_means = zeros(Float64, n_species)
    N_species_stderr = zeros(Float64, n_species)
    
    for species_idx in 1:n_species
        samples = [N_species_samples[i][species_idx] for i in 1:n_samples]
        
        ba = MolSim.MC.BlockAverager(block_size)
        for s in samples
            push!(ba, Float64(s))
        end
        
        N_species_means[species_idx] = MolSim.MC.mean(ba)
        N_species_stderr[species_idx] = MolSim.MC.stderr(ba)
    end
    
    # Block average for volume
    V_ba = MolSim.MC.BlockAverager(block_size)
    for V_val in V_samples
        push!(V_ba, V_val)
    end
    V_mean = MolSim.MC.mean(V_ba)
    V_stderr = MolSim.MC.stderr(V_ba)
    
    # Compute total density
    N_total_mean = sum(N_species_means)
    ρ_total_mean = N_total_mean / V_mean
    # Error propagation for density (simplified)
    ρ_total_stderr = ρ_total_mean * sqrt((V_stderr/V_mean)^2)  # Simplified
    
    # Compute acceptance rates
    trans_acc_rate = translation_attempted_total > 0 ? Float64(translation_accepted_total) / Float64(translation_attempted_total) : 0.0
    vol_acc_rate = volume_attempted_total > 0 ? Float64(volume_accepted_total) / Float64(volume_attempted_total) : 0.0
    rxn_acc_rate = reaction_attempted_total > 0 ? Float64(reaction_accepted_total) / Float64(reaction_attempted_total) : 0.0
    
    # Print results
    log_print(logger, "")
    log_print(logger, "Results:")
    log_print(logger, "-" ^ 80)
    log_print(logger, "Species | Mean N | StdErr N")
    log_print(logger, "-" ^ 80)
    for (idx, name) in enumerate(species_names)
        name_padded = rpad(name, 7)
        log_print(logger, "$name_padded | $(lpad(string(round(N_species_means[idx], digits=2)), 8)) | $(lpad(string(round(N_species_stderr[idx], digits=2)), 8))")
    end
    log_print(logger, "$(rpad("V", 7)) | $(lpad(string(round(V_mean, digits=4)), 8)) | $(lpad(string(round(V_stderr, digits=4)), 8))")
    log_print(logger, "$(rpad("ρ_total", 7)) | $(lpad(string(round(ρ_total_mean, digits=4)), 8)) | $(lpad(string(round(ρ_total_stderr, digits=4)), 8))")
    log_print(logger, "-" ^ 80)
    log_print(logger, "Acceptance rates:")
    log_print(logger, "  Translation: $(round(trans_acc_rate, digits=4))")
    log_print(logger, "  Volume: $(round(vol_acc_rate, digits=4))")
    log_print(logger, "  Reaction: $(round(rxn_acc_rate, digits=4))")
    
    # Forward/reverse reaction diagnostics (for ΔN=0 reactions)
    if reaction_forward_attempted_total > 0 || reaction_reverse_attempted_total > 0
        rxn_fwd_acc_rate = reaction_forward_attempted_total > 0 ? reaction_forward_accepted_total / reaction_forward_attempted_total : 0.0
        rxn_rev_acc_rate = reaction_reverse_attempted_total > 0 ? reaction_reverse_accepted_total / reaction_reverse_attempted_total : 0.0
        log_print(logger, "  Reaction Forward (A→B): attempts=$(reaction_forward_attempted_total), accepted=$(reaction_forward_accepted_total), rate=$(round(rxn_fwd_acc_rate, digits=4))")
        log_print(logger, "  Reaction Reverse (B→A): attempts=$(reaction_reverse_attempted_total), accepted=$(reaction_reverse_accepted_total), rate=$(round(rxn_rev_acc_rate, digits=4))")
    end
    log_print(logger, "Timing (seconds):")
    log_print(logger, "  Initialization: $(round(timing["initialization"], digits=2))")
    log_print(logger, "  Equilibration: $(round(timing["equilibration"], digits=2))")
    log_print(logger, "  Production: $(round(timing["production"], digits=2))")
    log_print(logger, "  Total: $(round(timing["total"], digits=2))") 
    
    # Compare with targets
    targets = REACTIONS[reaction_name]["target"]
    if !haskey(targets, "ρ_total")
        error("Missing ρ_total target for reaction $reaction_name in REACTIONS dict")
    end
    
    log_print(logger, "")
    log_print(logger, "=" ^ 80)
    log_print(logger, "FINAL SUMMARY")
    log_print(logger, "=" ^ 80)
    log_print(logger, "Species counts:")
    for (idx, name) in enumerate(species_names)
        log_print(logger, "  N_$name = $(round(N_species_means[idx], digits=2)) ± $(round(N_species_stderr[idx], digits=2))")
    end
    log_print(logger, "Total density: ρ_total = $(round(ρ_total_mean, digits=4)) ± $(round(ρ_total_stderr, digits=4))")
    log_print(logger, "Acceptance rates: translation=$(round(trans_acc_rate, digits=3)), volume=$(round(vol_acc_rate, digits=3)), reaction=$(round(rxn_acc_rate, digits=3))")
    log_print(logger, "Wall time: init=$(round(timing["initialization"], digits=2))s, equil=$(round(timing["equilibration"], digits=2))s, prod=$(round(timing["production"], digits=2))s, total=$(round(timing["total"], digits=2))s")
    log_print(logger, "")
    log_print(logger, "=" ^ 80)
    log_print(logger, "THESIS COMPARISON (Table 3.2 ReKMC row)")
    log_print(logger, "=" ^ 80)
    log_print(logger, "Quantity      | Measured      | Target        | % Error")
    log_print(logger, "-" ^ 80)
    
    for (idx, name) in enumerate(species_names)
        target_key = "N_$name"
        if haskey(targets, target_key)
            target = targets[target_key]
            measured = N_species_means[idx]
            pct_error = 100.0 * abs(measured - target) / target
            name_padded = rpad("N_$name", 13)
            log_print(logger, "$name_padded | $(lpad(string(round(measured, digits=2)), 13)) | $(lpad(string(round(target, digits=1)), 13)) | $(lpad(string(round(pct_error, digits=2)) * "%", 7))")
        else
            error("Missing target N_$name for reaction $reaction_name in REACTIONS dict")
        end
    end
    
    target_ρ = targets["ρ_total"]
    pct_error = 100.0 * abs(ρ_total_mean - target_ρ) / target_ρ
    log_print(logger, "$(rpad("ρ_total", 13)) | $(lpad(string(round(ρ_total_mean, digits=4)), 13)) | $(lpad(string(round(target_ρ, digits=3)), 13)) | $(lpad(string(round(pct_error, digits=2)) * "%", 7))")
    log_print(logger, "-" ^ 80)
    
    # Prepare JSON output
    json_data = Dict(
        "meta" => Dict(
            "reaction" => reaction_name,
            "seed" => seed,
            "timestamp" => string(Dates.now()),
        ),
        "statepoint" => Dict(
            "T" => T_TARGET,
            "P" => P_TARGET,
            "rc" => RC,
            "lj_model" => "shifted",
            "use_lrc" => false,
            "mixing_rule" => "Lorentz-Berthelot",
        ),
        "species" => Dict(),
        "reaction" => Dict(
            "stoichiometry" => Dict(species_names[i] => stoichiometry[i] for i in 1:n_species),
        ),
        "moves" => Dict(
            "max_disp" => max_disp,
            "max_dlnV" => max_dlnV,
            "vol_move_every" => vol_move_every,
            "p_reaction" => p_reaction,
        ),
        "run_lengths" => Dict(
            "sweeps_equil" => sweeps_equil,
            "sweeps_prod" => sweeps_prod,
            "sample_every" => sample_every,
            "block_size" => block_size,
        ),
        "results" => Dict(
            "means" => Dict(),
            "stderrs" => Dict(),
            "acceptance_rates" => Dict(
                "translation" => trans_acc_rate,
                "volume" => vol_acc_rate,
                "reaction" => rxn_acc_rate,
            ),
            "targets" => targets,
            "timing" => timing,
        ),
    )
    
    # Add species data
    for (idx, name) in enumerate(species_names)
        json_data["species"][name] = Dict(
            "sigma" => σ_types[idx],
            "epsilon" => ϵ_types[idx],
            "q" => exp(logq[idx]),
            "logq" => logq[idx],
        )
        json_data["results"]["means"]["N_$name"] = N_species_means[idx]
        json_data["results"]["stderrs"]["N_$name"] = N_species_stderr[idx]
    end
    
    json_data["results"]["means"]["V"] = V_mean
    json_data["results"]["stderrs"]["V"] = V_stderr
    json_data["results"]["means"]["rho_total"] = ρ_total_mean
    json_data["results"]["stderrs"]["rho_total"] = ρ_total_stderr
    
    # Replace NaN with null for JSON compatibility
    function replace_nan!(d)
        if d isa Dict
            for (k, v) in d
                if v isa Float64 && isnan(v)
                    d[k] = nothing
                elseif v isa Dict
                    replace_nan!(v)
                end
            end
        end
        return d
    end
    replace_nan!(json_data)
    
    # Write per-reaction JSON
    mkpath(output_dir)
    json_filename = joinpath(output_dir, "table3_2_$(reaction_name)_seed$(seed).json")
    open(json_filename, "w") do f
        JSON.print(f, json_data, 2)
    end
    
    log_print(logger, "")
    log_print(logger, "JSON output written to: $json_filename")
    log_print(logger, "")
    
    # Return summary data for combined output
    summary_data = Dict(
        "reaction" => reaction_name,
        "means" => Dict("N_$name" => N_species_means[idx] for (idx, name) in enumerate(species_names)),
        "stderrs" => Dict("N_$name" => N_species_stderr[idx] for (idx, name) in enumerate(species_names)),
        "mean_rho_total" => ρ_total_mean,
        "stderr_rho_total" => ρ_total_stderr,
        "acceptance_rates" => Dict(
            "translation" => trans_acc_rate,
            "volume" => vol_acc_rate,
            "reaction" => rxn_acc_rate,
        ),
        "timing" => timing,
        "targets" => targets,
    )
    
    return (json_data, summary_data)
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    # Parse CLI args (simple)
    local seed = 12345
    local sweeps_equil = 5000
    local sweeps_prod = 10000
    local output_dir = "results"
    local sample_every = 10
    local block_size = 50
    local p_reaction = 0.1
    local rho_init = 0.75
    local proposal_mode = :com_insert  # Default to COM insertion
    local logfile_path = nothing
    local smoke_mode = false
    local run_single_reaction = nothing  # None = run all, otherwise string with reaction name
    
    local i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--seed" && i < length(ARGS)
            seed = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--equil" && i < length(ARGS)
            sweeps_equil = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--prod" && i < length(ARGS)
            sweeps_prod = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--stride" && i < length(ARGS)
            sample_every = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--blocks" && i < length(ARGS)
            block_size = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--outdir" && i < length(ARGS)
            output_dir = ARGS[i+1]
            i += 2
        elseif arg == "--logfile" && i < length(ARGS)
            logfile_path = ARGS[i+1]
            i += 2
        elseif arg == "--reaction" && i < length(ARGS)
            run_single_reaction = ARGS[i+1]
            i += 2
        elseif arg == "--smoke"
            smoke_mode = true
            i += 1
        else
            i += 1
        end
    end
    
    # Check environment variables (override CLI)
    if haskey(ENV, "SEED")
        seed = parse(Int, ENV["SEED"])
    end
    
    # Set up logging
    if logfile_path === nothing
        logfile_path = joinpath(output_dir, "table3_2_run.log")
    end
    mkpath(output_dir)
    logfile_io = open(logfile_path, "w")
    logger = SimpleLogger(logfile_io)
    
    # Also create per-reaction log files (they'll be opened/closed per reaction)
    per_reaction_loggers = Dict{String, SimpleLogger}()
    
    t_total_start = time()
    
    log_print(logger, "=" ^ 80)
    log_print(logger, "Table 3.2 Reaction Ensemble Monte Carlo Benchmarks")
    log_print(logger, "=" ^ 80)
    log_print(logger, "Seed: $seed")
    log_print(logger, "Equilibration sweeps: $sweeps_equil")
    log_print(logger, "Production sweeps: $sweeps_prod")
    log_print(logger, "Sample every: $sample_every")
    log_print(logger, "Block size: $block_size")
    log_print(logger, "p_reaction: $p_reaction")
    log_print(logger, "Initial density guess: $rho_init")
    log_print(logger, "Output directory: $output_dir")
    log_print(logger, "Log file: $logfile_path")
    if run_single_reaction !== nothing
        log_print(logger, "Running single reaction: $run_single_reaction")
    else
        log_print(logger, "Running all reactions")
    end
    log_print(logger, "")
    
    # Smoke mode: run only A⇌B for 100 sweeps
    if smoke_mode
        log_line(logger, "SMOKE MODE: Running A⇌B only (100 sweeps)")
        rxn_name = "A⇌B"
        per_rxn_log_path = joinpath(output_dir, "$(rxn_name)_seed$(seed).log")
        mkpath(output_dir)
        per_rxn_log_io = open(per_rxn_log_path, "w")
        per_rxn_logger = SimpleLogger(per_rxn_log_io)
        
        result, summary = run_reaction_benchmark(rxn_name, per_rxn_logger;
                                                seed=seed, sweeps_equil=50,
                                                sweeps_prod=100, output_dir=output_dir,
                                                sample_every=10, block_size=10,
                                                p_reaction=p_reaction, rho_init=rho_init)
        close(per_rxn_log_io)
        n_a = summary["means"]["N_A"]
        n_b = summary["means"]["N_B"]
        rho = summary["mean_rho_total"]
        log_line(logger, "Smoke test completed: A⇌B, N_A=$(round(n_a, digits=1)), N_B=$(round(n_b, digits=1)), rho=$(round(rho, digits=3))")
        close(logfile_io)
        exit(0)
    end
    
    # Run reactions (all or single)
    all_summaries = Dict{String, Dict}()
    all_timings = Dict{String, Float64}()
    
    reactions_to_run = if run_single_reaction !== nothing
        if haskey(REACTIONS, run_single_reaction)
            [run_single_reaction]
        else
            error("Unknown reaction: $run_single_reaction. Available reactions: $(join(keys(REACTIONS), ", "))")
        end
    else
        collect(keys(REACTIONS))
    end
    
    for rxn_name in reactions_to_run
        try
            # Open per-reaction log file
            per_rxn_log_path = joinpath(output_dir, "$(rxn_name)_seed$(seed).log")
            per_rxn_log_io = open(per_rxn_log_path, "w")
            per_rxn_logger = SimpleLogger(per_rxn_log_io)
            
            # Also write to combined log
            log_print(logger, "")
            log_print(logger, "=" ^ 80)
            log_print(logger, "Running: $rxn_name")
            log_print(logger, "=" ^ 80)
            
            result, summary = run_reaction_benchmark(rxn_name, per_rxn_logger;
                                                    seed=seed, sweeps_equil=sweeps_equil,
                                                    sweeps_prod=sweeps_prod, output_dir=output_dir,
                                                    sample_every=sample_every, block_size=block_size,
                                                    p_reaction=p_reaction, rho_init=rho_init,
                                                    proposal_mode=proposal_mode)
            all_summaries[rxn_name] = summary
            
            # Copy output to combined log (already written to per-reaction log)
            # Just add a summary line
            log_print(logger, "$rxn_name completed in $(round(summary["timing"]["total"], digits=2)) seconds")
            
            close(per_rxn_log_io)
        catch e
            log_print(logger, "Error running $rxn_name: $e") 
            rethrow(e)
        end
    end
    
    total_time = time() - t_total_start
    
    log_print(logger, "")
    log_print(logger, "=" ^ 80)
    log_print(logger, "All benchmarks completed")
    log_print(logger, "Total runtime: $(round(total_time, digits=2)) seconds ($(round(total_time / 60.0, digits=2)) minutes)") 
    log_print(logger, "=" ^ 80)
    
    # Write combined summary
    summary_file_txt = joinpath(output_dir, "summary_seed$(seed).txt")
    summary_file_json = joinpath(output_dir, "summary_seed$(seed).json")
    
    open(summary_file_txt, "w") do f
        println(f, "=" ^ 80)
        println(f, "Table 3.2 Reaction Ensemble MC - Combined Summary")
        println(f, "=" ^ 80)
        println(f, "Seed: $seed")
        println(f, "Equilibration: $sweeps_equil sweeps, Production: $sweeps_prod sweeps")
        println(f, "Total runtime: $total_time seconds ($(total_time / 60.0) minutes)")
        println(f, "")
        println(f, "Reaction | N_A (mean±stderr) | N_product (mean±stderr) | ρ_total (mean±stderr) | Trans acc | Vol acc | Rxn acc | Time (s)")
        println(f, "-" ^ 80)
        
        for rxn_name in keys(REACTIONS)
            if haskey(all_summaries, rxn_name)
                s = all_summaries[rxn_name]
                # Extract means for each species (simplified - would need species_names)
                # For now, just print a simplified version
                timing_total = s["timing"]["total"]
                trans_acc = s["acceptance_rates"]["translation"]
                vol_acc = s["acceptance_rates"]["volume"]
                rxn_acc = s["acceptance_rates"]["reaction"]
                rho_mean = s["mean_rho_total"]
                rho_stderr = s["stderr_rho_total"]
                
                # Get species names from reaction
                species_names = sort(collect(keys(REACTIONS[rxn_name]["stoichiometry"])))
                means_str = join(["$(round(s["means"]["N_$name"], digits=1))" for name in species_names], " / ")
                
                println(f, "$rxn_name | $means_str | $(round(rho_mean, digits=4))±$(round(rho_stderr, digits=4)) | $(round(trans_acc, digits=3)) | $(round(vol_acc, digits=3)) | $(round(rxn_acc, digits=3)) | $(round(timing_total, digits=2))")
            end
        end
    end
    
    # Write JSON summary
    summary_json = Dict(
        "meta" => Dict(
            "seed" => seed,
            "sweeps_equil" => sweeps_equil,
            "sweeps_prod" => sweeps_prod,
            "timestamp" => string(Dates.now()),
            "total_runtime_seconds" => total_time,
        ),
        "parameters" => Dict(
            "sweeps_equil" => sweeps_equil,
            "sweeps_prod" => sweeps_prod,
            "sample_every" => sample_every,
            "block_size" => block_size,
        ),
        "reactions" => all_summaries,
    )
    
    open(summary_file_json, "w") do f
        JSON.print(f, summary_json, 2)
    end
    
    log_print(logger, "Summary files written:")
    log_print(logger, "  Text: $summary_file_txt")
    log_print(logger, "  JSON: $summary_file_json") 
    
    close(logfile_io)
end
