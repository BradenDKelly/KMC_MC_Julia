"""
Table 3.2 Reaction Ensemble Monte Carlo benchmark runner.

Reproduces Table 3.2 from Kelly Braden's thesis using NPT Reaction Ensemble MC.
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

function print_statepoint_echo(reaction_name::String, species_names::Vector{String},
                                σ_types::Vector{Float64}, ϵ_types::Vector{Float64},
                                logq::Vector{Float64}, stoichiometry::Vector{Int},
                                T::Float64, P::Float64, rc::Float64,
                                N_init::Dict{String,Int}, ρ_init::Float64,
                                seed::Int, max_disp::Float64, max_dlnV::Float64,
                                vol_move_every::Int, p_reaction::Float64,
                                sweeps_equil::Int, sweeps_prod::Int,
                                sample_every::Int, block_size::Int)
    """Print structured statepoint echo."""
    println("=" ^ 80)
    println("STATEPOINT ECHO")
    println("=" ^ 80)
    println("Reaction: $reaction_name")
    print("Stoichiometry: ")
    for (idx, name) in enumerate(species_names)
        if idx > 1
            print(" + ")
        end
        ν = stoichiometry[idx]
        if ν < 0
            print("$(-ν)$name")
        elseif ν > 0
            print("$ν$name")
        end
    end
    println()
    println("Ensemble: NPT Reaction Ensemble MC (REMC)")
    println("Seed: $seed")
    println("T = $T")
    println("P = $P")
    println("rc = $rc")
    println("LJ model: shifted (cut-and-shifted)")
    println("LRC: false")
    println("Mixing rule: Lorentz-Berthelot (σ_ab = (σ_a + σ_b)/2, ε_ab = sqrt(ε_a * ε_b))")
    println()
    println("Species parameters:")
    for (idx, name) in enumerate(species_names)
        println("  $name: σ=$(σ_types[idx]), ε=$(ϵ_types[idx]), q=$(exp(logq[idx])), logq=$(logq[idx])")
    end
    println()
    println("Initial composition:")
    for name in species_names
        n_init = get(N_init, name, 0)
        println("  N_$name = $n_init")
    end
    println("Initial density guess: $ρ_init")
    println()
    println("Move parameters:")
    println("  Translation: max_disp = $max_disp")
    println("  Volume: max_dlnV = $max_dlnV, every $vol_move_every sweeps")
    println("  Reaction: p_reaction = $p_reaction per sweep")
    println()
    println("Run lengths:")
    println("  Equilibration: $sweeps_equil sweeps")
    println("  Production: $sweeps_prod sweeps")
    println("  Sample every: $sample_every")
    println("  Block size: $block_size")
    println("=" ^ 80)
    println()
end

function run_reaction_benchmark(reaction_name::String;
                                seed::Int=12345,
                                sweeps_equil::Int=50000,
                                sweeps_prod::Int=200000,
                                output_dir::String="results",
                                max_disp::Float64=0.1,
                                max_dlnV::Float64=0.01,
                                vol_move_every::Int=10,
                                p_reaction::Float64=0.1,
                                sample_every::Int=10,
                                block_size::Int=50)
    """Run a single reaction benchmark."""
    
    # Create species vectors
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
    
    # Initial composition
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
    print_statepoint_echo(reaction_name, species_names, σ_types, ϵ_types, logq,
                          stoichiometry, T_TARGET, P_TARGET, RC,
                          N_init, RHO_INIT_GUESS, seed, max_disp, max_dlnV,
                          vol_move_every, p_reaction, sweeps_equil, sweeps_prod,
                          sample_every, block_size)
    
    # Create initial state
    st = create_initial_state(species_names, N_A_INIT, RHO_INIT_GUESS,
                               T_TARGET, RC, max_disp, seed)
    
    # Equilibration
    println("Equilibration: $sweeps_equil sweeps...")
    for sweep in 1:sweeps_equil
        do_vol = (sweep % vol_move_every == 0)
        MolSim.MC.sweep_npt_with_reactions!(st, p, rxn; Pext=P_TARGET, max_dlnV=max_dlnV,
                                            p_reaction=p_reaction, do_volume_move=do_vol,
                                            rebuild_every=st.N)
    end
    
    # Production: accumulate data
    println("Production: $sweeps_prod sweeps...")
    N_species_samples = Vector{Vector{Int}}()
    V_samples = Float64[]
    
    translation_accepted_total = 0
    translation_attempted_total = 0
    volume_accepted_total = 0
    volume_attempted_total = 0
    reaction_accepted_total = 0
    reaction_attempted_total = 0
    
    for sweep in 1:sweeps_prod
        do_vol = (sweep % vol_move_every == 0)
        trans_acc, vol_acc, vol_att, rxn_acc, rxn_att = 
            MolSim.MC.sweep_npt_with_reactions!(st, p, rxn; Pext=P_TARGET, max_dlnV=max_dlnV,
                                                p_reaction=p_reaction, do_volume_move=do_vol,
                                                rebuild_every=st.N)
        
        translation_attempted_total += st.N
        translation_accepted_total += Int(round(trans_acc * st.N))
        volume_accepted_total += vol_acc
        volume_attempted_total += vol_att
        reaction_accepted_total += rxn_acc
        reaction_attempted_total += rxn_att
        
        if sweep % sample_every == 0
            counts = MolSim.MC.count_species(st, n_species)
            push!(N_species_samples, counts)
            V = st.L * st.L * st.L
            push!(V_samples, V)
        end
    end
    
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
    println()
    println("Results:")
    println("-" ^ 80)
    println("Species | Mean N | StdErr N")
    println("-" ^ 80)
    for (idx, name) in enumerate(species_names)
        println("$name      | $(round(N_species_means[idx], digits=2)) | $(round(N_species_stderr[idx], digits=2))")
    end
    println("V       | $(round(V_mean, digits=4)) | $(round(V_stderr, digits=4))")
    println("ρ_total | $(round(ρ_total_mean, digits=4)) | $(round(ρ_total_stderr, digits=4))")
    println("-" ^ 80)
    println("Acceptance rates:")
    println("  Translation: $(round(trans_acc_rate, digits=4))")
    println("  Volume: $(round(vol_acc_rate, digits=4))")
    println("  Reaction: $(round(rxn_acc_rate, digits=4))")
    
    # Compare with targets
    targets = REACTIONS[reaction_name]["target"]
    println()
    println("Comparison with Table 3.2 targets:")
    println("-" ^ 80)
    println("Quantity | Measured | Target | % Error")
    println("-" ^ 80)
    
    for (idx, name) in enumerate(species_names)
        target_key = "N_$name"
        if haskey(targets, target_key)
            target = targets[target_key]
            measured = N_species_means[idx]
            pct_error = 100.0 * abs(measured - target) / target
            println("N_$name | $(round(measured, digits=2)) | $target | $(round(pct_error, digits=2))%")
        end
    end
    
    if haskey(targets, "ρ_total")
        target_ρ = targets["ρ_total"]
        pct_error = 100.0 * abs(ρ_total_mean - target_ρ) / target_ρ
        println("ρ_total | $(round(ρ_total_mean, digits=4)) | $target_ρ | $(round(pct_error, digits=2))%")
    end
    println("-" ^ 80)
    
    # Prepare JSON output
    json_data = Dict(
        "meta" => Dict(
            "reaction" => reaction_name,
            "seed" => seed,
            "timestamp" => string(now()),
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
    
    # Write JSON (replace NaN with null for JSON compatibility)
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
    
    mkpath(output_dir)
    json_filename = joinpath(output_dir, "table3_2_$(reaction_name)_seed$(seed).json")
    open(json_filename, "w") do f
        JSON.print(f, json_data, 2)
    end
    println()
    println("JSON output written to: $json_filename")
    println()
    
    return json_data
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    
    # Parse CLI args (simple)
    seed = 12345
    sweeps_equil = 50000
    sweeps_prod = 200000
    output_dir = "results"
    
    if length(ARGS) >= 1
        seed = parse(Int, ARGS[1])
    end
    if length(ARGS) >= 2
        sweeps_equil = parse(Int, ARGS[2])
    end
    if length(ARGS) >= 3
        sweeps_prod = parse(Int, ARGS[3])
    end
    if length(ARGS) >= 4
        output_dir = ARGS[4]
    end
    
    println("=" ^ 80)
    println("Table 3.2 Reaction Ensemble Monte Carlo Benchmarks")
    println("=" ^ 80)
    println("Running all reactions with:")
    println("  seed = $seed")
    println("  sweeps_equil = $sweeps_equil")
    println("  sweeps_prod = $sweeps_prod")
    println("  output_dir = $output_dir")
    println()
    
    # Run all reactions
    results = Dict()
    for rxn_name in keys(REACTIONS)
        try
            result = run_reaction_benchmark(rxn_name; seed=seed, sweeps_equil=sweeps_equil,
                                            sweeps_prod=sweeps_prod, output_dir=output_dir)
            results[rxn_name] = result
        catch e
            println("Error running $rxn_name: $e")
            rethrow(e)
        end
        println()
    end
    
    println("=" ^ 80)
    println("All benchmarks completed")
    println("=" ^ 80)
end
