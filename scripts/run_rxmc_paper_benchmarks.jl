"""
Paper-style Reaction Ensemble Monte Carlo benchmark runs.

Implements benchmarks matching paper tables (Table 3.1 and 3.2).
Species A-F with specified σ, ε, q values.
Reactions: A⇌B, A⇌C, A⇌2D, A⇌2E, A⇌D+F, A⇌D+E
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using Random
using StaticArrays

# Species definitions from Table 3.1
# A: σ=1.0, ε=1.0, q=0.002
# B: σ=1.0, ε=1.0, q=0.002
# C: σ=1.1, ε=0.9, q=0.002
# D: σ=1.0, ε=1.0, q=0.02
# E: σ=1.1, ε=0.9, q=0.02
# F: σ=1.0, ε=1.0, q=0.02

const SPECIES_DATA = Dict(
    "A" => (σ=1.0, ε=1.0, q=0.002),
    "B" => (σ=1.0, ε=1.0, q=0.002),
    "C" => (σ=1.1, ε=0.9, q=0.002),
    "D" => (σ=1.0, ε=1.0, q=0.02),
    "E" => (σ=1.1, ε=0.9, q=0.02),
    "F" => (σ=1.0, ε=1.0, q=0.02),
)

# Reaction definitions
# Each reaction maps species names to stoichiometric coefficients
const REACTIONS = Dict(
    "A⇌B" => Dict("A" => -1, "B" => 1),
    "A⇌C" => Dict("A" => -1, "C" => 1),
    "A⇌2D" => Dict("A" => -1, "D" => 2),
    "A⇌2E" => Dict("A" => -1, "E" => 2),
    "A⇌D+F" => Dict("A" => -1, "D" => 1, "F" => 1),
    "A⇌D+E" => Dict("A" => -1, "D" => 1, "E" => 1),
)

# Target values from Table 3.2 (ReKMC row) - to be filled in from paper
# Format: reaction_name => Dict("N_A" => target, "N_product" => target, "ρ_total" => target)
const TARGET_VALUES = Dict(
    "A⇌B" => Dict("N_A" => NaN, "N_B" => NaN, "ρ_total" => NaN),  # Placeholder
    "A⇌C" => Dict("N_A" => NaN, "N_C" => NaN, "ρ_total" => NaN),
    "A⇌2D" => Dict("N_A" => NaN, "N_D" => NaN, "ρ_total" => NaN),
    "A⇌2E" => Dict("N_A" => NaN, "N_E" => NaN, "ρ_total" => NaN),
    "A⇌D+F" => Dict("N_A" => NaN, "N_D" => NaN, "N_F" => NaN, "ρ_total" => NaN),
    "A⇌D+E" => Dict("N_A" => NaN, "N_D" => NaN, "N_E" => NaN, "ρ_total" => NaN),
)

function create_species_vectors(reaction_name::String, N_A_init::Int)
    """Create σ_types, ϵ_types, and logq vectors for a reaction."""
    rxn = REACTIONS[reaction_name]
    species_names = sort(collect(keys(rxn)))  # Deterministic order
    n_species = length(species_names)
    
    σ_types = Vector{Float64}(undef, n_species)
    ϵ_types = Vector{Float64}(undef, n_species)
    logq = Vector{Float64}(undef, n_species)
    
    for (idx, name) in enumerate(species_names)
        data = SPECIES_DATA[name]
        σ_types[idx] = data.σ
        ϵ_types[idx] = data.ε
        logq[idx] = log(data.q)  # log(q/λ³), assuming λ³=1 in reduced units
    end
    
    return (species_names, σ_types, ϵ_types, logq)
end

function create_reaction_object(reaction_name::String, species_names::Vector{String}, logq::Vector{Float64})
    """Create Reaction object from reaction name and species order."""
    rxn_dict = REACTIONS[reaction_name]
    n_species = length(species_names)
    stoichiometry = zeros(Int, n_species)
    
    for (idx, name) in enumerate(species_names)
        if haskey(rxn_dict, name)
            stoichiometry[idx] = rxn_dict[name]
        end
    end
    
    # logK = 0.0 (default, can be overridden)
    return MolSim.MC.Reaction(reaction_name, stoichiometry, 0.0; logq=logq)
end

function create_initial_state(N_A_init::Int, species_names::Vector{String}, ρ_init::Float64,
                               T::Float64, rc::Float64, max_disp::Float64, seed::Int)
    """Create initial state with N_A_init particles of type A (first species), zero products."""
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
    
    # Parameters will be created separately with multicomponent data
    return st
end

function run_benchmark(reaction_name::String, N_A_init::Int;
                       T::Float64=2.0, P::Float64=5.0,
                       ρ_init::Float64=0.75,
                       rc::Float64=2.5, max_disp::Float64=0.1,
                       max_dlnV::Float64=0.01, vol_move_every::Int=10,
                       p_reaction::Float64=0.1,
                       equilibration_sweeps::Int=10000,
                       production_sweeps::Int=50000,
                       sample_every::Int=10,
                       block_size::Int=50,
                       seed::Int=12345)
    """
    Run a single benchmark for a reaction.
    
    State point metadata:
    - Ensemble: NPT RxMC
    - N_init: N_A_init (all A initially)
    - T: temperature
    - P: external pressure
    - ρ_init: initial density guess
    - rc: cutoff distance
    - lj_model: :truncated
    - use_lrc: false
    - apply_impulsive_correction: false
    - max_disp: maximum displacement
    - max_dlnV: maximum volume change
    - vol_move_every: volume move frequency
    - p_reaction: reaction move probability per sweep
    - equilibration_sweeps: equilibration phase length
    - production_sweeps: production phase length
    - sample_every: sampling stride
    - block_size: block averaging block size
    - seed: RNG seed
    """
    
    println("=" ^ 80)
    println("Benchmark: $reaction_name")
    println("=" ^ 80)
    println("State point:")
    println("  Ensemble: NPT RxMC")
    println("  N_init: $N_A_init")
    println("  T: $T")
    println("  P: $P")
    println("  ρ_init: $ρ_init")
    println("  rc: $rc")
    println("  lj_model: :truncated")
    println("  use_lrc: false")
    println("  apply_impulsive_correction: false")
    println("  max_disp: $max_disp")
    println("  max_dlnV: $max_dlnV")
    println("  vol_move_every: $vol_move_every")
    println("  p_reaction: $p_reaction")
    println("  equilibration_sweeps: $equilibration_sweeps")
    println("  production_sweeps: $production_sweeps")
    println("  sample_every: $sample_every")
    println("  block_size: $block_size")
    println("  seed: $seed")
    println()
    
    # Create species vectors
    species_names, σ_types, ϵ_types, logq = create_species_vectors(reaction_name, N_A_init)
    n_species = length(species_names)
    
    println("Species: $species_names")
    println("σ: $σ_types")
    println("ε: $ϵ_types")
    println("q: $(exp.(logq))")
    println()
    
    # Create multicomponent parameters
    p = MolSim.MC.LJParams(σ_types=σ_types, ϵ_types=ϵ_types,
                           rc=rc, T=T, max_disp=max_disp, use_lrc=false, lj_model=:truncated)
    
    # Create reaction
    rxn = create_reaction_object(reaction_name, species_names, logq)
    
    # Create initial state
    st = create_initial_state(N_A_init, species_names, ρ_init, T, rc, max_disp, seed)
    
    # Update parameters to multicomponent
    st.types = fill(1, N_A_init)  # All type 1 (A)
    MolSim.MC.rebuild_cells!(st)
    
    # Equilibration
    println("Equilibration: $equilibration_sweeps sweeps...")
    for sweep in 1:equilibration_sweeps
        do_vol = (sweep % vol_move_every == 0)
        MolSim.MC.sweep_npt_with_reactions!(st, p, rxn; Pext=P, max_dlnV=max_dlnV,
                                            p_reaction=p_reaction, do_volume_move=do_vol,
                                            rebuild_every=st.N)
    end
    
    # Production: accumulate data
    println("Production: $production_sweeps sweeps...")
    N_species_samples = Vector{Vector{Int}}()  # Each element is counts for one sample
    ρ_samples = Float64[]
    V_samples = Float64[]
    
    for sweep in 1:production_sweeps
        do_vol = (sweep % vol_move_every == 0)
        MolSim.MC.sweep_npt_with_reactions!(st, p, rxn; Pext=P, max_dlnV=max_dlnV,
                                            p_reaction=p_reaction, do_volume_move=do_vol,
                                            rebuild_every=st.N)
        
        if sweep % sample_every == 0
            counts = MolSim.MC.count_species(st, n_species)
            push!(N_species_samples, counts)
            V = st.L * st.L * st.L
            ρ = st.N / V
            push!(ρ_samples, ρ)
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
        # Extract samples for this species
        samples = [N_species_samples[i][species_idx] for i in 1:n_samples]
        
        # Block averaging
        ba = MolSim.MC.BlockAverager(block_size)
        for s in samples
            push!(ba, Float64(s))
        end
        # finish! is not needed - mean/stderr work on incomplete blocks
        
        N_species_means[species_idx] = MolSim.MC.mean(ba)
        N_species_stderr[species_idx] = MolSim.MC.stderr(ba)
    end
    
    # Block average for density
    ρ_ba = MolSim.MC.BlockAverager(block_size)
    for ρ_val in ρ_samples
        push!(ρ_ba, ρ_val)
    end
    # finish! is not needed - mean/stderr work on incomplete blocks
    ρ_mean = MolSim.MC.mean(ρ_ba)
    ρ_stderr = MolSim.MC.stderr(ρ_ba)
    
    # Print results
    println()
    println("Results:")
    println("-" ^ 80)
    println("Species | Mean N | StdErr N")
    println("-" ^ 80)
    for (idx, name) in enumerate(species_names)
        println("$name      | $(round(N_species_means[idx], digits=2)) | $(round(N_species_stderr[idx], digits=2))")
    end
    println("ρ_total | $(round(ρ_mean, digits=4)) | $(round(ρ_stderr, digits=4))")
    println("-" ^ 80)
    
    # Compare with targets if available
    if haskey(TARGET_VALUES, reaction_name)
        targets = TARGET_VALUES[reaction_name]
        println()
        println("Comparison with targets:")
        println("-" ^ 80)
        println("Quantity | Measured | Target | % Error")
        println("-" ^ 80)
        
        for (idx, name) in enumerate(species_names)
            if haskey(targets, "N_$name")
                target = targets["N_$name"]
                if !isnan(target)
                    measured = N_species_means[idx]
                    pct_error = 100.0 * abs(measured - target) / target
                    println("N_$name | $(round(measured, digits=2)) | $target | $(round(pct_error, digits=2))%")
                end
            end
        end
        
        if haskey(targets, "ρ_total")
            target_ρ = targets["ρ_total"]
            if !isnan(target_ρ)
                pct_error = 100.0 * abs(ρ_mean - target_ρ) / target_ρ
                println("ρ_total | $(round(ρ_mean, digits=4)) | $target_ρ | $(round(pct_error, digits=2))%")
            end
        end
        println("-" ^ 80)
    end
    
    return (species_names, N_species_means, N_species_stderr, ρ_mean, ρ_stderr)
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    println("=" ^ 80)
    println("Reaction Ensemble Monte Carlo Paper Benchmarks")
    println("=" ^ 80)
    println()
    
    # Benchmark parameters
    N_A_init = 400
    T = 2.0
    P = 5.0
    ρ_init = 0.75
    
    # Run all reactions
    results = Dict()
    for rxn_name in keys(REACTIONS)
        try
            result = run_benchmark(rxn_name, N_A_init; T=T, P=P, ρ_init=ρ_init, seed=12345)
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
