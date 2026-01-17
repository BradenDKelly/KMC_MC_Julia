using BenchmarkTools

include(joinpath(@__DIR__, "..", "src", "MolSim.jl"))
using .MolSim

function make_state(; N::Int=864, ρ::Float64=0.8, T::Float64=1.0, rc::Float64=2.5, seed::Int=1234)
    # Create a mostly-A system with a couple of D/F particles for reaction feasibility
    types = ones(Int, N)
    if N >= 2
        types[1] = 2  # D
        types[2] = 3  # F
    end
    p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, seed=seed, types=types)
    return p, st
end

function parse_args()
    N = 864
    ρ = 0.8
    T = 1.0
    rc = 2.5
    seed = 1234
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--N" && i < length(ARGS)
            N = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--rho" && i < length(ARGS)
            ρ = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--T" && i < length(ARGS)
            T = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--rc" && i < length(ARGS)
            rc = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--seed" && i < length(ARGS)
            seed = parse(Int, ARGS[i+1]); i += 2
        else
            error("Unknown or incomplete argument: $arg")
        end
    end
    return (N=N, ρ=ρ, T=T, rc=rc, seed=seed)
end

args = parse_args()
N = args.N
ρ = args.ρ
T = args.T
rc = args.rc
seed = args.seed

println("Benchmarking with N=$N, ρ=$ρ, T=$T, rc=$rc, seed=$seed")

# Baseline: total energy
println("\n-- total_energy --")
bench_total = @benchmarkable MolSim.MC.total_energy(state[2], state[1]) setup=(state = make_state(N=N, ρ=ρ, T=T, rc=rc, seed=seed))
display(run(bench_total))

# Local energy for a single particle
println("\n-- local_energy (particle 1) --")
bench_local = @benchmarkable MolSim.MC.local_energy(1, state[2], state[1]) setup=(state = make_state(N=N, ρ=ρ, T=T, rc=rc, seed=seed))
display(run(bench_local))

# Alchemical flip (ΔN=0), uses local ΔU path
println("\n-- alchemical_flip_trial! (A⇌B-style) --")
bench_flip = @benchmarkable MolSim.MC.alchemical_flip_trial!(state[2], state[1], reaction_flip, :forward) setup=(state = make_state(N=N, ρ=ρ, T=T, rc=rc, seed=seed); reaction_flip = MolSim.MC.Reaction("A⇌B", [-1, 1, 0], 0.0; logq=[0.0, 0.0, 0.0]))
display(run(bench_flip))

# Reaction move (ΔN≠0): A⇌D+F
println("\n-- reaction_trial! (A⇌D+F, forward) --")
bench_df = @benchmarkable MolSim.MC.reaction_trial!(state[2], state[1], reaction_df, :forward; insertion_mode=:standard_remc) setup=(state = make_state(N=N, ρ=ρ, T=T, rc=rc, seed=seed); reaction_df = MolSim.MC.Reaction("A⇌D+F", [-1, 1, 1], 0.0; logq=[0.0, 0.0, 0.0]))
display(run(bench_df))
