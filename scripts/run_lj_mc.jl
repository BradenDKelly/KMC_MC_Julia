"""
Example script to run Lennard-Jones Monte Carlo simulation.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Add src directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using BenchmarkTools
using Statistics
using Base.Threads
using Printf

# Simulation parameters
N = 108              # Number of particles
ρ = 0.8              # Reduced density (N*σ^3/V)
σ = 1.0              # LJ size parameter
ϵ = 1.0              # LJ energy parameter
T = 1.2              # Reduced temperature (kB*T/ϵ)
rc = 2.5 * σ         # Cutoff distance
max_disp = 0.1 * σ   # Maximum displacement for trial moves

# Compute box length from density
V = N / ρ
L = cbrt(V)

println("Lennard-Jones Monte Carlo Simulation")
println("=" ^ 50)
println("N = $N")
println("ρ = $ρ")
println("L = $L")
println("T = $T")
println("rc = $rc")
println("max_disp = $max_disp")
println("=" ^ 50)

# Create parameters
params = LJParams(ϵ, σ, rc, L, N, T, max_disp)

# Initialize state
println("\nInitializing state...")
state = init_lj_state(params; seed=42)

# Compute initial energy per particle
E0 = state.energy / N
println("Initial energy per particle: $E0")

# Run simulation
println("\nRunning MC simulation...")
nsteps = 100_000
sample_every = 100

acceptance_rate, energies = run!(state, params; nsteps=nsteps, sample_every=sample_every)

# Compute final energy per particle
E_final = state.energy / N

# Compute average energy over samples
E_avg = sum(energies) / (length(energies) * N)

println("\nResults:")
println("=" ^ 50)
println("Acceptance rate: $acceptance_rate")
println("Final energy per particle: $E_final")
println("Average energy per particle: $E_avg")
println("=" ^ 50)

# Benchmark section
println("\n" * "=" ^ 50)
println("BENCHMARK")
println("=" ^ 50)

# Create a fresh state for benchmarking
println("\nCreating fresh state for benchmarking...")
bench_state = init_lj_state(params; seed=12345)

# Warmup with @time
println("\nWarming up with @time...")
warmup_steps = 1000
@time begin
    for i in 1:warmup_steps
        mc_step!(bench_state, params)
    end
end

# Reset state after warmup
bench_state = init_lj_state(params; seed=12345)

# Benchmark with @btime (for comparison)
println("\nBenchmarking with @btime (block of 100 steps)...")
@btime begin
    for i in 1:100
        mc_step!($bench_state, $params)
    end
end

# Reset state again for detailed @benchmark analysis
bench_state = init_lj_state(params; seed=12345)

# Benchmark with @benchmark (captures results for detailed analysis)
println("\nDetailed benchmark with @benchmark (block of 100 steps)...")
bench_result = @benchmark begin
    for i in 1:100
        mc_step!($bench_state, $params)
    end
end samples=1000 evals=1

# Extract timing and allocation info from trial
trial = bench_result.times
time_per_100_steps_ns = median(trial)  # median time in nanoseconds
time_per_step_ns = time_per_100_steps_ns / 100  # in nanoseconds
time_per_step_us = time_per_step_ns / 1000  # in microseconds
steps_per_sec = 1e9 / time_per_step_ns  # convert nanoseconds to seconds

allocs_trial = bench_result.allocs
allocated_bytes = median(allocs_trial)  # median bytes allocated for 100 steps
allocated_per_step = allocated_bytes / 100  # bytes per step

println("\nBenchmark Results:")
println("-" ^ 50)
println("Time per step: $(round(time_per_step_us, digits=2)) μs")
println("Steps per second: $(round(steps_per_sec, digits=0))")
println("Allocated bytes per 100 steps: $allocated_bytes")
println("Allocated bytes per step: $allocated_per_step")

if allocated_per_step < 1.0
    println("✓ Zero-allocation inner loop achieved!")
else
    println("⚠ Warning: $(round(allocated_per_step, digits=2)) bytes allocated per step")
    println("  This may indicate temporary allocations in the inner loop.")
end
println("=" ^ 50)

# Parallel ensemble example
println("\n" * "=" ^ 50)
println("PARALLEL ENSEMBLE EXAMPLE")
println("=" ^ 50)

# Number of replicas
R = 4
println("\nCreating ensemble of $R replicas...")
println("Available threads: $(Threads.nthreads())")

# Create ensemble with independent seeds
ensemble = ReplicaEnsemble(params, R; base_seed=1000)

# Run ensemble in parallel
println("\nRunning ensemble in parallel ($R replicas, $(Threads.nthreads()) threads)...")
println("Each replica runs $(nsteps) steps independently...")

@time stats = run_ensemble!(ensemble, params; nsteps=nsteps, sample_every=sample_every)

# Print per-replica statistics
println("\nPer-Replica Statistics:")
println("-" ^ 80)
println("Replica | Acceptance | Initial Energy | Final Energy  | Average Energy")
println("-" ^ 80)
for stat in stats
    println(@sprintf("   %2d   |   %6.2f%%  |    %10.4f   |   %10.4f   |   %10.4f",
                     stat.replica_id,
                     stat.acceptance_rate * 100,
                     stat.initial_energy / N,
                     stat.final_energy / N,
                     stat.average_energy / N))
end
println("-" ^ 80)

# Summary statistics
avg_acceptance = mean([s.acceptance_rate for s in stats]) * 100
avg_final_energy = mean([s.final_energy / N for s in stats])
std_final_energy = std([s.final_energy / N for s in stats])
avg_avg_energy = mean([s.average_energy / N for s in stats])

println("\nEnsemble Summary:")
println("-" ^ 50)
println("Average acceptance rate: $(round(avg_acceptance, digits=2))%")
println("Average final energy per particle: $(round(avg_final_energy, digits=4)) ± $(round(std_final_energy, digits=4))")
println("Average sampled energy per particle: $(round(avg_avg_energy, digits=4))")
println("=" ^ 50)
