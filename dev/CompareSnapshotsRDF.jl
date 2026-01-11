#!/usr/bin/env julia
"""
Script to run LJ Metropolis NVT MC simulation, save snapshots, and compute RDF.

Usage: julia --project dev/CompareSnapshotsRDF.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using Random

# ============================================================================
# Parameters (edit these as needed)
# ============================================================================
const N = 500
const ρ = 0.5
const T = 1.0
const rc = 2.5
const max_disp = 0.1
const nsweeps = 10000
const seed = 1234

# RDF parameters
const rmax = nothing  # Will be set to L/2
const nbins = 200

# Output directory
const output_dir = joinpath(@__DIR__, "out")

# ============================================================================
# Helper functions
# ============================================================================

"""
Write RDF to CSV file.
"""
function write_rdf_csv(path::String, r::Vector{Float64}, g_r::Vector{Float64})
    open(path, "w") do io
        println(io, "r,g_r")
        for i in eachindex(r)
            println(io, "$(r[i]),$(g_r[i])")
        end
    end
    return nothing
end

# ============================================================================
# Main script
# ============================================================================

# Create output directory
if !isdir(output_dir)
    mkdir(output_dir)
end

println("=" ^ 80)
println("NVT MC: Snapshots and RDF")
println("=" ^ 80)
println()
println("Parameters:")
println("  N = $N")
println("  ρ = $ρ")
println("  T = $T")
println("  rc = $rc")
println("  nsweeps = $nsweeps")
println()

# Initialize system
Random.seed!(seed)
p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp, seed=seed, use_lrc=false)

L = st.L
rmax_actual = rmax === nothing ? L / 2.0 : rmax

println("Box size L = $L")
println("RDF rmax = $rmax_actual")
println("RDF nbins = $nbins")
println()

# Save initial snapshot
println("Running Metropolis MC...")
println("  Sweep 0 (initial)...")
MolSim.Analysis.write_xyz(joinpath(output_dir, "mc_start.xyz"), st.pos, 
                          types=st.types, comment="MC initial, T=$T, ρ=$ρ, N=$N")

# Run MC simulation
midpoint = nsweeps ÷ 2
for sweep in 1:nsweeps
    MolSim.MC.sweep!(st, p; rebuild_every=1)
    
    if sweep == midpoint
        println("  Sweep $sweep (midpoint)...")
        MolSim.Analysis.write_xyz(joinpath(output_dir, "mc_mid.xyz"), st.pos,
                                  types=st.types, comment="MC midpoint, T=$T, ρ=$ρ, N=$N, sweep=$sweep")
    end
end

println("  Sweep $nsweeps (final)...")
MolSim.Analysis.write_xyz(joinpath(output_dir, "mc_end.xyz"), st.pos,
                          types=st.types, comment="MC final, T=$T, ρ=$ρ, N=$N, sweeps=$nsweeps")

# Compute final energy and pressure
U_mc = MolSim.MC.total_energy(st, p) / N
P_mc = MolSim.MC.pressure(st, p, T)

println()
println("Computing RDF for final configuration...")
r_bins, g_r = MolSim.Analysis.rdf(st.pos, L, rmax_actual, nbins)
write_rdf_csv(joinpath(output_dir, "rdf_mc.csv"), r_bins, g_r)

# Summary
println()
println("=" ^ 80)
println("Summary")
println("=" ^ 80)
println("Energy per particle: $(U_mc)")
println("Pressure: $(P_mc)")
println()
println("Output files written to $(output_dir):")
println("  mc_start.xyz - Initial configuration")
println("  mc_mid.xyz - Midpoint configuration")
println("  mc_end.xyz - Final configuration")
println("  rdf_mc.csv - Radial distribution function (final config)")
println()
println("To view XYZ files: Use OVITO (https://www.ovito.org/) or VMD")
