"""
MolSim: Molecular Simulation package for Monte Carlo and Kinetic Monte Carlo.
"""

module MolSim

# Include submodules
include("MC/Backends.jl")
include("MC/PBC.jl")
include("MC/NeighborList.jl")
include("MC/LJMC.jl")

# Re-export important functions and types
export
    # Backends
    AbstractBackend, CPUBackend, CPU,
    # PBC
    wrap!, wrap, wrap_soa!, minimum_image, minimum_image!, minimum_image_distance_sq,
    # NeighborList
    CellBins, build_cells!, iterate_neighbor_cells, iterate_particles_in_cell, for_each_neighbor,
    # LJMC
    LJParams, LJState, lj_potential, local_energy, total_energy,
    init_lj_state, mc_step!, run!,
    # Ensemble
    ReplicaEnsemble, ReplicaStats, run_ensemble!

end # module MolSim
