"""
MolSim: Molecular Simulation package for Monte Carlo and Kinetic Monte Carlo.
"""

module MolSim

# MC submodule
    module MC
        include("MC/PBC.jl")
        include("MC/NeighborList.jl")
        include("MC/LJLongRange.jl")
        include("MC/LJMC.jl")
        include("MC/BlockAveraging.jl")
        include("MC/Observables.jl")
        include("MC/Widom.jl")
        include("MC/Molecules.jl")
        include("MC/Fluctuations.jl")
    include("MC/eKMC.jl")
        include("MC/validation/CompareNVT.jl")
    end

# EOS submodule
module EOS
    include("EOS/NKEOS.jl")            # Python reference port (NKEOS)
    include("EOS/NKEOSAuthor.jl")      # Author wrapper (unit conversions)
end

# Analysis submodule
module Analysis
    include("Analysis/RDF.jl")
    include("Analysis/Snapshots.jl")
end

# Re-export MC and Analysis symbols
export MC
export Analysis

end
