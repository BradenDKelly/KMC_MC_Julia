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
end

# EOS submodule
module EOS
    include("EOS/LJVirial.jl")
    include("EOS/LJKolafaNezbeda1994.jl")
    include("EOS/LJJohnson1993.jl")
    include("EOS/LJThol2016.jl")
end

# Re-export MC symbols
export MC

end
