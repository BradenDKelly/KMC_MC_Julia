include("src/MolSim.jl")
using .MolSim
pts = [(1.0,0.8),(1.2,0.8),(1.5,0.3),(0.8,0.5)]
for (T, ρ) in pts
    P = MolSim.EOS.pressure_nezbeda_author(T, ρ)
    U = MolSim.EOS.internal_energy_nezbeda_author(T, ρ)
    println("$(T) $(ρ) $(P) $(U)")
end
