"""
Sanity check for Kolafa SklogWiki implementation.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

include(joinpath(@__DIR__, "..", "src", "EOS", "LJKolafaSklogwiki1994.jl"))

println("Kolafa SklogWiki EOS sanity check")
println("=" ^ 60)

test_points = [
    (T=1.0, ρ=0.05),
    (T=1.0, ρ=0.1),
    (T=2.0, ρ=0.2),
]

for pt in test_points
    T = pt.T
    ρ = pt.ρ
    P = pressure_kolafa_sklogwiki(T, ρ)
    Z = P / (ρ * T)
    println("T=$T, ρ=$ρ: Z=$Z")
end
