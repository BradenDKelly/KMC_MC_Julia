using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim

# Check Z at required points
points = [
    (T=2.0, ρ=0.2),
    (T=1.0, ρ=0.1),
    (T=1.0, ρ=0.3),
]

println("Thol EOS Z values after fix:")
println("=" ^ 60)

for pt in points
    T = pt.T
    ρ = pt.ρ
    P = MolSim.EOS.pressure_thol(T, ρ)
    Z = P / (ρ * T)
    println("T=$T, ρ=$ρ: Z = $Z")
end
