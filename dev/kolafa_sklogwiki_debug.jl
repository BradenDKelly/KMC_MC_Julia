using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "EOS", "LJKolafaSklogwiki1994.jl"))

T = 1.0
rho = 0.05

d = dC(T)
eta = (π / 6.0) * rho * d^3
z_hs = zHS(eta)
BC_val = BC(T)
gamma = gammaBH(T)
rho2 = rho * rho
exp_val = exp(gamma * rho2)
BC_term = (BC_val / exp_val) * rho * (1.0 - 2.0 * gamma * rho2)

println("T=$T, rho=$rho:")
println("  d = $d")
println("  eta = $eta")
println("  z_hs = $z_hs")
println("  BC_val = $BC_val")
println("  gamma = $gamma")
println("  exp_val = $exp_val")
println("  BC_term = $BC_term")
println("  z_hs + BC_term = $(z_hs + BC_term)")

# Compute sum
let sum_poly = 0.0
    for (i, j, cij) in C_CIJ_SKW
        T_power = T^(i / 2.0)
        rho_power = rho^j
        sum_poly += cij * T_power * rho_power
    end
    println("  sum_poly = $sum_poly")
    
    PLJ = ((z_hs + BC_term) * T + sum_poly) * rho
    Z = PLJ / (rho * T)
    println("  PLJ = $PLJ")
    println("  Z = $Z")
end

