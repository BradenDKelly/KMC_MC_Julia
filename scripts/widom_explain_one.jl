using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using Printf

# Initialize NVT LJ system
N = 256
ρ = 0.8
T = 1.0
rc = 2.5
max_disp = 0.1

p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp, seed=42)

# Warmup
for i in 1:50
    MolSim.MC.sweep!(st, p)
end

# Draw ONE random insertion point
L = st.L
x = [rand(st.rng) * L, rand(st.rng) * L, rand(st.rng) * L]

# Compute ΔU with explanation
ΔU, nterms, r_small, u_small, rmin = MolSim.MC.widom_deltaU_explain(st, p, x)

println("Insertion point: [$(round(x[1], digits=6)), $(round(x[2], digits=6)), $(round(x[3], digits=6))]")
println("nterms: $nterms")
println("rmin (from scan): $(round(rmin, digits=6))")
println("\nSmallest 10 r and u(r):")
println("  r         u(r)")
println("  " * "-" ^ 20)
n_print = min(10, length(r_small))
for i in 1:n_print
    @printf("  %8.6f  %12.6f\n", r_small[i], u_small[i])
end
println("\nΔU: $(round(ΔU, digits=6))")
