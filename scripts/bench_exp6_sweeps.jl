using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using Printf

function run_case(N::Int; sweeps::Int=2000)
    rm = [1.0, 1.0, 1.0]
    ϵ = [1.0, 1.0, 1.0]
    α = [12.0, 12.0, 12.0]
    types = vcat(fill(1, N ÷ 4), fill(2, N ÷ 4), fill(3, N - 2*(N ÷ 4)))
    p, st = MolSim.MC.init_fcc_exp6(N=N, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42,
                                    types=types, rm_types=rm, ϵ_types=ϵ, α_types=α,
                                    cutoff_model=:truncated, mixing_rule=:lb)
    t0 = time()
    for _ in 1:sweeps
        MolSim.MC.sweep!(st, p; rebuild_every=st.N)
    end
    t = time() - t0
    @printf("N=%d sweeps=%d time_per_sweep=%.6f s\n", N, sweeps, t / sweeps)
end

run_case(100)
run_case(400)
