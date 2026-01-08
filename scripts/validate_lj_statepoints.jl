using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using Printf

# Flag to use plain Widom (false) or cavity-biased (true, default for dense)
use_cavity_biased = true
# Flag to enable long-range corrections
use_lrc = true

# State points to test
state_points = [
    (T=1.0, ρ=0.3),
    (T=1.0, ρ=0.8),
    (T=1.5, ρ=0.8)
]

N = 256
rc = 2.5
max_disp = 0.1
warmup_sweeps = 100
production_sweeps = 300
sample_every = 5
block_size = 20
widom_insertions = 50
rmin_cut = 0.85

println("T    ρ    u/N(mean±se)        p(mean±se)          μ_ex(mean±se)       acc")
println("-" ^ 80)

for sp in state_points
    T = sp.T
    ρ = sp.ρ
    
    # Initialize
    p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp, seed=42, use_lrc=use_lrc)
    
    # Print LJ parameters (once per state point)
    if sp == state_points[1]
        println("LJ: rc=$(rc), shifted=false, LRC=$(use_lrc)")
    end
    
    # Warmup
    for i in 1:warmup_sweeps
        MolSim.MC.sweep!(st, p)
    end
    
    # Production
    energy_ba = MolSim.MC.BlockAverager(block_size)
    pressure_ba = MolSim.MC.BlockAverager(block_size)
    mu_ex_ba = MolSim.MC.BlockAverager(block_size)
    widom_acc = MolSim.MC.WidomAccumulator()
    total_acc = 0.0
    
    for sweep_idx in 1:production_sweeps
        acc = MolSim.MC.sweep!(st, p)
        total_acc += acc
        
        if sweep_idx % sample_every == 0
            E_total = MolSim.MC.total_energy(st, p)
            u_per_particle = E_total / N
            P = MolSim.MC.pressure(st, p, T)
            push!(energy_ba, u_per_particle)
            push!(pressure_ba, P)
            
            MolSim.MC.reset!(widom_acc)
            # Use cavity-biased for dense cases (ρ >= 0.7) unless flag is set
            if ρ >= 0.7 && use_cavity_biased
                μ_ex, _ = MolSim.MC.widom_mu_ex_cavity!(widom_acc, st, p; ninsert=widom_insertions, rmin_cut=rmin_cut)
            else
                MolSim.MC.widom_mu_ex!(widom_acc, st, p; ninsert=widom_insertions)
                μ_ex = MolSim.MC.mu_ex(widom_acc, p.β)
            end
            push!(mu_ex_ba, μ_ex)
        end
    end
    
    avg_acc = total_acc / production_sweeps
    u_mean = MolSim.MC.mean(energy_ba)
    u_se = MolSim.MC.stderr(energy_ba)
    p_mean = MolSim.MC.mean(pressure_ba)
    p_se = MolSim.MC.stderr(pressure_ba)
    mu_ex_mean = MolSim.MC.mean(mu_ex_ba)
    mu_ex_se = MolSim.MC.stderr(mu_ex_ba)
    
    @printf("%.1f  %.1f  %.4f±%.4f  %.4f±%.4f  %.4f±%.4f  %.4f\n",
            T, ρ, u_mean, u_se, p_mean, p_se, mu_ex_mean, mu_ex_se, avg_acc)
end
