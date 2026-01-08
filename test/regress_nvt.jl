using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using Test

@testset "NVT regression" begin
    # Settings (fast version for test)
    N = 864
    T = 1.0
    ρ = 0.8
    rc = 2.5
    use_lrc = true
    seed = 12345
    warmup_sweeps = 50
    prod_sweeps = 300
    
    # Initialize
    p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=seed, use_lrc=use_lrc)
    T_val = 1.0 / p.β
    
    # Check initialization
    @test st.N == N
    @test p.use_lrc == use_lrc
    @test size(st.pos) == (3, N)
    
    # Warmup
    for i in 1:warmup_sweeps
        MolSim.MC.sweep!(st, p)
    end
    
    # Production
    sample_every = 10
    total_acc = 0.0
    energy_samples = Float64[]
    pressure_samples = Float64[]
    
    for sweep_idx in 1:prod_sweeps
        acc = MolSim.MC.sweep!(st, p)
        total_acc += acc
        
        if sweep_idx % sample_every == 0
            E_total = MolSim.MC.total_energy(st, p)
            u_per_particle = E_total / st.N
            P = MolSim.MC.pressure(st, p, T_val)
            push!(energy_samples, u_per_particle)
            push!(pressure_samples, P)
        end
    end
    
    avg_acc = total_acc / prod_sweeps
    u_mean = sum(energy_samples) / length(energy_samples)
    p_mean = sum(pressure_samples) / length(pressure_samples)
    
    # Robust assertions
    @test 0.15 <= avg_acc <= 0.85
    
    if p.use_lrc
        u_sampled = u_mean - p.lrc_u_per_particle
        p_sampled = p_mean - p.lrc_p
        
        @test u_sampled < 0.0  # sampled energy per particle is negative
        @test u_mean < 0.0     # corrected energy per particle is negative
        @test isfinite(p_sampled) && !isnan(p_sampled)
        @test isfinite(p_mean) && !isnan(p_mean)
    else
        @test u_mean < 0.0
        @test isfinite(p_mean) && !isnan(p_mean)
    end
    
    # Check array sizes
    @test length(energy_samples) > 0
    @test length(pressure_samples) > 0
    @test length(energy_samples) == length(pressure_samples)
end
