using Test
using MolSim
using Random

@testset "NVT energy fluctuations → C_V" begin
    # Test parameters
    N = 108
    ρ = 0.8
    T = 1.0
    rc = 2.5
    max_disp = 0.1
    seed = 12345
    warmup_sweeps = 10000
    prod_sweeps = 50000
    sample_every = 50
    block_size = 50
    
    p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp, 
                               seed=seed, use_lrc=false, lj_model=:truncated)
    
    # Warmup
    for _ in 1:warmup_sweeps
        MolSim.MC.sweep!(st, p; rebuild_every=1)
    end
    
    # Production: sample energies
    U_samples = Float64[]
    for sweep in 1:prod_sweeps
        MolSim.MC.sweep!(st, p; rebuild_every=1)
        if sweep % sample_every == 0
            U = MolSim.MC.total_energy(st, p)
            push!(U_samples, U)
        end
    end
    
    # Compute C_V (configurational, from potential energy fluctuations only)
    C_V, mean_U = MolSim.MC.heat_capacity_CV(U_samples, T, block_size)
    
    # Assertions
    @test C_V > 0.0
    @test isfinite(C_V)
    @test isfinite(mean_U)
    
    # Regression baseline: configurational C_V for LJ fluid at T=1.0, ρ=0.8
    # Note: This is configurational C_V (potential energy only), not full C_V.
    # For full C_V, add 3N/2 (ideal gas contribution).
    # Stored baseline: C_V/N ≈ 0.507 (deterministic seed, 1000 samples)
    C_V_per_particle = C_V / Float64(N)
    @test C_V_per_particle > 0.3   # Lower bound (configurational C_V is typically smaller)
    @test C_V_per_particle < 2.0   # Upper bound (reasonable for configurational C_V at this state point)
    
    # Compare against stored baseline with reasonable tolerance for MC statistics
    baseline_CV_per_particle = 0.507
    @test C_V_per_particle ≈ baseline_CV_per_particle rtol=0.3  # 30% tolerance for MC fluctuations
    
    # Energy should be negative and reasonable
    @test mean_U < 0.0
    @test mean_U / Float64(N) > -10.0  # Per-particle energy should be reasonable
end

@testset "NVT C_V consistency with different step sizes" begin
    # Test that C_V is relatively stable with different max_disp
    N = 108
    ρ = 0.8
    T = 1.0
    rc = 2.5
    seed = 54321
    warmup_sweeps = 5000
    prod_sweeps = 30000
    sample_every = 50
    block_size = 50
    
    C_V_values = Float64[]
    
    for max_disp in [0.05, 0.15]
        p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp,
                                   seed=seed, use_lrc=false, lj_model=:truncated)
        
        # Warmup
        for _ in 1:warmup_sweeps
            MolSim.MC.sweep!(st, p; rebuild_every=1)
        end
        
        # Production
        U_samples = Float64[]
        for sweep in 1:prod_sweeps
            MolSim.MC.sweep!(st, p; rebuild_every=1)
            if sweep % sample_every == 0
                U = MolSim.MC.total_energy(st, p)
                push!(U_samples, U)
            end
        end
        
        C_V, _ = MolSim.MC.heat_capacity_CV(U_samples, T, block_size)
        push!(C_V_values, C_V)
    end
    
    # C_V should be similar (within ~30% relative difference for short runs)
    @test length(C_V_values) == 2
    rel_diff = abs(C_V_values[1] - C_V_values[2]) / max(C_V_values[1], C_V_values[2])
    @test rel_diff < 0.5  # Loose tolerance for short runs
end

@testset "NPT volume fluctuations → κ_T" begin
    # Test parameters
    N = 108
    T = 2.0
    rc = 2.5
    max_disp = 0.1
    max_dlnV = 0.01
    Pext = 1.0
    seed = 67890
    warmup_sweeps = 10000
    prod_sweeps = 50000
    sample_every = 50
    vol_move_every = 10
    block_size = 50
    
    # Initialize at moderate density
    ρ_init = 0.5
    p, st = MolSim.MC.init_fcc(N=N, ρ=ρ_init, T=T, rc=rc, max_disp=max_disp,
                               seed=seed, use_lrc=false, lj_model=:truncated)
    
    # Warmup
    for sweep in 1:warmup_sweeps
        MolSim.MC.sweep!(st, p; rebuild_every=1)
        if sweep % vol_move_every == 0
            MolSim.MC.volume_trial!(st, p; max_dlnV=max_dlnV, Pext=Pext)
        end
    end
    
    # Production: sample volumes
    V_samples = Float64[]
    for sweep in 1:prod_sweeps
        MolSim.MC.sweep!(st, p; rebuild_every=1)
        if sweep % vol_move_every == 0
            MolSim.MC.volume_trial!(st, p; max_dlnV=max_dlnV, Pext=Pext)
            if sweep % sample_every == 0
                V = st.L * st.L * st.L
                push!(V_samples, V)
            end
        end
    end
    
    # Compute κ_T
    κ_T, mean_V = MolSim.MC.compressibility_kappaT(V_samples, T, block_size)
    
    # Assertions
    @test κ_T > 0.0
    @test isfinite(κ_T)
    @test isfinite(mean_V)
    @test mean_V > 0.0
    
    # Regression baseline: κ_T should be in a reasonable range
    # For LJ fluid at moderate density: κ_T typically O(0.1-1.0) in reduced units
    @test κ_T > 0.01
    @test κ_T < 10.0  # Very loose upper bound
end
