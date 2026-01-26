"""
Tests for MD barostat and NPT MD simulations.
"""

using Test
using MolSim
using Random

@testset "MD barostat basic functionality" begin
    # Initialize a small system
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    
    # Initialize MC state and convert to MD
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=12345,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    st_md = MC.init_md_state(st_mc.pos, T, Random.Xoshiro(12345))
    st_md.L = st_mc.L
    st_md.types = st_mc.types
    
    # Initialize forces
    MC.compute_forces!(st_md, p)
    
    # Store initial state
    L_initial = st_md.L
    V_initial = L_initial^3
    ρ_initial = N / V_initial
    
    # Test barostat with target pressure
    P_target = 1.0
    dt = 0.001
    tau_p = 1.0
    
    # Apply barostat a few times
    for i in 1:10
        MC.berendsen_barostat!(st_md, p, T, P_target, dt, tau_p)
        MC.compute_forces!(st_md, p)
        
        # Check that box size is valid
        @test st_md.L > 0.0
        @test isfinite(st_md.L)
        
        # Check that all positions are still in box
        for j in 1:N
            for d in 1:3
                @test 0.0 <= st_md.pos[d, j] < st_md.L
            end
        end
    end
    
    # Box size should have changed (unless pressure was exactly right)
    # But it should still be reasonable
    @test st_md.L > 0.5 * L_initial
    @test st_md.L < 2.0 * L_initial
end

@testset "MD NPT short run - box size stability" begin
    # Initialize a small system
    N = 32
    ρ_initial = 0.8
    T = 1.0
    rc = 2.5
    
    # Initialize MC state and convert to MD
    _, st_mc = MC.init_simple(N=N, ρ=ρ_initial, T=T, rc=rc, max_disp=0.1, seed=54321,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    st_md = MC.init_md_state(st_mc.pos, T, Random.Xoshiro(54321))
    st_md.L = st_mc.L
    st_md.types = st_mc.types
    
    L_initial = st_md.L
    V_initial = L_initial^3
    ρ_initial_calc = N / V_initial
    
    # Run short NPT MD simulation
    P_target = 1.0
    dt = 0.001
    nsteps = 1000
    thermostat_every = 10
    barostat_every = 10
    tau_p = 1.0
    
    # Initialize forces
    MC.compute_forces!(st_md, p)
    MC.velocity_rescale!(st_md, T)
    
    # Track observables
    count = 0
    L_samples = Float64[]
    P_samples = Float64[]
    ρ_samples = Float64[]
    L_max = L_initial
    L_min = L_initial
    
    for step in 1:nsteps
        MC.velocity_verlet_step!(st_md, p, dt)
        
        if step % thermostat_every == 0
            MC.velocity_rescale!(st_md, T)
        end
        
        if step % barostat_every == 0
            L_before = st_md.L
            MC.berendsen_barostat!(st_md, p, T, P_target, dt, tau_p)
            MC.compute_forces!(st_md, p)
            L_after = st_md.L
            
            # Check that box size change is reasonable (barostat should limit to ±5%)
            if L_before > 0.0
                scale = L_after / L_before
                @test 0.95 <= scale <= 1.05  # Barostat should limit to ±5% per step
            end
        end
        
        if step % thermostat_every == 0
            P_inst = MC.pressure(st_md, p, T)
            ρ_inst = N / (st_md.L^3)
            if isfinite(P_inst) && isfinite(ρ_inst) && st_md.L > 0.0 && ρ_inst > 0.0
                push!(L_samples, st_md.L)
                push!(P_samples, P_inst)
                push!(ρ_samples, ρ_inst)
                L_max = max(L_max, st_md.L)
                L_min = min(L_min, st_md.L)
                count += 1
            end
        end
        
        # Critical test: box size should never explode or collapse
        @test st_md.L > 0.1 * L_initial  # Shouldn't collapse to < 10% of initial
        @test st_md.L < 10.0 * L_initial  # Shouldn't explode to > 10x initial
        @test isfinite(st_md.L)
    end
    
    # Should have collected some samples
    @test count > 0
    @test length(L_samples) == count
    @test length(P_samples) == count
    @test length(ρ_samples) == count
    
    # All samples should be valid
    @test all(L -> L > 0.0 && isfinite(L), L_samples)
    @test all(P -> isfinite(P), P_samples)
    @test all(ρ -> ρ > 0.0 && isfinite(ρ), ρ_samples)
    
    # Critical test: box size should remain reasonable
    if count > 0
        L_avg = sum(L_samples) / count
        L_std = sqrt(sum((L - L_avg)^2 for L in L_samples) / count)
        P_avg = sum(P_samples) / count
        ρ_avg = sum(ρ_samples) / count
        
        @test isfinite(L_avg)
        @test isfinite(P_avg)
        @test isfinite(ρ_avg)
        
        # Box size should be within reasonable bounds (not exploded or collapsed)
        @test L_avg > 0.5 * L_initial  # Shouldn't collapse to < 50% of initial
        @test L_avg < 2.0 * L_initial  # Shouldn't explode to > 2x initial
        
        # Box size variation should be reasonable (barostat should control it)
        @test L_std < 0.5 * L_initial  # Standard deviation shouldn't be huge
        
        # Density should be reasonable (not near zero or extremely high)
        @test ρ_avg > 0.1  # Shouldn't be near zero
        @test ρ_avg < 10.0  # Shouldn't be extremely high
        
        # Pressure should be finite and reasonable
        @test abs(P_avg) < 100.0  # Pressure shouldn't be extremely high
    end
end

@testset "MD barostat compressibility formula" begin
    # Test that barostat uses correct compressibility formula
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=99999,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    st_md = MC.init_md_state(st_mc.pos, T, Random.Xoshiro(99999))
    st_md.L = st_mc.L
    st_md.types = st_mc.types
    
    MC.compute_forces!(st_md, p)
    
    L_before = st_md.L
    P_inst = MC.pressure(st_md, p, T)
    P_target = P_inst + 0.1  # Slightly higher target pressure
    
    # Apply barostat once
    MC.berendsen_barostat!(st_md, p, T, P_target, 0.001, 1.0)
    
    # Box should shrink slightly (P_inst < P_target, so volume should decrease)
    # But change should be small and controlled
    @test st_md.L < L_before  # Should shrink when P_target > P_inst
    @test st_md.L > 0.95 * L_before  # But shouldn't shrink too much in one step
    @test isfinite(st_md.L)
end

@testset "MD barostat guards against invalid states" begin
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=99999,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    st_md = MC.init_md_state(st_mc.pos, T, Random.Xoshiro(99999))
    st_md.L = st_mc.L
    st_md.types = st_mc.types
    
    MC.compute_forces!(st_md, p)
    
    L_before = st_md.L
    
    # Test with invalid temperature (should return early)
    MC.berendsen_barostat!(st_md, p, 0.0, 1.0, 0.001, 1.0)
    @test st_md.L == L_before  # Should not have changed
    
    # Test with NaN pressure (by corrupting state)
    # This is harder to test directly, but the guards should prevent issues
    MC.berendsen_barostat!(st_md, p, T, 1.0, 0.001, 1.0)
    @test st_md.L > 0.0
    @test isfinite(st_md.L)
end
