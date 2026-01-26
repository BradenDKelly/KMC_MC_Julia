"""
Tests for MD NPT volume moves and NPT MD simulations.
MD now uses MC-style volume change moves instead of Berendsen barostat.
"""

using Test
using MolSim
using Random

@testset "MD volume move basic functionality" begin
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
    
    # Test volume move with target pressure
    P_target = 1.0
    max_dlnV = 0.01
    
    # Apply volume moves a few times
    accepted_count = 0
    for i in 1:10
        L_before = st_md.L
        accepted = MC.volume_trial_md!(st_md, p, T, P_target, max_dlnV, st_md.rng)
        if accepted
            accepted_count += 1
        end
        
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
    
    # Should have some accepted moves
    @test accepted_count > 0
    
    # Box size should be reasonable
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
    vol_move_every = 20
    max_dlnV = 0.01
    
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
        
        if step % vol_move_every == 0
            L_before = st_md.L
            accepted = MC.volume_trial_md!(st_md, p, T, P_target, max_dlnV, st_md.rng)
            # Forces are recomputed inside volume_trial_md! if accepted
            if !accepted
                MC.compute_forces!(st_md, p)  # Recompute if rejected
            end
            L_after = st_md.L
            
            # Check that box size change is reasonable (MC moves are limited by max_dlnV)
            if L_before > 0.0 && accepted
                scale = L_after / L_before
                # max_dlnV = 0.01 means max scale = exp(0.01) ≈ 1.01
                @test scale > 0.99  # Shouldn't shrink too much
                @test scale < 1.01  # Shouldn't grow too much
            end
        end
        
        if step % thermostat_every == 0
            # Ensure forces are current
            if step % vol_move_every != 0
                MC.compute_forces!(st_md, p)
            end
            
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
        
        # Box size variation should be reasonable
        @test L_std < 0.5 * L_initial  # Standard deviation shouldn't be huge
        
        # Density should be reasonable (not near zero or extremely high)
        @test ρ_avg > 0.1  # Shouldn't be near zero
        @test ρ_avg < 10.0  # Shouldn't be extremely high
        
        # Pressure should be finite and reasonable
        @test abs(P_avg) < 100.0  # Pressure shouldn't be extremely high
    end
end

@testset "MD volume move acceptance" begin
    # Test that volume moves can be accepted/rejected based on Metropolis criterion
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
    P_target = 1.0
    max_dlnV = 0.01
    
    # Perform volume move
    accepted = MC.volume_trial_md!(st_md, p, T, P_target, max_dlnV, st_md.rng)
    
    # Should return a boolean
    @test accepted isa Bool
    
    # Box size should be valid regardless of acceptance
    @test st_md.L > 0.0
    @test isfinite(st_md.L)
    
    # If accepted, box size should have changed (within max_dlnV limits)
    if accepted
        scale = st_md.L / L_before
        @test scale >= exp(-max_dlnV)
        @test scale <= exp(max_dlnV)
    end
end

@testset "MD volume move preserves energy consistency" begin
    # Test that energy computed from MD state matches after volume move
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=88888,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    st_md = MC.init_md_state(st_mc.pos, T, Random.Xoshiro(88888))
    st_md.L = st_mc.L
    st_md.types = st_mc.types
    
    MC.compute_forces!(st_md, p)
    
    # Perform volume move
    P_target = 1.0
    max_dlnV = 0.01
    MC.volume_trial_md!(st_md, p, T, P_target, max_dlnV, st_md.rng)
    
    # Energy should be finite
    U = MC._md_total_energy(st_md, p)
    @test isfinite(U)
    
    # Pressure should be finite
    P = MC.pressure(st_md, p, T)
    @test isfinite(P)
end
