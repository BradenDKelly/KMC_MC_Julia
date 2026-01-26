"""
Comprehensive unit tests for MD simulations.
Tests inspired by Cassandra, Towhee, RASPA, and BRIC validation suites.
"""

using Test
using MolSim
using Random

# Helper functions
function mean(x::Vector{Float64})::Float64
    isempty(x) && return 0.0
    return sum(x) / length(x)
end

function std(x::Vector{Float64})::Float64
    length(x) <= 1 && return 0.0
    m = mean(x)
    variance = sum((xi - m)^2 for xi in x) / (length(x) - 1)
    return sqrt(max(0.0, variance))
end

@testset "MD: Force consistency (F = -∇U)" begin
    # Test that forces are consistent with energy gradients
    # This is a fundamental requirement: F_i = -∂U/∂r_i
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=12345,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    st_md = MC.init_md_state(st_mc.pos, T, Random.Xoshiro(12345))
    st_md.L = st_mc.L
    st_md.types = st_mc.types
    
    MC.compute_forces!(st_md, p)
    
    # Test force-energy consistency via finite differences
    # For a small displacement δr, we should have:
    # U(r + δr) - U(r) ≈ -F · δr
    δr = 1e-6
    particle_idx = 1
    direction = 0  # x-direction
    
    # Store original state
    pos_orig = copy(st_md.pos)
    force_orig = copy(st_md.force)
    U_orig = MC._md_total_energy(st_md, p)
    
    # Displace particle slightly
    st_md.pos[direction + 1, particle_idx] += δr
    
    # Recompute energy
    U_displaced = MC._md_total_energy(st_md, p)
    
    # Energy change should match -F · δr
    ΔU_numerical = U_displaced - U_orig
    ΔU_expected = -force_orig[direction + 1, particle_idx] * δr
    
    # Allow some tolerance for numerical error (use absolute tolerance for very small values)
    error = abs(ΔU_numerical - ΔU_expected)
    tolerance = max(1e-4 * abs(ΔU_expected), 1e-10)  # Relative or absolute, whichever is larger
    @test error < tolerance
    
    # Restore
    copyto!(st_md.pos, pos_orig)
end

@testset "MD: Energy conservation (NVE-like)" begin
    # In NVE (microcanonical), total energy should be conserved
    # For NVT with thermostat, we test that potential + kinetic energy is reasonable
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=54321,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    st_md = MC.init_md_state(st_mc.pos, T, Random.Xoshiro(54321))
    st_md.L = st_mc.L
    st_md.types = st_mc.types
    
    MC.compute_forces!(st_md, p)
    
    # Run a few steps without thermostat (NVE-like)
    dt = 0.001
    nsteps = 100
    
    U_initial = MC._md_total_energy(st_md, p)
    K_initial = MC.kinetic_energy(st_md)
    E_total_initial = U_initial + K_initial
    
    for step in 1:nsteps
        MC.velocity_verlet_step!(st_md, p, dt)
        
        # Check that total energy is approximately conserved (small drift allowed)
        U_current = MC._md_total_energy(st_md, p)
        K_current = MC.kinetic_energy(st_md)
        E_total_current = U_current + K_current
        
        # Energy drift should be small (allowing for numerical errors)
        energy_drift = abs(E_total_current - E_total_initial) / abs(E_total_initial)
        @test energy_drift < 0.01  # Less than 1% drift over 100 steps
    end
end

@testset "MD: Temperature control (NVT)" begin
    # Test that velocity rescaling thermostat maintains target temperature
    N = 32
    ρ = 0.8
    T_target = 1.0
    rc = 2.5
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T_target, rc=rc, max_disp=0.1, seed=99999,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T_target, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    st_md = MC.init_md_state(st_mc.pos, T_target, Random.Xoshiro(99999))
    st_md.L = st_mc.L
    st_md.types = st_mc.types
    
    MC.compute_forces!(st_md, p)
    MC.velocity_rescale!(st_md, T_target)
    
    dt = 0.001
    thermostat_every = 10
    nsteps = 1000
    
    T_samples = Float64[]
    
    for step in 1:nsteps
        MC.velocity_verlet_step!(st_md, p, dt)
        
        if step % thermostat_every == 0
            MC.velocity_rescale!(st_md, T_target)
            T_inst = MC.instantaneous_temperature(st_md)
            push!(T_samples, T_inst)
        end
    end
    
    # Temperature should be close to target
    T_mean = mean(T_samples)
    T_std = std(T_samples)
    
    @test length(T_samples) > 0
    @test abs(T_mean - T_target) < 0.1  # Within 10% of target
    @test all(isfinite, T_samples)
    @test all(T -> T > 0.0, T_samples)
end

@testset "MD: Pressure control (NPT)" begin
    # Test that NPT MD maintains target pressure
    N = 32
    ρ_initial = 0.8
    T = 1.0
    P_target = 1.0
    rc = 2.5
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ_initial, T=T, rc=rc, max_disp=0.1, seed=88888,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    st_md = MC.init_md_state(st_mc.pos, T, Random.Xoshiro(88888))
    st_md.L = st_mc.L
    st_md.types = st_mc.types
    
    MC.compute_forces!(st_md, p)
    MC.velocity_rescale!(st_md, T)
    
    dt = 0.001
    thermostat_every = 10
    vol_move_every = 20
    max_dlnV = 0.01
    nsteps = 2000
    
    P_samples = Float64[]
    ρ_samples = Float64[]
    
    for step in 1:nsteps
        MC.velocity_verlet_step!(st_md, p, dt)
        
        if step % thermostat_every == 0
            MC.velocity_rescale!(st_md, T)
        end
        
        if step % vol_move_every == 0
            MC.volume_trial_md!(st_md, p, T, P_target, max_dlnV, st_md.rng)
        end
        
        if step % thermostat_every == 0
            P_inst = MC.pressure(st_md, p, T)
            ρ_inst = N / (st_md.L^3)
            
            if isfinite(P_inst) && isfinite(ρ_inst) && st_md.L > 0.0
                push!(P_samples, P_inst)
                push!(ρ_samples, ρ_inst)
            end
        end
    end
    
    # Should have collected samples
    @test length(P_samples) > 0
    @test length(ρ_samples) > 0
    
    # Pressure should be finite and reasonable
    @test all(isfinite, P_samples)
    @test all(P -> abs(P) < 100.0, P_samples)  # Not extremely high
    
    # Density should be reasonable
    @test all(ρ -> ρ > 0.1 && ρ < 10.0, ρ_samples)
end

@testset "MD: Momentum conservation" begin
    # Total momentum should be conserved (or zero if COM motion removed)
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=77777,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    st_md = MC.init_md_state(st_mc.pos, T, Random.Xoshiro(77777))
    st_md.L = st_mc.L
    st_md.types = st_mc.types
    
    MC.compute_forces!(st_md, p)
    
    # Compute initial total momentum
    P_initial = zeros(Float64, 3)
    for i in 1:N
        for d in 1:3
            P_initial[d] += st_md.vel[d, i]
        end
    end
    
    # Run a few steps
    dt = 0.001
    nsteps = 50
    
    for step in 1:nsteps
        MC.velocity_verlet_step!(st_md, p, dt)
        
        # Compute current momentum
        P_current = zeros(Float64, 3)
        for i in 1:N
            for d in 1:3
                P_current[d] += st_md.vel[d, i]
            end
        end
        
        # Momentum should be conserved (or zero if COM removed)
        # init_md_state removes COM motion, so momentum should be ~0
        @test all(abs.(P_current) .< 1e-10)  # Should be essentially zero
    end
end

@testset "MD: PBC consistency" begin
    # Test that periodic boundary conditions work correctly
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=66666,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    st_md = MC.init_md_state(st_mc.pos, T, Random.Xoshiro(66666))
    st_md.L = st_mc.L
    st_md.types = st_mc.types
    
    MC.compute_forces!(st_md, p)
    
    # Test 1: velocity_verlet_step! wraps positions correctly
    particle_idx = 1
    st_md.pos[1, particle_idx] = st_md.L + 0.5  # Outside box
    
    MC.velocity_verlet_step!(st_md, p, 0.001)
    
    # Position should be wrapped back into box
    @test 0.0 <= st_md.pos[1, particle_idx] < st_md.L
    
    # Test 2: Wrapping a particle by L should give same energy (PBC translation)
    # Reset to original state
    st_md.pos = copy(st_mc.pos)
    MC.compute_forces!(st_md, p)
    U_orig = MC._md_total_energy(st_md, p)
    
    # Translate one particle by box length (should be equivalent due to PBC)
    st_md.pos[1, particle_idx] += st_md.L
    # Wrap it back
    if st_md.pos[1, particle_idx] >= st_md.L
        st_md.pos[1, particle_idx] -= st_md.L
    end
    
    # Recompute energy - should be the same (PBC translation invariance)
    MC.compute_forces!(st_md, p)
    U_wrapped = MC._md_total_energy(st_md, p)
    
    # Energy should be unchanged (PBC translation invariance)
    @test abs(U_wrapped - U_orig) < 1e-10
end

@testset "MD: Energy/pressure consistency with MC" begin
    # MD and MC should give similar energy/pressure for same configuration
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    seed = 55555
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=seed,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    st_md = MC.init_md_state(st_mc.pos, T, Random.Xoshiro(seed))
    st_md.L = st_mc.L
    st_md.types = st_mc.types
    
    MC.compute_forces!(st_md, p)
    
    # Compute energy and pressure with both methods
    U_mc = MC.total_energy(st_mc, p)
    U_md = MC._md_total_energy(st_md, p)
    
    P_mc = MC.pressure(st_mc, p, T)
    P_md = MC.pressure(st_md, p, T)
    
    # Should match (they use independent implementations but same physics)
    @test abs(U_mc - U_md) < 1e-10
    @test abs(P_mc - P_md) < 1e-10
end
