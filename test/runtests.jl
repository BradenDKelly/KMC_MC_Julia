"""
Tests for MolSim package.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Add src directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using Test

@testset "PBC tests" begin
    # Test wrap keeps coordinates in [0, L)
    L = 10.0
    positions = [-2.5  5.0  12.3;
                 3.5  -1.2  8.7;
                 0.0   9.9  10.0]
    
    wrap!(positions, L)
    
    @test all(0.0 .<= positions .< L)
    @test positions[1, 1] ≈ 7.5  # -2.5 + 10.0
    @test positions[1, 3] ≈ 2.3  # 12.3 - 10.0
    @test positions[2, 2] ≈ 8.8  # -1.2 + 10.0
end

@testset "Energy invariance tests" begin
    # Setup
    N = 32
    σ = 1.0
    ϵ = 1.0
    ρ = 0.8
    T = 1.2
    rc = 2.5 * σ
    max_disp = 0.1 * σ
    
    V = N / ρ
    L = cbrt(V)
    
    params = LJParams(ϵ, σ, rc, L, N, T, max_disp)
    
    # Create initial state
    state1 = init_lj_state(params; seed=42)
    E1 = state1.energy
    
    # Shift all particles by a constant vector (should not change energy with PBC)
    shift = [2.3, 1.7, 0.5]
    posx2 = copy(state1.posx)
    posy2 = copy(state1.posy)
    posz2 = copy(state1.posz)
    @inbounds for i in 1:N
        posx2[i] += shift[1]
        posy2[i] += shift[2]
        posz2[i] += shift[3]
    end
    
    # Wrap positions
    wrap_soa!(posx2, posy2, posz2, L)
    
    # Create new state with shifted positions
    state2 = init_lj_state(params; posx=posx2, posy=posy2, posz=posz2, seed=42)
    E2 = state2.energy
    
    # Energies should be the same (within numerical precision)
    @test abs(E1 - E2) < 1e-10
end

@testset "MC step tests" begin
    # Setup
    N = 32
    σ = 1.0
    ϵ = 1.0
    ρ = 0.8
    T = 1.2
    rc = 2.5 * σ
    max_disp = 0.0  # Zero displacement should not change energy
    
    V = N / ρ
    L = cbrt(V)
    
    params = LJParams(ϵ, σ, rc, L, N, T, max_disp)
    state = init_lj_state(params; seed=42)
    
    # Store initial energy
    E0 = state.energy
    
    # Perform one MC step with zero displacement
    accepted, ΔE = mc_step!(state, params)
    
    # Energy should not change
    @test abs(state.energy - E0) < 1e-12
    @test abs(ΔE) < 1e-12
    
    # Step should always be accepted (ΔE = 0)
    @test accepted
end

@testset "LJ potential tests" begin
    σ = 1.0
    ϵ = 1.0
    rc = 2.5 * σ
    L = 10.0
    N = 2
    T = 1.2
    max_disp = 0.1
    
    params = LJParams(ϵ, σ, rc, L, N, T, max_disp)
    
    # Test potential at minimum (r = 2^(1/6) * σ)
    r_min = 2.0^(1.0/6.0) * σ
    r_min_sq = r_min * r_min
    U_min = lj_potential(r_min_sq, params)
    
    # Should be -ϵ at minimum
    @test abs(U_min - (-ϵ)) < 1e-10
    
    # Test potential at cutoff
    rc_sq = rc * rc
    U_rc = lj_potential(rc_sq, params)
    @test U_rc ≈ 0.0
    
    # Test potential beyond cutoff
    r_beyond_sq = (rc + 0.1)^2
    U_beyond = lj_potential(r_beyond_sq, params)
    @test U_beyond == 0.0
end
