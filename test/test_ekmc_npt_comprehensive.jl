"""
Comprehensive unit tests for NPT eKMC simulations.
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

@testset "NPT eKMC: Volume move preserves pair energy consistency" begin
    # After volume move, pair_u matrix should be consistent with positions
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=11111,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    ekst = MC.init_ekmc_state(st_mc, p)
    
    # Store initial energy from pair_u
    U_pair_u_before = MC.total_energy_from_pair_u(ekst, p)
    
    # Perform volume move
    Pext = 1.0
    max_dV = 0.1
    MC.ekmc_volume_move_tan!(ekst, p, Pext, max_dV)
    
    # After volume move, pair_u should be rebuilt and consistent
    U_pair_u_after = MC.total_energy_from_pair_u(ekst, p)
    U_full_after = MC.total_energy(ekst, p)
    
    # pair_u energy should match full energy calculation
    @test abs(U_pair_u_after - U_full_after) < 1e-10
    
    # Energy should be finite
    @test isfinite(U_pair_u_after)
    @test isfinite(U_full_after)
end

@testset "NPT eKMC: Pressure consistency after volume moves" begin
    # Pressure computed from pair_u should be consistent
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=22222,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    ekst = MC.init_ekmc_state(st_mc, p)
    
    # Perform a few volume moves
    Pext = 1.0
    max_dV = 0.1
    
    for _ in 1:5
        # Compute pressure before move
        P_before = MC.pressure_from_pair_u(ekst, p, T)
        
        # Perform volume move
        MC.ekmc_volume_move_tan!(ekst, p, Pext, max_dV)
        
        # Compute pressure after move
        P_after = MC.pressure_from_pair_u(ekst, p, T)
        
        # Pressures should be finite
        @test isfinite(P_before)
        @test isfinite(P_after)
        
        # Box size should be valid
        @test ekst.L > 0.0
        @test isfinite(ekst.L)
        
        # All positions should be in box
        for i in 1:N
            for d in 1:3
                @test 0.0 <= ekst.pos[d, i] < ekst.L
            end
        end
    end
end

@testset "NPT eKMC: Time-weighted average correctness" begin
    # Test that time-weighted averages are computed correctly
    # For a simple case, verify the accumulator math
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=33333,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    ekst = MC.init_ekmc_state(st_mc, p)
    
    # Create accumulator
    acc = MC.TimeWeightedAccumulator()
    
    # Add some test values with known weights
    test_values = [1.0, 2.0, 3.0]
    test_dts = [0.1, 0.2, 0.3]
    
    for (val, dt) in zip(test_values, test_dts)
        MC.push!(acc, val, dt)
    end
    
    # Compute mean
    mean_val = MC.mean(acc)
    expected_mean = sum(val * dt for (val, dt) in zip(test_values, test_dts)) / sum(test_dts)
    
    @test abs(mean_val - expected_mean) < 1e-10
    @test acc.total_time ≈ sum(test_dts)
    @test acc.count == length(test_values)
end

@testset "NPT eKMC: Chemical potential accumulator (Equation A.132)" begin
    # Test NPT chemical potential accumulator
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=44444,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    ekst = MC.init_ekmc_state(st_mc, p)
    
    # Create NPT chemical potential accumulator
    acc = MC.NPTChemicalPotentialAccumulator()
    
    # Add some test values
    R_over_N = 1.0
    V = ekst.L^3
    dt = 0.1
    
    MC.push!(acc, R_over_N, V, dt)
    
    # Check that values are stored correctly
    @test acc.t_total ≈ dt
    @test acc.count == 1
    @test acc.V_avg ≈ V * dt
    
    # Compute chemical potential (should be finite)
    μ = MC.mu_ex_npt(acc, T)
    @test isfinite(μ)
end

@testset "NPT eKMC: Volume move frequency" begin
    # Test that volume moves occur at correct frequency using run_ekmc_npt!
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=55555,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    ekst = MC.init_ekmc_state(st_mc, p)
    acc_mu = MC.ChemicalPotentialAccumulator()
    
    vol_move_every = 10
    nsteps = vol_move_every * N * 2  # Enough to trigger 2 volume moves
    Pext = 1.0
    max_dV = 0.1
    
    # Run short NPT simulation
    obs, total_time, _ = MC.run_ekmc_npt!(ekst, p, acc_mu; nsteps=nsteps, Pext=Pext, max_dV=max_dV,
                                          vol_move_every=vol_move_every, sample_every=1000,
                                          collect_timeseries=false)
    
    # Should have performed 2 volume moves
    @test obs.vol_moves == 2
    @test isfinite(total_time)
end

@testset "NPT eKMC: Phi consistency after volume move" begin
    # After volume move, phi should be consistent with pair_u
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=66666,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    ekst = MC.init_ekmc_state(st_mc, p)
    
    # Perform volume move
    Pext = 1.0
    max_dV = 0.1
    MC.ekmc_volume_move_tan!(ekst, p, Pext, max_dV)
    
    # Check phi consistency: phi[i] should equal sum of pair_u[i, j] for j != i
    for i in 1:N
        phi_from_pair_u = 0.0
        for j in 1:N
            if j != i
                phi_from_pair_u += ekst.pair_u[i, j]
            end
        end
        
        @test abs(ekst.phi[i] - phi_from_pair_u) < 1e-10
    end
end

@testset "NPT eKMC: R (total rate) consistency" begin
    # Total rate R should equal sum of mobilities m
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=77777,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    ekst = MC.init_ekmc_state(st_mc, p)
    
    # Check R = sum(m)
    R_from_m = sum(ekst.m)
    @test abs(ekst.R - R_from_m) < 1e-10
    
    # Perform a volume move and check again
    Pext = 1.0
    max_dV = 0.1
    MC.ekmc_volume_move_tan!(ekst, p, Pext, max_dV)
    
    R_from_m_after = sum(ekst.m)
    @test abs(ekst.R - R_from_m_after) < 1e-10
    
    # R should be positive and finite
    @test ekst.R > 0.0
    @test isfinite(ekst.R)
end

@testset "NPT eKMC: Box size validity" begin
    # Box size should remain valid after volume moves
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=88888,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    ekst = MC.init_ekmc_state(st_mc, p)
    
    L_initial = ekst.L
    
    # Perform multiple volume moves
    Pext = 1.0
    max_dV = 0.1
    
    for _ in 1:10
        MC.ekmc_volume_move_tan!(ekst, p, Pext, max_dV)
        
        # Box size should be valid
        @test ekst.L > 0.0
        @test isfinite(ekst.L)
        
        # Box size should be reasonable (not exploded or collapsed)
        @test ekst.L > 0.1 * L_initial
        @test ekst.L < 10.0 * L_initial
        
        # All positions should be in box
        for i in 1:N
            for d in 1:3
                @test 0.0 <= ekst.pos[d, i] < ekst.L
            end
        end
    end
end

@testset "NPT eKMC: Energy/pressure finite after volume moves" begin
    # Energy and pressure should remain finite after volume moves
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    
    _, st_mc = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=99999,
                               use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    p = MC.LJParams(; σ_types=[1.0], ϵ_types=[1.0], rc=rc, T=T, max_disp=0.1,
                    use_lrc=false, lj_model=:shifted, apply_impulsive_correction=false)
    
    ekst = MC.init_ekmc_state(st_mc, p)
    
    Pext = 1.0
    max_dV = 0.1
    
    for _ in 1:5
        # Compute before move
        U_before = MC.total_energy_from_pair_u(ekst, p)
        P_before = MC.pressure_from_pair_u(ekst, p, T)
        
        @test isfinite(U_before)
        @test isfinite(P_before)
        
        # Perform volume move
        MC.ekmc_volume_move_tan!(ekst, p, Pext, max_dV)
        
        # Compute after move
        U_after = MC.total_energy_from_pair_u(ekst, p)
        P_after = MC.pressure_from_pair_u(ekst, p, T)
        
        @test isfinite(U_after)
        @test isfinite(P_after)
    end
end
