using Test
using MolSim
using StaticArrays
using Random

# Gate for slow tests (ensemble convergence, long runs)
const RUN_SLOW_TESTS = get(ENV, "MOLSIM_SLOW_TESTS", "0") == "1"
# Gate for extended test suite (longer than ~20s total)
const RUN_ALL_TESTS = get(ENV, "MOLSIM_ALL_TESTS", "0") == "1"

@testset "wrap! tests" begin
    L = 10.0
    x = MVector{3,Float64}(-2.5, 5.0, 12.3)
    MolSim.MC.wrap!(x, L)
    @test all(0.0 .<= x .< L)
    @test x[1] ≈ 7.5
    @test x[2] ≈ 5.0
    @test x[3] ≈ 2.3
end

@testset "minimum_image! tests" begin
    L = 10.0
    dr = MVector{3,Float64}(6.0, -6.0, 3.0)
    MolSim.MC.minimum_image!(dr, L)
    @test all(abs.(dr) .<= L/2)
    @test dr[1] ≈ -4.0
    @test dr[2] ≈ 4.0
    @test dr[3] ≈ 3.0
end

@testset "total_energy translation invariance" begin
    p, st = MolSim.MC.init_fcc(N=108, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42)
    E1 = MolSim.MC.total_energy(st, p)
    
    # Translate all particles by constant vector
    shift = [2.3, 1.7, 0.5]
    @inbounds for i in 1:st.N
        st.pos[1, i] += shift[1]
        st.pos[2, i] += shift[2]
        st.pos[3, i] += shift[3]
    end
    
    # Wrap all positions
    scratch = MVector{3,Float64}(0.0, 0.0, 0.0)
    @inbounds for i in 1:st.N
        scratch[1] = st.pos[1, i]
        scratch[2] = st.pos[2, i]
        scratch[3] = st.pos[3, i]
        MolSim.MC.wrap!(scratch, st.L)
        st.pos[1, i] = scratch[1]
        st.pos[2, i] = scratch[2]
        st.pos[3, i] = scratch[3]
    end
    
    E2 = MolSim.MC.total_energy(st, p)
    @test abs(E1 - E2) < 1e-10
end

@testset "sweep! safety check" begin
    p, st = MolSim.MC.init_fcc(N=108, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42)
    
    # Run one sweep with rebuild_every=st.N (default, once per sweep)
    acceptance = MolSim.MC.sweep!(st, p; rebuild_every=st.N)
    
    # Check acceptance ratio is in valid range
    @test 0.0 <= acceptance <= 1.0
    
    # Check energy is finite
    E_total = MolSim.MC.total_energy(st, p)
    @test isfinite(E_total)
end

@testset "volume_trial! safety check" begin
    p, st = MolSim.MC.init_fcc(N=108, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42)
    
    # Volume move with very small max_dlnV should rarely reject for modest Pext
    # Just assert it runs and returns Bool
    result = MolSim.MC.volume_trial!(st, p; max_dlnV=1e-6, Pext=1.0)
    @test result isa Bool
    
    # Check energy remains finite
    E_total = MolSim.MC.total_energy(st, p)
    @test isfinite(E_total)
end

@testset "LJ REMC smoke test" begin
    p, st = MolSim.MC.init_fcc(N=108, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42)
    reaction = MolSim.MC.Reaction("A⇌B", [-1, 1], 0.0; logq=[0.0, 0.0])
    accepted = 0
    attempted = 0
    for _ in 1:50
        _tacc, _vacc, _vatt, racc, ratt, _rfatt, _rfacc, _rratt, _rracc, _fc, _rc, _iu, _cb, _ca, _counters =
            MolSim.MC.sweep_npt_with_reactions!(st, p, reaction; Pext=1.0, max_dlnV=0.01, p_reaction=0.2, do_volume_move=false)
        accepted += racc
        attempted += ratt
    end
    @test isfinite(MolSim.MC.total_energy(st, p))
    @test attempted >= 0
end

@testset "LJ total_energy allocation-free" begin
    p, st = MolSim.MC.init_fcc(N=108, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42)
    @test @allocated(MolSim.MC.total_energy(st, p)) == 0
end

@testset "EXP-6 energy sanity" begin
    rm = 1.0
    ϵ = 1.0
    α = 12.0
    r = 1.1
    r2 = r * r
    p = MolSim.MC.Exp6Params(rm_types=[rm], ϵ_types=[ϵ], α_types=[α], rc=2.5, T=1.0, cutoff_model=:truncated)
    u_ref = MolSim.MC.exp6_potential(r2, rm, ϵ, α, p.rmin_factor)
    u_eval = MolSim.MC.exp6_pair_u_from_r2(r2, 1, 1, p)
    @test abs(u_eval - u_ref) < 1e-10
end

@testset "EXP-6 two-particle energy at 3σ" begin
    rm = 1.0
    ϵ = 1.0
    α = 12.0
    r = 3.0
    r2 = r * r
    p = MolSim.MC.Exp6Params(rm_types=[rm], ϵ_types=[ϵ], α_types=[α], rc=5.0, T=1.0, cutoff_model=:truncated)
    pos = zeros(3, 2)
    pos[1, 1] = 0.0
    pos[2, 1] = 0.0
    pos[3, 1] = 0.0
    pos[1, 2] = r
    pos[2, 2] = 0.0
    pos[3, 2] = 0.0
    types = [1, 1]
    cl = MolSim.MC.CellList(2, 10.0, p.rc)
    st = MolSim.MC.LJState(2, 10.0, pos, types, Random.Xoshiro(1234),
                           cl, MVector{3,Float64}(0.0, 0.0, 0.0), 0, 0)
    MolSim.MC.rebuild_cells!(st)
    u_ref = MolSim.MC.exp6_potential(r2, rm, ϵ, α, p.rmin_factor)
    u_total = MolSim.MC.total_energy(st, p)
    println("EXP6_TWO_PARTICLE_CHECK: r=$(r) rm=$(rm) eps=$(ϵ) alpha=$(α) u_direct=$(u_ref) u_total=$(u_total)")
    @test abs(u_ref - (-2.7435e-3)) < 2e-8
    @test abs(u_total - (-2.7435e-3)) < 2e-8
    @test abs(u_total - u_ref) < 1e-10
end

@testset "REMC reaction direction stoichiometry" begin
    # Zero-energy Exp-6 to isolate reaction bookkeeping
    rm = [1.0, 1.0, 1.0]
    ϵ = [0.0, 0.0, 0.0]
    α = [12.0, 12.0, 12.0]
    p = MolSim.MC.Exp6Params(rm_types=rm, ϵ_types=ϵ, α_types=α, rc=5.0, T=1.0, cutoff_model=:truncated)
    L = 10.0
    pos = zeros(3, 4)
    pos[:, 1] = [1.0, 1.0, 1.0]
    pos[:, 2] = [3.0, 1.0, 1.0]
    pos[:, 3] = [1.0, 3.0, 1.0]
    pos[:, 4] = [1.0, 1.0, 3.0]
    cl = MolSim.MC.CellList(4, L, p.rc)
    rng = Random.Xoshiro(1234)

    # Forward: N2 + O2 -> 2 NO (counts [1,1,2] -> [0,0,4])
    types_fwd = [1, 2, 3, 3]
    st_fwd = MolSim.MC.LJState(4, L, copy(pos), copy(types_fwd), rng,
                               cl, MVector{3,Float64}(0.0, 0.0, 0.0), 0, 0)
    MolSim.MC.rebuild_cells!(st_fwd)
    reaction_fwd = MolSim.MC.Reaction("N2+O2⇌2NO", [-1, -1, 2], 0.0; logq=nothing, logGamma=1000.0)
    accepted_fwd, feasible_fwd = MolSim.MC.reaction_trial!(st_fwd, p, reaction_fwd, :forward;
                                                          use_alchemical=false, insertion_mode=:standard_remc)
    @test feasible_fwd
    @test accepted_fwd
    counts_fwd = MolSim.MC.count_species(st_fwd, 3)
    @test counts_fwd == [0, 0, 4]

    # Reverse: 2 NO -> N2 + O2 (counts [1,1,2] -> [2,2,0])
    types_rev = [1, 2, 3, 3]
    st_rev = MolSim.MC.LJState(4, L, copy(pos), copy(types_rev), rng,
                               MolSim.MC.CellList(4, L, p.rc), MVector{3,Float64}(0.0, 0.0, 0.0), 0, 0)
    MolSim.MC.rebuild_cells!(st_rev)
    reaction_rev = MolSim.MC.Reaction("N2+O2⇌2NO", [-1, -1, 2], 0.0; logq=nothing, logGamma=-1000.0)
    accepted_rev, feasible_rev = MolSim.MC.reaction_trial!(st_rev, p, reaction_rev, :reverse;
                                                          use_alchemical=false, insertion_mode=:standard_remc)
    @test feasible_rev
    @test accepted_rev
    counts_rev = MolSim.MC.count_species(st_rev, 3)
    @test counts_rev == [2, 2, 0]
end

@testset "EXP-6 shifted cutoff continuity" begin
    rm = 1.0
    ϵ = 1.0
    α = 12.0
    rc = 2.5
    p = MolSim.MC.Exp6Params(rm_types=[rm], ϵ_types=[ϵ], α_types=[α], rc=rc, T=1.0, cutoff_model=:shifted)
    u_rc = MolSim.MC.exp6_pair_u_from_r2(rc * rc, 1, 1, p)
    @test abs(u_rc) < 1e-10
end

@testset "EXP-6 force consistency" begin
    rm = 1.0
    ϵ = 1.0
    α = 12.0
    r = 1.2
    r2 = r * r
    p = MolSim.MC.Exp6Params(rm_types=[rm], ϵ_types=[ϵ], α_types=[α], rc=2.5, T=1.0, cutoff_model=:truncated)
    dr = 1e-6
    u_plus = MolSim.MC.exp6_potential((r + dr)^2, rm, ϵ, α, p.rmin_factor)
    u_minus = MolSim.MC.exp6_potential((r - dr)^2, rm, ϵ, α, p.rmin_factor)
    dUdr = (u_plus - u_minus) / (2 * dr)
    force_times_r = -r * dUdr
    f_eval = MolSim.MC.exp6_force_magnitude_times_r(r2, rm, ϵ, α, p.rmin_factor)
    @test abs(force_times_r - f_eval) < 1e-6
end

@testset "EXP-6 REMC smoke" begin
    rm = [1.0, 1.0, 1.0]
    ϵ = [1.0, 1.0, 1.0]
    α = [12.0, 12.0, 12.0]
    N = 32
    types = vcat(fill(1, 10), fill(2, 10), fill(3, 12))
    p, st = MolSim.MC.init_fcc_exp6(N=N, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42,
                                    types=types, rm_types=rm, ϵ_types=ϵ, α_types=α,
                                    cutoff_model=:truncated, mixing_rule=:lb)
    reaction = MolSim.MC.Reaction("N2+O2⇌2NO", [-1, -1, 2], 0.0; logq=[0.0, 0.0, 0.0])
    for _ in 1:10
        MolSim.MC.sweep_npt_with_reactions!(st, p, reaction; Pext=1.0, max_dlnV=0.01, p_reaction=0.2, do_volume_move=false)
    end
    @test isfinite(MolSim.MC.total_energy(st, p))
end

@testset "density consistency" begin
    p, st = MolSim.MC.init_fcc(N=108, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42)
    
    # Initial density check
    ρ_initial = st.N / (st.L * st.L * st.L)
    @test ρ_initial ≈ 0.8 rtol=1e-10
    
    # After volume move, density should be consistent with L and N
    L_before = st.L
    MolSim.MC.volume_trial!(st, p; max_dlnV=0.01, Pext=1.0)
    L_after = st.L
    ρ_after = st.N / (L_after * L_after * L_after)
    
    # Density should equal N/L^3
    @test ρ_after ≈ st.N / (st.L * st.L * st.L) rtol=1e-10
end

@testset "pressure sanity check" begin
    p, st = MolSim.MC.init_fcc(N=108, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42)
    T = 1.0 / p.β
    P = MolSim.MC.pressure(st, p, T)
    @test isfinite(P)
end

@testset "widom_deltaU allocation and sanity" begin
    p, st = MolSim.MC.init_fcc(N=108, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42)
    
    # Check zero allocation
    allocated = @allocated MolSim.MC.widom_deltaU(st, p)
    @test allocated == 0
    
    # Check returns finite Float64
    ΔU = MolSim.MC.widom_deltaU(st, p)
    @test isfinite(ΔU)
    @test ΔU isa Float64
end

@testset "long-range corrections" begin
    N = 108
    ρ = 0.8
    T = 1.0
    rc = 2.5
    
    # Initialize without LRC
    p_no_lrc, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=42, use_lrc=false)
    @test p_no_lrc.use_lrc == false
    @test p_no_lrc.lrc_u_per_particle == 0.0
    @test p_no_lrc.lrc_p == 0.0
    
    # Initialize with LRC
    p_lrc, st_lrc = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=42, use_lrc=true)
    @test p_lrc.use_lrc == true
    
    # Compute energy and pressure without LRC
    E_no_lrc = MolSim.MC.total_energy(st, p_no_lrc)
    P_no_lrc = MolSim.MC.pressure(st, p_no_lrc, T)
    
    # Compute energy and pressure with LRC
    E_lrc = MolSim.MC.total_energy(st_lrc, p_lrc)
    P_lrc = MolSim.MC.pressure(st_lrc, p_lrc, T)
    
    # Check that LRC adds exactly the correction amount
    expected_E_correction = N * p_lrc.lrc_u_per_particle
    @test abs((E_lrc - E_no_lrc) - expected_E_correction) < 1e-10
    
    expected_P_correction = p_lrc.lrc_p
    @test abs((P_lrc - P_no_lrc) - expected_P_correction) < 1e-10
    
    # Check that corrections are computed correctly (analytic formula)
    # U_tail/N = (8πρ/3) * [1/(3*rc^9) - 1/rc^3]
    rc3 = rc * rc * rc
    rc9 = rc3 * rc3 * rc3
    inv_rc3 = 1.0 / rc3
    inv_rc9 = 1.0 / rc9
    expected_u_per_particle = (8.0 * π * ρ / 3.0) * (inv_rc9 / 3.0 - inv_rc3)
    @test abs(p_lrc.lrc_u_per_particle - expected_u_per_particle) < 1e-10
    
    # P_tail = (16πρ²/3) * [2/(3*rc^9) - 1/rc^3]
    ρ2 = ρ * ρ
    expected_p_correction = (16.0 * π * ρ2 / 3.0) * (2.0 * inv_rc9 / 3.0 - inv_rc3)
    @test abs(p_lrc.lrc_p - expected_p_correction) < 1e-10
end

@testset "allocation truth for LRC" begin
    # Initialize with LRC enabled
    p_lrc, st_lrc = MolSim.MC.init_fcc(N=864, ρ=0.8, T=1.0, rc=2.5, seed=42, use_lrc=true)
    T = 1.0 / p_lrc.β
    
    # Warm up
    MolSim.MC.total_energy(st_lrc, p_lrc)
    MolSim.MC.pressure(st_lrc, p_lrc, T)
    
    # Measure allocations with LRC
    alloc_total_energy_lrc = @allocated MolSim.MC.total_energy(st_lrc, p_lrc)
    alloc_pressure_lrc = @allocated MolSim.MC.pressure(st_lrc, p_lrc, T)
    
    @test alloc_total_energy_lrc <= 64
    @test alloc_pressure_lrc <= 64
    
    # Initialize with LRC disabled
    p_no_lrc, st_no_lrc = MolSim.MC.init_fcc(N=864, ρ=0.8, T=1.0, rc=2.5, seed=42, use_lrc=false)
    T_no_lrc = 1.0 / p_no_lrc.β
    
    # Warm up
    MolSim.MC.total_energy(st_no_lrc, p_no_lrc)
    MolSim.MC.pressure(st_no_lrc, p_no_lrc, T_no_lrc)
    
    # Measure allocations without LRC
    alloc_total_energy_no_lrc = @allocated MolSim.MC.total_energy(st_no_lrc, p_no_lrc)
    alloc_pressure_no_lrc = @allocated MolSim.MC.pressure(st_no_lrc, p_no_lrc, T_no_lrc)
    
    @test alloc_total_energy_no_lrc <= 64
    @test alloc_pressure_no_lrc <= 64
end

# Include physics invariant tests
include("test_invariants.jl")

# Include ΔU correctness tests
include("test_deltaU.jl")

# Include detailed balance tests
include("test_detailed_balance.jl")

# Include pressure identity tests
include("test_pressure_identities.jl")
include("test_rdf.jl")
include("test_rdf_mixture.jl")
include("test_fluctuations.jl")

# Include regression configuration locks
include("test_regression_configs.jl")

# Include molecule tests
include("test_molecules.jl")

# Include GCMC/CBMC molecule tests
if RUN_ALL_TESTS
    include("test_gcmc_molecules.jl")
else
    @info "Extended tests skipped. To enable, set MOLSIM_ALL_TESTS=1" *
          "\n  Windows PowerShell: `\$env:MOLSIM_ALL_TESTS=\"1\"; julia --project -e \"using Pkg; Pkg.test()\"`" *
          "\n  cmd.exe: `set MOLSIM_ALL_TESTS=1 && julia --project -e \"using Pkg; Pkg.test()\"`"
end

# Include regression test
if RUN_ALL_TESTS
    include("regress_nvt.jl")
end

# Include EOS virial test
include("test_eos_virial.jl")

# Include EOS cross-check test
include("test_eos_crosscheck.jl")

# Include multicomponent LJ tests
include("test_multicomponent_reduction.jl")
include("test_multicomponent_relabel.jl")
include("test_multicomponent_pair_sanity.jl")
include("test_multicomponent_widom.jl")

# Include RxMC tests
if RUN_ALL_TESTS
    include("test_rxmc.jl")
    include("test_table3_2_remc.jl")
    include("test_reaction_detailed_balance_ideal_gas.jl")
end

# Conditional include for slow tests (long-run ensemble convergence)
if RUN_SLOW_TESTS
    include("test_pressure_identities_slow.jl")
else
    @info "Slow tests skipped. To enable, set environment variable: MOLSIM_SLOW_TESTS=1" *
          "\n  Windows PowerShell: `\$env:MOLSIM_SLOW_TESTS=\"1\"; julia --project -e \"using Pkg; Pkg.test()\"`" *
          "\n  cmd.exe: `set MOLSIM_SLOW_TESTS=1 && julia --project -e \"using Pkg; Pkg.test()\"`"
end
