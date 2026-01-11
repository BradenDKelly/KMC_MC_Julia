using Test
using MolSim
using Random
using StaticArrays

@testset "Tail-correction toggle consistency" begin
    # Test parameters
    N = 64
    ρ = 0.5
    T = 1.0
    rc = 2.5
    seed = 22222
    
    for lj_model in [:truncated, :shifted]
        @testset "LJ model: $lj_model" begin
            # Initialize system WITHOUT tail corrections
            p_no_lrc, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1,
                                               seed=seed, use_lrc=false, lj_model=lj_model)
            
            # Compute pair energy (no tail)
            U_pair = MolSim.MC.total_energy(st, p_no_lrc)
            
            # Compute pair pressure (no tail, no impulsive correction)
            p_no_lrc_no_imp = MolSim.MC.LJParams(
                p_no_lrc.σ, p_no_lrc.ϵ, p_no_lrc.rc, p_no_lrc.rc2, p_no_lrc.β,
                p_no_lrc.max_disp, false,  # use_lrc = false
                0.0, 0.0,  # lrc_u_per_particle, lrc_p
                p_no_lrc.lj_model, false,  # apply_impulsive_correction = false
                p_no_lrc.u_rc,
                p_no_lrc.n_types, p_no_lrc.σ_types, p_no_lrc.ϵ_types,
                p_no_lrc.σ_mix, p_no_lrc.ϵ_mix
            )
            P_pair = MolSim.MC.pressure(st, p_no_lrc_no_imp, T)
            
            # Initialize system WITH tail corrections
            p_with_lrc, st2 = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1,
                                                  seed=seed, use_lrc=true, lj_model=lj_model)
            
            # Compute total energy (with tail)
            U_total = MolSim.MC.total_energy(st2, p_with_lrc)
            
            # Compute total pressure (with tail, no impulsive correction)
            p_with_lrc_no_imp = MolSim.MC.LJParams(
                p_with_lrc.σ, p_with_lrc.ϵ, p_with_lrc.rc, p_with_lrc.rc2, p_with_lrc.β,
                p_with_lrc.max_disp, true,  # use_lrc = true
                p_with_lrc.lrc_u_per_particle, p_with_lrc.lrc_p,
                p_with_lrc.lj_model, false,  # apply_impulsive_correction = false
                p_with_lrc.u_rc,
                p_with_lrc.n_types, p_with_lrc.σ_types, p_with_lrc.ϵ_types,
                p_with_lrc.σ_mix, p_with_lrc.ϵ_mix
            )
            P_total = MolSim.MC.pressure(st2, p_with_lrc_no_imp, T)
            
            # Compute expected tail corrections using the same functions as production
            U_tail_analytic = N * MolSim.MC.compute_lrc_energy_per_particle(ρ, rc)
            P_tail_analytic = MolSim.MC.compute_lrc_pressure(ρ, rc)
            
            # Check energy consistency
            U_diff = U_total - U_pair
            @test abs(U_diff - U_tail_analytic) <= 1e-10 "Energy tail correction mismatch (diff=$U_diff, expected=$U_tail_analytic)"
            
            # Check pressure consistency
            P_diff = P_total - P_pair
            @test abs(P_diff - P_tail_analytic) <= 1e-10 "Pressure tail correction mismatch (diff=$P_diff, expected=$P_tail_analytic)"
            
            # Also verify that p_with_lrc has the correct precomputed values
            @test abs(p_with_lrc.lrc_u_per_particle - MolSim.MC.compute_lrc_energy_per_particle(ρ, rc)) <= 1e-12
            @test abs(p_with_lrc.lrc_p - MolSim.MC.compute_lrc_pressure(ρ, rc)) <= 1e-12
        end
    end
end
