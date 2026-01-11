using Test
using MolSim
using Random
using StaticArrays

@testset "PBC translation + wrap invariance" begin
    # Test parameters
    N = 64
    ρ = 0.5
    T = 1.0
    rc = 2.5
    seed = 11111
    
    for lj_model in [:truncated, :shifted]
        @testset "LJ model: $lj_model" begin
            # Initialize system
            p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1,
                                       seed=seed, use_lrc=false, lj_model=lj_model)
            
            # Compute initial energy and pressure
            U0 = MolSim.MC.total_energy(st, p)
            P0 = MolSim.MC.pressure(st, p, T)
            
            # Apply uniform translation that pushes particles across boundaries
            # Use ~0.6L to ensure many cross boundaries
            shift = [0.6 * st.L, 0.6 * st.L, 0.6 * st.L]
            
            # Translate all positions
            pos_translated = copy(st.pos)
            @inbounds for i in 1:st.N
                pos_translated[1, i] += shift[1]
                pos_translated[2, i] += shift[2]
                pos_translated[3, i] += shift[3]
            end
            
            # Wrap all positions into [0, L)
            scratch = MVector{3,Float64}(0.0, 0.0, 0.0)
            @inbounds for i in 1:st.N
                scratch[1] = pos_translated[1, i]
                scratch[2] = pos_translated[2, i]
                scratch[3] = pos_translated[3, i]
                MolSim.MC.wrap!(scratch, st.L)
                pos_translated[1, i] = scratch[1]
                pos_translated[2, i] = scratch[2]
                pos_translated[3, i] = scratch[3]
            end
            
            # Create state with translated and wrapped positions
            st_translated = MolSim.MC.LJState(st.N, st.L, pos_translated, st.types, st.rng, st.cl, st.scratch_dr, 0, 0)
            MolSim.MC.rebuild_cells!(st_translated)
            
            # Recompute energy and pressure
            U1 = MolSim.MC.total_energy(st_translated, p)
            P1 = MolSim.MC.pressure(st_translated, p, T)
            
            # Check invariance
            @test abs(U0 - U1) <= 1e-10 "Energy not invariant under translation+wrap (U0=$U0, U1=$U1)"
            @test abs(P0 - P1) <= 1e-10 "Pressure not invariant under translation+wrap (P0=$P0, P1=$P1)"
        end
    end
end
