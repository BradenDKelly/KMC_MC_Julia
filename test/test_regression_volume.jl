using Test
using MolSim
using Random
using StaticArrays

# Import brute-force energy function from test_regression_deltaU
include("test_regression_deltaU.jl")

@testset "NPT volume-move acceptance oracle vs brute-force" begin
    # Test parameters
    N = 64
    ρ_init = 0.5
    T = 1.0
    rc = 2.5
    max_dlnV = 0.01
    Pext = 1.0
    seed = 98765
    K = 100  # Number of proposed volume moves
    
    for lj_model in [:truncated, :shifted]
        @testset "LJ model: $lj_model" begin
            # Initialize system
            p, st = MolSim.MC.init_fcc(N=N, ρ=ρ_init, T=T, rc=rc, max_disp=0.1,
                                       seed=seed, use_lrc=false, lj_model=lj_model)
            
            # Pre-generate volume proposals deterministically
            rng_seq = Random.Xoshiro(seed)
            volume_proposals = Float64[]
            for k in 1:K
                dlnV = (rand(rng_seq) - 0.5) * 2.0 * max_dlnV
                push!(volume_proposals, dlnV)
            end
            
            V_old = st.L * st.L * st.L
            L_old = st.L
            
            for (move_idx, dlnV) in enumerate(volume_proposals)
                # Compute new volume
                V_new = V_old * exp(dlnV)
                L_new = cbrt(V_new)
                scale = L_new / L_old
                
                # Store old state
                pos_old = copy(st.pos)
                
                # Fast-path: use production volume_trial! logic
                # We'll simulate it step by step to extract ΔU_fast
                
                # Scale all positions
                pos_scaled = copy(st.pos)
                @inbounds for i in 1:st.N
                    pos_scaled[1, i] *= scale
                    pos_scaled[2, i] *= scale
                    pos_scaled[3, i] *= scale
                end
                
                # Wrap scaled positions (as volume_trial! does)
                scratch = MVector{3,Float64}(0.0, 0.0, 0.0)
                @inbounds for i in 1:st.N
                    scratch[1] = pos_scaled[1, i]
                    scratch[2] = pos_scaled[2, i]
                    scratch[3] = pos_scaled[3, i]
                    MolSim.MC.wrap!(scratch, L_new)
                    pos_scaled[1, i] = scratch[1]
                    pos_scaled[2, i] = scratch[2]
                    pos_scaled[3, i] = scratch[3]
                end
                
                # Create temporary state with scaled positions
                st_temp = MolSim.MC.LJState(st.N, L_new, pos_scaled, st.types, st.rng, st.cl, st.scratch_dr, 0, 0)
                MolSim.MC.rebuild_cells!(st_temp)
                
                # Compute energies
                U_old_fast = MolSim.MC.total_energy(st, p)
                U_new_fast = MolSim.MC.total_energy(st_temp, p)
                ΔU_fast = U_new_fast - U_old_fast
                
                # Compute Metropolis argument (fast path)
                ΔV = V_new - V_old
                log_V_ratio = log(V_new / V_old)
                Δ_fast = p.β * (ΔU_fast + Pext * ΔV) - N * log_V_ratio
                
                # Brute-force: recompute energy before and after scaling
                U_old_brute = brute_force_energy_test(pos_old, L_old, p, false)
                U_new_brute = brute_force_energy_test(pos_scaled, L_new, p, false)
                ΔU_brute = U_new_brute - U_old_brute
                
                # Compute Metropolis argument (brute-force)
                Δ_brute = p.β * (ΔU_brute + Pext * ΔV) - N * log_V_ratio
                
                # Check agreement
                @test abs(ΔU_fast - ΔU_brute) <= 1e-10 "Move $move_idx: ΔU mismatch (fast=$ΔU_fast, brute=$ΔU_brute)"
                @test abs(Δ_fast - Δ_brute) <= 1e-10 "Move $move_idx: Metropolis argument mismatch (fast=$Δ_fast, brute=$Δ_brute)"
                
                # Check accept/reject decision matches
                u_test = rand(Random.Xoshiro(seed + move_idx))  # Deterministic test value
                accept_fast = (Δ_fast <= 0.0) || (u_test < exp(-Δ_fast))
                accept_brute = (Δ_brute <= 0.0) || (u_test < exp(-Δ_brute))
                @test accept_fast == accept_brute "Move $move_idx: Accept/reject mismatch"
            end
        end
    end
end
