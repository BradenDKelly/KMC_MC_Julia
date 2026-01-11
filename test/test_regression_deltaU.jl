using Test
using MolSim
using Random
using StaticArrays

"""
    brute_force_energy_test(pos, L, p, use_lrc)

Compute total energy using O(N²) brute-force loop for testing.
Supports both truncated and shifted LJ models.
"""
function brute_force_energy_test(pos::Matrix{Float64}, L::Float64, p::MolSim.MC.LJParams, use_lrc::Bool)::Float64
    N = size(pos, 2)
    rc2 = p.rc2
    L_half = L / 2.0
    
    energy = 0.0
    
    @inbounds for i in 1:N
        for j in (i+1):N
            # Compute distance vector
            dr_x = pos[1, j] - pos[1, i]
            dr_y = pos[2, j] - pos[2, i]
            dr_z = pos[3, j] - pos[3, i]
            
            # Apply minimum image convention
            if dr_x > L_half
                dr_x -= L
            elseif dr_x < -L_half
                dr_x += L
            end
            if dr_y > L_half
                dr_y -= L
            elseif dr_y < -L_half
                dr_y += L
            end
            if dr_z > L_half
                dr_z -= L
            elseif dr_z < -L_half
                dr_z += L
            end
            
            # Squared distance
            r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
            
            if r2 < rc2 && r2 > 0.0
                # Use type information if multicomponent
                type_i = p.n_types > 1 ? 1 : 1  # For single-component tests, type is always 1
                type_j = p.n_types > 1 ? 1 : 1
                u_val = MolSim.MC.lj_pair_u_from_r2_mixed(r2, type_i, type_j, p)
                energy += u_val
            end
        end
    end
    
    # Add tail correction if enabled
    if use_lrc && p.use_lrc
        energy += N * p.lrc_u_per_particle
    end
    
    return energy
end

@testset "NVT ΔU oracle vs brute-force (displacement moves)" begin
    # Test parameters
    N = 64  # Small system for fast tests
    ρ = 0.5
    T = 1.0
    rc = 2.5
    max_disp = 0.1
    seed = 12345
    K = 200  # Number of attempted moves
    
    # Test both truncated and shifted LJ
    for lj_model in [:truncated, :shifted]
        @testset "LJ model: $lj_model" begin
            # Initialize system deterministically
            p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp, 
                                       seed=seed, use_lrc=false, lj_model=lj_model)
            
            # Store initial state for brute-force recompute
            pos_current = copy(st.pos)
            
            # Pre-generate move sequence deterministically
            rng_test = Random.Xoshiro(seed)
            moves = Vector{Tuple{Int, Float64, Float64, Float64, Float64}}()
            for k in 1:K
                i = rand(rng_test, 1:N)
                dx = (rand(rng_test) - 0.5) * 2.0 * max_disp
                dy = (rand(rng_test) - 0.5) * 2.0 * max_disp
                dz = (rand(rng_test) - 0.5) * 2.0 * max_disp
                u_accept = rand(rng_test)  # Uniform for acceptance test
                push!(moves, (i, dx, dy, dz, u_accept))
            end
            
            # Run moves and check ΔU at each step
            for (move_idx, (i, dx, dy, dz, u_accept)) in enumerate(moves)
                # Store old position
                x_old = pos_current[1, i]
                y_old = pos_current[2, i]
                z_old = pos_current[3, i]
                
                # Compute fast-path ΔU using production code
                # First compute old local energy (sum of interactions between i and all others)
                Eold_fast = MolSim.MC.local_energy(i, st, p)
                
                # Apply trial move to state
                st.pos[1, i] = x_old + dx
                st.pos[2, i] = y_old + dy
                st.pos[3, i] = z_old + dz
                
                # Wrap position (as mc_trial! does)
                dr = st.scratch_dr
                dr[1] = st.pos[1, i]
                dr[2] = st.pos[2, i]
                dr[3] = st.pos[3, i]
                MolSim.MC.wrap!(dr, st.L)
                st.pos[1, i] = dr[1]
                st.pos[2, i] = dr[2]
                st.pos[3, i] = dr[3]
                
                # Compute new local energy
                Enew_fast = MolSim.MC.local_energy(i, st, p)
                ΔU_fast = Enew_fast - Eold_fast
                
                # Compute brute-force ΔU
                pos_new = copy(pos_current)
                pos_new[1, i] = st.pos[1, i]  # Use wrapped position
                pos_new[2, i] = st.pos[2, i]
                pos_new[3, i] = st.pos[3, i]
                
                U_old_brute = brute_force_energy_test(pos_current, st.L, p, false)
                U_new_brute = brute_force_energy_test(pos_new, st.L, p, false)
                ΔU_brute = U_new_brute - U_old_brute
                
                # Check agreement
                @test abs(ΔU_fast - ΔU_brute) <= 1e-10 "Move $move_idx: ΔU mismatch (fast=$ΔU_fast, brute=$ΔU_brute)"
                
                # Check accept/reject decision matches
                # Metropolis: accept if ΔU <= 0 or rand() < exp(-β*ΔU)
                accept_fast = (ΔU_fast <= 0.0) || (u_accept < exp(-p.β * ΔU_fast))
                accept_brute = (ΔU_brute <= 0.0) || (u_accept < exp(-p.β * ΔU_brute))
                @test accept_fast == accept_brute "Move $move_idx: Accept/reject mismatch (fast=$accept_fast, brute=$accept_brute)"
                
                # Update state if move would be accepted
                if accept_fast
                    pos_current = pos_new
                    # Rebuild cell list for next move (as production does)
                    MolSim.MC.rebuild_cells!(st)
                else
                    # Restore old position in state
                    st.pos[1, i] = x_old
                    st.pos[2, i] = y_old
                    st.pos[3, i] = z_old
                end
            end
        end
    end
end
