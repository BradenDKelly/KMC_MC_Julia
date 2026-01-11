using Test
using MolSim
using Random
using StaticArrays

@testset "NVT neighbor-list rebuild equivalence" begin
    # Test parameters
    N = 64
    ρ = 0.5
    T = 1.0
    rc = 2.5
    max_disp = 0.15  # Larger to trigger rebuilds
    seed = 54321
    K = 150  # Number of attempted moves
    
    for lj_model in [:truncated, :shifted]
        @testset "LJ model: $lj_model" begin
            # Pre-generate deterministic move sequence
            rng_seq = Random.Xoshiro(seed)
            moves = Vector{Tuple{Int, Float64, Float64, Float64}}()
            for k in 1:K
                i = rand(rng_seq, 1:N)
                dx = (rand(rng_seq) - 0.5) * 2.0 * max_disp
                dy = (rand(rng_seq) - 0.5) * 2.0 * max_disp
                dz = (rand(rng_seq) - 0.5) * 2.0 * max_disp
                push!(moves, (i, dx, dy, dz))
            end
            
            # Run A: rebuild every step
            p_A, st_A = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp,
                                            seed=seed, use_lrc=false, lj_model=lj_model)
            rng_A = Random.Xoshiro(seed)
            energies_A = Float64[]
            accepts_A = Bool[]
            
            for (move_idx, (i, dx, dy, dz)) in enumerate(moves)
                # Store old position
                x_old = st_A.pos[1, i]
                y_old = st_A.pos[2, i]
                z_old = st_A.pos[3, i]
                
                # Compute old energy
                Eold = MolSim.MC.local_energy(i, st_A, p_A)
                
                # Apply trial move
                st_A.pos[1, i] = x_old + dx
                st_A.pos[2, i] = y_old + dy
                st_A.pos[3, i] = z_old + dz
                
                # Wrap
                dr = st_A.scratch_dr
                dr[1] = st_A.pos[1, i]
                dr[2] = st_A.pos[2, i]
                dr[3] = st_A.pos[3, i]
                MolSim.MC.wrap!(dr, st_A.L)
                st_A.pos[1, i] = dr[1]
                st_A.pos[2, i] = dr[2]
                st_A.pos[3, i] = dr[3]
                
                # Rebuild cell list every step (mode A)
                MolSim.MC.rebuild_cells!(st_A)
                
                # Compute new energy
                Enew = MolSim.MC.local_energy(i, st_A, p_A)
                ΔE = Enew - Eold
                
                # Accept/reject
                u = rand(rng_A)
                accepted = (ΔE <= 0.0) || (u < exp(-p_A.β * ΔE))
                push!(accepts_A, accepted)
                
                if accepted
                    # Rebuild again after acceptance (redundant but ensures consistency)
                    MolSim.MC.rebuild_cells!(st_A)
                    U_total = MolSim.MC.total_energy(st_A, p_A)
                    push!(energies_A, U_total)
                else
                    # Restore position
                    st_A.pos[1, i] = x_old
                    st_A.pos[2, i] = y_old
                    st_A.pos[3, i] = z_old
                    # Rebuild to restore state
                    MolSim.MC.rebuild_cells!(st_A)
                    U_total = MolSim.MC.total_energy(st_A, p_A)
                    push!(energies_A, U_total)
                end
            end
            
            # Run B: rebuild only when needed (production behavior)
            p_B, st_B = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp,
                                            seed=seed, use_lrc=false, lj_model=lj_model)
            rng_B = Random.Xoshiro(seed)
            energies_B = Float64[]
            accepts_B = Bool[]
            
            # Initial rebuild
            MolSim.MC.rebuild_cells!(st_B)
            
            for (move_idx, (i, dx, dy, dz)) in enumerate(moves)
                # Use production sweep! which handles rebuild logic
                # But we need to control the moves exactly, so simulate manually
                # Store old position
                x_old = st_B.pos[1, i]
                y_old = st_B.pos[2, i]
                z_old = st_B.pos[3, i]
                
                # Compute old energy (uses current cell list)
                Eold = MolSim.MC.local_energy(i, st_B, p_B)
                
                # Apply trial move
                st_B.pos[1, i] = x_old + dx
                st_B.pos[2, i] = y_old + dy
                st_B.pos[3, i] = z_old + dz
                
                # Wrap
                dr = st_B.scratch_dr
                dr[1] = st_B.pos[1, i]
                dr[2] = st_B.pos[2, i]
                dr[3] = st_B.pos[3, i]
                MolSim.MC.wrap!(dr, st_B.L)
                st_B.pos[1, i] = dr[1]
                st_B.pos[2, i] = dr[2]
                st_B.pos[3, i] = dr[3]
                
                # Compute new energy (cell list may be stale, but that's OK for comparison)
                Enew = MolSim.MC.local_energy(i, st_B, p_B)
                ΔE = Enew - Eold
                
                # Accept/reject
                u = rand(rng_B)
                accepted = (ΔE <= 0.0) || (u < exp(-p_B.β * ΔE))
                push!(accepts_B, accepted)
                
                if accepted
                    # Rebuild after acceptance (production behavior)
                    MolSim.MC.rebuild_cells!(st_B)
                    U_total = MolSim.MC.total_energy(st_B, p_B)
                    push!(energies_B, U_total)
                else
                    # Restore position
                    st_B.pos[1, i] = x_old
                    st_B.pos[2, i] = y_old
                    st_B.pos[3, i] = z_old
                    # No rebuild needed on rejection (cell list unchanged)
                    U_total = MolSim.MC.total_energy(st_B, p_B)
                    push!(energies_B, U_total)
                end
            end
            
            # Compare results
            @test length(accepts_A) == length(accepts_B)
            @test length(energies_A) == length(energies_B)
            
            for k in 1:length(accepts_A)
                @test accepts_A[k] == accepts_B[k] "Move $k: Accept/reject mismatch"
                @test abs(energies_A[k] - energies_B[k]) <= 1e-10 "Move $k: Energy mismatch (A=$(energies_A[k]), B=$(energies_B[k]))"
            end
        end
    end
end
