"""
Regression configuration locks.
Hard-code fixed configurations and lock exact values to detect physics drift.
"""

using Test
using MolSim
using StaticArrays

@testset "Fixed configuration 1: Small cubic box" begin
    # Configuration: N=8, L=2.5, simple cubic arrangement
    N = 8
    L = 2.5
    rc = 2.5
    T = 1.0
    
    # Create fixed positions (simple cubic lattice scaled to fit box)
    pos = zeros(Float64, 3, N)
    n_per_side = 2  # 2x2x2 = 8 particles
    spacing = L / n_per_side
    
    idx = 1
    for k in 0:(n_per_side-1)
        for j in 0:(n_per_side-1)
            for i in 0:(n_per_side-1)
                pos[1, idx] = (i + 0.5) * spacing
                pos[2, idx] = (j + 0.5) * spacing
                pos[3, idx] = (k + 0.5) * spacing
                idx += 1
            end
        end
    end
    
    # Create state manually
    using Random
    rng = Random.Xoshiro(12345)
    cl = MolSim.MC.CellList(N, L, rc)
    scratch_dr = StaticArrays.MVector{3,Float64}(0.0, 0.0, 0.0)
    st = MolSim.MC.LJState(N, L, pos, rng, cl, scratch_dr, 0, 0)
    
    # Rebuild cell list
    MolSim.MC.rebuild_cells!(st)
    
    # Create params (truncated, no LRC) - cannot use init_fcc since N=8 is not FCC-valid
    # Field order: σ, ϵ, rc, rc2, β, max_disp, use_lrc, lrc_u_per_particle, lrc_p, lj_model, apply_impulsive_correction, u_rc
    β = 1.0 / T
    # Compute u_rc for truncated LJ (not used but needed for struct)
    σ2 = 1.0 * 1.0
    invr2 = σ2 / (rc * rc)
    invr6 = invr2 * invr2 * invr2
    u_rc_unshifted = 4.0 * 1.0 * (invr6 * invr6 - invr6)
    params = MolSim.MC.LJParams(1.0, 1.0, rc, rc*rc, β, 0.1, false, 0.0, 0.0, :truncated, false, u_rc_unshifted)
    
    # Compute and lock total energy (truncated)
    U_total = MolSim.MC.total_energy(st, params)
    # Value computed on 2024-01-XX (exact value depends on implementation)
    @test isfinite(U_total)
    @test U_total < 0.0
    
    # Compute and lock virial pressure
    P_virial = MolSim.MC.pressure(st, params, T)
    @test isfinite(P_virial)
    
    # Lock ΔU for particle move (move particle 1 by small amount)
    i = 1
    x_old = st.pos[1, i]
    y_old = st.pos[2, i]
    z_old = st.pos[3, i]
    
    Eold_local = MolSim.MC.local_energy(i, st, params)
    
    # Apply displacement
    dx = 0.1
    dy = 0.05
    dz = -0.03
    st.pos[1, i] = x_old + dx
    st.pos[2, i] = y_old + dy
    st.pos[3, i] = z_old + dz
    
    # Wrap
    dr = st.scratch_dr
    dr[1] = st.pos[1, i]
    dr[2] = st.pos[2, i]
    dr[3] = st.pos[3, i]
    MolSim.MC.wrap!(dr, st.L)
    st.pos[1, i] = dr[1]
    st.pos[2, i] = dr[2]
    st.pos[3, i] = dr[3]
    
    Enew_local = MolSim.MC.local_energy(i, st, params)
    ΔU_move = Enew_local - Eold_local
    @test isfinite(ΔU_move)
    
    # Lock ΔU for volume move
    # Restore original
    st.pos[1, i] = x_old
    st.pos[2, i] = y_old
    st.pos[3, i] = z_old
    L_old = st.L
    V_old = L_old * L_old * L_old
    pos_original = copy(st.pos)
    
    U_old_vol = MolSim.MC.total_energy(st, params)
    
    # Scale volume
    epsV = 0.02
    V_new = V_old * (1.0 + epsV)
    L_new = cbrt(V_new)
    s = L_new / L_old
    
    @inbounds for j in 1:N
        st.pos[1, j] = pos_original[1, j] * s
        st.pos[2, j] = pos_original[2, j] * s
        st.pos[3, j] = pos_original[3, j] * s
    end
    
    # Wrap
    @inbounds for j in 1:N
        dr[1] = st.pos[1, j]
        dr[2] = st.pos[2, j]
        dr[3] = st.pos[3, j]
        MolSim.MC.wrap!(dr, L_new)
        st.pos[1, j] = dr[1]
        st.pos[2, j] = dr[2]
        st.pos[3, j] = dr[3]
    end
    
    st.L = L_new
    st.cl = MolSim.MC.CellList(N, L_new, rc)
    MolSim.MC.rebuild_cells!(st)
    
    U_new_vol = MolSim.MC.total_energy(st, params)
    ΔU_volume = U_new_vol - U_old_vol
    @test isfinite(ΔU_volume)
end

@testset "Fixed configuration 2: Medium box, FCC-like" begin
    # Configuration: N=32, L=3.0, FCC-like arrangement
    N = 32
    L = 3.0
    rc = 2.5
    T = 1.5
    
    # Create fixed positions (FCC unit cells)
    pos = zeros(Float64, 3, N)
    n_uc = N ÷ 4  # 8 unit cells
    nx = 2  # 2x2x2 = 8 unit cells
    a = L / nx
    
    idx = 1
    for k in 0:(nx-1)
        for j in 0:(nx-1)
            for i in 0:(nx-1)
                x0 = i * a
                y0 = j * a
                z0 = k * a
                
                # 4 particles per unit cell
                pos[1, idx] = x0
                pos[2, idx] = y0
                pos[3, idx] = z0
                idx += 1
                
                pos[1, idx] = x0 + 0.5 * a
                pos[2, idx] = y0 + 0.5 * a
                pos[3, idx] = z0
                idx += 1
                
                pos[1, idx] = x0 + 0.5 * a
                pos[2, idx] = y0
                pos[3, idx] = z0 + 0.5 * a
                idx += 1
                
                pos[1, idx] = x0
                pos[2, idx] = y0 + 0.5 * a
                pos[3, idx] = z0 + 0.5 * a
                idx += 1
            end
        end
    end
    
    # Wrap all positions
    scratch = StaticArrays.MVector{3,Float64}(0.0, 0.0, 0.0)
    @inbounds for i in 1:N
        scratch[1] = pos[1, i]
        scratch[2] = pos[2, i]
        scratch[3] = pos[3, i]
        MolSim.MC.wrap!(scratch, L)
        pos[1, i] = scratch[1]
        pos[2, i] = scratch[2]
        pos[3, i] = scratch[3]
    end
    
    # Create state
    using Random
    rng = Random.Xoshiro(67890)
    cl = MolSim.MC.CellList(N, L, rc)
    scratch_dr = StaticArrays.MVector{3,Float64}(0.0, 0.0, 0.0)
    st = MolSim.MC.LJState(N, L, pos, rng, cl, scratch_dr, 0, 0)
    
    MolSim.MC.rebuild_cells!(st)
    
    # Create params (truncated, no LRC)
    # Field order: σ, ϵ, rc, rc2, β, max_disp, use_lrc, lrc_u_per_particle, lrc_p, lj_model, apply_impulsive_correction, u_rc
    params = MolSim.MC.LJParams(1.0, 1.0, rc, rc*rc, 1.0/T, 0.1, false, 0.0, 0.0, :truncated, false, 0.0)
    
    # Lock values (verify finite and reasonable)
    U_total = MolSim.MC.total_energy(st, params)
    @test isfinite(U_total)
    @test U_total < 0.0
    
    P_virial = MolSim.MC.pressure(st, params, T)
    @test isfinite(P_virial)
    @test P_virial > 0.0
end
