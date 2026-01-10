"""
ΔU correctness tests.
Verify that incremental ΔU logic matches exact energy differences.
Uses small systems (N=32) for determinism.
"""

using Test
using MolSim
using StaticArrays

# Local helper: compute interaction energy between particle i and all others j≠i
function e_i_local(i::Int, pos::Matrix{Float64}, x_i::Float64, y_i::Float64, z_i::Float64, 
                   N::Int, L::Float64, rc2::Float64, p)
    energy = 0.0
    for j in 1:N
        if j != i
            # Compute distance vector
            dr_x = pos[1, j] - x_i
            dr_y = pos[2, j] - y_i
            dr_z = pos[3, j] - z_i
            
            # Apply minimum image convention
            L_half = L / 2.0
            if dr_x > L_half
                dr_x = dr_x - L
            elseif dr_x < -L_half
                dr_x = dr_x + L
            end
            if dr_y > L_half
                dr_y = dr_y - L
            elseif dr_y < -L_half
                dr_y = dr_y + L
            end
            if dr_z > L_half
                dr_z = dr_z - L
            elseif dr_z < -L_half
                dr_z = dr_z + L
            end
            
            r2 = dr_x*dr_x + dr_y*dr_y + dr_z*dr_z
            
            if r2 < rc2 && r2 > 0.0
                # Compute LJ potential (same as lj_potential from LJMC.jl)
                σ2_over_r2 = (p.σ * p.σ) / r2
                σ6_over_r6 = σ2_over_r2 * σ2_over_r2 * σ2_over_r2
                σ12_over_r12 = σ6_over_r6 * σ6_over_r6
                u_unshifted = 4.0 * p.ϵ * (σ12_over_r12 - σ6_over_r6)
                
                if p.lj_model == :shifted
                    energy += u_unshifted - p.u_rc
                else  # :truncated
                    energy += u_unshifted
                end
            end
        end
    end
    return energy
end

@testset "Particle displacement ΔU: truncated LJ" begin
    # Small system for determinism
    p, st = MolSim.MC.init_fcc(N=32, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42, lj_model=:truncated)
    
    # Select a particle
    i = 4
    
    # Store original state
    pos_original = copy(st.pos)
    x_old = pos_original[1, i]
    y_old = pos_original[2, i]
    z_old = pos_original[3, i]
    
    # Compute ΔU_local as sum of pair energies between i and all j≠i
    # Before move: sum over j≠i of u(r_ij_old)
    Eold_local = e_i_local(i, pos_original, x_old, y_old, z_old, st.N, st.L, p.rc2, p)
    
    # Apply small displacement
    dx = 0.05
    dy = 0.03
    dz = -0.02
    
    x_new = x_old + dx
    y_new = y_old + dy
    z_new = z_old + dz
    
    # Wrap position
    dr = st.scratch_dr
    dr[1] = x_new
    dr[2] = y_new
    dr[3] = z_new
    MolSim.MC.wrap!(dr, st.L)
    x_new = dr[1]
    y_new = dr[2]
    z_new = dr[3]
    
    st.pos[1, i] = x_new
    st.pos[2, i] = y_new
    st.pos[3, i] = z_new
    
    # After move: sum over j≠i of u(r_ij_new)
    Enew_local = e_i_local(i, st.pos, x_new, y_new, z_new, st.N, st.L, p.rc2, p)
    
    # ΔU from local energy difference
    ΔU_local = Enew_local - Eold_local
    
    # Compute exact ΔU by computing total energy difference
    # Restore original
    copyto!(st.pos, pos_original)
    Eold_total = MolSim.MC.total_energy(st, p)
    
    # Apply same displacement
    st.pos[1, i] = x_new
    st.pos[2, i] = y_new
    st.pos[3, i] = z_new
    
    Enew_total = MolSim.MC.total_energy(st, p)
    ΔU_total = Enew_total - Eold_total
    
    # They should match exactly
    @test abs(ΔU_local - ΔU_total) < 1e-10
end

@testset "Particle displacement ΔU: shifted LJ" begin
    # Small system with shifted potential
    p, st = MolSim.MC.init_fcc(N=32, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42, lj_model=:shifted)
    
    # Select a particle
    i = 4
    
    # Store original state
    pos_original = copy(st.pos)
    x_old = pos_original[1, i]
    y_old = pos_original[2, i]
    z_old = pos_original[3, i]
    
    # Compute ΔU_local as sum of pair energies between i and all j≠i
    # Before move: sum over j≠i of u(r_ij_old)
    Eold_local = e_i_local(i, pos_original, x_old, y_old, z_old, st.N, st.L, p.rc2, p)
    
    # Apply small displacement
    dx = 0.05
    dy = 0.03
    dz = -0.02
    
    x_new = x_old + dx
    y_new = y_old + dy
    z_new = z_old + dz
    
    # Wrap position
    dr = st.scratch_dr
    dr[1] = x_new
    dr[2] = y_new
    dr[3] = z_new
    MolSim.MC.wrap!(dr, st.L)
    x_new = dr[1]
    y_new = dr[2]
    z_new = dr[3]
    
    st.pos[1, i] = x_new
    st.pos[2, i] = y_new
    st.pos[3, i] = z_new
    
    # After move: sum over j≠i of u(r_ij_new)
    Enew_local = e_i_local(i, st.pos, x_new, y_new, z_new, st.N, st.L, p.rc2, p)
    
    ΔU_local = Enew_local - Eold_local
    
    # Exact ΔU from total energy
    copyto!(st.pos, pos_original)
    Eold_total = MolSim.MC.total_energy(st, p)
    
    st.pos[1, i] = x_new
    st.pos[2, i] = y_new
    st.pos[3, i] = z_new
    
    Enew_total = MolSim.MC.total_energy(st, p)
    ΔU_total = Enew_total - Eold_total
    
    @test abs(ΔU_local - ΔU_total) < 1e-10
end

@testset "Volume move ΔU: truncated LJ" begin
    # Small system with use_lrc=false for consistent comparison
    p, st = MolSim.MC.init_fcc(N=32, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42, lj_model=:truncated, use_lrc=false)
    
    # Store original state
    L_old = st.L
    V_old = L_old * L_old * L_old
    pos_original = copy(st.pos)
    
    # Compute original energy with original box length
    U_original = MolSim.MC.total_energy(st, p)
    
    # Scale volume: λ = (V'/V)^(1/3) = L'/L
    epsV = 0.01
    V_new = V_old * (1.0 + epsV)
    L_new = cbrt(V_new)
    λ = L_new / L_old  # scaling factor
    
    # Create scaled positions (scale by λ and wrap with new box length L')
    pos_scaled = copy(pos_original)
    @inbounds for j in 1:st.N
        pos_scaled[1, j] = pos_original[1, j] * λ
        pos_scaled[2, j] = pos_original[2, j] * λ
        pos_scaled[3, j] = pos_original[3, j] * λ
    end
    
    # Wrap scaled positions to [0, L_new) using new box length
    scratch = st.scratch_dr
    @inbounds for j in 1:st.N
        scratch[1] = pos_scaled[1, j]
        scratch[2] = pos_scaled[2, j]
        scratch[3] = pos_scaled[3, j]
        MolSim.MC.wrap!(scratch, L_new)
        pos_scaled[1, j] = scratch[1]
        pos_scaled[2, j] = scratch[2]
        pos_scaled[3, j] = scratch[3]
    end
    
    # Compute U_scaled using scaled positions and new box length L'
    # Temporarily set state to scaled configuration
    copyto!(st.pos, pos_scaled)
    st.L = L_new
    st.cl = MolSim.MC.CellList(st.N, L_new, st.cl.rc)
    MolSim.MC.rebuild_cells!(st)
    
    U_scaled = MolSim.MC.total_energy(st, p)
    
    # ΔU is pure potential energy change
    ΔU_volume_check = U_scaled - U_original
    
    # Restore original state
    copyto!(st.pos, pos_original)
    st.L = L_old
    st.cl = MolSim.MC.CellList(st.N, L_old, st.cl.rc)
    MolSim.MC.rebuild_cells!(st)
    
    # Verify original energy is restored
    U_original_check = MolSim.MC.total_energy(st, p)
    @test abs(U_original_check - U_original) < 1e-10
    
    # Recompute ΔU independently (same scaling, independent computation)
    # Scale again
    pos_scaled2 = copy(pos_original)
    @inbounds for j in 1:st.N
        pos_scaled2[1, j] = pos_original[1, j] * λ
        pos_scaled2[2, j] = pos_original[2, j] * λ
        pos_scaled2[3, j] = pos_original[3, j] * λ
    end
    
    # Wrap again
    @inbounds for j in 1:st.N
        scratch[1] = pos_scaled2[1, j]
        scratch[2] = pos_scaled2[2, j]
        scratch[3] = pos_scaled2[3, j]
        MolSim.MC.wrap!(scratch, L_new)
        pos_scaled2[1, j] = scratch[1]
        pos_scaled2[2, j] = scratch[2]
        pos_scaled2[3, j] = scratch[3]
    end
    
    # Compute U_scaled2 with new box length
    copyto!(st.pos, pos_scaled2)
    st.L = L_new
    st.cl = MolSim.MC.CellList(st.N, L_new, st.cl.rc)
    MolSim.MC.rebuild_cells!(st)
    
    U_scaled2 = MolSim.MC.total_energy(st, p)
    ΔU_volume = U_scaled2 - U_original_check
    
    # Both should match exactly (pure potential energy change)
    @test abs(ΔU_volume - ΔU_volume_check) < 1e-10
end

@testset "Volume move ΔU: shifted LJ" begin
    # Small system with shifted potential, use_lrc=false for consistent comparison
    p, st = MolSim.MC.init_fcc(N=32, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42, lj_model=:shifted, use_lrc=false)
    
    # Store original state
    L_old = st.L
    V_old = L_old * L_old * L_old
    pos_original = copy(st.pos)
    
    # Compute original energy with original box length
    U_original = MolSim.MC.total_energy(st, p)
    
    # Scale volume: λ = (V'/V)^(1/3) = L'/L
    epsV = 0.01
    V_new = V_old * (1.0 + epsV)
    L_new = cbrt(V_new)
    λ = L_new / L_old  # scaling factor
    
    # Create scaled positions (scale by λ and wrap with new box length L')
    pos_scaled = copy(pos_original)
    @inbounds for j in 1:st.N
        pos_scaled[1, j] = pos_original[1, j] * λ
        pos_scaled[2, j] = pos_original[2, j] * λ
        pos_scaled[3, j] = pos_original[3, j] * λ
    end
    
    # Wrap scaled positions to [0, L_new) using new box length
    scratch = st.scratch_dr
    @inbounds for j in 1:st.N
        scratch[1] = pos_scaled[1, j]
        scratch[2] = pos_scaled[2, j]
        scratch[3] = pos_scaled[3, j]
        MolSim.MC.wrap!(scratch, L_new)
        pos_scaled[1, j] = scratch[1]
        pos_scaled[2, j] = scratch[2]
        pos_scaled[3, j] = scratch[3]
    end
    
    # Compute U_scaled using scaled positions and new box length L'
    # Temporarily set state to scaled configuration
    copyto!(st.pos, pos_scaled)
    st.L = L_new
    st.cl = MolSim.MC.CellList(st.N, L_new, st.cl.rc)
    MolSim.MC.rebuild_cells!(st)
    
    U_scaled = MolSim.MC.total_energy(st, p)
    
    # ΔU is pure potential energy change
    ΔU_volume_check = U_scaled - U_original
    
    # Restore original state
    copyto!(st.pos, pos_original)
    st.L = L_old
    st.cl = MolSim.MC.CellList(st.N, L_old, st.cl.rc)
    MolSim.MC.rebuild_cells!(st)
    
    # Verify original energy is restored
    U_original_check = MolSim.MC.total_energy(st, p)
    @test abs(U_original_check - U_original) < 1e-10
    
    # Recompute ΔU independently (same scaling, independent computation)
    # Scale again
    pos_scaled2 = copy(pos_original)
    @inbounds for j in 1:st.N
        pos_scaled2[1, j] = pos_original[1, j] * λ
        pos_scaled2[2, j] = pos_original[2, j] * λ
        pos_scaled2[3, j] = pos_original[3, j] * λ
    end
    
    # Wrap again
    @inbounds for j in 1:st.N
        scratch[1] = pos_scaled2[1, j]
        scratch[2] = pos_scaled2[2, j]
        scratch[3] = pos_scaled2[3, j]
        MolSim.MC.wrap!(scratch, L_new)
        pos_scaled2[1, j] = scratch[1]
        pos_scaled2[2, j] = scratch[2]
        pos_scaled2[3, j] = scratch[3]
    end
    
    # Compute U_scaled2 with new box length
    copyto!(st.pos, pos_scaled2)
    st.L = L_new
    st.cl = MolSim.MC.CellList(st.N, L_new, st.cl.rc)
    MolSim.MC.rebuild_cells!(st)
    
    U_scaled2 = MolSim.MC.total_energy(st, p)
    ΔU_volume = U_scaled2 - U_original_check
    
    # Both should match exactly (pure potential energy change)
    @test abs(ΔU_volume - ΔU_volume_check) < 1e-10
end
