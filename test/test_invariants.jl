"""
Hamiltonian invariance tests.
Tests verify physical invariants without Monte Carlo loops.
All tests must be deterministic.
"""

using Test
using MolSim
using StaticArrays

@testset "Translation invariance" begin
    # Initialize configuration
    p, st = MolSim.MC.init_fcc(N=108, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42)
    
    # Compute initial energy
    E_initial = MolSim.MC.total_energy(st, p)
    
    # Translate all particles by integer multiples of box vectors
    # This should leave energy unchanged due to periodic boundary conditions
    shifts = [
        [st.L, 0.0, 0.0],
        [0.0, st.L, 0.0],
        [0.0, 0.0, st.L],
        [2.0*st.L, -3.0*st.L, 1.0*st.L],
    ]
    
    for shift in shifts
        # Store original positions
        pos_original = copy(st.pos)
        
        # Translate all particles
        @inbounds for i in 1:st.N
            st.pos[1, i] += shift[1]
            st.pos[2, i] += shift[2]
            st.pos[3, i] += shift[3]
        end
        
        # Wrap all positions (required after translation)
        scratch = st.scratch_dr
        @inbounds for i in 1:st.N
            scratch[1] = st.pos[1, i]
            scratch[2] = st.pos[2, i]
            scratch[3] = st.pos[3, i]
            MolSim.MC.wrap!(scratch, st.L)
            st.pos[1, i] = scratch[1]
            st.pos[2, i] = scratch[2]
            st.pos[3, i] = scratch[3]
        end
        
        # Energy should be unchanged
        E_translated = MolSim.MC.total_energy(st, p)
        @test abs(E_translated - E_initial) < 1e-10
        
        # Restore original positions
        copyto!(st.pos, pos_original)
    end
end

@testset "Permutation invariance" begin
    # Initialize configuration
    p, st = MolSim.MC.init_fcc(N=32, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42)
    
    # Compute initial energy and virial
    E_initial = MolSim.MC.total_energy(st, p)
    W_initial = MolSim.MC.total_virial(st, p)
    
    # Store original positions
    pos_original = copy(st.pos)
    
    # Permute particle indices: swap first and last half
    N = st.N
    pos_permuted = similar(st.pos)
    @inbounds for i in 1:N÷2
        pos_permuted[1, i] = pos_original[1, N÷2 + i]
        pos_permuted[2, i] = pos_original[2, N÷2 + i]
        pos_permuted[3, i] = pos_original[3, N÷2 + i]
    end
    @inbounds for i in (N÷2 + 1):N
        pos_permuted[1, i] = pos_original[1, i - N÷2]
        pos_permuted[2, i] = pos_original[2, i - N÷2]
        pos_permuted[3, i] = pos_original[3, i - N÷2]
    end
    
    copyto!(st.pos, pos_permuted)
    
    # Energy and virial should be unchanged (same configuration, different labeling)
    E_permuted = MolSim.MC.total_energy(st, p)
    W_permuted = MolSim.MC.total_virial(st, p)
    
    @test abs(E_permuted - E_initial) < 1e-10
    @test abs(W_permuted - W_initial) < 1e-10
end

@testset "Potential-shift consistency" begin
    # Initialize same configuration with truncated and shifted potentials
    N = 32
    ρ = 0.8
    T = 1.0
    rc = 2.5
    seed = 42
    
    p_truncated, st_truncated = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=seed, lj_model=:truncated)
    p_shifted, st_shifted = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=seed, lj_model=:shifted)
    
    # Ensure positions are identical
    copyto!(st_shifted.pos, st_truncated.pos)
    
    # Compute energies
    U_unshifted = MolSim.MC.total_energy(st_truncated, p_truncated)
    U_shifted = MolSim.MC.total_energy(st_shifted, p_shifted)
    
    # Count pairs within cutoff
    N_pairs = 0
    L = st_truncated.L
    rc2 = p_truncated.rc2
    L_half = L / 2.0
    pos = st_truncated.pos
    
    @inbounds for i in 1:N
        for j in (i+1):N
            dr_x = pos[1, j] - pos[1, i]
            dr_y = pos[2, j] - pos[2, i]
            dr_z = pos[3, j] - pos[3, i]
            
            # Minimum image
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
                N_pairs += 1
            end
        end
    end
    
    # Expected relationship: U_shifted = U_unshifted - N_pairs * u(rc)
    u_rc = p_shifted.u_rc
    U_shifted_expected = U_unshifted - Float64(N_pairs) * u_rc
    
    @test abs(U_shifted - U_shifted_expected) < 1e-10
end

@testset "Volume-scaling symmetry" begin
    # Initialize configuration
    p, st = MolSim.MC.init_fcc(N=32, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42)
    
    # Store original state
    pos_original = copy(st.pos)
    L_initial = st.L
    V_initial = L_initial * L_initial * L_initial
    U_initial = MolSim.MC.total_energy(st, p)
    
    # Test single epsilon value (smaller test for speed)
    eps = 1e-3
    
    # Helper to compute energy at scaled volume
    function energy_at_volume(V::Float64, pos_ref, L_ref, p_ref, st_ref)
        L_new = cbrt(V)
        s = L_new / L_ref
        
        # Scale positions
        pos_scaled = similar(pos_ref)
        @inbounds for i in 1:size(pos_ref, 2)
            pos_scaled[1, i] = pos_ref[1, i] * s
            pos_scaled[2, i] = pos_ref[2, i] * s
            pos_scaled[3, i] = pos_ref[3, i] * s
        end
        
        # Wrap positions
        scratch = st_ref.scratch_dr
        @inbounds for i in 1:size(pos_ref, 2)
            scratch[1] = pos_scaled[1, i]
            scratch[2] = pos_scaled[2, i]
            scratch[3] = pos_scaled[3, i]
            MolSim.MC.wrap!(scratch, L_new)
            pos_scaled[1, i] = scratch[1]
            pos_scaled[2, i] = scratch[2]
            pos_scaled[3, i] = scratch[3]
        end
        
        # Temporarily modify state
        pos_orig = copy(st_ref.pos)
        L_orig = st_ref.L
        cl_orig = st_ref.cl
        
        copyto!(st_ref.pos, pos_scaled)
        st_ref.L = L_new
        st_ref.cl = MolSim.MC.CellList(st_ref.N, L_new, st_ref.cl.rc)
        MolSim.MC.rebuild_cells!(st_ref)
        
        E_scaled = MolSim.MC.total_energy(st_ref, p_ref)
        
        # Restore
        copyto!(st_ref.pos, pos_orig)
        st_ref.L = L_orig
        st_ref.cl = cl_orig
        MolSim.MC.rebuild_cells!(st_ref)
        
        return E_scaled
    end
    
    # Compute ΔU for +eps and -eps
    V_pos = V_initial * (1.0 + eps)
    V_neg = V_initial * (1.0 - eps)
    
    U_pos = energy_at_volume(V_pos, pos_original, L_initial, p, st)
    U_neg = energy_at_volume(V_neg, pos_original, L_initial, p, st)
    
    ΔU_pos = U_pos - U_initial
    ΔU_neg = U_initial - U_neg  # Note: sign flipped for negative volume change
    
    # For small eps, ΔU(+eps) and ΔU(-eps) should be approximately symmetric
    # The difference should be O(eps²)
    asymmetry = abs(ΔU_pos - ΔU_neg)
    eps_magnitude = abs(eps)
    
    # Asymmetry should scale as eps², so ratio should be O(eps)
    if eps_magnitude > 0.0
        asymmetry_normalized = asymmetry / eps_magnitude
            @test asymmetry_normalized < 1e-2
    end
end
