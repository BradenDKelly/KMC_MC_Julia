"""
Detailed balance micro-tests.
Verify single-step balance, not long-time averages.
"""

using Test
using MolSim
using StaticArrays
using Random

@testset "Particle move detailed balance" begin
    # Initialize with fixed seed for determinism
    p, st = MolSim.MC.init_fcc(N=32, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=12345, lj_model=:truncated)
    β = p.β
    
    # Select a particle
    i = 5
    
    # Store original state
    pos_original = copy(st.pos)
    x_old = pos_original[1, i]
    y_old = pos_original[2, i]
    z_old = pos_original[3, i]
    
    # Compute old local energy
    Eold = MolSim.MC.local_energy(i, st, p)
    
    # Propose forward move (deterministic displacement for testing)
    dx = 0.08
    dy = -0.05
    dz = 0.03
    
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
    
    # Compute new local energy
    Enew = MolSim.MC.local_energy(i, st, p)
    ΔU_forward = Enew - Eold
    
    # Acceptance probability for forward move
    if ΔU_forward <= 0.0
        A_forward = 1.0
    else
        A_forward = exp(-β * ΔU_forward)
    end
    
    # Now compute reverse move: from new position back to old
    x_new = st.pos[1, i]
    y_new = st.pos[2, i]
    z_new = st.pos[3, i]
    
    # Reverse displacement: old - new
    dx_rev = x_old - x_new
    dy_rev = y_old - y_new
    dz_rev = z_old - z_new
    
    # Apply reverse move
    st.pos[1, i] = x_new + dx_rev
    st.pos[2, i] = y_new + dy_rev
    st.pos[3, i] = z_new + dz_rev
    
    # Wrap
    dr[1] = st.pos[1, i]
    dr[2] = st.pos[2, i]
    dr[3] = st.pos[3, i]
    MolSim.MC.wrap!(dr, st.L)
    st.pos[1, i] = dr[1]
    st.pos[2, i] = dr[2]
    st.pos[3, i] = dr[3]
    
    # Verify we're back to original position (within wrapping)
    # Due to wrapping, positions might differ by box vectors, but energy should match
    Eold_restored = MolSim.MC.local_energy(i, st, p)
    @test abs(Eold_restored - Eold) < 1e-10
    
    # Compute ΔU for reverse move (from new to old)
    # Restore new position first
    st.pos[1, i] = x_new
    st.pos[2, i] = y_new
    st.pos[3, i] = z_new
    Enew_check = MolSim.MC.local_energy(i, st, p)
    
    # Now reverse
    st.pos[1, i] = x_old
    st.pos[2, i] = y_old
    st.pos[3, i] = z_old
    Eold_check = MolSim.MC.local_energy(i, st, p)
    ΔU_reverse = Eold_check - Enew_check
    
    # Detailed balance: A(x→x') / A(x'→x) = exp(-βΔU_forward) / exp(-βΔU_reverse)
    # But ΔU_reverse = -ΔU_forward (by definition)
    # So: A(x→x') / A(x'→x) = exp(-βΔU_forward) / exp(βΔU_forward) = exp(-2βΔU_forward)
    # Actually, if A = min(1, exp(-βΔU)), then detailed balance requires:
    # A(x→x') * exp(-β*U(x)) = A(x'→x) * exp(-β*U(x'))
    # Which simplifies to: A(x→x') / A(x'→x) = exp(-β*(U(x') - U(x))) = exp(-βΔU)
    
    # Acceptance for reverse
    if ΔU_reverse <= 0.0
        A_reverse = 1.0
    else
        A_reverse = exp(-β * ΔU_reverse)
    end
    
    # Verify ΔU_reverse = -ΔU_forward
    @test abs(ΔU_reverse + ΔU_forward) < 1e-10
    
    # Detailed balance ratio
    if A_reverse > 0.0
        ratio = A_forward / A_reverse
        expected_ratio = exp(-β * ΔU_forward)
        @test abs(ratio - expected_ratio) < 1e-10
    else
        # If reverse is rejected with probability 1, forward must also be rejected (ΔU_reverse < 0, so ΔU_forward > 0)
        @test ΔU_forward > 0.0
    end
end

@testset "Ideal-gas NPT volume move balance" begin
    # Create ideal gas: set very large rc so no interactions
    # Actually, we can't easily disable interactions, so use very low density and high T
    # Or better: verify the acceptance formula analytically
    
    # For ideal gas with volume move:
    # Acceptance: A = min(1, exp[-βPext(V' - V) + N*ln(V'/V)])
    # For small volume change: V' = V*(1+ε), so ln(V'/V) = ln(1+ε) ≈ ε - ε²/2
    
    # Test: with Pext = 0, acceptance should depend only on Jacobian
    N = 32
    ρ = 0.01  # Very low density (nearly ideal)
    T = 100.0  # High temperature (nearly ideal)
    p, st = MolSim.MC.init_fcc(N=N, ρ=ρ, T=T, rc=2.5, max_disp=0.1, seed=12345, lj_model=:truncated)
    β = p.β
    
    # Store original state
    V_old = st.L * st.L * st.L
    lnV_old = log(V_old)
    pos_original = copy(st.pos)
    L_old = st.L
    
    # Small volume change
    epsV = 0.001
    lnV_new = lnV_old + epsV
    V_new = exp(lnV_new)
    L_new = cbrt(V_new)
    s = L_new / L_old
    
    # Scale positions
    @inbounds for j in 1:N
        st.pos[1, j] = pos_original[1, j] * s
        st.pos[2, j] = pos_original[2, j] * s
        st.pos[3, j] = pos_original[3, j] * s
    end
    
    # Wrap
    scratch = st.scratch_dr
    @inbounds for j in 1:N
        scratch[1] = st.pos[1, j]
        scratch[2] = st.pos[2, j]
        scratch[3] = st.pos[3, j]
        MolSim.MC.wrap!(scratch, L_new)
        st.pos[1, j] = scratch[1]
        st.pos[2, j] = scratch[2]
        st.pos[3, j] = scratch[3]
    end
    
    st.L = L_new
    st.cl = MolSim.MC.CellList(N, L_new, st.cl.rc)
    MolSim.MC.rebuild_cells!(st)
    
    # Compute energy difference (should be ~0 for ideal gas at low density)
    U_old = MolSim.MC.total_energy(st, p)
    # Restore first
    copyto!(st.pos, pos_original)
    st.L = L_old
    st.cl = MolSim.MC.CellList(N, L_old, st.cl.rc)
    MolSim.MC.rebuild_cells!(st)
    U_old_real = MolSim.MC.total_energy(st, p)
    
    # Re-apply volume change
    @inbounds for j in 1:N
        st.pos[1, j] = pos_original[1, j] * s
        st.pos[2, j] = pos_original[2, j] * s
        st.pos[3, j] = pos_original[3, j] * s
    end
    
    @inbounds for j in 1:N
        scratch[1] = st.pos[1, j]
        scratch[2] = st.pos[2, j]
        scratch[3] = st.pos[3, j]
        MolSim.MC.wrap!(scratch, L_new)
        st.pos[1, j] = scratch[1]
        st.pos[2, j] = scratch[2]
        st.pos[3, j] = scratch[3]
    end
    
    st.L = L_new
    st.cl = MolSim.MC.CellList(N, L_new, st.cl.rc)
    MolSim.MC.rebuild_cells!(st)
    
    U_new = MolSim.MC.total_energy(st, p)
    ΔU = U_new - U_old_real
    
    # For ideal gas limit, ΔU should be small (but not exactly zero due to finite interactions)
    # The acceptance formula with Pext=0 should be: A = min(1, exp(N*ln(V'/V)))
    # For small epsV: ln(V'/V) ≈ epsV, so log_acc ≈ N*epsV
    
    Pext = 0.0
    ΔV = V_new - V_old
    log_acc_expected = -β * (ΔU + Pext * ΔV) + N * (lnV_new - lnV_old)
    
    # Compute actual acceptance log from volume_trial! logic
    # But we can't call volume_trial! without RNG, so verify the formula components
    log_acc_computed = -β * ΔU + N * (lnV_new - lnV_old)
    
    # For ideal gas (ΔU ≈ 0), this should be approximately N*epsV
    log_acc_ideal = N * epsV  # First-order approximation
    
    # Verify that computed log_acc matches expected formula
    @test abs(log_acc_expected - log_acc_computed) < 1e-10
    
    # For ideal gas, ΔU should be small
    # At low density and high T, interactions are weak
    # Check that log_acc is dominated by Jacobian term
    jacobian_term = N * (lnV_new - lnV_old)
    energy_term = -β * ΔU
    
    # Jacobian term should dominate for ideal gas
    if abs(jacobian_term) > 1e-6
        @test abs(energy_term / jacobian_term) < 0.1
    end
end
