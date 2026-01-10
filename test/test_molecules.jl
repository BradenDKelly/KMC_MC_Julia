"""
Tests for rigid molecule support.
All tests are additive - they do not modify existing atomic tests.
"""

using Test
using MolSim
using StaticArrays
using Random

@testset "Single-site molecule ≡ atom" begin
    # Create single-site molecule template
    template = MolSim.MC.create_single_site_molecule_template()
    @test template.n_sites == 1
    
    # Initialize atomic system
    N_atoms = 32  # FCC-valid: 4 * 2^3 = 32
    ρ = 0.5
    T = 1.0
    rc = 2.5
    p_atom, st_atom = MolSim.MC.init_fcc(N=N_atoms, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=12345)
    
    # Initialize molecular system with single-site molecules
    atom_pos = zeros(Float64, 3, 0)  # No atoms
    molecules = Vector{MolSim.MC.MoleculeState}()
    templates = [template]
    
    # Create N_atoms single-site molecules at same positions as atoms
    for i in 1:N_atoms
        com = MVector{3,Float64}(st_atom.pos[1, i], st_atom.pos[2, i], st_atom.pos[3, i])
        quat = MVector{4,Float64}(1.0, 0.0, 0.0, 0.0)  # Identity quaternion
        mol = MolSim.MC.MoleculeState(com, quat, 1)
        push!(molecules, mol)
    end
    
    L = st_atom.L
    sys_mol = MolSim.MC.init_molecular_system(atom_pos, molecules, templates, L, rc, 12345)
    
    # Update site positions
    MolSim.MC.update_site_positions!(sys_mol)
    
    # Energies should match (within numerical precision)
    U_atom = MolSim.MC.total_energy(st_atom, p_atom)
    U_mol = MolSim.MC.molecular_total_energy(sys_mol, p_atom)
    
    @test abs(U_atom - U_mol) < 1e-10
    
    # Pressures should match
    P_atom = MolSim.MC.pressure(st_atom, p_atom, T)
    # For molecular system, we'd need to compute pressure similarly
    # For now, just check energy equivalence
end

@testset "Translation invariance (molecules)" begin
    # Create diatomic molecule template
    template = MolSim.MC.create_diatomic_molecule_template(1.0)
    
    # Initialize system with one diatomic molecule
    atom_pos = zeros(Float64, 3, 0)
    molecules = Vector{MolSim.MC.MoleculeState}()
    templates = [template]
    
    L = 5.0
    rc = 2.5
    T = 1.0
    com = MVector{3,Float64}(L/2, L/2, L/2)
    quat = MVector{4,Float64}(1.0, 0.0, 0.0, 0.0)
    mol = MolSim.MC.MoleculeState(com, quat, 1)
    push!(molecules, mol)
    
    sys = MolSim.MC.init_molecular_system(atom_pos, molecules, templates, L, rc, 12345)
    
    p = MolSim.MC.LJParams(1.0, 1.0, rc, rc*rc, 1.0/T, 0.1, false, 0.0, 0.0, :truncated, false, 0.0)
    
    # Compute initial energy
    U_initial = MolSim.MC.molecular_total_energy(sys, p)
    
    # Translate molecule by integer multiple of box vectors
    shift = [L, 0.0, 0.0]
    sys.molecules[1].com[1] += shift[1]
    sys.molecules[1].com[2] += shift[2]
    sys.molecules[1].com[3] += shift[3]
    
    # Wrap COM
    MolSim.MC.wrap!(sys.molecules[1].com, L)
    
    # Update site positions
    MolSim.MC.update_site_positions!(sys)
    
    # Energy should be unchanged
    U_translated = MolSim.MC.molecular_total_energy(sys, p)
    @test abs(U_translated - U_initial) < 1e-10
end

@testset "Rotation invariance (isolated molecule)" begin
    # Create diatomic molecule template
    template = MolSim.MC.create_diatomic_molecule_template(1.0)
    
    # Initialize system with one isolated diatomic molecule
    atom_pos = zeros(Float64, 3, 0)
    molecules = Vector{MolSim.MC.MoleculeState}()
    templates = [template]
    
    L = 10.0  # Large box to isolate molecule
    rc = 2.5
    T = 1.0
    com = MVector{3,Float64}(L/2, L/2, L/2)
    quat = MVector{4,Float64}(1.0, 0.0, 0.0, 0.0)
    mol = MolSim.MC.MoleculeState(com, quat, 1)
    push!(molecules, mol)
    
    sys = MolSim.MC.init_molecular_system(atom_pos, molecules, templates, L, rc, 12345)
    
    p = MolSim.MC.LJParams(1.0, 1.0, rc, rc*rc, 1.0/T, 0.1, false, 0.0, 0.0, :truncated, false, 0.0)
    
    # Compute initial energy (should be ~0 for isolated molecule)
    U_initial = MolSim.MC.molecular_total_energy(sys, p)
    
    # Rotate molecule
    sys.molecules[1].quat = MolSim.MC.uniform_random_quaternion(Random.Xoshiro(99999))
    MolSim.MC.update_site_positions!(sys)
    
    # Energy should be unchanged (no intermolecular interactions)
    U_rotated = MolSim.MC.molecular_total_energy(sys, p)
    @test abs(U_rotated - U_initial) < 1e-10
end

@testset "Molecule move ΔU exactness" begin
    # Create diatomic molecule template
    template = MolSim.MC.create_diatomic_molecule_template(1.0)
    
    # Initialize system with two diatomic molecules
    atom_pos = zeros(Float64, 3, 0)
    molecules = Vector{MolSim.MC.MoleculeState}()
    templates = [template]
    
    L = 5.0
    rc = 2.5
    T = 1.0
    
    # First molecule
    com1 = MVector{3,Float64}(L/3, L/3, L/3)
    quat1 = MVector{4,Float64}(1.0, 0.0, 0.0, 0.0)
    mol1 = MolSim.MC.MoleculeState(com1, quat1, 1)
    push!(molecules, mol1)
    
    # Second molecule
    com2 = MVector{3,Float64}(2*L/3, 2*L/3, 2*L/3)
    quat2 = MVector{4,Float64}(1.0, 0.0, 0.0, 0.0)
    mol2 = MolSim.MC.MoleculeState(com2, quat2, 1)
    push!(molecules, mol2)
    
    sys = MolSim.MC.init_molecular_system(atom_pos, molecules, templates, L, rc, 12345)
    p = MolSim.MC.LJParams(1.0, 1.0, rc, rc*rc, 1.0/T, 0.1, false, 0.0, 0.0, :truncated, false, 0.0)
    
    # Compute initial energy
    U_old = MolSim.MC.molecular_total_energy(sys, p)
    
    # Store old COM
    com_old = copy(sys.molecules[1].com)
    
    # Compute ΔU using local energy (translation move)
    mol_idx = 1
    max_disp = 0.1
    dx = 0.05
    dy = 0.03
    dz = -0.02
    
    # Compute old local energy (sum over sites)
    Eold_local = 0.0
    site_start = sys.n_atoms
    for m in 1:(mol_idx-1)
        site_start += sys.templates[sys.molecules[m].template_idx].n_sites
    end
    template = sys.templates[sys.molecules[mol_idx].template_idx]
    for site_local_idx in 1:template.n_sites
        site_idx = site_start + site_local_idx
        Eold_local += MolSim.MC.molecular_local_energy(site_idx, sys, p)
    end
    
    # Apply translation
    sys.molecules[1].com[1] = com_old[1] + dx
    sys.molecules[1].com[2] = com_old[2] + dy
    sys.molecules[1].com[3] = com_old[3] + dz
    MolSim.MC.wrap!(sys.molecules[1].com, L)
    MolSim.MC.update_site_positions!(sys)
    
    # Compute new local energy
    Enew_local = 0.0
    for site_local_idx in 1:template.n_sites
        site_idx = site_start + site_local_idx
        Enew_local += MolSim.MC.molecular_local_energy(site_idx, sys, p)
    end
    
    ΔU_local = Enew_local - Eold_local
    
    # Compute exact ΔU from total energy
    U_new = MolSim.MC.molecular_total_energy(sys, p)
    ΔU_total = U_new - U_old
    
    @test abs(ΔU_local - ΔU_total) < 1e-10
end
