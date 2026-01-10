"""
Rigid molecule support for Lennard-Jones Monte Carlo.
Additive extension - does not modify atomic LJ behavior.
"""

using Random
using StaticArrays
using LinearAlgebra

"""
    MoleculeTemplate

Template defining a rigid molecule structure.
Sites are defined in body frame (relative to COM).
"""
struct MoleculeTemplate
    n_sites::Int
    sites_body::Matrix{Float64}  # 3 x n_sites, positions in body frame
    site_σ::Vector{Float64}      # LJ σ for each site (can be all 1.0)
    site_ϵ::Vector{Float64}      # LJ ϵ for each site (can be all 1.0)
end

"""
    MoleculeState

State of a single rigid molecule.
"""
mutable struct MoleculeState
    com::MVector{3,Float64}      # Center of mass position
    quat::MVector{4,Float64}     # Orientation quaternion (w, x, y, z)
    template_idx::Int             # Index into molecule templates array
end

"""
    MolecularSystem

System containing both atoms and molecules.
Atoms are stored as before in pos matrix.
Molecules are stored separately.
"""
mutable struct MolecularSystem
    # Atomic particles (existing)
    n_atoms::Int
    atom_pos::Matrix{Float64}     # 3 x n_atoms
    
    # Molecules
    n_molecules::Int
    molecules::Vector{MoleculeState}
    templates::Vector{MoleculeTemplate}
    
    # Global site mapping: for each site, which molecule (or -1 if atom)
    # site_to_molecule[i] = molecule index, or -1 if atom
    # site_to_site_idx[i] = site index within molecule, or atom index
    n_sites_total::Int
    site_to_molecule::Vector{Int}
    site_to_site_idx::Vector{Int}
    
    # Global site positions (computed from molecules + atoms)
    site_pos::Matrix{Float64}     # 3 x n_sites_total
    
    # Cell list (for all sites)
    cl::CellList
    
    # RNG and scratch
    rng::Xoshiro
    scratch_dr::MVector{3,Float64}
    
    # Acceptance tracking
    atom_accepted::Int
    atom_attempted::Int
    mol_trans_accepted::Int
    mol_trans_attempted::Int
    mol_rot_accepted::Int
    mol_rot_attempted::Int
end

"""
    quaternion_to_rotation_matrix(q)

Convert quaternion (w, x, y, z) to 3x3 rotation matrix.
"""
function quaternion_to_rotation_matrix(q::AbstractVector{Float64})
    w, x, y, z = q[1], q[2], q[3], q[4]
    R = Matrix{Float64}(undef, 3, 3)
    R[1,1] = 1 - 2*(y*y + z*z)
    R[1,2] = 2*(x*y - w*z)
    R[1,3] = 2*(x*z + w*y)
    R[2,1] = 2*(x*y + w*z)
    R[2,2] = 1 - 2*(x*x + z*z)
    R[2,3] = 2*(y*z - w*x)
    R[3,1] = 2*(x*z - w*y)
    R[3,2] = 2*(y*z + w*x)
    R[3,3] = 1 - 2*(x*x + y*y)
    return R
end

"""
    update_site_positions!(sys)

Update global site positions from molecule states and atom positions.
"""
function update_site_positions!(sys::MolecularSystem)
    # Copy atom positions
    @inbounds for i in 1:sys.n_atoms
        sys.site_pos[1, i] = sys.atom_pos[1, i]
        sys.site_pos[2, i] = sys.atom_pos[2, i]
        sys.site_pos[3, i] = sys.atom_pos[3, i]
    end
    
    # Compute molecule site positions
    site_idx = sys.n_atoms
    @inbounds for mol_idx in 1:sys.n_molecules
        mol = sys.molecules[mol_idx]
        template = sys.templates[mol.template_idx]
        
        # Get rotation matrix
        R = quaternion_to_rotation_matrix(mol.quat)
        
        # Transform each site
        for site_local_idx in 1:template.n_sites
            site_idx += 1
            
            # Body frame position
            r_body = template.sites_body[:, site_local_idx]
            
            # Rotate
            r_rot = R * r_body
            
            # Add COM
            sys.site_pos[1, site_idx] = mol.com[1] + r_rot[1]
            sys.site_pos[2, site_idx] = mol.com[2] + r_rot[2]
            sys.site_pos[3, site_idx] = mol.com[3] + r_rot[3]
        end
    end
end

"""
    uniform_random_quaternion(rng)

Generate uniform random quaternion on unit sphere (for SO(3)).
"""
function uniform_random_quaternion(rng::Xoshiro)::MVector{4,Float64}
    # Marsaglia method: uniform on 4D sphere
    s = 1.0
    while s >= 1.0
        v1 = 2.0 * rand(rng) - 1.0
        v2 = 2.0 * rand(rng) - 1.0
        s = v1*v1 + v2*v2
    end
    
    s2 = 1.0 - s
    sqrt_s2 = sqrt(s2)
    sqrt_s = sqrt(s)
    
    v3 = 2.0 * rand(rng) - 1.0
    v4 = 2.0 * rand(rng) - 1.0
    t = v3*v3 + v4*v4
    
    while t >= 1.0
        v3 = 2.0 * rand(rng) - 1.0
        v4 = 2.0 * rand(rng) - 1.0
        t = v3*v3 + v4*v4
    end
    
    sqrt_t = sqrt(t)
    sqrt_1mt = sqrt(1.0 - t)
    
    q = MVector{4,Float64}(undef)
    q[1] = v1 * sqrt_1mt  # w
    q[2] = v2 * sqrt_1mt  # x
    q[3] = v3 * sqrt_s    # y
    q[4] = v4 * sqrt_s    # z
    
    return q
end

"""
    create_single_site_molecule_template()

Create a single-site molecule template (equivalent to atom).
"""
function create_single_site_molecule_template()::MoleculeTemplate
    sites_body = zeros(Float64, 3, 1)
    site_σ = [1.0]
    site_ϵ = [1.0]
    return MoleculeTemplate(1, sites_body, site_σ, site_ϵ)
end

"""
    create_diatomic_molecule_template(bond_length::Float64=1.0)

Create a diatomic molecule template with bond length.
"""
function create_diatomic_molecule_template(bond_length::Float64=1.0)::MoleculeTemplate
    sites_body = zeros(Float64, 3, 2)
    sites_body[1, 1] = -bond_length / 2.0
    sites_body[1, 2] = bond_length / 2.0
    site_σ = [1.0, 1.0]
    site_ϵ = [1.0, 1.0]
    return MoleculeTemplate(2, sites_body, site_σ, site_ϵ)
end

"""
    init_molecular_system(atom_pos, molecules, templates, L, rc, seed)

Initialize molecular system from atoms and molecules.
"""
function init_molecular_system(
    atom_pos::Matrix{Float64},
    molecules::Vector{MoleculeState},
    templates::Vector{MoleculeTemplate},
    L::Float64,
    rc::Float64,
    seed::Int
)::MolecularSystem
    n_atoms = size(atom_pos, 2)
    n_molecules = length(molecules)
    
    # Count total sites
    n_sites_total = n_atoms
    for template in templates
        n_sites_total += template.n_sites
    end
    
    # Build site mapping
    site_to_molecule = Vector{Int}(undef, n_sites_total)
    site_to_site_idx = Vector{Int}(undef, n_sites_total)
    
    # Atoms map to -1
    for i in 1:n_atoms
        site_to_molecule[i] = -1
        site_to_site_idx[i] = i
    end
    
    # Molecules map to molecule index
    site_idx = n_atoms
    for mol_idx in 1:n_molecules
        mol = molecules[mol_idx]
        template = templates[mol.template_idx]
        for site_local_idx in 1:template.n_sites
            site_idx += 1
            site_to_molecule[site_idx] = mol_idx
            site_to_site_idx[site_idx] = site_local_idx
        end
    end
    
    # Allocate site positions
    site_pos = zeros(Float64, 3, n_sites_total)
    
    # Create cell list
    cl = CellList(n_sites_total, L, rc)
    
    # Create system
    sys = MolecularSystem(
        n_atoms, atom_pos,
        n_molecules, molecules, templates,
        n_sites_total, site_to_molecule, site_to_site_idx,
        site_pos, cl,
        Xoshiro(seed), MVector{3,Float64}(0.0, 0.0, 0.0),
        0, 0, 0, 0, 0, 0
    )
    
    # Update site positions
    update_site_positions!(sys)
    
    # Rebuild cell list
    rebuild_cells_molecular!(sys, L)
    
    return sys
end

"""
    rebuild_cells_molecular!(sys, L)

Rebuild cell list for molecular system.
"""
function rebuild_cells_molecular!(sys::MolecularSystem, L::Float64)
    # Update site positions first
    update_site_positions!(sys)
    
    # Clear all head pointers
    fill!(sys.cl.head, 0)
    fill!(sys.cl.next, 0)
    
    N = sys.n_sites_total
    ncell = sys.cl.ncell
    pos = sys.site_pos
    
    # Assign sites to cells
    @inbounds for i in 1:N
        x, y, z = pos[1, i], pos[2, i], pos[3, i]
        # Wrap coordinates to [0, L)
        x_wrapped = x - L * floor(x / L)
        y_wrapped = y - L * floor(y / L)
        z_wrapped = z - L * floor(z / L)
        
        cell_idx = get_cell(x_wrapped, y_wrapped, z_wrapped, L, ncell)
        sys.cl.cell_of[i] = cell_idx
        sys.cl.next[i] = sys.cl.head[cell_idx]
        sys.cl.head[cell_idx] = i
    end
    
    return nothing
end

# Note: Since Molecules.jl is included in MC module, types from other included files
# are in the same namespace. We reference them directly without imports.

"""
    molecular_local_energy(site_i::Int, sys::MolecularSystem, p::LJParams)::Float64

Compute local energy for site i (sum over neighbors within rc).
Excludes intramolecular interactions (sites from same molecule).
"""
function molecular_local_energy(site_i::Int, sys::MolecularSystem, p::LJParams)::Float64
    energy = 0.0
    L = sys.cl.L
    rc2 = p.rc2
    ncell = sys.cl.ncell
    pos = sys.site_pos
    dr = sys.scratch_dr
    
    # Get site i position
    pix = pos[1, site_i]
    piy = pos[2, site_i]
    piz = pos[3, site_i]
    
    # Get molecule index for site i
    mol_i = sys.site_to_molecule[site_i]
    
    # Compute cell index
    x_wrapped = pix - L * floor(pix / L)
    y_wrapped = piy - L * floor(piy / L)
    z_wrapped = piz - L * floor(piz / L)
    i_cell_idx = get_cell(x_wrapped, y_wrapped, z_wrapped, L, ncell)
    
    # Convert to 3D cell indices
    k = ((i_cell_idx - 1) % ncell) + 1
    j = (((i_cell_idx - 1) ÷ ncell) % ncell) + 1
    i_cell = ((i_cell_idx - 1) ÷ (ncell * ncell)) + 1
    
    # Check all 27 neighboring cells
    @inbounds for di in -1:1
        for dj in -1:1
            for dk in -1:1
                cell_i = ((i_cell - 1 + di + ncell) % ncell) + 1
                cell_j = ((j - 1 + dj + ncell) % ncell) + 1
                cell_k = ((k - 1 + dk + ncell) % ncell) + 1
                
                neighbor_cell = cell_index(cell_i, cell_j, cell_k, ncell)
                
                # Iterate through sites in this cell
                site_j = sys.cl.head[neighbor_cell]
                while site_j > 0
                    if site_j != site_i
                        # Check if intramolecular (same molecule)
                        mol_j = sys.site_to_molecule[site_j]
                        if mol_i != mol_j  # Only intermolecular interactions
                            # Compute distance vector
                            dr[1] = pos[1, site_j] - pix
                            dr[2] = pos[2, site_j] - piy
                            dr[3] = pos[3, site_j] - piz
                            
                            # Apply minimum image convention
                            minimum_image!(dr, L)
                            
                            r2 = dr[1]*dr[1] + dr[2]*dr[2] + dr[3]*dr[3]
                            
                            if r2 < rc2 && r2 > 0.0
                                energy += lj_pair_u_from_r2(r2, p)
                            end
                        end
                    end
                    site_j = sys.cl.next[site_j]
                end
            end
        end
    end
    
    return energy
end

"""
    molecular_total_energy(sys::MolecularSystem, p::LJParams)::Float64

Compute total intermolecular energy (excludes intramolecular terms).
"""
function molecular_total_energy(sys::MolecularSystem, p::LJParams)::Float64
    energy = 0.0
    N = sys.n_sites_total
    L = sys.cl.L
    rc2 = p.rc2
    pos = sys.site_pos
    
    @inbounds for i in 1:N
        mol_i = sys.site_to_molecule[i]
        for j in (i+1):N
            mol_j = sys.site_to_molecule[j]
            
            # Skip intramolecular interactions
            if mol_i == mol_j
                continue
            end
            
            # Compute distance vector
            dr_x = pos[1, j] - pos[1, i]
            dr_y = pos[2, j] - pos[2, i]
            dr_z = pos[3, j] - pos[3, i]
            
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
                energy += lj_pair_u_from_r2(r2, p)
            end
        end
    end
    
    return energy
end

"""
    molecule_translation_trial!(sys::MolecularSystem, mol_idx::Int, p::LJParams, max_disp::Float64)::Bool

Perform translation trial move for molecule mol_idx.
Returns true if accepted, false if rejected.
"""
function molecule_translation_trial!(sys::MolecularSystem, mol_idx::Int, p::LJParams, max_disp::Float64)::Bool
    mol = sys.molecules[mol_idx]
    template = sys.templates[mol.template_idx]
    L = sys.cl.L
    
    # Store old COM
    com_old = copy(mol.com)
    
    # Compute old energy (sum over all sites in molecule)
    Eold = 0.0
    site_start = sys.n_atoms
    for m in 1:(mol_idx-1)
        site_start += sys.templates[sys.molecules[m].template_idx].n_sites
    end
    for site_local_idx in 1:template.n_sites
        site_idx = site_start + site_local_idx
        Eold += molecular_local_energy(site_idx, sys, p)
    end
    
    # Generate trial displacement
    dx = (rand(sys.rng) - 0.5) * 2.0 * max_disp
    dy = (rand(sys.rng) - 0.5) * 2.0 * max_disp
    dz = (rand(sys.rng) - 0.5) * 2.0 * max_disp
    
    # Apply trial move
    mol.com[1] = com_old[1] + dx
    mol.com[2] = com_old[2] + dy
    mol.com[3] = com_old[3] + dz
    
    # Wrap COM
    wrap!(mol.com, L)
    
    # Update site positions
    update_site_positions!(sys)
    
    # Compute new energy
    Enew = 0.0
    for site_local_idx in 1:template.n_sites
        site_idx = site_start + site_local_idx
        Enew += molecular_local_energy(site_idx, sys, p)
    end
    
    # Metropolis acceptance
    ΔE = Enew - Eold
    accepted = false
    if ΔE <= 0.0 || rand(sys.rng) < exp(-p.β * ΔE)
        accepted = true
    else
        # Reject: restore old COM
        copyto!(mol.com, com_old)
        update_site_positions!(sys)
    end
    
    return accepted
end

"""
    molecule_rotation_trial!(sys::MolecularSystem, mol_idx::Int, p::LJParams)::Bool

Perform rotation trial move for molecule mol_idx.
Returns true if accepted, false if rejected.
"""
function molecule_rotation_trial!(sys::MolecularSystem, mol_idx::Int, p::LJParams)::Bool
    mol = sys.molecules[mol_idx]
    template = sys.templates[mol.template_idx]
    L = sys.cl.L
    
    # Store old quaternion
    quat_old = copy(mol.quat)
    
    # Compute old energy
    Eold = 0.0
    site_start = sys.n_atoms
    for m in 1:(mol_idx-1)
        site_start += sys.templates[sys.molecules[m].template_idx].n_sites
    end
    for site_local_idx in 1:template.n_sites
        site_idx = site_start + site_local_idx
        Eold += molecular_local_energy(site_idx, sys, p)
    end
    
    # Generate new random quaternion (uniform on SO(3))
    mol.quat = uniform_random_quaternion(sys.rng)
    
    # Update site positions
    update_site_positions!(sys)
    
    # Compute new energy
    Enew = 0.0
    for site_local_idx in 1:template.n_sites
        site_idx = site_start + site_local_idx
        Enew += molecular_local_energy(site_idx, sys, p)
    end
    
    # Metropolis acceptance (no Jacobian needed for uniform rotation)
    ΔE = Enew - Eold
    accepted = false
    if ΔE <= 0.0 || rand(sys.rng) < exp(-p.β * ΔE)
        accepted = true
    else
        # Reject: restore old quaternion
        copyto!(mol.quat, quat_old)
        update_site_positions!(sys)
    end
    
    return accepted
end
