# MolSim

Monte Carlo simulation package for Lennard-Jones systems.

## Testing

By default, `Pkg.test()` runs fast, deterministic tests:

```bash
julia --project -e "using Pkg; Pkg.test()"
```

### Slow Tests (Long-Run Ensemble Convergence)

Slow tests (e.g., NPT–NVT density consistency) are skipped by default. To enable them, set the `MOLSIM_SLOW_TESTS` environment variable:

**Windows PowerShell:**
```powershell
$env:MOLSIM_SLOW_TESTS="1"; julia --project -e "using Pkg; Pkg.test()"
```

**Windows cmd.exe:**
```cmd
set MOLSIM_SLOW_TESTS=1 && julia --project -e "using Pkg; Pkg.test()"
```

**Linux/macOS:**
```bash
MOLSIM_SLOW_TESTS=1 julia --project -e "using Pkg; Pkg.test()"
```

Slow tests require long MC runs and may take several minutes to complete.

## Grand-Canonical Monte Carlo (GCMC) with CBMC

The package supports grand-canonical μVT ensemble moves for rigid molecules using Configurational Bias Monte Carlo (CBMC).

### Usage

```julia
using MolSim

# Create molecule template (e.g., single-site or diatomic)
template = MolSim.MC.create_single_site_molecule_template()
# or
template = MolSim.MC.create_diatomic_molecule_template(bond_length=1.0)

# Initialize molecular system
atom_pos = zeros(Float64, 3, 0)  # No atoms
molecules = Vector{MolSim.MC.MoleculeState}()
templates = [template]
L = 10.0
rc = 2.5
sys = MolSim.MC.init_molecular_system(atom_pos, molecules, templates, L, rc, seed=12345)

# Set up parameters
T = 1.0
p = MolSim.MC.LJParams(1.0, 1.0, rc, rc*rc, 1.0/T, 0.1, false, 0.0, 0.0, :truncated, false, 0.0)
beta = 1.0 / T
z = 1.0  # Activity (fugacity-like parameter in reduced units)
k_trials = 10  # Number of CBMC trial configurations

# Attempt insertion
accepted = MolSim.MC.cbmc_insert_trial!(sys, 1, p; beta=beta, z=z, k_trials=k_trials)

# Attempt deletion
accepted = MolSim.MC.cbmc_delete_trial!(sys, 1, p; beta=beta, z=z, k_trials=k_trials)
```

### Activity Parameter `z`

The activity `z` is a fugacity-like parameter in reduced units (σ=ε=kB=1). It controls the equilibrium number of molecules:
- Large `z` → favors more molecules (insertions more likely)
- Small `z` → favors fewer molecules (deletions more likely)
- For ideal gas: equilibrium density ρ ≈ z

The acceptance probabilities (Frenkel-Smit formulation):
- **Insertion**: `A_ins = min(1, (z*V/(N+1)) * (W_ins/k_trials))` where `W_ins = Σ_j w_j` (sum over k_trials candidates), `w_j = exp(-β*ΔU_j)`
- **Deletion**: `A_del = min(1, (N/(z*V)) * (k_trials/W_del))` where `W_del = w_real + Σ_j w_j` (real molecule weight + k_trials-1 decoys)

The `k_trials` factors ensure detailed balance: insertion includes `1/k_trials` (selection probability), deletion includes `k_trials` (symmetric compensation).

### CBMC Algorithm

CBMC uses multi-try sampling to improve acceptance rates:

**Insertion**:
1. Generate `k_trials` independent candidate configurations (uniform COM in box, uniform quaternion in SO(3))
2. Compute weights `w_j = exp(-β*ΔU_j)` for each candidate
3. Compute Rosenbluth weight `W_ins = Σ_j w_j` (sum, not average)
4. Select candidate `j*` with probability `w_j*/W_ins` (categorical distribution)
5. Accept with probability `A_ins = min(1, (z*V/(N+1)) * (W_ins/k_trials))`

**Deletion**:
1. Select existing molecule uniformly (probability 1/N)
2. Compute `w_real = exp(-β*ΔU_real)` for the selected molecule
3. Generate `k_trials-1` decoy candidates (same distribution as insertion)
4. Compute Rosenbluth weight `W_del = w_real + Σ_j w_j` (sum over real + decoys)
5. Accept with probability `A_del = min(1, (N/(z*V)) * (k_trials/W_del))`

**Detailed Balance**: The formulas above ensure exact detailed balance between insertion and deletion moves. Uniform proposals (COM and quaternion) have no Jacobian factors.

### Running CBMC Tests

CBMC tests are included in the fast test suite:
```bash
julia --project -e "using Pkg; Pkg.test()"
```

The tests verify ΔU correctness, detailed balance, and single-site/diatomic molecule behavior.
