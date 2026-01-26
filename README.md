# MolSim

A comprehensive molecular simulation package for Lennard-Jones systems supporting Monte Carlo (MC), Equilibrium Kinetic Monte Carlo (eKMC), and Molecular Dynamics (MD) methods.

## Overview

MolSim provides implementations of three complementary simulation methods for studying Lennard-Jones fluids:

- **Metropolis Monte Carlo (MC)**: Traditional rejection-based MC with importance sampling
- **Equilibrium Kinetic Monte Carlo (eKMC)**: Rejection-free MC using time-weighted averages (Tan et al. method)
- **Molecular Dynamics (MD)**: Velocity Verlet integrator with thermostat and barostat (independent verifier)

All three methods support **NVT** (canonical) and **NPT** (isothermal-isobaric) ensembles, with thermodynamic consistency validation between methods.

## Core Features

### Ensembles

- **NVT (Canonical)**: Constant number of particles (N), volume (V), and temperature (T)
  - Particle displacement moves (MC/eKMC)
  - Velocity rescaling thermostat (MD)
  - Energy, pressure, and chemical potential calculations

- **NPT (Isothermal-Isobaric)**: Constant N, pressure (P), and T
  - Volume change moves (MC/eKMC/MD)
  - Adaptive acceptance tuning for optimal move sizes
  - Density, energy, and pressure fluctuations
  - Chemical potential with volume terms (Equation A.132 for eKMC)

### Simulation Methods

#### Metropolis Monte Carlo (MC)
- Particle displacement moves with Metropolis acceptance criterion
- Volume change moves for NPT ensemble
- Adaptive acceptance tuning (default 45% acceptance rate)
- Long-range corrections (LRC) for truncated potentials
- Impulsive pressure correction for shifted potentials
- Widom insertion method for chemical potential

#### Equilibrium Kinetic Monte Carlo (eKMC)
- Rejection-free particle moves using mobility-based selection
- Tan et al. auxiliary pressure method for NPT volume moves
- Time-weighted averages for thermodynamic observables
- Chemical potential from total rate (R) accumulation
- Efficient pair energy storage and updates
- Overlap handling with energy capping

#### Molecular Dynamics (MD)
- Velocity Verlet integrator
- Velocity rescaling thermostat (NVT)
- MC-style volume change moves (NPT)
- Independent energy, force, and pressure calculations (serves as external verifier)
- Momentum conservation
- Periodic boundary conditions

### Advanced Features

#### Grand-Canonical Monte Carlo (GCMC) with CBMC
- Configurational Bias Monte Carlo for rigid molecules
- Single-site and multi-site molecules
- Insertion/deletion moves with Rosenbluth weights
- Activity-based fugacity control

#### Multicomponent Systems
- Multiple species with different LJ parameters
- Species-resolved observables
- Mixture RDF calculations
- Widom insertion per species

#### Analysis Tools
- **Radial Distribution Function (RDF)**: g(r) calculation and plotting
- **Block averaging**: Statistical error estimation
- **Fluctuation analysis**: Heat capacity (C_V) and isothermal compressibility (κ_T)
- **EOS comparison**: Validation against Nezbeda-Kolafa and Thol et al. equations of state

#### Potential Models
- Truncated Lennard-Jones (cutoff at r_c)
- Shifted Lennard-Jones (U(r_c) = 0, continuous)
- Cut-and-shift (with impulsive pressure correction)
- Long-range corrections (LRC) for truncated potentials

## Installation

```julia
using Pkg
Pkg.add("MolSim")  # Or clone from repository
```

## Quick Start

### NVT Monte Carlo

```julia
using MolSim
using Random

# Initialize system
N = 500
ρ = 0.8
T = 1.0
rc = 2.5
p, st = MC.init_simple(N=N, ρ=ρ, T=T, rc=rc, max_disp=0.1, seed=12345)

# Run NVT simulation
obs = MC.run_nvt!(p, st; 
    burnin_sweeps=5000, 
    prod_sweeps=20000, 
    sample_every=10)

# Access results
println("Energy per particle: ", obs.U_mean / N, " ± ", obs.U_stderr / N)
println("Pressure: ", obs.P_mean, " ± ", obs.P_stderr)
```

### NPT Monte Carlo

```julia
# Run NPT simulation with adaptive tuning
obs_npt = MC.run_npt!(p, st; 
    P=1.0,
    burnin_sweeps=5000,
    prod_sweeps=20000,
    sample_every=10,
    vol_move_every=20,
    target_acceptance=0.45)

println("Density: ", obs_npt.rho_mean, " ± ", obs_npt.rho_stderr)
println("Energy: ", obs_npt.U_mean / N, " ± ", obs_npt.U_stderr / N)
```

### Equilibrium Kinetic Monte Carlo (NVT)

```julia
# Initialize eKMC state
ekst = MC.init_ekmc_state(st, p)
acc_mu = MC.ChemicalPotentialAccumulator()

# Run eKMC simulation
obs_ekmc = MC.run_ekmc!(ekst, p, acc_mu;
    burnin_events=1000000,
    prod_events=5000000,
    sample_every=1000)

# Chemical potential
μ_ex = MC.mu_ex(acc_mu, T)
println("Excess chemical potential: ", μ_ex)
```

### NPT eKMC

```julia
# Run NPT eKMC (Tan et al. method)
obs_ekmc_npt = MC.run_ekmc_npt!(ekst, p, acc_mu;
    P=1.0,
    burnin_events=10000000,
    prod_events=5000000,
    sample_every=1000,
    vol_move_every=100)

# NPT chemical potential (Equation A.132)
μ_ex_npt = MC.mu_ex_npt(acc_mu_npt, T)
```

### Molecular Dynamics

```julia
# Initialize MD state
st_md = MC.init_md_state(st.pos, T, Random.Xoshiro(12345))
st_md.L = st.L
st_md.types = st.types

# NVT MD
MC.run_md_nvt!(st_md, p; 
    nsteps=10000, 
    dt=0.001, 
    T=T, 
    thermostat_every=10)

# NPT MD
MC.run_md_npt!(st_md, p;
    nsteps=10000,
    dt=0.001,
    T=T,
    P=1.0,
    thermostat_every=10,
    vol_move_every=20,
    max_dlnV=0.01)
```

### Radial Distribution Function

```julia
using MolSim.Analysis

# Collect snapshots during simulation
snapshots = []
# ... during simulation loop ...
push!(snapshots, copy(st.pos))

# Compute RDF
r, g_r = Analysis.compute_rdf(snapshots, st.L, nbins=200, rmax=st.L/2)

# Plot (requires Plots.jl)
using Plots
plot(r, g_r, xlabel="r / σ", ylabel="g(r)", title="Radial Distribution Function")
```

## Validation Scripts

The `scripts/` directory contains validation and comparison scripts:

- **`validate_nvt.jl`**: Compare MC and eKMC NVT results
- **`validate_npt_consistency.jl`**: Validate NPT eKMC against MC and check NVT-NPT consistency
- **`validate_md.jl`**: Compare MC, eKMC, and MD for both NVT and NPT, including RDF comparison
- **`validate_lj_statepoints.jl`**: Validate against literature state points

Run validation scripts:
```bash
julia --project=. scripts/validate_md.jl
```

## Testing

### Fast Tests (Default)

By default, `Pkg.test()` runs fast, deterministic tests:

```bash
julia --project -e "using Pkg; Pkg.test()"
```

### Slow Tests (Long-Run Ensemble Convergence)

Slow tests (e.g., NPT–NVT density consistency, fluctuation analysis) are skipped by default. To enable them:

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

### Comprehensive Unit Tests

The package includes extensive unit tests:

- **`test_md_comprehensive.jl`**: MD force consistency, energy conservation, temperature/pressure control, momentum conservation, PBC
- **`test_ekmc_npt_comprehensive.jl`**: NPT eKMC volume move consistency, pressure calculations, time-weighted averages, chemical potential
- **`test_md_barostat.jl`**: MD NPT volume move stability and correctness

Run specific test files:
```bash
julia --project=. test/test_md_comprehensive.jl
julia --project=. test/test_ekmc_npt_comprehensive.jl
```

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

## Key Algorithms

### eKMC NPT Volume Moves (Tan et al. Method)

The NPT eKMC implementation uses the auxiliary pressure method from Tan et al.:
- Compare current pressure with auxiliary pressure
- Deterministic volume change based on pressure difference
- Rebuild pair energies, mobilities, and total rate after volume change
- Time-weighted averages for observables

### Adaptive Acceptance Tuning

MC simulations automatically tune move sizes during equilibration to achieve target acceptance rates:
- Default: 45% for both particle and volume moves
- Adjusts `max_disp` (particle moves) and `max_dlnV` (volume moves)
- Only active during burn-in; parameters fixed during production

### Chemical Potential

- **MC**: Widom insertion method
- **eKMC NVT**: `μ_ex = -T * ln(⟨R/N⟩)` where R is total rate
- **eKMC NPT**: `μ_ex = -T * ln(⟨R*V/N⟩)` (Equation A.132, includes volume terms)

## Thermodynamic Consistency

The package validates thermodynamic consistency:
- NVT and NPT give same density at matching pressure
- MC, eKMC, and MD agree on energy, pressure, and density
- RDFs match across all three methods
- Pressure from virial equation matches thermodynamic pressure

## References

- Tan, Z., et al. "On the consistency of NVT, NPT, μVT and Gibbs ensembles in the framework of Kinetic Monte Carlo." *Chemical Engineering Journal* (2020).
- Frenkel, D. & Smit, B. *Understanding Molecular Simulation* (Academic Press, 2002).

## License

[Add your license here]
