# Reaction Ensemble Monte Carlo (RxMC/REMC)

This document describes the Reaction Ensemble Monte Carlo implementation in MolSim.jl.

## Overview

The Reaction Ensemble Monte Carlo (REMC/RxMC) method samples equilibrium compositions in systems with chemical reactions at fixed temperature and either fixed volume (NVT) or fixed pressure (NPT). This implementation supports single-site atomic Lennard-Jones species.

## Reaction Specification

Reactions are defined using the `MolSim.MC.Reaction` struct:

```julia
struct Reaction
    label::String             # Human-readable label, e.g., "2A ⇌ B"
    stoichiometry::Vector{Int} # Stoichiometric coefficients [nu_1, nu_2, ...]. Negative for reactants, positive for products.
    logK::Float64             # log of equilibrium constant (deprecated for Table 3.2, set to 0.0)
    logq::Union{Vector{Float64}, Nothing}  # log(q/λ³) per species, or nothing
end
```

For example, the reaction `2A ⇌ B` would be defined with `stoichiometry = [-2, 1]` (assuming species 1 is A and species 2 is B).

The `logq` parameter specifies the ideal-gas partition function factors. For species `i`, `logq[i] = log(q_i/λ³)` where `q_i` is the single-particle partition function and `λ` is the de Broglie wavelength. In reduced LJ units, these are typically treated as constants.

## Acceptance Criterion

The acceptance criterion for a reaction move in the NVT or NPT ensemble is based on the standard Metropolis criterion, incorporating the change in potential energy, volume factors, ideal-gas partition function terms, and combinatorial factors.

For a general reaction `Σ_i ν_i S_i = 0` (where `ν_i` is the stoichiometric coefficient for species `i`, negative for reactants and positive for products), the acceptance probability uses the log acceptance ratio:

```
ln_acc = -β * ΔU
       + ΔN * log(V)
       + Σ_i ν_i * log(q_i/λ³)
       + Σ_i [log(N_i!) - log((N_i + ν_i)!)]
```

where:
- `β = 1/(kT)` is the inverse temperature (reduced units: k=1)
- `ΔU` is the change in potential energy
- `ΔN = Σ_i ν_i` is the change in total particle number
- `V` is the system volume
- `N_i` is the number of particles of species `i` before the move
- `ν_i` is the stoichiometric coefficient for species `i`
- `q_i/λ³` is the ideal-gas partition function per volume for species `i`

The factorial terms are computed using `loggamma` for numerical stability: `log(n!) = loggamma(n+1)`.

Then accept with probability `min(1, exp(ln_acc))`.

### Alchemical Conversion (ΔN=0 reactions)

For reactions where `ΔN = 0` (e.g., `A ⇌ B`, `A ⇌ C`), the implementation uses alchemical conversion:
- A particle of the reactant species is selected uniformly at random
- Its type is changed to the product species (or vice versa for reverse moves)
- The energy change is computed via the standard energy evaluation

This is more efficient than insertion/deletion moves and maintains the same equilibrium distribution.

### Insertion/Deletion (ΔN≠0 reactions)

For reactions where `ΔN ≠ 0` (e.g., `A ⇌ 2D`, `A ⇌ D+E`), the implementation uses insertion/deletion:
- Reactant particles are deleted uniformly at random
- Product particles are inserted at random positions in the box
- The cell list is rebuilt after the move

## Integration into Simulation

### NVT Ensemble

The `MolSim.MC.sweep_with_reactions!` function allows for integrating reaction moves into a standard NVT simulation. It performs `N` particle translation moves (via `MolSim.MC.sweep!`) and then, with a given probability `p_reaction`, attempts one reaction move (either forward or reverse, chosen stochastically).

### NPT Ensemble

The `MolSim.MC.sweep_npt_with_reactions!` function integrates reaction moves into NPT simulations:
- Performs `N` particle translation moves
- Optionally attempts a volume move (if `do_volume_move=true`)
- Optionally attempts a reaction move (with probability `p_reaction`)

Volume moves use the standard NPT acceptance criterion with the Jacobian term. Reaction moves use the acceptance criterion described above.

## Table 3.2 Benchmarks

The `scripts/run_table3_2_reaction_ensemble.jl` script reproduces the benchmarks from Table 3.2 of Kelly Braden's thesis.

### Running the Script

```bash
julia --project scripts/run_table3_2_reaction_ensemble.jl [seed] [sweeps_equil] [sweeps_prod] [output_dir]
```

Default parameters:
- `seed = 12345`
- `sweeps_equil = 50000`
- `sweeps_prod = 200000`
- `output_dir = "results"`

### Statepoint

The script runs at:
- Temperature: T = 2.0 (reduced units)
- Pressure: P = 5.0 (reduced units)
- Cutoff: rc = 2.5
- LJ model: shifted (cut-and-shifted)
- Mixing rule: Lorentz-Berthelot (σ_ab = (σ_a + σ_b)/2, ε_ab = sqrt(ε_a * ε_b))
- Initial composition: N_A = 400, others = 0
- Initial density guess: ρ = 0.75

### Species Definitions (Table 3.1)

- A: σ=1.0, ε=1.0, q=0.002
- B: σ=1.0, ε=1.0, q=0.002
- C: σ=1.1, ε=0.9, q=0.002
- D: σ=1.0, ε=1.0, q=0.02
- E: σ=1.1, ε=0.9, q=0.02
- F: σ=1.0, ε=1.0, q=0.02

### Reactions Tested

The script runs the following reactions and compares results to Table 3.2 targets:

- `A ⇌ B`
- `A ⇌ C`
- `A ⇌ 2D`
- `A ⇌ 2E`
- `A ⇌ D+F`
- `A ⇌ D+E`

### Output

The script prints:
1. **Statepoint echo**: Structured output containing all simulation parameters
2. **Results table**: Mean compositions, densities, and uncertainties
3. **Comparison table**: Measured values vs. Table 3.2 targets with percent error

Additionally, JSON files are written to the output directory for each reaction:
- Filename: `table3_2_<reaction>_seed<seed>.json`
- Contains: metadata, statepoint, species parameters, results (means, stderrs, acceptance rates), and targets

### JSON Schema

Each JSON file contains:

```json
{
  "meta": {
    "reaction": "A⇌B",
    "seed": 12345,
    "timestamp": "..."
  },
  "statepoint": {
    "T": 2.0,
    "P": 5.0,
    "rc": 2.5,
    "lj_model": "shifted",
    "use_lrc": false,
    "mixing_rule": "Lorentz-Berthelot"
  },
  "species": {
    "A": {"sigma": 1.0, "epsilon": 1.0, "q": 0.002, "logq": -6.2146...},
    "B": {"sigma": 1.0, "epsilon": 1.0, "q": 0.002, "logq": -6.2146...}
  },
  "reaction": {
    "stoichiometry": {"A": -1, "B": 1}
  },
  "moves": {
    "max_disp": 0.1,
    "max_dlnV": 0.01,
    "vol_move_every": 10,
    "p_reaction": 0.1
  },
  "run_lengths": {
    "sweeps_equil": 50000,
    "sweeps_prod": 200000,
    "sample_every": 10,
    "block_size": 50
  },
  "results": {
    "means": {"N_A": 200.5, "N_B": 199.5, "V": 523.4, "rho_total": 0.764},
    "stderrs": {"N_A": 0.2, "N_B": 0.2, "V": 0.5, "rho_total": 0.001},
    "acceptance_rates": {
      "translation": 0.45,
      "volume": 0.30,
      "reaction": 0.15
    },
    "targets": {"N_A": 200.0, "N_B": 200.0, "rho_total": 0.766}
  }
}
```

## Tests

The implementation is validated by tests in `test/test_table3_2_remc.jl`:

1. **Invariants and finiteness**: Verifies that counts remain non-negative, energy stays finite, and volume remains positive for both ΔN=0 and ΔN≠0 reactions.

2. **A⇌B symmetry**: For the symmetric A⇌B reaction (identical LJ parameters and q values), verifies that composition drifts from all-A to a mixed composition.

3. **Table 3.2 match (long test)**: Gated behind `LONG_TESTS=1` environment variable. Runs a longer simulation and compares results to Table 3.2 targets with loose tolerance (30%).

To run long tests:
```bash
LONG_TESTS=1 julia --project test/test_table3_2_remc.jl
```

## Notes

- This implementation uses REMC/RxMC equilibrium sampling. It does not introduce physical time; reactions are accepted/rejected based on equilibrium criteria only.
- The implementation is currently limited to single-site atomic species. Extension to molecular systems with CBMC insertion/deletion is planned for future work.
- For Table 3.2 benchmarks, `logK` is set to 0.0; equilibrium is determined solely by the `logq` values and the acceptance criterion.
