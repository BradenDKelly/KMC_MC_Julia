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
