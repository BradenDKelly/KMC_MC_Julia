"""
Fluctuation-based observables for NVT and NPT ensembles.
Computes heat capacity C_V (NVT) and isothermal compressibility κ_T (NPT).

Important distinction:
- Variance of the distribution (Var(U), Var(V)): Used in fluctuation formulas.
  Computed using standard sample variance from all samples. This gives the
  actual variance needed for C_V = Var(U)/(kT²) and κ_T = Var(V)/(kT⟨V⟩).
  
- Uncertainty estimates for means/variances: Used to estimate statistical errors.
  May use blocking, integrated autocorrelation time (IAT), or bootstrap methods.
  See BlockAverager.stderr() for uncertainty estimation of means.
"""

"""
    FluctuationAccumulator

Accumulator for computing mean via block averaging.
Stores block means internally. Used here only for computing the mean
(for numerical stability), not for variance estimation.
Variance is computed directly from samples using standard sample variance.
"""
mutable struct FluctuationAccumulator
    block_size::Int
    current_block::Vector{Float64}
    block_means::Vector{Float64}
    block_idx::Int
    
    function FluctuationAccumulator(block_size::Int)
        new(block_size, Float64[], Float64[], 0)
    end
end

"""
    push!(acc::FluctuationAccumulator, x::Float64)

Add a single observation to the accumulator.
Automatically forms blocks and stores block means.
"""
function Base.push!(acc::FluctuationAccumulator, x::Float64)
    push!(acc.current_block, x)
    if length(acc.current_block) >= acc.block_size
        block_mean = sum(acc.current_block) / Float64(length(acc.current_block))
        push!(acc.block_means, block_mean)
        empty!(acc.current_block)
    end
    return acc
end

"""
    finish!(acc::FluctuationAccumulator)

Finalize accumulation: if current_block has data, treat it as a partial block and include it.
"""
function finish!(acc::FluctuationAccumulator)
    if !isempty(acc.current_block)
        block_mean = sum(acc.current_block) / Float64(length(acc.current_block))
        push!(acc.block_means, block_mean)
        empty!(acc.current_block)
    end
    return acc
end

"""
    mean(acc::FluctuationAccumulator)::Float64

Compute mean from block means.
"""
function mean(acc::FluctuationAccumulator)::Float64
    if isempty(acc.block_means)
        return 0.0
    end
    return sum(acc.block_means) / Float64(length(acc.block_means))
end

# Note: variance() is not used directly; variance is computed from samples in heat_capacity_CV and compressibility_kappaT

"""
    heat_capacity_CV(U_samples::Vector{Float64}, T::Float64, block_size::Int=50)::Tuple{Float64, Float64}

Compute configurational heat capacity at constant volume from potential energy fluctuations.
U_samples: vector of potential energies (per configuration)
T: temperature
block_size: block size for block averaging the mean (default 50)

Returns: (C_V, mean_U)
In reduced LJ units (kB=1):
  C_V = Var(U) / (T^2)

Notes:
- This computes configurational C_V from potential energy fluctuations only.
  For full C_V (including kinetic energy), add 3N/2 (ideal gas contribution).
- Variance Var(U) is computed using standard sample variance from all samples.
  This gives the actual variance needed for the fluctuation formula.
- Block averaging (block_size) is used only for computing the mean (numerical stability),
  not for variance estimation. For uncertainty estimates, use BlockAverager.stderr().
"""
function heat_capacity_CV(U_samples::Vector{Float64}, T::Float64, block_size::Int=50)::Tuple{Float64, Float64}
    if isempty(U_samples)
        return (0.0, 0.0)
    end
    
    # Use block averaging for mean (for numerical stability), but compute variance directly from samples
    acc = FluctuationAccumulator(block_size)
    for u in U_samples
        push!(acc, u)
    end
    finish!(acc)
    
    mean_U = mean(acc)
    
    # Compute sample variance directly from all samples.
    # For fluctuation formulas, we need the actual variance of the distribution,
    # not a correlation-adjusted estimate. Standard sample variance gives Var(U).
    n = length(U_samples)
    if n < 2
        return (0.0, mean_U)
    end
    sum_sq_diff = 0.0
    for u in U_samples
        sum_sq_diff += (u - mean_U)^2
    end
    var_U = sum_sq_diff / Float64(n - 1)
    
    # C_V = Var(U) / (kB * T^2), with kB=1 in reduced units
    C_V = var_U / (T * T)
    
    return (C_V, mean_U)
end

"""
    compressibility_kappaT(V_samples::Vector{Float64}, T::Float64, block_size::Int=50)::Tuple{Float64, Float64}

Compute isothermal compressibility from volume fluctuations.
V_samples: vector of volumes (per configuration)
T: temperature
block_size: block size for block averaging the mean (default 50)

Returns: (κ_T, mean_V)
In reduced LJ units (kB=1):
  κ_T = Var(V) / (kB * T * ⟨V⟩)

Notes:
- Variance Var(V) is computed using standard sample variance from all samples.
  This gives the actual variance needed for the fluctuation formula.
- Block averaging (block_size) is used only for computing the mean (numerical stability),
  not for variance estimation. For uncertainty estimates, use BlockAverager.stderr().
"""
function compressibility_kappaT(V_samples::Vector{Float64}, T::Float64, block_size::Int=50)::Tuple{Float64, Float64}
    if isempty(V_samples)
        return (0.0, 0.0)
    end
    
    # Use block averaging for mean (for numerical stability), but compute variance directly from samples
    acc = FluctuationAccumulator(block_size)
    for v in V_samples
        push!(acc, v)
    end
    finish!(acc)
    
    mean_V = mean(acc)
    
    # Compute sample variance directly from all samples.
    # For fluctuation formulas, we need the actual variance of the distribution,
    # not a correlation-adjusted estimate. Standard sample variance gives Var(V).
    n = length(V_samples)
    if n < 2
        return (0.0, mean_V)
    end
    sum_sq_diff = 0.0
    for v in V_samples
        sum_sq_diff += (v - mean_V)^2
    end
    var_V = sum_sq_diff / Float64(n - 1)
    
    # κ_T = Var(V) / (kB * T * ⟨V⟩), with kB=1 in reduced units
    if mean_V > 0.0
        kappa_T = var_V / (T * mean_V)
    else
        kappa_T = 0.0
    end
    
    return (kappa_T, mean_V)
end
