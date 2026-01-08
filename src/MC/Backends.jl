"""
Backend abstraction for MC simulations.
Supports multiple backends (CPU, GPU) with unified interface.
"""

"""
    AbstractBackend

Abstract type for simulation backends (CPU, GPU, etc.).
"""
abstract type AbstractBackend end

"""
    CPUBackend

CPU backend implementation using standard Julia arrays.
"""
struct CPUBackend <: AbstractBackend end

# Default CPU backend instance (singleton)
const CPU = CPUBackend()
