"""
Block averaging utility for computing means and standard errors from block means.
"""

"""
    BlockAverager

Block averager that accumulates values into blocks and computes statistics from block means.
"""
mutable struct BlockAverager
    block_size::Int
    block_means::Vector{Float64}
    current_block::Vector{Float64}
end

"""
    BlockAverager(block_size::Int)

Create a new block averager with given block size.
"""
function BlockAverager(block_size::Int)
    return BlockAverager(block_size, Float64[], Float64[])
end

"""
    push!(ba::BlockAverager, x::Float64)

Add a value to the block averager.
When a block is complete, its mean is computed and stored.
"""
function Base.push!(ba::BlockAverager, x::Float64)
    push!(ba.current_block, x)
    if length(ba.current_block) >= ba.block_size
        block_mean = sum(ba.current_block) / length(ba.current_block)
        push!(ba.block_means, block_mean)
        empty!(ba.current_block)
    end
    return ba
end

"""
    mean(ba::BlockAverager)::Float64

Compute the mean over all block means.
"""
function mean(ba::BlockAverager)::Float64
    if isempty(ba.block_means)
        if isempty(ba.current_block)
            return NaN
        else
            return sum(ba.current_block) / length(ba.current_block)
        end
    end
    n = length(ba.block_means)
    return sum(ba.block_means) / n
end

"""
    stderr(ba::BlockAverager)::Float64

Compute the standard error from block means.
Returns NaN if there are fewer than 2 blocks.
"""
function stderr(ba::BlockAverager)::Float64
    if length(ba.block_means) < 2
        return NaN
    end
    m = mean(ba)
    n = length(ba.block_means)
    sum_sq_diff = 0.0
    for x in ba.block_means
        diff = x - m
        sum_sq_diff += diff * diff
    end
    variance = sum_sq_diff / (n - 1)
    return sqrt(variance / n)
end
