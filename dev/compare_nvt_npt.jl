"""
Compare NVT and NPT simulation results.
Checks that NPT ⟨ρ⟩ matches NVT ρ_target within combined error.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Helper function for robust CSV parsing
function parse_csv_summary(filepath::String, required_fields::Vector{String})
    if !isfile(filepath)
        error("File not found: $filepath")
    end
    
    # Read file as text
    lines = String[]
    open(filepath, "r") do io
        for line in eachline(io)
            stripped = strip(line)
            if !isempty(stripped)
                push!(lines, stripped)
            end
        end
    end
    
    if length(lines) < 2
        error("File $filepath has fewer than 2 non-empty lines (need header + data)")
    end
    
    header_line = lines[1]
    data_line = lines[2]
    
    # Split on comma
    header_fields = split(header_line, ',')
    data_fields = split(data_line, ',')
    
    if length(header_fields) != length(data_fields)
        error("Header and data line have different number of fields in $filepath")
    end
    
    # Build Dict mapping column name => index
    col_dict = Dict{String, Int}()
    for (idx, col_name) in enumerate(header_fields)
        col_dict[strip(col_name)] = idx
    end
    
    # Check for required fields
    missing_fields = String[]
    for field in required_fields
        if !haskey(col_dict, field)
            push!(missing_fields, field)
        end
    end
    
    if !isempty(missing_fields)
        available = sort(collect(keys(col_dict)))
        error("Required fields missing in $filepath:\n  Missing: $(join(missing_fields, ", "))\n  Available: $(join(available, ", "))")
    end
    
    # Parse fields
    function get_value(field_name::String, default_type::Type{T}) where T
        idx = col_dict[field_name]
        val_str = strip(data_fields[idx])
        if T == Bool
            if val_str == "true"
                return true
            elseif val_str == "false"
                return false
            else
                error("Cannot parse '$val_str' as Bool for field '$field_name'")
            end
        else
            return parse(T, val_str)
        end
    end
    
    return (get_value, col_dict)
end

results_dir = joinpath(@__DIR__, "results")
nvt_summary_file = joinpath(@__DIR__, "results", "nvt_summary.csv")
npt_summary_file = joinpath(@__DIR__, "results", "npt_summary.csv")

# Read NVT summary
get_val_nvt, _ = parse_csv_summary(nvt_summary_file, ["rho"])
nvt_rho_target = get_val_nvt("rho", Float64)

# Read NPT summary
get_val_npt, _ = parse_csv_summary(npt_summary_file, ["rho_mean", "rho_stderr"])
npt_rho_mean = get_val_npt("rho_mean", Float64)
npt_rho_stderr = get_val_npt("rho_stderr", Float64)

# Compute comparison
delta = npt_rho_mean - nvt_rho_target
z_score = delta / npt_rho_stderr

println("=" ^ 80)
println("NVT-NPT Comparison")
println("=" ^ 80)
println()
println("NVT target density: $nvt_rho_target")
println("NPT mean density:   $npt_rho_mean ± $npt_rho_stderr")
println()
println("Difference:")
println("  delta = $delta")
println("  z-score = $z_score")
println()

# Write comparison file
compare_file = joinpath(@__DIR__, "results", "compare.txt")
open(compare_file, "w") do io
    println(io, "NVT-NPT Comparison")
    println(io, "=" ^ 80)
    println(io)
    println(io, "NVT target density: $nvt_rho_target")
    println(io, "NPT mean density:   $npt_rho_mean ± $npt_rho_stderr")
    println(io)
    println(io, "Difference:")
    println(io, "  delta = $delta")
    println(io, "  z-score = $z_score")
    println(io)
    println(io, "Interpretation:")
    if abs(z_score) <= 2.0
        println(io, "  ✓ Agreement within 2σ (|z-score| ≤ 2.0)")
    else
        println(io, "  ⚠ Disagreement: |z-score| > 2.0")
    end
end

println("Wrote comparison to: $compare_file")
println()

if abs(z_score) <= 2.0
    println("✓ Agreement within 2σ (|z-score| ≤ 2.0)")
else
    println("⚠ Disagreement: |z-score| > 2.0")
end

println("=" ^ 80)
