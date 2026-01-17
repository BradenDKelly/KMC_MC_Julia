"""
Profile EXP-6 REMC sweeps for performance hotspots.

Example:
  julia --project=. "scripts/profile_exp6_remc.jl" --profile_sweeps 200 --equil 50 --p_reaction 1.0 --units reduced --T 40.0 --P 2225.6 --rc 2.25 --rho_init 3.5 --use_lrc true
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using Random
using Profile

const SPECIES = ["N2", "O2", "NO"]
const EXP6_PARAMS = Dict(
    "N2" => (rm=4.251, ε=75.0, α=13.474),
    "O2" => (rm=4.110, ε=75.0, α=13.117),
    "NO" => (rm=3.995, ε=117.1, α=12.08),
)

function parse_args()
    params = Dict(
        "seed" => 12345,
        "equil" => 50,
        "profile_sweeps" => 200,
        "p_reaction" => 1.0,
        "cutoff_model" => "truncated",
        "mixing_rule" => "lb",
        "rmin_factor" => 0.8,
        "rho_init" => 0.7,
        "units" => "reduced",
        "T" => 1.0,
        "P" => 1.0,
        "rc" => 2.5,
        "T_K" => 3000.0,
        "P_GPa" => 30.0,
        "rc_A" => 12.0,
        "N_total" => 400,
        "N2_init" => 100,
        "O2_init" => 100,
        "NO_init" => 200,
        "use_lrc" => true,
        "mu0_N2_kjmol" => 0.0,
        "mu0_O2_kjmol" => 0.0,
        "mu0_NO_kjmol" => 52.43,
        "deltaG0_kjmol" => NaN,
        "profile_out" => "",
    )
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--seed" && i < length(ARGS)
            params["seed"] = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--equil" && i < length(ARGS)
            params["equil"] = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--profile_sweeps" && i < length(ARGS)
            params["profile_sweeps"] = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--p_reaction" && i < length(ARGS)
            params["p_reaction"] = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--cutoff_model" && i < length(ARGS)
            params["cutoff_model"] = ARGS[i+1]; i += 2
        elseif arg == "--mixing_rule" && i < length(ARGS)
            params["mixing_rule"] = ARGS[i+1]; i += 2
        elseif arg == "--rmin_factor" && i < length(ARGS)
            params["rmin_factor"] = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--rho_init" && i < length(ARGS)
            params["rho_init"] = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--units" && i < length(ARGS)
            params["units"] = ARGS[i+1]; i += 2
        elseif arg == "--T" && i < length(ARGS)
            params["T"] = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--P" && i < length(ARGS)
            params["P"] = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--rc" && i < length(ARGS)
            params["rc"] = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--T_K" && i < length(ARGS)
            params["T_K"] = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--P_GPa" && i < length(ARGS)
            params["P_GPa"] = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--rc_A" && i < length(ARGS)
            params["rc_A"] = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--N_total" && i < length(ARGS)
            params["N_total"] = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--N2_init" && i < length(ARGS)
            params["N2_init"] = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--O2_init" && i < length(ARGS)
            params["O2_init"] = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--NO_init" && i < length(ARGS)
            params["NO_init"] = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--use_lrc" && i < length(ARGS)
            params["use_lrc"] = parse(Bool, ARGS[i+1]); i += 2
        elseif arg == "--mu0_N2_kjmol" && i < length(ARGS)
            params["mu0_N2_kjmol"] = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--mu0_O2_kjmol" && i < length(ARGS)
            params["mu0_O2_kjmol"] = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--mu0_NO_kjmol" && i < length(ARGS)
            params["mu0_NO_kjmol"] = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--deltaG0_kjmol" && i < length(ARGS)
            params["deltaG0_kjmol"] = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--profile_out" && i < length(ARGS)
            params["profile_out"] = ARGS[i+1]; i += 2
        else
            error("Unknown or incomplete argument: $arg")
        end
    end
    return params
end

function build_params()
    rm_types = [EXP6_PARAMS[s].rm for s in SPECIES]
    ϵ_types = [EXP6_PARAMS[s].ε for s in SPECIES]
    α_types = [EXP6_PARAMS[s].α for s in SPECIES]
    return (rm_types, ϵ_types, α_types)
end

function build_initial_types(N_total, n2, o2, no)
    @assert n2 + o2 + no == N_total "Initial counts must sum to N_total"
    types = Vector{Int}(undef, N_total)
    idx = 1
    for _ in 1:n2
        types[idx] = 1; idx += 1
    end
    for _ in 1:o2
        types[idx] = 2; idx += 1
    end
    for _ in 1:no
        types[idx] = 3; idx += 1
    end
    return types
end

function main()
    args = parse_args()
    rm_types_phys, ϵ_types_phys, α_types_phys = build_params()
    N_total = args["N_total"]
    types = build_initial_types(N_total, args["N2_init"], args["O2_init"], args["NO_init"])
    cutoff_model = Symbol(args["cutoff_model"])
    mixing_rule = Symbol(args["mixing_rule"])
    
    # Unit conversion (reference species: N2)
    sigma_ref = EXP6_PARAMS["N2"].rm
    eps_ref = EXP6_PARAMS["N2"].ε
    kB = 1.380649e-23
    angstrom = 1.0e-10
    gpa = 1.0e9
    sigma_ref_m = sigma_ref * angstrom
    eps_ref_J = eps_ref * kB
    
    rm_star = rm_types_phys ./ sigma_ref
    eps_star = ϵ_types_phys ./ eps_ref
    alpha_star = copy(α_types_phys)
    
    units_mode = lowercase(args["units"])
    if !(units_mode in ("reduced", "physical"))
        error("Invalid --units value: $(args["units"]). Must be reduced or physical.")
    end
    
    if units_mode == "reduced"
        T_used = args["T"]
        P_used = args["P"]
        rc_used = args["rc"]
    else
        T_used = args["T_K"] / eps_ref
        P_used = (args["P_GPa"] * gpa) * (sigma_ref_m^3) / eps_ref_J
        rc_used = args["rc_A"] / sigma_ref
        println("WARNING: physical units mode assumes P in GPa, rm in Å, ε in K.")
    end
    
    p, st = MolSim.MC.init_fcc_exp6(N=N_total, ρ=args["rho_init"], T=T_used, rc=rc_used,
                                    seed=args["seed"], types=types,
                                    rm_types=rm_star, ϵ_types=eps_star, α_types=alpha_star,
                                    cutoff_model=cutoff_model, mixing_rule=mixing_rule,
                                    rmin_factor=args["rmin_factor"], use_lrc=args["use_lrc"])
    
    # Chemistry term for reaction
    mu0_phys_kjmol = Dict(
        "N2" => args["mu0_N2_kjmol"],
        "O2" => args["mu0_O2_kjmol"],
        "NO" => args["mu0_NO_kjmol"],
    )
    avogadro = 6.02214076e23
    if isfinite(args["deltaG0_kjmol"])
        delta_g0_star = ((args["deltaG0_kjmol"] * 1000.0) / avogadro) / eps_ref_J
        logGamma = -(1.0 / T_used) * delta_g0_star
    else
        mu0_star = [((mu0_phys_kjmol[s] * 1000.0) / avogadro) / eps_ref_J for s in SPECIES]
        delta_g0_star = -mu0_star[1] - mu0_star[2] + 2.0 * mu0_star[3]
        logGamma = -(1.0 / T_used) * delta_g0_star
    end
    reaction = MolSim.MC.Reaction("N2+O2⇌2NO", [-1, -1, 2], 0.0; logq=nothing, logGamma=logGamma)
    
    # Warmup
    for _ in 1:args["equil"]
        MolSim.MC.sweep_npt_with_reactions!(st, p, reaction;
            Pext=P_used, max_dlnV=0.01, p_reaction=args["p_reaction"],
            do_volume_move=true, rebuild_every=-1, insertion_mode=:standard_remc,
            use_alchemical=false)
    end
    
    # Profile + timing
    Profile.clear()
    t_start = time_ns()
    @profile begin
        for _ in 1:args["profile_sweeps"]
            MolSim.MC.sweep_npt_with_reactions!(st, p, reaction;
                Pext=P_used, max_dlnV=0.01, p_reaction=args["p_reaction"],
                do_volume_move=true, rebuild_every=-1, insertion_mode=:standard_remc,
                use_alchemical=false)
        end
    end
    t_elapsed = (time_ns() - t_start) / 1e9
    sweeps = args["profile_sweeps"]
    println("Timing: $(round(t_elapsed, digits=3)) s for $(sweeps) sweeps ($(round(sweeps / t_elapsed, digits=3)) sweeps/s)")
    
    if args["profile_out"] == ""
        Profile.print()
    else
        open(args["profile_out"], "w") do io
            Profile.print(io=io)
        end
        println("Profile written to $(args["profile_out"])")
    end
end

main()
