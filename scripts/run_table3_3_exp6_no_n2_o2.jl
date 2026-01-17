"""
Table 3.3-style EXP-6 REMC for N2/O2/NO (reduced units).

Reaction: N2 + O2 ⇌ 2 NO
Reduced units use N2 as reference (rm_ref, ε_ref).
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using Random
using Dates

const SPECIES = ["N2", "O2", "NO"]

const EXP6_PARAMS_PHYSICAL = Dict(
    "N2" => (rm=4.251, ε=75.0, α=13.474),
    "O2" => (rm=4.110, ε=75.0, α=13.117),
    "NO" => (rm=3.995, ε=117.1, α=12.08),
)

function parse_args()
    params = Dict(
        "ff" => "exp6",
        "seed" => 12345,
        "equil" => 10000,
        "prod" => 10000,
        "stride" => 10,
        "p_reaction" => 0.1,
        "rho_init" => 0.7,
        "T_phys" => 3000.0,
        "P_phys" => 30.0,
        "rc" => 2.5,
        "N_total" => 400,
        "N2_init" => 100,
        "O2_init" => 100,
        "NO_init" => 200,
        "cutoff_model" => "truncated",
        "mixing_rule" => "lb",
        "rmin_factor" => 0.2,
    )
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--seed" && i < length(ARGS)
            params["seed"] = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--ff" && i < length(ARGS)
            params["ff"] = ARGS[i+1]; i += 2
        elseif arg == "--equil" && i < length(ARGS)
            params["equil"] = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--prod" && i < length(ARGS)
            params["prod"] = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--stride" && i < length(ARGS)
            params["stride"] = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--p_reaction" && i < length(ARGS)
            params["p_reaction"] = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--rho_init" && i < length(ARGS)
            params["rho_init"] = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--T" && i < length(ARGS)
            params["T_phys"] = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--P" && i < length(ARGS)
            params["P_phys"] = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--rc" && i < length(ARGS)
            params["rc"] = parse(Float64, ARGS[i+1]); i += 2
        elseif arg == "--N_total" && i < length(ARGS)
            params["N_total"] = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--N2_init" && i < length(ARGS)
            params["N2_init"] = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--O2_init" && i < length(ARGS)
            params["O2_init"] = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--NO_init" && i < length(ARGS)
            params["NO_init"] = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--cutoff_model" && i < length(ARGS)
            params["cutoff_model"] = ARGS[i+1]; i += 2
        elseif arg == "--mixing_rule" && i < length(ARGS)
            params["mixing_rule"] = ARGS[i+1]; i += 2
        elseif arg == "--rmin_factor" && i < length(ARGS)
            params["rmin_factor"] = parse(Float64, ARGS[i+1]); i += 2
        else
            error("Unknown or incomplete argument: $arg")
        end
    end
    return params
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
    if lowercase(args["ff"]) != "exp6"
        error("This script supports only --ff exp6. Use LJ scripts for LJ runs.")
    end
    rm_phys = [EXP6_PARAMS_PHYSICAL[s].rm for s in SPECIES]
    ϵ_phys = [EXP6_PARAMS_PHYSICAL[s].ε for s in SPECIES]
    α_phys = [EXP6_PARAMS_PHYSICAL[s].α for s in SPECIES]
    
    rm_ref = EXP6_PARAMS_PHYSICAL["N2"].rm
    ϵ_ref = EXP6_PARAMS_PHYSICAL["N2"].ε
    rm_red, ϵ_red, α_red, T_red, P_red = MolSim.MC.exp6_to_reduced(
        rm_phys, ϵ_phys, α_phys;
        rm_ref_A=rm_ref, ϵ_ref_K=ϵ_ref,
        T_K=args["T_phys"], P_GPa=args["P_phys"]
    )
    
    N_total = args["N_total"]
    types = build_initial_types(N_total, args["N2_init"], args["O2_init"], args["NO_init"])
    cutoff_model = Symbol(args["cutoff_model"])
    mixing_rule = Symbol(args["mixing_rule"])
    
    p, st = MolSim.MC.init_fcc_exp6(N=N_total, ρ=args["rho_init"], T=T_red, rc=args["rc"],
                                    seed=args["seed"], types=types,
                                    rm_types=rm_red, ϵ_types=ϵ_red, α_types=α_red,
                                    cutoff_model=cutoff_model, mixing_rule=mixing_rule,
                                    rmin_factor=args["rmin_factor"])
    
    reaction = MolSim.MC.Reaction("N2+O2⇌2NO", [-1, -1, 2], 0.0; logq=[0.0, 0.0, 0.0])
    
    ts() = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
    println("[", ts(), "] " * "=" ^ 79)
    println("[", ts(), "] EXP-6 REMC N2/O2/NO (reduced units)")
    println("[", ts(), "] " * "=" ^ 79)
    println("[", ts(), "] Seed: $(args["seed"])")
    println("[", ts(), "] Equilibration sweeps: $(args["equil"])")
    println("[", ts(), "] Production sweeps: $(args["prod"])")
    println("[", ts(), "] Sample every: $(args["stride"])")
    println("[", ts(), "] p_reaction: $(args["p_reaction"])")
    println("[", ts(), "] Initial density guess: $(args["rho_init"])")
    println("[", ts(), "] T* = $(round(T_red, digits=6)) (T=$(args["T_phys"]) K)")
    println("[", ts(), "] P* = $(round(P_red, digits=6)) (P=$(args["P_phys"]) GPa)")
    println("[", ts(), "] rc = $(args["rc"]) (reduced)")
    println("[", ts(), "] Mixing rule: $(args["mixing_rule"])")
    println("[", ts(), "] Cutoff model: $(args["cutoff_model"])")
    println("[", ts(), "] rmin_factor: $(args["rmin_factor"])")
    println("[", ts(), "] Species parameters (reduced):")
    for (idx, s) in enumerate(SPECIES)
        println("[", ts(), "]   $s: rm=$(round(rm_red[idx], digits=6)), ε=$(round(ϵ_red[idx], digits=6)), α=$(round(α_red[idx], digits=6))")
    end
    println("[", ts(), "] Initial composition: N2=$(args["N2_init"]), O2=$(args["O2_init"]), NO=$(args["NO_init"])")
    println("[", ts(), "] " * "=" ^ 79)
    
    println("[", ts(), "] Equilibration: $(args["equil"]) sweeps...")
    for _ in 1:args["equil"]
        MolSim.MC.sweep_npt_with_reactions!(st, p, reaction;
            Pext=P_red, max_dlnV=0.01, p_reaction=args["p_reaction"],
            do_volume_move=true, rebuild_every=-1, insertion_mode=:standard_remc)
    end
    println("[", ts(), "] Production: $(args["prod"]) sweeps...")
    counts_sum = zeros(Float64, 3)
    n_samples = 0
    for sweep in 1:args["prod"]
        MolSim.MC.sweep_npt_with_reactions!(st, p, reaction;
            Pext=P_red, max_dlnV=0.01, p_reaction=args["p_reaction"],
            do_volume_move=true, rebuild_every=-1, insertion_mode=:standard_remc)
        if sweep % args["stride"] == 0
            counts = MolSim.MC.count_species(st, 3)
            counts_sum .+= counts
            n_samples += 1
        end
    end
    
    avg_counts = counts_sum ./ max(n_samples, 1)
    x_no = avg_counts[3] / sum(avg_counts)
    println("[", ts(), "] Results:")
    println("[", ts(), "]   N2=$(round(avg_counts[1], digits=3))")
    println("[", ts(), "]   O2=$(round(avg_counts[2], digits=3))")
    println("[", ts(), "]   NO=$(round(avg_counts[3], digits=3))")
    println("[", ts(), "]   x_NO=$(round(x_no, digits=6))")
end

main()
