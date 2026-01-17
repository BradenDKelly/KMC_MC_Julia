"""
Run NPT Reaction Ensemble MC for N2/O2/NO with exp-6 (Buckingham) force field.

Reaction: N2 + O2 ⇌ 2 NO
Units: reduced or physical (converted to reduced)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using Random
using Dates

const SPECIES = ["N2", "O2", "NO"]

const EXP6_PARAMS = Dict(
    "N2" => (rm=4.251, ε=75.0, α=13.474),
    "O2" => (rm=4.110, ε=75.0, α=13.117),
    "NO" => (rm=3.995, ε=117.1, α=12.08),
)

function parse_args()
    params = Dict(
        "seed" => 12345,
        "equil" => 10000,
        "prod" => 20000,
        "stride" => 10,
        "p_reaction" => 0.1,
        "cutoff_model" => "truncated",
        "mixing_rule" => "lb",
        "rmin_factor" => 0.2,
        "rho_init" => 0.7,
        "units" => "physical",
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
        "rmin_factor" => 0.8,
        "mu0_N2_kjmol" => 0.0,
        "mu0_O2_kjmol" => 0.0,
        "mu0_NO_kjmol" => 52.43,
        "deltaG0_kjmol" => NaN,
    )
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--seed" && i < length(ARGS)
            params["seed"] = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--equil" && i < length(ARGS)
            params["equil"] = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--prod" && i < length(ARGS)
            params["prod"] = parse(Int, ARGS[i+1]); i += 2
        elseif arg == "--stride" && i < length(ARGS)
            params["stride"] = parse(Int, ARGS[i+1]); i += 2
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
        # physical → reduced
        T_used = args["T_K"] / eps_ref
        P_used = (args["P_GPa"] * gpa) * (sigma_ref_m^3) / eps_ref_J
        rc_used = args["rc_A"] / sigma_ref
        println("WARNING: physical units mode assumes P in GPa, rm in Å, ε in K.")
    end
    
    println("UNITS_AUDIT:")
    println("  mode = $units_mode")
    println("  sigma_ref = $sigma_ref Å, eps_ref = $eps_ref K")
    println("  T_used = $T_used, P_used = $P_used, beta_used = $(1.0 / T_used)")
    println("  rc_used = $rc_used ($(units_mode == "reduced" ? "reduced" : "reduced from Å"))")
    println("  params (reduced):")
    for (idx, s) in enumerate(SPECIES)
        println("    $s: rm*=$(rm_star[idx]), eps*=$(eps_star[idx]), alpha=$(alpha_star[idx])")
    end
    
    p, st = MolSim.MC.init_fcc_exp6(N=N_total, ρ=args["rho_init"], T=T_used, rc=rc_used,
                                    seed=args["seed"], types=types,
                                    rm_types=rm_star, ϵ_types=eps_star, α_types=alpha_star,
                                    cutoff_model=cutoff_model, mixing_rule=mixing_rule,
                                    rmin_factor=args["rmin_factor"], use_lrc=args["use_lrc"])
    
    # Neighbor list audit (cell list)
    MolSim.MC.rebuild_cells!(st)
    if get(ENV, "EXP6_NEIGHBOR_DEBUG", "0") == "1"
        total_neighbors = 0
        rc2 = p.rc2
        L = st.L
        L_half = 0.5 * L
        pos = st.pos
        ncell = st.cl.ncell
        for i in 1:st.N
            i_cell_idx = st.cl.cell_of[i]
            k = ((i_cell_idx - 1) % ncell) + 1
            j = (((i_cell_idx - 1) ÷ ncell) % ncell) + 1
            i_cell = ((i_cell_idx - 1) ÷ (ncell * ncell)) + 1
            for di in -1:1
                for dj in -1:1
                    for dk in -1:1
                        cell_i = ((i_cell - 1 + di + ncell) % ncell) + 1
                        cell_j = ((j - 1 + dj + ncell) % ncell) + 1
                        cell_k = ((k - 1 + dk + ncell) % ncell) + 1
                        neighbor_cell = MolSim.MC.cell_index(cell_i, cell_j, cell_k, ncell)
                        pj = st.cl.head[neighbor_cell]
                        while pj > 0
                            if pj != i
                                dx = pos[1, pj] - pos[1, i]
                                dy = pos[2, pj] - pos[2, i]
                                dz = pos[3, pj] - pos[3, i]
                                if dx > L_half
                                    dx -= L
                                elseif dx < -L_half
                                    dx += L
                                end
                                if dy > L_half
                                    dy -= L
                                elseif dy < -L_half
                                    dy += L
                                end
                                if dz > L_half
                                    dz -= L
                                elseif dz < -L_half
                                    dz += L
                                end
                                r2 = dx*dx + dy*dy + dz*dz
                                if r2 < rc2 && r2 > 0.0
                                    total_neighbors += 1
                                end
                            end
                            pj = st.cl.next[pj]
                        end
                    end
                end
            end
        end
        println("NEIGHBOR_AUDIT: enabled=true avg_neighbors=$(total_neighbors / st.N)")
    end
    
    # Reaction: N2 + O2 ⇌ 2 NO (species order: N2, O2, NO)
    # Standard-state chemistry term: logGamma = -β * ΔG°(T), ΔG° = Σ ν_i μ_i°
    # μ° values are in kJ/mol; override with --deltaG0_kjmol if desired
    mu0_phys_kjmol = Dict(
        "N2" => args["mu0_N2_kjmol"],
        "O2" => args["mu0_O2_kjmol"],
        "NO" => args["mu0_NO_kjmol"],
    )
    avogadro = 6.02214076e23
    if isfinite(args["deltaG0_kjmol"])
        delta_g0_star = ((args["deltaG0_kjmol"] * 1000.0) / avogadro) / eps_ref_J
        logGamma = -(1.0 / T_used) * delta_g0_star
        println("logGamma used: $(logGamma) (ΔG° kJ/mol=$(args["deltaG0_kjmol"]))")
    else
        mu0_star = [((mu0_phys_kjmol[s] * 1000.0) / avogadro) / eps_ref_J for s in SPECIES]
        delta_g0_star = -mu0_star[1] - mu0_star[2] + 2.0 * mu0_star[3]
        logGamma = -(1.0 / T_used) * delta_g0_star
        println("logGamma used: $(logGamma) (ΔG°*=$(delta_g0_star), μ° kJ/mol=$mu0_phys_kjmol)")
    end
    reaction = MolSim.MC.Reaction("N2+O2⇌2NO", [-1, -1, 2], 0.0; logq=nothing, logGamma=logGamma)
    
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] " * "=" ^ 79)
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] EXP-6 REMC N2/O2/NO")
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] " * "=" ^ 79)
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] Seed: $(args["seed"])")
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] Equilibration sweeps: $(args["equil"])")
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] Production sweeps: $(args["prod"])")
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] Sample every: $(args["stride"])")
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] p_reaction: $(args["p_reaction"])")
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] Initial density guess: $(args["rho_init"])")
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] T_used = $(T_used)")
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] P_used = $(P_used)")
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] rc_used = $(rc_used)")
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] FF: exp6 (Buckingham)")
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] Mixing rule: $(args["mixing_rule"])")
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] Cutoff model: $(args["cutoff_model"])")
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] rmin_factor: $(args["rmin_factor"])")
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] use_lrc: $(args["use_lrc"])")
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] Species parameters (reduced):")
    for (idx, s) in enumerate(SPECIES)
        println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "]   $s: rm=$(rm_star[idx]), ε=$(eps_star[idx]), α=$(alpha_star[idx])")
    end
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] Initial composition:")
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "]   N2=$(args["N2_init"]), O2=$(args["O2_init"]), NO=$(args["NO_init"])")
    println("[", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "] " * "=" ^ 79)

    println("Starting equilibration...")
    eq_react_attempts = 0
    eq_react_accepted = 0
    eq_fwd_attempts = 0
    eq_fwd_accepted = 0
    eq_rev_attempts = 0
    eq_rev_accepted = 0
    equil_start = time()
    for sweep in 1:args["equil"]
        sweep_start = time()
        _tacc, _vacc, _vatt, racc, ratt, rfatt, rfacc, rratt, rracc, _fc, _rc, _iu, _cb, _ca, _counters =
            MolSim.MC.sweep_npt_with_reactions!(st, p, reaction;
            Pext=P_used, max_dlnV=0.01, p_reaction=args["p_reaction"],
            do_volume_move=true, rebuild_every=-1, insertion_mode=:standard_remc,
            use_alchemical=false)
        eq_react_attempts += ratt
        eq_react_accepted += racc
        eq_fwd_attempts += rfatt
        eq_fwd_accepted += rfacc
        eq_rev_attempts += rratt
        eq_rev_accepted += rracc
        sweep_elapsed = time() - sweep_start
    end
    println("Equilibration reaction stats:")
    println("  attempts=$(eq_react_attempts), accepted=$(eq_react_accepted)")
    println("  forward: attempts=$(eq_fwd_attempts), accepted=$(eq_fwd_accepted)")
    println("  reverse: attempts=$(eq_rev_attempts), accepted=$(eq_rev_accepted)")
    
    println("Starting production...")
    counts_sum = zeros(Float64, 3)
    n_samples = 0
    prod_react_attempts = 0
    prod_react_accepted = 0
    prod_fwd_attempts = 0
    prod_fwd_accepted = 0
    prod_rev_attempts = 0
    prod_rev_accepted = 0
    prod_start = time()
    for sweep in 1:args["prod"]
        sweep_start = time()
        _tacc, _vacc, _vatt, racc, ratt, rfatt, rfacc, rratt, rracc, _fc, _rc, _iu, _cb, _ca, _counters =
            MolSim.MC.sweep_npt_with_reactions!(st, p, reaction;
            Pext=P_used, max_dlnV=0.01, p_reaction=args["p_reaction"],
            do_volume_move=true, rebuild_every=-1, insertion_mode=:standard_remc,
            use_alchemical=false)
        prod_react_attempts += ratt
        prod_react_accepted += racc
        prod_fwd_attempts += rfatt
        prod_fwd_accepted += rfacc
        prod_rev_attempts += rratt
        prod_rev_accepted += rracc
        sweep_elapsed = time() - sweep_start
        if sweep % args["stride"] == 0
            counts = MolSim.MC.count_species(st, 3)
            counts_sum .+= counts
            n_samples += 1
        end
    end
    println("Production reaction stats:")
    println("  attempts=$(prod_react_attempts), accepted=$(prod_react_accepted)")
    println("  forward: attempts=$(prod_fwd_attempts), accepted=$(prod_fwd_accepted)")
    println("  reverse: attempts=$(prod_rev_attempts), accepted=$(prod_rev_accepted)")
    
    avg_counts = counts_sum ./ max(n_samples, 1)
    println("Average counts:")
    println("  N2 = $(avg_counts[1])")
    println("  O2 = $(avg_counts[2])")
    println("  NO = $(avg_counts[3])")
end

main()
