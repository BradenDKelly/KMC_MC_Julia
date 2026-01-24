"""
NVT validation harness comparing Metropolis MC vs eKMC (single-component LJ).
Also compares against Kolafa-Nezbeda EOS (pressure and internal energy).
"""

using Random
using Plots

mutable struct TimeWeightedObservable
    weighted_sum::Float64
    total_time::Float64
    weighted_sum_sq::Float64
    total_time_sq::Float64
    count::Int
end

TimeWeightedObservable() = TimeWeightedObservable(0.0, 0.0, 0.0, 0.0, 0)

function Base.push!(obs::TimeWeightedObservable, x::Float64, dt::Float64)
    obs.weighted_sum += x * dt
    obs.weighted_sum_sq += x * x * dt
    obs.total_time += dt
    obs.total_time_sq += dt * dt
    obs.count += 1
    return obs
end

function mean(obs::TimeWeightedObservable)::Float64
    return obs.total_time > 0.0 ? obs.weighted_sum / obs.total_time : NaN
end

function stderr(obs::TimeWeightedObservable)::Float64
    if obs.total_time <= 0.0 || obs.count < 2
        return NaN
    end
    μ = obs.weighted_sum / obs.total_time
    var_pop = obs.weighted_sum_sq / obs.total_time - μ * μ
    var_pop = max(var_pop, 0.0)
    n_eff = (obs.total_time * obs.total_time) / max(obs.total_time_sq, eps(Float64))
    return n_eff > 1.0 ? sqrt(var_pop / n_eff) : NaN
end

mutable struct NVTObservables
    U_per_particle::TimeWeightedObservable
    pressure::TimeWeightedObservable
    exp_vplus::TimeWeightedObservable
    exp_vminus::TimeWeightedObservable
    mu_ex::Float64
    mu_ex_valid::Bool
    widom_n::Int
    mu_ex_err::Float64
    energy_samples::Vector{Float64}  # For energy distribution analysis
end

NVTObservables() = NVTObservables(TimeWeightedObservable(), TimeWeightedObservable(),
                                  TimeWeightedObservable(), TimeWeightedObservable(), NaN, false, 0, NaN, Float64[])

function reset!(obs::NVTObservables)
    obs.U_per_particle = TimeWeightedObservable()
    obs.pressure = TimeWeightedObservable()
    obs.exp_vplus = TimeWeightedObservable()
    obs.exp_vminus = TimeWeightedObservable()
    obs.mu_ex = NaN
    obs.mu_ex_valid = false
    obs.widom_n = 0
    obs.mu_ex_err = NaN
    obs.energy_samples = Float64[]
    return nothing
end

struct NVTComparisonResult
    N::Int
    ρ::Float64
    T::Float64
    rc::Float64
    use_lrc::Bool
    lj_model::Symbol

    mc_U_per_particle::Float64
    mc_U_err::Float64
    mc_C_V::Float64  # Heat capacity per particle from energy fluctuations
    mc_P::Float64
    mc_P_err::Float64
    mc_P_virtual::Float64
    mc_P_virtual_err::Float64
    mc_mu_ex::Float64
    mc_mu_ex_lrc::Float64
    mc_widom_n::Int
    mc_mu_ex_err::Float64
    mc_n_samples::Int
    mc_wall_time::Float64

    ekmc_U_per_particle::Float64
    ekmc_U_err::Float64
    ekmc_C_V::Float64  # Heat capacity per particle from energy fluctuations
    ekmc_P::Float64
    ekmc_P_err::Float64
    ekmc_P_virtual::Float64
    ekmc_P_virtual_err::Float64
    ekmc_mu_ex::Float64
    ekmc_mu_ex_lrc::Float64
    ekmc_mu_ex_err::Float64
    ekmc_total_time::Float64
    ekmc_n_events::Int
    ekmc_wall_time::Float64

    eos_U_per_particle::Float64
    eos_P::Float64

    delta_U::Float64
    delta_P::Float64
    delta_mu_ex::Float64
    mu_lrc::Float64
end

function run_metropolis_mc(p::LJParams, st::LJState;
                           burnin_sweeps::Int=10000,
                           prod_sweeps::Int=100000,
                           sample_every::Int=10,
                           dlnV_virtual::Float64=1e-4,
                           widom_every::Int=50,
                           widom_ninsert::Int=200,
                           collect_timeseries::Bool=false,
                           rdf_nbins::Int=100,
                           rdf_rmax::Float64=-1.0)
    T = 1.0 / p.β
    N = st.N
    L = st.L
    
    # RDF parameters
    if rdf_rmax < 0.0
        rdf_rmax = L / 2.0  # Default to half box length
    end

    # Time series storage
    timeseries = collect_timeseries ? (
        sweeps_burnin = Float64[],
        U_burnin = Float64[],
        P_burnin = Float64[],
        mu_burnin = Float64[],
        sweeps_prod = Float64[],
        U_prod = Float64[],
        P_prod = Float64[],
        mu_prod = Float64[]
    ) : nothing
    
    # RDF will be computed once at end of production

    # Burn-in phase
    widom_acc_burnin = WidomAccumulator()
    for sweep_idx in 1:burnin_sweeps
        sweep!(st, p; rebuild_every=1)
        
        if collect_timeseries && sweep_idx % sample_every == 0
            U = total_energy(st, p)
            P = pressure(st, p, T)
            push!(timeseries.sweeps_burnin, Float64(sweep_idx))
            push!(timeseries.U_burnin, U / N)
            push!(timeseries.P_burnin, P)
            # Compute running μ_ex estimate
            if widom_acc_burnin.n > 0
                mu_est = mu_ex(widom_acc_burnin, p.β)
                push!(timeseries.mu_burnin, isfinite(mu_est) ? mu_est : NaN)
            else
                push!(timeseries.mu_burnin, NaN)
            end
        end
        
        if sweep_idx % widom_every == 0
            for _ in 1:widom_ninsert
                ΔU = widom_deltaU(st, p)
                push!(widom_acc_burnin, p.β, ΔU)
            end
        end
    end

    obs = NVTObservables()
    reset!(obs)
    widom_acc = WidomAccumulator()

    for sweep_idx in 1:prod_sweeps
        sweep!(st, p; rebuild_every=1)

        if sweep_idx % sample_every == 0
            U = total_energy(st, p)
            P = pressure(st, p, T)
            exp_plus, exp_minus = virtual_volume_exp_factors(st, p, dlnV_virtual)
            push!(obs.U_per_particle, U / N, 1.0)
            push!(obs.pressure, P, 1.0)
            push!(obs.exp_vplus, exp_plus, 1.0)
            push!(obs.exp_vminus, exp_minus, 1.0)
            # Collect energy sample for distribution analysis
            push!(obs.energy_samples, U / N)
            
            
            if collect_timeseries
                push!(timeseries.sweeps_prod, Float64(sweep_idx))
                push!(timeseries.U_prod, U / N)
                push!(timeseries.P_prod, P)
                # Compute running μ_ex estimate if we have Widom data
                if widom_acc.n > 0
                    mu_est = mu_ex(widom_acc, p.β)
                    push!(timeseries.mu_prod, isfinite(mu_est) ? mu_est : NaN)
                else
                    push!(timeseries.mu_prod, NaN)
                end
            end
        end

        if sweep_idx % widom_every == 0
            for _ in 1:widom_ninsert
                ΔU = widom_deltaU(st, p)
                push!(widom_acc, p.β, ΔU)
            end
        end
    end

    if widom_acc.n > 0
        obs.mu_ex = mu_ex(widom_acc, p.β)
        obs.mu_ex_valid = isfinite(obs.mu_ex)
        if widom_acc.n > 1 && widom_acc.m > 0.0
            var_exp = widom_acc.s / (widom_acc.n - 1)
            σ_m = sqrt(var_exp / widom_acc.n)
            Tcur = 1.0 / p.β
            obs.mu_ex_err = Tcur * (σ_m / widom_acc.m)
        end
    end
    obs.widom_n = widom_acc.n

    # Compute RDF from final configuration
    rdf_result = Main.MolSim.Analysis.rdf(st.pos, L, rdf_rmax, rdf_nbins; types=nothing)

    if collect_timeseries
        return obs, timeseries, rdf_result
    else
        return obs, rdf_result
    end
end

function run_ekmc(p::LJParams, ekst::eKMCState, acc_mu::ChemicalPotentialAccumulator;
                  burnin_events::Int=10000,
                  prod_events::Int=100000,
                  sample_every::Int=10,
                  dlnV_virtual::Float64=1e-4,
                  collect_timeseries::Bool=false,
                  rdf_nbins::Int=100,
                  rdf_rmax::Float64=-1.0)
    T = 1.0 / p.β
    N = ekst.N
    L = ekst.L
    
    # RDF parameters
    if rdf_rmax < 0.0
        rdf_rmax = L / 2.0  # Default to half box length
    end
    
    # Time series storage
    timeseries = collect_timeseries ? (
        events_burnin = Float64[],
        time_burnin = Float64[],
        U_burnin = Float64[],
        P_burnin = Float64[],
        mu_burnin = Float64[],
        events_prod = Float64[],
        time_prod = Float64[],
        U_prod = Float64[],
        P_prod = Float64[],
        mu_prod = Float64[]
    ) : nothing
    
    println("Burning in eKMC...")
    reset!(acc_mu)
    burnin_time = 0.0
    for event_idx in 1:burnin_events
        dt = ekmc_step!(ekst, p, acc_mu)
        burnin_time += dt
        
        if collect_timeseries && event_idx % sample_every == 0
            U = total_energy(ekst, p)
            P = pressure(ekst, p, T)
            push!(timeseries.events_burnin, Float64(event_idx))
            push!(timeseries.time_burnin, burnin_time)
            push!(timeseries.U_burnin, U / N)
            push!(timeseries.P_burnin, P)
            # Running μ_ex estimate from accumulator
            mu_est = mu_ex(acc_mu, T)
            push!(timeseries.mu_burnin, isfinite(mu_est) ? mu_est : NaN)
        end
    end
    println("Burning in eKMC... done")
    obs = NVTObservables()
    reset!(obs)
    # Create separate accumulator for production-only final result
    acc_mu_prod = ChemicalPotentialAccumulator()
    # Keep acc_mu accumulating for continuous timeseries (no reset)

    total_time = 0.0
    
    println("Running eKMC production...")
    for event_idx in 1:prod_events
        R_before = ekst.R  # Store R before move (same as used in ekmc_step!)

        # According to Tan et al. (CEJ 2020) recipe:
        # - We're currently in a state with rate R_before
        # - ekmc_step! will compute dt = -ln(ξ)/R_before (residence time in current state)
        # - Then it will perform the move to a new state
        # - For time-weighted averages, we should weight observables by dt
        # - So we sample BEFORE the move, using the dt that will be computed in the step

        do_sample = event_idx % sample_every == 0
        # Sample observables from current state BEFORE the move (sampling steps only).
        # Weighting uses dt from this specific step (no accumulated dt).
        if do_sample
            U = total_energy(ekst, p)
            P = pressure(ekst, p, T)
            exp_plus, exp_minus = virtual_volume_exp_factors(ekst, p, dlnV_virtual)
        end

        # Perform KMC step: computes dt for current state, then moves to new state
        dt = ekmc_step!(ekst, p, acc_mu)  # dt is residence time in state BEFORE move

        # Also update production-only accumulator with same R value
        R_over_N = R_before / N
        acc_mu_prod.t_total += dt
        acc_mu_prod.S += R_over_N * dt
        acc_mu_prod.S_sq += (R_over_N * R_over_N) * dt
        acc_mu_prod.t_total_sq += dt * dt
        acc_mu_prod.count += 1
        total_time += dt

        # Accumulate observables only on sampling steps, using dt from this step
        if do_sample && dt > 0.0
            push!(obs.U_per_particle, U / N, dt)
            push!(obs.pressure, P, dt)
            push!(obs.exp_vplus, exp_plus, dt)
            push!(obs.exp_vminus, exp_minus, dt)
            push!(obs.energy_samples, U / N)

            if collect_timeseries
                push!(timeseries.events_prod, Float64(event_idx))
                push!(timeseries.time_prod, total_time)
                push!(timeseries.U_prod, U / N)
                push!(timeseries.P_prod, P)
                # Running μ_ex estimate from accumulator
                mu_est = mu_ex(acc_mu, T)
                push!(timeseries.mu_prod, isfinite(mu_est) ? mu_est : NaN)
            end
        end
    end

    obs.mu_ex = mu_ex(acc_mu_prod, T)  # Use production-only accumulator
    obs.mu_ex_err = mu_ex_stderr(acc_mu_prod, T)
    obs.mu_ex_valid = isfinite(obs.mu_ex)

    # Diagnostic: Compare pair matrix energy with total_energy
    E_pair = 0.5 * sum(ekst.pair_u)
    E_full = total_energy(ekst, p)
    ΔE = E_pair - E_full
    if abs(ΔE) > 1e-6
        println("eKMC WARNING: Pair matrix energy ($(E_pair/N)) differs from total_energy ($(E_full/N)) by $(ΔE/N) per particle")
        println("  This suggests the pair_u matrix is out of sync with actual positions!")
    end
    
    # Diagnostic: Check if time-weighted averaging might be the issue
    # Compare instantaneous energy at final state
    U_final_ekmc = total_energy(ekst, p) / N
    println("eKMC final instantaneous U/N = ", U_final_ekmc)
    println("eKMC time-weighted average U/N = ", mean(obs.U_per_particle))
    println("  (If these differ significantly, time-weighted avg may be biased)")
    
    # Diagnostic: Check phi distribution and mobilities
    min_phi = minimum(ekst.phi)
    max_phi = maximum(ekst.phi)
    mean_phi = sum(ekst.phi) / length(ekst.phi)
    min_m = minimum(ekst.m)
    max_m = maximum(ekst.m)
    mean_m = sum(ekst.m) / length(ekst.m)
    println("eKMC final phi distribution: min=$(min_phi), max=$(max_phi), mean=$(mean_phi)")
    println("eKMC final mobilities: min=$(min_m), max=$(max_m), mean=$(mean_m), R=$(ekst.R)")
    println("  (phi should be negative for attractive LJ, mobilities should be exp(β*phi))")
    
    # Compute RDF from final configuration
    rdf_result = Main.MolSim.Analysis.rdf(ekst.pos, L, rdf_rmax, rdf_nbins; types=nothing)

    if collect_timeseries
        return obs, total_time, timeseries, rdf_result
    else
        return obs, total_time, rdf_result
    end
end

function eos_u_per_particle(T::Float64, ρ::Float64)::Float64
    return Main.MolSim.EOS.internal_energy_nkeos(T, ρ)
end

function _print_timed(label::String, f)
    t = @timed f()
    println("$label time = $(round(t.time, digits=3)) s, alloc = $(round(t.bytes / 1e6, digits=3)) MB, gctime = $(round(t.gctime, digits=3)) s")
    return t.value
end

function compare_nvt(; N::Int=500,
                     ρ::Float64=0.8,
                     T::Float64=1.2,
                     rc::Float64=2.5,
                     max_disp::Float64=0.1,
                     use_lrc::Bool=false,
                     lj_model::Symbol=:truncated,
                     apply_impulsive_correction::Bool=false,
                     init_lattice::Symbol=:fcc,
                     seed_mc::Int=1234,
                     seed_ekmc::Int=4321,
                     burnin_sweeps_mc::Int=10000,
                     prod_sweeps_mc::Int=100000,
                     sample_every_mc::Int=10,
                     widom_every_mc::Int=50,
                     widom_ninsert_mc::Int=200,
                     burnin_events_ekmc::Int=10000,
                     prod_events_ekmc::Int=100000,
                     sample_every_ekmc::Int=10,
                     dlnV_virtual::Float64=1e-4,
                     debug_check_every_ekmc::Int=0,
                     profile::Bool=false,
                     collect_timeseries::Bool=false,
                     rdf_nbins::Int=100,
                     rdf_rmax::Float64=-1.0)

    Random.seed!(seed_mc)
    if init_lattice != :fcc
        throw(ArgumentError("init_lattice must be :fcc for this branch, got :$init_lattice"))
    end

    p_mc, st_mc = init_fcc(; N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp,
                           seed=seed_mc, use_lrc=use_lrc, lj_model=lj_model,
                           apply_impulsive_correction=apply_impulsive_correction)

    p_ekmc = p_mc
    st_ekmc = LJState(
        st_mc.N, st_mc.L, copy(st_mc.pos), copy(st_mc.types),
        Xoshiro(seed_ekmc),
        CellList(st_mc.N, st_mc.L, rc),
        MVector{3,Float64}(0.0, 0.0, 0.0),
        0, 0
    )
    rebuild_cells!(st_ekmc)

    # Diagnostic: Print force field parameters
    println("Force field parameters:")
    println("  σ = ", p_mc.σ, ", ε = ", p_mc.ϵ)
    println("  rc = ", p_mc.rc, ", rc2 = ", p_mc.rc2)
    println("  lj_model = ", p_mc.lj_model, ", use_lrc = ", p_mc.use_lrc)
    println("  u_rc (shift value) = ", p_mc.u_rc)
    println("  apply_impulsive_correction = ", apply_impulsive_correction)
    println("  Same parameters for eKMC? ", (p_ekmc === p_mc))
    
    # Compute LRC values for curiosity (even if not used)
    # (LJLongRange functions are available in MC module)
    lrc_u_per_particle = compute_lrc_energy_per_particle(ρ, rc)
    lrc_p = compute_lrc_pressure(ρ, rc)
    println()
    println("Long-range corrections (for reference, not applied):")
    println("  LRC U/N = ", lrc_u_per_particle)
    println("  LRC P   = ", lrc_p)
    println("  (Note: These are NOT applied since use_lrc = false and model is cut-and-shifted)")
    
    # Verify impulsive corrections are not applied
    println()
    println("Impulsive correction verification:")
    println("  p_mc.apply_impulsive_correction = ", p_mc.apply_impulsive_correction)
    println("  p_ekmc.apply_impulsive_correction = ", p_ekmc.apply_impulsive_correction)
    if p_mc.apply_impulsive_correction || p_ekmc.apply_impulsive_correction
        println("  WARNING: Impulsive corrections ARE enabled!")
    else
        println("  ✓ Impulsive corrections are disabled (correct for cut-and-shifted LJ)")
    end
    # Reporting-only impulsive correction value (not applied to energy or pressure unless enabled)
    ρ_inst = N / (st_mc.L * st_mc.L * st_mc.L)
    g_rc = compute_g_rc(st_mc, p_mc)
    ΔP_imp = -(2.0 * π / 3.0) * ρ_inst * ρ_inst * p_mc.rc * p_mc.rc * p_mc.rc * p_mc.u_rc * g_rc
    println("  g(rc) = ", g_rc)
    println("  ΔP_imp = ", ΔP_imp)
    
    # Verify shift value is identical
    println()
    println("Shift value verification (cut-and-shifted LJ):")
    println("  p_mc.u_rc   = ", p_mc.u_rc)
    println("  p_ekmc.u_rc = ", p_ekmc.u_rc)
    println("  Difference  = ", p_ekmc.u_rc - p_mc.u_rc)
    if abs(p_ekmc.u_rc - p_mc.u_rc) > 1e-15
        println("  WARNING: Shift values differ!")
    else
        println("  ✓ Shift values are identical")
    end
    if p_ekmc === p_mc
        println("  ✓ MC and eKMC share the same parameter object (guaranteed identical)")
    end
    
    # Diagnostic: Compare energies at same initial state
    U_mc_init = total_energy(st_mc, p_mc) / N
    U_ekmc_init = total_energy(st_ekmc, p_ekmc) / N
    println()
    println("Initial energy comparison (should be identical):")
    println("  MC initial U/N   = ", U_mc_init)
    println("  eKMC initial U/N = ", U_ekmc_init)
    println("  Difference       = ", U_ekmc_init - U_mc_init)
    if abs(U_ekmc_init - U_mc_init) > 1e-10
        println("  WARNING: Initial energies differ! Force fields may not match.")
    end

    ekst = init_ekmc_state(st_ekmc, p_ekmc)
    ekst.debug_check_every = debug_check_every_ekmc
    acc_mu = ChemicalPotentialAccumulator()
    
    # Diagnostic: Compare eKMC pair matrix energy with total_energy at initialization
    E_pair_init = 0.5 * sum(ekst.pair_u) / N
    E_full_init = total_energy(ekst, p_ekmc) / N
    println("eKMC pair matrix vs total_energy at init:")
    println("  Pair matrix U/N = ", E_pair_init)
    println("  total_energy U/N = ", E_full_init)
    println("  Difference      = ", E_pair_init - E_full_init)
    if abs(E_pair_init - E_full_init) > 1e-10
        println("  WARNING: Pair matrix doesn't match total_energy at initialization!")
    end

    println("Starting MC...")
    time_mc_start = Base.time()
    mc_result = profile ? _print_timed("MC run", () -> run_metropolis_mc(p_mc, st_mc;
                                                                       burnin_sweeps=burnin_sweeps_mc,
                                                                       prod_sweeps=prod_sweeps_mc,
                                                                       sample_every=sample_every_mc,
                                                                       dlnV_virtual=dlnV_virtual,
                                                                       widom_every=widom_every_mc,
                                                                       widom_ninsert=widom_ninsert_mc,
                                                                       collect_timeseries=collect_timeseries,
                                                                       rdf_nbins=rdf_nbins,
                                                                       rdf_rmax=rdf_rmax)) :
                        run_metropolis_mc(p_mc, st_mc;
                                          burnin_sweeps=burnin_sweeps_mc,
                                          prod_sweeps=prod_sweeps_mc,
                                          sample_every=sample_every_mc,
                                          dlnV_virtual=dlnV_virtual,
                                          widom_every=widom_every_mc,
                                          widom_ninsert=widom_ninsert_mc,
                                          collect_timeseries=collect_timeseries,
                                          rdf_nbins=rdf_nbins,
                                          rdf_rmax=rdf_rmax)
    time_mc = Base.time() - time_mc_start
    println("Finished MC.")
    
    if collect_timeseries
        obs_mc, timeseries_mc, rdf_mc = mc_result
    else
        obs_mc, rdf_mc = mc_result
        timeseries_mc = nothing
    end

    println("Starting eKMC...")
    time_ekmc_start = Base.time()
    if profile
        timed_val = _print_timed("eKMC run", () -> run_ekmc(p_ekmc, ekst, acc_mu;
                                                            burnin_events=burnin_events_ekmc,
                                                            prod_events=prod_events_ekmc,
                                                            sample_every=sample_every_ekmc,
                                                            dlnV_virtual=dlnV_virtual,
                                                            collect_timeseries=collect_timeseries,
                                                            rdf_nbins=rdf_nbins,
                                                            rdf_rmax=rdf_rmax))
        if collect_timeseries
            obs_ekmc, total_time_sim, timeseries_ekmc, rdf_ekmc = timed_val
        else
            obs_ekmc, total_time_sim, rdf_ekmc = timed_val
            timeseries_ekmc = nothing
        end
    else
        ekmc_result = run_ekmc(p_ekmc, ekst, acc_mu;
                              burnin_events=burnin_events_ekmc,
                              prod_events=prod_events_ekmc,
                              sample_every=sample_every_ekmc,
                              dlnV_virtual=dlnV_virtual,
                              collect_timeseries=collect_timeseries,
                              rdf_nbins=rdf_nbins,
                              rdf_rmax=rdf_rmax)
        if collect_timeseries
            obs_ekmc, total_time_sim, timeseries_ekmc, rdf_ekmc = ekmc_result
        else
            obs_ekmc, total_time_sim, rdf_ekmc = ekmc_result
            timeseries_ekmc = nothing
        end
    end
    time_ekmc = Base.time() - time_ekmc_start
    println("Finished eKMC.")

    U_mc = mean(obs_mc.U_per_particle)
    U_mc_err = stderr(obs_mc.U_per_particle)
    # Calculate C_V from energy fluctuations: C_V = σ²(U) / (k_B T²) = σ²(U) / T² in reduced units
    var_U_mc = obs_mc.U_per_particle.weighted_sum_sq / obs_mc.U_per_particle.total_time - U_mc * U_mc
    var_U_mc = max(var_U_mc, 0.0)
    C_V_mc = var_U_mc / (T * T)  # Heat capacity per particle from fluctuations
    P_mc = mean(obs_mc.pressure)
    P_mc_err = stderr(obs_mc.pressure)
    V = (N / ρ)
    β = p_mc.β
    dV = V * sinh(dlnV_virtual)
    exp_plus_mc = mean(obs_mc.exp_vplus)
    exp_minus_mc = mean(obs_mc.exp_vminus)
    P_mc_virtual = (N * T / V) + (1.0 / (2.0 * β * dV)) * log(exp_plus_mc / exp_minus_mc)
    B = (1.0 / (2.0 * β * dV))
    var_m_plus = stderr(obs_mc.exp_vplus)^2
    var_m_minus = stderr(obs_mc.exp_vminus)^2
    if exp_plus_mc > 0.0 && exp_minus_mc > 0.0 && isfinite(var_m_plus) && isfinite(var_m_minus)
        P_mc_virtual_err = abs(B) * sqrt(var_m_plus / (exp_plus_mc * exp_plus_mc) +
                                         var_m_minus / (exp_minus_mc * exp_minus_mc))
    else
        P_mc_virtual_err = NaN
    end
    μ_mc = obs_mc.mu_ex_valid ? obs_mc.mu_ex : NaN
    μ_mc_err = obs_mc.mu_ex_err
    μ_lrc = p_mc.use_lrc ? 2.0 * p_mc.lrc_u_per_particle : 0.0
    μ_mc_lrc = isfinite(μ_mc) ? μ_mc + μ_lrc : μ_mc

    U_ekmc = mean(obs_ekmc.U_per_particle)
    U_ekmc_err = stderr(obs_ekmc.U_per_particle)
    # Calculate C_V from energy fluctuations: C_V = σ²(U) / (k_B T²) = σ²(U) / T² in reduced units
    var_U_ekmc = obs_ekmc.U_per_particle.weighted_sum_sq / obs_ekmc.U_per_particle.total_time - U_ekmc * U_ekmc
    var_U_ekmc = max(var_U_ekmc, 0.0)
    C_V_ekmc = var_U_ekmc / (T * T)  # Heat capacity per particle from fluctuations
    P_ekmc = mean(obs_ekmc.pressure)
    P_ekmc_err = stderr(obs_ekmc.pressure)
    exp_plus_ekmc = mean(obs_ekmc.exp_vplus)
    exp_minus_ekmc = mean(obs_ekmc.exp_vminus)
    P_ekmc_virtual = (N * T / V) + (1.0 / (2.0 * β * dV)) * log(exp_plus_ekmc / exp_minus_ekmc)
    var_m_plus_ek = stderr(obs_ekmc.exp_vplus)^2
    var_m_minus_ek = stderr(obs_ekmc.exp_vminus)^2
    if exp_plus_ekmc > 0.0 && exp_minus_ekmc > 0.0 && isfinite(var_m_plus_ek) && isfinite(var_m_minus_ek)
        P_ekmc_virtual_err = abs(B) * sqrt(var_m_plus_ek / (exp_plus_ekmc * exp_plus_ekmc) +
                                           var_m_minus_ek / (exp_minus_ekmc * exp_minus_ekmc))
    else
        P_ekmc_virtual_err = NaN
    end
    μ_ekmc = obs_ekmc.mu_ex_valid ? obs_ekmc.mu_ex : NaN
    μ_ekmc_err = obs_ekmc.mu_ex_err
    μ_ekmc_lrc = isfinite(μ_ekmc) ? μ_ekmc + μ_lrc : μ_ekmc

    eos_P = Main.MolSim.EOS.pressure_nkeos(T, ρ)
    eos_U = eos_u_per_particle(T, ρ)

    result = NVTComparisonResult(
        N, ρ, T, rc, use_lrc, lj_model,
        U_mc, U_mc_err, C_V_mc, P_mc, P_mc_err, P_mc_virtual, P_mc_virtual_err, μ_mc, μ_mc_lrc, obs_mc.widom_n, μ_mc_err, obs_mc.U_per_particle.count, time_mc,
        U_ekmc, U_ekmc_err, C_V_ekmc, P_ekmc, P_ekmc_err, P_ekmc_virtual, P_ekmc_virtual_err, μ_ekmc, μ_ekmc_lrc, μ_ekmc_err, total_time_sim, prod_events_ekmc, time_ekmc,
        eos_U, eos_P,
        U_ekmc - U_mc,
        P_ekmc - P_mc,
        μ_ekmc - μ_mc,
        μ_lrc
    )
    
    # Energy distribution analysis to verify Boltzmann sampling
    println()
    println("=" ^ 70)
    println("Energy Distribution Analysis (Boltzmann Sampling Verification):")
    println("=" ^ 70)
    
    U_mc_samples = obs_mc.energy_samples
    U_ekmc_samples = obs_ekmc.energy_samples
    
    if length(U_mc_samples) > 0 && length(U_ekmc_samples) > 0
        # Compute statistics
        mean_mc = sum(U_mc_samples) / length(U_mc_samples)
        mean_ekmc = sum(U_ekmc_samples) / length(U_ekmc_samples)
        
        # Variance
        var_mc = sum((x - mean_mc)^2 for x in U_mc_samples) / (length(U_mc_samples) - 1)
        var_ekmc = sum((x - mean_ekmc)^2 for x in U_ekmc_samples) / (length(U_ekmc_samples) - 1)
        std_mc = sqrt(var_mc)
        std_ekmc = sqrt(var_ekmc)
        
        # Skewness (third moment)
        skew_mc = length(U_mc_samples) > 2 ? sum(((x - mean_mc) / std_mc)^3 for x in U_mc_samples) / length(U_mc_samples) : NaN
        skew_ekmc = length(U_ekmc_samples) > 2 ? sum(((x - mean_ekmc) / std_ekmc)^3 for x in U_ekmc_samples) / length(U_ekmc_samples) : NaN
        
        println("MC energy distribution:")
        println("  N samples = ", length(U_mc_samples))
        println("  Mean U/N  = ", mean_mc)
        println("  Std dev   = ", std_mc)
        println("  Skewness  = ", skew_mc)
        println("  Min       = ", minimum(U_mc_samples))
        println("  Max       = ", maximum(U_mc_samples))
        println()
        println("eKMC energy distribution:")
        println("  N samples = ", length(U_ekmc_samples))
        println("  Mean U/N  = ", mean_ekmc)
        println("  Std dev   = ", std_ekmc)
        println("  Skewness  = ", skew_ekmc)
        println("  Min       = ", minimum(U_ekmc_samples))
        println("  Max       = ", maximum(U_ekmc_samples))
        println()
        println("Distribution comparison:")
        println("  Δ mean    = ", mean_ekmc - mean_mc)
        println("  Δ std     = ", std_ekmc - std_mc)
        println("  Δ skew    = ", isfinite(skew_ekmc) && isfinite(skew_mc) ? skew_ekmc - skew_mc : NaN)
        println()
        println("  If eKMC is sampling from Boltzmann distribution, distributions should match.")
        println("  Large differences indicate potential sampling bias in eKMC.")
    else
        println("  Insufficient samples for distribution analysis")
    end
    
    if collect_timeseries
        return result, timeseries_mc, timeseries_ekmc, rdf_mc, rdf_ekmc
    else
        return result, rdf_mc, rdf_ekmc
    end
end

function print_comparison(result::NVTComparisonResult)
    println("=" ^ 70)
    println("NVT Comparison: Metropolis MC vs eKMC vs EOS")
    println("=" ^ 70)
    println("System parameters:")
    println("  N = $(result.N), ρ = $(result.ρ), T = $(result.T)")
    println("  rc = $(result.rc), use_lrc = $(result.use_lrc), model = $(result.lj_model)")
    println()
    println("Metropolis MC:")
    println("  U/N = $(result.mc_U_per_particle) ± $(result.mc_U_err)")
    println("  C_V = $(result.mc_C_V)  (from energy fluctuations: σ²(U)/T²)")
    println("  P (virial)  = $(result.mc_P) ± $(result.mc_P_err)")
    println("  P (virtual) = $(result.mc_P_virtual) ± $(result.mc_P_virtual_err)")
    println("  μ_ex = $(result.mc_mu_ex) ± $(result.mc_mu_ex_err)")
    println("  μ_ex + μ_LRC = $(result.mc_mu_ex_lrc)")
    println("  Widom inserts = $(result.mc_widom_n)")
    println("  Samples = $(result.mc_n_samples), Wall time = $(round(result.mc_wall_time, digits=3)) s")
    println()
    println("eKMC:")
    println("  U/N = $(result.ekmc_U_per_particle) ± $(result.ekmc_U_err)")
    println("  C_V = $(result.ekmc_C_V)  (from energy fluctuations: σ²(U)/T²)")
    println("  P (virial)  = $(result.ekmc_P) ± $(result.ekmc_P_err)")
    println("  P (virtual) = $(result.ekmc_P_virtual) ± $(result.ekmc_P_virtual_err)")
    println("  μ_ex = $(result.ekmc_mu_ex) ± $(result.ekmc_mu_ex_err)")
    println("  μ_ex + μ_LRC = $(result.ekmc_mu_ex_lrc)")
    println("  Events = $(result.ekmc_n_events), Simulated time = $(round(result.ekmc_total_time, digits=6))")
    println("  Wall time = $(round(result.ekmc_wall_time, digits=3)) s")
    println()
    println("EOS (Kolafa-Nezbeda):")
    println("  U/N = $(result.eos_U_per_particle)")
    println("  P   = $(result.eos_P)")
    println()
    println("Differences (eKMC - MC):")
    println("  Δ(U/N) = $(result.delta_U)")
    println("  ΔP     = $(result.delta_P)")
    println("  Δμ_ex  = $(result.delta_mu_ex)")
    println("=" ^ 70)
end

function write_comparison_csv(result::NVTComparisonResult, filename::String)
    open(filename, "w") do io
        println(io, "parameter,value")
        println(io, "N,$(result.N)")
        println(io, "rho,$(result.ρ)")
        println(io, "T,$(result.T)")
        println(io, "rc,$(result.rc)")
        println(io, "use_lrc,$(result.use_lrc)")
        println(io, "lj_model,$(result.lj_model)")
        println(io, "")
        println(io, "observable,mc_value,ekmc_value,eos_value")
    println(io, "U_per_particle,$(result.mc_U_per_particle),$(result.ekmc_U_per_particle),$(result.eos_U_per_particle)")
    println(io, "U_per_particle_err,$(result.mc_U_err),$(result.ekmc_U_err),")
    println(io, "P_virial,$(result.mc_P),$(result.ekmc_P),$(result.eos_P)")
    println(io, "P_virial_err,$(result.mc_P_err),$(result.ekmc_P_err),")
    println(io, "P_virtual,$(result.mc_P_virtual),$(result.ekmc_P_virtual),")
    println(io, "P_virtual_err,$(result.mc_P_virtual_err),$(result.ekmc_P_virtual_err),")
    println(io, "mu_ex,$(result.mc_mu_ex),$(result.ekmc_mu_ex),")
    println(io, "mu_ex_err,$(result.mc_mu_ex_err),$(result.ekmc_mu_ex_err),")
    println(io, "mu_ex_lrc,$(result.mc_mu_ex_lrc),$(result.ekmc_mu_ex_lrc),")
    println(io, "mu_lrc,$(result.mu_lrc),$(result.mu_lrc),")
        println(io, "")
        println(io, "metric,mc_value,ekmc_value")
    println(io, "widom_inserts,$(result.mc_widom_n),")
        println(io, "n_samples,$(result.mc_n_samples),$(result.ekmc_n_events)")
        println(io, "wall_time_s,$(result.mc_wall_time),$(result.ekmc_wall_time)")
        println(io, "simulated_time,$(NaN),$(result.ekmc_total_time)")
    end
end

function plot_rdf_comparison(rdf_mc::Tuple{Vector{Float64}, Vector{Float64}},
                              rdf_ekmc::Tuple{Vector{Float64}, Vector{Float64}},
                              filename::String="rdf_comparison.png")
    """Plot MC and eKMC RDF on same figure for comparison."""
    try
        r_mc, g_mc = rdf_mc
        r_ekmc, g_ekmc = rdf_ekmc
        
        p = plot(xlabel="r / σ", ylabel="g(r)", title="Radial Distribution Function: MC vs eKMC",
                 legend=:topright, grid=true, dpi=300)
        plot!(p, r_mc, g_mc, label="MC", linewidth=2, color=:blue)
        plot!(p, r_ekmc, g_ekmc, label="eKMC", linewidth=2, color=:red, linestyle=:dash)
        
        savefig(p, filename)
        println("Saved RDF comparison plot: $filename")
    catch e
        println("Warning: Could not create RDF plot (Plots.jl may not be available): $e")
    end
end
