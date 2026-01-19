"""
NVT validation harness comparing Metropolis MC vs eKMC (single-component LJ).
Also compares against Kolafa-Nezbeda EOS (pressure and internal energy).
"""

using Random

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
end

NVTObservables() = NVTObservables(TimeWeightedObservable(), TimeWeightedObservable(),
                                  TimeWeightedObservable(), TimeWeightedObservable(), NaN, false, 0, NaN)

function reset!(obs::NVTObservables)
    obs.U_per_particle = TimeWeightedObservable()
    obs.pressure = TimeWeightedObservable()
    obs.exp_vplus = TimeWeightedObservable()
    obs.exp_vminus = TimeWeightedObservable()
    obs.mu_ex = NaN
    obs.mu_ex_valid = false
    obs.widom_n = 0
    obs.mu_ex_err = NaN
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
                           widom_ninsert::Int=200)
    T = 1.0 / p.β
    N = st.N

    for _ in 1:burnin_sweeps
        sweep!(st, p; rebuild_every=1)
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

    return obs
end

function run_ekmc(p::LJParams, ekst::eKMCState, acc_mu::ChemicalPotentialAccumulator;
                  burnin_events::Int=10000,
                  prod_events::Int=100000,
                  sample_every::Int=10,
                  dlnV_virtual::Float64=1e-4)
    T = 1.0 / p.β
    N = ekst.N
    println("Burning in eKMC...")
    reset!(acc_mu)
    for _ in 1:burnin_events
        ekmc_step!(ekst, p, acc_mu)
    end
    println("Burning in eKMC... done")
    obs = NVTObservables()
    reset!(obs)
    reset!(acc_mu)

    total_time = 0.0
    accumulated_dt = 0.0
    println("Running eKMC production...")
    for event_idx in 1:prod_events
        dt = ekmc_step!(ekst, p, acc_mu)
        total_time += dt
        accumulated_dt += dt

        if event_idx % sample_every == 0
            U = total_energy(ekst, p)
            P = pressure(ekst, p, T)
            if accumulated_dt > 0.0
                push!(obs.U_per_particle, U / N, accumulated_dt)
                push!(obs.pressure, P, accumulated_dt)
                exp_plus, exp_minus = virtual_volume_exp_factors(ekst, p, dlnV_virtual)
                push!(obs.exp_vplus, exp_plus, accumulated_dt)
                push!(obs.exp_vminus, exp_minus, accumulated_dt)
            end
            accumulated_dt = 0.0
        end
    end
    
    if accumulated_dt > 0.0
        U = total_energy(ekst, p)
        P = pressure(ekst, p, T)
        push!(obs.U_per_particle, U / N, accumulated_dt)
        push!(obs.pressure, P, accumulated_dt)
        exp_plus, exp_minus = virtual_volume_exp_factors(ekst, p, dlnV_virtual)
        push!(obs.exp_vplus, exp_plus, accumulated_dt)
        push!(obs.exp_vminus, exp_minus, accumulated_dt)
    end

    obs.mu_ex = mu_ex(acc_mu, T)
    obs.mu_ex_valid = isfinite(obs.mu_ex)

    return obs, total_time
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
                     init_lattice::Symbol=:auto,
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
                     profile::Bool=false)

    Random.seed!(seed_mc)
    if init_lattice == :fcc
        p_mc, st_mc = init_fcc(; N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp,
                               seed=seed_mc, use_lrc=use_lrc, lj_model=lj_model,
                               apply_impulsive_correction=apply_impulsive_correction)
    elseif init_lattice == :sc
        p_mc, st_mc = init_sc(; N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp,
                              seed=seed_mc, use_lrc=use_lrc, lj_model=lj_model,
                              apply_impulsive_correction=apply_impulsive_correction)
    elseif init_lattice == :auto
        if N % 4 == 0 && _is_perfect_cube(N ÷ 4)
            p_mc, st_mc = init_fcc(; N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp,
                                   seed=seed_mc, use_lrc=use_lrc, lj_model=lj_model,
                                   apply_impulsive_correction=apply_impulsive_correction)
        else
            p_mc, st_mc = init_sc(; N=N, ρ=ρ, T=T, rc=rc, max_disp=max_disp,
                                  seed=seed_mc, use_lrc=use_lrc, lj_model=lj_model,
                                  apply_impulsive_correction=apply_impulsive_correction)
        end
    else
        throw(ArgumentError("init_lattice must be :fcc, :sc, or :auto, got :$init_lattice"))
    end

    p_ekmc = p_mc
    st_ekmc = LJState(
        st_mc.N, st_mc.L, copy(st_mc.pos), copy(st_mc.types),
        Xoshiro(seed_ekmc),
        CellList(st_mc.N, st_mc.L, rc),
        MVector{3,Float64}(0.0, 0.0, 0.0),
        0, 0
    )
    rebuild_cells!(st_ekmc)

    ekst = init_ekmc_state(st_ekmc, p_ekmc)
    ekst.debug_check_every = debug_check_every_ekmc
    acc_mu = ChemicalPotentialAccumulator()

    println("Starting MC...")
    time_mc_start = Base.time()
    obs_mc = profile ? _print_timed("MC run", () -> run_metropolis_mc(p_mc, st_mc;
                                                                       burnin_sweeps=burnin_sweeps_mc,
                                                                       prod_sweeps=prod_sweeps_mc,
                                                                       sample_every=sample_every_mc,
                                                                       dlnV_virtual=dlnV_virtual,
                                                                       widom_every=widom_every_mc,
                                                                       widom_ninsert=widom_ninsert_mc)) :
                        run_metropolis_mc(p_mc, st_mc;
                                          burnin_sweeps=burnin_sweeps_mc,
                                          prod_sweeps=prod_sweeps_mc,
                                          sample_every=sample_every_mc,
                                          dlnV_virtual=dlnV_virtual,
                                          widom_every=widom_every_mc,
                                          widom_ninsert=widom_ninsert_mc)
    time_mc = Base.time() - time_mc_start
    println("Finished MC.")

    println("Starting eKMC...")
    time_ekmc_start = Base.time()
    if profile
        timed_val = _print_timed("eKMC run", () -> run_ekmc(p_ekmc, ekst, acc_mu;
                                                            burnin_events=burnin_events_ekmc,
                                                            prod_events=prod_events_ekmc,
                                                            sample_every=sample_every_ekmc,
                                                            dlnV_virtual=dlnV_virtual))
        obs_ekmc, total_time_sim = timed_val
    else
        obs_ekmc, total_time_sim = run_ekmc(p_ekmc, ekst, acc_mu;
                                            burnin_events=burnin_events_ekmc,
                                            prod_events=prod_events_ekmc,
                                            sample_every=sample_every_ekmc,
                                            dlnV_virtual=dlnV_virtual)
    end
    time_ekmc = Base.time() - time_ekmc_start
    println("Finished eKMC.")

    U_mc = mean(obs_mc.U_per_particle)
    U_mc_err = stderr(obs_mc.U_per_particle)
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
    μ_ekmc_err = NaN
    μ_ekmc_lrc = isfinite(μ_ekmc) ? μ_ekmc + μ_lrc : μ_ekmc

    eos_P = Main.MolSim.EOS.pressure_nkeos(T, ρ)
    eos_U = eos_u_per_particle(T, ρ)

    return NVTComparisonResult(
        N, ρ, T, rc, use_lrc, lj_model,
        U_mc, U_mc_err, P_mc, P_mc_err, P_mc_virtual, P_mc_virtual_err, μ_mc, μ_mc_lrc, obs_mc.widom_n, μ_mc_err, obs_mc.U_per_particle.count, time_mc,
        U_ekmc, U_ekmc_err, P_ekmc, P_ekmc_err, P_ekmc_virtual, P_ekmc_virtual_err, μ_ekmc, μ_ekmc_lrc, μ_ekmc_err, total_time_sim, prod_events_ekmc, time_ekmc,
        eos_U, eos_P,
        U_ekmc - U_mc,
        P_ekmc - P_mc,
        μ_ekmc - μ_mc,
        μ_lrc
    )
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
    println("  P (virial)  = $(result.mc_P) ± $(result.mc_P_err)")
    println("  P (virtual) = $(result.mc_P_virtual) ± $(result.mc_P_virtual_err)")
    println("  μ_ex = $(result.mc_mu_ex) ± $(result.mc_mu_ex_err)")
    println("  μ_ex + μ_LRC = $(result.mc_mu_ex_lrc)")
    println("  Widom inserts = $(result.mc_widom_n)")
    println("  Samples = $(result.mc_n_samples), Wall time = $(round(result.mc_wall_time, digits=3)) s")
    println()
    println("eKMC:")
    println("  U/N = $(result.ekmc_U_per_particle) ± $(result.ekmc_U_err)")
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
