using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using Printf

function main()
    # Statepoints to compare (low density only - virial EOS is only valid at ρ < 0.2)
    statepoints = [
        (T=1.0, ρ=0.01),
        (T=1.0, ρ=0.05),
        (T=1.0, ρ=0.1),
        (T=1.5, ρ=0.05),
        (T=1.5, ρ=0.1)
    ]
    
    println("MC vs EOS Comparison (LJ Virial EOS - Low Density Only)")
    println("=" ^ 80)
    println("NOTE: Virial EOS is only valid at low density (ρ < 0.2)")
    println("=" ^ 80)
    println(@sprintf("%-6s %-6s %-15s %-15s %-15s %-15s %-10s %-10s %-10s",
                     "T", "ρ", "MC u/N", "EOS u", "MC p", "EOS p", "Z", "Δu%", "Δp%"))
    println("-" ^ 80)
    
    for sp in statepoints
        T = sp.T
        ρ = sp.ρ
        
        # Warn if density is too high for virial EOS
        if ρ > 0.2
            println()
            println("=" ^ 80)
            println("WARNING: Virial EOS not valid at this density (ρ = $ρ > 0.2)")
            println("Shown for qualitative reference only.")
            println("=" ^ 80)
            println()
        end
        
        # Run short MC simulation
        p_mc, st = MolSim.MC.init_fcc(N=864, ρ=ρ, T=T, rc=2.5, max_disp=0.1, seed=42, use_lrc=true)
        T_val = 1.0 / p_mc.β
        
        # Warmup
        for _ in 1:50
            MolSim.MC.sweep!(st, p_mc)
        end
        
        # Production (short for speed)
        prod_sweeps = 200
        sample_every = 20
        total_acc = 0.0
        energy_samples = Float64[]
        pressure_samples = Float64[]
        
        for sweep_idx in 1:prod_sweeps
            acc = MolSim.MC.sweep!(st, p_mc)
            total_acc += acc
            
            if sweep_idx % sample_every == 0
                E_total = MolSim.MC.total_energy(st, p_mc)
                u_per_particle = E_total / st.N
                P = MolSim.MC.pressure(st, p_mc, T_val)
                push!(energy_samples, u_per_particle)
                push!(pressure_samples, P)
            end
        end
        
        # MC averages (corrected values)
        u_mc = sum(energy_samples) / length(energy_samples)
        p_mc = sum(pressure_samples) / length(pressure_samples)
        
        # EOS values
        u_eos = MolSim.EOS.internal_energy(T, ρ)
        p_eos = MolSim.EOS.pressure(T, ρ)
        
        # Compressibility factor
        Z = p_eos / (ρ * T)
        
        # Percent differences
        Δu_pct = 100.0 * (u_mc - u_eos) / abs(u_eos)
        Δp_pct = 100.0 * (p_mc - p_eos) / abs(p_eos)
        
        println(@sprintf("%-6.1f %-6.1f %-15.6f %-15.6f %-15.6f %-15.6f %-10.4f %-10.2f %-10.2f",
                         T, ρ, u_mc, u_eos, p_mc, p_eos, Z, Δu_pct, Δp_pct))
    end
end

main()
