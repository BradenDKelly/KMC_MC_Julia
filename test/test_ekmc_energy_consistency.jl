using Test
using MolSim

@testset "eKMC pair_u energy consistency" begin
    # Use no LRC so total_energy matches 0.5 * sum(pair_u)
    p, st = MolSim.MC.init_fcc(N=108, ρ=0.8, T=1.0, rc=2.5,
                               max_disp=0.1, seed=1234, use_lrc=false)

    ekst = MolSim.MC.init_ekmc_state(st, p)

    pair_energy = 0.5 * sum(ekst.pair_u)
    total = MolSim.MC.total_energy(st, p)

    @test isfinite(pair_energy)
    @test abs(pair_energy - total) < 1e-10
end
