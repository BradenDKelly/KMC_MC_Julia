using Test
using MolSim

@testset "Nezbeda-Kolafa EOS regression" begin
    # Ideal-gas limit at very low density
    T_ig = 1.0
    ρ_ig = 1e-6
    P_ig = MolSim.EOS.pressure(T_ig, ρ_ig)
    Z_ig = P_ig / (ρ_ig * T_ig)
    @test abs(Z_ig - 1.0) < 1e-5

    # Table 1 data from Kolafa & Nezbeda (1994): P and U with reported errors
    table1 = [
        (T=0.81, ρ=0.8645, P=1.0954,   P_err=0.0036, U=-6.10212, U_err=0.00053),
        (T=1.2,  ρ=0.5,    P=0.02816,  P_err=0.007,  U=-3.4809,  U_err=0.006),
        (T=1.2,  ρ=0.7,    P=0.6710,   P_err=0.006,  U=-4.7593,  U_err=0.0018),
        (T=1.3,  ρ=0.2,    P=0.12113,  P_err=0.00087,U=-1.5685,  U_err=0.0038),
        (T=1.3,  ρ=0.4,    P=0.1117,   P_err=0.0045, U=-2.8293,  U_err=0.008),
        (T=1.3,  ρ=0.5,    P=0.15495,  P_err=0.004,  U=-3.4204,  U_err=0.0025),
        (T=1.3,  ρ=0.6,    P=0.35725,  P_err=0.005,  U=-4.0555,  U_err=0.0026),
        (T=1.4,  ρ=0.2,    P=0.15148,  P_err=0.00081,U=-1.4912,  U_err=0.003),
        (T=1.4,  ρ=0.4,    P=0.19532,  P_err=0.0045, U=-2.7597,  U_err=0.0035),
        (T=1.45, ρ=0.3,    P=0.1988,   P_err=0.0012, U=-2.1179,  U_err=0.0017),
        (T=4.85, ρ=1.0,    P=31.474,   P_err=0.040,  U=-2.2925,  U_err=0.008),
        (T=10.0, ρ=1.0,    P=54.032,   P_err=0.030,  U=1.4171,   U_err=0.0060),
        (T=10.0, ρ=1.2,    P=99.178,   P_err=0.060,  U=5.4480,   U_err=0.010),
    ]

    # Allow a few-sigma envelope around reported errors
    for pt in table1
        P = MolSim.EOS.pressure(pt.T, pt.ρ)
        U = MolSim.EOS.internal_energy(pt.T, pt.ρ)
        @test abs(P - pt.P) <= 5.0 * pt.P_err
        @test abs(U - pt.U) <= 5.0 * pt.U_err
    end
end
