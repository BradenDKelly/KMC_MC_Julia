"""
NKEOS: Kolafa-Nezbeda 1994 Lennard-Jones EOS (Python reference port).

Ported directly from the provided Python implementation. All quantities are in
reduced units (ε = σ = k_B = 1).
"""

const NKEOS_GAMMA = 1.92907278
const NKEOS_PI = 3.141592654

# CAlj coefficient table (indices map to i = 0,1,2,4 and j = 2..6)
const NKEOS_CALJ = let C = zeros(Float64, 7, 7)
    C[1, 3] = 2.01546797
    C[1, 4] = -28.17881636
    C[1, 5] = 28.28313847
    C[1, 6] = -10.42402873
    C[2, 3] = -19.58371655
    C[2, 4] = 75.62340289
    C[2, 5] = -120.70586598
    C[2, 6] = 93.92740328
    C[2, 7] = -27.37737354
    C[3, 3] = 29.34470520
    C[3, 4] = -112.35356937
    C[3, 5] = 170.64908980
    C[3, 6] = -123.06669187
    C[3, 7] = 34.42288969
    C[5, 3] = -13.37031968
    C[5, 4] = 65.38059570
    C[5, 5] = -115.09233113
    C[5, 6] = 88.91973082
    C[5, 7] = -25.62099890
    C
end

const NKEOS_CDH_BH = let C = zeros(Float64, 8)
    C[8] = -0.58544978
    C[7] = 0.43102052
    C[6] = 0.87361369
    C[5] = -4.13749995
    C[4] = 2.90616279
    C[3] = -7.02181962
    C[1] = 0.02459877
    C
end

@inline function nkeos_zhs(η::Float64)::Float64
    return (1.0 + η * (1.0 + η * (1.0 - η / 1.5 * (1.0 + η)))) / (1.0 - η)^3
end

@inline function nkeos_betaAHS(η::Float64)::Float64
    return log(1.0 - η) / 0.6 + η * ((4.0 / 6.0 * η - 33.0 / 6.0) * η + 34.0 / 6.0) / (1.0 - η)^2
end

@inline function nkeos_dC(T::Float64)::Float64
    sT = sqrt(T)
    return -0.063920968 * log(T) + 0.011117524 / T - 0.076383859 / sT + 1.080142248 + 0.000693129 * sT
end

@inline function nkeos_dCdT(T::Float64)::Float64
    sT = sqrt(T)
    return 0.063920968 * T + 0.011117524 + (-0.5 * 0.076383859 - 0.5 * 0.000693129 * T) * sT
end

@inline function nkeos_dB2hBH(T::Float64)::Float64
    sumv = 0.0
    for i in -7:0
        if i == -1
            continue
        end
        sumv += NKEOS_CDH_BH[abs(i)+1] * T^(i / 2.0)
    end
    return sumv
end

@inline function nkeos_BC(T::Float64)::Float64
    isT = 1.0 / sqrt(T)
    sum1 = (((-0.58544978 * isT + 0.43102052) * isT + 0.87361369) * isT - 4.13749995) * isT + 2.90616279
    return (sum1 * isT - 7.02181962) / T + 0.02459877
end

@inline function nkeos_BCdT(T::Float64)::Float64
    isT = 1.0 / sqrt(T)
    sum1 = ((-0.58544978 * 3.5 * isT + 0.43102052 * 3.0) * isT + 0.87361369 * 2.5) * isT -
           4.13749995 * 2.0
    return (sum1 * isT + 2.90616279 * 1.5) * isT - 7.02181962
end

@inline function nkeos_sumCALJ(T::Float64, ρ::Float64)::Float64
    sumv = 0.0
    for i in -4:0
        if i == -3
            continue
        end
        for j in 2:6
            if i == 0 && j == 6
                continue
            end
            sumv += NKEOS_CALJ[abs(i)+1, j+1] * T^(i / 2.0) * ρ^j
        end
    end
    return sumv
end

@inline function nkeos_DALJ(T::Float64, ρ::Float64)::Float64
    sum1 = 2.01546797 + ρ * (-28.17881636 + ρ * (28.28313847 + ρ * (-10.42402873)))
    sum2 = -19.58371655 + ρ * (75.62340289 + ρ * ((-120.70586598) + ρ * (93.92740328 + ρ * (-27.37737354))))
    sum2 /= sqrt(T)
    sum3 = 29.34470520 + ρ * ((-112.35356937) + ρ * (170.64908980 + ρ * ((-123.06669187) + ρ * 34.42288969)))
    sum4 = -13.37031968 + ρ * (65.38059570 + ρ * ((-115.09233113) + ρ * (88.91973082 + ρ * (-25.62099890))))
    sum4 /= T
    return (sum1 + sum2 + (sum3 + sum4) / T) * ρ * ρ
end

"""
    nkeos_alj_res(T, ρ)

Residual Helmholtz free energy (without ideal term), reduced units.
"""
function nkeos_alj_res(T::Float64, ρ::Float64)::Float64
    η = NKEOS_PI / 6.0 * ρ * nkeos_dC(T)^3
    return (nkeos_betaAHS(η) + ρ * nkeos_BC(T) / exp(NKEOS_GAMMA * ρ^2)) * T + nkeos_DALJ(T, ρ)
end

"""
    nkeos_free_energy(T, ρ)

Helmholtz free energy including ideal term, reduced units.
"""
function nkeos_free_energy(T::Float64, ρ::Float64)::Float64
    η = NKEOS_PI / 6.0 * ρ * nkeos_dC(T)^3
    alj = log(ρ) + nkeos_betaAHS(η) + ρ * nkeos_BC(T) / exp(NKEOS_GAMMA * ρ^2)
    return alj * T + nkeos_DALJ(T, ρ)
end

"""
    pressure_nkeos(T, ρ)

Pressure from the NKEOS (Python reference) formulation.
"""
function pressure_nkeos(T::Float64, ρ::Float64)::Float64
    η = NKEOS_PI / 6.0 * ρ * nkeos_dC(T)^3
    sum1 = 2.01546797 * 2.0 + ρ * (-28.17881636 * 3.0 + ρ * (28.28313847 * 4.0 + ρ * (-10.42402873) * 5.0))
    sum2 = -19.58371655 * 2.0
    sum2 += ρ * (75.62340289 * 3.0 + ρ * (-120.70586598 * 4.0 + ρ * (93.92740328 * 5.0 + ρ * (-27.37737354) * 6.0)))
    sum2 /= sqrt(T)
    sum3 = 29.34470520 * 2.0 + ρ * ((-112.35356937) * 3.0 + ρ * (170.64908980 * 4.0 + ρ * ((-123.06669187) * 5.0 + ρ * 34.42288969 * 6.0)))
    sum4 = -13.37031968 * 2.0 + ρ * (65.38059570 * 3.0 + ρ * (-115.09233113 * 4.0 + ρ * (88.91973082 * 5.0 + ρ * (-25.62099890) * 6.0)))
    sum4 /= T
    sum5 = (sum1 + sum2 + (sum3 + sum4) / T) * ρ^2

    plj = ((nkeos_zhs(η) + nkeos_BC(T) / exp(NKEOS_GAMMA * ρ^2) * ρ * (1.0 - 2.0 * NKEOS_GAMMA * ρ^2)) * T + sum5) * ρ
    return plj
end

"""
    internal_energy_nkeos(T, ρ)

Internal energy from the NKEOS (Python reference) formulation.
"""
function internal_energy_nkeos(T::Float64, ρ::Float64)::Float64
    dBHdT = nkeos_dCdT(T)
    dB2BHdT = nkeos_BCdT(T)
    d = nkeos_dC(T)
    η = NKEOS_PI / 6.0 * ρ * d^3
    sum1 = 2.01546797 + ρ * ((-28.17881636) + ρ * (28.28313847 + ρ * (-10.42402873)))
    sum2 = -19.58371655 * 1.5 + ρ * (75.62340289 * 1.5 + ρ * ((-120.70586598) * 1.5 + ρ * (93.92740328 * 1.5 + ρ * (-27.37737354) * 1.5)))
    sum2 /= sqrt(T)
    sum3 = 29.34470520 * 2.0 + ρ * (-112.35356937 * 2.0 + ρ * (170.64908980 * 2.0 + ρ * (-123.06669187 * 2.0 + ρ * 34.42288969 * 2.0)))
    sum4 = -13.37031968 * 3.0 + ρ * (65.38059570 * 3.0 + ρ * (-115.09233113 * 3.0 + ρ * (88.91973082 * 3.0 + ρ * (-25.62099890) * 3.0)))
    sum4 /= T
    sum5 = (sum1 + sum2 + (sum3 + sum4) / T) * ρ * ρ
    return 3.0 * (nkeos_zhs(η) - 1.0) * dBHdT / d + ρ * dB2BHdT / exp(NKEOS_GAMMA * ρ^2) + sum5
end

"""
    pressure(T, ρ)

Compatibility wrapper for EOS pressure in reduced units.
"""
pressure(T::Float64, ρ::Float64)::Float64 = pressure_nkeos(T, ρ)

"""
    internal_energy(T, ρ)

Compatibility wrapper for EOS internal energy per particle in reduced units.
"""
internal_energy(T::Float64, ρ::Float64)::Float64 = internal_energy_nkeos(T, ρ)

"""
    chemical_potential_nkeos(T, ρ)

Residual chemical potential in reduced units.
For a pure component: μ_res = a_res + P_res/ρ
where a_res = A_res/N (residual Helmholtz free energy per particle).
"""
function chemical_potential_nkeos(T::Float64, ρ::Float64)::Float64
    a_res = nkeos_alj_res(T, ρ)  # A_res/N (per particle)
    P_res = pressure_nkeos(T, ρ)
    return a_res + P_res / ρ
end
