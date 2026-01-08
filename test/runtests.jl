using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MolSim
using Test
using StaticArrays

@testset "wrap! tests" begin
    L = 10.0
    x = MVector{3,Float64}(-2.5, 5.0, 12.3)
    MolSim.MC.wrap!(x, L)
    @test all(0.0 .<= x .< L)
    @test x[1] ≈ 7.5
    @test x[2] ≈ 5.0
    @test x[3] ≈ 2.3
end

@testset "minimum_image! tests" begin
    L = 10.0
    dr = MVector{3,Float64}(6.0, -6.0, 3.0)
    MolSim.MC.minimum_image!(dr, L)
    @test all(abs.(dr) .<= L/2)
    @test dr[1] ≈ -4.0
    @test dr[2] ≈ 4.0
    @test dr[3] ≈ 3.0
end

@testset "total_energy translation invariance" begin
    p, st = MolSim.MC.init_fcc(N=108, ρ=0.8, T=1.0, rc=2.5, max_disp=0.1, seed=42)
    E1 = MolSim.MC.total_energy(st, p)
    
    # Translate all particles by constant vector
    shift = [2.3, 1.7, 0.5]
    @inbounds for i in 1:st.N
        st.pos[1, i] += shift[1]
        st.pos[2, i] += shift[2]
        st.pos[3, i] += shift[3]
    end
    
    # Wrap all positions
    scratch = MVector{3,Float64}(0.0, 0.0, 0.0)
    @inbounds for i in 1:st.N
        scratch[1] = st.pos[1, i]
        scratch[2] = st.pos[2, i]
        scratch[3] = st.pos[3, i]
        MolSim.MC.wrap!(scratch, st.L)
        st.pos[1, i] = scratch[1]
        st.pos[2, i] = scratch[2]
        st.pos[3, i] = scratch[3]
    end
    
    E2 = MolSim.MC.total_energy(st, p)
    @test abs(E1 - E2) < 1e-10
end
