

sol = solveModel(modd)


aa=zeros(modd.N_z)
@unpack ψ, ε, q, κ, hp, σ_η, hbar = modd
a_min = 10^-8
a_max = 20

for (iz,z) in enumerate(modd.zgrid)
    #aa[iz] = solve(ZeroProblem( x -> (x > a_min)*(x - max( (z*x/sol.w_0 - (ψ/ε)*(hp(x)*σ_η)^2)/hbar, eps() )^(ε/(1+ε))) + (x <= a_min)*10^10, 1.0))
    println(find_zeros( x -> (x > a_min)*(x - max( (z*x/sol.w_0 - (ψ/ε)*(hp(x)*σ_η)^2)/hbar, eps() )^(ε/(1+ε))) + (x <= a_min)*10^10,  a_min, a_max))
end

function optAA(e, modd, z, w_0)
    @unpack ψ, ε, q, κ, hp, σ_η, hbar = modd
    a    =  find_zeros( x -> (x > a_min)*(x - max( (z*x/w_0 - (ψ/e)*(hp(x)*σ_η)^2)/hbar, eps() )^(e/(1+e))) + (x <= a_min)*10^10,  a_min, a_max)[1]
    gap = a -  ((z*a/w_0 - (ψ/e)*(hp(a)*σ_η)^2)/hbar)^(e/(1+e))

    return a, gap
end

plot(e->optAA(e, modd, modd.zgrid[1], sol.w_0)[1], 0, 5 )
plot(e->optAA(e, modd, modd.zgrid[1], sol.w_0)[2], 0, 5 )