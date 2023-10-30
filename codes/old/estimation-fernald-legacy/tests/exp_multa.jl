using IntervalArithmetic, IntervalRootFinding

sol = solveModel(modd; tol1 = 10^(-13), tol2 = 10^(-13), tol3 =  10^(-13) )

@unpack ψ, ε, q, κ, hp, σ_η, hbar = modd

a_min = 10^(-6)
a_max = 20

for (iz,z) in enumerate(modd.zgrid)
    println(roots( x -> x - ((z*x/sol.w_0 - (ψ/ε)*(hp(x)*σ_η)^2)/hbar)^(ε/(1+ε)) ,  a_min..a_max))
end
