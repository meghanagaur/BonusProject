using IntervalArithmetic, IntervalRootFinding

sol = solveModel(modd; tol1 = 10^(-13), tol2 = 10^(-13), tol3 =  10^(-13) )
aa  = zeros(modd.N_z)

@unpack ψ, ε, q, κ, hp, σ_η, hbar = modd

a_min = 10^(-8)
a_max = 20

for (iz,z) in enumerate(modd.zgrid)
    println(roots( x -> x - ((z*x/sol.w_0 - (ψ/ε)*(hp(x)*σ_η)^2)/hbar)^(ε/(1+ε)) ,  a_min..a_max))
end

function solveA(x, ε, modd, z, w_0)
    @unpack ψ, q, κ, σ_η, hbar = modd
    hp(a) = hbar*max(a, 0)^(1/ε) 
    return (x - max( (z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2)/hbar, eps() )^(ε/(1+ε))) 
end

#look at what we get for diferent values of ε
p1 = plot(legend=:topleft)
for e in 0.3:0.5:2.0
    plot!(p1, x -> solveA(x, e, modd, modd.zgrid[1], sol.w_0)[1], 0, 1.2 , label=L"\varepsilon = "*string(e))
end
p1