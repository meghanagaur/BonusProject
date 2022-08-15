using DynamicModel, BenchmarkTools, DataStructures, Distributions, 
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase

"""
Given z, solve for a(z) under the infinite horizon assumption.
""" 
function optA(mod)
    @unpack β, s, κ, ε, σ_η = m  

end

"""
Simulate EGSS, given parameters defined in tuple m.
Solve the model with finite horizon contract. Solve 
for θ and Y on every point in the productivity grid (z0 = μ).
Then, approximate effort optimal effort a as a(z), i.e. assuming
the contract is in the infinite horizon, so we do not have to keep
track of the time period t.
"""
function simulate(m)
    m = model()
    @unpack T, β, r, s, κ, ι, ε, σ_η, ω, N_z, q, u, h, hp, zgrid, P_z, ψ, procyclical = m  
    
    θ = zeros(length(zgrid))
    Y = zeros(length(zgrid))

    @inbounds for (iz,z) in enumerate(zgrid)
       sol   = solveModel(model(z0 = log(z)); noisy = false)
       θ[iz] = sol.θ
       Y[iz] = sol.Y
    end
end