using DynamicModel, BenchmarkTools, DataStructures, Distributions, 
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase

"""
Solve for a(z) under the infinite horizon assumption, 
given z, θ(z), and Y(z).
""" 
function optA(z, modd, Y, θ)
    @unpack β, s, κ, ε, σ_η, q, hp, κ = modd
    ψ  = 1 - β*(1-s)       # infinite horizon pass-through coefficient 
    w0 = ψ*(Y - κ/q(θ))    # time-0 earnings (constant)
    
    if ε == 1 # can solve analytically
        aa = (z/w0 + sqrt((z/w0)^2))/2(1 + ψ*σ_η^2)
    else # exclude the choice of zero effort
        aa = find_zeros(x -> x - max(z*x/w0 -  (ψ/ε)*(hp(x)*σ_η)^2, 0)^(ε/(1+ε)) + Inf*(x==0), 0, 20) 
    end
    a    = ~isempty(aa) ? aa[1] : 0
    flag = ~isempty(aa) ? ((z*aa[1]/w0 + (ψ/ε)*(hp(aa[1])*σ_η)^2) < 0) : isempty(aa) 
    flag += (w0 < 0)
    return a, max(w0, eps()), flag
end

"""
Simulate EGSS, given parameters defined in tuple m.
Solve the model with finite horizon contract. Solve 
for θ and Y on every point in the productivity grid (z0 = μ).
Then, approximate effort optimal effort a as a(z), i.e. assuming
the contract is in the infinite horizon, so we do not have to keep
track of the time period t. u0 = initial unemployment rate.
"""
function simulate(endogParams, η_shocks, z_shocks, z_shocks_idx; u0 = 0.06)
    ε   = endogParams[1] 
    σ_η = endogParams[2]
    χ   = endogParams[3]
    γ   = endogParams[4]
    @unpack β, s, κ, hp, zgrid, f = model(ε = ε, σ_η = σ_η, χ = χ, γ = γ)

    θ    = zeros(length(zgrid))
    Y    = zeros(length(zgrid))
    a    = zeros(length(zgrid))
    flag = zeros(Int64,length(zgrid))
    w0   = zeros(length(zgrid))
    @inbounds for (iz,z) in enumerate(zgrid)
        modd  = model(z0 = log(z), ε = ε, σ_η = σ_η, χ = χ, γ = γ)
        sol   = solveModel(modd; noisy = false)
        θ[iz] = sol.θ
        Y[iz] = sol.Y
        a[iz], w0[iz], flag[iz]  = optA(z, modd, Y[iz], θ[iz])
    end

    # store simulated series
    a_z     = a[z_shocks_idx]
    y_z     = Y[z_shocks_idx]
    θ_z     = θ[z_shocks_idx]
    w0_z    = w0[z_shocks_idx]
    hp_az   = hp.(az)

    # Compute wage changes
    ψ       = 1 - β*(1-s)       # infinite horizon pass-through coefficient 
    delta_w = ψ*hp_az.*η_shocks - 0.5*(ψ*hp_az*σ_η).^2
    σ_Δw    = var(delta_w) 
    # Compute expected log wage at time 1 
    w1      = log.(w0_z) -  0.5*(ψ*hp_az*σ_η).^2
    # Compute evolution of unemployment for different z paths
    T       = size(z_shocks,1)
    u       = zeros(size(z_shocks))
    u[1,:] .= u0
    @inbounds for t=2:T
        u[t,:] .= u[t-1,:].*(1 .- f.(θ_z[t-1,:])) + s*(1 .- u[t-1,:])
    end
    # Compute d log w_1 / d u given z_t
    dw1_du = regress(vec(w1), vec(u))

    return σ_Δw, dw1_du, flag
end

"""
OLS regression
"""
function regress(Y,X)
    return inv(X'X)*X'*Y
end

#= for wrapper script 
Simulate shocks externally
Export flag matrix
Construct objective function, covariance matrix
=#
 m = model(savings = false, procyclical = true)
ZZ, probs, IZ  = simulateProd(P_z, zgrid, T) # T X N

endogParams    = zeros(4)
endogParams[1] = 0.5                    # ε
endogParams[2] = 0.05                   # σ_η
endogParams[3] = 0.1                    # χ
endogParams[4] = 0.66 - endogParams[3]  # γ
η_shocks       = rand(Normal(0,endogParams[2]), size(ZZ))
