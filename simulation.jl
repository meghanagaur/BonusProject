#=
Helper functions for SMM.
=#
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
Then, approximate the effort optimal effort a and wage w,
as a(z) and w(z), i.e. assume an infinite horizon, so we do not have to keep
track of the time period t. Note: u0 = initial unemployment rate.
"""
function simulate(endogParams, shocks; u0 = 0.06)
    ε   = endogParams[1] 
    σ_η = endogParams[2]
    χ   = endogParams[3]
    γ   = endogParams[4]
    # speed this up
    @unpack β, s, κ, hp, zgrid, f = model(ε = ε, σ_η = σ_η, χ = χ, γ = γ)

    θ      = zeros(length(zgrid))
    Y      = zeros(length(zgrid))
    a      = zeros(length(zgrid))
    a_flag = zeros(Int64,length(zgrid)) # effort flag
    c_flag = zeros(Int64,length(zgrid)) # convergence flag
    w0     = zeros(length(zgrid))
    @inbounds for (iz,z) in enumerate(zgrid)
        modd  = model(z0 = log(z), ε = ε, σ_η = σ_η, χ = χ, γ = γ)
        sol   = solveModel(modd; noisy = false)
        θ[iz] = sol.θ
        Y[iz] = sol.Y
        a[iz], w0[iz], a_flag[iz]  = optA(z, modd, Y[iz], θ[iz])
        c_flag[iz]  = sol.exit_flag1
    end

    # store simulated series
    @unpack η_shocks, z_shocks, z_shocks_idx, burnin = shocks
    a_z     = a[z_shocks_idx]    # a_t(z_t)
    y_z     = Y[z_shocks_idx]    # y_t(z_t)
    θ_z     = θ[z_shocks_idx]    # w0_t(z_t)
    w0_z    = w0[z_shocks_idx]   # w0 (constant)
    hp_az   = hp.(a_z)           # hprime
    ψ       = 1 - β*(1-s)        # infinite horizon pass-through coefficient 
    # Compute wage changes given η_t and z_t shocks
    delta_w = ψ*hp_az[burnin+1:end,:].*η_shocks - 0.5*(ψ*hp_az[burnin+1:end,:]*σ_η).^2
    σ_Δw    = var(delta_w) 
    # Compute expected log wage at time 1 
    w1      = log.(w0_z) -  0.5*(ψ*hp_az*σ_η).^2
    # Compute evolution of unemployment for the different z_t paths
    T       = size(z_shocks,1)
    u       = zeros(size(z_shocks))
    u[1,:] .= u0
    @inbounds for t=2:T
        u[t,:] .= (1 .- f.(θ_z[t-1,:])).*u[t-1,:] + s*(1 .- u[t-1,:])
    end
    # Compute d log w_1 / d u given z_t
    dw1_du = ols(vec(w1[burnin+1:end]), vec(u[burnin+1:end]))

    return (σ_Δw = σ_Δw, dw1_du = dw1_du, θ = θ, c_flag = c_flag, a_flag = a_flag) 
end

"""
Run OLS regression of Y on X.
"""
function ols(Y,X)
    return inv(X'X)*X'*Y
end

"""
Simulate the shock panel for z and η.
Allow a burnin period of 1000.
"""
function simulateShocks(mod; burnin = 1000, N = 1000, T = 5000 + burnin, seed = 512)
    Random.seed!(seed)
    @unpack P_z, zgrid             =  model()
    # simulate shocks
    z_shocks, probs, z_shocks_idx  = simulateProd(P_z, zgrid, T; N = N, set_seed = false) # T X N
    η_shocks                       = rand(Normal(0, endogParams[2]), T-burnin, N)          # T x N
    return (z_shocks = z_shocks, z_shocks_idx = z_shocks_idx, η_shocks = η_shocks, burnin = burnin)
end