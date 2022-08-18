#=
Define some important helpfer functions for SMM.
=#
using DynamicModel, BenchmarkTools, DataStructures, Distributions, 
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase

"""
Solve for a(z) under the infinite horizon assumption, 
i.e. given z, θ(z), and Y(z).
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
function simulate(baseline, shocks; u0 = 0.06)
    # Get all of the relevant parameters for the model
    @unpack β, s, κ, hp, zgrid, f, ε, σ_η, χ = baseline 
    # Initialize series for each point on zgrid
    θ      = zeros(length(zgrid))
    Y      = zeros(length(zgrid))
    a      = zeros(length(zgrid))
    a_flag = zeros(Int64,length(zgrid)) # effort flag
    c_flag = zeros(Int64,length(zgrid)) # convergence flag
    w0     = zeros(length(zgrid))
    # Solve the model for every point on zgrid
    @inbounds for (iz,z) in enumerate(zgrid)
        modd  = model(z0 = log(z), ε = ε, σ_η = σ_η, χ = χ) #, γ = γ)
        sol   = solveModel(modd; noisy = false)
        θ[iz] = sol.θ
        Y[iz] = sol.Y
        a[iz], w0[iz], a_flag[iz]  = optA(z, modd, Y[iz], θ[iz])
        c_flag[iz]  = sol.exit_flag1
    end
   
    # Get all of the z_t and η_t shocks
    @unpack z_shocks, z_shocks_idx, burnin, N, T, seed = zshocks # all N X T (including burnin-in)
    η_shocks =  simulateEShocks(mod; N = N, T = T - burnin, seed = seed)
    # Compute simulated series (trim to post-burn-in for z_t when computing moments)
    a_z     = a[z_shocks_idx]    # a_t(z_t)
    y_z     = Y[z_shocks_idx]    # E_0 [Y | z_0]
    θ_z     = θ[z_shocks_idx]    # w0_t(z_t)
    w0_z    = w0[z_shocks_idx]   # w0 (constant)
    hp_az   = hp.(a_z)           # hprime
    ψ       = 1 - β*(1-s)        # infinite horizon pass-through coefficient 

    # Compute log wage changes given η_t and z_t shocks (discard burnin for z_t)
    Δlw      = ψ*hp_az[:,burnin+1:end].*η_shocks - 0.5*(ψ*hp_az[:,burnin+1:end]*σ_η).^2
    var_Δlw  = var(Δlw) 

    # Compute expected log wage at time 1 (given z_t)
    lw1       = log.(w0_z) -  0.5*(ψ*hp_az*σ_η).^2

    # Compute individual log output y_it
    ly       = log.(z_shocks[:,burnin+1:end].*(a_z[:,burnin+1:end] + η_shocks))

    # Regress Δ log w_it on log y_it
    dΔlw_dy = ols(vec(Δlw), vec(ly))

    # Compute evolution of unemployment for the different z_t paths
    T       = size(z_shocks,2)
    u       = zeros(size(z_shocks))
    u[:,1] .= u0
   @views @inbounds for t=2:T
        u[:,t] .= (1 .- f.(θ_z[:,t-1])).*u[:,t-1] + s*(1 .- u[:,t-1])
    end
    # Compute d log w_1 / d u (pooled ols)
    dlw1_du = ols(vec(lw1[:,burnin+1:end]), vec(u[:,burnin+1:end]))

    # Return simulation results
    return (var_Δlw = var_Δlw, dlw1_du = dlw1_du, dΔlw_dy = dΔlw_dy,
    θ = θ, c_flag = c_flag, a_flag = a_flag) 
end

"""
Run an OLS regression of Y on X.
"""
function ols(Y,X)
    return inv(X'X)*X'*Y
end

"""
Simulate N X T shock panel for z. Include a burn-in period.
"""
function simulateZShocks(mod; burnin = 1000, N = 10000, T = 5000 + burnin, seed = 512)
    @unpack P_z, zgrid, σ_η        =  mod
    z_shocks, probs, z_shocks_idx  = simulateProd(P_z, zgrid, T; N = N, set_seed = seed) # N X T 
    return (z_shocks = z_shocks, z_shocks_idx = z_shocks_idx, burnin = burnin, 
    N = N, T = T, seed = seed)
end

"""
Simulate N X T shock panel for η. 
"""
function simulateEShocks(mod; N, T, seed)
    Random.seed!(seed)
    η_shocks = rand(Normal(0, σ_η), N, T) # N x T 
    return η_shocks
end