# Define important functions for the SMM.

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
    @unpack β, s, κ, hp, zgrid, f, ε, σ_η, χ, ψ, z0_idx = baseline 

    # Initialize series for each point on zgrid 
    θ_z      = zeros(length(zgrid))  # θ(z) 
    lw1_z    = zeros(length(zgrid))  # E[log w1(z)] <- wages of new hires
    w1_z     = zeros(length(zgrid))  # E[w1(z)] (= w_0(z) by martingale property)
    a_z      = zeros(length(zgrid))  # a(z | z_0 = z)
    flag_z   = zeros(Int64,length(zgrid)) # flags

    # Loop through every point on zgrid
    sol    = OrderedDict{Int64, Any}()
    @inbounds for (iz,z) in enumerate(zgrid)
        
        modd          = model(z0 = z, ε = ε, σ_η = σ_η, χ = χ) #, γ = γ)
        sol[iz]       = solveModel(modd; noisy = false)

        @unpack exit_flag1, exit_flag2, exit_flag3, wage_flag, effort_flag, az, w_0, θ = sol[iz]
        z_idx         = modd.z0_idx
        flag_z[iz]    = maximum([exit_flag1, exit_flag2, exit_flag3, wage_flag, effort_flag])
        θ_z[iz]       = θ             
        a_z[iz]       = az[z_idx]     
        lw1_z[iz]     = log(w_0) - 0.5*(ψ*hp(a_z[iz])*σ_η)^2 
        w1_z[iz]      = w_0           
    end

    # for regressions with initial z0 FIXED at z_ss (continuing hires)
    az_z0 = sol[z0_idx].az  # a(z | z_0 = z_ss)
    hp_z0 = hp.(az_z0)      # h'(a(z | z_0 = z_ss))
    yz_z0 = sol[z0_idx].yz  # z*a(z | z_0 = z_ss)

    # Get all of the z_t and η_t shocks <- beginning at z_ss
    @unpack z_shocks, z_shocks_idx, burnin, N, T, seed = zshocks # all N X T (including burnin-in)
    η_shocks =  simulateEShocks(baseline; N = N, T = T - burnin, seed = seed)
    
    # Compute simulated series (trim to post-burn-in for z_t when computing moments)
    @views lw1   = lw1_z[z_shocks_idx]  # E[w_1 | z_t]
    @views θ     = θ_z[z_shocks_idx]    # θ(z_t)

    # Compute log wage changes for continuing workers given η_t and z_t shocks (discard burnin for z_t)
    @views hp_az = hp_z0[z_shocks_idx][:,burnin+1:end]
    t1           = ψ*hp_az.*η_shocks 
    t2           = 0.5*(ψ*hp_az*σ_η).^2 
    Δlw          = t1 - t2
    var_Δlw      = var(Δlw) 
    histogram(vec(Δlw))

    # Compute individual log output y_it
    @views y   = yz_z0[z_shocks_idx][:,burnin+1:end]
    @views ηz  = z_shocks[:,burnin+1:end].*η_shocks
    ly         = log.(y + ηz)

    # Regress wage changes Δ log w_it on individual log output log y_it
    dΔlw_dy = ols(vec(Δlw), vec(ly))

    # Compute evolution of unemployment for the different z_t paths
    T       = size(z_shocks,2)
    u       = zeros(size(z_shocks))
    u[:,1] .= u0
    @views @inbounds for t=2:T
        u[:,t] .= (1 .- f.(θ[:,t-1])).*u[:,t-1] + s*(1 .- u[:,t-1])
    end

    # Compute d log w_1 / d u (pooled ols)
    @views dlw1_du = ols(vec(lw1[:,burnin+1:end]), vec(u[:,burnin+1:end]))
    
    # Return simulation results
    return (var_Δlw = var_Δlw, dlw1_du = dlw1_du, dΔlw_dy = dΔlw_dy, θ = θ, flag = maximum(flag_z))
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