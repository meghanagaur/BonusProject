# Define important functions for the SMM.

"""
Simulate EGSS, given parameters defined in tuple m.
Solve the model with finite horizon contract. Solve 
for θ and Y on every point in the productivity grid (z0 = μ).
Then, approximate the effort optimal effort a and wage w,
as a(z) and w(z), i.e. assume an infinite horizon, so we do not have to keep
track of the time period t. Note: u0 = initial unemployment rate.
"""
function simulate(baseline, zshocks; u0 = 0.06)
    
    # Get all of the relevant parameters for the model
    @unpack β, s, κ, hp, zgrid, N_z, ψ, z0_idx, f, ε, σ_η, χ, γ = baseline 

    # Unpack the relevant shocks
    @unpack z_shocks, z_shocks_idx, distr, zstring, burnin, z_ss_dist = zshocks

    # Loop through every point on zgrid
    indexes = cumsum(distr, dims=2)*zShocks.T
    Δlw     = zeros(indexes[end])        # Δlog w_it
    ly      = zeros(indexes[end])        # log y_it 
    w_y_z   = zeros(length(zgrid))       # W/Y
    θ_z     = zeros(length(zgrid))       # θ(z) 
    lw1_z   = zeros(length(zgrid))       # E[log w1(z)] <- wages of new hires
    flag_z  = zeros(Int64,length(zgrid)) # error flags

    Threads.@threads for iz = 1:N_z
        
        modd          = model(z0 = zgrid[iz], ε = ε, σ_η = σ_η, χ = χ) #, γ = γ)
        sol           = solveModel(modd; noisy = false)

        @unpack exit_flag1, exit_flag2, exit_flag3, wage_flag, effort_flag, az, yz, w_0, θ, Y = sol
        
        flag_z[iz]    = maximum([exit_flag1, exit_flag2, exit_flag3, wage_flag, effort_flag])
        z_idx         = modd.z0_idx # index of z on the productivity grid
        
        # log wage of new hires, given z0 = z
        lw1_z[iz]     = log(w_0) - 0.5*(ψ*hp(az[z_idx])*σ_η)^2 
        # tightness, given z0 = z
        θ_z[iz]       = θ             
        w_y_z[iz]     = (w_0./ψ)/Y

        # now, let's think about wages and output for continuing hires
        @views z_shocks_z     = z_shocks[iz]
        @views z_shocks_idx_z = z_shocks_idx[iz]
        η_shocks              = simulateEShocks(σ_η; N = length(z_shocks_idx_z), T = 1)
        
        start_idx    = (iz==1) ? 1 : indexes[iz-1] + 1 
        end_idx      = indexes[iz]
        hp_z0        = hp.(az)  # h'(a(z | z_0 = z_ss))
        @views hp_az = hp_z0[z_shocks_idx_z]
        t1           = ψ*hp_az.*η_shocks 
        t2           = 0.5*(ψ*hp_az*σ_η).^2 
        @views y     = yz[z_shocks_idx_z]
        ηz           = z_shocks_z.*η_shocks

        Δlw[start_idx:end_idx]  = t1 - t2
        ly[start_idx:end_idx]   = log.(y + ηz)
    end

    # Weighted average of labor share
    w_y          = sum(w_y_z.*vec(z_ss_dist))
    
    # Variance of wage changes for incumbents
    var_Δlw      = var(Δlw) 
    
    # Regress wage changes Δ log w_it on individual log output log y_it
    dΔlw_dy = ols(vec(Δlw), vec(ly))[2]

    # Compute simulated series (trim to post-burn-in for z_t when computing moments)
    @unpack z_shocks_idx = zstring
    @views lw1_t  = lw1_z[z_shocks_idx]   # E[log w_1 | z_t]
    @views θ_t    = θ_z[z_shocks_idx]     # θ(z_t)

    # Compute evolution of unemployment for the different z_t paths
    T        = length(θ_t)
    u_t      = zeros(T)
    u_t[1]   = u0
    @views @inbounds for t=2:T
        u_t[t] = (1 - f(θ_t[t-1]))*u_t[t-1] + s*(1 - u_t[t-1])
    end

    # Compute d log w_1 / d u (pooled ols)
    @views dlw1_du = ols(vec(lw1_t[burnin+1:end]), vec(u_t[burnin+1:end]))[2]
    
    # Return simulation results
    return (var_Δlw = var_Δlw, dlw1_du = dlw1_du, dΔlw_dy = dΔlw_dy, w_y = w_y, flag = maximum(flag_z))
end

"""
Run an OLS regression of Y on X.
"""
function ols(Y, X; intercept = true)
    if intercept == true
        X = [ones(size(X,1)) X]
    end
    return inv(X'X)*X'*Y
end

"""
Simulate N X T shock panel for z. Include a burn-in period.
"""
function simulateZShocks(mod; N = 10000, T = 500, z0_idx = median(1:length(zgrid)), seed = 512)

    @unpack P_z, zgrid, σ_η        = mod
    z_shocks, probs, z_shocks_idx  = simulateProd(P_z, zgrid, T; z0_idx = z0_idx, N = N, seed = seed) # N X T 
    
    return (z_shocks = z_shocks, z_shocks_idx = z_shocks_idx, N = N, T = T, z0_idx = z0_idx, seed = seed)
end

"""
Simulate N X T shock panel for η. 
"""
function simulateEShocks(σ_η; N, T, seed = 512)
    Random.seed!(seed)
    η_shocks = rand(Normal(0, σ_η), N, T) # N x T 
    return η_shocks
end