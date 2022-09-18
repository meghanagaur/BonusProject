# Define important functions for the SMM.
"""
Simulate EGSS, given parameters defined in tuple m.
Solve the model with infinite horizon contract. Solve 
for θ and Y on every point in the productivity grid.
Then, compute the effort optimal effort a and wage w,
as a(z|z_0) and w(z|z_0). 
Note: u0 = initial unemployment rate.
"""
function simulate(baseline, zshocks; u0 = 0.06)
    
    # initialize moments to export (default to NaN)
    std_Δlw = NaN
    dlw1_du = NaN
    dΔlw_dy = NaN
    u_ss    = NaN

    # Get all of the relevant parameters for the model
    @unpack β, s, κ, hp, zgrid, N_z, ψ, z_1_idx, f, ε, σ_η, χ, γ = baseline 

    # Unpack the relevant shocks
    @unpack z_shocks, z_shocks_idx, λ_N_z, zstring, burnin, z_ss_dist = zshocks

    # Loop through every point on zgrid
    indices = cumsum(λ_N_z, dims=2)*zshocks.T
    Δlw     = zeros(indices[end])        # Δlog w_it, given z_1 and z_t
    ly      = zeros(indices[end])        # log y_it, given z_1 and z_t
    w_y_z   = zeros(length(zgrid))       # PV of W/Y, given z_1
    θ_z     = zeros(length(zgrid))       # θ(z_1)
    lw1_z   = zeros(length(zgrid))       # E[log w1|z] <- wages of new hires
    flag_z  = zeros(Int64,length(zgrid)) # error flags

    Threads.@threads for iz = 1:N_z
        
        modd          = model(z_1 = zgrid[iz], ε = ε, σ_η = σ_η, χ = χ, γ = γ)
        sol           = solveModel(modd; noisy = false)

        @unpack exit_flag1, exit_flag2, exit_flag3, wage_flag, effort_flag, az, yz, w_0, θ, Y = sol
        
        flag_z[iz]    = maximum([exit_flag1, exit_flag2, exit_flag3, wage_flag, effort_flag])

        if flag_z[iz] < 1             
            
            # log wage of new hires, given z_1 = z
            lw1_z[iz]     = log(w_0) - 0.5*(ψ*hp(az[iz])*σ_η)^2 
           
            # tightness, given z_1 = z
            θ_z[iz]       = θ             

            # now, let's think about wages and output for continuing hires
            @views z_shocks_z     = z_shocks[iz]
            @views z_shocks_idx_z = z_shocks_idx[iz]
            η_shocks              = simulateEShocks(σ_η; N = length(z_shocks_idx_z), T = 1)
            
            start_idx    = (iz==1) ? 1 : indices[iz-1] + 1 
            end_idx      = indices[iz]
            hpz_z1       = hp.(az)  # h'(a(z|z_1))
            @views hp_az = hpz_z1[z_shocks_idx_z]

            t1           = ψ*hp_az.*η_shocks 
            t2           = 0.5*(ψ*hp_az*σ_η).^2 
            Δlw[start_idx:end_idx]  = t1 - t2

            @views y     = yz[z_shocks_idx_z] # a(z|z_1)*z
            ηz           = z_shocks_z.*η_shocks

            ly[start_idx:end_idx]   = log.(max.(y + ηz, 0.01)) # avoid run-time error
        end
    end

    # only compute moments for reasonable parameters
    if maximum(flag_z) < 1
        
        # Standard deviation of wage changes for job-stayers
        #histogram(Δlw)
        std_Δlw = std(Δlw) 
        
        # Regress wage changes Δ log w_it on ndividual log output log y_it
        dΔlw_dy = ols(vec(Δlw), vec(ly))[2]

        # Compute long simulated time series  (trim to post-burn-in for z_t when computing moment)
        z_shocks_idx_str     = zstring.z_shocks_idx
        @views lw1_t         = lw1_z[z_shocks_idx_str]   # E[log w_1 | z_t]
        @views θ_t           = θ_z[z_shocks_idx_str]     # θ(z_t)

        # Compute evolution of unemployment for the different z_t paths
        T        = zstring.T
        u_t      = zeros(T)
        u_t[1]   = u0
        @views @inbounds for t=2:T
            u_t[t] = (1 - f(θ_t[t-1]))*u_t[t-1] + s*(1 - u_t[t-1])
        end

        # Estimate d E[log w_1] / d u (pooled ols)
        @views dlw1_du = ols(vec(lw1_t[burnin+1:end]), vec(u_t[burnin+1:end]))[2]
        # Compute u_ss as mean of unemployment rate post-burn period in for now
        u_ss = mean(u_t[burnin+1:end])
    end
    
    # Return simulation results
    return (std_Δlw = std_Δlw, dlw1_du = dlw1_du, dΔlw_dy = dΔlw_dy, u_ss = u_ss, flag = maximum(flag_z))
end

"""
Run an OLS regression of Y on X.
"""
function ols(Y, X; intercept = true)
    if intercept == true
        X = hcat(ones(size(X,1)), X)
    end
    return (X'X)\(X'*Y) #inv(X'X)*(X'*Y)
end

"""
Simulate N X T shock panel for z. Include a burn-in period.
"""
function simulateZShocks(mod; N = 10000, T = 100, z_1_idx = median(1:length(zgrid)), set_seed = true, seed = 512)
    if set_seed == true
        Random.seed!(seed)
    end
    @unpack P_z, zgrid, σ_η        = mod
    z_shocks, probs, z_shocks_idx  = simulateProd(P_z, zgrid, T; z_1_idx = z_1_idx, N = N, set_seed = false) # N X T  
    return (z_shocks = z_shocks, z_shocks_idx = z_shocks_idx, N = N, T = T, z_1_idx = z_1_idx, seed = seed)
end

"""
Simulate N X T shock panel for η. 
"""
function simulateEShocks(σ_η; N, T, set_seed = true, seed = 512)
    if set_seed == true
        Random.seed!(seed)
    end
    η_shocks = rand(Normal(0, σ_η), N, T) # N x T 
    return η_shocks
end