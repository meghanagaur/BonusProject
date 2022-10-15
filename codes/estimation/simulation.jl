"""
Simulate EGSS, given parameters defined in tuple m.
Solve the model with infinite horizon contract. Solve 
for θ and Y on every point in the productivity grid.
Then, compute the effort optimal effort a and wage w,
as a(z|z_0) and w(z|z_0). u0 = initial unemployment rate.
"""
function simulate(modd, shocks; u0 = 0.06)
    
    # initialize moments to export (default to NaN)
    std_Δlw  = NaN  # st dev of quarterly wage growth
    avg_Δlw  = NaN  # avg quarterly wage growth 
    dlw1_du  = NaN  # d log w_1 / d u
    dlw_dly  = NaN  # d log w_it / d log y_it
    u_ss     = NaN  # u_ss
    dlY_dlz  = NaN  # d log Y / d log z
    dlu_dlz  = NaN  # d log u / d log z
    dlw1_dlz = NaN  # d log w_1  / d log z
    std_u    = NaN  # # st dev of u_t
    std_z    = NaN  # st dev of z_t
    std_Y    = NaN  # st dev of Y_t
    std_w0   = NaN  # std dev of w_0

    # Get all of the relevant parameters for the model
    @unpack hp, zgrid, N_z, ψ, f, s, ε, σ_η, χ, γ, hbar = modd 

    # Unpack the relevant shocks
    @unpack η_shocks, z_shocks, z_shocks_idx, λ_N_z, zstring, burnin, z_ss_dist, indices, indices_q, T_sim = shocks

    # Generate model data for every point on zgrid:
    
    # Results from panel simulation
    lw      = zeros(indices[end])        # log w_it, given z_1 and z_t
    ly      = zeros(indices[end])        # log y_it, given z_1 and z_t
    Δlw_q   = zeros(indices_q[end])      # Δlog w_it <- QUARTERLY

    # Values corresponding to new contracts (i.e. starting at z_t)
    w0_z    = zeros(length(zgrid))       # E[w_t], given z_1
    Y_z     = zeros(length(zgrid))       # PV of Y, given z_1
    θ_z     = zeros(length(zgrid))       # θ(z_1)
    lw1_z   = zeros(length(zgrid))       # E[log w1|z] <- wages of new hires
    flag_z  = zeros(Int64,length(zgrid)) # error flags

    Threads.@threads for iz = 1:N_z

        modd          = model(z_1 = zgrid[iz], ε = ε, σ_η = σ_η, hbar = hbar, χ = χ, γ = γ)
        sol           = solveModel(modd; noisy = false)

        @unpack exit_flag1, exit_flag2, exit_flag3, wage_flag, effort_flag, az, yz, w_0, θ, Y = sol
        
        flag_z[iz]    = maximum([exit_flag1, exit_flag2, exit_flag3, wage_flag, effort_flag])

        if flag_z[iz] < 1             
            
            # PV of wages, given z_1
            w0_z[iz]      = w_0

            # PV of Output given z_1
            Y_z[iz]       = Y

            # Expectation of the log wage of new hires, given z_1 = z
            lw1_z[iz]     = log(w_0) - 0.5*(ψ*hp(az[iz])*σ_η)^2 
           
            # Tightness, given z_1 = z
            θ_z[iz]       = θ             

            # Get z, η-shocks to compute wages and output for continuing hires
            @views z_shocks_z     = z_shocks[iz]
            @views z_shocks_idx_z = z_shocks_idx[iz]
            η_shocks_z            = σ_η*η_shocks[iz] # scale η to avoid re-simulating 
            
            # Get indices for filling out log wages
            start_idx    = (iz==1) ? 1 : indices[iz-1] + 1 
            end_idx      = indices[iz]

            # Compute relevant terms for log wages and log output
            hpz_z1                  = hp.(az)  # h'(a(z|z_1))
            @views hp_az            = hpz_z1[z_shocks_idx_z]
            t1                      = ψ*hp_az.*η_shocks_z
            t2                      = 0.5*(ψ*hp_az*σ_η).^2 
            lw_mat                  = log(w_0) .+ cumsum(t1, dims=2) - cumsum(t2, dims=2)
            lw[start_idx:end_idx]   = vec(lw_mat)

            # Compute log individual output
            @views y                = yz[z_shocks_idx_z]     # a_t(z_t|z_1)*z_t
            ηz                      = z_shocks_z.*η_shocks_z # η_t*z_t
            ly[start_idx:end_idx]   = vec(log.(max.(y + ηz, 0.01))) # nudge up to avoid run-time error

            # Make some adjustments to compute quarterly wage changes
            start_idx_q  = (iz==1) ? 1 : indices_q[iz-1] + 1 
            end_idx_q    = indices_q[iz]
            Δlw_q[start_idx_q:end_idx_q] = [lw_mat[i,t+3] - lw_mat[i,t] for  i = 1:size(z_shocks_z,1), t = 1:T_sim-3]
        end
    end

    # only compute moments if equilibria were found for all z
    if maximum(flag_z) < 1
        
        # Stdev & avg of quarterly log wage changes for job-stayers
        std_Δlw  = std(Δlw_q) 
        avg_Δlw  = mean(Δlw_q)
        #histogram(Δlw_q)
        
        # Regress log w_it on log y_it 
        dlw_dly  = ols(vec(lw), vec(ly))[2]

        # Compute model data for long time series  (trim to post-burn-in when computing moment)
        z_shocks_idx_str     = zstring.z_shocks_idx
        z_shocks_str         = zstring.z_shocks
        lz_shocks_str        = log.(zstring.z_shocks)
        @views lw1_t         = lw1_z[z_shocks_idx_str]       # E[log w_1 | z_t]
        w_0_t                = w0_z[z_shocks_idx_str]        # E[w_0 | z_t]
        @views Y_t           = Y_z[z_shocks_idx_str]         # Y_1 | z_t
        lY_t                 = log.(max.(Y_t, eps()))        # log Y_1 | z_t, nudge up to avoid runtime error
        @views θ_t           = θ_z[z_shocks_idx_str]         # θ(z_t)

        # Compute evolution of unemployment for the different z_t paths
        T        = zstring.T
        u_t      = zeros(T)
        u_t[1]   = u0
        @views @inbounds for t=2:T
            u_t[t] = (1 - f(θ_t[t-1]))*u_t[t-1] + s*(1 - u_t[t-1])
        end

        # Estimate d E[log w_1] / d u (pooled ols)
        @views dlw1_du  = ols(vec(lw1_t[burnin+1:end]), vec(u_t[burnin+1:end]))[2]
        
        # Estimate d E[log w_1] / d log z (pooled ols)
        @views dlw1_dlz = ols(vec(lw1_t[burnin+1:end]), vec(lz_shocks_str[burnin+1:end]))[2]
        
        # Estimate d log Y / d log z (pooled OLS)
        @views dlY_dlz   = ols(vec(lY_t[burnin+1:end]), vec(lz_shocks_str[burnin+1:end]))[2]

        # Estimate d log u / d  log z (pooled OLS), nudge up to avoid runtime error
        @views dlu_dlz  = ols(log.( max.(u_t[burnin+1:end], eps()) ), vec(lz_shocks_str[burnin+1:end]))[2]

        # Compute u_ss as mean of unemployment rate post-burn period in for now
        u_ss   = mean(u_t[burnin+1:end])

        # Compute some standard deviations
        std_u  = std(u_t)
        std_z  = std(z_shocks_str)
        std_Y  = std(Y_t)
        std_w0 = std(w_0_t)
    end
    
    # Export the simulation results
    return (std_Δlw = std_Δlw, dlw1_du = dlw1_du, dlw_dly = dlw_dly, u_ss = u_ss, 
            avg_Δlw = avg_Δlw, dlw1_dlz = dlw1_dlz, dlY_dlz = dlY_dlz, dlu_dlz = dlu_dlz, 
            std_u = std_u, std_z = std_z, std_Y = std_Y, std_w0 = std_w0, flag = maximum(flag_z))
end

"""
Run an OLS regression of Y on X.
"""
function ols(Y, X; intercept = true)
    if intercept == true
        X = hcat(ones(size(X,1)), X)
    end
    return  (X'X)\(X'*Y) #inv(X'X)*(X'*Y)
end

"""
Simulate N X T shock panel for z. Include a burn-in period.
"""
function simulateZShocks(P_z, zgrid; N = 10000, T = 100, z_1_idx = median(1:length(zgrid)), set_seed = true, seed = 512)
    if set_seed == true
        Random.seed!(seed)
    end
    z_shocks, probs, z_shocks_idx  = simulateProd(P_z, zgrid, T; z_1_idx = z_1_idx, N = N, set_seed = false) # N X T  
    return (z_shocks = z_shocks, z_shocks_idx = z_shocks_idx, N = N, T = T, z_1_idx = z_1_idx, seed = seed)
end
