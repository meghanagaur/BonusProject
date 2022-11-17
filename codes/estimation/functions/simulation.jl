"""
Simulate EGSS, given parameters defined in tuple m.
Solve the model with infinite horizon contract. Solve 
for θ and Y on every point in the productivity grid.
Then, compute the effort optimal effort a and wage w,
as a(z|z_0) and w(z|z_0). u0 = initial unemployment rate.
"""
function simulate(modd, shocks; u0 = 0.067)
    
    # initialize moments to export (default to NaN)
    std_Δlw   = NaN  # st dev of quarterly wage growth
    avg_Δlw   = NaN  # avg quarterly wage growth 
    dlw1_du   = NaN  # d log w_1 / d u
    dlw_dly   = NaN  # d log w_it / d log y_it
    u_ss      = NaN  # u_ss
    dlY_dlz   = NaN  # d log Y / d log z
    dlu_dlz   = NaN  # d log u / d log z
    dlw1_dlz  = NaN  # d log w_1  / d log z
    std_u     = NaN  # st dev of u_t
    std_z     = NaN  # st dev of z_t
    std_Y     = NaN  # st dev of Y_t
    std_w0    = NaN  # std dev of w_0
    dlw_dly_2 = NaN  # d log w_it / d log y_it <- alt definition
    u_ss_2    = NaN  # u_ss <- alt definition

    # Get all of the relevant parameters for the model
    @unpack hp, zgrid, N_z, ψ, f, s, σ_η, χ, γ, hbar, ε = modd 

    # Unpack the relevant shocks
    @unpack η_shocks, z_shocks, z_shocks_idx, λ_N_z, zstring, burnin, z_ss_dist, indices, indices_y, T_sim = shocks

    # Generate model data for every point on zgrid:
    
    # Results from panel simulation
    lw        = zeros(indices[end])        # log w_it, given z_1 and z_t
    ly        = zeros(indices[end])        # log y_it, given z_1 and z_t
    η_idx     = zeros(indices[end])        # index for selecting based on η
    pt        = zeros(indices[end])        # direct computation of pass-through
    Δlw_y     = zeros(indices_y[end])      # Δlog w_it <- yoy

    # Values corresponding to new contracts (i.e. starting at z_t)
    w0_z      = zeros(length(zgrid))       # E[w_t], given z_1
    Y_z       = zeros(length(zgrid))       # PV of Y, given z_1
    θ_z       = zeros(length(zgrid))       # θ(z_1)
    lw1_z     = zeros(length(zgrid))       # E[log w1|z] <- wages of new hires
    flag_z    = zeros(Int64,length(zgrid)) # convergence/effort/wage flags
    flag_IR_z = zeros(Int64,length(zgrid)) # IR flags
    err_IR_z  = zeros(length(zgrid))       # IR flags

    Threads.@threads for iz = 1:N_z

        # solve the model for z_1 = zgrid[iz]
        modd          = model(z_1 = zgrid[iz], ε = ε, σ_η = σ_η, hbar = hbar, χ = χ, γ = γ)
        sol           = solveModel(modd; noisy = false)

        @unpack conv_flag1, conv_flag2, conv_flag3, wage_flag, effort_flag, IR_err, flag_IR, az, yz, w_0, θ, Y = sol
        
        # record the flags
        flag_z[iz]    = maximum([conv_flag1, conv_flag2, conv_flag3, wage_flag, effort_flag])
        flag_IR_z[iz] = flag_IR
        err_IR_z[iz]  = IR_err

        if flag_z[iz] < 1             
            
            # PV of wages, given z_1
            w0_z[iz]      = w_0

            # PV of Output given z_1
            Y_z[iz]       = Y

            # Expectation of the log wage of new hires, given z_1 = z
            hpz_z1        = hp.(az)  # h'(a(z|z_1))
            lw1_z[iz]     = log(w_0) - 0.5*(ψ*hpz_z1[iz]*σ_η)^2 
           
            # Tightness, given z_1 = z
            θ_z[iz]       = θ             

            # Get indices for filling out log wages
            start_idx    = (iz==1) ? 1 : indices[iz-1] + 1 
            end_idx      = indices[iz]

            # Get z, η-shocks to compute wages and output for continuing hires
            @views z_shocks_z        = z_shocks[iz]
            @views z_shocks_idx_z    = z_shocks_idx[iz]
            η_shocks_z               = σ_η*η_shocks[iz]  # scale η
            #η_idx[start_idx:end_idx] = vec((η_shocks_z .>= -0.6).*(η_shocks_z .<= 0.6)) # limit to [-0.6,0.6] for log y computation
            
            # Compute relevant terms for log wages and log output
            @views hp_az            = hpz_z1[z_shocks_idx_z]
            t1                      = ψ*hp_az.*η_shocks_z
            t2                      = 0.5*(ψ*hp_az*σ_η).^2 
            lw_mat                  = log(w_0) .+ cumsum(t1, dims=2) - cumsum(t2, dims=2)
            lw[start_idx:end_idx]   = vec(lw_mat)

            # Compute log individual output
            #@views y                = yz[z_shocks_idx_z]                            # a_t(z_t|z_1)*z_t
            #ηz                      = z_shocks_z.*η_shocks_z                        # truncate η_t
            #ly[start_idx:end_idx]   = vec(log.(max.(y + ηz, eps())))                # nudge up to avoid any runtime errors

            # Compute directly pass-through for comparison
            @views pt[start_idx:end_idx]   = az[z_shocks_idx_z].^(1 + 1/ε)

            # Make some adjustments to compute annual wage changes
            start_idx_y = (iz == 1) ? 1 : indices_y[iz-1] + 1 
            end_idx_y   = indices_y[iz]
            @views Δlw_y[start_idx_y:end_idx_y] = vec([lw_mat[i,t+12] - lw_mat[i,t] for  i = 1:size(z_shocks_z,1), t = 1:T_sim-12])
        end
    end

    # only compute moments if equilibria were found for all z
    if maximum(flag_z) < 1
        
        # Stdev & avg of quarterly log wage changes for job-stayers
        std_Δlw  = std(Δlw_y) 
        #avg_Δlw  = mean(Δlw_y)
        #histogram(Δlw_y)
        
        # Regress log w_it on log y_it 
        #dlw_dly_2  = ols(lw[η_idx.==1], ly[η_idx.==1] )[2]
        dlw_dly     = ψ*hbar*mean(pt)

        # Compute model data for long time series  (trim to post-burn-in when computing moment)
        z_shocks_idx_str     = zstring.z_shocks_idx
        z_shocks_str         = zstring.z_shocks
        lz_shocks_str        = log.(zstring.z_shocks)
        @views lw1_t         = lw1_z[z_shocks_idx_str]       # E[log w_1 | z_t]
        @views w_0_t         = w0_z[z_shocks_idx_str]        # E[w_0 | z_t]
        @views Y_t           = Y_z[z_shocks_idx_str]         # Y_1 | z_t
        lY_t                 = log.(max.(Y_t, eps() ))       # log Y_1 | z_t, nudge up to avoid runtime error
        @views θ_t           = θ_z[z_shocks_idx_str]         # θ(z_t)

        # Compute evolution of unemployment for the different z_t paths
        T        = zstring.T
        u_t      = zeros(T)
        u_t[1]   = u0
        @views @inbounds for t = 2:T
            u_t[t] = (1 - f(θ_t[t-1]))*u_t[t-1] + s*(1 - u_t[t-1])
        end

        # Estimate d E[log w_1] / d u (pooled ols)
        @views dlw1_du  = ols(vec(lw1_t[burnin+1:end]), vec(u_t[burnin+1:end]))[2]

        # Estimate d E[log w_1] / d log z (pooled ols)
        #@views dlw1_dlz = ols(vec(lw1_t[burnin+1:end]), vec(lz_shocks_str[burnin+1:end]))[2]
        
        # Estimate d log Y / d log z (pooled OLS)
        #@views dlY_dlz  = ols(vec(lY_t[burnin+1:end]), vec(lz_shocks_str[burnin+1:end]))[2]

        # Estimate d log u / d  log z (pooled OLS), nudge to avoid runtime error
        #@views dlu_dlz  = ols(log.( max.(u_t[burnin+1:end], eps() )), vec(lz_shocks_str[burnin+1:end]))[2]

        # Compute nonstochastic SS unemployment: define u_ss = s/(s + f(θ(z_ss)), at log z_ss = μ_z
        idx    = Int64(median(1:length(zgrid)))
        u_ss   = s/(s  + f(θ_z[idx]))

        # Compute stochastic SS unemployment: define u_ss = E[u_t | t > burnin]
        u_ss_2   = mean(u_t[burnin+1:end])

        # Compute some standard deviations
        std_u  = std(log.(max.(eps(), u_t)))
        std_z  = std(lz_shocks_str)
        std_Y  = std(lY_t)
    end
    
    IR_err = sum(abs.(err_IR_z))

    # Export the simulation results
    return (std_Δlw = std_Δlw, dlw1_du = dlw1_du, dlw_dly = dlw_dly, u_ss = u_ss, 
            avg_Δlw = avg_Δlw, dlw1_dlz = dlw1_dlz, dlY_dlz = dlY_dlz, dlu_dlz = dlu_dlz, 
            std_u = std_u, std_z = std_z, std_Y = std_Y, std_w0 = std_w0, dlw_dly_2 = dlw_dly_2,
            u_ss_2 = u_ss_2, flag = maximum(flag_z), flag_IR = maximum(flag_IR_z), IR_err = IR_err)
end

"""
Run an OLS regression of Y on X.
"""
function ols(Y, X; intercept = true)
    
    if intercept == true
        XX = [ones(size(X,1)) X]
    end    
    return  (XX'XX)\(XX'*Y)
    #return  inv(XX'XX)(XX'*YY)
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

"""
Construct the η, z shocks for simulation
"""
function build_shocks( N_z, P_z, zgrid, N_sim, T_sim, burnin; set_seed = true, seed = 512)

    if set_seed == true
        Random.seed!(seed)
    end
    
    # Compute the invariant distribution of logz
    A           = P_z - Matrix(1.0I, N_z, N_z)
    A[:,end]   .= 1
    O           = zeros(1,N_z)
    O[end]      = 1
    z_ss_dist   = (O*inv(A))
    @assert(isapprox(sum(z_ss_dist),1))

    # Create z and η shocks
    λ_N_z        = floor.(Int64, N_sim*z_ss_dist)
    indices      = cumsum(λ_N_z, dims = 2)*T_sim
    indices_y    = cumsum(λ_N_z, dims = 2)*(T_sim - 12) # indices for yearly log wage changes
    z_shocks     = OrderedDict{Int, Array{Real,2}}()
    z_shocks_idx = OrderedDict{Int, Array{Real,2}}()
    η_shocks     = OrderedDict{Int, Array{Real,2}}()
    Threads.@threads for iz = 1:length(zgrid)
        temp                = simulateZShocks(P_z, zgrid, N = λ_N_z[iz], T = T_sim, z_1_idx = iz, set_seed = false)
        z_shocks[iz]        = temp.z_shocks
        z_shocks_idx[iz]    = temp.z_shocks_idx
        η_shocks[iz]        = rand(Normal(0, 1), size(z_shocks[iz])) # N x T  <- standard normal
    end

    # Create one long z_t string: set z_1 to default value of 1.
    zstring  = simulateZShocks(P_z, zgrid, N = 1, T = N_sim + burnin, set_seed = false)

    # Create an ordered tuple that contains the zshocks
    shocks   = (η_shocks = η_shocks, z_shocks = z_shocks, z_shocks_idx = z_shocks_idx, indices = indices, indices_y = indices_y,
        λ_N_z = λ_N_z, N_sim = N_sim, T_sim = T_sim, zstring = zstring, burnin = burnin, z_ss_dist = z_ss_dist)

    return shocks
end