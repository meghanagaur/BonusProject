"""
Simulate EGSS, given parameters defined in tuple m.
Solve the model with infinite horizon contract. Solve 
for θ and Y on every point in the productivity grid.
Then, compute the effort optimal effort a and wage w,
as a(z|z_0) and w(z|z_0). u0 = initial unemployment rate.
"""
function simulate(modd, shocks; u0 = 0.069, check_mult = false)
    
    # Initialize moments to export (default = NaN)
    std_Δlw   = NaN  # st dev of YoY wage growth
    dlw1_du   = NaN  # d log w_1 / d u
    dlw_dly   = NaN  # passthrough: d log w_it / d log y_it
    u_ss      = NaN  # u_ss <- nonstochastic steady state
    alp_ρ     = NaN  # autocorrelation of quarterly average labor prod.
    alp_σ     = NaN  # st dev of quarterly average labor prod.
    dlu_dlz   = NaN  # d log u / d log z
    std_u     = NaN  # st dev of log u_t
    std_z     = NaN  # st dev of log z_t
    u_ss_2    = NaN  # stochastic mean of unemployment

    # Get all of the relevant parameters, functions for the model
    @unpack hp, zgrid, logz, N_z, P_z, p_z, ψ, f, s, σ_η, χ, γ, hbar, ε = modd 

    # Generate model data for every point on zgrid:

    # Build vectors     
    θ_z       = zeros(N_z)               # θ(z_1)
    f_z       = zeros(N_z)               # f(θ(z_1))
    hp_z      = zeros(N_z, N_z)          # h'(a(z_i | z_j))
    y_z       = zeros(N_z, N_z)          # a(z_i | z_j)*z_i
    w0_z      = zeros(N_z)               # w0 constant
    lw1_z     = zeros(N_z)               # E[log w1|z] <- wages of new hires    
    pt_z      = zeros(N_z, N_z)          # pass-through: ψ*hbar*a(z_i | z_j)^(1 + 1/ε)
    flag_z    = zeros(Int64,N_z)         # convergence/effort/wage flags
    flag_IR_z = zeros(Int64,N_z)         # IR flags
    err_IR_z  = zeros(N_z)               # IR error

    Threads.@threads for iz = 1:N_z

        # Solve the model for z_1 = zgrid[iz]
        sol = solveModel(modd; z_1 = zgrid[iz], noisy = false, check_mult = check_mult)

        @unpack conv_flag1, conv_flag2, conv_flag3, wage_flag, effort_flag, IR_err, flag_IR, az, yz, w_0, θ, Y = sol
        
        # Record flags
        flag_z[iz]    = maximum([conv_flag1, conv_flag2, conv_flag3, wage_flag, effort_flag])
        flag_IR_z[iz] = flag_IR
        err_IR_z[iz]  = IR_err

        if flag_z[iz] < 1             
            
            # E[w_t], given z_1
            w0_z[iz]      = w_0

            # Expected output  a(z_i | z_j)*z_i
            y_z[:,iz]     = yz

            # Expectation of the log wage of new hires, given z_1 = z
            hp_z[:,iz]    = hp.(az)  # h'(a(z|z_1))

            # Expectation of the log wage of new hires, given z_1 = z
            lw1_z[iz]     = log(w_0) - 0.5*(ψ*hp_z[iz,iz]*σ_η)^2 
           
            # Tightness and job-finding rate, given z_1 = z
            θ_z[iz]       = θ      
            f_z[iz]       = f(θ)       

            # Compute passthrough moment: elasticity of w_it wrt y_it 
            pt_z[:,iz]    = ψ*hbar*az.^(1 + 1/ε)

        end
    end

    # Composite flag
    flag = maximum(flag_z)

    # only compute moments if equilibria were found for all z
    if flag < 1

        # Unpack the relevant shocks
        @unpack N_sim_micro, T_sim_micro, N_sim_macro, T_sim_macro,  burnin, z_shocks_micro, η_shocks_micro, z_shocks_macro, s_shocks, jf_shocks, burnin, N_sim_macro_est_alp = shocks

        # Build shocks for wages: z_it, η_it
        η_shocks       = η_shocks_micro*σ_η # scale normal shocks by σ_η
        z_idx_micro    = simulateZShocks(P_z, p_z, z_shocks_micro, N_sim_micro, T_sim_micro)

        # Simulate annual wage changes + passthrough
        @unpack std_Δlw, dlw_dly = simulateWageMoments(η_shocks, z_idx_micro, N_sim_micro, T_sim_micro, w0_z, hp_z, pt_z, ψ, σ_η)        

        # Compute model data for long z_t series (trim to post-burn-in when computing moment)
        logz_ss_idx    = Int64(median(1:N_z))
        z_idx_macro    = simulateZShocks(P_z, p_z, z_shocks_macro, N_sim_macro, T_sim_macro + burnin; z_1_idx = logz_ss_idx)
       
        # Macro moments 
        logz_t         = logz[z_idx_macro]      # log z_t series 
        @views lw1_t   = lw1_z[z_idx_macro]     # E[log w_1 | z_t] series
        @views θ_t     = θ_z[z_idx_macro]       # θ(z_t) series
        @views f_t     = f_z[z_idx_macro]       # f(θ(z_t)) series

        # Bootstrap across N_sim_macro simulations
        dlw1_du_n     = zeros(N_sim_macro)
        dlu_dlz_n     = zeros(N_sim_macro)
        std_u_n       = zeros(N_sim_macro)
        std_z_n       = zeros(N_sim_macro)
        # Bootstrap across N_sim_macro_est_alp simulations
        alp_ρ_n       = zeros(N_sim_macro_est_alp) 
        alp_σ_n       = zeros(N_sim_macro_est_alp)

        # Compute evolution of unemployment for the z_t path
        T             = T_sim_macro + burnin
        u_t           = zeros(T, N_sim_macro)
        u_t[1,:]     .= u0

        Threads.@threads for n = 1:N_sim_macro

            @views @inbounds for t = 2:T
                u_t[t, n] = (1 - f_t[t-1,n])*u_t[t-1,n] + s*(1 - u_t[t-1,n])
            end

            # Estimate d E[log w_1] / d u (pooled ols)
            #@views dlw1_du_n[n]  = ols(vec(lw1_t[burnin+1:end, n]), vec(u_t[burnin+1:end, n]))[2]
            @views dlw1_du_n[n]  = cov(lw1_t[burnin+1:end, n], u_t[burnin+1:end, n])/max(eps(), var(u_t[burnin+1:end, n]))

            # Estimate d log u / d  log z (pooled OLS), nudge to avoid runtime error
            @views logu_t = log.(max.(u_t[burnin+1:end, n], eps()))

            #@views dlu_dlz_n[n]  = ols(log.( max.(u_t[burnin+1:end, n], eps() )), vec(logz_t[burnin+1:end, n]))[2]
            @views dlu_dlz_n[n]  = cov(logu_t, logz_t[burnin+1:end, n])/max(eps(), var(logz_t[burnin+1:end, n]))

            # Compute standard deviation of log u_t and log z_t
            std_u_n[n]  = std(lut)
            @views std_z_n[n]  = std(logz_t[burnin+1:end, n])

        end

        # Standard deviation and persistence of average labor productivity 
        Threads.@threads for n = 1:N_sim_macro_est_alp
            alp_ρ_n[n], alp_σ_n[n] = simulateALP(z_idx_macro[:,n], s_shocks, jf_shocks, N_sim_macro_workers, T_sim_macro, burnin, s, f_z, y_z)
        end

        # Compute cross-simulation averages
        dlw1_du = mean(dlw1_du_n)
        dlu_dlz = mean(dlu_dlz_n)
        std_u   = mean(std_u_n) 
        std_z   = mean(std_z_n)
        alp_ρ   = mean(alp_ρ_n)
        alp_σ   = mean(alp_σ_n)

        # Compute stochastic mean of unemployment: E[u_t | t > burnin]
        u_ss_2  = mean(vec(mean(u_t[burnin+1:end,:], dims = 1)))

        # Compute nonstochastic SS unemployment: define u_ss = s/(s + f(θ(z_ss)), at log z_ss = μ_z
        u_ss   = s/(s  + f(θ_z[logz_ss_idx]))
    end
    
    # determine an IR error for all initial z
    IR_err = sum(abs.(err_IR_z))

    # Export the simulation results
    return (std_Δlw = std_Δlw, dlw1_du = dlw1_du, dlw_dly = dlw_dly, u_ss = u_ss, u_ss_2 = u_ss_2, 
            alp_ρ = alp_ρ, alp_σ = alp_σ, dlu_dlz = dlu_dlz, std_u = std_u, std_z = std_z,
            flag = flag, flag_IR = maximum(flag_IR_z), IR_err = IR_err)
end

"""
Build random shocks for the simulation.
"""
function rand_shocks(N_sim_micro, T_sim_micro, N_sim_macro, N_sim_macro_workers, T_sim_macro, burnin, N_sim_macro_est_alp; set_seed = true, seed = 512)

    if set_seed == true
        Random.seed!(seed)
    end

    # draw uniform and standard normal shocks for micro moments
    z_shocks_micro  = rand(Uniform(0,1), T_sim_micro, N_sim_micro)   # z shocks: T X N
    η_shocks_micro  = rand(Normal(0,1), T_sim_micro,  N_sim_micro)   # η shocks: T X N

    # draw uniform shocks for macro moments
    z_shocks_macro  = rand(Uniform(0,1), T_sim_macro + burnin, N_sim_macro)          # z shocks: T x 1
    s_shocks        = rand(Uniform(0,1), N_sim_macro_workers, T_sim_macro + burnin)  # separation shocks: N x T
    jf_shocks       = rand(Uniform(0,1), N_sim_macro_workers, T_sim_macro + burnin)  # job-finding shocks: N x T
    
    return (N_sim_micro = N_sim_micro, T_sim_micro = T_sim_micro, N_sim_macro = N_sim_macro, N_sim_macro_workers = N_sim_macro_workers,
    T_sim_macro = T_sim_macro, burnin = burnin, z_shocks_micro = z_shocks_micro, η_shocks_micro = η_shocks_micro,
    z_shocks_macro = z_shocks_macro, s_shocks = s_shocks, jf_shocks = jf_shocks, N_sim_macro_est_alp)

end

"""
Simulate T x N panel of productivity draws.
"""
function simulateZShocks(P_z, p_z, z_shocks, N, T; z_1_idx = 0)
   
    PI_z         = cumsum(P_z, dims = 2) # CDF of transition density
    pi_z         = cumsum(p_z)           # CDF of invariant distribution
    z_shocks_idx = zeros(Int32, T, N)    # z_it indices

    Threads.@threads for n = 1:N
        @views @inbounds for t = 1:T
            if t == 1
                if z_1_idx == 0
                    z_shocks_idx[t,n] = findfirst(x -> x >= z_shocks[t,n], pi_z)
                else
                    z_shocks_idx[t,n] = z_1_idx
                end
            else
                z_shocks_idx[t,n] = findfirst(x-> x >= z_shocks[t,n], PI_z[z_shocks_idx[t-1,n], :]) 
            end

        end
    end

    return (z_shocks_idx = z_shocks_idx)
end

"""
Simulate wage moments givenn T x N panel of z_it and η_it
"""
function simulateWageMoments(η_shocks, z_shocks_idx, N, T, w0_z, hp_z, pt_z, ψ, σ_η)

    # Elements of wages
    t1 = zeros(T,N)
    t2 = zeros(T,N)
    # Pass-through moment
    pt = zeros(T,N)

    Threads.@threads for n = 1:N
        z_1_idx = z_shocks_idx[1, n]
        @views @inbounds for t = 1:T
            t1[t,n] = ψ*hp_z[z_shocks_idx[t,n], z_1_idx].*η_shocks[t,n]
            t2[t,n] = 0.5*(ψ*hp_z[z_shocks_idx[t,n], z_1_idx]*σ_η).^2
            pt[t,n] = pt_z[z_shocks_idx[t,n], z_1_idx]
        end
    end

    # Stdev of YoY log wage changes for job-stayers
    lwages   = log.(w0_z[z_shocks_idx[1,:]])' .+ cumsum(t1, dims = 1) - cumsum(t2, dims = 1) # T x N panel of wages
    std_Δlw  = std(lwages[end, :] - lwages[1, :]) 

    return (std_Δlw = std_Δlw, dlw_dly = mean(pt))

end

"""
Simulate average labor productivity a*z for N X T jobs,
ignoring η shocks (zero in expectation).
HP-filter log average output with smoothing parameter λ.
"""
function simulateALP(z_shocks_idx, s_shocks, jf_shocks, N, T_sim, burnin, s, f_z, y_z; λ = 10^5)

    # Active jobs
    T          = T_sim + burnin
    y_m        = zeros(N, T) # N x T panel of output
    y_m[:,1]  .= y_z[z_shocks_idx[1], z_shocks_idx[1]]
    active     = ones(N, T)
    unemp      = zeros(N)
    z_1        = fill(z_shocks_idx[1], N)

    @inbounds for t = 2:T

        zt = z_shocks_idx[t]
        ft = f_z[zt]

         @views @inbounds for n = 1:N

            if unemp[n] == false   
                # separation shock 
                if s_shocks[n, t] < s              # become unemployed
                    unemp[n]     = true
                    active[n, t] = 0
                else
                    y_m[n, t]    = y_z[zt, z_1[n]] # remain employed      
                end
            elseif unemp[n] == true
                # job-finding shock
                if jf_shocks[n, t] < ft            # find a job
                    unemp[n]    = false            # become employed
                    z_1[n]      = zt               # new initial z for contract
                    y_m[n, t]   = y_z[zt, zt]      # new output level     
                else                               # remain unemployed
                    active[n,t] = 0
                end 
            end

        end
    end

    # construct quarterly averages
    T_q       = Int64(T_sim/4)
    ly_q      = zeros(T_q) # quarterly log output

    @views @inbounds for t = 1:T_q
        output  = vec(y_m[:, burnin+1:end][:,t:t+3])
        emp     = vec(active[:, burnin+1:end][:,t:t+3])
        ly_q[t] = log.(mean(output[emp.==1]))
    end

    # hp-filter the quarterly log output series
    ly_q_resid, ly_q_trend = hp_filter(ly_q, λ)

    # Compute standard deviation of log ALP
    alp_σ  = std(ly_q_resid)

    # Compute persistence of log ALP (OLS)
    alp_ρ  = cov(ly_q_resid[1:end-1], ly_q_resid[2:end])/max(eps(), var(ly_q_resid[1:end-1]))
        
    return (alp_ρ = alp_ρ, alp_σ = alp_σ)
end
