"""
Simulate EGSS, given parameters defined in modd.
Solve the model with infinite horizon contract. Solve 
for θ and Y on every point in the productivity grid.
Then, compute the effort optimal effort a and wage w,
as a(z|z_0) and w(z|z_0). u0 = initial unemployment rate.
λ = HP filtering parameter.
"""
function simulate(modd, shocks; u0 = 0.06, check_mult = false, λ = 10^5)
    
    # Initialize moments to export (default = NaN)
    std_Δlw   = 0.0  # st dev of YoY wage growth
    dlw1_du   = 0.0  # d log w_1 / d u
    dlw_dly   = 0.0  # passthrough: d log w_it / d log y_it
    u_ss      = 0.0  # stochastic mean of unemployment  
    alp_ρ     = 0.0  # autocorrelation of quarterly endog average labor prod.
    alp_σ     = 0.0  # st dev of quarterly endog average labor prod.
    dlu_dly   = 0.0  # d log u / d log y
    std_u     = 0.0  # st dev of log u_t
    std_z     = 0.0  # st dev of log z_t
    u_ss_2    = 0.0  # u_ss <- nonstochastic steady state

    # Get all of the relevant parameters, functions for the model
    @unpack hp, zgrid, logz, N_z, P_z, p_z, ψ, f, s, σ_η, χ, γ, hbar, ε, z_ss_idx, ρ, σ_ϵ = modd 

    # Generate model objects for every point on zgrid:

    # Build vectors     
    θ_z       = zeros(N_z)               # θ(z_1)
    f_z       = zeros(N_z)               # f(θ(z_1))
    hp_z      = zeros(N_z, N_z)          # h'(a(z_i | z_j))
    y_z       = zeros(N_z, N_z)          # a(z_i | z_j)*z_i
    lw1_z     = zeros(N_z)               # E[log w1|z] <- wages of new hires -- denote it by time 1 for simplicity (w_0 is the constant)
    pt_z      = zeros(N_z, N_z)          # pass-through: ψ*hbar*a(z_i | z_j)^(1 + 1/ε)
    flag_z    = zeros(Int64, N_z)        # convergence/effort/wage flags
    flag_IR_z = zeros(Int64, N_z)        # IR flags
    err_IR_z  = zeros(N_z)               # IR error

    Threads.@threads for iz = 1:N_z

        # Solve the model for z_0 = zgrid[iz]
        sol = solveModel(modd; z_0 = zgrid[iz], noisy = false, check_mult = check_mult)

        @unpack conv_flag1, conv_flag2, conv_flag3, wage_flag, effort_flag, IR_err, flag_IR, az, yz, w_0, θ, Y = sol
        
        # Record flags
        flag_z[iz]    = maximum([conv_flag1, conv_flag2, conv_flag3, wage_flag, effort_flag])
        flag_IR_z[iz] = flag_IR
        err_IR_z[iz]  = IR_err

        if flag_z[iz] < 1             
            # Expected output  a(z_i | z_j)*z_i
            y_z[:,iz]     = yz

            # Marginal disutility of effort, given z_0 = z
            hp_z[:,iz]    = hp.(az)  # h'(a(z|z_1))

            # Expectation of the log wage of new hires, given z_0 = z
            lw1_z[iz]     = log(max(eps(), w_0)) - 0.5*(ψ*hp_z[iz, iz]*σ_η)^2 
           
            # Tightness and job-finding rate, given z_0 = z
            θ_z[iz]       = θ      
            f_z[iz]       = f(θ)       

            # Compute expected passthrough: elasticity of w_it wrt y_it 
            pt_z[:,iz]    = ψ*hbar*az.^(1 + 1/ε)
        end

    end

    # Composite flag; truncate simulation and only penalize IR flags for log z values within 3 standard deviations of μ_z 
    σ_z     = σ_ϵ/sqrt(1 - ρ^2)
    idx_1   = findfirst(x-> x > -5σ_z, logz) 
    idx_2   = findlast(x-> x <= 5σ_z, logz) 
    flag    = maximum(flag_z[idx_1:idx_2])
    flag_IR = maximum(flag_IR_z[idx_1:idx_2])

    # only compute moments if equilibria were found for all log z values within 3 standard deviations of μ_z 
    if  max(flag_IR, flag) < 1 

        # Unpack the relevant shocks
        @unpack N_sim_micro, T_sim_micro, burnin_micro, N_sim_macro, T_sim_macro, z_shocks_macro, burnin_macro, 
                    z_shocks_micro, η_shocks_micro, s_shocks_micro, jf_shocks_micro = shocks

        # Scale η shocks by σ_η
        η_shocks_micro = η_shocks_micro*σ_η 

        # Build shocks for wages: z_it, η_it
        z_idx_micro    = simulateZShocks(P_z, z_shocks_micro, 1, T_sim_micro + burnin_micro; z0_idx = z_ss_idx)
        z_idx_micro    = min.(max.(z_idx_micro, idx_1), idx_2)

        # Compute model data for long z_t series (trim to post-burn-in when computing moment)
        z_idx_macro    = simulateZShocks(P_z, z_shocks_macro, N_sim_macro, T_sim_macro + burnin_macro; z0_idx = z_ss_idx)
        z_idx_macro    = min.(max.(z_idx_macro, idx_1), idx_2)  

        # Simulate annual wage changes, passthrough, and ALP
        @unpack std_Δlw, dlw_dly, alp_ρ, alp_σ, dlu_dly =  simulateWagesAndOuput(N_sim_micro, T_sim_micro, burnin_micro,
                                z_idx_micro, s_shocks_micro, jf_shocks_micro, η_shocks_micro, s, f_z, y_z, hp_z, pt_z, ψ, σ_η)

        # Other macro moments 
        @views lw1_t   = lw1_z[z_idx_macro]     # E[log w_1 | z_t] series
        @views θ_t     = θ_z[z_idx_macro]       # θ(z_t) series
        @views f_t     = f_z[z_idx_macro]       # f(θ(z_t)) series
        zshocks_macro  = zgrid[z_idx_macro]     # z shocks

        # Bootstrap across N_sim_macro simulations
        dlw1_du_n      = zeros(N_sim_macro)     # cyclicality of new hire wages 
        std_u_n        = zeros(N_sim_macro)     # standard deviation of log quarterly unemployment
        std_z_n        = zeros(N_sim_macro)     # standard deviation of log quarterly productivity 

        # Compute evolution of unemployment for the z_t path
        T             = T_sim_macro + burnin_macro
        T_q_macro     = Int(T_sim_macro/3)
        u_t           = zeros(T, N_sim_macro)
        u_t[1,:]     .= u0

        Threads.@threads for n = 1:N_sim_macro

            # Law of motion for unemployment
            @views @inbounds for t = 2:T
                u_t[t, n] = u_t[t-1,n] + s*(1 - u_t[t-1,n]) - (1-s)*f_t[t-1,n]*u_t[t-1,n]
            end

            # Estimate d E[log w_1] / d u (pooled ols)
            @views dlw1_du_n[n]  = cov(lw1_t[burnin_macro+1:end, n], u_t[burnin_macro+1:end, n])/max(eps(), var(u_t[burnin_macro+1:end, n]))

            # Compute quarterly average of u_t, z_t in post-burn-in period
            @views u_q           = [mean(u_t[burnin_macro+1:end, n][(t_q*3 - 2):t_q*3]) for t_q = 1:T_q_macro] 
            @views z_q           = [mean(zshocks_macro[burnin_macro+1:end, n][(t_q*3 - 2):t_q*3]) for t_q = 1:T_q_macro] 

            # hp-filter the quarterly log unemployment series, nudge to avoid runtime error
            logu_q_resid, _      = hp_filter(log.(max.(u_q, eps())), λ)   #hp_filter(log.(max.(u_t[burnin_macro+1:end, n], eps())), λ)   
            logz_q_resid, _      = hp_filter(log.(max.(z_q, eps())), λ)  

            # Compute the standard deviation of log u_t and log z_t
            std_u_n[n]           = std(logu_q_resid)
            std_z_n[n]           = std(logz_q_resid)
        end

        # Compute cross-simulation averages
        dlw1_du = mean(dlw1_du_n)
        std_u   = mean(std_u_n) 
        std_z   = mean(std_z_n)

        # Compute cross-simulation average of stochastic mean of unemployment: E[u_t | t > burnin]
        u_ss    = mean(vec(mean(u_t[burnin_macro+1:end,:], dims = 1)))
 
        # Compute nonstochastic SS unemployment: define u_ss = s/(s + f(θ(z_ss)), at log z_ss = μ_z
        #u_ss_2  = s/(s  + f(θ_z[z_ss_idx]))    
    end
    
    # determine an IR error for all initial z within 3 uncond. standard deviations
    IR_err = sqrt(sum((err_IR_z[idx_1:idx_2]).^2))

    # Export the simulation results
    return (std_Δlw = std_Δlw, dlw1_du = dlw1_du, dlw_dly = dlw_dly, u_ss = u_ss, u_ss_2 = u_ss_2, alp_ρ = alp_ρ, 
    alp_σ = alp_σ, dlu_dly = dlu_dly, std_u = std_u, std_z = std_z, flag = flag, flag_IR = flag_IR, IR_err = IR_err)
end

"""
Build random shocks for the simulation.
N_sim_micro             = num workers for micro wage moments
T_sim_micro             = mum periods for micro wage moments  
burnin_micro            = length burn-in for micro moments
N_sim_macro             = num seq to avg across for macro moments
T_sim_macro             = num periods for agg sequences: 69 years 
burnin_macro            = length burn-in for macro moments
"""
function rand_shocks(; N_sim_micro = 5*10^4, T_sim_micro = 828, burnin_micro = 250,
    N_sim_macro = 10^4, T_sim_macro = 828, burnin_macro = 250, set_seed = true, seed = 512)

    if set_seed == true
        Random.seed!(seed)
    end

    # Draw uniform and standard normal shocks for micro moments
    η_shocks_micro  = rand(Normal(0,1), N_sim_micro,  T_sim_micro + burnin_micro)              # η shocks: N x T
    z_shocks_micro  = rand(Uniform(0,1), T_sim_micro + burnin_micro)                           # z shocks: N x T
    s_shocks_micro  = rand(Uniform(0,1), N_sim_micro, T_sim_micro + burnin_micro)              # separation shocks: N x T
    jf_shocks_micro = rand(Uniform(0,1), N_sim_micro, T_sim_micro + burnin_micro)              # job-finding shocks: N x T

    # Draw uniform shocks for macro moments
    z_shocks_macro  = rand(Uniform(0,1), T_sim_macro + burnin_macro, N_sim_macro)              # z shocks: T x 1

    return (z_shocks_micro = z_shocks_micro, z_shocks_macro = z_shocks_macro, N_sim_micro = N_sim_micro, 
            T_sim_micro = T_sim_micro, s_shocks_micro = s_shocks_micro, jf_shocks_micro = jf_shocks_micro, 
            η_shocks_micro = η_shocks_micro, N_sim_macro = N_sim_macro, T_sim_macro = T_sim_macro,
            burnin_micro = burnin_micro, burnin_macro = burnin_macro)
end

"""
Simulate T_sim x N_sim panel of productivity draws, given uniform z_shocks.
"""
function simulateZShocks(P_z, z_shocks, N_sim, T_sim; z0_idx = 0)
   
    PI_z         = cumsum(P_z, dims = 2)         # CDF of transition density
    z_shocks_idx = zeros(Int32, T_sim, N_sim)    # z_it indices

    Threads.@threads for n = 1:N_sim
        @views @inbounds for t = 1:T_sim
            if t == 1
                if z0_idx == 0
                    z_shocks_idx[t,n] = Int64(median(1:length(p_z))) #findfirst(x -> x >= z_shocks[t,n], pi_z)
                else
                    z_shocks_idx[t,n] = z0_idx
                end
            else
                z_shocks_idx[t,n] = findfirst(x-> x >= z_shocks[t,n], PI_z[z_shocks_idx[t-1,n], :]) 
            end

        end
    end

    return (z_shocks_idx = z_shocks_idx)
end

"""
Simulate wage moments given N_sim x T_sim panel of z_it and η_it.
Simulate average labor productivity a*z for N_sim x T_sim jobs,
ignoring η shocks (iid). HP-filter log average output with smoothing parameter λ.
"""
function simulateWagesAndOuput(N_sim, T_sim, burnin, z_idx, s_shocks, jf_shocks, η_shocks, 
                                s, f_z, y_z, hp_z, pt_z, ψ, σ_η; λ = 10^5)

    # Active jobs
    T          = T_sim + burnin
    y_m        = zeros(N_sim, T)                      # N x T panel of output
    y_m[:,1]  .= y_z[z_idx[1], z_idx[1]]              # initialize time 1 output
    active     = ones(N_sim, T)                       # active within the period
    unemp      = zeros(N_sim)                         # current unemployment status
    unemp_beg  = zeros(N_sim, T)                      # beginning of period unemployment status
    z_1        = fill(z_idx[1], N_sim)                # everyone begins employed

    # Micro wage information about job/employment spells
    T            = T_sim + burnin
    lw           = zeros(N_sim, T)                    # N x T panel of log wages
    pt           = fill(NaN, N_sim, T)                # N x T panel of pass-through
    tenure       = zeros(N_sim, T)                    # N x T panel of tenure

    # Initialize period 1 values (production) -- ignore the constant w-1 in log wages, since drops out for wage changes
    lw[:,1]     .= ψ*hp_z[z_idx[1], z_idx[1]]*η_shocks[:,1]  .- 0.5*(ψ*hp_z[z_idx[1], z_idx[1]]*σ_η)^2 
    pt[:,1]     .= pt_z[z_idx[1], z_idx[1]] #.+ ψ*hp_z[z_idx[1], z_idx[1]]*η_shocks_micro[:,1]  
    tenure[:,1] .= 1

    @views @inbounds for t = 2:T

        zt   = z_idx[t]       # current z_t
        ft   = f_z[zt]        # job-finding rate given current z_t 
        y_zz = y_z[zt, :]     # expected output given CURRENT z_t  
        hp_zz = hp_z[zt,:]    # h'(a) given CURRENT z_t
        pt_zz = pt_z[zt,:]    # pass-through given CURRENT z_t

        Threads.@threads for n = 1:N_sim

            if unemp[n] == false   
               
                # experienced separation shock at end of t-1
                if s_shocks[n, t-1] < s               
                   
                    unemp_beg[n,t]       =  1.0             # became unemployed at start of period
                    
                    if  jf_shocks[n, t] < ft                # find a job and produce within the period t
                        z_1[n]      = zt                    # new initial z for contract
                        y_m[n, t]   = y_zz[zt]              # current output      
                        lw[n, t]    = ψ*hp_zz[zt]*η_shocks[n,t]  - 0.5*(ψ*hp_zz[zt]*σ_η)^2  
                        pt[n, t]    = pt_zz[zt] # +  ψ*hp_zz[z_1[n]]*η_shocks_micro[n,t] <- ignore 2nd term for large sample
                        tenure[n,t] = 1
                    else
                        unemp[n]     = true
                        active[n, t] = 0
                    end  
                # no separation shock, remain employed  
                else        
                    # remain employed in period t 
                    lw[n, t]    = lw[n,t-1] + ψ*hp_zz[z_1[n]]*η_shocks[n,t]  - 0.5*(ψ*hp_zz[z_1[n]]*σ_η)^2  
                    pt[n, t]    = pt_zz[z_1[n]] # +  ψ*hp_zz[z_1[n]]*η_shocks[n,t] <- ignore for large sample
                    tenure[n,t] = tenure[n,t-1] + 1            
                    y_m[n, t]   = y_zz[z_1[n]]               # remain employed      
                end

            elseif unemp[n] == true

                unemp_beg[n,t]  = 1.0                        # unemployed at end of last period 

                # job-finding shock
                if jf_shocks[n, t] < ft                      # found a job
                    z_1[n]      = zt                         # new initial z for contract
                    y_m[n, t]   = y_zz[zt]                   # new output level     
                    unemp[n]    = false                      # became employed
                    lw[n, t]    = ψ*hp_zz[zt]*η_shocks[n,t]  - 0.5*(ψ*hp_zz[zt]*σ_η)^2 
                    pt[n, t]    = pt_zz[zt]                  # +  ψ*hp_zz[zt]*η_shocks_micro[n,t]  <- ignore for large sample 
                    tenure[n,t] = 1      
                else                                         # remain unemployed
                    active[n,t] = 0                 
                end 
            end

        end
    end

    # compute micro wage moments using post-burn-in data
    @views ten_pb        = tenure[:, burnin+1:end]'  # reshape to T x N
    @views lw_pb         = lw[:, burnin+1:end]'      # reshape to T x N 
    @views pt_pb         = pt[:, burnin+1:end]'      # reshape to T x N
    @views avg_pt        = mean(pt_pb[isnan.(pt_pb).==0])

    # compute YoY wage changes for 13 month spells
    dlw = fill(NaN, Int64(ceil(T_sim/12)),  N_sim)
    Threads.@threads for n = 1:N_sim

        # log wages
        lw_n  = lw_pb[:, n]
        ten_n = ten_pb[:,n]

        # record new hires for given worker
        new   = findall(isequal(1), ten_n)
        new   = [1; new; T_sim + 1]

        # go through employment spells
        @inbounds for i = 2:lastindex(new)
            
            t0    = new[i-1]                             # beginning of job spell 
            t1    = findfirst(isequal(0), ten_n[t0:new[i] - 1])
            t1    = isnothing(t1) ? new[i] - 1 : t1 - 1  # end of job spell (right before new job spell)
            
            if t1 - t0 >= 12
                t2                                 = collect(t0:12:t1)[end-1]
                idx                                = findfirst(isnan, dlw[:, n])
                dlw_s                              = [lw_n[t+12] - lw_n[t] for t = t0:12:t2]
                dlw[idx:idx+length(dlw_s)-1, n]   .= dlw_s
            end

        end

    end

    # Stdev of YoY log wage changes for job-stayers
    @views Δlw      = dlw[isnan.(dlw).==0]
    std_Δlw         = isempty(Δlw) ? NaN : std(Δlw) 

    # construct quarterly measures
    T_q_macro     = Int(T_sim/3)
    ly_q          = zeros(T_q_macro)                                        # quarterly log output
    u_t           = mean(unemp_beg[:, burnin+1:end], dims=1)                # beginning of period monthly unemployment

    # Compute log average output
    @views @inbounds for t = 1:T_q_macro
        t_q      = t*3
        output   = vec(y_m[:, burnin+1:end][:, (t_q - 2):t_q])       # output of all workers
        emp      = vec(active[:, burnin+1:end][:, (t_q - 2):t_q])    # number of workers producing 
        ly_q[t]  = log.(max(mean(output[emp.==1]), eps()))           # avg output/worker (averaged across workers/hours in the quarter)
    end

    # Estimate d log u_t+1 / d  log ALP_t (quarterly, pooled OLS)
    lu_q            = log.([mean(u_t[(t_q*3 - 2):t_q*3]) for t_q = 1:T_q_macro])
    @views dlu_dly  = cov(lu_q[2:end], ly_q[1:end-1])/max(eps(), var(ly_q[1:end-1]))

    # hp-filter the quarterly log output and unemployment series
    ly_q_resid, _ = hp_filter(ly_q, λ)

    # Compute standard deviation of log ALP
    alp_σ  = std(ly_q_resid)

    # Compute persistence of log ALP (OLS)
    alp_ρ  = first(autocor(ly_q_resid, [1]))

    return (alp_ρ = alp_ρ, alp_σ = alp_σ, std_Δlw = std_Δlw, dlw_dly = avg_pt, dlu_dly = dlu_dly)
end

"""
Directly simulate N X T shock panel for z without drawing uniform shocks externally.
Include a burn-in period.
"""
function drawZShocks(P_z, zgrid; N = 10000, T = 100, z_0_idx = median(1:length(zgrid)), set_seed = true, seed = 512)
    
    if set_seed == true
        Random.seed!(seed)
    end

    z_shocks, z_shocks_idx = simulateProd(P_z, zgrid, T; z_0_idx = z_0_idx, N = N, set_seed = false) # N X T  

    return (z_shocks = z_shocks, z_shocks_idx = z_shocks_idx, N = N, T = T, z_0_idx = z_0_idx, seed = seed)
end