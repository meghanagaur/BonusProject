"""
Simulate the model with fixed effort.
"""
function simulateFixedEffort(modd, shocks; u0 = 0.06, a = 1.0, λ = 10^5, sd_cut = 3.0, 
                            pv = true, fix_wages = false, smm = false, est_z = false)

    # Initialize moments to export (default = 0)

    # Targeted
    std_Δlw      = 0.0  # st dev of YoY wage growth
    dlw1_du      = 0.0  # d log w_1 / d u
    dlw_dly      = 0.0  # passthrough: d log w_it / d log y_it
    u_ss         = 0.0  # stochastic mean of unemployment  
    y_ρ          = 0.0  # autocorrelation of quarterly GDP
    y_σ          = 0.0  # st dev of quarterly GDP
    p_ρ          = 0.0  # autocorrelation of quarterly ALP
    p_σ          = 0.0  # st dev of quarterly ALP
    dlw_dlp      = 0.0  # averages wages/output (quarterly)

    # Untargeted simulated moments (quarterly)
    macro_vars   = [:y, :p, :u, :v, :θ, :w]
    N_macro_vars = length(macro_vars)
    rho_lx       = zeros(N_macro_vars)
    std_lx       = zeros(N_macro_vars)
    corr_lx      = zeros(N_macro_vars, N_macro_vars)
    
    # Additional cyclicality measures (quarterly)
    dlw1_dlp     = 0.0  # averages new hire wages/output (quarterly)
    dlW_dlY      = 0.0  # present value wages/output (monthly)
    dlW_dlz      = 0.0  # present value wages (monthly)
    dlY_dlz      = 0.0  # present value output (monthly)
    dlθ_dlz      = 0.0  # present value tightness (monthly)

    # Get all of the relevant parameters, functions for the model
    @unpack zgrid, logz, N_z, P_z, p_z, ψ, f, s, z_ss_idx, ρ, σ_ϵ = modd 

    # Get θ(z), f(z), w(z), lw0(z), y(z)
    @unpack θ_z, f_z, w_z, lw1_z, y_z, W, Y, flag_z, flag_IR_z, err_IR_z = getFixedEffort(modd; a = a, fix_wages = fix_wages)
   
    # Composite flag; truncate simulation and only penalize IR flags for log z values within XX standard deviations of μ_z 
    σ_z     = σ_ϵ/sqrt(1 - ρ^2)
    idx_1   = findfirst(x-> x > -sd_cut*σ_z, logz) 
    idx_2   = findlast(x-> x <= sd_cut*σ_z, logz) 
    flag    = maximum(flag_z[idx_1:idx_2])
    flag_IR = maximum(flag_IR_z[idx_1:idx_2])
    IR_err  = sqrt(sum((err_IR_z[idx_1:idx_2]).^2))

    # Only compute moments if equilibria were found for all z
    if  max(flag_IR, flag) < 1 

        # Unpack the relevant shocks
        if est_z == true 
            @unpack N_sim_macro, T_sim, burnin, z_shocks = shocks
            z_idx    = simulateZShocks(P_z, z_shocks, N_sim_macro, T_sim + burnin; z0_idx = z_ss_idx)
        elseif est_z == false 
            @unpack N_sim_macro, T_sim, burnin, z_idx = shocks  
        end

        # Truncate z shock realizations
        z_idx    = min.(max.(z_idx, idx_1), idx_2)  

        # Macro moments 
        z_idx_macro    = min.(max.(z_idx, idx_1), idx_2)  
        @views lW_t    = log.(W[z_idx])   # E[log w_1 | z_t] <- PRESENT VALUE
        @views lw1_t   = pv ? lW_t : lw1_z[z_idx]   
        @views θ_t     = θ_z[z_idx]       # θ(z_t) series
        @views f_t     = f_z[z_idx]       # f(θ(z_t)) series
        @views y_t     = y_z[z_idx]       # y(z_t) series

        # Bootstrap across N_sim_macro simulations
        dlw1_du_n      = zeros(N_sim_macro)                  # cyclicality of new hire wages (monthly)
        y_ρ_n          = est_z ? zeros(N_sim_macro) : 0.0    # autocorrelation of GDP
        y_σ_n          = est_z ? zeros(N_sim_macro) : 0.0    # output volatility of GDP
        p_ρ_n          = est_z ? zeros(N_sim_macro) : 0.0    # autocorrelation of ALP
        p_σ_n          = est_z ? zeros(N_sim_macro) : 0.0    # output volatility of ALP

        # Compute evolution of unemployment for the z_t path
        T              = T_sim + burnin
        T_q            = Int(T_sim/3)
        u_t            = zeros(T, N_sim_macro)
        u_t[1,:]      .= u0
        u_t_prod       = copy(u_t)

        Threads.@threads for n = 1:N_sim_macro

            # Beginning-of-period unemployment
            @views @inbounds for t = 2:T
                u_t[t, n]      = uLM(u_t[t-1,n], s, f_t[t-1,n]) 
                u_t_prod[t, n] = uLM_prod(u_t_prod[t-1,n], s, f_t[t-1,n]) 
            end

            # Estimate d E[log w_1] / d u (pooled ols)
            @views dlw1_du_n[n]   = cov(lw1_t[burnin+1:end, n], u_t[burnin+1:end, n])/max(eps(), var(u_t[burnin+1:end, n]))
        end 

        # Production weight
        @views emp_wgt    = 1 .- u_t_prod[burnin+1:end,:]
 
        if est_z == true 
            Threads.@threads for n = 1:N_sim_macro

                # Total output (GDP)
                ly_q            = quarterlyTotal(y_t[burnin+1:end, n], T_q; weights = emp_wgt[:,n]) 
                ly_q_resid, _   = hp_filter(log.(max.(ly_q, eps())), λ)  
            
                # Average output/worker (ALP)
                @views lp_q     = quarterlyAverage(y_t[burnin+1:end, n], T_q; weights = emp_wgt[:,n])    
                lp_q_resid, _   = hp_filter(log.(max.(lp_q, eps())), λ)  
            
                # Autocorrelation and standard deviation of GDP
                y_ρ_n[n]  = first(autocor(ly_q_resid, [1]))
                y_σ_n[n]  = std(ly_q_resid)

                # Autocorrelation and standard deviation of ALP
                p_ρ_n[n]  = first(autocor(lp_q_resid, [1]))
                p_σ_n[n]  = std(lp_q_resid)

            end
    
        end
        
        # TARGETED MOMENTS 

        # Cyclicality of new hire wages 
        dlw1_du = mean(dlw1_du_n)

        # Stochastic mean of unemployment: E[u_t | t > burnin]
        u_ss    = mean(vec(mean(u_t[burnin+1:end,:], dims = 1)))

        # Ouptut autocorrelation and standard deviation
        y_ρ = mean(y_ρ_n)
        y_σ = mean(y_σ_n)
        p_ρ = mean(p_ρ_n)
        p_σ = mean(p_σ_n)

        # UNTARGETED MOMENTS
        if smm == false 

            # Unpack additional shocks for macro wage/output moments
            @unpack N_sim_micro, s_shocks_micro, jf_shocks_micro, N_sim_macro_workers, s_shocks_macro, jf_shocks_macro = shocks

            # Initialize some series
            @views v_t        = θ_t[burnin+1:end, :].*u_t[burnin+1:end, :]
            @views w_t        = w_z[z_idx]       # w(z_t) series
            @views lY_t       = log.(Y[z_idx])   # E[log w_1 | z_t] series <- PRESENT VALUE
            @views lz_t       = logz[z_idx]      # log z grid
            @views w1_wgt     = u_t[burnin+1:end,:].*f_t[burnin+1:end, :]

            # Simulate on-the-job wage volatility
            std_Δlw = simulateSDWagesFixedEffort(s_shocks_micro, jf_shocks_micro, z_idx[:,1], 
                        N_sim_micro, T_sim, burnin, f_z, lw1_z, s; fix_wages = fix_wages)

            # 5 variables x N simulation
            lx_q       = zeros(T_q, N_macro_vars, N_sim_macro)
            dlw_dlp_n  = zeros(N_sim_macro)     # cyclicality of wages     
            dlw1_dlp_n = zeros(N_sim_macro)     # cyclicality of new hire wages
            dlθ_dlz_n  = zeros(N_sim_macro)     # cyclicality of tightness
            dlW_dlz_n  = zeros(N_sim_macro)     # cyclicality of wages (present value) 
            dlY_dlz_n  = zeros(N_sim_macro)     # cyclicality of output (present value) 
            dlW_dlY_n  = zeros(N_sim_macro)     # cyclicality of new hire wages (present value) 

            Threads.@threads for n = 1:N_sim_macro

                # Monthly cyclicality measures
                @views dlθ_dlz_n[n]  = cov(log.(θ_t[burnin+1:end, n]), lz_t[burnin+1:end, n])/max(eps(), var(lz_t[burnin+1:end, n]))
                @views dlW_dlY_n[n]  = cov(lW_t[burnin+1:end, n], lY_t[burnin+1:end, n])/max(eps(), var(lY_t[burnin+1:end, n]))
                @views dlW_dlz_n[n]  = cov(lW_t[burnin+1:end, n], lz_t[burnin+1:end, n])/max(eps(), var(lz_t[burnin+1:end, n]))
                @views dlY_dlz_n[n]  = cov(lY_t[burnin+1:end, n], lz_t[burnin+1:end, n])/max(eps(), var(lz_t[burnin+1:end, n]))
            
                # Compute quarterly series for u_t, z_t, θ_t, y_t, p_t, w_t, lw1_t in post-burn-in period
                @views u_q           = quarterlyAverage(u_t[burnin+1:end, n], T_q)            
                @views v_q           = quarterlyAverage(v_t[:,n], T_q) 
                @views θ_q           = quarterlyAverage(θ_t[burnin+1:end, n], T_q)    
                @views w1_q          = quarterlyAverage(w_t[burnin+1:end, n], T_q; weights = w1_wgt[:,n])  
                @views y_q           = quarterlyTotal(y_t[burnin+1:end, n], T_q; weights = emp_wgt[:,n]) 
                @views p_q           = quarterlyAverage(y_t[burnin+1:end, n], T_q; weights = emp_wgt[:,n])    

                # Average wages
                if fix_wages == false

                    @views w_q     = quarterlyAverage(w_z[z_idx_macro[burnin+1:end,n]] , T_q; weights = emp_wgt[:,n])      
               
                elseif fix_wages == true
                    
                    # Simulate wages
                    @unpack lw, active = simulateWagesFixedEffort(s_shocks_macro, jf_shocks_macro, z_idx[:,n], 
                                            N_sim_macro_workers, T_sim, burnin, f_z, lw1_z, s; fix_wages = fix_wages)
                    
                    # Compute quarterly average    
                    w_q          = zeros(T_q)                                                  
                    @views @inbounds for t = 1:T_q
                        t_q      = t*3
                        wages    = exp.(vec(lw[:, (t_q - 2):t_q]))    # wages
                        emp_2    = vec(active[:, (t_q - 2):t_q])      # who is employed
                        w_q[t]   = mean(wages[emp_2.==1])             # log average wages/worker
                    end

                end 

                # HP-filter the quarterly log unemployment series, nudge to avoid runtime error
                ly_q_resid, _      = hp_filter(log.(max.(y_q, eps())), λ)  
                lp_q_resid, _      = hp_filter(log.(max.(p_q, eps())), λ)  
                lu_q_resid, _      = hp_filter(log.(max.(u_q, eps())), λ)   
                lv_q_resid, _      = hp_filter(log.(max.(v_q, eps())), λ)   
                lθ_q_resid, _      = hp_filter(log.(max.(θ_q, eps())), λ)   
                lw_q_resid, _      = hp_filter(log.(max.(w_q, eps())), λ)  
                lw1_q_resid, _     = hp_filter(log.(max.(w1_q, eps())), λ)  

                # Moments
                lx_q[:, :, n]     .= [ly_q_resid lp_q_resid lu_q_resid lv_q_resid lθ_q_resid lw_q_resid]
                dlw_dlp_n[n]       = cov(lw_q_resid, ly_q_resid)/max(eps(), var(ly_q_resid))
                dlw1_dlp_n[n]      = cov(lw1_q_resid, ly_q_resid)/max(eps(), var(ly_q_resid))

            end

            # Simulated Moments
            rho_n          = zeros(N_macro_vars, N_sim_macro)
            std_n          = zeros(N_macro_vars, N_sim_macro)
            corr_n         = zeros(N_macro_vars, N_macro_vars, N_sim_macro)

            Threads.@threads for n = 1:N_sim_macro
                
                @inbounds for j = 1:N_macro_vars

                    rho_n[j,n]           = first(autocor(lx_q[:,j,n], [1]))
                    std_n[j,n]           = std(lx_q[:,j,n])
                    
                    @inbounds for i = 1:N_macro_vars
                        corr_n[i,j,n]    = cor(lx_q[:,i,n], lx_q[:,j,n])
                    end
                end
            end 

            # Compute cross-simulation averages
            rho_lx       = mean(rho_n, dims = 2) 
            std_lx       = mean(std_n, dims = 2) 
            corr_lx      = mean(corr_n, dims = 3)[:,:,1]
            dlθ_dlz      = mean(dlθ_dlz_n)
            dlw_dlp      = mean(dlw_dlp_n)
            dlw1_dlp     = mean(dlw1_dlp_n)
            dlW_dlz      = mean(dlW_dlz_n)
            dlY_dlz      = mean(dlY_dlz_n)
            dlW_dlY      = mean(dlW_dlY_n)

        end

    end

    return (std_Δlw = std_Δlw, dlw1_du = dlw1_du, dlw_dly = dlw_dly, u_ss = u_ss, 
            y_ρ = y_ρ, y_σ = y_σ, p_ρ = p_ρ, p_σ = p_σ,
            rho_lx = rho_lx, std_lx = std_lx, corr_lx = corr_lx, 
            dlθ_dlz = dlθ_dlz, dlw_dlp = dlw_dlp, dlw1_dlp = dlw1_dlp, 
            dlW_dlz = dlW_dlz, dlY_dlz = dlY_dlz, dlW_dlY = dlW_dlY,
            macro_vars = macro_vars, flag = flag, flag_IR = flag_IR, IR_err = IR_err)
end

"""
Return correct bargaining solution
"""
function getFixedEffort(modd; a = 1.0, fix_wages = false) 

    # Get all of the relevant parameters, functions for the model
    @unpack zgrid, logz, N_z, P_z, p_z, ψ, f, s, z_ss_idx, ρ, σ_ϵ = modd 
    y_z     = zgrid*a             

    # Generate model data for every point on zgrid:

    # Solve the model for z_0 = zgrid[iz]
    if fix_wages == true 
        @unpack IR_err, IR_flag, conv_flag1, conv_flag2, wage_flag, w_z, θ_z, W, Y = solveFixedEffortWages(modd; a = a)
    else
        @unpack IR_err, IR_flag, conv_flag1, conv_flag2, wage_flag, w_z, θ_z, W, Y = solveFixedEffortFlexWages(modd; a = a)
    end
    
    flag_z = maximum([conv_flag1, conv_flag2, wage_flag])*ones(N_z)

    # Build vectors     
    lw1_z     = log.(max.(eps(), w_z))   # new hire wage 
    f_z       = f.(θ_z)                  # job-finding rate

    # Export the main objects
    return (θ_z = θ_z, f_z = f_z, w_z = w_z, lw1_z = lw1_z, 
            a_z = a*ones(N_z), W = W, Y = Y, y_z = y_z,
            flag_z = flag_z, flag_IR_z = IR_flag, err_IR_z = IR_err)
end

"""
Solve the infinite horizon model 
with fixed effort a and period-by-period wage w(z).
"""
function solveFixedEffortWages(modd; a = 1.0, max_iter1 = 1000, max_iter2 = 1000, 
    tol1 = 10^-10, tol2 =  10^-10, noisy = false)

    @unpack h, β, s, κ, ι, ω, N_z, q, u, zgrid, P_z, ψ, N_z = modd  

    # set tolerance and convergence parameters
    err1    = 10
    iter1   = 1
    err2    = 10
    iter2   = 1
    IR_err  = 10
    IR_flag = 0

    # Initialize default values and search parameters
    α      = 0               # dampening parameter
    Y_0    = 0               # initalize Y for export
    yz     = a*zgrid         # per-period output for each z_t 

    # Solve for the present value of output (exogenous)    
    Y_0    = ones(N_z)
    
    @inbounds while err1 > tol1 && iter1 <= max_iter1   
                
        Y_1    = yz + β*(1-s)*P_z*Y_0    
        err1   = maximum(abs.(Y_0 - Y_1))  # Error       
        
        if (err1 > tol1) 
            iter1 += 1
            if (iter1 < max_iter1) 
                Y_0    = α*Y_0 + (1 - α)*Y_1 
            end
        end
        
        if noisy 
            println("Y_0 error: "*string(err1))
        end
    end

    # Solve for the worker's continuation value upon separation 
    W_0   = copy(ω) # initial guess
    flow  = β*s*(P_z*ω)

    @inbounds while err2 > tol2 && iter2 <= max_iter2
        
        W_1  = flow + β*(1-s)*(P_z*W_0)
        err2 = maximum(abs.(W_1 - W_0))
        
        if (err2 > tol2) 
            iter2 += 1
            if (iter2 < max_iter2) 
                W_0  = α*W_0 + (1 - α)*W_1
            end
        end
       
        if noisy 
            println("U_0 error: "*string(err2))
        end
    end

    # Solve for the wage that satisfies the participation constraint 
    logw    = h(a) .- W_0*ψ + ω*ψ
    w_z     = exp.(logw)
    
    # Solve for tightness from the firm's free-entry condition
    W_0     = w_z./ψ
    J       = Y_0 - W_0
    J_0     = max.(J, 0)
    q       = (κ./(J_0))
    q_0     = min.(1.0, q) 
    θ_z     = ((q_0.^(-ι) .- 1).^(1/ι))
    IR_flag = (q_0 .>= 1)
    IR_err  = abs.(q.-1).*IR_flag # no vacancies posted 

    return (θ_z = θ_z, Y = Y_0, W = W_0, w_z = w_z, IR_err = IR_err, IR_flag = IR_flag,
            err1 = err1, err2 = err2, iter1 = iter1, iter2 = iter2, wage_flag = maximum(w_z .<= 0),
            conv_flag1 = (err1 > tol1), conv_flag2 = (err2 > tol2))
end

"""
Solve the infinite horizon model using a bisection search on θ,
with fixed effort a and fixed wage w_0 throughout the contract.
"""
function solveFixedEffortFlexWages(modd; a = 1.0, max_iter1 = 1000, max_iter2 = 1000,
    tol1 = 10^-10, tol2 =  10^-10, noisy = false)

    @unpack h, β, s, κ, ι, ω, N_z, q, u, zgrid, P_z, ψ, N_z, ξ = modd  

    # set tolerance and convergence parameters
    err1    = 10
    iter1   = 1
    err2    = 10
    iter2   = 1

    # Initialize default values and search parameters
    Y_0    = zeros(N_z)      # initalize Y for export
    yz     = a*zgrid         # per-period output for each z_t 
    
    # Solve for EPDV of output (exogenous)
    @inbounds while err1 > tol1 && iter1 <= max_iter1
                
        Y_1    = yz + β*(1-s)*P_z*Y_0    
        err1   = maximum(abs.(Y_0 - Y_1))  # Error       
        
        if (err1 > tol1) 
            iter1 += 1
            if (iter1 < max_iter1) 
                Y_0    = copy(Y_1)
            end
        end

        if noisy 
            println("Y_0 iter: "*string(iter1))
            println("Y_0 error: "*string(err1))
        end

    end

    #=Solve for the worker's continuation value upon separation 
    U_0   = zeros(N_z) # initial guess
    flow  = β*s*(P_z*ω)

    @inbounds while err2 > tol2 && iter2 <= max_iter2
        U_1  = flow + β*(1-s)*(P_z*U_0)
        err2 = maximum(abs.(U_1 - U_0))
       
        if (err2 > tol2) 
            iter2 += 1
            if (iter2 < max_iter2) 
                U_0  = copy(U_1)
            end
        end

        if noisy 
            println("U_0 iter: "*string(iter2))
            println("U_0 error: "*string(err2))
        end
    end
    
    Ω      = ω .+ h(a)/ψ - U_0
    logw   = Ω - β*(1-s)*(P_z*Ω) 
    @assert maximum(abs.(logw - log.(ξ.(zgrid))) .- h(a) ) < 10^-6 
    =#
    
    # Solve for wages 
    logw    = log.(ξ.(zgrid)) .+ h(a)

    # Solve for the EPDV of wages (to compute profits)
    W_0    = zeros(N_z) 
    w_z    = exp.(logw) 

    # solve via simple value function iteration
    @inbounds while err2 > tol2 && iter2 < max_iter2
        W_1    = w_z + β*(1-s)*P_z*W_0
        err2   = maximum(abs.(W_1 - W_0))
        W_0    = copy(W_1)
        iter2   +=1

        if noisy 
            println("W_0 iter: "*string(iter2))
            println("W_0 error: "*string(err2))
        end
    end

    # Solve for tightness from the firm's free-entry condition
    J       = Y_0 - W_0
    J_0     = max.(J, 0)
    q       = (κ./(J_0))
    q_0     = min.(1.0, q) 
    θ_z     = ((q_0.^(-ι) .- 1).^(1/ι))
    IR_flag = (q_0 .>= 1)
    IR_err  = abs.(q.-1).*IR_flag # no vacancies posted 

    return (θ_z = θ_z, Y = Y_0, W = W_0, w_z = w_z, IR_err = IR_err, IR_flag = IR_flag,
            err1 = err1, err2 = err2, iter1 = iter1, iter2 = iter2, wage_flag = maximum(w_z .<= 0),
            conv_flag1 = (err1 > tol1), conv_flag2 = (err2 > tol2))
            
end

"""
Simulate wages given N x T panel of z_it
"""
function simulateWagesFixedEffort(s_shocks, jf_shocks, z_idx, N_sim, 
                    T_sim, burnin, f_z, lw1_z, s; fix_wages = false)

    # Information about job/employment spells
    T            = T_sim + burnin
    lw           = zeros(N_sim, T)           # N x T panel of log wages
    tenure       = zeros(N_sim, T)           # N x T panel of tenure

    # Initialize period 1 values (production) 
    lw[:,1]     .= lw1_z[z_idx[1]] 
    tenure[:,1] .= 1

    # N x 1 vectors 
    unemp      = zeros(N_sim)            # current unemployment status
    z_1        = fill(z_idx[1], N_sim)   # relevant initial z for contract

    @views @inbounds for t = 2:T

        # index by current z_t: aggregate variables 
        zt    = z_idx[t]                # current productivity
        ft    = f_z[zt]                 # people find jobs in period t

        Threads.@threads for n = 1:N_sim

            if unemp[n] == false  
               
                # separation shock at the end of t-1 
                if s_shocks[n, t-1] < s              
                    
                    # find a job and produce within the period t
                    if  jf_shocks[n, t] < ft               
                        z_1[n]      = zt                    # new initial z for contract
                        lw[n, t]    = lw1_z[zt]             
                        tenure[n,t] = 1                    

                    else
                        unemp[n]     = true
                    end

                # no separation shock, remain employed    
                else        
                    if fix_wages == true
                        lw[n, t]    = lw1_z[z_1[n]]
                    else
                        lw[n, t]    = lw1_z[zt]
                    end
                    tenure[n,t] = tenure[n,t-1] + 1
                end

            elseif unemp[n] == true
                
                # job-finding shock
                if jf_shocks[n, t] < ft             
                    unemp[n]    = false              
                    z_1[n]      = zt                  # new initial z for contract
                    lw[n, t]    = lw1_z[zt]       
                    tenure[n,t] = 1                
                end 

            end

        end

    end

    # compute micro wage moments using post-burn-in data
    @views ten_pb        = tenure[:, burnin+1:end] 
    @views lw_pb         = lw[:, burnin+1:end]     
    active               = ten_pb .>= 1

    return (ten = ten_pb, lw = lw_pb, active = active)
end

"""
Compute standard deviation of YoY wage growth 
"""
function simulateSDWagesFixedEffort(s_shocks, jf_shocks, z_idx, N_sim, 
    T_sim, burnin, f_z, lw1_z, s; fix_wages = false)

    @unpack ten, lw = simulateWagesFixedEffort(s_shocks, jf_shocks, z_idx, N_sim, 
                        T_sim, burnin, f_z, lw1_z, s; fix_wages = fix_wages)

    # reshape to T x N
    ten = ten' 
    lw  = lw' 
    
    # compute YoY wage changes for 13 month spells
    dlw = fill(NaN, Int64(ceil(T_sim/12)),  N_sim)
    Threads.@threads for n = 1:N_sim

        # log wages
        lw_n  = lw[:, n]
        ten_n = ten[:,n]

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

    return (std_Δlw = std_Δlw)
end
