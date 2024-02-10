"""
Simulate the model with fixed effort.
"""
function simulateFixedEffort(modd, shocks; u0 = 0.06, a = 1.0, λ = 10^5, sd_cut = 3.0, fix_wages = false, smm = false)

    # Initialize moments to export (default = 0)

    # Targeted
    std_Δlw   = 0.0  # st dev of YoY wage growth
    dlw1_du   = 0.0  # d log w_1 / d u
    dlw_dly   = 0.0  # passthrough: d log w_it / d log y_it
    u_ss      = 0.0  # stochastic mean of unemployment  

    # Initialize untargeted moments
    macro_vars   = [:p, :u, :v, :θ, :w]
    N_macro_vars = length(macro_vars)
    rho_lx       = zeros(N_macro_vars)
    std_lx       = zeros(N_macro_vars)
    corr_lx      = zeros(N_macro_vars, N_macro_vars)

    # Get all of the relevant parameters, functions for the model
    @unpack zgrid, logz, N_z, P_z, p_z, ψ, f, s, z_ss_idx, ρ, σ_ϵ = modd 

    # Get θ(z), f(z), w(z), lw0(z), y(z)
    y_z     = zgrid*a             
    @unpack θ_z, f_z, w_z, lw1_z, flag_z, flag_IR_z, err_IR_z = getFixedEffort(modd; a = 1.0, fix_wages = fix_wages)
   
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
        @unpack burnin_macro, z_idx_macro, T_sim_macro, N_sim_macro, N_sim_alp = shocks

        # Compute model data for long z_t series (trim to post-burn-in when computing moment)

        # Macro moments 
        z_idx_macro    = min.(max.(z_idx_macro, idx_1), idx_2)  
        @views lw1_t   = lw1_z[z_idx_macro]     # E[log w_1 | z_t] series
        @views θ_t     = θ_z[z_idx_macro]       # θ(z_t) series
        @views f_t     = f_z[z_idx_macro]       # f(θ(z_t)) series
        @views y_t     = y_z[z_idx_macro]       # y(z_t) series
        @views w_t     = w_z[z_idx_macro]       # w(z_t) series

        # Bootstrap across N_sim_macro simulations
        dlw1_du_n      = zeros(N_sim_macro)     # cyclicality of new hire wages (monthly)

        # Compute evolution of unemployment for the z_t path
        T              = T_sim_macro + burnin_macro
        T_q            = Int(T_sim_macro/3)
        u_t            = zeros(T, N_sim_macro)
        u_t[1,:]      .= u0

        Threads.@threads for n = 1:N_sim_macro

            @views @inbounds for t = 2:T
                u_t[t, n] = uLM(u_t[t-1,n], s, f_t[t-1,n]) 
            end

            # Estimate d E[log w_1] / d u (pooled ols)
            @views dlw1_du_n[n]  = cov(lw1_t[burnin_macro+1:end, n], u_t[burnin_macro+1:end, n])/max(eps(), var(u_t[burnin_macro+1:end, n]))
        end

        # TARGETED MOMENTS 

        # Cyclicality of new hire wages 
        dlw1_du = mean(dlw1_du_n)

        # Stochastic mean of unemployment: E[u_t | t > burnin]
        u_ss    = mean(vec(mean(u_t[burnin_macro+1:end,:], dims = 1)))

        # UNTARGETED MOMENTS
        if smm == false 

            # Simulate on-the-job wage volatility
            std_Δlw = simulateWageMomentsFixedEffort(s_shocks_micro, jf_shocks_micro, z_idx_micro, 
                        N_sim_micro, T_sim_micro, burnin_micro, f_z, lw1_z, s; fix_wages = fix_wages)

            # 5 variables x N simulation
            lx_q  = zeros(T_q_macro, N_macro_vars, N_sim_alp)

            Threads.@threads for n = 1:N_sim_alp
        
                @views v_t         = θ_t[burnin_macro+1:end, n].*u_t[burnin_macro+1:end, n]

                # Compute quarterly average of u_t, z_t, θ_t, and y_t in post-burn-in period
                @views y_q         = quarterlyAverage(y_t[burnin_macro+1:end, n], T_q; weights = u_t[burnin_macro+1:end, n])    
                @views u_q         = quarterlyAverage(u_t[burnin_macro+1:end, n], T_q)            
                @views v_q         = quarterlyAverage(v_t, T_q) 
                @views θ_q         = quarterlyAverage(θ_t[burnin_macro+1:end, n], T_q)          
                @views w_q         = quarterlyAverage(w_t[burnin_macro+1:end, n], T_q; weights = u_t[burnin_macro+1:end, n])    

                # HP-filter the quarterly log unemployment series, nudge to avoid runtime error
                ly_q_resid, _      = hp_filter(log.(max.(y_q, eps())), λ)  
                lu_q_resid, _      = hp_filter(log.(max.(u_q, eps())), λ)   
                lv_q_resid, _      = hp_filter(log.(max.(v_q, eps())), λ)   
                lθ_q_resid, _      = hp_filter(log.(max.(θ_q, eps())), λ)   
                lw_q_resid, _      = hp_filter(log.(max.(w_q, eps())), λ)  
                
                # Shimer ordering
                lx_q[:, :, n]      .= [ly_q_resid lu_q_resid lv_q_resid lθ_q_resid lw_q_resid]
            end

            # Moments
            rho_n          = zeros(N_macro_vars, N_sim_alp)
            std_n          = zeros(N_macro_vars, N_sim_alp)
            corr_n         = zeros(N_macro_vars, N_macro_vars, N_sim_alp)

            Threads.@threads for n = 1:N_sim_alp
                
                @inbounds for j = 1:N_macro_vars

                    rho_n[j,n]           = first(autocor(lx_q[:,j,n], [1]))
                    std_n[j,n]           = std(lx_q[:,j,n])
                    
                    @inbounds for i = 1:N_macro_vars
                        corr_n[i,j,n]    = cor(lx_q[:,i,n], lx_q[:,j,n])
                    end
                end
            end 

            # Compute cross-simulation averages
            rho_lx    = mean(rho_n, dims = 2) 
            std_lx    = mean(std_n, dims = 2) 
            corr_lx   = mean(corr_n, dims = 3)[:,:,1]
        end

    end

    return (std_Δlw = std_Δlw, dlw1_du = dlw1_du, dlw_dly = dlw_dly, u_ss = u_ss, rho_lx = rho_lx, std_lx = std_lx,
            corr_lx = corr_lx, macro_vars = macro_vars, flag = flag, flag_IR = flag_IR, IR_err = IR_err)
end

"""
Return correct bargaining solution
"""
function getFixedEffort(modd; a = 1.0, fix_wages = false) 
    if fix_wages == true 
        return getFixedEffortWages(modd; a = a)
    else
        return getFixedEffortFlexWages(modd; a = a)
    end
end

"""
Simulate the model with bargaining; fixed wages within the match.
"""
function getFixedEffortWages(modd; a = 1.0)
        
    # Generate model data for every point on zgrid:
    @unpack zgrid, logz, N_z, P_z, p_z, ψ, f, s, z_ss_idx, ρ, σ_ϵ = modd 

    # Build vectors     
    θ_z       = zeros(N_z)               # θ(z)
    f_z       = zeros(N_z)               # f(θ(z))
    W_z       = zeros(N_z)               # W
    Y_z       = zeros(N_z)               # Y
    lw0_z     = zeros(N_z)               # new hire log wage  
    flag_z    = zeros(Int64, N_z)        # convergence/effort/wage flags
    flag_IR_z = zeros(Int64, N_z)        # IR flags
    err_IR_z  = zeros(N_z)               # IR error

    Threads.@threads for iz = 1:N_z

        # Solve the model for z_0 = zgrid[iz]
        @unpack conv_flag1, conv_flag2, conv_flag3, wage_flag, IR_err, IR_flag, w_0, θ, W, Y = solveFixedEffortWages(modd; z_0 = zgrid[iz], a = a)
        
        # Record flags
        flag_z[iz]    = maximum([conv_flag1, conv_flag2, conv_flag3, wage_flag])
        flag_IR_z[iz] = IR_flag
        err_IR_z[iz]  = IR_err

        if flag_z[iz] < 1             
            
            # log wage of new hires (wage is constant throughout contract)
            lw1_z[iz]     = log(max(eps(), w_0))
           
            # Tightness and job-finding rate, given z_0 = z
            θ_z[iz]       = θ      
            f_z[iz]       = f(θ)   
            Y_z[iz]       = Y
            W_z[iz]       = W

        end
    end

    # Export the main objects 
    return (θ_z = θ_z, f_z = f_z, lw1_z = lw1_z, 
            a_z = a*ones(N_z), W = W_z, Y = Y_z, w_z = exp.(lw0_z),
            flag_z = flag_z, flag_IR_z = flag_IR_z, err_IR_z = err_IR_z)
end

"""
Simulate the model with bargaining; flexible wages.
"""
function getFixedEffortFlexWages(modd; a = 1.0)
        
    # Get all of the relevant parameters, functions for the model
    @unpack zgrid, logz, N_z, P_z, p_z, ψ, f, s, z_ss_idx, ρ, σ_ϵ = modd 

    # Generate model data for every point on zgrid:

    # Solve the model for z_0 = zgrid[iz]
    @unpack conv_flag1, conv_flag2, conv_flag3, wage_flag, w_z, θ_z, W, Y = solveFixedEffortFlexWages(modd; a = a)
    flag_z = maximum([conv_flag1, conv_flag2, conv_flag3, wage_flag])*ones(N_z)

    # Build vectors     
    flag_IR_z = zeros(Int64, N_z)        # IR flags
    err_IR_z  = zeros(N_z)               # IR error
    lw1_z     = log.(max.(eps(), w_z))   # new hire wage 
    f_z       = f.(θ_z)                  # job-finding rate

    # Export the main objects
    return (θ_z = θ_z, f_z = f_z, w_z = w_z, lw1_z = lw1_z, 
            a_z = a*ones(N_z), W = W, Y = Y, 
            flag_z = flag_z, flag_IR_z = flag_IR_z, err_IR_z = err_IR_z)
end

"""
Solve the infinite horizon model 
with fixed effort a and period-by-period wage w(z).
"""
function solveFixedEffortWages(modd; a = 1.0, z_0 = nothing, max_iter1 = 1000, max_iter2 = 1000, max_iter3 = 50,
    tol1 = 10^-10, tol2 =  10^-10, tol3 =  10^-8, noisy = false, q_lb_0 =  0.0, q_ub_0 = 1.0)

    @unpack h, β, s, κ, ι, ω, N_z, q, u, zgrid, P_z, ψ, procyclical, N_z, z_ss_idx = modd  
    
    # Find index of z_0 on the productivity grid 
    if isnothing(z_0)
        z_0_idx = z_ss_idx
    else
        z_0_idx = findfirst(isapprox(z_0, atol = 1e-6), zgrid)  
    end

    # set tolerance and convergence parameters
    err1    = 10
    iter1   = 1
    err2    = 10
    iter2   = 1
    err3    = 10
    iter3   = 1
    IR_err  = 10
    IR_flag = 0

    # Initialize default values and search parameters
    ω_0    = procyclical ? ω[z_0_idx] : ω # unemployment value at z_0
    ω_vec  = procyclical ?  ω : ω*ones(N_z)
    q_lb   = q_lb_0          # lower search bound for q
    q_ub   = q_ub_0          # upper search bound for q
    q_0    = (q_lb + q_ub)/2 # initial guess for q
    α      = 0               # dampening parameter
    Y_0    = 0               # initalize Y for export
    U      = 0               # initalize worker's EU from contract for export
    w_0    = 0               # initialize initial wage constant for export                      
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
    W_0   = copy(ω_vec) # initial guess
    flow  = β*s*(P_z*ω_vec)

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

    # Look for a fixed point in θ_0
    @inbounds while err3 > tol3 && iter3 <= max_iter3

        if noisy 
            println("Solving for q_0...")
            println("iter:\t"*string(iter3))
            println("error:\t"*string(err3))
            println("q_0:\t"*string(q_0))
        end

        # wages from free-entry condition
        w_0    = ψ*(Y_0[z_0_idx] - κ/q_0) 
        
        # Check the IR constraint (must bind)
        U      = (1/ψ)*log(max(eps(), w_0)) - (1/ψ)*h(a) + W_0[z_0_idx] # nudge w_0 to avoid runtime error
        IR_err = U - ω_0                                                # check whether IR constraint holds

        # Update θ accordingly: U decreasing in θ (increasing in q)
        if IR_err < 0             # increase q (decrease θ)
            q_lb  = copy(q_0)
        elseif IR_err > 0         # decrease q (increase θ)
            q_ub  = copy(q_0)
        end

        # Bisection
        q_1    = (q_lb + q_ub)/2   # update q
        err3   = abs(IR_err)

        # Record info on IR constraint
        IR_flag = (err1 > tol1)

        # Export the accurate iter and current q
        if err3 > tol3
            # stuck in a corner (0 or 1), so break
            if min(abs(q_1 - q_ub_0), abs(q_1 - q_lb_0))  < tol3
                break
            else
                q_0     = α*q_0 + (1 - α)*q_1
                iter3  += 1
            end
        end

    end

    return (θ = (q_0^(-ι) - 1)^(1/ι), Y = Y_0[z_0_idx], W = w_0/ψ, w_0 = w_0, IR_err = err1*IR_flag, IR_flag = IR_flag,
            err1 = err1, err2 = err2, err3 = err3, iter1 = iter1, iter2 = iter2, iter3 = iter3, wage_flag = (w_0 <= 0),
            conv_flag1 = (iter1 > max_iter1), conv_flag2 = (iter2 > max_iter2), conv_flag3 = (iter3 > max_iter3))
end

"""
Solve the infinite horizon model using a bisection search on θ,
with fixed effort a and fixed wage w_0 throughout the contract.
"""
function solveFixedEffortFlexWages(modd; a = 1.0, max_iter1 = 1000, max_iter2 = 1000, max_iter3 = 1000,
    tol1 = 10^-10, tol2 =  10^-10, tol3 =  10^-10, noisy = false)

    @unpack h, β, s, κ, ι, ω, N_z, q, u, zgrid, P_z, ψ, N_z, ξ = modd  

    # set tolerance and convergence parameters
    err1    = 10
    iter1   = 1
    err2    = 10
    iter2   = 1
    err3    = 10
    iter3   = 1

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
    @inbounds while err3 > tol3 && iter3 < max_iter3
        W_1    = w_z + β*(1-s)*P_z*W_0
        err3   = maximum(abs.(W_1 - W_0))
        W_0    = copy(W_1)
        iter3   +=1

        if noisy 
            println("W_0 iter: "*string(iter3))
            println("W_0 error: "*string(err3))
        end
    end

    # Solve for tightness from the firm's free-entry condition
    J_0    = Y_0 - W_0
    q_0    = κ./(J_0) 

    return (θ_z = (q_0.^(-ι) .- 1).^(1/ι), Y = Y_0, W = W_0, w_z = w_z, 
            err1 = err1, err2 = err2, err3 = err3, iter1 = iter1, iter2 = iter2, iter3 = iter3, wage_flag = maximum(w_z .<= 0),
            conv_flag1 = (iter1 > max_iter1), conv_flag2 = (iter2 > max_iter2), conv_flag3 = (iter3 > max_iter3))
end

"""
Simulate wage moments givenn N x T panel of z_it and η_it
"""
function simulateWageMomentsFixedEffort(s_shocks, jf_shocks, z_idx, N_sim, 
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
    @views ten_pb        = tenure[:, burnin+1:end]'  # reshape to T x N
    @views lw_pb         = lw[:, burnin+1:end]'      # reshape to T x N 

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

    return (std_Δlw = std_Δlw)
end
