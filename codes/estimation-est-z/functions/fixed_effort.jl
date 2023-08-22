"""
Solve for ρ, σ_ϵ such that we match quarterly ALP moments
"""
function calibrateZ(shocks; ρ_y = 0.89, σ_y = 0.017, N_z = 11, a = 1.0, zbar = 1.0)

    # Objective function
    @unpack burnin_macro, z_shocks_macro, T_sim_macro, N_sim_macro = shocks
    opt                       = Opt(:LN_BOBYQA, 2) 
    obj(x, dummy_gradient!)   = simulateALP(x, z_shocks_macro, T_sim_macro, burnin, N_sim_macro; ρ_y = ρ_y, σ_y = σ_y,  N_z = N_z, a = a, zbar = zbar)
    opt.min_objective         = obj


    # Bound constraints
    opt.lower_bounds        = [0.9, 0.0001] 
    opt.upper_bounds        = [0.999, 0.01] 

    # tolerance and time settings 
    opt.stopval             = 1e-8
    opt.ftol_rel            = 1e-7
    opt.ftol_abs            = 1e-7
    opt.xtol_rel            = 0.0  
    opt.maxtime             = 60*5

    # Search for parameters 
    init                  = [0.977, 0.0054]
    (min_f, arg_min, ret) = NLopt.optimize(opt, init)

    println("minimum:\t\t"*string(min_f))
    println("minimizer:\t\t"*string(arg_min))
    println("reason for stopping:\t"*string(ret))
    
    return (ρ = arg_min[1], σ_ϵ = arg_min[2])
end

"""
Simulate quarterly ALP series and compare to data moments
"""
function simulateALP(x, z_shocks, T_sim, burnin, N_sim; ρ_y = 0.89, σ_y = 0.017, N_z = 13, a = 1.0, zbar = 1, λ = 10^5)
    
    ρ               = x[1]
    σ_ϵ             = x[2]
    μ_z             = log(zbar) - (σ_ϵ^2)/(2*(1-ρ^2))   # normalize E[logz], so that E[z_t] = 1
    logz, P_z, p_z  = rouwenhorst(μ_z, ρ, σ_ϵ, N_z)     # log z grid, transition matrix, invariant distribution
    zgrid           = exp.(logz)                        # z grid in levels
    z_ss_idx        = findfirst(isapprox(μ_z, atol = 1e-6), logz )

    # simulate ALP
    z_idx_macro    = simulateZShocks(P_z, z_shocks, N_sim, T_sim + burnin; z0_idx = z_ss_idx)
    zshocks_macro  = zgrid[z_idx_macro]
    T_q_macro      = Int(T_sim/3)
    alp_ρ_n        = zeros(N_sim)
    alp_σ_n        = zeros(N_sim)

    Threads.@threads for n = 1:N_sim

        # Compute the quarterly average of output
        @views ly_q            = log.(a*[mean(zshocks_macro[burnin+1:end, n][(t_q*3 - 2):t_q*3]) for t_q = 1:T_q_macro])
        @views ly_q_resid, _   = hp_filter(ly_q, λ)  
        @views alp_ρ_n[n]      = first(autocor(ly_q_resid, [1]))
        @views alp_σ_n[n]      = std(ly_q_resid)
    end

    alp_ρ   = mean(alp_ρ_n)
    alp_σ   = mean(alp_σ_n)

    err     = (alp_ρ - ρ_y)^2 + (alp_σ - σ_y)^2

    return err
end

"""
Solve the infinite horizon model using a bisection search on θ,
with fixed effort a and fixed wage w_0 throughout the contract.
"""
function solveModelFixedEffort(modd; a = 1.0, z_0 = nothing, max_iter1 = 50, max_iter2 = 1000, max_iter3 = 1000,
    tol1 = 10^-8, tol2 =  10^-10, tol3 =  10^-10, noisy = true, q_lb_0 =  0.0, q_ub_0 = 1.0)

    @unpack h, β, s, κ, ι, ω, N_z, q, u, zgrid, P_z, ψ, procyclical, N_z, z_ss_idx = modd  
    
    # find index of z_0 on the productivity grid 
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
    flag_IR = 0

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
    az     = fill(a, N_z)    # effort for each z_t (constant)
    yz     = a*zgrid         # per-period output for each z_t 

    # Solve for the present value of output
    err2   = 10
    iter2  = 1      
    Y_0    = ones(N_z)
    
    @inbounds while err2 > tol2 && iter2 <= max_iter2   
                
        Y_1    = yz + β*(1-s)*P_z*Y_0    
        err2   = maximum(abs.(Y_0 - Y_1))  # Error       
        if (err2 > tol2) 
            iter2 += 1
            if (iter2 < max_iter2) 
                Y_0    = α*Y_0 + (1 - α)*Y_1 
            end
        end
        #println(err2)
    end

    # Solve for the worker's continuation value upon separation 
    err3  = 10
    iter3 = 1  
    W_0   = copy(ω_vec) # initial guess
    flow  = β*s*(P_z*ω_vec)

    @inbounds while err3 > tol3 && iter3 <= max_iter3
        W_1  = flow + β*(1-s)*(P_z*W_0)
        err3 = maximum(abs.(W_1 - W_0))
        if (err3 > tol3) 
            iter3 += 1
            if (iter3 < max_iter3) 
                W_0  = α*W_0 + (1 - α)*W_1
            end
        end
        #println(err3)
    end

    # Look for a fixed point in θ_0
    @inbounds while err1 > tol1 && iter1 <= max_iter1  

        if noisy 
            println("iter:\t"*string(iter1))
            println("error:\t"*string(err1))
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
        err1   = abs(IR_err)

        # Record info on IR constraint
        flag_IR = (err1 > tol1)

        # Export the accurate iter and current q
        if err1 > tol1
            # stuck in a corner (0 or 1), so break
            if min(abs(q_1 - q_ub_0), abs(q_1 - q_lb_0))  < tol1
                break
            else
                q_0     = α*q_0 + (1 - α)*q_1
                iter1  += 1
            end
        end

    end

    return (θ = (q_0^(-ι) - 1)^(1/ι), Y = Y_0[z_0_idx], U = U, ω = ω_0, w_0 = w_0, mod = modd, IR_err = err1*flag_IR, flag_IR = flag_IR,
            az = az, yz = yz, err1 = err1, err2 = err2, err3 = err3, iter1 = iter1, iter2 = iter2, iter3 = iter3, wage_flag = (w_0 <= 0),
            effort_flag = false, conv_flag1 = (iter1 > max_iter1), conv_flag2 = (iter2 > max_iter2), conv_flag3 = (iter3 > max_iter3))
end

"""
Simulate the model with fixed effort.
"""
function simulateFixedEffort(modd, shocks; u0 = 0.06, a = 1.0, λ = 10^5, sd_cut = 5.0)
    
    # Initialize moments to export (default = NaN)
    std_Δlw   = 0.0  # st dev of YoY wage growth
    dlw1_du   = 0.0  # d log w_1 / d u
    dlw_dly   = 0.0  # passthrough: d log w_it / d log y_it
    alp_ρ     = 0.0  # autocorrelation of quarterly average labor prod.
    alp_σ     = 0.0  # st dev of quarterly average labor prod.
    dlu_dly   = 0.0  # d log u / d log y
    std_u     = 0.0  # st dev of log u_t
    std_z     = 0.0  # st dev of log z_t
    u_ss      = 0.0  # stochastic mean of unemployment
    u_ss_2    = 0.0  # u_ss <- nonstochastic steady state

    # Get all of the relevant parameters, functions for the model
    @unpack zgrid, logz, N_z, P_z, p_z, ψ, f, s, z_ss_idx, ρ, σ_ϵ = modd 

    # Generate model data for every point on zgrid:

    # Build vectors     
    θ_z       = zeros(N_z)               # θ(z_1)
    f_z       = zeros(N_z)               # f(θ(z_1))
    lw1_z     = zeros(N_z)               # log wage  
    flag_z    = zeros(Int64, N_z)        # convergence/effort/wage flags
    flag_IR_z = zeros(Int64, N_z)        # IR flags
    err_IR_z  = zeros(N_z)               # IR error

    Threads.@threads for iz = 1:N_z

        # Solve the model for z_0 = zgrid[iz]
        @unpack conv_flag1, conv_flag2, conv_flag3, wage_flag, IR_err, flag_IR, w_0, θ = solveModelFixedEffort(modd; z_0 = zgrid[iz], a = a, noisy = false)
        
        # Record flags
        flag_z[iz]    = maximum([conv_flag1, conv_flag2, conv_flag3, wage_flag])
        flag_IR_z[iz] = flag_IR
        err_IR_z[iz]  = IR_err

        if flag_z[iz] < 1             
            
            # log wage of new hires (wage is constant throughout contract)
            lw1_z[iz]     = log(max(eps(), w_0))
           
            # Tightness and job-finding rate, given z_0 = z
            θ_z[iz]       = θ      
            f_z[iz]       = f(θ)       
        end

    end

    # Composite flag; truncate simulation and only penalize IR flags for log z values within XX standard deviations of μ_z 
    σ_z     = σ_ϵ/sqrt(1 - ρ^2)
    idx_1   = findfirst(x-> x > -sd_cut*σ_z, logz) 
    idx_2   = findlast(x-> x <= sd_cut*σ_z, logz) 
    flag    = maximum(flag_z[idx_1:idx_2])
    flag_IR = maximum(flag_IR_z[idx_1:idx_2])

    # only compute moments if equilibria were found for all z
    if  max(flag_IR, flag) < 1 

        # Unpack the relevant shocks
        @unpack burnin_macro, z_shocks_macro, T_sim_macro, N_sim_macro = shocks

        # Compute model data for long z_t series (trim to post-burn-in when computing moment)

        # Macro moments 
        z_idx_macro    = simulateZShocks(P_z, z_shocks_macro, N_sim_macro, T_sim_macro + burnin_macro; z0_idx = z_ss_idx)
        z_idx_macro    = min.(max.(z_idx_macro, idx_1), idx_2)  
        @views lw1_t   = lw1_z[z_idx_macro]     # E[log w_1 | z_t] series
        @views θ_t     = θ_z[z_idx_macro]       # θ(z_t) series
        @views f_t     = f_z[z_idx_macro]       # f(θ(z_t)) series
        zshocks_macro  = zgrid[z_idx_macro]     # z shocks

        # Bootstrap across N_sim_macro simulations
        dlw1_du_n      = zeros(N_sim_macro)
        std_u_n        = zeros(N_sim_macro)
        alp_ρ_n        = zeros(N_sim_macro) 
        alp_σ_n        = zeros(N_sim_macro)
        dlu_dly_n      = zeros(N_sim_macro)

        # Compute evolution of unemployment for the z_t path
        T              = T_sim_macro + burnin_macro
        T_q_macro      = Int(T_sim_macro/3)
        u_t            = zeros(T, N_sim_macro)
        u_t[1,:]      .= u0

        Threads.@threads for n = 1:N_sim_macro

            @views @inbounds for t = 2:T
                u_t[t, n] = u_t[t-1,n] + s*(1 - u_t[t-1,n]) - (1-s)*f_t[t-1,n]*u_t[t-1,n]
            end

            # Estimate d E[log w_1] / d u (pooled ols)
            @views dlw1_du_n[n]    = cov(lw1_t[burnin_macro+1:end, n], u_t[burnin_macro+1:end, n])/max(eps(), var(u_t[burnin_macro+1:end, n]))

            # Compute the quarterly average of various series
            @views ly_q            = log.(a*[mean(zshocks_macro[burnin_macro+1:end, n][(t_q*3 - 2):t_q*3]) for t_q = 1:T_q_macro])
            @views ly_q_resid, _   = hp_filter(ly_q, λ)  
            @views alp_ρ_n[n]      = first(autocor(ly_q_resid, [1]))
            @views alp_σ_n[n]      = std(ly_q_resid)

            # Compute quarterly average of u_t in post-burn-in period + hp-filter the log
            @views lu_q            = log.(max.([mean(u_t[burnin_macro+1:end, n][(t_q*3 - 2):t_q*3]) for t_q = 1:T_q_macro], eps()))
            lu_q_resid, _          = hp_filter(lu_q, λ)  
            std_u_n[n]             = std(lu_q_resid)
            
            # Compute d log u_t+1 / d log y
            @views dlu_dly_n[n]    = cov(lu_q[2:end], ly_q[1:end-1])/max(eps(), var(ly_q[1:end-1]))
        end
  
        # Compute cross-simulation averages
        dlw1_du = mean(dlw1_du_n)
        std_u   = mean(std_u_n) 
        alp_ρ   = mean(alp_ρ_n)
        alp_σ   = mean(alp_σ_n)
        std_z   = alp_σ
        dlu_dly = mean(dlu_dly_n)

        # Compute stochastic mean of unemployment: E[u_t | t > burnin]
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
