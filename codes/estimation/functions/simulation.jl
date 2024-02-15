"""
Simulate EGSS, given parameters defined in modd.
Solve the model with infinite horizon contract. Solve 
for θ and Y on every point in the productivity grid.
Then, compute the effort optimal effort a and wage w,
as a(z|z_0) and w(z|z_0). u0 = initial unemployment rate.
λ = HP filtering parameter.
"""
function simulate(modd, shocks; u0 = 0.06, check_mult = false, smm = false, λ = 10^5, sd_cut = 3)
    
    # Initialize moments to export (default = 0)
    
    # Targeted
    std_Δlw   = 0.0  # st dev of YoY wage growth
    dlw1_du   = 0.0  # d log w_1 / d u
    dlw_dly   = 0.0  # passthrough: d log w_it / d log y_it
    u_ss      = 0.0  # stochastic mean of unemployment  

    # Untargeted simulated moments (quarterly)
    macro_vars   = [:p, :u, :v, :θ, :w]
    N_macro_vars = length(macro_vars)
    rho_lx       = zeros(N_macro_vars)
    std_lx       = zeros(N_macro_vars)
    corr_lx      = zeros(N_macro_vars, N_macro_vars)
    
    # Additional cyclicality measures (quarterly)
    dlw_dlp      = 0.0  # averages wages/output (quarterly)
    dlw1_dlp     = 0.0  # averages new hire wages/output (quarterly)
    dlW_dlY      = 0.0  # present value wages/output (monthly)
    dlW_dlz      = 0.0  # present value wages (monthly)
    dlY_dlz      = 0.0  # present value output (monthly)
    dlθ_dlz      = 0.0  # present value tightness (monthly)


    # Get all of the relevant parameters, functions for the model
    @unpack hp, zgrid, logz, N_z, P_z, p_z, ψ, f, s, σ_η, χ, γ, hbar, ε, z_ss_idx, ρ, σ_ϵ = modd 

    # Generate model objects for every point on zgrid:
    @unpack θ_z, f_z, hp_z, y_z, lw1_z, w0_z, pt_z, W, Y, flag_z, flag_IR_z, err_IR_z = getModel(modd; check_mult = check_mult) 

    # Composite flag; truncate simulation and only penalize IR flags for log z values within sd_cut standard deviations of μ_z 
    σ_z     = σ_ϵ/sqrt(1 - ρ^2)
    idx_1   = findfirst(x-> x > -sd_cut*σ_z, logz) 
    idx_2   = findlast(x-> x <= sd_cut*σ_z, logz) 
    flag    = maximum(flag_z[idx_1:idx_2])
    flag_IR = maximum(flag_IR_z[idx_1:idx_2])
    IR_err  = sqrt(sum((err_IR_z[idx_1:idx_2]).^2))

    # only compute moments if equilibria were found for all z
    if  max(flag_IR, flag) < 1 

        # Unpack the relevant shocks
        @unpack N_sim_micro, T_sim_micro, N_sim_macro, T_sim_macro, burnin_micro, burnin_macro, z_idx_macro,
                N_sim_alp_workers, η_shocks_macro, s_shocks_macro, jf_shocks_macro,
                z_idx_micro, η_shocks_micro, s_shocks_micro, jf_shocks_micro = shocks

        # scale normal output shocks by σ_η
        η_shocks_micro = η_shocks_micro*σ_η 
        η_shocks_macro = η_shocks_macro*σ_η 

        # truncate z shock realizations
        z_idx_micro    = min.(max.(z_idx_micro, idx_1), idx_2)
        z_idx_macro    = min.(max.(z_idx_macro, idx_1), idx_2)  

        # Simulate annual wage changes + passthrough
        @unpack std_Δlw, dlw_dly = simulateWageMoments(η_shocks_micro, s_shocks_micro, jf_shocks_micro, 
                                    z_idx_micro, N_sim_micro, T_sim_micro, burnin_micro, hp_z, pt_z, f_z, ψ, σ_η, s)        

        # Macro moments 
        @views lw1_t   = lw1_z[z_idx_macro]     # E[log w_1 | z_t] series
        @views θ_t     = θ_z[z_idx_macro]       # θ(z_t) series
        @views f_t     = f_z[z_idx_macro]       # f(θ(z_t)) series

        # Bootstrap across N_sim_macro simulations
        dlw1_du_n      = zeros(N_sim_macro)     # cyclicality of new hire wages (monthly)

        # Compute evolution of unemployment for the z_t path
        T             = T_sim_macro + burnin_macro
        u_t           = zeros(T, N_sim_macro)
        u_t[1,:]     .= u0

        # Bootstrap N_sim_macro times 
        Threads.@threads for n = 1:N_sim_macro
            
            # Law of motion for unemployment
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

            # quarterly data
            T_q           = Int(T_sim_macro/3)

            # initialize additional series
            @views y_t     = y_z[z_idx_macro]       # y(z_t) series
            @views lW_t    = log.(W[z_idx_macro])   # E[log w_1 | z_t] series <- PRESENT VALUE
            @views lY_t    = log.(Y[z_idx_macro])   # E[log w_1 | z_t] series <- PRESENT VALUE
            @views lz_t    = logz[z_idx_macro]      # log z grid
            @views w0_t    = w0_z[z_idx_macro] 
            @views w1_wgt  = u_t[burnin_macro+1:end,:].*f_t[burnin_macro+1:end, :]

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
                @views dlθ_dlz_n[n]  = cov(log.(θ_t[burnin_macro+1:end, n]), lz_t[burnin_macro+1:end, n])/max(eps(), var(lz_t[burnin_macro+1:end, n]))
                @views dlW_dlY_n[n]  = cov(lW_t[burnin_macro+1:end, n], lY_t[burnin_macro+1:end, n])/max(eps(), var(lY_t[burnin_macro+1:end, n]))
                @views dlW_dlz_n[n]  = cov(lW_t[burnin_macro+1:end, n], lz_t[burnin_macro+1:end, n])/max(eps(), var(lz_t[burnin_macro+1:end, n]))
                @views dlY_dlz_n[n]  = cov(lY_t[burnin_macro+1:end, n], lz_t[burnin_macro+1:end, n])/max(eps(), var(lz_t[burnin_macro+1:end, n]))
        
                # Compute quarterly average of u_t, z_t, θ_t, and y_t in post-burn-in period
                @views v_t         = θ_t[burnin_macro+1:end, n].*u_t[burnin_macro+1:end, n]
                @views u_q         = quarterlyAverage(u_t[burnin_macro+1:end, n], T_q_macro)            
                @views v_q         = quarterlyAverage(v_t, T_q_macro) 
                @views θ_q         = quarterlyAverage(θ_t[burnin_macro+1:end, n], T_q_macro)  
                # note that E[w_1] = w_0, so we approximate 
                @views w1_q        = quarterlyAverage(w0_t[burnin_macro+1:end, n], T_q_macro; weights = w1_wgt[:,n])       

                # HP-filter the quarterly log unemployment series, nudge to avoid runtime error
                lu_q_resid, _      = hp_filter(log.(max.(u_q, eps())), λ)   
                lv_q_resid, _      = hp_filter(log.(max.(v_q, eps())), λ)   
                lθ_q_resid, _      = hp_filter(log.(max.(θ_q, eps())), λ)   
                lw1_q_resid, _     = hp_filter(log.(max.(w1_q, eps())), λ)   

                # Simulate endogenous ALP and wages
                ly_q_resid, lw_q_resid = simulateWagesOutput(z_idx_macro[:,n], s_shocks_macro, jf_shocks_macro, η_shocks_macro, N_sim_alp_workers, 
                                                T_sim_macro, burnin_macro, T_q_macro, s, f_z, y_z, hp_z, lw1_z, ψ; λ = λ)
                
                # Moments
                lx_q[:, :, n]      .= [ly_q_resid lu_q_resid lv_q_resid lθ_q_resid lw_q_resid]  
                dlw_dlp_n[n]        = cov(lw_q_resid, ly_q_resid)/max(eps(), var(ly_q_resid))
                dlw1_dlp_n[n]       = cov(lw1_q_resid, ly_q_resid)/max(eps(), var(ly_q_resid))

            end

            # Moments
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

    return (std_Δlw = std_Δlw, dlw1_du = dlw1_du, dlw_dly = dlw_dly, u_ss = u_ss, rho_lx = rho_lx, 
            std_lx = std_lx, corr_lx = corr_lx, dlθ_dlz = dlθ_dlz, dlw_dlp = dlw_dlp, 
            dlw1_dlp = dlw1_dlp, dlW_dlz = dlW_dlz, dlY_dlz = dlY_dlz, dlW_dlY = dlW_dlY,
            macro_vars = macro_vars, flag = flag, flag_IR = flag_IR, IR_err = IR_err)
end

"""
Build random shocks for the simulation.
N_sim_micro             = num workers for micro wage moments
T_sim_micro             = mum periods for micro wage moments  
burnin_micro            = length burn-in for micro moments
N_sim_macro             = num seq to avg across for macro moments
T_sim_macro             = num periods for agg sequences: 69 years 
burnin_macro            = length burn-in for macro moments
N_sim_alp               = num seq to avg across for endog ALP moments (<= N_sim_macro)
N_sim_alp_workers       = num workers for endog ALP moments
"""
function drawShocks(P_z; z0_idx = 0, N_sim_micro = 5*10^4, T_sim_micro = 1000, burnin_micro = 500,
    N_sim_macro = 10^4, T_sim_macro = 828, burnin_macro = 500, N_sim_alp_workers = 10^4, 
    smm = false, fix_a = false, set_seed = true, seed = 512)

    if set_seed == true
        Random.seed!(seed)
    end

    # ALP estimation
    N_sim_alp         = !smm ? N_sim_macro : 1
    T_sim_alp         = !smm ? T_sim_macro : 1
    burnin_alp        = !smm ? burnin_macro : 1 
    N_sim_alp_workers = !smm ? N_sim_alp_workers : 1

    # Draw uniform and standard normal shocks for micro moments
    if (smm == false) | (fix_a == false)
        z_shocks_micro  = rand(Uniform(0,1), T_sim_micro + burnin_micro)                    # z shocks: N x T
        s_shocks_micro  = rand(Uniform(0,1), N_sim_micro, T_sim_micro + burnin_micro)       # separation shocks: N x T
        jf_shocks_micro = rand(Uniform(0,1), N_sim_micro, T_sim_micro + burnin_micro)       # job-finding shocks: N x T
        z_idx_micro     = simulateZShocks(P_z, z_shocks_micro, 1, T_sim_micro + burnin_micro; z0_idx = z0_idx)
    else 
        z_shocks_micro  = 0
        s_shocks_micro  = 0
        jf_shocks_micro = 0
        z_idx_micro     = 0
    end

    # Draw uniform shocks for macro moments
    z_shocks_macro  = rand(Uniform(0,1), T_sim_macro + burnin_macro, N_sim_macro)          # z shocks: T x 1
    s_shocks_macro  = rand(Uniform(0,1), N_sim_alp_workers, T_sim_alp + burnin_alp)        # separation shocks: N x T
    jf_shocks_macro = rand(Uniform(0,1), N_sim_alp_workers, T_sim_alp + burnin_alp)        # job-finding shocks: N x T
    z_idx_macro     = simulateZShocks(P_z, z_shocks_macro, N_sim_macro, T_sim_macro + burnin_macro; z0_idx = z0_idx)

    # Idiosyncratic output shocks (only for model with incentives)
    if fix_a == false 
        η_shocks_micro  = rand(Normal(0,1),  N_sim_micro, T_sim_micro + burnin_micro)          # η shocks: N x T
        η_shocks_macro  = rand(Normal(0,1),  N_sim_alp_workers, T_sim_alp + burnin_alp)        # η shocks: N x T
    else
        η_shocks_micro = 0
        η_shocks_macro = 0
    end
            
    # make sure N_sim_alp <= N_sim_macro
    @assert(N_sim_macro >= N_sim_alp)

    # Don't pass in η shocks
    return (z_idx_micro = z_idx_micro, z_idx_macro = z_idx_macro, 
            N_sim_micro = N_sim_micro, T_sim_micro = T_sim_micro, burnin_micro = burnin_micro,
            s_shocks_micro = s_shocks_micro, jf_shocks_micro = jf_shocks_micro, η_shocks_micro = η_shocks_micro,
            N_sim_macro = N_sim_macro,  N_sim_alp_workers = N_sim_alp_workers,
            T_sim_macro = T_sim_macro, burnin_macro = burnin_macro, 
            s_shocks_macro = s_shocks_macro, jf_shocks_macro = jf_shocks_macro, η_shocks_macro = η_shocks_macro)
    
end

"""
Simulate wage moments givenn N x T panel of z_it and η_it
"""
function simulateWageMoments(η_shocks, s_shocks, jf_shocks, z_idx, N_sim, T_sim, burnin, hp_z, pt_z, f_z, ψ, σ_η, s)

    # Information about job/employment spells
    T            = T_sim + burnin
    lw           = zeros(N_sim, T)           # N x T panel of log wages
    pt           = fill(NaN, N_sim, T)       # N x T panel of pass-through
    tenure       = zeros(N_sim, T)           # N x T panel of tenure

    # Initialize period 1 values (production) -- ignore the constant w-1 in log wages, since drops out for wage changes
    lw[:,1]     .= ψ*hp_z[z_idx[1], z_idx[1]]*η_shocks[:,1]  .- 0.5*(ψ*hp_z[z_idx[1], z_idx[1]]*σ_η)^2 
    pt[:,1]     .= pt_z[z_idx[1], z_idx[1]] #.+ ψ*hp_z[z_idx_micro[1], z_idx_micro[1]]*η_shocks_micro[:,1]  
    tenure[:,1] .= 1

    # N x 1 vectors 
    unemp      = zeros(N_sim)            # current unemployment status
    z_1        = fill(z_idx[1], N_sim)   # relevant initial z for contract

    @views @inbounds for t = 2:T

        # index by current z_t: aggregate variables 
        zt    = z_idx[t]                # current productivity
        ft    = f_z[zt]                 # people find jobs in period t
        hp_zz = hp_z[zt,:]              # h'(a) given CURRENT z_t
        pt_zz = pt_z[zt,:]              # pass-through given CURRENT z_t

        Threads.@threads for n = 1:N_sim

            if unemp[n] == false  
               
                # separation shock at the end of t-1 
                if s_shocks[n, t-1] < s              
                    
                    if  jf_shocks[n, t] < ft                # find a job and produce within the period t
                        z_1[n]      = zt                    # new initial z for contract
                        lw[n, t]    = ψ*hp_zz[z_1[n]]*η_shocks[n,t]  - 0.5*(ψ*hp_zz[z_1[n]]*σ_η)^2  
                        pt[n, t]    = pt_zz[z_1[n]]         # +  ψ*hp_zz[z_1[n]]*η_shocks_micro[n,t] <- ignore 2nd term for large sample
                        tenure[n,t] = 1

                    else
                        unemp[n]     = true
                    end

                # no separation shock, remain employed    
                else        
                    lw[n, t]    = lw[n,t-1] + ψ*hp_zz[z_1[n]]*η_shocks[n,t]  - 0.5*(ψ*hp_zz[z_1[n]]*σ_η)^2  
                    pt[n, t]    = pt_zz[z_1[n]]             # +  ψ*hp_zz[z_1[n]]*η_shocks_micro[n,t] <- ignore for large sample
                    tenure[n,t] = tenure[n,t-1] + 1

                end

            elseif unemp[n] == true
                
                # job-finding shock
                if jf_shocks[n, t] < ft               # find a job at beginning of period 
                    unemp[n]    = false               # become employed
                    z_1[n]      = zt                  # new initial z for contract
                    lw[n, t]    = ψ*hp_zz[zt]*η_shocks[n,t]  - 0.5*(ψ*hp_zz[zt]*σ_η)^2 
                    pt[n, t]    = pt_zz[zt]           # +  ψ*hp_zz[zt]*η_shocks_micro[n,t]  <- ignore for large sample 
                    tenure[n,t] = 1                
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

    return (std_Δlw = std_Δlw, dlw_dly = avg_pt)
end

"""
Simulate average labor productivity a*z for N_sim x T_sim observations,
ignoring η shocks. HP-filter log average output with smoothing parameter λ.
"""
function simulateWagesOutput(z_idx, s_shocks, jf_shocks, η_shocks, N_sim,
                     T_sim, burnin, T_q, s, f_z, y_z, hp_z, lw1_z, ψ; λ = 10^5)
    # Active jobs
    T          = T_sim + burnin           # total time 
    y_m        = zeros(N_sim, T)          # N x T panel of output
    lw_m       = zeros(N_sim, T)          # N x T panel of output
    y_m[:,1]  .= y_z[z_idx[1], z_idx[1]]  # initial output
    lw_m[:,1] .= lw1_z[z_idx[1]] .+ ψ*hp_z[z_idx[1], z_idx[1]]*η_shocks[:,1] # initial wages
    active     = ones(N_sim, T)           # active within the period
    unemp      = zeros(N_sim)             # current unemployment status
    unemp_beg  = zeros(N_sim, T)          # beginning of period unemployment
    z_1        = fill(z_idx[1], N_sim)    # initial productivity

    @inbounds for t = 2:T

        zt     = z_idx[t]
        ft     = f_z[z_idx[t-1]]          # job-finding occurs at end of period t-1
        hp_zz  = hp_z[zt,:]               # h'(a) given CURRENT z_t
        lw1_zz = lw1_z[zt]                # initial new hire wages (systematic component)
        y_zz   = y_z[zt,:]                # expected output given CURRENT z_t

         @views @inbounds for n = 1:N_sim

            if unemp[n] == false   
               
                # separation shock at end of t-1
                if s_shocks[n, t-1] < s               
                   
                    # unemployed at start of period
                    unemp_beg[n,t]   =  1.0 
                    
                    # find a job and produce within the period t
                    if  jf_shocks[n, t] < ft  
                        z_1[n]       = zt                                     # new initial z for contract
                        y_m[n, t]    = y_zz[zt]                               # current output 
                        lw_m[n, t]   = lw1_zz + ψ*hp_zz[zt]*η_shocks[n,t] # current wages    
                    
                    # don't find a job and remain unemployed
                    else
                        unemp[n]     = true
                        active[n, t] = 0
                    end
                
                # no separation shock, remain employed
                else
                    y_m[n, t]   = y_zz[z_1[n]]    
                    lw_m[n, t]  = lw_m[n,t-1] + ψ*hp_zz[z_1[n]]*η_shocks[n,t]  - 0.5*(ψ*hp_zz[z_1[n]]*σ_η)^2  
                end
        
            # unemployed at end of last period
            elseif unemp[n] == true

                unemp_beg[n,t]  = 1.0              

                # job-finding shock at beginning of current period
                if jf_shocks[n, t] < ft             
                    z_1[n]      = zt                                    # new initial z for contract
                    y_m[n, t]   = y_zz[zt]                              # new output level     
                    lw_m[n, t]  = lw1_zz + ψ*hp_zz[zt]*η_shocks[n,t]    # new hire wage
                    unemp[n]    = false                                 # became employed
                else                                                    # remain unemployed
                    active[n,t] = 0                 
                end 
            end

        end
    end

    # Construct quarterly averages of wages and output
    ly_q = zeros(T_q)                                       
    lw_q = zeros(T_q)                                       

    @views @inbounds for t = 1:T_q
        t_q      = t*3
        output   = vec(y_m[:, burnin+1:end][:, (t_q - 2):t_q])         # output
        wages    = exp.(vec(lw_m[:, burnin+1:end][:, (t_q - 2):t_q]))  # wages
        emp      = vec(active[:, burnin+1:end][:, (t_q - 2):t_q])      # who is employed
        ly_q[t]  = log.(max(mean(output[emp.==1]), eps()))             # log avg output/worker (averaged across workers/hours in the quarter)
        lw_q[t]  = log.(max(mean(wages[emp.==1]), eps()))              # log average wages/worker
    end

    # HP-filter log wages and output
    ly_q_resid, _  = hp_filter(ly_q, λ)   
    lw_q_resid, _  = hp_filter(lw_q, λ)   

    return (ly_q_resid = ly_q_resid, lw_q_resid = lw_q_resid)
end

"""
Simulate T_sim x N_sim panel of productivity draws, given uniform shocks as inputs.
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
Simulate N X T shock panel for z.
Include a burn-in period.
"""
function drawZShocks(P_z, zgrid; N = 10000, T = 100, z_0_idx = median(1:length(zgrid)), set_seed = true, seed = 512)
    
    if set_seed == true
        Random.seed!(seed)
    end

    z_shocks, z_shocks_idx = simulateProd(P_z, zgrid, T; z_0_idx = z_0_idx, N = N, set_seed = false) # N X T  

    return (z_shocks = z_shocks, z_shocks_idx = z_shocks_idx, N = N, T = T, z_0_idx = z_0_idx, seed = seed)
end

