"""
Simulate EGSS, given parameters defined in modd.
Solve the model with infinite horizon contract. Solve 
for θ and Y on every point in the productivity grid.
Then, compute the effort optimal effort a and wage w,
as a(z|z_0) and w(z|z_0). u0 = initial unemployment rate.
λ       = HP filtering parameter.
output  = "gdp" or "alp"

"""
function simulateEstZ(modd, shocks; u0 = 0.06, output_target = "gdp", check_mult = false, λ = 10^5, sd_cut = 3.0)
    
    # Initialize moments to export (default = 0)
    
    # Targeted
    std_Δlw   = 0.0  # st dev of YoY wage growth
    dlw1_du   = 0.0  # d log w_1 / d u
    dlw_dly   = 0.0  # passthrough: d log w_it / d log y_it
    u_ss      = 0.0  # stochastic mean of unemployment  
    y_ρ       = 0.0  # autocorrelation of quarterly output
    y_σ       = 0.0  # st dev of quarterly output

    # Get all of the relevant parameters, functions for the model
    @unpack hp, zgrid, logz, N_z, P_z, p_z, ψ, f, s, σ_η, χ, γ, hbar, ε, z_ss_idx, ρ, σ_ϵ = modd 

    # Generate model objects for every point on zgrid:
     @unpack θ_z, f_z, hp_z, y_z, lw1_z, pt_z, W, Y, flag_z, flag_IR_z, err_IR_z = getModel(modd; check_mult = check_mult) 

    # Composite flag; truncate simulation and only penalize IR flags for log z values within XX standard deviations of μ_z 
    σ_z     = σ_ϵ/sqrt(1 - ρ^2)
    idx_1   = findfirst(x-> x > -sd_cut*σ_z, logz) 
    idx_2   = findlast(x-> x <= sd_cut*σ_z, logz) 
    flag    = maximum(flag_z[idx_1:idx_2])
    flag_IR = maximum(flag_IR_z[idx_1:idx_2])

    # only compute moments if equilibria were found for all z
    if  max(flag_IR, flag) < 1 

        # Unpack the relevant shocks
        @unpack N_sim_micro, N_sim_macro, T_sim, burnin, z_shocks, η_shocks, s_shocks, jf_shocks = shocks

        # Scale η shocks by σ_η
        η_shocks = η_shocks*σ_η 

        # Compute model data for long z_t series (trim to post-burn-in when computing moment)
        z_idx    = simulateZShocks(P_z, z_shocks, N_sim_macro, T_sim + burnin; z0_idx = z_ss_idx)
        z_idx    = min.(max.(z_idx, idx_1), idx_2)  

        # Simulate annual wage changes, passthrough, and output
        @unpack std_Δlw, dlw_dly, y_ρ, y_σ = simulateWagesOutputEstZ(N_sim_micro, T_sim, burnin, z_idx[:,1], 
            s_shocks, jf_shocks, η_shocks, s, f_z, y_z, hp_z, pt_z, ψ, σ_η; λ = λ, output_target = output_target)

        # Other macro moments 
        @views lw1_t   = lw1_z[z_idx]     # E[log w_1 | z_t] series
        @views θ_t     = θ_z[z_idx]       # θ(z_t) series
        @views f_t     = f_z[z_idx]       # f(θ(z_t)) series

        # Bootstrap across N_sim_macro simulations
        dlw1_du_n      = zeros(N_sim_macro)     # cyclicality of new hire wages 

        # Compute evolution of unemployment for the z_t path
        T             = T_sim + burnin
        u_t           = zeros(T, N_sim_macro)
        u_t[1,:]     .= u0

        Threads.@threads for n = 1:N_sim_macro

            # Law of motion for unemployment
            @views @inbounds for t = 2:T
                u_t[t, n] = uLM(u_t[t-1,n], s, f_t[t-1,n])  #u_t[t-1,n] + s*(1 - u_t[t-1,n]) - (1-s)*f_t[t-1,n]*u_t[t-1,n]
            end

            # Estimate d E[log w_1] / d u (pooled ols)
            @views dlw1_du_n[n]  = cov(lw1_t[burnin+1:end, n], u_t[burnin+1:end, n])/max(eps(), var(u_t[burnin+1:end, n])) 
        
        end

        # Cyclicality of new hire wages 
        dlw1_du = mean(dlw1_du_n)

        # Stochastic mean of unemployment: E[u_t | t > burnin]
        u_ss    = mean(vec(mean(u_t[burnin+1:end,:], dims = 1)))

    end
    
    # determine an IR error for all initial z within 3 uncond. standard deviations
    IR_err = sqrt(sum((err_IR_z[idx_1:idx_2]).^2))

    # Export the simulation results
    return (std_Δlw = std_Δlw, dlw1_du = dlw1_du, dlw_dly = dlw_dly, u_ss = u_ss, y_ρ = y_ρ, 
            y_σ = y_σ, flag = flag, flag_IR = flag_IR, IR_err = IR_err)
    
end

"""
Build random shocks for the simulation when we are internally calibrating z.
N_sim_micro             = num workers for micro wage moments
T_sim                   = mum periods for micro wage moments  
burnin                  = length burn-in for micro moments
N_sim_macro             = num of seq to avg across for macro moments
"""
function drawShocksEstZ(; N_sim_micro = 2*10^4, T_sim = 1200, burnin = 250,
    N_sim_macro = 10^4, fix_a = false, set_seed = true, seed = 512)

    if set_seed == true
        Random.seed!(seed)
    end

    # Draw uniform and standard normal shocks for micro moments
    if fix_a == false
        η_shocks  = rand(Normal(0,1),  N_sim_micro, T_sim + burnin)              # η shocks: N x T
        s_shocks  = rand(Uniform(0,1), N_sim_micro, T_sim + burnin)              # separation shocks: N x T
        jf_shocks = rand(Uniform(0,1), N_sim_micro, T_sim + burnin)              # job-finding shocks: N x T
    elseif fix_a == true
        η_shocks  = 0
        s_shocks  = 0
        jf_shocks = 0
    end

    # Draw uniform shocks for macro moments
    z_shocks  = rand(Uniform(0,1), T_sim + burnin, N_sim_macro)              # z shocks: T x 1

    return (z_shocks = z_shocks, N_sim_micro = N_sim_micro, 
            N_sim_macro = N_sim_macro, T_sim = T_sim, burnin = burnin, 
            s_shocks = s_shocks, jf_shocks = jf_shocks, η_shocks = η_shocks)
end

"""
Simulate wage moments given N_sim x T_sim panel of z_it and η_it.
Simulate average labor productivity a*z for N_sim x T_sim jobs,
ignoring η shocks (iid). HP-filter log average output with smoothing parameter λ.
"""
function simulateWagesOutputEstZ(N_sim, T_sim, burnin, z_idx, s_shocks, jf_shocks, η_shocks, 
                                s, f_z, y_z, hp_z, pt_z, ψ, σ_η; λ = 10^5, output_target = "gdp")

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

    # Simulate workers, given job-finding, separation, and η shocks
    @views @inbounds for t = 2:T

        zt   = z_idx[t]       # current z_t
        ft   = f_z[zt]        # job-finding rate given current z_t 
        y_zz = y_z[zt, :]     # expected output given CURRENT z_t  
        hp_zz = hp_z[zt,:]    # h'(a) given CURRENT z_t
        pt_zz = pt_z[zt,:]    # pass-through given CURRENT z_t

        Threads.@threads for n = 1:N_sim

            if unemp[n] == false   
               
                # experience separation shock at end of t-1
                if s_shocks[n, t-1] < s               
                   
                    unemp_beg[n,t]       =  1.0             # became unemployed at start of period
                    
                    if  jf_shocks[n, t] < ft                # find a job and produce within the period t
                        z_1[n]      = zt                    # new initial z for contract
                        y_m[n, t]   = y_zz[zt]              # current output      
                        lw[n, t]    = ψ*hp_zz[zt]*η_shocks[n,t]  - 0.5*(ψ*hp_zz[zt]*σ_η)^2  
                        pt[n, t]    = pt_zz[zt] # +  ψ*hp_zz[z_1[n]]*η_shocks_micro[n,t] <- ignore for large sample
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

    # Compute micro wage moments using post-burn-in data
    @views ten_pb        = tenure[:, burnin+1:end]'  # reshape to T x N
    @views lw_pb         = lw[:, burnin+1:end]'      # reshape to T x N 
    @views pt_pb         = pt[:, burnin+1:end]'      # reshape to T x N
    @views avg_pt        = mean(pt_pb[isnan.(pt_pb).==0])

    # Compute YoY wage changes for 13 month spells
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
    @views Δlw     = dlw[isnan.(dlw).==0]
    std_Δlw        = isempty(Δlw) ? NaN : std(Δlw) 

    # Construct quarterly measures
    T_q            = Int(T_sim/3)
    ly_q           = zeros(T_q)                                      

    # Compute quarterly output series average
    @views @inbounds for t = 1:T_q
       
        t_q        = t*3
        y_i        = vec(y_m[:, burnin+1:end][:, (t_q - 2):t_q])       # output of all workers
        emp        = vec(active[:, burnin+1:end][:, (t_q - 2):t_q])    # workers that were producing 

        # total output across all people
        if output_target == "gdp"
            ly_q[t]    = log.(max(sum(y_i), eps()))   
        # avg output/worker (averaged across workers/hours in the quarter)
        elseif output_target == "alp"
            ly_q[t]    = log.(max(mean(y_i[emp.==1]), eps()))           
        end

    end

    # HP-filter the quarterly log output and unemployment series
    ly_q_resid, _ = hp_filter(ly_q, λ)

    # Compute standard deviation of output
    y_σ  = std(ly_q_resid)

    # Compute persistence of output
    y_ρ  = first(autocor(ly_q_resid, [1]))

    return (y_ρ = y_ρ, y_σ = y_σ, std_Δlw = std_Δlw, dlw_dly = avg_pt)
end
