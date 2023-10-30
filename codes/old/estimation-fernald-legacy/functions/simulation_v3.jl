"""
Simulate EGSS, given parameters defined in tuple m.
Solve the model with infinite horizon contract. Solve 
for θ and Y on every point in the productivity grid.
Then, compute the effort optimal effort a and wage w,
as a(z|z_0) and w(z|z_0). u0 = initial unemployment rate.
"""
function simulate(modd, shocks; u0 = 0.06, check_mult = false, est_alp = false, λ = 10^5)
    
    # Initialize moments to export (default = NaN)
    std_Δlw   = 0.0  # st dev of YoY wage growth
    dlw1_du   = 0.0  # d log w_1 / d u
    dlw_dly   = 0.0  # passthrough: d log w_it / d log y_it
    u_ss      = 0.0  # u_ss <- nonstochastic steady state
    alp_ρ     = 0.0  # autocorrelation of quarterly average labor prod.
    alp_σ     = 0.0  # st dev of quarterly average labor prod.
    dlu_dly   = 0.0  # d log u / d log y
    std_u     = 0.0  # st dev of log u_t
    std_z     = 0.0  # st dev of log z_t
    u_ss_2    = 0.0  # stochastic mean of unemployment

    # Get all of the relevant parameters, functions for the model
    @unpack hp, zgrid, logz, N_z, P_z, p_z, ψ, f, s, σ_η, χ, γ, hbar, ε, z_ss_idx, ρ, σ_ϵ = modd 

    # Generate model data for every point on zgrid:

    # Build vectors     
    θ_z       = zeros(N_z)               # θ(z_1)
    f_z       = zeros(N_z)               # f(θ(z_1))
    hp_z      = zeros(N_z, N_z)          # h'(a(z_i | z_j))
    y_z       = zeros(N_z, N_z)          # a(z_i | z_j)*z_i
    lw1_z     = zeros(N_z)               # E[log w1|z] <- wages of new hires    
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

            # Expectation of the log wage of new hires, given z_0 = z
            hp_z[:,iz]    = hp.(az)  # h'(a(z|z_1))

            # Expectation of the log wage of new hires, given z_0 = z
            lw1_z[iz]     = log(max(eps(), w_0)) - 0.5*(ψ*hp_z[iz, iz]*σ_η)^2 
           
            # Tightness and job-finding rate, given z_0 = z
            θ_z[iz]       = θ      
            f_z[iz]       = f(θ)       

            # Compute passthrough: elasticity of w_it wrt y_it 
            pt_z[:,iz]    = ψ*hbar*az.^(1 + 1/ε)
        end

    end

    # Composite flag; compute simulation for 3 standard deviations of z shock
    σ_z  = σ_ϵ/sqrt(1 - ρ^2)
    idx  = findfirst(x-> x > -3σ_z, logz) 
    flag = max(maximum(flag_z), sum(flag_IR_z) == N_z)

    # only compute moments if equilibria were found for all z
    if (flag < 1) 

        # Unpack the relevant shocks
        @unpack N_sim_micro, T_sim_micro, N_sim_macro, T_sim_macro, z_idx_macro, N_sim_macro_est_alp, N_sim_macro_workers, 
            burnin, z_idx_micro, η_shocks_micro, s_shocks_micro, jf_shocks_micro, s_shocks_macro, jf_shocks_macro = shocks

        # scale normal shocks by σ_η
        η_shocks_micro = η_shocks_micro*σ_η 
        z_idx_micro    = max.(z_idx_micro, idx)

        # Simulate annual wage changes + passthrough
        @unpack std_Δlw, dlw_dly = simulateWageMoments(η_shocks_micro, s_shocks_micro, jf_shocks_micro, z_idx_micro, N_sim_micro, T_sim_micro, burnin, hp_z, pt_z, f_z, ψ, σ_η, s)        

        # Macro moments 
        z_idx_macro    = max.(z_idx_macro, idx)
        @views lw1_t   = lw1_z[z_idx_macro]     # E[log w_1 | z_t] series
        @views θ_t     = θ_z[z_idx_macro]       # θ(z_t) series
        @views f_t     = f_z[z_idx_macro]       # f(θ(z_t)) series
        zshocks_macro  = zgrid[z_idx_macro]     # z shocks

        # Bootstrap across N_sim_macro simulations
        dlw1_du_n      = zeros(N_sim_macro)
        std_u_n        = zeros(N_sim_macro)
        std_z_n        = zeros(N_sim_macro)

        # Bootstrap across N_sim_macro_est_alp simulations
        alp_ρ_n        = zeros(N_sim_macro_est_alp) 
        alp_σ_n        = zeros(N_sim_macro_est_alp)
        dlu_dly_n      = zeros(N_sim_macro_est_alp)

        # Compute evolution of unemployment for the z_t path
        T             = T_sim_macro + burnin
        T_q_macro     = Int(T_sim_macro/3)
        u_t           = zeros(T, N_sim_macro)
        u_t[1,:]     .= u0

        Threads.@threads for n = 1:N_sim_macro

            @views @inbounds for t = 2:T
                u_t[t, n] = (1 - f_t[t-1,n])*u_t[t-1,n] + s*(1 - u_t[t-1,n])
            end

            # Estimate d E[log w_1] / d u (pooled ols)
            @views dlw1_du_n[n]  = cov(lw1_t[burnin+1:end, n], u_t[burnin+1:end, n])/max(eps(), var(u_t[burnin+1:end, n]))

            # Compute quarterly average of log u_t, z_t in post-burn-in period
            @views u_q           = [mean(u_t[burnin+1:end, n][(t_q*3 - 2):t_q*3]) for t_q = 1:T_q_macro] 
            @views z_q           = [mean(zshocks_macro[burnin+1:end, n][(t_q*3 - 2):t_q*3]) for t_q = 1:T_q_macro] 

            # hp-filter the quarterly log unemployment series, nudge to avoid runtime error
            logu_q_resid, _      = hp_filter(log.(max.(u_q, eps())), λ)  
            logz_q_resid, _      = hp_filter(log.(max.(z_q, eps())), λ)  

            # Compute the standard deviation
            std_u_n[n]           = std(logu_q_resid)
            std_z_n[n]           = std(logz_q_resid)
        end

        # Standard deviation and persistence of average labor productivity 
        if est_alp == true
            
            Threads.@threads for n = 1:N_sim_macro_est_alp
                alp_ρ_n[n], alp_σ_n[n], dlu_dly_n[n] = simulateALP(z_idx_macro[:,n], s_shocks_macro, jf_shocks_macro, N_sim_macro_workers, T_sim_macro, 
                                                                   burnin, T_q_macro, s, f_z, y_z; λ = λ)
            end

            alp_ρ   = mean(alp_ρ_n)
            alp_σ   = mean(alp_σ_n)

        end

        # Compute cross-simulation averages
        dlw1_du = mean(dlw1_du_n)
        std_u   = mean(std_u_n) 
        std_z   = mean(std_z_n)
        dlu_dly = mean(dlu_dly_n)

        # Compute cross-simulation average of stochastic mean of unemployment: E[u_t | t > burnin]
        u_ss    = mean(vec(mean(u_t[burnin+1:end,:], dims = 1)))
 
        # Compute nonstochastic SS unemployment: define u_ss = s/(s + f(θ(z_ss)), at log z_ss = μ_z
        u_ss_2  = s/(s  + f(θ_z[z_ss_idx]))
    end
    
    # determine an IR error for all initial z within 3 uncond. standard deviations
    IR_err = sum(abs.(err_IR_z[idx:end]))

    # Export the simulation results
    return (std_Δlw = std_Δlw, dlw1_du = dlw1_du, dlw_dly = dlw_dly, u_ss = u_ss, u_ss_2 = u_ss_2, alp_ρ = alp_ρ, 
    alp_σ = alp_σ, dlu_dly = dlu_dly, std_u = std_u, std_z = std_z, flag = flag, flag_IR = maximum(flag_IR_z[idx:end]), IR_err = IR_err)
end

"""
Build random shocks for the simulation.
N_sim_micro             = num workers for micro wage moments
T_sim_micro             = mum periods for micro wage moments  
N_sim_macro             = num seq to avg across for macro moments
N_sim_macro_workers     = num workers for prod moments
T_sim_macro             = num periods for agg sequences: 69 years 
burnin                  = length burn-in for agg sequence
N_sim_macro_est_alp     = num seq to avg across for prod moments (<= N_sim_macro)
"""
function rand_shocks(P_z, p_z; z0_idx = 0, N_sim_micro = 10^4, T_sim_micro = 2000, N_sim_macro = 10^4, 
    N_sim_macro_workers = 10^3, T_sim_macro = 828, burnin = 1000, N_sim_macro_est_alp = 10^3, set_seed = true, seed = 512)

    if set_seed == true
        Random.seed!(seed)
    end

    # Draw uniform and standard normal shocks for micro moments
    η_shocks_micro  = rand(Normal(0,1), N_sim_micro,  T_sim_micro + burnin)          # η shocks: N x T
    z_shocks_micro  = rand(Uniform(0,1), T_sim_micro + burnin)                       # z shocks: N x T
    s_shocks_micro  = rand(Uniform(0,1), N_sim_micro, T_sim_micro + burnin)          # separation shocks: N x T
    jf_shocks_micro = rand(Uniform(0,1), N_sim_micro, T_sim_micro + burnin)          # job-finding shocks: N x T

    # Draw uniform shocks for macro moments
    z_shocks_macro  = rand(Uniform(0,1), T_sim_macro + burnin, N_sim_macro)          # z shocks: T x 1
    s_shocks_macro  = rand(Uniform(0,1), N_sim_macro_workers, T_sim_macro + burnin)  # separation shocks: N x T
    jf_shocks_macro = rand(Uniform(0,1), N_sim_macro_workers, T_sim_macro + burnin)  # job-finding shocks: N x T

    # Build shocks for wages: z_it, η_it
    z_idx_micro    = simulateZShocks(P_z, p_z, z_shocks_micro, 1, T_sim_micro + burnin; z0_idx = z0_idx)
    # Compute model data for long z_t series (trim to post-burn-in when computing moment)
    z_idx_macro    = simulateZShocks(P_z, p_z, z_shocks_macro, N_sim_macro, T_sim_macro + burnin; z0_idx = z0_idx)
        
    return (z_idx_micro = z_idx_micro, z_idx_macro = z_idx_macro, N_sim_micro = N_sim_micro, T_sim_micro = T_sim_micro, 
            s_shocks_micro = s_shocks_micro, jf_shocks_micro = jf_shocks_micro, η_shocks_micro = η_shocks_micro,
            N_sim_macro = N_sim_macro, N_sim_macro_workers = N_sim_macro_workers, T_sim_macro = T_sim_macro, burnin = burnin, 
            s_shocks_macro = s_shocks_macro, jf_shocks_macro = jf_shocks_macro, N_sim_macro_est_alp = N_sim_macro_est_alp)
end

"""
Simulate T_sim x N_sim panel of productivity draws.
"""
function simulateZShocks(P_z, p_z, z_shocks, N_sim, T_sim; z0_idx = 0)
   
    PI_z         = cumsum(P_z, dims = 2)         # CDF of transition density
    pi_z         = cumsum(p_z)                   # CDF of invariant distribution
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
Simulate wage moments givenn N x T panel of z_it and η_it
"""
function simulateWageMoments(η_shocks_micro, s_shocks_micro, jf_shocks_micro, z_idx_micro, N_sim_micro, T_sim_micro, burnin, hp_z, pt_z, f_z, ψ, σ_η, s)

    # Active jobs -- ignore the constant in log wages, since drops out for wage changes
    T          = T_sim_micro + burnin
    lw         = zeros(N_sim_micro, T) # N x T panel of log wages
    pt         = zeros(N_sim_micro, T) # N x T panel of pass-through
    lw[:,1]   .= ψ*hp_z[z_idx_micro[1], z_idx_micro[1]]*η_shocks_micro[:,1]  .- 0.5*(ψ*hp_z[z_idx_micro[1], z_idx_micro[1]]*σ_η)^2 
    pt[:,1]   .= pt_z[z_idx_micro[1], z_idx_micro[1]] #.+ ψ*hp_z[z_idx_micro[1], z_idx_micro[1]]*η_shocks_micro[:,1]  
    active     = ones(N_sim_micro, T)
    unemp      = zeros(N_sim_micro)
    z_1        = fill(z_idx_micro[1], N_sim_micro)

    @views @inbounds for t = 2:T

        # index by current z_t
        zt    = z_idx_micro[t]
        ft    = f_z[zt]
        hp_zz = hp_z[zt,:]  
        pt_zz = pt_z[zt,:] 
        
        Threads.@threads for n = 1:N_sim_micro

            if unemp[n] == false   
                # separation shock 
                if s_shocks_micro[n, t] < s           # become unemployed
                    unemp[n]     = true
                    active[n, t] = 0                  # remain employed
                else
                    lw[n, t]    = lw[n,t-1] + ψ*hp_zz[z_1[n]]*η_shocks_micro[n,t]  - 0.5*(ψ*hp_zz[z_1[n]]*σ_η)^2  
                    pt[n, t]    = pt_zz[z_1[n]]       # +  ψ*hp_zz[z_1[n]]*η_shocks_micro[n,t] 
                end
            elseif unemp[n] == true
                # job-finding shock
                if jf_shocks_micro[n, t] < ft         # find a job
                    unemp[n]    = false               # become employed
                    z_1[n]      = zt                  # new initial z for contract
                    lw[n, t]    = ψ*hp_zz[zt]*η_shocks_micro[n,t]  - 0.5*(ψ*hp_zz[zt]*σ_η)^2 
                    pt[n, t]    = pt_zz[zt]           # +  ψ*hp_zz[zt]*η_shocks_micro[n,t] 
                else                                  # remain unemployed
                    active[n,t] = 0                 
                end 
            end

        end
    end

    # compute micro wage moments using post-burn-in data
    act_pb = active[:, burnin+1:end]'  # reshape to T x N
    lw_pb  = lw[:, burnin+1:end]'      # reshape to T x N 
    pt_pb  = pt[:, burnin+1:end]'      # reshape T x N
    avg_pt = mean(pt_pb[act_pb.==1.0])

    # compute YoY wage changes for 13 month spells
    dlw_dict = OrderedDict{Int, Array{Real,1}}()
    @views @inbounds for n = 1:N_sim_micro
        
        # record separations for given worker
        sep   = findall(isequal(0), act_pb[:, n])
        sep   = [0; sep; T_sim_micro + 1]
        lw_n  = lw_pb[:, n]
        dlw_n = []

        # go through employment spells
        @inbounds for i = 2:lastindex(sep)
            
            t0    = sep[i-1] + 1 # beginning of job spell
            t1    = sep[i] - 1   # end of job spell
            
            if t1 - t0 >= 12
                spells = collect(t0:12:t1)
                if spells[end] + 12 > t1
                    t2 = spells[end-1]
                else
                    t2 = spells[end]
                end
                dlw_s  = [lw_n[t+12] - lw_n[t] for t = t0:12:t2]
                dlw_n  = vcat(dlw_n, dlw_s)
            end

        end

        dlw_dict[n] = dlw_n
       
    end
  
    dlw=[]
    for (k,v) in dlw_dict
       dlw = vcat(dlw, v)
    end

    # Stdev of YoY log wage changes for job-stayers
    std_Δlw  = isempty(dlw) ? NaN : std(dlw) 

    return (std_Δlw = std_Δlw, dlw_dly = avg_pt)
end

"""
Simulate average labor productivity a*z for N_sim x T_sim jobs,
ignoring η shocks.
HP-filter log average output with smoothing parameter λ.
"""
function simulateALP(z_idx_macro, s_shocks_macro, jf_shocks_macro, 
                        N_sim_macro_workers, T_sim_macro, burnin, T_q_macro, s, f_z, y_z; λ = 10^5)

    # Active jobs
    T          = T_sim_macro + burnin
    y_m        = zeros(N_sim_macro_workers, T) # N x T panel of output
    y_m[:,1]  .= y_z[z_idx_macro[1], z_idx_macro[1]]
    active     = ones(N_sim_macro_workers, T)
    unemp      = zeros(N_sim_macro_workers)
    z_1        = fill(z_idx_macro[1], N_sim_macro_workers)

    @inbounds for t = 2:T

        zt = z_idx_macro[t]
        ft = f_z[zt]

         @views @inbounds for n = 1:N_sim_macro_workers

            if unemp[n] == false   
                # separation shock 
                if s_shocks_macro[n, t] < s              # become unemployed
                    unemp[n]     = true
                    active[n, t] = 0
                else
                    y_m[n, t]    = y_z[zt, z_1[n]] # remain employed      
                end
            elseif unemp[n] == true
                # job-finding shock
                if jf_shocks_macro[n, t] < ft      # find a job
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
    ly_q = zeros(T_q_macro)                           # quarterly log output
    u_t  = mean(1 .- active[:, burnin+1:end], dims=1) # monthly unemployment

    @views @inbounds for t = 1:T_q_macro
        t_q      = t*3
        output   = vec(y_m[:, burnin+1:end][:, (t_q - 2):t_q])    # output
        emp      = vec(active[:, burnin+1:end][:, (t_q - 2):t_q]) # who is employed
        ly_q[t]  = log.(max(mean(output[emp.==1]), eps()))        # avg output/worker (quarterly)
        #lu_q[t] = log.(max(1 - mean(emp), eps()))                # average quarterly unemployment
    end

    # Estimate d log u_t+1 / d  log ALP_t (quarterly, pooled OLS)
    lu_q            = log.([mean(u_t[(t_q*3 - 2):t_q*3]) for t_q = 1:T_q_macro])
    @views dlu_dlz  = cov(lu_q[2:end], ly_q[1:end-1])/max(eps(), var(ly_q[1:end-1]))

    # hp-filter the quarterly log output and unemployment series
    ly_q_resid, _ = hp_filter(ly_q, λ)

    # Compute standard deviation of log ALP
    alp_σ  = std(ly_q_resid)

    # Compute persistence of log ALP (OLS)
    alp_ρ  = first(autocor(ly_q_resid, [1]))

    return (alp_ρ = alp_ρ, alp_σ = alp_σ, dlu_dlz = dlu_dlz)
end

"""
Directly simulate N X T shock panel for z. Include a burn-in period.
"""
function drawZShocks(P_z, zgrid; N = 10000, T = 100, z_0_idx = median(1:length(zgrid)), set_seed = true, seed = 512)
    
    if set_seed == true
        Random.seed!(seed)
    end

    z_shocks, z_shocks_idx = simulateProd(P_z, zgrid, T; z_0_idx = z_0_idx, N = N, set_seed = false) # N X T  

    return (z_shocks = z_shocks, z_shocks_idx = z_shocks_idx, N = N, T = T, z_0_idx = z_0_idx, seed = seed)
end