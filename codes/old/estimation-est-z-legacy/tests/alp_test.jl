cd(dirname(@__FILE__))

include("../functions/smm_settings.jl")

println(Threads.nthreads())

N_sim_macro_workers = 5*10^4

function simulateALP_alt(z_shocks_idx, s_shocks, jf_shocks, η_shocks_macro, z_grid, 
    N_sim, T_sim, burnin, s, f_z, y_z; λ = 10^5, eta = 0)

    # Active jobs
    s_shocks   = s_shocks'
    jf_shocks  = jf_shocks'
    T          = T_sim + burnin
    y_m        = zeros(T, N_sim) # N x T panel of output
    inactive   = zeros(T, N_sim)

    Threads.@threads for n = 1:N_sim

        unemp = false
        z_1   = z_shocks_idx[1]

        @inbounds for t = 2:T

            zt = z_shocks_idx[t]
            ft = f_z[zt]

            if unemp[n] == false   
                # separation shock 
                if s_shocks[t, n] < s              # become unemployed
                    unemp          = true
                    inactive[t, n] = 0     
                end
            elseif unemp == true
                # job-finding shock
                if jf_shocks[t,n] < ft            # find a job
                    unemp         = false          # become employed
                    z_1           = zt             # new initial z for contract
                else                               # remain unemployed
                    inactive[t, n] = 0
                end 
            end

            if t > burnin
                y_m[t, n]  = y_z[zt, z_1]       # remain employed      
            end

        end

    end

    # construct quarterly averages
    T_q       = Int64(T_sim/3)
    ly_q      = zeros(T_q) # quarterly log output
    y_pb      = y_m[burnin+1:end,:]' 
    ina       = inactive[burnin+1:end,:]' 
    y_pb     += eta*η_shocks_macro.*zgrid[z_shocks_idx[burnin+1:end]]'

    @views @inbounds for t = 1:T_q
        t_q     = t*3
        output  = vec(y_pb[:,t_q - 2:t_q])
        unemp   = vec(ina[:,t_q - 2:t_q])
        ly_q[t] = log.(max(mean(output[unemp.==0]), eps()))
    end

    # hp-filter the quarterly log output series
    ly_q_resid, _ = hp_filter(ly_q, λ)

    # Compute standard deviation of log ALP
    alp_σ  = std(ly_q_resid)

    # Compute persistence of log ALP (OLS)
    alp_ρ  = first(autocor(ly_q_resid, [1]))

    return (alp_ρ = alp_ρ, alp_σ = alp_σ)
end

# Unpack the relevant shocks
shocks  = rand_shocks(N_sim_macro_workers = N_sim_macro_workers, N_sim_macro = 1)

 # Get all of the relevant parameters, functions for the model
 modd = model()
 @unpack hp, zgrid, logz, N_z, P_z, p_z, ψ, f, s, σ_η, χ, γ, hbar, ε = modd

 # Generate model data for every point on zgrid:

 # Build vectors     
 θ_z       = zeros(N_z)               # θ(z_1)
 f_z       = zeros(N_z)               # f(θ(z_1))
 hp_z      = zeros(N_z, N_z)          # h'(a(z_i | z_j))
 y_z       = zeros(N_z, N_z)          # a(z_i | z_j)*z_i
 lw1_z     = zeros(N_z)               # E[log w1|z] <- wages of new hires    
 pt_z      = zeros(N_z, N_z)          # pass-through: ψ*hbar*a(z_i | z_j)^(1 + 1/ε)
 flag_z    = zeros(Int64,N_z)         # convergence/effort/wage flags
 flag_IR_z = zeros(Int64,N_z)         # IR flags
 err_IR_z  = zeros(N_z)               # IR error

 Threads.@threads for iz = 1:N_z

     # Solve the model for z_1 = zgrid[iz]
     sol = solveModel(modd; z_1 = zgrid[iz], noisy = false, check_mult = false)

     @unpack conv_flag1, conv_flag2, conv_flag3, wage_flag, effort_flag, IR_err, flag_IR, az, yz, w_0, θ, Y = sol
     
     # Record flags
     flag_z[iz]    = maximum([conv_flag1, conv_flag2, conv_flag3, wage_flag, effort_flag])
     flag_IR_z[iz] = flag_IR
     err_IR_z[iz]  = IR_err

     if flag_z[iz] < 1             

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
 flag = max(maximum(flag_z), sum(flag_IR_z) == N_z)

 # only compute moments if equilibria were found for all z
 if (flag < 1) 

     # Unpack the relevant shocks
     @unpack  N_sim_macro, T_sim_macro, N_sim_macro_workers, burnin, η_shocks_macro, z_shocks_macro, s_shocks, jf_shocks = shocks

     # scale normal shocks by σ_η
     η_shocks_macro = η_shocks_macro*σ_η 

     # Compute model data for long z_t series (trim to post-burn-in when computing moment)
     logz_ss_idx    = Int64(median(1:N_z))
     z_idx_macro    = simulateZShocks(P_z, p_z, z_shocks_macro, N_sim_macro, T_sim_macro + burnin; z_1_idx = logz_ss_idx)
    
     # Macro moments 

    # Standard deviation and persistence of average labor productivity 
    alp_eta = simulateALP_alt(z_idx_macro, s_shocks, jf_shocks, η_shocks_macro, zgrid,
                                         N_sim_macro_workers, T_sim_macro, burnin, s, f_z, y_z; eta = 1)

    println("with eta")
    println("alp_ρ:"*string(alp_eta.alp_ρ))
    println("alp_σ:"*string(alp_eta.alp_σ))

    # Standard deviation and persistence of average labor productivity 
    alp_no_eta = simulateALP_alt(z_idx_macro, s_shocks, jf_shocks, η_shocks_macro, zgrid,
                                         N_sim_macro_workers, T_sim_macro, burnin, s, f_z, y_z, eta = 0)

    println("without eta")
    println("alp_ρ:"*string(alp_no_eta.alp_ρ))
    println("alp_σ:"*string(alp_no_eta.alp_σ))

 end