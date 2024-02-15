"""
Solve for ρ, σ_ϵ such that we match quarterly ALP moments
"""
function calibrateZ(; ρ_y = 0.89, σ_y = 0.017, N_z = 13, a = 1.0, zbar = 1.0,
                    burnin_macro = 250, T_sim_macro = 828, N_sim_macro = 10^4,
                    set_seed = true, seed = 512)

    # Draw uniform shocks for macro moments
    if set_seed == true
        Random.seed!(seed)
    end

    z_shocks_macro  = rand(Uniform(0,1), T_sim_macro + burnin_macro, N_sim_macro) # z shocks: T x 1

    # Objective function
    opt                       = Opt(:LN_BOBYQA, 2) 
    obj(x, dummy_gradient!)   = simulateExogALP(x, z_shocks_macro, T_sim_macro, burnin, N_sim_macro; ρ_y = ρ_y, σ_y = σ_y,  N_z = N_z, a = a, zbar = zbar)
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
function simulateExogALP(x, z_shocks, T_sim, burnin, N_sim; ρ_y = 0.89, σ_y = 0.017, N_z = 13, a = 1.0, zbar = 1, λ = 10^5)
    
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


