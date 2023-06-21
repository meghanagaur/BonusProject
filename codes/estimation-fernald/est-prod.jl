#= Estimate productivity process parameters to match
the autocorrelation and standard deviation of the Fernald TFP series.
=#

include("functions/smm_settings.jl") 

"""
Solve for ρ, σ_ϵ such that we match the moments form the
log quarterly TFP process. Noramlize mean such that E[z_t] = 1.
"""
function calibrateZ(shocks; λ = 10^5, ρ_z = 0.87, σ_z = 0.016, N_z = 13, zbar = 1.0)

    # Objective function
    @unpack burnin, z_shocks_macro, T_sim_macro, N_sim_macro = shocks
    opt                       = Opt(:LN_BOBYQA, 2) # :LN_BOBYQA :LN_SBPLX :LN_COBYLA :LN_NELDERMEAD
    obj(x, dummy_gradient!)   = simulateTFP(x, z_shocks_macro, T_sim_macro, burnin, N_sim_macro; λ = λ, ρ_z = ρ_z, σ_z = σ_z,  N_z = N_z, zbar = zbar)
    opt.min_objective         = obj

    # Bound constraints
    opt.lower_bounds        = [0.9, 0.0001] 
    opt.upper_bounds        = [0.999, 0.01] 

    # tolerance and time settings 
    opt.stopval            = 1e-8
    opt.ftol_rel           = 1e-8
    opt.ftol_abs           = 1e-8
    opt.xtol_rel           = 0.0  
    opt.maxtime            = 60*5

    # Search for parameters 
    init                  = [0.98, 0.005]
    (min_f, arg_min, ret) = NLopt.optimize(opt, init)

    println("minimum:\t\t"*string(min_f))
    println("minimizer:\t\t"*string(arg_min))
    println("reason for stopping:\t"*string(ret))
    
    return (ρ = arg_min[1], σ_ϵ = arg_min[2])
end

"""
Simulate quarterly TFP series and compare to data moments.
"""
function simulateTFP(x, z_shocks, T_sim, burnin, N_sim; λ = 10^5, ρ_z = 0.87, σ_z = 0.016, N_z = 13, zbar = 1)
    
    ρ               = x[1]
    σ_ϵ             = x[2]
    μ_z             = log(zbar) - (σ_ϵ^2)/(2*(1-ρ^2))   # normalize E[logz], so that E[z_t] = 1
    logz, P_z, p_z  = rouwenhorst(μ_z, ρ, σ_ϵ, N_z)     # log z grid, transition matrix, invariant distribution
    zgrid           = exp.(logz)                        # z grid in levels
    z_ss_idx        = findfirst(isapprox(μ_z, atol = 1e-6), logz )

    # simulate ALP
    z_idx_macro     = simulateZShocks(P_z, p_z, z_shocks, N_sim, T_sim + burnin; z0_idx = z_ss_idx)
    zshocks_macro   = zgrid[z_idx_macro]
    T               = T_sim + burnin
    T_q_macro       = Int(T_sim/3)
    tfp_ρ_n         = zeros(N_sim)
    tfp_σ_n         = zeros(N_sim)

    Threads.@threads for n = 1:N_sim
        # Compute the quarterly average of output
        @views lz_q            = log.([mean(zshocks_macro[burnin+1:end, n][(t_q*3 - 2):t_q*3]) for t_q = 1:T_q_macro])
        @views lz_q_resid, _   = hp_filter(lz_q, λ)  
        @views tfp_ρ_n[n]      = first(autocor(lz_q_resid, [1]))
        @views tfp_σ_n[n]      = std(lz_q_resid)
    end

    # Take cross-simulation averages
    tfp_ρ   = mean(tfp_ρ_n)
    tfp_σ   = mean(tfp_σ_n)

    # Compute the error
    err     = (tfp_ρ - ρ_z)^2 + (tfp_σ - σ_z)^2

    return err
end

# Get productivity parameters 
N_sim_macro     = 10^4
burnin          = 5000
T_sim_macro     = 828
z_shocks_macro  = rand(Uniform(0,1), T_sim_macro + burnin, N_sim_macro)          # z shocks: T x 1

shocks  = (z_shocks_macro = z_shocks_macro, T_sim_macro = T_sim_macro, burnin =burnin, N_sim_macro = N_sim_macro)
ρ, σ_ϵ  = calibrateZ(shocks)
    
#=
minimum:                7.0008927135002045e-9
minimizer:              [0.9662368625721576, 0.005553431765652789]
reason for stopping:    STOPVAL_REACHED
(ρ = 0.9662368625721576, σ_ϵ = 0.005553431765652789)
=#