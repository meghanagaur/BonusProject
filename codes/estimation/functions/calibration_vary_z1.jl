"""
Vary z_1, and compute relevant aggregates.
B denotes Bonus model.
"""
function vary_z1(modd; check_mult = false)

    @unpack ψ, zgrid, z_ss, N_z = modd
    modds      = OrderedDict{Int64, Any}()
    z_ss_idx   = Int64(median(1:N_z))

    # Solve the model for different z_0
    Threads.@threads for iz = 1:N_z
        modds[iz] =  solveModel(modd; z_1 = zgrid[iz], noisy = false, check_mult = check_mult)
    end

    ## Store series of interest
    w_0    = [modds[i].w_0 for i = 1:N_z]          # w0 (constant)
    θ_1    = [modds[i].θ for i = 1:N_z]            # tightness
    W_1    = w_0/ψ                                 # EPV of wages
    Y_1    = [modds[i].Y for i = 1:N_z]            # EPV of output
    ω_1    = [modds[i].ω_0 for i = 1:N_z]          # EPV value of unemployment at z0
    J_1    = Y_1 - W_1                             # EPV profits
    a_1    = [modds[i].az[i] for i = 1:N_z]        # optimal effort @ start of contract
    aflag  = [modds[i].effort_flag for i = 1:N_z] 

    return (modds = modds, w_0_B = w_0, θ_B = θ_1, W_B = W_1, Y_B = Y_1, ω_B = ω_1, J_B = J_1, 
            a_B = a_1, z_ss_idx = z_ss_idx, zgrid = zgrid, aflag = aflag)
end

"""
Produce Hall economy aggregates, matching SS Y and W from Bonus economy.
"""
function solveHall(modd, z_ss_idx, Y_B, W_B)

    @unpack zgrid, κ, ι, s, β, N_z, P_z = modd
    
    # Solve for expected PV of sum of the z_t's
    exp_z = zeros(length(zgrid)) 

    Threads.@threads for iz = 1:N_z

        # initialize guesses
        v0     = zgrid./(1-β*(1-s))
        v0_new = zeros(N_z)
        iter   = 1
        err    = 10
        
        # solve via simple value function iteration
        @inbounds while err > 10^-10 && iter < 500
            v0_new = zgrid + β*(1-s)*P_z*v0
            err    = maximum(abs.(v0_new - v0))
            v0     = copy(v0_new)
            iter +=1
        end

        exp_z[iz]   = v0[iz]
    end

    aa       = Y_B[z_ss_idx]./exp_z[z_ss_idx]     # exactly match SS PV of output in the 2 models
    WW       = W_B[z_ss_idx]                      # match SS PV of wages (E_0[w_t] = w_0 from martingale property)
    YY       = aa.*exp_z                          # Hall economy output 
    JJ       = YY .- WW                           # Hall economy profits
    qθ       = min.(1, max.(0, κ./JJ))            # job-filling rate
    θ        = (qθ.^(-ι) .- 1).^(1/ι).*(qθ .!= 0) # implied tightness

    return (a_H = aa, W_H = WW, J_H = JJ, Y_H = YY, θ_H = θ, zz = exp_z)
end

"""
Approximate slope of y(x) by forward or central finite differences,
where y and x are both vectors.
"""
function slope(y, x; diff = "forward")
    if diff == "forward"
        return (y[2:end] - y[1:end-1])./(x[2:end] - x[1:end-1])
        dydx = [dydx ; NaN]
    elseif diff == "central" 
        dydx = (y[3:end] - y[1:end-2])./(x[3:end] - x[1:end-2])
        dydx = [NaN; dydx ; NaN]
    end
    return dydx
end

"""
Return d log theta/d log z at the steady state μ_z.
"""
function dlogtheta(modd; N_z = 21)

    # vary initial z1
    modd2 = model(N_z = N_z, σ_η = modd.σ_η, χ = modd.χ, γ = modd.γ, hbar = modd.hbar, ε = modd.ε)
    @unpack θ_B, z_ss_idx, zgrid = vary_z1(modd2)

    # compute d log theta/d log z
    dlogθ  = slope(θ_B, zgrid).*zgrid[1:end-1]./θ_B[1:end-1]

    return  dlogθ[z_ss_idx]
end

"""
Simulate moments for heatmaps
"""
function heatmap_moments(; σ_η = 0.406231, χ = 0.578895, γ = 0.562862, hbar = 3.52046, ε = 0.3)

    baseline     = model(σ_η = σ_η, hbar = hbar, ε = ε, γ = γ,  χ = χ) 
    out          = simulate(baseline, shocks)
    dlogθ_dlogz  = dlogtheta(baseline)

    mod_mom  = [out.std_Δlw, out.dlw1_du, out.dlw_dly, out.u_ss, dlogθ_dlogz, out.dlw_dly_2, out.u_ss_2]  
    # out.dlw1_dlz, out.avg_Δlw,  out.dlY_dlz, out.dlu_dlz, out.std_u, out.std_z, out.std_Y, out.std_w0]

    # Flags
    flag     = out.flag
    flag_IR  = out.flag_IR
    IR_err   = out.IR_err

    return [mod_mom, flag, flag_IR, IR_err]
end

"""
Simulate employment, given θ(z_t) path
"""
function simulate_employment(modd, T_sim, burnin, θ; minz_idx = 1, u0 = 0.067, seed = 512)

    @unpack s, f, zgrid, P_z = modd

    # Get sequence of productivity shocks
    shocks  = simulateZShocks(P_z, zgrid, N = 1, T = T_sim + burnin, set_seed = true, seed = seed)
    @unpack z_shocks, z_shocks_idx, T = shocks

    # Truncate the simulated productivity series
    z_shocks_idx = max.(z_shocks_idx, minz_idx)

    # Get the θ(z_t) series.
    @views θ_t   = θ[z_shocks_idx]   

    # Compute evolution of unemployment for z_t path
    u_t      = zeros(T)
    u_t[1]   = u0

    @inbounds for t = 2:T
        u_t[t] = (1 - f(θ_t[t-1]))*u_t[t-1] + s*(1 - u_t[t-1])
    end

    return (nt = 1 .- u_t[burnin+1:end], zt_idx = z_shocks_idx[burnin+1:end])

end