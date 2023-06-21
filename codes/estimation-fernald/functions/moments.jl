"""
Vary z_1, and compute relevant aggregates.
B denotes Bonus model.
"""
function vary_z1(modd; check_mult = false, fix_a = false, a = 1.0)

    @unpack ψ, zgrid, z_ss_idx, N_z = modd

    w_0     = zeros(N_z)        # w0 (constant)
    θ_0     = zeros(N_z)        # tightness
    W_0     = zeros(N_z)        # EPV of wages
    Y_0     = zeros(N_z)        # EPV of output
    ω_0     = zeros(N_z)        # EPV value of unemployment at z0
    J_0     = zeros(N_z)        # EPV profits
    a1      = zeros(N_z)        # optimal effort @ start of contract
    az      = zeros(N_z, N_z)   # optimal effort (j = initial z)
    aflag   = zeros(N_z)        # effort flag
    IR_flag = zeros(N_z)        # IR flag 

    # Solve the model for different z_0
    Threads.@threads for iz = 1:N_z
        
        if fix_a == true
            sol = solveModelFixedEffort(modd; a = a, z_0 = zgrid[iz], noisy = false)
        elseif fix_a == false
            sol = solveModel(modd; z_0 = zgrid[iz], noisy = false, check_mult = check_mult)
        end
        
        ## Store series of interest
        w_0[iz]     = sol.w_0         
        θ_0[iz]     = sol.θ       
        W_0[iz]     = sol.w_0/ψ       
        Y_0[iz]     = sol.Y            
        ω_0[iz]     = sol.ω             
        J_0[iz]     = Y_0[iz] - W_0[iz] 
        a1[iz]      = sol.az[iz]      
        aflag[iz]   = sol.effort_flag
        IR_flag[iz] = sol.flag_IR  
        az[:,iz]    = sol.az         
    end

    return (az = az, w_0 = w_0, θ = θ_0, W = W_0, Y = Y_0, ω = ω_0, J = J_0, 
            a1 = a1, zgrid = zgrid, aflag = aflag, IR_flag = IR_flag)
end

"""
Produce Hall economy aggregates, matching SS Y_B and W_B from Bonus economy.
"""
function solveHall(modd, Y_B, W_B)

    @unpack zgrid, κ, ι, s, β, N_z, P_z, z_ss_idx = modd
    
    # Solve for expected PV of sum of the z_t's 
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
    

    a       = Y_B[z_ss_idx]./v0[z_ss_idx]        # exactly match SS PV of output in the 2 models
    W       = W_B[z_ss_idx]                      # match SS PV of wages (E_0[w_t] = w_0 from martingale property)
    Y       = a.*v0                              # Hall economy output 
    J       = Y .- W                             # Hall economy profits
    qθ      = min.(1, max.(0, κ./J))             # job-filling rate
    θ       = (qθ.^(-ι) .- 1).^(1/ι).*(qθ .!= 0) # implied tightness

    return (a = a, W = W, J = J, Y = Y, θ = θ, Z = v0)
end

"""
Return d log theta/d log z at the steady state μ_z.
"""
function dlogtheta(modd; N_z = 21)

    # vary initial z1
    modd2 = model(N_z = N_z, σ_η = modd.σ_η, χ = modd.χ, γ = modd.γ, hbar = modd.hbar, ε = modd.ε,
                    ρ = modd.ρ, σ_ϵ = modd.σ_ϵ, ι = modd.ι)

    @unpack θ, zgrid = vary_z1(modd2)

    # compute d log theta/d log z
    dlogθ  = slopeFD(θ, zgrid; diff = "forward").*zgrid[1:end-1]./θ[1:end-1]

    return  dlogθ[modd2.z_ss_idx]
end

"""
Simulate employment, given θ(z_t) path
"""
function simulate_employment(modd, T_sim, burnin, θ; minz_idx = 1, u0 = 0.069, seed = 512)

    @unpack s, f, zgrid, P_z = modd

    # Get sequence of productivity shocks
    shocks  = drawZShocks(P_z, zgrid, N = 1, T = T_sim + burnin, set_seed = true, seed = seed)
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

"""
Compute the optimal effort, given z_t and w_0, 
without the additional checks implemented in DynamicModel.
"""
function effort(z::T, w_0::T,  ψ, ε, hp, σ_η, hbar) where T<:AbstractFloat 
    #return find_zero(x-> x - max(eps(), ((z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2)/hbar))^(ε/(1+ε)), 1.0)
    return solve(ZeroProblem( x-> x - max(eps(), ((z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2)/hbar))^(ε/(1+ε)), 1.0))
end

"""
Compute da/dz_0 and the components of da/dz_0
"""
function dadz(z::T, w_0::T, modd) where T<:AbstractFloat 
    
    @unpack ψ, ε, hp, σ_η, hbar, hp = modd

    a      = effort(z, w_0,  ψ, ε, hp, σ_η, hbar)
    Ω      = (ε/(1+ε))*hbar^(-ε/(1+ε))*(max(eps(), z*a/w_0 - (ψ/ε)*(hp(a)*σ_η)^2))^(-1/(1+ε))
    denom  = (1 - Ω*(z/w_0 - (2*ψ/ε)*hp(a)*(σ_η^2)*(hbar/ε)*a^(1/ε - 1)) )

    da_dz  =  Ω*a/(w_0*denom)
    da_dw0 = -Ω*a*z/((w_0^2)*denom)

    A      = z*Ω*a/(w_0*denom)
    Da_z   = A*z
    Da_w   = A*z/w_0

    return (da_dz = da_dz, da_dw0 = da_dw0, Da_z = Da_z, Da_w = Da_w)
end

"""
Compute relelevant objects from our profit and wage decompositions 
in the paper. modd = model object, bonus = model solves for each initial z_0.
"""
function decomposition(modd, bonus; fix_a = false)

    @unpack P_z, zgrid, N_z, ρ, β, s, z_ss_idx, q, ψ, hp, κ, ι = modd

    # Solve for dJ/dz when C term = 0 (direct effect), conditional on initial z_1
    JJ_EVT   = zeros(N_z) 
    Threads.@threads for iz = 1:N_z

        # Initialize guess of direct effect
        v0     = zgrid./(1-ρ*β*(1-s))
        v0_new = zeros(N_z)
        iter   = 1
        err    = 10
        
        # solve via simple value function iteration
        @inbounds while err > 10^-10 && iter < 1000
            v0_new = bonus.az[:,iz].*zgrid + ρ*β*(1-s)*P_z*v0            
            err    = maximum(abs.(v0_new - v0))
            v0     = copy(v0_new)
            iter +=1
        end

        ## Bargained vs Incentive Wage Flexibility
        JJ_EVT[iz]   = v0[iz]/zgrid[iz]
    end

    # total wage flexibility
    WC              = slopeFD(bonus.W, zgrid)

    # compute the residual 
    qq(x)           = -(x^(-1 + ι))*(1 + x^ι)^(-1 - 1/ι) # q'(θ)
    total_resid     = -(κ./(q.(bonus.θ)).^2).*qq.(bonus.θ).*slopeFD(bonus.θ, zgrid) # d kappa/q(θ(z_0)) / d z_0

    if fix_a == true
        
        return (JJ_EVT = JJ_EVT, WF = WF, BWC = zeros(N_z), IWF = zeros(N_z), resid = zeros(N_z), total_resid = total_resid) 

    elseif fix_a == false

        # Solve for IWC
        IWC          = zeros(N_z) 
        dw0_dz1      = slopeFD(bonus.w_0, zgrid)
        dw0_dz1[1]   = slopeFD(bonus.w_0, zgrid; diff = "forward")[1]
        dw0_dz1[end] = slopeFD(bonus.w_0, zgrid; diff = "backward")[end]

        Threads.@threads for iz = 1:N_z

            w_0   = bonus.w_0[iz]
            Da_dz = [dadz(z, w_0, modd).Da_z for z in zgrid]
            Da_dw = [dadz(z, w_0, modd).Da_w for z in zgrid]

            # Initialize first term (effect of z_1 on z_t -> a)
            da_dz     = zeros(N_z)
            da_dz_new = zeros(N_z)
            iter   = 1
            err    = 10
            
            # solve via simple value function iteration
            @inbounds while err > 10^-10 && iter < 1000
                da_dz_new = Da_dz + ρ*β*(1-s)*P_z*da_dz
                err       = maximum(abs.(da_dz_new - da_dz))
                da_dz     = copy(da_dz_new)
                iter +=1
            end

            # Initialize second term (effect of z_1 on w_0 -> a)
            da_dw     = zeros(N_z)
            da_dw_new = zeros(N_z)
            iter      = 1
            err       = 10
            
            # solve via simple value function iteration
            @inbounds while err > 10^-10 && iter < 1000
                da_dw_new = Da_dw + β*(1-s)*P_z*da_dw
                err       = maximum(abs.(da_dw_new - da_dw))
                da_dw     = copy(da_dw_new)
                iter +=1
            end

            IWC[iz]      =  da_dz[iz]/zgrid[iz] - da_dw[iz]*dw0_dz1[iz] 
        end

        # Solve for the BWF
        BWC             = WC - IWC
        resid           = BWC - JJ_EVT  # partial kappa/q(θ(z_0)) / partial z_0

        return (JJ_EVT = JJ_EVT, WC = WC, BWC = BWC, IWC = IWC, resid = resid, total_resid = total_resid) 
    end

end

"""
Simulate moments for heatmaps
"""
function heatmap_moments(; σ_η = 0.406231, χ = 0.578895, γ = 0.562862, hbar = 1.0, ε = 0.3)

    baseline     = model(σ_η = σ_η, hbar = hbar, ε = ε, γ = γ,  χ = χ) 
    out          = simulate(baseline, shocks)
    dlogθ_dlogz  = dlogtheta(baseline)

    mod_mom  = [out.std_Δlw, out.dlw1_du, out.dlw_dly, out.u_ss, dlogθ_dlogz]  
    
    # Flags
    flag     = out.flag
    flag_IR  = out.flag_IR
    IR_err   = out.IR_err

    return [mod_mom, flag, flag_IR, IR_err]
end