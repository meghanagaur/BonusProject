"""
Vary z_1, and compute relevant aggregates.
B denotes Bonus model.
"""
function vary_z1(modd; check_mult = false)

    @unpack ψ, zgrid, z_ss, N_z = modd
    modds      = OrderedDict{Int64, Any}()
    z_ss_idx   = Int64(median(1:N_z))

    # Solve the model for different z_0
    @time Threads.@threads for iz = 1:N_z
        modds[iz] =  solveModel(modd; z_1 = zgrid[iz], noisy = false, check_mult = check_mult)
    end

    ## Store series of interest
    w_0    = [modds[i].w_0 for i = 1:N_z]      # w0 (constant)
    θ_1    = [modds[i].θ for i = 1:N_z]        # tightness
    W_1    = w_0/ψ                             # EPV of wages
    Y_1    = [modds[i].Y for i = 1:N_z]        # EPV of output
    ω_1    = [modds[i].ω_0 for i = 1:N_z]      # EPV value of unemployment at z0
    J_1    = Y_1 - W_1                         # EPV profits
    a_1    = [modds[i].az[i] for i = 1:N_z]    # optimal effort @ start of contract
    aflag  = [modds[i].effort_flag for i = 1:N_z] 

    return (w_0_B = w_0, θ_B = θ_1, W_B = W_1, Y_B = Y_1, ω_B = ω_1, J_B = J_1, 
            a_B = a_1, z_ss_idx = z_ss_idx, zgrid = zgrid, aflag = aflag)
end

"""
Produce Hall economy aggregates, matching SS Y and W from Bonus economy.
"""
function solveHall(modd, z_ss_idx, Y_B, W_B)

    @unpack zgrid, κ, ι, s, β, N_z, P_z = modd
    
    # Solve for expected PV of sum of the z_t's
    exp_z = zeros(length(zgrid)) 
    @inbounds for (iz, z1) in enumerate(zgrid)

        z0_idx  = findfirst(isequal(z1), zgrid)  # index of z0 on zgrid
        
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

        exp_z[iz]   = v0[z0_idx]

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
Approximate slope of y(x) using forward differences,
where y and x are both vectors.
"""
function slope(y, x)
    return (y[2:end] - y[1:end-1])./(x[2:end] - x[1:end-1])
end

"""
Return dlogtheta at the steady state μ_z.
"""
function dlogtheta(modd; N_z = 21)

    # vary z0 
    modd2 = model(N_z = N_z, σ_η = modd.σ_η, χ = modd.χ, γ = modd.γ, hbar = modd.hbar, ε = modd.ε)
    @unpack θ_B, z_ss_idx, zgrid = vary_z1(modd2)

    # compute dtheta/dz
    dlogθ  = slope(θ_B, zgrid).*zgrid[1:end-1]./θ_B[1:end-1]

    return  dlogθ[z_ss_idx]

end
 
"""
Simulate moments for heatmaps
"""
function heatmap_moments(; σ_η = 0.3, hbar = 1.0, ε = 0.3, γ = 0.5, χ = 0.0)

    baseline = model(σ_η = σ_η, hbar = hbar, ε = ε, γ = γ,  χ = χ) 
    out      = simulate(baseline, shocks)
    dlogθ    = dlogtheta(baseline)

    mod_mom  = [out.std_Δlw, out.dlw1_du, out.dlw_dly, out.u_ss, dlogθ] #, out.avg_Δlw,
    #out.dlw1_dlz, out.dlY_dlz, out.dlu_dlz, out.std_u, out.std_z, out.std_Y, out.std_w0]

    # Flags
    flag     = out.flag
    flag_IR  = out.flag_IR
    IR_err   = out.IR_err

    # Extra moments (check)
    dlw_dly_2  = out.dlw_dly_2
    u_ss_2     = out.u_ss_2

    return [mod_mom, flag, flag_IR, IR_err]
end