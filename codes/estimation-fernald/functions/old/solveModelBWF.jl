
function BWF(modd)

    @unpack β, s, κ, ι, ε, σ_η, ω, logz, q, u, h, hp, zgrid, P_z, ψ, procyclical, z_ss_idx, N_z, μ_z = modd  

    ω_0 = procyclical ? modd.ω[modd.z_ss_idx] : ω
    δ   = 10^-4 
    dW  = (solveModelE(modd, ω_0 + δ).w_0 - solveModelE(modd, ω_0  - δ).w_0)/(2ψ*δ)
    
    # Compute the direct effect on the IR constraint
    B    = (χ/(1-β*ρ))
    A    = (log(γ) + β*B*(1-ρ)*μ_z)/(1-β)
    ω_2  = A .+ B*logz
    dω   = B./zgrid
    BWF2  = dW*dω[z_ss_idx]

    # now perturb z, but holding fixed value of unemployment
    z_lb = zgrid[z_ss_idx-1]
    z_ub = zgrid[z_ss_idx+1]
    θ    = solveModelE(modd, ω_0).θ
    IWF2  = (solveModelE(modd; q_0 = q(θ), z_1 = z_ub, max_iter1 =1).w_0 -
                solveModelE(modd; q_0 = q(θ), z_1 = z_lb, max_iter1 =1).w_0)/(ψ*(z_ub - z_lb))
    
    IWF2  = (solveModelE(modd; ω_0 = ω_0, z_1 = z_ub).w_0 -
             solveModelE(modd; ω_0 = ω_0, z_1 = z_lb).w_0)/(ψ*(z_ub - z_lb))

end


"""
Solve the infinite horizon EGSS model using a bisection search on θ.
"""
function solveModelE(modd; ω_0 = nothing, q_0 = nothing, z_1 = nothing, max_iter1 = 50, max_iter2 = 1000, max_iter3 = 1000, a_min = 10^-6,
    tol1 = 10^-8, tol2 = 10^-8, tol3 =  10^-8, noisy = true, q_lb_0 =  0.0, q_ub_0 = 1.0, check_mult = false)

    @unpack β, s, κ, ι, ε, σ_η, ω, N_z, q, u, h, hp, zgrid, P_z, ψ, procyclical, N_z, z_ss_idx = modd  
    
    # find index of z_1 on the productivity grid 
    if isnothing(z_1)
        z_1_idx = z_ss_idx
    else
        z_1_idx = findfirst(isapprox(z_1, atol = 1e-6), zgrid)  
    end

    # set tolerance parameters for outermost loop
    err1    = 10
    iter1   = 1
    # initialize tolerance parameters for inner loops (for export)
    err2    = 10
    iter2   = 1
    err3    = 10
    iter3   = 1
    IR_err  = 10
    flag_IR = 0

    # Initialize default values and search parameters
    ω_0    = procyclical ? ω[z_1_idx] : ω # unemployment value at z_0
    ω_vec  = procyclical ?  ω : ω*ones(N_z)
    q_lb   = q_lb_0          # lower search bound for q
    q_ub   = q_ub_0          # upper search bound for q
    q_0    = isnothing(q_0) ? (q_lb + q_ub)/2  : q_0 # initial guess for q
    α      = 0               # dampening parameter
    Y_0    = 0               # initalize Y for export
    U      = 0               # initalize worker's EU from contract for export
    w_0    = 0               # initialize initial wage constant for export

    # Initialize series
    az     = zeros(N_z)   # a(z|z_1)                         
    yz     = zeros(N_z)   # y(z|z_1)                         
    a_flag = zeros(N_z)   # flag for a(z|z_1)                         

    # Look for a fixed point in θ_0
    @inbounds while err1 > tol1 && iter1 <= max_iter1  

        if noisy 
            println(q_0)
        end

        # Look for a fixed point in Y(z | z_1), ∀ z
        err2   = 10
        iter2  = 1      
        Y_0    = ones(N_z)*(50*κ/q_0)   # initial guess for Y(z | z_1)
        
        @inbounds while err2 > tol2 && iter2 <= max_iter2   
           
            w_0  = ψ*(Y_0[z_1_idx] - κ/q_0) # constant for wage difference equation
           
            # Solve for optimal effort a(z | z_1)
            @inbounds for (iz,z) in enumerate(zgrid)
                az[iz], yz[iz], a_flag[iz] = optA(z, modd, w_0; check_mult = check_mult, a_min = a_min)
            end
            Y_1    = yz + β*(1-s)*P_z*Y_0    
            err2   = maximum(abs.(Y_0 - Y_1))  # Error       
            if (err2 > tol2) 
                iter2 += 1
                if (iter2 < max_iter2) 
                    Y_0    = α*Y_0 + (1 - α)*Y_1 
                end
            end
            #println(Y_0[z_1_idx])
        end

        # Solve recursively for the PV utility from the contract
        err3  = 10
        iter3 = 1  
        W_0   = copy(ω_vec) # initial guess
        flow  = -(1/(2ψ))*(ψ*hp.(az)*σ_η).^2 - h.(az) + β*s*(P_z*ω_vec)

        @inbounds while err3 > tol3 && iter3 <= max_iter3
            W_1  = flow + β*(1-s)*(P_z*W_0)
            err3 = maximum(abs.(W_1 - W_0))
            if (err3 > tol3) 
                iter3 += 1
                if (iter3 < max_iter3) 
                    W_0  = α*W_0 + (1 - α)*W_1
                end
            end
            #println(W_0[z_1_idx])
        end
        
        # Check the IR constraint (must bind)
        U      = (1/ψ)*log(max(eps(), w_0)) + W_0[z_1_idx] # nudge w_0 to avoid runtime error
        
        # Upate θ accordingly: note U is decreasing in θ (=> increasing in q)
        if U < ω_0              # increase q (decrease θ)
            q_lb  = copy(q_0)
        elseif U > ω_0          # decrease q (increase θ)
            q_ub  = copy(q_0)
        end

        # Bisection
        IR_err = U - ω_0                             # check whether IR constraint holds
        q_1    = (q_lb + q_ub)/2                     # update q
        #err1   = min(abs(IR_err), abs(q_1 - q_0))   # compute convergence criterion
        err1    = abs(IR_err)

        # Record info on IR constraint
        flag_IR = (IR_err < 0)*(abs(IR_err) > tol1)

        # Export the accurate iter & q value
        if err1 > tol1
            if min(abs(q_1 - q_ub_0), abs(q_1 - q_lb_0))  < 10^(-10) 
                break
            else
                q_0     = α*q_0 + (1 - α)*q_1
                iter1  += 1
            end
        end

    end

    return (θ = (q_0^(-ι) - 1)^(1/ι), Y = Y_0[z_1_idx], U = U, ω = ω_0, w_0 = w_0, mod = modd, IR_err = IR_err*flag_IR, flag_IR = flag_IR,
    az = az, yz = yz, err1 = err1, err2 = err2, err3 = err3, iter1 = iter1, iter2 = iter2, iter3 = iter3, wage_flag = (w_0 <= 0),
    effort_flag = maximum(a_flag), conv_flag1 = (iter1 > max_iter1), conv_flag2 = (iter2 > max_iter2), conv_flag3 = (iter3 > max_iter3))
end
