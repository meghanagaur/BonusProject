#= Solve the finite horizon dynamic EGSS model with TIOLI offers. 
Monthly frequency, no savings. =#
"""
Set up the dynamic EGSS model:

matching tech: m(u,v)   = (uv)/(u^ι + v^⟦)^(1/ι)
LoM for prod:  log(z_t) = (1 - ρ)μ_z + ρ*log(z_t - 1 ) + ϵ_t, where ϵ_t ∼ N(0, σ_ϵ^2),
production:    y_t      = z_t(a_t + η_t), where η ∼ N(0, σ_η^2).
unemp benefit: ξ(z) = γ*z^χ

β    = discount factor
s    = exogenous separation rate
ι    = controls elasticity of matching function 
κ    = vacancy-posting cost
χ    = elasticity of unemp benefit to z 
γ    = level of unemp benefit 
σ_η  = st dev of η distribution
ε    = disutility of effort (Frisch elasticity)
T    = number of periods for contract horizon
""" 
function model_FH(; β = 0.99^(1/3), s = 0.031, κ = 0.45, ε = 2.713, σ_η = 0.532, ι = 0.9,
    ρ =  0.966, χ = 0.467, γ = 0.461, T = 240)

    # Basic parameterization
    q(θ)    = (1 + θ^ι)^(-1/ι)                          # job-filling rate
    f(θ)    = (1 + θ^-ι)^(-1/ι)                         # job-finding rate
    u(c)    = log(max(c, eps()))                        # utility from consumption                
    h(a)    = (max(a, 0)^(1 + 1/ε))/(1 + 1/ε)           # disutility from effort  
    hp(a)   = max(a, 0)^(1/ε)                           # h'(a)

    # pass-through parameters
    ψ      = zeros(T)
    temp = reverse(1 ./[mapreduce(t-> (β*(1-s))^t, +, [0:xx;]) for xx = 0:T-1])
    for t = 1:T
        ψ[t] = temp[t]
    end
    
    return (β = β, s = s, κ = κ, ε = ε, σ_η = σ_η, ι = ι, q = q, ρ = ρ,
            f = f, ψ = ψ, h = h, hp = hp, χ = χ, γ = γ, T = T)
end

"""
Solve for the optimal effort, given z, ψ, ε, hp, σ_η, and w_0
"""
function optA_FH(z, ψ, ε, hp, σ_η, w_0; check_mult = false, a_min = 10^-6, a_max = 100.0, hbar = 1.0)
        
    if ε == 1 # can solve analytically for positive root
        a      = (z/w_0)/(1 + ψ*σ_η^2)
        a_flag = 0
    else 

        # solve for the positive root. nudge to avoid any runtime errors.
        if check_mult == false 
            aa          = solve(ZeroProblem( x -> (x > a_min)*(x - max( (z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2)/hbar, eps() )^(ε/(1+ε))) + (x <= a_min)*10^10, 1.0))
        elseif check_mult == true
            aa          = find_zeros( x -> (x > a_min)*(x - max( (z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2)/hbar, eps() )^(ε/(1+ε))) + (x <= a_min)*10^10,  a_min, a_max)
        end

        if ~isempty(aa) 
            if (maximum(isnan.(aa)) == 0 )
                a      = aa[1] 
                a_flag = max(a < a_min , max( ((z*a/w_0 - (ψ/ε)*(hp(a)*σ_η)^2) < 0), (length(aa) > 1) ) )
            else
                a       = 0.0
                a_flag  = 1
            end
        elseif isempty(aa) 
            a           = 0.0
            a_flag      = 1
        end
    end

    y      = a*z # Expectation of y_t = z_t*(a_t+ η_t) over η_t 
    return a, y, a_flag
end

"""
Solve the finite horizon non-stochastic EGSS model using a bisection search on θ.
lz_t = string of log z_t realizations (0 = SS)
ω    = worker's PV from unemployment (infinite horizon)
"""
function solveModel_FH(modd, lz_t; max_iter1 = 50, max_iter2 = 1000, a_min = 10^-6,
                    tol1 = 10^-8, tol2 = 10^(-10), noisy = true, q_lb_0 =  0.0, 
                    q_ub_0 = 1.0, check_mult = false)
    
    @unpack T, β, s, ρ, κ, ι, ε, σ_η, q, h, hp, ψ, γ, χ = modd

    # PV of unemp at time t = PDV utility from consuming unemployment benefit forever 
    if (length(lz_t) < T+1) error("need length of z_t sequence to be at least T+1") end 
    lz1     = lz_t[1]
    ω       = [log(γ)/(1-β) + (χ*lz1*ρ^(t-1))/(1 - ρ*β) for t = 1:T+1]
    
    # exponetiate log z_t to get z_t
    z_t     = exp.(lz_t)
 
    # set tolerance parameters for outermost loop
    err1    = 10
    iter1   = 1
    # initialize tolerance parameters for inner loops (for export)
    err2    = 10
    iter2   = 1
    IR_err  = 10
    flag_IR = 0  

    # Initialize default values and search parameters
    q_lb    = q_lb_0           # lower search bound
    q_ub    = q_ub_0           # upper search bound
    q_1     = (q_lb + q_ub)/2  # initial guess for θ
    α       = 0.25             # dampening parameter
    Y_0     = 0                # initalize Y_0 for export
    w_0     = 0                # initialize initial wage constant for export
    U       = 0                # initalize worker's EU from contract for export

    #  effort and output given z path 
    az      = zeros(T)    # T                         
    yz      = zeros(T)    # T                        
    a_flag  = zeros(T)    # T                                           

    # Look for a fixed point in θ_0
    @inbounds while err1 > tol1 && iter1 < max_iter1
       
        if noisy 
            println("iter:\t"*string(iter1))
            println("error:\t"*string(err1))
            println("q_1:\t"*string(q_1))
        end

        # Look for a fixed point in Y_0
        err2   = 10
        iter2  = 1      
        Y_lb   = κ/q_1              # lower search bound
        Y_ub   = 100*κ/q_1          # upper search bound
        Y_0    = (Y_lb + Y_ub)/2    # initial guess for Y
        
        @inbounds while err2 > tol2 && iter2 < max_iter2

            w_0  = ψ[1]*(Y_0 - κ/q_1) # constant for wage difference equation

            # Solve for optimal effort a_t and implied (expected) y_t
            @inbounds for t = 1:T
                az[t], yz[t], a_flag[t] = optA_FH(z_t[t], ψ[t], ε, hp, σ_η, w_0; check_mult = check_mult, a_min = a_min) 
            end

            # compute EPDV of output
            Y_1  = mapreduce(t -> yz[t]*(β*(1-s))^(t-1), +, 1:T)  
            err2 = abs(Y_0 - Y_1)       

            #= if doing bisection search on Y_0 
            if Y_1 < Y_0 
                Y_ub  = copy(Y_0)
            elseif Y_1 > Y_0 || w0 < 0
                Y_lb  = copy(Y_0)
            end
            Y_0  = 0.5(Y_lb + Y_ub) 
            
            # Note: delivers ≈ Y_0, but converges more slowly. =#
            # increase dampening parameter if not converging
            α_1    = iter2 > 50 ? 0.75 : α 
            Y_0    = (1 - α_1)*Y_0 +  α_1*Y_1
            iter2 += 1

            if (noisy) println("errror 2: ", string(err2)) end
        end

        # Solve for the EPDV offered by the contract (over η realizations)

        # compute LHS of IR constraint
        U  = 0
        uc = log(max(eps(), w_0))
        @views @inbounds for t = 1:T
            # utility from consumption
            uc   -= 0.5*(ψ[t]*hp(az[t])*σ_η)^2 
            # continuation value upon separation
            cv    = (t == T) ? ω[t+1]*β : ω[t+1]*β*s          
            # worker's flow utility at time t         
            U += (uc - h(az[t]) + cv)*(β*(1-s))^(t-1) 
        end

        # Check the IR constraint (must bind with TIOLI)
        IR_err = U - ω[1]                                  

        # Update θ accordingly: U decreasing in θ (increasing in q)
        if IR_err < 0             # increase q (decrease θ)
            q_lb  = copy(q_1)
        elseif IR_err > 0         # decrease q (increase θ)
            q_ub  = copy(q_1)
        end

        # Bisection
        q_new   = (q_lb + q_ub)/2                     # update q
        #err1   = min(abs(IR_err), abs(q_1 - q_0))   # compute convergence criterion
        err1    = abs(IR_err)

        # Record info on TIOLI/IR constraint violations 
        flag_IR = (err1 > tol1)

        # Export the accurate iter & q value
        if err1 > tol1
            # stuck in a corner (0 or 1), so break
            if abs(q_ub - q_lb)  < tol1/2
                break
            else
                q_1     = q_new
                iter1  += 1
            end
        end

    end

    return (θ = (q_1^(-ι) - 1)^(1/ι), Y = Y_0, U = U, ω = ω[1], w_0 = w_0, mod = modd, IR_err = err1*flag_IR, flag_IR = flag_IR,
    az = az, yz = yz, err1 = err1, err2 = err2, iter1 = iter1, iter2 = iter2, wage_flag = (w_0 <= 0),
    effort_flag = maximum(a_flag), conv_flag1 = (iter1 > max_iter1), conv_flag2 = (iter2 > max_iter2))
end

"""
Solve the finite horizon non-stochastic EGSS model using a bisection search on θ.
lz_t = string of log z_t realizations (0 = SS)
ω    = worker's PV from unemployment (infinite horizon)
"""
function solveModelFixedEffort_FH(modd, lz_t; max_iter1 = 50, a = 1.0,
                    tol1 = 10^-8, noisy = true, q_lb_0 =  0.0, q_ub_0 = 1.0)
    
    @unpack T, β, s, ρ, κ, ι, ε, σ_η, q, h, hp, ψ, γ, χ = modd

    # PV of unemp at time t = PDV utility from consuming unemployment benefit forever 
    if (length(lz_t) < T+1) error("need length of z_t sequence to be at least T+1") end 
    lz1     = lz_t[1]
    ω       = [log(γ)/(1-β) + (χ*lz1*ρ^(t-1))/(1 - ρ*β) for t = 1:T+1]
    
    # exponentiate log z_t to get z_t
    z_t     = exp.(lz_t)

    # set tolerance parameters for outermost loop
    err1    = 10
    iter1   = 1
    IR_err  = 10
    flag_IR = 0  

    # Initialize default values and search parameters
    q_lb    = q_lb_0           # lower search bound
    q_ub    = q_ub_0           # upper search bound
    q_1     = (q_lb + q_ub)/2  # initial guess for θ
    w_0     = 0                # initialize initial wage constant for export
    U       = 0                # initalize worker's EU from contract for export

    #  effort and output given z path 
    az      = a.*ones(T)                           
    yz      = z_t                                 
    Y_0     = mapreduce(t -> yz[t] *(β*(1-s))^(t-1), +, 1:T)  

    # Look for a fixed point in θ_0
    @inbounds while err1 > tol1 && iter1 < max_iter1
       
        if noisy 
            println("iter:\t"*string(iter1))
            println("error:\t"*string(err1))
            println("q_1:\t"*string(q_1))
        end

        # Solve for the EPDV offered by the contract (from free entry condition)
        w_0  = ψ[1]*(Y_0 - κ/q_1) 

        # compute LHS of IR constraint
        U  = 0
        uc = log(max(eps(), w_0))
        @views @inbounds for t = 1:T
            # continuation value upon separation
            cv    = (t == T) ? ω[t+1]*β : ω[t+1]*β*s          
            # worker's flow utility at time t         
            U += (uc - h(az[t]) + cv)*(β*(1-s))^(t-1) 
        end

        # Check the IR constraint (must bind with TIOLI)
        IR_err = U - ω[1]                                  

        # Update θ accordingly: U decreasing in θ (increasing in q)
        if IR_err < 0             # increase q (decrease θ)
            q_lb  = copy(q_1)
        elseif IR_err > 0         # decrease q (increase θ)
            q_ub  = copy(q_1)
        end

        # Bisection
        q_new   = (q_lb + q_ub)/2                     # update q
        #err1   = min(abs(IR_err), abs(q_1 - q_0))   # compute convergence criterion
        err1    = abs(IR_err)

        # Record info on TIOLI/IR constraint violations 
        flag_IR = (err1 > tol1)

        # Export the accurate iter & q value
        if err1 > tol1
            # stuck in a corner (0 or 1), so break
            if abs(q_ub - q_lb)  < tol1/2
                break
            else
                q_1     = q_new
                iter1  += 1
            end
        end

    end

    return (θ = (q_1^(-ι) - 1)^(1/ι), Y = Y_0, U = U, ω = ω[1], w_0 = w_0, mod = modd, IR_err = err1*flag_IR, flag_IR = flag_IR,
    az = az, yz = yz, err1 = err1, iter1 = iter1, wage_flag = (w_0 <= 0), effort_flag = false, conv_flag1 = (iter1 > max_iter1))
end

"""
Compute the nonlinear perfect foresight IRFs given log z_t path 
Let u_0 be the unemployment rate at time 1.
T   = contract horizon (in months)
N   = length of transition path
lz1 = initial z_t (0 = SS)
u1  = initial unemployment rate (state variable)
"""
function IRFs(; ρ = 0.966, T = 12*20, N = 200, lz1 = 0.0, u1 = 0.06, ι = 0.9,
                ε = 2.713, σ_η = 0.532, χ = 0.467, γ = 0.461, fix_a = false) 
    
    TT      = N + T                      # length of full z_t path
    lz_t    = [lz1*ρ^(t-1) for t = 1:TT] # lz_t path
    θ_t     = zeros(N)          
    u_t     = zeros(N)
    u_t[1]  = u1 
    modd    = model_FH(; ρ =  ρ, ε = ε, σ_η = σ_η, χ = χ, γ = γ, T = T, ι = ι) 

    # solve to get θ_t (jump variable so can be done asychronously)
    @Threads.threads for n = 1:N
        if fix_a == false
            θ_t[n]   = solveModel_FH(modd, lz_t[n:n+240]; noisy = false).θ
        else
            θ_t[n]   = solveModelFixedEffort_FH(modd, lz_t[n:n+240]; noisy = false).θ
        end
    end 

    # now iterate forwards to solve for u_t
    @unpack s, f = modd 
    @inbounds for n = 1:N-1
        u_t[n+1] = u_t[n] + s*(1 - u_t[n]) - f(θ_t[n])*u_t[n]*(1 - s)
    end

    # check u_ss 
    u_ss = s/(s + f(θ_t[end])*(1-s))

    return (θ_t = θ_t, u_t = u_t, lz_t = lz_t, u_ss = u_ss)
end




