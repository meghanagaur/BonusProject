"""
Objective function to be minimized during SMM -- WITHOUT MANUAL BOUNDS.
Variable descriptions below.
xx           = evaluate objFunction @ these parameters 
shocks       = shocks for the simulation
param_vals   = baseline parameters 
param_est    = parameters we are estimating
data_mom     = data moments
W            = weight matrix for SMM
    ORDERING OF PARAMETERS
ε            = 1st param
σ_η          = 2nd param
χ            = 3rd param
γ            = 4th param
ρ            = 5th param
σ_ϵ          = 6th param
    ORDERING OF MOMENTS
std_Δlw      = 1st moment (st dev of wage growth)
dlw1_du      = 2nd moment (dlog w_1 / d u)
dlw_dly      = 3rd moment (dlw/dly)
u_ss         = 4th moment (SS unemployment rate)
y_ρ         = 5th moment (persistence of output)
y_σ         = 6th moment (std of output)
"""
function objFunction(xx, param_vals, param_est, shocks, data_mom, W; smm = true,
                     fix_a = false, fix_wages = false, pv = false, est_z = false, output_target = "alp")

    # Get the relevant parameters
    Params =  OrderedDict{Symbol, Float64}()
    for (k, v) in param_vals
        if haskey(param_est, k)
            Params[k]  = xx[param_est[k]]
        else
            Params[k]  = v
        end
    end

    @unpack σ_η, χ, γ, hbar, ε, ρ, σ_ϵ, ι = Params
    baseline = model(σ_η = σ_η, χ = χ, γ = γ, hbar = hbar, ε = ε, ρ = ρ, σ_ϵ = σ_ϵ, ι = ι) 

    # Simulate the model and compute moments
    if fix_a == true 

            out = simulateFixedEffort(baseline, shocks; smm = smm, fix_wages = fix_wages, pv = pv, 
                    a = Params[:a], est_z = est_z, output_target = output_target)

    elseif fix_a == false

        if est_z == false
            out        = simulate(baseline, shocks; smm = smm)
        
        elseif est_z == true
            out        = simulateEstZ(baseline, shock; output_target = output_target)
        end

    end

    # Get moments
    @unpack std_Δlw, dlw1_du, dlw_dly, u_ss, y_ρ, y_σ, flag, flag_IR, IR_err = out

    mod_mom    = [std_Δlw, dlw1_du, dlw_dly, u_ss, y_ρ, y_σ]

    # Compute objective function
    d          = (mod_mom - data_mom)./abs.(data_mom) #0.5(abs.(mod_mom) + abs.(data_mom)) # arc % differences
    f          = (d'*W*d)*(1-flag)*(1-flag_IR) + (1-flag_IR)*flag*10.0^8 + flag_IR*(100.0 + IR_err)

    # Extra checks for NaN
    flag       = isnan(f) ? 1 : flag
    f          = isnan(f) ? 10.0^8 : f

    return [f, mod_mom, flag, flag_IR, IR_err]
end

"""
Objective function to be minimized during SMM -- WITH MANUAL BOUNDS.
Variable descriptions below.
xx           = evaluate objFunction @ parameters = xx for optimization (before transformation)
x0           = actual starting point for the local optimization (after transformation)
shocks       = shocks for the simulation
param_vals   = baseline parameters 
param_est    = parameters we are estimating
param_bounds = parameter bounds
data_mom     = data moments
W            = weight matrix for SMM
    ORDERING OF PARAMETERS
ε            = 1st param
σ_η          = 2nd param
χ            = 3rd param
γ            = 4th param
ρ            = 5th param
σ_ϵ          = 6th param
    ORDERING OF MOMENTS
std_Δlw      = 1st moment (st dev of wage growth)
dlw1_du      = 2nd moment (dlog w_1 / d u)
dlw_dly      = 3rd moment (dlw/dly)
u_ss         = 4th moment (SS unemployment rate)
y_ρ          = 5th moment (persistence of output)
y_σ          = 6th moment (std of output)
"""
function objFunction_WB(xx, x0, param_bounds, param_vals, param_est, shocks, data_mom, W; 
            smm = true, fix_a = false, fix_wages = false, pv = false, est_z = false, output_target = "alp")

    Params =  OrderedDict{Symbol, Float64}()
    for (k, v) in param_vals
        if haskey(param_est, k)
            idx        = param_est[k]  
            Params[k]  = transform_params(xx[idx], param_bounds[k], x0[idx])
        else
            Params[k]  = v
        end
    end

    @unpack σ_η, χ, γ, hbar, ε, ρ, σ_ϵ, ι = Params
    baseline   = model(σ_η = σ_η, χ = χ, γ = γ, hbar = hbar, ε = ε, ρ = ρ, σ_ϵ = σ_ϵ, ι = ι) 

    # Simulate the model and compute moments
    if fix_a == true 

        out = simulateFixedEffort(baseline, shocks; smm = smm, fix_wages = fix_wages, pv = pv, 
                a = Params[:a], est_z = est_z, output_target = output_target)

    elseif fix_a == false

        if est_z == false
            out        = simulate(baseline, shocks; smm = smm)
        
        elseif est_z == true
            out        = simulateEstZ(baseline, shock; output_target = output_target)
        end

    end

    # Get moments
    @unpack std_Δlw, dlw1_du, dlw_dly, u_ss, y_ρ, y_σ, flag, flag_IR, IR_err = out
    mod_mom    = [std_Δlw, dlw1_du, dlw_dly, u_ss, y_ρ, y_σ]

    # Compute objective function
    d          = (mod_mom - data_mom)./abs.(data_mom) #0.5(abs.(mod_mom) + abs.(data_mom)) # arc % differences
    f          = (d'*W*d)*(1-flag)*(1-flag_IR) + (1-flag_IR)*flag*10.0^8 + flag_IR*(100.0 + IR_err)

    # Extra checks for NaN
    flag       = isnan(f) ? 1 : flag
    f          = isnan(f) ? 10.0^8 : f

    return [f, mod_mom, flag, flag_IR, IR_err]
end

"""
Logit transformation to transform x to [min, max].
"""
function logit(x; x0 = 0, min = -1, max = 1, λ = 1.0)
   return (max - min)/(1 + exp(-(x - x0)/λ)) + min
end

""" 
Transform parameters to lie within their specified bounds in pb.
xx = current (logit transformed) position
x1 = current (actual) position
p0 = actual initial position
"""
function transform_params(xx, pb, p0; λ = 1)
    
    # Rescales ALL of the parameters to lie between -1 and 1 
    xx2 =   logit.(xx; λ = λ) 

    # Transform each parameter, so that the boundrary conditions are satisfied 
    if xx2 > 0
        x1 = xx2*(pb[2] - p0) + p0
    else
        x1 = xx2*(p0 - pb[1]) + p0  
    end

    #= Could localize the search even further
    δ  = min(pb[2] - p0, p0 - pb[1])
    x1 = xx2*δ + p0=#

    return x1
end

#= Define a new simplexer for NM without explicit bound constraints
"""
Draw random points in the parameter space
"""
function draw_params(pb)
    pars = zeros(K)
    for i = 1:K
        pars[i] = rand(Uniform(param_bounds[i][1], param_bounds[i][2]))
    end
    return pars
end

struct RandSimplexer <: Optim.Simplexer end
function Optim.simplexer(S::RandSimplexer, initial_x::AbstractArray{T, N}) where {T, N}
    initial_simplex = Array{T, N}[initial_x for i = 1:K+1]
    for k = 2:K+1
        initial_simplex[k] .= draw_params(param_bounds) 
    end
    initial_simplex
end
=#