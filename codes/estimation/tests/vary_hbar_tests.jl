cd(dirname(@__FILE__))

include("functions/smm_settings.jl")        # SMM inputs, settings, packages, etc.
include("functions/calibration_vary_z1.jl") # vary z1 functions

using DelimitedFiles, LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =13, yguidefontsize=13, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 9)

## Logistics
file_str     = "fix_chi0"
file_pre     = "runs/jld/pretesting_"*file_str*".jld2"  # pretesting data location
file_est     = "runs/jld/estimation_"*file_str*".txt"   # estimation output location
file_save    = "figs/vary-z1/"*file_str*"/"             # file to-save 
mkpath(file_save)

# Load output
est_output = readdlm(file_est, ',', Float64)            # open output across all jobs
@unpack moms, fvals, pars, mom_key, param_bounds, param_est, param_vals, data_mom, J, W = load(file_pre) 

# Get the final minimum 
idx        = argmin(est_output[:,1])                    # check for the lowest function value across processes 
pstar      = est_output[idx, 2:(2+J-1)]                 # get parameters 

# Get the relevant parameters
Params =  OrderedDict{Symbol, Float64}()
for (k, v) in param_vals
    if haskey(param_est, k)
        Params[k]  = pstar[param_est[k]]
    else
        Params[k]  = v
    end
end

## Vary z1, hbar experiments
function vary_hbar(hbar_val, Params)
   
    # Get the Bonus model aggregates
    @unpack σ_η, χ, γ, hbar, ε = Params
    modd  = model(N_z = 21, χ = χ, γ = γ, hbar = hbar_val, ε = ε, σ_η = σ_η)
    @unpack w_0_B, θ_B, W_B, Y_B, ω_B, J_B, a_B, z_ss_idx, zgrid = vary_z1(modd)
    a_H, W_H, J_H, Y_H, θ_H = solveHall(modd, z_ss_idx, Y_B, W_B);

    JJ_B   = slope(J_B, zgrid)
    JJ_H   = slope(J_H, zgrid)

    return JJ_B[z_ss_idx] - JJ_H[z_ss_idx]
end

hgrid  = 0.5:0.1:2.0
c_term = [vary_hbar(h, Params) for h in hgrid]
plot(hgrid, c_term)


@unpack σ_η, χ, γ, hbar, ε = Params
modd  = model(N_z = 11, χ = χ, γ = γ, hbar = 1.2, ε = ε, σ_η = σ_η)
sol1 = solveModel(modd; check_mult=false)
sol2 = solveModel(modd; check_mult=false)

sol1.w_0-sol2.w_0
sol1.Y-sol2.Y
sol1.az-sol2.az

#a=1.0
#plot(h->(h*a^(1 + 1/ε))/(1+ε),0.1,2.0)
#plot!(h->(h*a^(1/ε)),0.1,2.0)
w_0 = sol1.w_0
@unpack hbar, ψ, ε, hp = modd

z = zgrid[z_ss_idx]
a_min = 10^-12
aa1         = solve(ZeroProblem( x -> (x > a_min)*(x - max( (z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2)/1, eps() )^(ε/(1+ε))) + (x <= a_min)*10^10, 1.0))
aa2         = solve(ZeroProblem( x -> (x > a_min)*(x - max( (z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2)/hbar, eps() )^(ε/(1+ε))) + (x <= a_min)*10^10, 1.0))

maximum(abs.(sol1.az - sol2.az))

function optA2(z, modd, w_0; a_min = 10^-12, a_max = 100.0, check_mult = false)
   
    @unpack ψ, ε, q, κ, hp, σ_η = modd
    
    if ε == 1 # can solve analytically for positive root
        a      = (z/w_0)/(1 + ψ*σ_η^2)
        a_flag = 0
    else 

        # solve for the positive root. nudge to avoid any runtime errors.
        if check_mult == false 
            aa          = solve(ZeroProblem( x -> (x > a_min)*(x - max( (z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2)/hbar, eps() )^(ε/(1+ε))) + (x <= a_min)*10^10, 1.0))

            #aa         = fzero(x -> (x > a_min)*(x - max( z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2, eps() )^(ε/(1+ε))) + (x <= a_min)*10^10, 1.0)
            #aa         = find_zero(x -> x - max(z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2, 0)^(ε/(1+ε)), (a_min, a_max)) # requires bracketing
        
        elseif check_mult == true
            #aa          = find_zeros(x -> x - max(z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2, 0)^(ε/(1+ε)), a_min, a_max)  
            aa          = find_zeros(x -> x - max( (z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2)/hbar, 0)^(ε/(1+ε)), a_min, a_max)  
        end

        if ~isempty(aa) & (maximum(isnan.(aa)) == 0 )

            a      = aa[1] 
            a_flag = max( ((z*a/w_0 - (ψ/ε)*(hp(a)*σ_η)^2) < 0), (length(aa) > 1) ) 
        
        elseif isempty(aa) || (maximum(isnan.(aa))==1)
            a       = 0
            a_flag  = 1
        end
    end

    y      = a*z # Expectation of y_t = z_t*(a_t+ η_t) over η_t (given z_t)

    return a, y, a_flag

end

function optA1(z, modd, w_0; a_min = 10^-12, a_max = 100.0, check_mult = false)
   
    @unpack ψ, ε, q, κ, hp, σ_η = modd
    
    if ε == 1 # can solve analytically for positive root
        a      = (z/w_0)/(1 + ψ*σ_η^2)
        a_flag = 0
    else 

        # solve for the positive root. nudge to avoid any runtime errors.
        if check_mult == false 
            aa          = solve(ZeroProblem( x -> (x > a_min)*(x - max( z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2, eps() )^(ε/(1+ε))) + (x <= a_min)*10^10, 1.0))

            #aa         = fzero(x -> (x > a_min)*(x - max( z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2, eps() )^(ε/(1+ε))) + (x <= a_min)*10^10, 1.0)
            #aa         = find_zero(x -> x - max(z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2, 0)^(ε/(1+ε)), (a_min, a_max)) # requires bracketing
        
        elseif check_mult == true
            #aa          = find_zeros(x -> x - max(z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2, 0)^(ε/(1+ε)), a_min, a_max)  
            aa          = find_zeros(x -> x - max( (z*x/w_0 - (ψ/ε)*(hp(x)*σ_η)^2)/hbar, 0)^(ε/(1+ε)), a_min, a_max)  
        end

        if ~isempty(aa) & (maximum(isnan.(aa)) == 0 )

            a      = aa[1] 
            a_flag = max( ((z*a/w_0 - (ψ/ε)*(hp(a)*σ_η)^2) < 0), (length(aa) > 1) ) 
        
        elseif isempty(aa) || (maximum(isnan.(aa))==1)
            a       = 0
            a_flag  = 1
        end
    end

    y      = a*z # Expectation of y_t = z_t*(a_t+ η_t) over η_t (given z_t)

    return a, y, a_flag

end

optA1(z, modd, w_0)
optA2(z, modd, w_0)