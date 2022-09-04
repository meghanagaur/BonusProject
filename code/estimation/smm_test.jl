#using Pkg; Pkg.add(url="https://github.com/meghanagaur/DynamicModel")
using DynamicModel, BenchmarkTools, DataStructures, Distributions, Plots, Optim,
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase, JLD2

include("simulation.jl") # simulation functions
include("smm_settings.jl") # simulation functions

"""
Objective function to be minimized during SMM. 
Variable descriptions below.
xx           = evaluate objFunction @ these parameters (before transformation)
endogParams2 = evaluate objFunction @ these parameters (after transformation)
endogParams  = actual starting point for local optimization
zshocks      = z shocks for the simulation
pb           = parameter bounds
data_mom     = data moments
W            = weight matrix for SMM
    ORDERING OF PARAMETERS/MOMENTS
ε            = 1st param
σ_η          = 2nd param
χ            = 3rd param
γ            = 4th param
var_Δlw      = 1st moment (variance of log wage changes)
dlw1_du      = 2nd moment (dlog w_1 / d u)
dΔlw_dy      = 3rd moment (d Δ log w_it / y_it)
w_y          = 4th moment (PV of labor share)
"""
function objFunction(xx, x0, pb, zshocks, data_mom, W)

    endogParams  = [ transform(xx[i], pb[i], x0[i]) for i = 1:length(x0)] 
    baseline     = model(ε = endogParams[1] , σ_η = endogParams[2], χ = endogParams[3]) #, γ = endogParams[4]) 

    # Simulate the model and compute moments
    out     = simulate(baseline, zshocks)
    mod_mom = [out.var_Δlw, out.dlw1_du, out.dΔlw_dy] #, out.w_y]
    d       = (mod_mom - data_mom)./2(mod_mom + data_mom) # arc percentage differences
    f       = out.flag < 1 ? d'*W*d : 10000
    println(string(f))
    return f, mod_mom, out.flag
end

## evaluate the objective function 
init_x = zeros(J)
@time fval, mod_mom, flag = objFunction(init_x, endogParams, pb, zshocks, data_mom, W)

## test local optimization by setting the truth to the originally obtained model moments
endogParams2 = endogParams + rand(Normal(0, 0.005), J)               # add some noise
endogParams2 = [clamp(endogParams2[i], pb[i][1],pb[i][2]) for i=1:J] # make sure new initial guess lies within the bounds

# NM from Optim WITH bounds 
objFunc(x) = objFunction(x, endogParams2, pb, zshocks, mod_mom, W)[1]
opt        = optimize(objFunc, init_x, NelderMead(), 
                    Optim.Options(g_tol = 1e-6, x_tol = 1e-6,  f_tol = 1e-6, iterations = 50, show_trace = true))

# rescales all of the parameters 
minimizer_t   = Optim.minimizer(opt)  # transformed
minimizer     = [ transform(minimizer_t[i], pb[i], endogParams2[i]) for i = 1:length(endogParams2) ] 
#orig          = [ transform(init_x[i], pb[i], endogParams2[i]) for i = 1:length(endogParams2) ] 

# save the results
save("smm_test.jld2", Dict("min" =>  Optim.minimum(opt), "argmin" =>  minimizer,
                        "initial_x" =>   endogParams2,
                        "truth" => endogParams, "opt" => opt))