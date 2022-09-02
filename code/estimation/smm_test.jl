#using Pkg; Pkg.add(url="https://github.com/meghanagaur/DynamicModel")
using DynamicModel, BenchmarkTools, DataStructures, Distributions, Plots, Optim,
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase, JLD2

include("simulation.jl") # simulation functions
include("smm_settings.jl") # simulation functions

# evaluate the objective function 
init_x = 100*ones(J)
@time fval, mod_mom, flag = objFunction(init_x, init_x, endogParams, pb, zshocks, data_mom, W)

## test local optimization by setting the truth to the originally obtained model moments
endogParams2 = endogParams + rand(Normal(0, 0.005), J)               # add some noise
endogParams2 = [clamp(endogParams2[i], pb[i][1],pb[i][2]) for i=1:J] # make sure new initial guess lies within the bounds

# NM from Optim, adding bounds 
objFunc(x) = objFunction(x, init_x, endogParams2, pb, zshocks, mod_mom, W)[1]
opt        = optimize(objFunc, init_x, NelderMead(), 
                    Optim.Options(g_tol = 1e-3, x_tol = 1e-5,  f_tol = 1e-5, iterations = 50,
                    show_trace = true))

# rescales all of the parameters 
minimizer_t   = Optim.minimizer(opt)  # transformed
minimizer     = [ transform(minimizer_t[i], pb[i], init_x[i], endogParams2[i]) for i = 1:length(endogParams2) ] 

# save the results
save("example.jld2", Dict("min" =>  Optim.minimum(opt), "argmin" =>  minimizer,
                        initial_x  =>   endogParams2,
                        "truth" => endogParams, "opt" => opt))