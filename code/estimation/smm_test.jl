#using Pkg; Pkg.add(url="https://github.com/meghanagaur/DynamicModel")
using DynamicModel, BenchmarkTools, DataStructures, Distributions, Plots, Optim,
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase, JLD2

include("simulation.jl") # simulation functions
include("smm_settings.jl") # simulation functions


endogParams
# evaluate the objective function 
init_x = 100*ones(J)
@time fval, mod_mom, flag = objFunction(zeros(J), init_x, endogParams, pb, zshocks, data_mom, W)

## test local optimization by setting the truth to the originally obtained model moments
x0 = endogParams + rand(Normal(0, 0.005), J) # add some noise
x0 = [clamp(x0[i], pb[i][1],pb[i][2]) for i=1:J] # make sure it lies inside the bounds

# NM from Optim, adding bounds 
objFunc(x) = objFunction(x, init_x, x0, pb, zshocks, mod_mom, W)[1]
opt              = optimize(objFunc, init_x, NelderMead(), 
                    Optim.Options(g_tol = 1e-5, x_tol = 1e-6,  f_tol = 1e-6, iterations = 50))

# save the results
save("example.jld2", Dict("min" =>  Optim.minimum(opt), "argmin" =>  Optim.minimizer(opt),
                        "truth" => endogParams, "opt" => opt))