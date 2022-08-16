using DynamicModel, BenchmarkTools, DataStructures, Distributions, 
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase

include("simulation.jl") # simulation functions

"""
Return objective function.
Endogenous params = params to be estimated.
Shocks            = z and η shocks.
data_mom          = targeted moments
W                 = weight matrix 
"""
function objFunction(endogParams, shocks, data_mom, W)
    # Unpack the shocks 
    # Simulate the model and compute moments
    out     = simulate(endogParams, shocks)
    mod_mom = [out.σ_Δw, out.dw1_du]
    d       = mod_mom - data_mom
    f       = maximum([out.c_flag; out.a_flag]) < 1 ? d'*W*d : Inf
    println(string(d'*W*d))
    return f, mod_mom, out.c_flag, out.θ
end

# test parameters / initial guess
endogParams    = zeros(4)
endogParams[1] = 0.5                    # ε
endogParams[2] = 0.05                   # σ_η
endogParams[3] = 0.3                    # χ
endogParams[4] = 0.68                   # γ

shocks         = simulateShocks(model())
W              = Matrix(1.0I, 2, 2) # to do: set to inverse of variance covariance matrix
data_mom       =[10^-3, -0.5]

test = objFunction(endogParams, shock2, data_mom, W)