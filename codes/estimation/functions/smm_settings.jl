#= Specifications for the SMM:

J = number of moments, K = number of parameters.

How to install necessary packages:
using Pkg; Pkg.add(url="https://github.com/meghanagaur/DynamicModel")

=#

## Requred pacakges
using DynamicModel, BenchmarkTools, DataStructures, Distributions, Optim, Sobol,
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase, JLD2

## Required functions
include("obj_func.jl")       # objective functions
include("simulation.jl")     # simulation functions

## Empirical moments that we are targeting
function moment_targets(; std_Δlw = 0.064, dlw1_du = -1.0, dlw_dly = 0.05, u_ss = 0.03/(0.03+0.42) )
    
    mom_key       = OrderedDict{Symbol, Real}([   # parameter bounds
                            (:std_Δlw, std_Δlw),
                            (:dlw1_du, dlw1_du),
                            (:dlw_dly, dlw_dly),
                            (:u_ss, u_ss) ]) 

    data_mom      = [std_Δlw, dlw1_du, dlw_dly, u_ss]   

    return data_mom, mom_key
end

## Identity weight matrix
function getW(K)
    return Matrix(1.0I, K, K) 
end

# Bounds on parameters to be estimated
param_bounds  = OrderedDict{Symbol, Array{Real,1}}([ # parameter bounds
                        (:ε,  [0.1,   3.0]),     # ε
                        (:σ_η, [0.001, 0.5]),    # σ_η 
                        (:χ, [0.0, 1.0]),        # χ
                        (:γ, [0.05, 0.9]),       # γ
                        (:hbar, [0.1, 4.0]) ])   # hbar

"""
Get lower, upper bounds given parameters
to be estimated.
"""
function get_bounds(param_est, param_bounds)

    lb = zeros(length(param_est))
    ub = zeros(length(param_est))

   for (k,v) in param_est
        lb[v] = param_bounds[k][1]
        ub[v] = param_bounds[k][2]
   end

   return lb, ub

end

## Specifciations for the shocks in simulation
@unpack N_z, P_z, zgrid = model()
N_sim                   = 50000
T_sim                   = 120         
burnin                  = 10000
shocks                  = build_shocks(N_z, P_z, zgrid, N_sim, T_sim, burnin)