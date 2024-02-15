"""
Specifications for the SMM:

K = number of moments, J = number of parameters.

How to install necessary packages:
using Pkg; Pkg.add(url="https://github.com/meghanagaur/DynamicModel")
"""

## Requred pacakges
using DynamicModel, BenchmarkTools, DataStructures, Distributions, Optim, Sobol, SparseArrays,
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase, JLD2, NLopt, Combinatorics

## Required functions
include("utils.jl")                     # basic utility functions  
include("obj_func_ALP.jl")              # objective functions
include("simulation_ALP.jl")            # simulation functions
include("bargaining_ALP.jl")            # solve the model with fixed effort
include("tik-tak.jl")                   # tik-tak code 

"""
Empirical moments that we are targeting;
Use the identity weight matrix.
K = number of targeted moments.
"""
function moment_targets(; std_Δlw = 0.064, dlw1_du = -1.000, dlw_dly = 0.039, u_ss = 0.06, alp_ρ = 0.89, alp_σ = 0.017, drop_mom = nothing)
    
    # ordering of the moments
    mom_key       = OrderedDict{Symbol, Int64}([   
                            (:std_Δlw, 1),
                            (:dlw1_du, 2),
                            (:dlw_dly, 3),
                            (:u_ss, 4),
                            (:alp_ρ, 5), 
                            (:alp_σ, 6) ]) 

    # vector of moments
    data_mom      = [std_Δlw, dlw1_du, dlw_dly, u_ss, alp_ρ, alp_σ]   
    W             = Matrix(1.0I, length(data_mom), length(data_mom)) 
    
    if !isnothing(drop_mom)
        for (k,v) in mom_key
            if haskey(drop_mom, k)
                W[v,v] = 0
            end
        end
    end

    K  = Int64(sum(diag(W)))

    return (data_mom = data_mom, mom_key = mom_key, K = K, W = W)
end

"""
Bounds on parameters to be estimated
"""
function get_param_bounds(; ε_lb = 0.3, ε_ub = 3.0)

    # parameter bounds
    return OrderedDict{Symbol, Array{Real,1}}([ 
                        (:ε,  [ε_lb, ε_ub]),        # ε
                        (:σ_η, [0.2, 0.6]),         # σ_η 
                        (:χ, [0.0, 1.0]),           # χ
                        (:γ, [0.2, 0.8]),           # γ
                        (:ρ, [0.93, 0.995]),        # ρ
                        (:σ_ϵ, [0.0005, 0.01]) ])   # σ_ϵ
end

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