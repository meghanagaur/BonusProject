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
include("utils.jl")                         # basic utility functions  
include("obj_func.jl")                      # objective functions
include("simulation.jl")                    # simulation functions
include("simulation_est_z.jl")              # simulate the model, when internally calibrating z
include("bargaining.jl")                    # solve and simulate the model with fixed effort
include("tik-tak.jl")                       # tik-tak code 

"""
Empirical moments that we are targeting;
Use the identity weight matrix.
K = number of targeted moments.
"""
function moment_targets(; std_Δlw = 0.064, dlw1_du = -1.000, dlw_dly = 0.039, u_ss = 0.06, 
                          p_ρ = 0.886, p_σ = 0.0167, y_ρ = 0.938, y_σ = 0.0244, dlw_dlp = 0.449, 
                          est_mom = Dict(:std_Δlw => true, :dlw1_du => true, :dlw_dly => true, :u_ss => true))
    
    # ordering of the moments
    mom_key       = OrderedDict{Symbol, Int64}([   
                            (:std_Δlw, 1),
                            (:dlw1_du, 2),
                            (:dlw_dly, 3),
                            (:u_ss,    4),
                            (:p_ρ,     5), 
                            (:p_σ,     6),
                            (:y_ρ,     7), 
                            (:y_σ,     8),
                            (:dlw_dlp, 9) ]) 

    # vector of moments
    data_mom      = [std_Δlw, dlw1_du, dlw_dly, u_ss, p_ρ, p_σ, y_ρ, y_σ, dlw_dlp]   
    W             = zeros(length(data_mom), length(data_mom)) 
    
    # fill out weight matrix
    for (k,v) in mom_key
        if haskey(est_mom, k)
            W[v,v] = 1.0
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
                        (:σ_η, [0.3, 0.7]),         # σ_η 
                        (:χ, [0.1, 0.9]),           # χ  
                        (:γ, [0.2, 0.8]),           # γ
                        (:ρ, [0.9, 0.999]),         # ρ
                        (:σ_ϵ, [0.0001, 0.01]) ])   # σ_ϵ
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
