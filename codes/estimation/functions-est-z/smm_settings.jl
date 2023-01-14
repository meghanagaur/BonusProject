"""
Specifications for the SMM:

J = number of moments, K = number of parameters.

How to install necessary packages:
using Pkg; Pkg.add(url="https://github.com/meghanagaur/DynamicModel")
"""

## Requred pacakges
using DynamicModel, BenchmarkTools, DataStructures, Distributions, Optim, Sobol, SparseArrays,
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase, JLD2

## Required functions
include("utils.jl")          
include("obj_func.jl")       # objective functions
include("simulation.jl")     # simulation functions

## Empirical moments that we are targeting
function moment_targets(; std_Δlw = 0.064, dlw1_du = -1.0, dlw_dly = 0.039, u_ss = 0.068, alp_ρ = 0.89, alp_σ = 0.017)
    
    mom_key       = OrderedDict{Symbol, Real}([   # parameter bounds
                            (:std_Δlw, std_Δlw),
                            (:dlw1_du, dlw1_du),
                            (:dlw_dly, dlw_dly),
                            (:u_ss, u_ss),
                            (:alp_ρ, alp_ρ), 
                            (:alp_σ, alp_σ) ]) 

    data_mom      = [std_Δlw, dlw1_du, dlw_dly, u_ss, alp_ρ, alp_σ]   

    return data_mom, mom_key
end

## Identity weight matrix
function getW(K)
    return Matrix(1.0I, K, K) 
end

# Bounds on parameters to be estimated
param_bounds  = OrderedDict{Symbol, Array{Real,1}}([ # parameter bounds
                        (:ε,  [0.1,   3.0]),        # ε
                        (:σ_η, [0.001, 0.5]),       # σ_η 
                        (:χ, [0.0, 1.0]),           # χ
                        (:γ, [0.05, 0.9]),          # γ
                        (:hbar, [0.1, 4.0]),        # hbar
                        (:ρ, [0.85, 0.999]),        # ρ
                        (:σ_ϵ, [0.0005, 0.01]) ])   # σ_ϵ
                 
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
N_sim_micro             = 10^5
T_sim_micro             = 13         
N_sim_macro             = 5*10^3 
N_sim_macro_workers     = 1000
T_sim_macro             = 828    # for aggregate sequence: 69 years 
burnin                  = 5000   # for aggregate sequence
N_sim_macro_est_alp     = 1000   # for estimating productivity process
shocks                  = rand_shocks(N_sim_micro, T_sim_micro, N_sim_macro, N_sim_macro_workers, 
                            T_sim_macro, burnin, N_sim_macro_est_alp)