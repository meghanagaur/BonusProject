#= 
Main file to build required functions and random vectors for the estimation.
Note: J = number of moments, K = number of parameters.
=#

# Load necessary packages
#using Pkg; Pkg.add(url="https://github.com/meghanagaur/DynamicModel")
using DynamicModel, BenchmarkTools, DataStructures, Distributions, Optim, Sobol,
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase, JLD2

# Load necessary helper functions for simulation
include("simulation.jl")   

"""
Objective function to be minimized during SMM -- WITHOUT BOUNDS.
Variable descriptions below.
xx           = evaluate objFunction @ these parameters 
zshocks      = z shocks for the simulation
pb           = parameter bounds
data_mom     = data moments
W            = weight matrix for SMM
    ORDERING OF PARAMETERS
ε            = 1st param
σ_η          = 2nd param
χ            = 3rd param
γ            = 4th param
hbar         = 5th param
    ORDERING OF MOMENTS
std_Δlw      = 1st moment (st dev of log wage changes)
E[Δlw]       = 2nd moment (avg log wage change)
dlw1_du      = 3rd moment (dlog w_1 / d u)
dly_dΔlw     = 4th moment (d log y_it / d Δ log w_it )
u_ss         = 5th moment (SS unemployment rate)
"""
function objFunction(xx, pb, shocks, data_mom, W)

    inbounds = minimum( [ pb[i][1] <= xx[i] <= pb[i][2] for i = 1:J]) >= 1

    if inbounds == 0
        f        = 10^5
        mod_mom  = ones(K)*NaN
        flag     = 1
    elseif inbounds == 1
        baseline = model(ε = xx[1] , σ_η = xx[2], χ = xx[3], γ = xx[4], hbar = xx[5]) 
        # Simulate the model and compute moments
        out      = simulate(baseline, shocks)
        flag     = out.flag
        mod_mom  = [out.std_Δlw, out.avg_Δlw, out.dlw1_du, out.dlw_dly, out.u_ss]
        d        = (mod_mom - data_mom) #./abs.(data_mom) #0.5(abs.(mod_mom) + abs.(data_mom)) # arc % change
        f        = flag < 1 ? d'*W*d : 10^5

        # add extra flags
        flag     = isnan(f) ? 1 : flag
        f        = isnan(f) ? 10^5 : f

    end

    return [f, mod_mom, flag]
end

"""
Objective function to be minimized during SMM -- WITH BOUNDS.
Variable descriptions below.
xx           = evaluate objFunction @ parameters = xx (before transformation)
x0           = actual starting point for the local optimization (after transformation)
zshocks      = z shocks for the simulation
pb           = parameter bounds
data_mom     = data moments
W            = weight matrix for SMM
    ORDERING OF PARAMETERS
ε            = 1st param
σ_η          = 2nd param
χ            = 3rd param
γ            = 4th param
hbar         = 5th param
    ORDERING OF MOMENTS
std_Δlw      = 1st moment (st dev of log wage changes)
E[Δlw]       = 2nd moment (avg log wage change)
dlw1_du      = 3rd moment (dlog w_1 / d u)
dly_dΔlw     = 4th moment (d log y_it / d Δ log w_it )
u_ss         = 5th moment (SS unemployment rate)
"""
function objFunction_WB(xx, x0, pb, shocks, data_mom, W)

    endogParams  = [ transform_params(xx[i], pb[i], x0[i]) for i = 1:J] 
    baseline     = model(ε = endogParams[1] , σ_η = endogParams[2], χ = endogParams[3], γ = endogParams[4], hbar = endogParams[5]) 

    # Simulate the model and compute moments
    out     = simulate(baseline, shocks)
    flag    = out.flag
    mod_mom = [out.std_Δlw, out.avg_Δlw, out.dlw1_du, out.dlw_dly, out.u_ss]
    d       = (mod_mom - data_mom) #./abs.(data_mom) #0.5(abs.(mod_mom) + abs.(data_mom)) # arc % differences
    f       = flag < 1 ? d'*W*d : 10^5

    # add extra flags
    flag     = isnan(f) ? 1 : flag
    f        = isnan(f) ? 10^5 : f

    return [f, mod_mom, flag]
end

"""
Logit transformation to transform x to [min,max].
"""
function logit(x; x0 = 0, min = -1, max = 1, λ = 1.0)
    (max - min)/(1 + exp(-(x - x0)/λ)) + min
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

## Empirical moments that we are targeting
moms_key             = OrderedDict{Int, Symbol}([   # parameter bounds
                        (1, :std_Δlw),
                        (2, :avg_Δlw),
                        (3, :dlw1_du),
                        (4, :dy_dΔlw),
                        (5, :u_ss) ]) 
data_mom             = [0.016, 0.01, -0.5, 0.15, 0.06]    # annual -> quarterly stdev 0.064/4
const K              = length(data_mom)

## Parameter bounds and weight matrix
W                    = Matrix(1.0I, K, K) # inverse of covariance matrix of data_mom?

#Parameters to be estimated
param_key            = OrderedDict{Int, Symbol}([
                        (1, :ε),
                        (2, :σ_η),
                        (3, :χ),
                        (4, :γ),
                        (5, :hbar)])
const J              = length(param_key)
param_bounds         = OrderedDict{Int,Array{Real,1}}([ # parameter bounds
                        (1, [0.15,  2.0]),      # ε
                        (2, [0.001, 0.5]),      # σ_η 
                        (3, [-1, 1]),           # χ
                        (4, [0.3, 0.9]),        # γ
                        (5, [0.15, 2]) ])       # hbar

#= Corrction for the χ, γ bounds (enforce log b(z) < log z for all z)
param_bounds[3] = [ max(1 - log(param_bounds[4][2])/log(model().zgrid[1]), param_bounds[3][1]) ,
min(1 - log(param_bounds[4][2])/log(model().zgrid[end]), param_bounds[3][2])] =#

## Build shocks for the simulation
@unpack N_z, P_z, zgrid = model()
N_sim                   = 30000
T_sim                   = 80         
burnin                  = 1000

# Compute the invariant distribution of logz
A           = P_z - Matrix(1.0I, N_z, N_z)
A[:,end]   .= 1
O           = zeros(1,N_z)
O[end]      = 1
z_ss_dist   = (O*inv(A))
@assert(isapprox(sum(z_ss_dist),1))

# Create z and η shocks
λ_N_z        = floor.(Int64, N_sim*z_ss_dist)
indices      = cumsum(λ_N_z, dims = 2)*T_sim
indices_q    = cumsum(λ_N_z, dims = 2)*(T_sim-3) # indices for quarterly log wage changes
z_shocks     = OrderedDict{Int, Array{Real,2}}()
z_shocks_idx = OrderedDict{Int, Array{Real,2}}()
η_shocks     = OrderedDict{Int, Array{Real,2}}()
Threads.@threads for iz = 1:length(zgrid)
    temp                = simulateZShocks(P_z, zgrid, N = λ_N_z[iz], T = T_sim, z_1_idx = iz, set_seed = true)
    z_shocks[iz]        = temp.z_shocks
    z_shocks_idx[iz]    = temp.z_shocks_idx
    η_shocks[iz]        = rand(Normal(0, 1), size(z_shocks[iz])) # N x T 
end

# Create one long z_t string: set z_1 to default value of 1.
zstring  = simulateZShocks(P_z, zgrid, N = 1, T = N_sim + burnin, set_seed = false)

# Create an ordered tuple that contains the zshocks
shocks   = (η_shocks = η_shocks, z_shocks = z_shocks, z_shocks_idx = z_shocks_idx, indices = indices, indices_q = indices_q,
    λ_N_z = λ_N_z, N_sim = N_sim, T_sim = T_sim, zstring = zstring, burnin = burnin, z_ss_dist = z_ss_dist)

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