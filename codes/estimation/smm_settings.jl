#using Pkg; Pkg.add(url="https://github.com/meghanagaur/DynamicModel")
using DynamicModel, BenchmarkTools, DataStructures, Distributions, Optim, Sobol,
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase, JLD2

include("simulation.jl")   # simulation functions

"""
Objective function to be minimized during SMM -- WITHOUT BOUNDS.
Variable descriptions below.
xx           = evaluate objFunction @ these parameters 
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
function objFunction(xx, pb, zshocks, data_mom, W)
    inbounds = minimum( [ pb[i][1] <= xx[i] <= pb[i][2] for i = 1:length(xx) ]) >= 1
    if inbounds == 0
        f        = 10000
        mod_mom  = NaN
        flag     = 1
    elseif inbounds == 1
        baseline = model(ε = xx[1] , σ_η = xx[2], χ = xx[3]) #, γ = xx[4]) 
        # Simulate the model and compute moments
        out      = simulate(baseline, zshocks)
        mod_mom  = [out.var_Δlw, out.dlw1_du, out.dΔlw_dy] #, out.w_y]
        d        = (mod_mom - data_mom)./0.5(abs.(mod_mom) + abs.(data_mom)) # arc % change
        f        = out.flag < 1 ? d'*W*d : 10000
    end
    return [f, vec(mod_mom), out.flag]
end

"""
Objective function to be minimized during SMM -- WITH BOUNDS.
Variable descriptions below.
xx           = evaluate objFunction @ these parameters (before transformation)
x0           = actual starting point for local optimization (after transformation)
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
function objFunction_WB(xx, x0, pb, zshocks, data_mom, W)
    endogParams  = [ transform(xx[i], pb[i], x0[i]) for i = 1:length(x0)] 
    baseline     = model(ε = endogParams[1] , σ_η = endogParams[2], χ = endogParams[3]) #, γ = endogParams[4]) 

    # Simulate the model and compute moments
    out     = simulate(baseline, zshocks)
    mod_mom = [out.var_Δlw, out.dlw1_du, out.dΔlw_dy] #, out.w_y]
    d       = (mod_mom - data_mom)./0.5(abs.(mod_mom) + abs.(data_mom)) # arc % differences
    f       = out.flag < 1 ? d'*W*d : 10000
    return [f, mod_mom, out.flag]
end

"""
Logit transformation to transform x to desired bounds.
"""
function logit(x; x0 = 0, min =-1, max=1, λ = 1.0)
    (max - min)/(1 + exp(-(x - x0)/λ)) + min
end

""" 
Transform parameters to lie within specified bounds in pb.
xx = current (transformed) position
x1 = current (actual) position
p0 = actual initial position
"""
function transform(xx, pb, p0; λ = 1)
    # Rescales ALL of the parameters to lie between -1 and 1 
    xx2          =   logit.(xx) 

    # Transform each parameter, so that the boundrary conditions are satisfied 
    if xx2 > 0
        x1 = xx2*(pb[2] - p0) + p0
    else
        x1 = xx2*(p0 - pb[1]) + p0  
    end

    # Can also force localize the search even further
    #δ  = min(pb[2] - p0, p0 - pb[1])
    #x1 = xx2*δ + p0

    return x1
end

## Empirical moments that we are targeting
data_mom             = [0.53^2, -0.5, .05] ##, 0.6]  # may need to update 
moms_key             = OrderedDict{Int, Symbol}([    # parameter bounds
                        (1, :var_Δlw),
                        (2, :dlw1_du),
                        (3, :dΔlw_dy) ])
const K              = length(data_mom)

## Parameter bounds and weight matrix
const W              = Matrix(1.0I, K, K) # inverse of covariance matrix of data_mom?

#Parameters we are estimating
params_key           = OrderedDict{Int, Symbol}([ # parameter bounds
                        (1, :ε),
                        (2, :σ_η),
                        (3, :χ) ])
const J              = length(params_key)
pb                   = OrderedDict{Int,Array{Real,1}}([ # parameter bounds
                        (1, [0  , 1.0]),
                        (2, [0.0, 0.36]),
                        (3, [-1, 1]),
                        (4, [0.3, 0.9]) ])

## Build zshocks for the simulation
baseline     = model()
const N_z    = baseline.N_z
const P_z    = baseline.P_z
const zgrid  = baseline.zgrid
const N      = 100000
const T      = 100
const burnin = 1000

# Compute the invariant distribution of z
A           = P_z - Matrix(1.0I, N_z, N_z)
A[:,end]   .= 1
O           = zeros(1,N_z)
O[end]      = 1
z_ss_dist   = (O*inv(A))
@assert(isapprox(sum(z_ss_dist),1))

# Create z shocks
λ_N_z        = floor.(Int64, N*z_ss_dist)
z_shocks     = OrderedDict{Int, Array{Real,1}}()
z_shocks_idx = OrderedDict{Int, Array{Real,1}}()

Threads.@threads for iz = 1:length(zgrid)
    temp                = simulateZShocks(baseline, N = λ_N_z[iz], T = T, z_1_idx = iz, set_seed = true)
    z_shocks[iz]        = vec(temp.z_shocks)
    z_shocks_idx[iz]    = vec(temp.z_shocks_idx)
end

# Create one long z_t string: set z_1 to default value of 1.
zstring  = simulateZShocks(baseline, N = 1, T = N + burnin, set_seed = false)

# Create an ordered tuple for the zshocks
zshocks = (z_shocks = z_shocks, z_shocks_idx = z_shocks_idx, λ_N_z = λ_N_z, N = N,
T = T, zstring = zstring, burnin = burnin, z_ss_dist = z_ss_dist)

