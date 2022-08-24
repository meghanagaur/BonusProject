using DynamicModel, BenchmarkTools, DataStructures, Distributions, Plots,
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase

include("simulation.jl") # simulation functions

"""
The bjective function. Variable descripiton below.
endogParams  = params to be estimated
zshocks      = z shocks for simulation
pb           = parameter bounds
data_mom     = data moments
W            = weight matrix for SMM
    ORDERING OF PARAMETERS/MOMENTS
ε            = 1st param
σ_η          = 2nd param
χ            = 3rd param
γ            = 4th param
σ_Δw         = 1st moment 
dw1_du       = 2nd moment
"""
function objFunction(endogParams, pb, zshocks, data_mom, W)
    baseline  = model(ε = endogParams[1] , σ_η = endogParams[2], χ = endogParams[3]) 
    # Check boundrary conditions
    inbounds  = sum([pb[i][1] <= endogParams[i] <= pb[i][2] for i = 1:length(endogParams)])
    if inbounds < length(endogParams)
        f       = Inf
    else
        # Simulate the model and compute moments
        out     = simulate(baseline, zshocks)
        mod_mom = [out.var_Δlw, out.dlw1_du, out.dΔlw_dy]
        d       = (mod_mom - data_mom)./2(mod_mom + data_mom) # arc percentage differences
        f       = out.flag < 1 ? d'*W*d : Inf
        println(string(d'*W*d))
        return f, mod_mom, out.flag, out.θ
    end
end

# test parameters / initial guess (for now, do not calibrate γ)
endogParams    = zeros(3)
endogParams[1] = 0.5      # ε
endogParams[2] = 0.05     # σ_η
endogParams[3] = 0.3      # χ
zshocks        = simulateZShocks(model()) # do  OUTSIDE of estimation 
data_mom       =[0.53^2, -0.5, .05] # may need to update 
J              = length(data_mom)
W              = Matrix(1.0I, J, J) # inverse of covariance matrix of data_mom?
pb             = OrderedDict{Int,Array{Real,1}}([ # parameter bounds
                (1, [0.1, 1.0]),
                (2, [0.0, 0.1]),
                (3, [0.0, 0.5])])

@time f, mod_mom, flag, θ = objFunction(endogParams, pb, zshocks, data_mom, W)


# check to make sure we fall within bounds for all χ
zgrid = model().zgrid

# min χ
mod1 = solveModel(model(z0 = log(minimum(zgrid)), χ = 0.0))
mod2 = solveModel(model(z0 = log(maximum(zgrid)), χ = 0.0))

# max χ
mod3 = solveModel(model(z0 = log(minimum(zgrid)), χ = 0.5))
mod4 = solveModel(model(z0 = log(maximum(zgrid)), χ = 0.5))

# median χ
mod5 = solveModel(model(z0 = log(minimum(zgrid)), χ = 0.3))
mod6 = solveModel(model(z0 = log(maximum(zgrid)), χ = 0.3))



