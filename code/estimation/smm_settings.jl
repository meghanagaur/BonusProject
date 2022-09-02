"""
Objective function to be minimized during SMM. 
Variable descriptions below.
endogParams  = evaluate objFunction @ these endog. parameters
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
function objFunction(xx, x0, endogParams, pb, zshocks, data_mom, W)
    
    # rescales all of the parameters to lie between -1 and 1 
    xx2          = 2 ./(1 .+ exp.(-(xx .- x0)./50) ) .- 1
    # Transform function so that boundrary conditions are satisfied 
    endogParams2 = [ transform(xx2[i], pb[i], endogParams[i]) for i = 1:length(endogParams) ] 
    baseline     = model(ε = endogParams2[1] , σ_η = endogParams2[2], χ = endogParams2[3], γ = endogParams2[4]) 

    # Simulate the model and compute moments
    out     = simulate(baseline, zshocks)
    mod_mom = [out.var_Δlw, out.dlw1_du, out.dΔlw_dy, out.w_y]
    d       = (mod_mom - data_mom)./2(mod_mom + data_mom) # arc percentage differences
    f       = out.flag < 1 ? d'*W*d : Inf
    println(string(d'*W*d))
    return f, mod_mom, out.flag
end

""" 
Transform parameters to lie within bounds
"""
function transform(xx2, pb, p0)
    if xx2 > 0
        x1 = xx2*(pb[2] - p0) + p0
    else
        x1 = xx2*(p0 - pb[1]) + p0  
    end
    return x1
end

# initial parameter values
endogParams    = zeros(4)
endogParams[1] = 0.5   # ε
endogParams[2] = 0.05  # σ_η
endogParams[3] = 0.3   # χ
endogParams[4] = 0.66  # γ

## Moments we are targeting
data_mom       =[0.53^2, -0.5, .05, 0.6]  # may need to update 
J              = length(data_mom)

## Parameter bounds and weight matrix
W              = Matrix(1.0I, J, J)       # inverse of covariance matrix of data_mom?
pb             = OrderedDict{Int,Array{Real,1}}([ # parameter bounds
                (1, [0  , 3.0]),
                (2, [0.0, .36]),
                (3, [-1, 1]),
                (4, [0.3, 0.9]) ])

## Build zshocks
baseline     = model()
@unpack N_z, P_z, zgrid  = baseline
N            = 100000
T            = 100
burnin       = 1000

# compute the invariant distribution of z
A           = P_z- Matrix(1.0I, N_z, N_z)
A[:,end]   .= 1
O           = zeros(1,N_z)
O[end]      = 1
z_ss_dist   = (O*inv(A))
@assert(isapprox(sum(z_ss_dist),1))

# create z shocks. do this out OUTSIDE of estimation.
distr        = floor.(Int64, N*z_ss_dist)
z_shocks     = OrderedDict{Int, Array{Real,1}}()
z_shocks_idx = OrderedDict{Int, Array{Real,1}}()

for iz = 1:length(zgrid)
    temp               = simulateZShocks(baseline, N = distr[iz], T = T, z0_idx = iz)
    z_shocks[iz]        = vec(temp.z_shocks)
    z_shocks_idx[iz]    = vec(temp.z_shocks_idx)
end

# create one z_t string: set z_1 to default value of 1.
zstring  = simulateZShocks(baseline, N = 1, T = N + burnin)

# create an ordered tuple for the zShocks
zshocks = (z_shocks = z_shocks, z_shocks_idx = z_shocks_idx, distr = distr, N = N,
T = T, zstring = zstring, burnin = burnin, z_ss_dist = z_ss_dist)