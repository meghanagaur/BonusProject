"""
Logit transformation to achieve desired bounds.
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

# initial parameter values
endogParams    = zeros(3)
endogParams[1] = 0.5   # ε
endogParams[2] = 0.05  # σ_η
endogParams[3] = 0.3   # χ
#endogParams[4] = 0.66  # γ

## Moments we are targeting
data_mom       =[0.53^2, -0.5, .05] ##, 0.6]  # may need to update 
J              = length(data_mom)

## Parameter bounds and weight matrix
W              = Matrix(1.0I, J, J)       # inverse of covariance matrix of data_mom?
pb             = OrderedDict{Int,Array{Real,1}}([ # parameter bounds
                (1, [0  , 1.0]),
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
A           = P_z - Matrix(1.0I, N_z, N_z)
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

#=
# checks
plot(x->logit(x;x0=0),-10,10)
hline!([-1])
hline!([1])

i=4
init_x = 10*ones(4)
δ=0.5*init_x[1]
plot(x -> transform(x, pb[i], init_x[i], endogParams[i], λ = 1), init_x[i] - δ, init_x[i] + δ,legend=:false)
hline!([pb[i][1]])
hline!([pb[i][2]])
hline!([endogParams[i]])
=#