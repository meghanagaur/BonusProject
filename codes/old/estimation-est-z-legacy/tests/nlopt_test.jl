cd(dirname(@__FILE__))

using NLopt

include("../functions/smm_settings.jl")                 # SMM inputs, settings, packages, etc.

# Local optimization options 
algorithm      = :LN_BOBYQA                             # :LN_NELDERMEAD
ftol_rel       = 1e-6                                   # relative function tolerance: set to 0 if no tol
xtol_rel       = 0                                      # relative parameter tolerance: set to 0 if no tol
max_time       = 60*90                                  # max time for local optimization (in seconds), 0 = no limit

println("Algorithm:\t"*string(algorithm))

# get moment targets
data_mom, mom_key = moment_targets()
K                 = length(data_mom)
W                 = getW(K)
W[end-1,end-1]    = 0    # fix rho, drop alp_ρ for now

# Draw the shocks for simulation
shocks  = rand_shocks()

# Define the baseline values
param_vals  = OrderedDict{Symbol, Real}([ 
                (:ε,   0.3),         # ε
                (:σ_η, 0.5  ),       # σ_η 
                (:χ, 0.0),           # χ
                (:γ, 0.6),           # γ
                (:hbar, 1.0),        # hbar
                (:ρ, 0.95^(1/3)),    # ρ
                (:σ_ϵ, 0.003) ])     # σ_ϵ

params_fix  = [:ε :ρ] 
for p in params_fix
    delete!(param_bounds, p)
end

# Parameters that we will estimate
J           = length(param_bounds)
param_est   = OrderedDict{Symbol, Int64}()
for (i, dict) in enumerate(collect(param_bounds))
    local key = dict[1]
    param_est[key] = i
end

# Define the optimization object
opt = Opt(algorithm, J)

# Deine the objective function
# add dummy gradient: https://discourse.julialang.org/t/nlopt-forced-stop/47747/3
obj(x, dummy_gradient!)  = objFunction(x, param_vals, param_est, shocks, data_mom, W)[1]
opt.min_objective = obj

# bound constraints
lower, upper     = get_bounds(param_est, param_bounds)
opt.lower_bounds = lower 
opt.upper_bounds = upper

# tolerance values 
opt.stopval  = 0
opt.ftol_rel = ftol_rel
opt.xtol_rel = xtol_rel
opt.maxtime  = max_time

@unpack σ_η, χ, γ, hbar, σ_ϵ = param_vals
start = [σ_η, χ, γ, hbar, σ_ϵ ] # inital guess

@time (min_f, arg_min, ret) = NLopt.optimize(opt, start)

println("minimum:\t\t"*string(min_f))
println("minimizer:\t\t"*string(arg_min))
println("reason for stopping:\t"*string(ret))