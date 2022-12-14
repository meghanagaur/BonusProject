using Distributed, SlurmClusterManager
cd(dirname(@__FILE__))

# Start the worker processes
addprocs(SlurmManager())

# File location for saving jld output + slurm idx
@everywhere ε_val = 0.3
file  = "pretesting_fix_eps"*replace(string(ε_val), "." => "")*"_low_pt_high_cyc_w1"

@everywhere begin

    include("functions/smm_settings.jl") # SMM inputs, settings, packages, etc.

    # get moment targets
    data_mom, mom_key = moment_targets(dlw_dly = 0.05, dlw1_du = -3.0)
    K                 = length(data_mom)
    W                 = getW(K)

    # Evalute objective function at i-th parameter vector
    function evaluate!(i, sob_seq, param_vals, param_est, shocks, data_mom, W)
        return objFunction(sob_seq[:,i], param_vals, param_est, shocks, data_mom, W)
    end

    # Define the baseline values
    param_vals  = OrderedDict{Symbol, Real}([ 
                    (:ε,   ε_val),       # ε
                    (:σ_η, 0.2759),      # σ_η 
                    (:χ, 0.4417),        # χ
                    (:γ, 0.4916),        # γ
                    (:hbar, 1.0) ])      # hbar

    # Parameters we will fix (if any) in ε, σ_η, χ, γ, hbar 
    params_fix  = [:ε] 
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

    # Sample I Sobol vectors from the parameter space
    I_max        = 30000
    lb           = zeros(J)
    ub           = zeros(J)

    for (key, value) in param_est
        lb[value]   = param_bounds[key][1]
        ub[value]   = param_bounds[key][2]
    end

    s            = SobolSeq(lb, ub)
    seq          = skip(s, 10000, exact = true)
    sob_seq      = reduce(hcat, next!(seq) for i = 1:I_max)
end

# Evaluate the objective function for each parameter vector
@time output = pmap(i -> evaluate!(i, sob_seq, param_vals, param_est, shocks, data_mom, W), 1:I_max) 

# Kill the processes
rmprocs(nprocs())

# Clean the output 

# Retain the valid vectors (i.e. solutions without flags)
N_old   = length(output)
indices = [output[i][3] == 0 for i = 1:N_old]
out_new = output[indices]
N       = length(out_new)

# Record the function values
fvals   = [out_new[i][1] for i = 1:N]
# Record the moments
moms    = reduce(hcat, out_new[i][2] for i = 1:N)'
# Record the parameters
pars    = sob_seq[:,indices]' 
# Record the IR flag
IR_flag = reduce(hcat, out_new[i][4] for i = 1:N)
# Record the IR flag
IR_err  = reduce(hcat, out_new[i][5] for i = 1:N)

# Save the output
save("runs/jld/"*file*".jld2",  Dict("moms" => moms, "fvals" => fvals, "mom_key" => mom_key, "param_est" => param_est, "param_vals" => param_vals, 
                            "param_bounds" => param_bounds, "pars" => pars, "IR_flag" => IR_flag, "IR_err" => IR_err, "J" => J, "K" => K,
                            "W" => W, "data_mom" => data_mom))
