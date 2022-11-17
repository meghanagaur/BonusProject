using Distributed, SlurmClusterManager

cd(dirname(@__FILE__))

# Start the worker processes
addprocs(SlurmManager())

@everywhere begin

    # Get slurm idx
    ja_idx  = parse(Int64, ENV["SLURM_ARRAY_TASK_ID"])

    include("functions/smm_settings.jl") # SMM inputs, settings, packages, etc.

    # get moment targets
    data_mom, mom_key = target_moments()
    K                 = length(data_mom)
    W                 = getW()

    # combinations of parameters we are fixing and varying
    symbols    = [:ε, :ε, :ε, :ε, :hbar]
    ε_vals     = [0.2, 0.3, 0.4, 0.5, 0.3]
    hbar_vals  = ones(5)

    # File location for saving jld output 
    files   = ["pretesting_fix_eps"*replace(string(ε_vals[i]), "." => "") for i = 1:length(ε_vals)-1]
    files   = [files; "pretesting_fix_hbar1"]
    file    = files[ja_idx]

    # Evalute objective function at i-th parameter vector
    function evaluate!(i, sob_seq, param_vals, param_est, shocks, data_mom, W)
        return objFunction(sob_seq[:,i], param_vals, param_est, shocks, data_mom, W)
    end

    # Define the values for parameters, we are fixing
    param_vals  = OrderedDict{Symbol, Real}([ 
                (:ε,  ε_vals[ja_idx]),         # ε
                (:σ_η, 0.2759),                # σ_η 
                (:χ, 0.4417),                  # χ
                (:γ, 0.4916),                  # γ
                (:hbar, hbar_vals[ja_idx]) ])  # hbar

    # Parameters we will fix (if any) in ε, σ_η, χ, γ, hbar 
    delete!(param_bounds, symbols[ja_idx])

    # Parameters that we will estimate
    J           = length(param_bounds)
    param_est   = OrderedDict{Symbol, Int64}()
    for (i, dict) in enumerate(collect(param_bounds))
        local key = dict[1]
        param_est[key] = i
    end

    # Sample I Sobol vectors from the parameter space
    I_max        = 20000
    lb           = zeros(J)
    ub           = zeros(J)

    for (key, value) in param_est
        lb[value]   = param_bounds[key][1]
        ub[value]   = param_bounds[key][2]
    end

    s            = SobolSeq(lb, ub)
    seq          = skip(s, 10000, exact=true)
    sob_seq      = reduce(hcat, next!(seq) for i = 1:I_max)

end

# Evaluate the objective function for each parameter vector i=1
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
