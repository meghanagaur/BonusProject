using Distributed, SlurmClusterManager
addprocs(SlurmManager())

# Start the worker processes
#num_tasks = parse(Int, ENV["SLURM_NTASKS"])
#addprocs(2)

@everywhere begin

    include("smm_settings.jl") # SMM inputs, settings, packages, etc.

    # Evalute objective function at i-th parameter vector
    function evaluate!(i, sob_seq, pb, shocks, data_mom, W)
        return objFunction(sob_seq[:,i], pb, shocks, data_mom, W)
    end

    # Sample I Sobol vectors from the parameter space
    I_max        = 25000
    lb           = zeros(J)
    ub           = zeros(J)
    for i = 1:J
        lb[i] = param_bounds[i][1]
        ub[i] = param_bounds[i][2]
    end
    s             = SobolSeq(lb, ub)
    seq           = skip(s, 10000, exact=true)
    sob_seq       = reduce(hcat, next!(seq) for i = 1:I_max)
end

# Evaluate the objective function for each parameter vector
@time output = pmap(i -> evaluate!(i, sob_seq, param_bounds, shocks, data_mom, W), 1:I_max) 

# Kill the processes
rmprocs(nprocs())

# Save the raw results
save(loc*"jld/pretesting.jld2", Dict("output" => output, "sob_seq" => sob_seq, "baseline_model" => model() ))

# Clean the results 

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
pars    = sob_seq[:,indices] 
# Record the IR flag
IR_flag = reduce(hcat, out_new[i][4] for i = 1:N)
# Record the IR flag
IR_err  = reduce(hcat, out_new[i][5] for i = 1:N)

# Save the output
save("jld/pretesting_clean.jld2",  Dict("moms" => moms, "fvals" => fvals, 
                                    "pars" => pars', "IR_flag" => IR_flag, "IR_err" => IR_err))
