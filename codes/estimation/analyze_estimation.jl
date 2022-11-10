using Distributed, SlurmClusterManager, DelimitedFiles

cd(dirname(@__FILE__))

# Start the worker processes
addprocs(SlurmManager())
#addprocs(2)

@everywhere begin

    include("smm_settings.jl") # SMM inputs, settings, packages, etc.
    using DelimitedFiles

    # Evalute objective function at i-th parameter vector
    function evaluate!(i, sob_seq, pb, shocks, data_mom, W)
        return objFunction(sob_seq[:,i], pb, shocks, data_mom, W)
    end

    # Clean the results 
    est_output = readdlm("jld/estimatio.txt", ',', Float64)      # open current output across all jobs
    sob_seq    = est_output[:, 2:(2+J-1)]'        # get parameters 

end

# Evaluate the objective function for each parameter vector
@time output = pmap(i -> evaluate!(i, sob_seq, param_bounds, shocks, data_mom, W), 1:size(sob_seq,2)) 

# Kill the processes
rmprocs(nprocs())

# Save the raw results
#save(loc*"jld/pretesting.jld2", Dict("output" => output, "sob_seq" => sob_seq, "baseline_model" => model() ))

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
save("jld/est_output.jld2",  Dict("moms" => moms, "fvals" => fvals, "pars" => pars', "IR_flag" => IR_flag, "IR_err" => IR_err))
