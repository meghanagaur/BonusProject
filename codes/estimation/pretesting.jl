using Distributed, SlurmClusterManager
addprocs(SlurmManager())

# Start the worker processes
#num_tasks = parse(Int, ENV["SLURM_NTASKS"])
#addprocs(num_tasks)

loc = "/Users/meghanagaur/BonusProject/codes/estimation/"
@everywhere begin

    include("smm_settings.jl") # SMM inputs, settings, packages, etc.

    # Evalute objective function
    function evaluate!(i, sob_seq, pb, zshocks, data_mom, W)
        return objFunction(sob_seq[i,:], pb, zshocks, data_mom, W)
    end

    # Sample I Sobol vectors from the parameter space
    const II     = 10000
    lb           = zeros(J)
    ub           = zeros(J)
    for i = 1:J
        lb[i] = pb[i][1]
        ub[i] = pb[i][2]
    end
    s             = SobolSeq(lb, ub)
    seq           = skip(s, 10000, exact=true)
    sob_seq       = reduce(hcat, next!(seq) for i = 1:II)'
end

# Let's evaluate the objective function for each parameter value
@time output = pmap(i -> evaluate!(i, sob_seq, pb, zshocks, data_mom, W), 1:II) 

# Kill the processes
rmprocs(nprocs())

# Save the raw results
save(loc*"jld/pretesting.jld2", Dict("output" => output, "sob_seq" => sob_seq, "baseline_model" => model() ))


# Clean the results 
#@unpack output = load(loc*"jld/pretesting.jld2")

# Retain the valid indices
N_old   = length(output)
indices = [output[i][3] == 0 for i =1:N_old]
out_new = output[indices]
N       = length(out_new)

# Record the function values
fvals   = [out_new[i][1] for i = 1:N]
# Record the moments
moms   = reduce(hcat, out_new[i][2] for i = 1:N)'
# Record the parameters
params = sob_seq[indices,:] 

# Retain the valid indices
N_old   = length(output)
indices = [output[i][3] == 0 for i =1:N_old]
out_new = output[indices]
N       = length(out_new)

# Record the function values
fvals   = [out_new[i][1] for i = 1:N]
# Record the moments
moms   = reduce(hcat, out_new[i][2] for i = 1:N)'
# Record the parameters
params = sob_seq[indices,:] 

save(loc*"jld/pretesting_clean.jld2", 
    Dict("moms" => moms, "fvals" => fvals, "params" => params) )
