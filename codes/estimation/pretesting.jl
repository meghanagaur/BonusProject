using Distributed 

# Start the worker processes
num_tasks = parse(Int, ENV["SLURM_NTASKS"])
addprocs(num_tasks)

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

# Save the results
save("jld/pre-testing.jld2", Dict("output" => output, "sob_seq" => sob_seq, "baseline_model" => model() ))


                        