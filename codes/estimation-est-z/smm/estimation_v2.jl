using DelimitedFiles

cd(dirname(@__FILE__))

## Logistics
file_str       = "filename"
file_load      = "../jld/pretesting_"*file_str*".jld2"  # file to-load location
file_save      = "../jld/estimation_"*file_str*".txt"   # file to-save 
N_procs        = 20                                     # number of jobs in job array
N_string       = 25                                     # length of each worker string

# Local optimization settings if using NLopt
algo_nlopt           = :LN_NELDERMEAD                   # :LN_BOBYQA, algorithm for NLopt, set to :OPTIM if using Optim
ftol_rel_nlopt       = 1e-6                             # relative function tolerance: set to 0 if no tol
xtol_rel_nlopt       = 0                                # relative parameter tolerance: set to 0 if no tol
max_time_nlopt       = 60*60                            # max time for local optimization (in seconds), 0 = no limit

# Task number for job array
idx = parse(Int64, ENV["SLURM_ARRAY_TASK_ID"])

println("JLD FILE = ", file_str)

# Load helper functions
include("../../functions/smm_settings.jl")         # SMM inputs, settings, packages, etc.
include("../../functions/tik-tak_job_array_v2.jl") # tik-tak code 

# Load the pretesting ouput. Use the "best" Sobol points for our starting points.
@unpack moms, fvals, pars, mom_key, param_bounds, param_est, param_vals, data_mom, J, K, W = load(file_load) 

## Specifciations for the shocks in simulation
shocks  = rand_shocks()

# Define the NLopt optimization object
if algo_nlopt == :OPTIM
    opt = nothing
else
    opt = Opt(algo_nlopt, J) 

    # Objective function
    # need to add dummy gradient: https://discourse.julialang.org/t/nlopt-forced-stop/47747/3
    obj(x, dummy_gradient!)  = objFunction(x, param_vals, param_est, shocks, data_mom, W)[1]
    opt.min_objective        = obj

    # Bound constraints
    lower, upper = get_bounds(param_est, param_bounds)
    opt.lower_bounds = lower 
    opt.upper_bounds = upper

    # Tolerance values 
    opt.stopval  = 0
    opt.ftol_rel = ftol_rel_nlopt 
    opt.xtol_rel = xtol_rel_nlopt
    opt.maxtime  = max_time_nlopt

end

# Sort and reshape the parameters for distribution across jobs
Nend           = N_procs*N_string                  # number of initial points
sorted_indices = reverse(sortperm(fvals)[1:Nend])  # sort by function values in descending order
sobol_sort     = pars[sorted_indices,:]            # get parameter values: N_TASKS*NSTRING x J

# reshape parameter vector 
sob_int        = reshape(sobol_sort,  (N_procs, N_string, J )) # N_TASKS x NSTRING x J

# Final parameter matrix (reshape for column-memory access: NSTRING x J x N_TASKS
sobol = zeros(J, N_string, N_procs)
@views @inbounds for j = 1:N_procs
    @inbounds for i = 1:N_string
        sobol[:,i,j] = sob_int[j,i,:]
    end
end

# Starting values
init_points = sobol[:,:,idx]

# Maximal iterations across all nodes
I_max       = N_string*N_procs

# print the number of threads
println("Threads: ", Threads.nthreads())

# Run the optimization code 
@time output = tiktak(init_points, file_save, param_bounds, param_vals, param_est, shocks, data_mom, W, I_max, opt = opt)

# Print output 
for i = 1:N_string
    println(output[:,i])
end
