using DelimitedFiles

cd(dirname(@__FILE__))

## Logistics
file_str       = "fix_eps03"
file_load      = "runs/jld/pretesting_"*file_str*".jld2"  # file to-load location
file_save      = "jld/estimation_"*file_str*".txt"   # file to-save 
N_procs        = 20                                  # number of jobs in job array

# Task number for job array
idx = parse(Int64, ENV["SLURM_ARRAY_TASK_ID"])

println("JLD FILE = ", file_str)

# Load helper functions
include("functions/smm_settings.jl")      # SMM inputs, settings, packages, etc.
include("functions/tik-tak_job_array.jl") # tik-tak code 

# Load the pretesting ouput. Use the "best" Sobol points for our starting points.
@unpack moms, fvals, pars, mom_key, param_bounds, param_est, param_vals, data_mom, J, K, W = load(file_load) 

# Sort and reshape the parameters for distribution across jobs
N_string       = 25                                # length of each worker string
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

# Starting point for local optimization step with bounds (in (-1,1) interval because these are transformed parameters)
init_x      = zeros(J)

# Run the optimization code 
@time output = tiktak(init_points, file_save, init_x, param_bounds, param_vals, param_est, shocks, data_mom, W, I_max)

# Print output 
for i = 1:N_string
    println(output[:,i])
end

