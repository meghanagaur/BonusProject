using Distributed, SlurmClusterManager

cd(dirname(@__FILE__))

idx = parse(Int64, ENV["SLURM_ARRAY_TASK_ID"])

# Load helper functions
include("smm_settings.jl") # SMM inputs, settings, packages, etc.
include("tik-tak_job_array.jl") # SMM inputs, settings, packages, etc.

# Load the pretesting ouput. Use the "best" Sobol points for our starting points.
@unpack moms, fvals, pars = load("jld/pretesting_clean.jld2") 

file           = "jld/estimation.txt"  # file save
N_procs        = 20                    # number of jobs in job array

# sort and reshape the parameters for distribution across worksers
N_string       = 25                                # length of each worker string
Nend           = N_procs*N_string                  # number of initial points
sorted_indices = reverse(sortperm(fvals)[1:Nend])  # sort by function values in descending order
sobol_sort     = pars[sorted_indices,:]            # get parameter values: N_TASKS*NSTRING x J
# reshape parameter vector 
sob_int        = reshape(sobol_sort,  (N_procs, N_string, J )) # N_TASKS x NSTRING x J

# final parameter matrix (reshape for column-memory access: NSTRING x J x N_TASKS
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

# starting point for local optimization step with bounds (in (-1,1) interval for transformed parameters )
init_x      = zeros(J)

# Run the optimization code 
@time output = tiktak(init_points, fvals, argmin, I_max, file, init_x, param_bounds, shocks, data_mom, W)

for i = 1:N_string
    println(output[:,i])
end

