using DelimitedFiles

cd(dirname(@__FILE__))

## Logistics
file_str         = "filename"
file_load        = "../jld/pretesting_"*file_str*".jld2"  # file to-load location
file_save        = "../jld/estimation_"*file_str*".txt"   # file to-save 
N_procs          = 20                                     # number of jobs in job array
N_string         = 50                                     # length of each worker string

# Local optimization settings if using NLopt
algo             = :NLopt                                 # set to :OPTIM if using Optim

# Options 
smm       = true
est_z     = true

# Task number for job array
idx = parse(Int64, ENV["SLURM_ARRAY_TASK_ID"])

println("JLD FILE = ", file_str)

# Load helper functions
include("../../functions/smm_settings.jl")                # SMM inputs, settings, packages, etc.

# Load the pretesting ouput. Use the "best" Sobol points for our starting points.
@unpack moms, fvals, pars, mom_key, param_bounds, param_est, param_vals, data_mom, J, K, W = load(file_load) 

## Specifciations for the shocks in simulation
@unpack P_z, p_z, z_ss_idx = model(ρ = param_vals[:ρ], σ_ϵ = param_vals[:σ_ϵ])

# Load Shocks
shocks  = drawShocksALP(P_z, p_z; smm = true, z0_idx = z_ss_idx)

# Define the NLopt optimization object
if algo == :NLopt

    opt_1  = Opt(:LN_NELDERMEAD, J) #Opt(:LN_NELDERMEAD, J)  #Opt(:LN_SBPLX, J) nothing
    opt_2  = Opt(:LN_NELDERMEAD, J) 

    # Objective function

    # need to add dummy gradient: https://discourse.julialang.org/t/nlopt-forced-stop/47747/3
    obj(x, dummy_gradient!)       = objFunction(x, param_vals, param_est, shocks, data_mom, W; smm = smm, est_z = est_z)[1]

    # Bound constraints
    lower, upper                  = get_bounds(param_est, param_bounds)

    if !isnothing(opt_1)
        opt_1.min_objective       = obj 
        # tolerance and time settings 
        opt_1.stopval             = 1e-3
        opt_1.ftol_rel            = 1e-5
        opt_1.ftol_abs            = 1e-5
        opt_1.xtol_rel            = 0.0  
        opt_1.maxtime             = (60*60) 
        opt_1.lower_bounds        = lower 
        opt_1.upper_bounds        = upper
    end

    if !isnothing(opt_2)
        opt_2.min_objective       = obj
        # tolerance and time settings 
        opt_2.stopval             = 1e-5
        opt_2.ftol_rel            = 1e-8
        opt_2.ftol_abs            = 1e-8
        opt_2.xtol_rel            = 0.0  
        opt_2.maxtime             = (60*60)*1.5
        opt_2.lower_bounds        = lower 
        opt_2.upper_bounds        = upper
    end
end

# Sort and reshape the parameters for distribution across jobs
Nend           = N_procs*N_string                  # number of initial points
#sorted_indices = reverse(sortperm(fvals)[1:Nend]) # sort by function values in DESCENDING order
sorted_indices = sortperm(fvals)[1:Nend]           # sort by function values in ASCENDING order
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
@time output = tiktak(init_points, file_save, param_bounds, param_vals, param_est, shocks, data_mom, W, I_max; 
                        opt_1 = opt_1, opt_2 = opt_2, smm = smm, est_z = est_z)

# Print output 
for i = 1:N_string
    println(output[:,i])
end
