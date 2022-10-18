using Distributed, SlurmClusterManager

#addprocs(SlurmManager())
#addprocs(2)

@everywhere begin
    using DistributedArrays, DistributedArrays.SPMD
    #oc = "/Users/meghanagaur/BonusProject/codes/estimation/"
    loc = ""
    include(loc*"smm_settings.jl") # SMM inputs, settings, packages, etc.
end  

@everywhere begin
    include(loc*"tik-tak.jl") # SMM inputs, settings, packages, etc.
    # Load the pretesting ouput. Use the "best" Sobol points for our starting points.
    @unpack moms, fvals, pars = load(loc*"jld/pretesting_clean.jld2") 

    # sort and reshape the parameters for distribution across worksers
    N_string       = 20                                # length of each worker string
    N_procs        = nworkers()                        # number of processes
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

end

# initialize the big distributed vectors 
fvals_d     = dzeros(size(sobol;)[2:3], workers()[1:end], [1, nworkers()])     # distributed function values
argmin_d    = dzeros(size(sobol),  workers()[1:end], [1, 1, nworkers()])       # distributed argmin vector

# initialize the small distributed vectors 
iter_p      = ddata();  
min_p       = dfill(10.0^5, nworkers());
argmin_p    = dzeros( (J, nworkers()) , workers()[1:end], [1, nworkers()] )

# starting point for local optimization step with bounds (in (-1,1) interval for transformed parameters )
init_x      = zeros(J)

# Run the parallel optimization code 
@time spmd(tiktak_spmd, sobol, fvals_d, argmin_d, min_p, argmin_p, iter_p, 
    init_x, param_bounds, shocks, data_mom, W; pids = workers()) # executes on all tasks

# Look at indices to verify they match expected assignemnt
[@fetchfrom p localindices(fvals_d) for p in workers()]
[@fetchfrom p localindices(argmin_d) for p in workers()]
[@fetchfrom p localindices(iter_p) for p in workers()]
[@fetchfrom p localindices(min_p) for p in workers()]
[@fetchfrom p localindices(argmin_p) for p in workers()]

# Get estimation output (function values, arg mins)
fvals_d   = convert( Matrix{Float64}, fvals_d)
argmin_d  = convert( Array{Float64, 3}, argmin_d)
argmin_p  = convert( Matrix{Float64}, argmin_p )
min_p     = convert( Vector{Float64}, min_p)
iter_p    = [@fetchfrom p iter_p[:L] for p in workers()]

# Kill all processes
rmprocs(workers())

# Save the output
save(loc*"jld/estimation.jld2", Dict("sobol" => sobol, "fvals_d" => fvals_d, 
    "argmin_d" => argmin_d, "argmin_p" => argmin_p, "min_p" => min_p, "iter_p" => iter_p) )

