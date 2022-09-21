using Distributed, SlurmClusterManager

#addprocs(SlurmManager())
#addprocs(2)

@everywhere begin
    using DistributedArrays, DistributedArrays.SPMD
    #loc = "/Users/meghanagaur/BonusProject/codes/estimation/"
    loc = ""
    include(loc*"smm_settings.jl") # SMM inputs, settings, packages, etc.
end  

@everywhere begin
    include(loc*"tik-tak.jl") # SMM inputs, settings, packages, etc.
    # Load the pretesting ouput. Use the "best" Sobol points for our starting points.
    @unpack moms, fvals, pars = load(loc*"jld/pretesting_clean.jld2") 

    # sort and reshape the parameters for distribution across worksers
    N_string       = 10 # length of each worker string
    N_procs        = nworkers()
    sorted_indices = sortperm(fvals)
    Nend           = nworkers()*N_string #ceil(Int64, 0.1*length(fvals))
    sobol_sort     = pars[sorted_indices,:]
    sobol_sort     = sobol_sort[1:Nend,:] 
    sob_int        = reshape(sobol_sort,  (N_procs, N_string, size(sobol_sort,2) )) # NWORKERS X NSTRING X K

   sobol = zeros(J, N_string, N_procs)
   @views @inbounds for j = 1:nworkers()
        @inbounds for i = 1:N_string
            sobol[:,i,j] = sob_int[j,i,:]
        end
   end
end

# initialize the big distributed vectors 
fvals_d     = dzeros(size(sobol;)[2:3], workers()[1:end], [1, nworkers()])     # distributed function values
argmin_d    = dzeros(size(sobol),  workers()[1:end], [1, 1, nworkers()])     # distributed argmin vector

# initialize the small distributed vectors 
iter_p      = ddata(); # dzeros(nworkers()) #
min_p       = dfill(1000.0, nworkers()); #ddata(); 
argmin_p    = dzeros( (J, nworkers()) , workers()[1:end], [1, nworkers()] )

# starting point for local optimization step with bounds (in (-1,1) interval for transformed parameters )
init_x      = zeros(J)

# Run the parallel optimization code 
@time spmd(tiktak_spmd, sobol, fvals_d, argmin_d, min_p, argmin_p, iter_p, 
    init_x, param_bounds, zshocks, data_mom, W; pids = workers()) # executes on all workers

[@fetchfrom p localindices(fvals_d) for p in workers()]
[@fetchfrom p localindices(argmin_d) for p in workers()]
[@fetchfrom p localindices(iter_p) for p in workers()]
[@fetchfrom p localindices(min_p) for p in workers()]
[@fetchfrom p localindices(argmin_p) for p in workers()]

fvals_d   = convert( Matrix{Float64}, fvals_d)
argmin_d  = convert( Array{Float64, 3}, argmin_d)
argmin_p  = convert( Matrix{Float64}, argmin_p )
min_p     = convert( Vector{Float64}, min_p)

# kill processes
rmprocs(workers())

# Save the output
save(loc*"jld/estimation.jld2", Dict("sobol" => sobol, "fvals_d" => fvals_d, 
    "argmin_d" => argmin_d, "argmin_p" => argmin_p, "min_p" => min_p))

