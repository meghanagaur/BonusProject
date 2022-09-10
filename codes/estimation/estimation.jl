using Distributed, SlurmClusterManager

#addprocs(5)
addprocs(SlurmManager())

@everywhere begin
    using DistributedArrays, DistributedArrays.SPMD
    loc = "" #"/Users/meghanagaur/BonusProject/codes/estimation/"
    include(loc*"smm_settings.jl") # SMM inputs, settings, packages, etc.
    include(loc*"tik-tak.jl") # SMM inputs, settings, packages, etc.
end

# Load the pretesting ouput. Use the "best" Sobol points for our starting points.
@unpack moms, fvals, params = load(loc*"jld/pretesting_clean.jld2") 

sorted_indices = sortperm(fvals)
Nend           = ceil(Int64,0.1*length(fvals))
sobol_sort     = params[sorted_indices,:]
sobol          = sobol_sort[shuffle(1:Nend),:]
save(loc*"jld/sobol.jld2", Dict("sobol" => sobol ))

# initialize the big distributed vectors 
sob_d       = distribute(sobol)         # distributed sobol points
fvals_d     = dzeros(size(sobol,1))     # distributed function values
argmin_d    = dzeros(size(sobol))       # distributed argmin vector

# initialize the small distributed vectors 
iter_p      = ddata(); 
min_p       = ddata();
argmin_p    = dzeros( (nworkers(),J) , workers()[1:end], [nworkers(), 1] )

# starting point for local optimization step (in (-1,1) interval for transformed parameters )
init_x      = zeros(J)

# Run the parallel optimization code 
@time spmd(tiktak_spmd, sob_d, fvals_d, argmin_d, min_p, argmin_p, iter_p, 
    init_x, pb, zshocks, data_mom, W; pids = workers()) # executes on all workers

[@fetchfrom p localindices(fvals_d) for p in workers()]
[@fetchfrom p localindices(sob_d) for p in workers()]
[@fetchfrom p localindices(argmin_d) for p in workers()]
[@fetchfrom p localindices(iter_p) for p in workers()]
[@fetchfrom p localindices(min_p) for p in workers()]
[@fetchfrom p localindices(argmin_p) for p in workers()]


sob_d
fvals_d
argmin_d
argmin_p
iter_p
min_p