using Distributed, SlurmClusterManager

#addprocs(SlurmManager())
addprocs(2)

@everywhere begin
    loc = "/Users/meghanagaur/BonusProject/codes/estimation/"
    #loc = ""
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
    sobol_sort     = hcat(sobol_sort, zeros(size(sobol_sort,1)) )[1:Nend,:] 
    sob_int          = reshape(sobol_sort,  (N_procs, N_string, size(sobol_sort,2) )) # NWORKERS X NSTRING X K

   sobol = zeros(J, N_string, N_procs)
   @views @inbounds for j = 1:nworkers()
        @inbounds for i = 1:N_string
            sobol[:,i,j] = sob_int[j,i,:]
        end
   end
end