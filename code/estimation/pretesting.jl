using Distributed 

# start processes
num_tasks = parse(Int, ENV["SLURM_NTASKS"])
num_cores = parse(Int, ENV["SLURM_CPUS_PER_TASK"])

addprocs(num_tasks*num_cores)

@everywhere begin

    using DynamicModel, BenchmarkTools, DataStructures, Distributions, Optim, Sobol,
    ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase, JLD2

    include("simulation.jl")   # simulation functions
    include("smm_settings.jl") # smm functions

    """
    Objective function to be minimized during SMM WITHOUT BOUNDS.
    Variable descriptions below.
    xx           = evaluate objFunction @ these parameters 
    zshocks      = z shocks for the simulation
    pb           = parameter bounds
    data_mom     = data moments
    W            = weight matrix for SMM
        ORDERING OF PARAMETERS/MOMENTS
    ε            = 1st param
    σ_η          = 2nd param
    χ            = 3rd param
    γ            = 4th param
    var_Δlw      = 1st moment (variance of log wage changes)
    dlw1_du      = 2nd moment (dlog w_1 / d u)
    dΔlw_dy      = 3rd moment (d Δ log w_it / y_it)
    w_y          = 4th moment (PV of labor share)
    """
    function objFunction(xx, pb, zshocks, data_mom, W)
        inbounds = minimum( [ pb[i][1] <= xx[i] <= pb[i][2] for i = 1:length(xx) ]) >= 1
        if inbounds == 0
            f        = 10000
            mod_mom  = NaN
            flag     = 1
        elseif inbounds == 1
            baseline = model(ε = xx[1] , σ_η = xx[2], χ = xx[3]) #, γ = xx[4]) 
            # Simulate the model and compute moments
            out      = simulate(baseline, zshocks)
            mod_mom  = [out.var_Δlw, out.dlw1_du, out.dΔlw_dy] #, out.w_y]
            d        = (mod_mom - data_mom)./0.5(mod_mom + data_mom) # arc % change
            f        = out.flag < 1 ? d'*W*d : 10000
        end
        return [f, vec(mod_mom), out.flag]
    end

    # Evalute objective function
    function evaluate!(i, sob_seq, pb, zshocks, data_mom, W)
        return objFunction(sob_seq[i,:], pb, zshocks, data_mom, W)
    end

    # Sample N Sobol vectors from the parameter space
    N  = 10000
    lb = zeros(J)
    ub = zeros(J)
    for i = 1:J
        lb[i] = pb[i][1]
        ub[i] = pb[i][2]
    end

    s             = SobolSeq(lb, ub)
    seq           = skip(s, 10000, exact=true)
    sob_seq       = reduce(hcat, next!(seq) for i = 1:N)'
end

# Let's evaluate the objective function for each parameter value
@time output = pmap(i -> evaluate!(i, sob_seq, pb, zshocks, data_mom, W), 1:N) 

rmprocs(rmprocs())

# Save the results
save("jld/pre-testing.jld2", Dict("output" => output, "sob_seq" => sob_seq, "model_type" => model() ))


                        