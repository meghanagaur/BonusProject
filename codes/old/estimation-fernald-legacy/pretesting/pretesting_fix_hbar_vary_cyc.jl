using Distributed, SlurmClusterManager
cd(dirname(@__FILE__))

# Start the worker processes
addprocs(SlurmManager())

# File location for saving jld output + slurm idx
@everywhere hbar_val = 1.0

# Load SMM inputs, settings, packages, etc.
@everywhere include("../functions/smm_settings.jl") 

@everywhere begin

    # Get slurm job array idx
    ja_idx  = parse(Int64, ENV["SLURM_ARRAY_TASK_ID"])

    # different values of the cyclicality of new hire wages
    cyc_vals    = 1*[0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    cyc         = -cyc_vals[ja_idx]
    file        = "pretesting_fix_hbar"*replace(string(hbar_val), "." => "")*"_cyc"*replace(string(cyc_vals[ja_idx]), "." => "")  

    # get moment targets and weight matrix
    @unpack data_mom, mom_key, K, W = moment_targets(dlw1_du = cyc)
    
    # Define the baseline values
    @unpack ρ, σ_ϵ, ι = model()
    param_vals        = OrderedDict{Symbol, Real}([ 
                        (:a, 1.0),           # effort 
                        (:ε, 0.5),           # ε
                        (:σ_η, 0.0),         # σ_η 
                        (:χ, 0.0),           # χ
                        (:γ, 0.4916),        # γ
                        (:hbar, 1.0),        # hbar
                        (:ρ, ρ),             # ρ
                        (:σ_ϵ, σ_ϵ),         # σ_ϵ
                        (:ι, ι) ])           # ι

    # Specifciations for the shocks in simulation
    @unpack P_z, p_z, z_ss_idx = model(ρ = param_vals[:ρ], σ_ϵ = param_vals[:σ_ϵ])
    shocks  = rand_shocks(P_z, p_z; N_sim_macro_workers = 1, z0_idx = z_ss_idx)

    # Parameters we will fix (if any) in ε, σ_η, χ, γ, hbar 
    params_fix   = [:hbar, :ρ, :σ_ϵ] 
    param_bounds = get_param_bounds()
    for p in params_fix
        delete!(param_bounds, p)
    end

    # Parameters that we will estimate
    J           = length(param_bounds)
    
    @assert(K >= J)

    param_est   = OrderedDict{Symbol, Int64}()
    for (i, dict) in enumerate(collect(param_bounds))
        local key = dict[1]
        param_est[key] = i
    end

    # Sample I Sobol vectors from the parameter space
    I_max        = 5*10^4
    lb           = zeros(J)
    ub           = zeros(J)

    for (key, value) in param_est
        lb[value]   = param_bounds[key][1]
        ub[value]   = param_bounds[key][2]
    end

    s            = SobolSeq(lb, ub)
    seq          = skip(s, 10000, exact = true)
    sob_seq      = reduce(hcat, next!(seq) for i = 1:I_max)
end

# Evaluate the objective function for each parameter vector
@time output = pmap(i -> objFunction(sob_seq[:,i], param_vals, param_est, shocks, data_mom, W), 1:I_max) 

# Kill the processes
#rmprocs(nprocs())

# Clean the output 

# Retain the valid vectors (i.e. solutions without flags)
N_old   = length(output)
indices = [output[i][3] == 0 for i = 1:N_old]
out_new = output[indices]
N       = length(out_new)

# Record the function values
fvals   = [out_new[i][1] for i = 1:N]
# Record the moments
moms    = reduce(hcat, out_new[i][2] for i = 1:N)'
# Record the parameters
pars    = sob_seq[:,indices]' 
# Record the IR flag
IR_flag = reduce(hcat, out_new[i][4] for i = 1:N)
# Record the IR flag
IR_err  = reduce(hcat, out_new[i][5] for i = 1:N)

# Save the output
save("../smm/jld/"*file*".jld2",  Dict("moms" => moms, "fvals" => fvals, "mom_key" => mom_key, "param_est" => param_est, "param_vals" => param_vals, 
                            "param_bounds" => param_bounds, "pars" => pars, "IR_flag" => IR_flag, "IR_err" => IR_err, "J" => J, "K" => K,
                            "W" => W, "data_mom" => data_mom, "fix_a" => false))
