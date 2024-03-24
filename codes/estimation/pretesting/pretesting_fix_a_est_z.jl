using Distributed, SlurmClusterManager
cd(dirname(@__FILE__))

# Start the worker processes
addprocs(SlurmManager())
println(nprocs())

# File location for saving jld output + slurm idx
@everywhere cyc             = 1.0
file                        = "pretesting_fix_a_bwc"*replace(string(cyc), "." => "")
fix_wages                   = true
pv                          = true        
@everywhere hm              = true
@everywhere output_target   = "alp" # "alp", can be anything if est_z = false 

# Update file name
file            = fix_wages ? file*"_fix_wages" : file 
file            = pv ? file*"_pv" : file 
file            = file*"_est_z"
file            = file*"_"*output_target
file            = hm ? file*"_hm" : file 

# Load SMM inputs, settings, packages, etc.
@everywhere include("../functions/smm_settings.jl") 

@everywhere begin

    # get moment targets and weight matrix
    est_mom = Dict(:dlw1_du => true, :u_ss => true) 

    if hm == true
        est_mom[:dlw_dlp] = true
        est_mom[:dlw1_du] = false
    end
    if output_target == "gdp"
        est_mom[:y_ρ] = true 
        est_mom[:y_σ] = true 
    elseif output_target == "alp"
        est_mom[:p_ρ] = true 
        est_mom[:p_σ] = true 
    end

    @unpack data_mom, mom_key, K, W = moment_targets(dlw1_du = -cyc; est_mom = est_mom)

    # Define the baseline values
    @unpack ρ, σ_ϵ, ι, P_z, z_ss_idx, ε, χ, γ, σ_η, hbar = model()
    param_vals        = OrderedDict{Symbol, Real}([ 
                        (:a, 1.0),           # effort 
                        (:ε,   ε),           # ε
                        (:σ_η, σ_η),         # σ_η 
                        (:χ, χ),             # χ
                        (:γ, γ),             # γ
                        (:hbar, hbar),       # hbar
                        (:ρ, ρ),             # ρ
                        (:σ_ϵ, σ_ϵ),         # σ_ϵ
                        (:ι, ι) ])           # ι

    # Draw shocks
    shocks            = drawShocksEstZ(; fix_a = true)  

    # Parameters we will fix (if any) in: ε, σ_η, χ, γ, ρ, σ_ϵ
    params_fix   = [:hbar, :ε, :σ_η] 
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
    I_max        = Int64(2*10^4)
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
@time output = pmap(i -> objFunction(sob_seq[:,i], param_vals, param_est, shocks, data_mom, W; 
                fix_a = true, fix_wages = fix_wages, pv = pv, est_z = true), 1:I_max) 

# Kill the processes
rmprocs(workers())

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
                            "W" => W, "data_mom" => data_mom, "fix_a" => true, "fix_wages" => fix_wages, "pv" => pv, 
                            "est_z" => true, "output_target" => output_target))
