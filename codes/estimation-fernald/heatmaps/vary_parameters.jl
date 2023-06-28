# Solve the model moments on a grid of ε and χ
using Distributed, SlurmClusterManager

cd(dirname(@__FILE__))

addprocs(SlurmManager())

# Load helper files
@everywhere include("../functions/smm_settings.jl")        # SMM inputs, settings, packages, etc.
@everywhere include("../functions/moments.jl")             # produce relevant model moments

# Get the baseline parameter values
param_vals  = OrderedDict{Symbol, Real}([ 
                (:ε, 2.3855),           # ε
                (:σ_η, 0.5357),         # σ_η 
                (:χ, 0.5159),           # χ
                (:γ, 0.4743) ])         # γ

# Get the parameter combination 
param_bounds = get_param_bounds()
parameters   = collect(keys(param_bounds))
combos       = collect(combinations(parameters, 2))
idx          = parse(Int64, ENV["SLURM_ARRAY_TASK_ID"])
combo        = combos[idx]

println("Vary: "*string(combo))

# Build the parameter grids 
par_1       = combo[1]
par_2       = combo[2]
N_grid      = 50
par_1_grid  = LinRange(param_bounds[par_1][1], param_bounds[par_1][2], N_grid)
par_2_grid  = LinRange(param_bounds[par_2][1], param_bounds[par_2][2], N_grid)

# Stack the parameter vectors for parallel computation
par_mat = zeros(2, N_grid^2)
t = 1
@inbounds for j = 1:N_grid
    @inbounds for k = 1:N_grid
        par_mat[1, t] = par_1_grid[j]
        par_mat[2, t] = par_2_grid[k]
        global t+=1
    end
end

# Solve the model for all parameter combinations 
@unpack P_z, p_z, z_ss_idx = model()
shocks       = rand_shocks(P_z, p_z; N_sim_macro_alp_workers = 1, z0_idx = z_ss_idx)
@time output = pmap(i -> heatmap_moments(par_mat[:,i], combo, param_vals, shocks), 1:size(par_mat, 2))

rmprocs(workers())

# Save the output
save("jld/"*"heatmap_vary_"*string(combo[1])*"_"*string(combo[2])* ".jld2", 
                                    Dict("output" => output, "par_mat" => par_mat, 
                                        "baseline_params" =>  param_vals, "combo" => combo))


