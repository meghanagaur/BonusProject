# Solve the model moments on a grid of ε and σ_η
using Distributed, SlurmClusterManager

cd(dirname(@__FILE__))
addprocs(SlurmManager())

file = "calibration_vary_sigma_hbar"

@everywhere begin
    
    include("functions/smm_settings.jl")        # SMM inputs, settings, packages, etc.
    include("functions/calibration_vary_z1.jl") # for computing dlogθ/dlogz

    # Build the grids 
    N_grid    = 50
    σ_η_grid  = LinRange(param_bounds[:σ_η][1], param_bounds[:σ_η][2], N_grid)
    hbar_grid = LinRange(param_bounds[:hbar][1], param_bounds[:hbar][2], N_grid)

end

# Stack the parameter vectors for parallel computation
par_grid = zeros(2, N_grid^2)
t = 1
@inbounds for j = 1:N_grid
    @inbounds for k = 1:N_grid
        par_grid[1,t] = σ_η_grid[j]
        par_grid[2,t] = hbar_grid[k]
        global t+=1
    end
end

# Save the output
@time output = pmap(i -> heatmap_moments(; σ_η = par_grid[1,i], hbar = par_grid[2,i]), 1:size(par_grid,2)) 

save("jld/"*file*".jld2", Dict("output" => output, "par_grid" => par_grid, "baseline_model" => model() ))


