# Solve the model moments on a grid of ε and σ_η
using Distributed, SlurmClusterManager
addprocs(SlurmManager())

@everywhere begin
    include("smm_settings.jl") # SMM inputs, settings, packages, etc.

    # Build the grids 
    N_grid   = 100
    ε_grid   = LinRange(param_bounds[1][1],param_bounds[1][2],N_grid)
    σ_η_grid = LinRange(param_bounds[2][1],param_bounds[2][2],N_grid)

    # Simulate moments over 
    function simulate_moments(xx)
        baseline = model(ε = xx[1] , σ_η = xx[2], χ = 0, γ = .66) 
        out      = simulate(baseline, shocks)
        mod_mom  = [out.std_Δlw, out.dlw1_du, out.dly_dlw, out.u_ss, out.dW_du, out.da_du]
        flag     = out.flag
        return [mod_mom, flag]
    end
end

# Stack the parameter vectors for parallel computation
par_grid = zeros(2, N_grid^2)
t = 1
@inbounds for i =1:N_grid
    @inbounds for j = 1:N_grid
        par_grid[1,t] = ε_grid[i]
        par_grid[2,t] = σ_η_grid[j]
        t+=1
    end
end

# Save the output
@time output = pmap(i -> simulate_moments(par_grid[:,i]), 1:size(par_grid,2)) 
save("jld/calibration_data.jld2", Dict("output" => output, "par_grid" => par_grid, "baseline_model" => model() ))


