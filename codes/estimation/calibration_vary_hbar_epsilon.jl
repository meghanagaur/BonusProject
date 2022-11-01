# Solve the model moments on a grid of ε and σ_η
using Distributed, SlurmClusterManager
addprocs(SlurmManager())
#addprocs(2)

@everywhere begin
    include("smm_settings.jl") # SMM inputs, settings, packages, etc.

    # Build the grids 
    N_grid    = 50
    ε_grid    = LinRange(0.05, 2, N_grid)
    hbar_grid = LinRange(param_bounds[4][1], param_bounds[4][2], N_grid)

    # Simulate moments  
    function simulate_moments(xx)
        
        baseline = model(ε = xx[1], hbar = xx[2], γ = 0.4,  χ = 0) 
        out      = simulate(baseline, shocks)
        mod_mom  = [out.std_Δlw, out.dlw1_du, out.dlw_dly, out.u_ss] #, out.avg_Δlw,
        #out.dlw1_dlz, out.dlY_dlz, out.dlu_dlz, out.std_u, out.std_z, out.std_Y, out.std_w0]

        flag     = out.flag
        flag_IR  = out.flag_IR
        IR_err   = out.IR_err

        # Extra moments (check)
        dlw_dly_2  = out.dlw_dly_2
        u_ss_2     = out.u_ss_2

        return [mod_mom, flag, flag_IR, IR_err, dlw_dly_2, u_ss_2]
    end

end

# Stack the parameter vectors for parallel computation
par_grid = zeros(2, N_grid^2)
t = 1
@inbounds for j = 1:N_grid
    @inbounds for k = 1:N_grid
        par_grid[1,t] = ε_grid[j]
        par_grid[2,t] = hbar_grid[k]
        global t+=1
    end
end

# Save the output
@time output = pmap(i -> simulate_moments(par_grid[:,i]), 1:size(par_grid,2)) 
save("jld/calibration_data_vary_hbar_epsilon.jld2", Dict("output" => output, "par_grid" => par_grid, "baseline_model" => model() ))


