include("smm_settings.jl") # SMM inputs, settings, packages, etc.

println(Threads.nthreads())

# Initial parameter values
endogParams    = zeros(K) 
endogParams[1] = 0.5      # ε 
endogParams[2] = 0.05     # σ_η
endogParams[3] = 0.3      # χ
endogParams[4] = 0.66     # γ

## evaluate the objective function 
init_x    = zeros(J)

# NM from Optim WITH bounds 
objFunc(x) = objFunction_WB(x, endogParams, param_bounds, shocks, data_mom, W)[1]
@time opt  = optimize(objFunc, init_x, NelderMead(), 
                    Optim.Options(g_tol = 1e-6, x_tol = 1e-6,  f_tol = 1e-6, iterations = 50, show_trace = true))

# rescales all of the parameters 
minimizer_t   = Optim.minimizer(opt)  # transformed
minimizer     = [ transform_params(minimizer_t[i], param_bounds[i], endogParams[i]) for i = 1:length(endogParams) ] 

# save the results
save("jld/local_NM_optim_bounds.jld2", Dict("min" =>  Optim.minimum(opt), "argmin" =>  minimizer,
                        "initial_x" =>   endogParams, "opt" => opt))

