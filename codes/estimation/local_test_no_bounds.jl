include("smm_settings.jl") # SMM inputs, settings, packages, etc.

println(Threads.nthreads())

# Initial parameter values
endogParams    = zeros(K) 
endogParams[1] = 0.5      # ε
endogParams[2] = 0.05     # σ_η
endogParams[3] = 0.3      # χ
endogParams[4] = 0.7      # γ

# evaluate the objective function 
@time out = objFunction(endogParams, param_bounds, shocks, data_mom, W)
fval      = out[1]
mod_mom   = out[2]
flag      = out[3]

## test local optimization by setting the truth to the originally obtained model moments
endogParams2 = endogParams + rand(Normal(0, 0.005), J)               # add some noise
endogParams2 = [clamp(endogParams2[i], param_bounds[i][1], param_bounds[i][2]) for i=1:J] # make sure new initial guess lies within the bounds

# NM from Optim WITHOUT bounds 
objFunc(x) = objFunction(x, param_bounds, shocks, mod_mom, W)[1]
@time opt  = optimize(objFunc, endogParams2, NelderMead(), 
                    Optim.Options(g_tol = 1e-6, x_tol = 1e-6, f_tol = 1e-6, iterations = 50, show_trace = true))

# save the results
save("jld/local_test_optim_no_bounds.jld2", Dict("min" =>  Optim.minimum(opt), "argmin" =>  Optim.minimizer(opt),
                        "initial_x" =>   endogParams2, "mod_mom" => mod_mom,
                        "truth" => endogParams, "opt" => opt))


