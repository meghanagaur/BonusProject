include("smm_settings.jl") # SMM settings and functions

println(Threads.nthreads())

# Initial parameter values
endogParams    = zeros(3) # zeros(4)
endogParams[1] = 0.5      # ε 
endogParams[2] = 0.05     # σ_η
endogParams[3] = 0.3      # χ
#endogParams[4] = 0.66    # γ

## evaluate the objective function 
init_x    = zeros(J)
@time out = objFunction_WB(init_x, endogParams, pb, zshocks, data_mom, W)
fval      = out[1]
mod_mom   = out[2]
flag      = out[3]

## test local optimization by setting the truth to the originally obtained model moments
endogParams2 = endogParams + rand(Normal(0, 0.005), J)               # add some noise
endogParams2 = [clamp(endogParams2[i], pb[i][1],pb[i][2]) for i=1:J] # make sure new initial guess lies within the bounds

# NM from Optim WITH bounds 
objFunc(x) = objFunction_WB(x, endogParams2, pb, zshocks, mod_mom, W)[1]
opt        = optimize(objFunc, init_x, NelderMead(), 
                    Optim.Options(g_tol = 1e-6, x_tol = 1e-6,  f_tol = 1e-6, iterations = 50, show_trace = true))

# rescales all of the parameters 
minimizer_t   = Optim.minimizer(opt)  # transformed
minimizer     = [ transform(minimizer_t[i], pb[i], endogParams2[i]) for i = 1:length(endogParams2) ] 
#orig          = [ transform(init_x[i], pb[i], endogParams2[i]) for i = 1:length(endogParams2) ] 

# save the results
save("jld/local_test_optim_bounds.jld2", Dict("min" =>  Optim.minimum(opt), "argmin" =>  minimizer,
                        "initial_x" =>   endogParams2, "mod_mom" => mod_mom,
                        "truth" => endogParams, "opt" => opt))

## NLOpt Routines