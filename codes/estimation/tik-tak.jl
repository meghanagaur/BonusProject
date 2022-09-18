"""
Workhouse function for the global multistart optimization algorithm
"""
function tiktak_spmd(sobol, fvals_d, argmin_d, min_p, argmin_p, iter_p, 
    init_x, param_bounds, zshocks, data_mom, W; I_min  = 1, real = true, bounds = true)
    
    #init_points = @fetchfrom 2 sob_d[:L] 
    #init_points = sob_d[:L] 
    #idx = @fetchfrom 2 myid()
    id          = myid() - 1 
    N_str       = size(sobol,2)
    I_max       = N_str*size(sobol,3)
    init_points = sobol[:,:,id]
    iter_p[:L]  = 0

    @inbounds for i = 1:N_str
        if i <= I_min
            start     = init_points[:,i]
        elseif i >  I_min
            i_last    = sum(iter_p)            
            θ         = min( max(0.1, (i_last/I_max)^(1/2) ), 0.995 ) 
            idx       = argmin(min_p) # check for the lowest function value across processes 
            start     = @views (1-θ)*init_points[:,i] + θ*argmin_p[:,idx] # set new start value
        end
            
        if real == true
            # LOCAL OPTIMIZATION STEP
            if bounds == true
                opt       = optimize(x -> objFunction_WB(x, start, param_bounds, zshocks, data_mom, W)[1], init_x, NelderMead(), 
                    Optim.Options(g_tol = 1e-5, x_tol = 1e-5,  f_tol = 1e-5, iterations = 50))
                
                arg_min_t = Optim.minimizer(opt)
                arg_min   = [ transform_params(arg_min_t[j], param_bounds[j], start[j]) for j = 1:length(param_bounds) ] 
            else
                opt       = optimize(x -> objFunction(x, param_bounds, zshocks, mod_mom, W)[1], start, NelderMead(), 
                        Optim.Options(g_tol = 1e-5, x_tol = 1e-5, f_tol = 1e-5, iterations = 50))
                arg_min   = Optim.minimizer(opt)
            end
            min_f         = Optim.minimum(opt) 
        else
            arg_min  = start
            min_f    = id == 2 ? 0 : 3 
        end

        # record results in big vectors
        fvals_d[:L][i]      = min_f      # function value
        argmin_d[:L][:,i]  .= arg_min    # arg min 

        # update the results in small vectors 
        if i == 1
            min_p[:L][:]      .= min_f
            argmin_p[:L][:]   .= arg_min
        elseif i > 1
            if min_f < first(min_p[:L])
                min_p[:L][:]    .= min_f
                argmin_p[:L][:] .= arg_min
            end
        end  

        iter_p[:L] += 1
    
    end 

end
