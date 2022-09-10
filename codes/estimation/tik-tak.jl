"""
Workhouse function for the global multistart optimization algorithm
"""
function tiktak_spmd(sob_d, fvals_d, argmin_d, min_p, argmin_p, iter_p, init_x, pb, zshocks, data_mom, W)
    
    #init_points = @fetchfrom 2 sob_d[:L] 
    init_points = sob_d[:L] 
    
    num_points  = size(init_points,1)
    θ           = 1
    I_max       = size(sob_d,1)
    
    iter_p[:L]   = 0
    I_min        = 5

    for i = 1:num_points
        println(i)  
        if i < I_min
            
            start     = init_points[i,:]
            
            #= LOCAL OPTIMIZATION
            opt       = optimize(x -> objFunction_WB(x, start, pb, zshocks, data_mom, W)[1], init_x, NelderMead(), 
                Optim.Options(g_tol = 1e-6, x_tol = 1e-6,  f_tol = 1e-6, iterations = 1))
            =#

            min_f     = 1            #Optim.minimum(opt)
            arg_min   = 3*ones(J)    #Optim.minimizer(opt)  # needs to be transformed
            argmin_t  = [ transform_params(arg_min[j], pb[j], start[j]) for j = 1:J ] 

            # record results in big vectors
            fvals_d[:L][i]    = copy(min_f)      # function value
            argmin_d[:L][i,:] = copy(argmin_t)  # arg min 
            # update the results in small vectors 
            if i ==1
                min_p[:L]        = copy(min_f)
                argmin_p[:L][:]  = copy(argmin_t)
            elseif i > 1
                if min_f < min_p[:L]  
                    min_p[:L]        = copy(min_f)
                    argmin_p[:L][:]  = copy(argmin_t)
                end
            end  
        elseif i >=  I_min

            i_last    = sum(iter_p)            
            θ         = min( max(0.1, (i_last/I_max)^(1/2) ), 0.995 ) 
            idx       = argmin(min_p) # check for the lowest function value

            start     = init_points[i,:]*θ + (1-θ)*argmin_p[idx,:]

            #=
            # LOCAL OPTIMIZATION
            opt       = optimize(x -> objFunction_WB(x, start, pb, zshocks, data_mom, W)[1], init_x, NelderMead(), 
            Optim.Options(g_tol = 1e-6, x_tol = 1e-6,  f_tol = 1e-6, iterations = 1))
            =#
            
            min_f     = 1           # Optim.minimum(opt)    
            arg_min   = 3*ones(J)   # Optim.minimizer(opt)  # needs to be transformed
            argmin_t  = [ transform_params(arg_min[j], pb[j], start[j]) for j = 1:J ] 
            
            # record results in big vectors
            fvals_d[:L][i]    = copy(min_f)      # function value
            argmin_d[:L][i,:] = copy(argmin_t)   # arg min 
            
            # update the results in small vectors 
            if min_f < min_p[:L]  
                min_p[:L]        = copy(min_f)
                argmin_p[:L][:]  = copy(argmin_t)
            end   
            
        end 

        iter_p[:L] += 1

    end 

end