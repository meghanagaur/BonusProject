"""
Workhouse function for the global multistart optimization algorithm
"""
function tiktak_spmd(sobol, fvals_d, argmin_d, min_p, argmin_p, iter_p, init_x, param_bounds, zshocks, data_mom, W)
    
    #init_points = @fetchfrom 2 sob_d[:L] 
    #init_points = sob_d[:L] 
    #idx = @fetchfrom 2 myid()

    id          = myid() - 1 
    init_points = sobol[:,:,id]
    
    num_points  = size(init_points,2)
    θ           = 1
    I_max       = size(sobol,2)* size(sobol,3)
    I_min       = 1 # number of iterations before looking at past values
    iter_p[:L]  = 0

    @inbounds for i = 1:num_points
        if i <= I_min
            start     = init_points[:,i]
        elseif i >  I_min
            i_last    = sum(iter_p)            
            θ         = min( max(0.1, (i_last/I_max)^(1/2) ), 0.995 ) 
            idx       = argmin(min_p) # check for the lowest function value across processes 
            start     = init_points[:,i]*θ + (1-θ)*argmin_p[:,idx] # set new start value
        end
            
        #= LOCAL OPTIMIZATION
        opt       = optimize(x -> objFunction_WB(x, start, param_bounds, zshocks, data_mom, W)[1], init_x, NelderMead(), 
            Optim.Options(g_tol = 1e-6, x_tol = 1e-6,  f_tol = 1e-6, iterations = 1))
        =# 
        
        opt       = 0#objFunction(start, param_bounds, zshocks, data_mom, W)[1]
        min_f     = opt #Optim.minimum(opt)
        arg_min   = start #Optim.minimizer(opt)  # needs to be transformed
        argmin_t  = [ transform_params(arg_min[j], param_bounds[j], start[j]) for j = 1:length(param_bounds) ] 

        # record results in big vectors
        fvals_d[:L][i]    .= min_f     # function value
        argmin_d[:L][:,i] .= argmin_t  # arg min 

        # update the results in small vectors 
        if i == 1
            min_p[:L][:]     .= min_f
            argmin_p[:L][:]  .= argmin_t
        elseif i > 1
            if min_f < first(min_p[:L])
                min_p[:L][:]     .= min_f
                argmin_p[:L][:]  .= argmin_t
            end
        end  

        iter_p[:L] .+= 1
    
    end 

end