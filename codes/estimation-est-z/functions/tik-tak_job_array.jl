"""
Workhouse function for the global multistart optimization algorithm, loosely following
Guvenen et al (2019), with NM simplex for local optimization step.
"""
function tiktak(init_points, file, init_x, param_bounds, param_vals, param_est, shocks, data_mom, W, I_max; 
    I_min  = 10, test = false, bounds = true, max_iter_1 = 50, max_iter_2 = 50, crit = 1e-4, manual_bounds = true)

    JJ          = length(param_vals)         # total num params (fixed + estimating)
    J           = length(param_bounds)       # num params we are estimating
    N_str       = size(init_points, 2)       # number of initial points
    output      = zeros(JJ+1, N_str)         # JJ + 1 x N_str, record function + param values

    if bounds == true
        lower, upper = get_bounds(param_est, param_bounds)
    end

    @inbounds for i = 1:N_str

        println("----------------------------")
        println("GLOBAL ITERATION STEP: ", i)
        println("----------------------------")

        if i <= I_min

            start      = init_points[:,i]
            max_iter   = max_iter_1

        elseif i >  I_min

            cur_out    = readdlm(file, ',', Float64)                     # open current output across all jobs
            i_last     = size(cur_out, 1)                                # sum completed iterations across all workers    
            θ          = min((i_last/I_max)^2, 0.995)                    # updating parameter
            #θ          = min( max(0.1, (i_last/I_max)^(1/2) ), 0.995 )  # updating parameter
            idx        = argmin(cur_out[:,1])                            # check for the lowest function value across processes 
            pstar      = cur_out[idx, 2:2+J-1]                           # get parameters 
            start      = @views (1-θ)*init_points[:,i] + θ*pstar         # set new start value for local optimization
            max_iter   = max_iter_2                                      # max iterations 

            println("I_last: ", i_last)
            println("----------------------------")
            
        end

        if test == false
            
            # LOCAL OPTIMIZATION WITH BOUNDS
            if bounds == true

                if manual_bounds == true
                
                    opt       = optimize(x -> objFunction_WB(x, start, param_bounds, param_vals, param_est, shocks, data_mom, W)[1], init_x, NelderMead(), 
                                Optim.Options(g_tol = crit, x_tol = crit,  f_tol = crit, iterations = max_iter, show_trace = true))

                elseif manual_bounds == false

                    opt       = optimize(x -> objFunction(x, param_vals, param_est, shocks, data_mom, W)[1], lower, upper, start, Fminbox(NelderMead()), 
                                Optim.Options(g_tol = crit, x_tol = crit,  f_tol = crit, iterations = max_iter, show_trace = true))
                
                end

                arg_min_t = Optim.minimizer(opt)
               
                arg_min   = zeros(length(arg_min_t))
                for (k, v) in param_est
                    arg_min[v]  = transform_params(arg_min_t[v], param_bounds[k], start[v])
                end

            # LOCAL OPTIMIZATION WITHOUT BOUNDS
            else

                opt        = optimize(x -> objFunction(x, param_vals, param_est, shocks, data_mom, W)[1], start, NelderMead(), 
                            Optim.Options(g_tol = crit, x_tol = crit, f_tol = crit, iterations = max_iter, show_trace = true))

                arg_min    = Optim.minimizer(opt)

            end

            min_f    = Optim.minimum(opt) 

        elseif test == true

            arg_min    = start
            min_f      = i

        end

        # add any fixed paramaters
        new_out = vcat(min_f, arg_min)
        for (k, v) in param_vals
            if !haskey(param_est, k)
                push!(new_out, v)
            end
        end

        # record results in .txt file
        output[:, i]   = new_out

        open(file, "a+") do io
            writedlm(io, new_out', ',')
        end;


    end 

    return output

end

