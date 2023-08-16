"""
Workhouse function for TikTak, a global multistart optimization algorithm, loosely following
Guvenen et al (2019). Derivative-free local optimization.
"""
function tiktak(init_points, file, param_bounds, param_vals, param_est, shocks, data_mom, W, I_max; test = false,
    I_min  = 100, max_iters = 60, crit_1 = 1e-4, crit_2 = 1e-8, opt_1 = nothing, opt_2  = nothing, switch_opt = 0.7, fix_a = false)

    JJ          = length(param_vals)         # total num params (fixed + estimating)
    J           = length(param_bounds)       # num params we are estimating
    N_str       = size(init_points, 2)       # number of initial points
    output      = zeros(JJ+1, N_str)         # JJ + 1 x N_str, record function + param values

    @inbounds for i = 1:N_str

        println("----------------------------")
        println("PROCESS SEARCH ITERATION: ", i)
        println("----------------------------")

        if i < 2 

            start      = init_points[:,i]
            i_last     = i
        
        elseif   i >= 2 
            
            # read current output
            cur_out    = readdlm(file, ',', Float64)                          # open current output across all jobs
            i_last     = size(cur_out, 1)                                     # sum completed iterations across all workers  
                    
            if i_last <= I_min

                start      = init_points[:,i]       

            else

                println("----------------------------")
                println("GLOBAL SEARCH ITERATION: ", i_last)
                println("----------------------------")
                
                θ          = min((i_last/I_max)^2, 0.995)                     # updating parameter: convex updating parameter
                #θ          = min( max(0.1, (i_last/I_max)^(1/2) ), 0.995 )   # updating parameter: concave updating parameter
                idx        = argmin(cur_out[:,1])                             # check for the lowest function value across processes 
                pstar      = cur_out[idx, 2:2+J-1]                            # get parameters 
                start      = @views (1-θ)*init_points[:,i] + θ*pstar          # set new start value for local optimization

            end            

        end

        if test == false
            
            # Local NM-Simplex algorithm with manually enforced bound constraints via logistic transformation
            if (   (isnothing(opt_1) && i_last/I_max <= switch_opt) || (i_last/I_max > switch_opt && isnothing(opt_2)) )

                crit            = i_last/I_max > switch_opt ? crit_2 : crit_1
                optim           = Optim.optimize(x -> objFunction_WB(x, start, param_bounds, param_vals, param_est, shocks, data_mom, W; fix_a = fix_a)[1], 
                                    zeros(J), NelderMead(), Optim.Options(g_tol = crit, f_tol = crit, x_tol = crit, iterations = max_iters))
                arg_min_t       = Optim.minimizer(optim)
                min_f           = Optim.minimum(optim) 
                arg_min         = zeros(length(arg_min_t))

                println(optim)

                # transform the parameters back into the parameter space
                for (k, v) in param_est
                    arg_min[v]  = transform_params(arg_min_t[v], param_bounds[k], start[v])
                end

            # Local optimization using NLopt algorithm/settings 
            else 
                
                if (!isnothing(opt_1) && i_last/I_max <= switch_opt) 

                    (min_f, arg_min, ret) = NLopt.optimize(opt_1, start)
                
                elseif (!isnothing(opt_2) && i_last/I_max > switch_opt) 

                    (min_f, arg_min, ret) = NLopt.optimize(opt_2, start)
                end

                println("minimum:\t\t"*string(min_f))
                println("minimizer:\t\t"*string(arg_min))
                println("reason for stopping:\t"*string(ret))
            end

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

