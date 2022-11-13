"""
Workhouse function for the global multistart optimization algorithm, loosely following
Guvenen et al (2019), with basic NM simplex algorithm for local optimization.
"""
function tiktak(init_points, file, init_x, param_bounds, param_vals, param_est, 
    shocks, data_mom, W, I_max; I_min  = 5, test = false, bounds = true)

    JJ          = length(param_vals)         # total num params (fixed + estimating)
    J           = length(param_bounds)       # num params we are estimating
    N_str       = size(init_points, 2)       # number of initial points
    output      = zeros(JJ+1, N_str)         # JJ + 1 x N_str, record function + param values

    @inbounds for i = 1:N_str

        println("----------------------------")
        println("GLOBAL ITERATION STEP: ", i)
        println("----------------------------")

        if i <= I_min

            start      = init_points[:,i]

        elseif i >  I_min

            cur_out    = readdlm(file, ',', Float64)                     # open current output across all jobs
            i_last     = size(cur_out, 1)                                # sum completed iterations across all workers       
            θ          = min( max(0.1, (i_last/I_max)^(1/2) ), 0.995 )   # updating parameter
            idx        = argmin(cur_out[:,1])                            # check for the lowest function value across processes 
            pstar      = cur_out[idx, 2:2+J-1]                           # get parameters 
            start      = @views (1-θ)*init_points[:,i] + θ*pstar         # set new start value for local optimization

            println("I_last: ", i_last)
            println("----------------------------")
        end

        if test == false
            
            # LOCAL OPTIMIZATION 
            if bounds == true

                opt       = optimize(x -> objFunction_WB(x, start, param_bounds, param_vals, param_est, shocks, data_mom, W)[1], init_x, NelderMead(), 
                            Optim.Options(g_tol = 1e-4, x_tol = 1e-4,  f_tol = 1e-4, iterations = 60, show_trace = true))

                arg_min_t = Optim.minimizer(opt)
               
                arg_min = zeros(length(arg_min_t))
                for (k, v) in param_est
                    arg_min[v]  = transform_params(arg_min_t[v], param_bounds[k], start[v])
                end

            else

                opt        = optimize(x -> objFunction(x, param_vals, param_est, shocks, data_mom, W)[1], start, 
                            NelderMead(), Optim.Options(g_tol = 1e-4, x_tol = 1e-4, f_tol = 1e-4, iterations = 60, show_trace = true))

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

