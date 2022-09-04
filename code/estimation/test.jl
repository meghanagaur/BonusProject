function test(xx)
    xx2          = 2 ./(1 .+ exp.(-(xx)) ) .- 1
    x1 = zeros(length(xx))
    for i =1:length(xx)
        if xx2[i] > 0
            x1[i] = xx2[i]*(3 - 2) .+ 2
        else
            x1[i] = xx2[i]*(2 - 1.5) .+ 2 
        end
    end
   return sum((x1 .- 1.7).^2)
 end


opt        = optimize(test, [0.0; 0.0], NelderMead(), Optim.Options(g_tol = 1e-10, iterations = 100,show_trace = true))
test(xx)

xx = Optim.minimizer(opt)
xx2          = 2 ./(1 .+ exp.(-(xx .- 0)./1) ) .- 1
x1 = zeros(length(xx))
for i =1:length(xx)
    if xx2[i] > 0
        x1[i] = xx2[i]*(3 - 2) .+ 2
    else
        x1[i] = xx2[i]*(2 - 1.5) .+ 2 
    end
end


function test(xx)
    return sum((xx .- 1.7).^2)
end

opt        = optimize(test, [1.5; 1.5], NelderMead(), Optim.Options(g_tol = 1e-10, iterations = 100,show_trace = true))
xx         = Optim.minimizer(opt)