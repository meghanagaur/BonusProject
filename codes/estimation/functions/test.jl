f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

lower = [1.25, -2.1]
upper = [Inf, Inf]
initial_x = [2.0, 2.0]

od = OnceDifferentiable(f, initial_x; autodiff = :forward)

results = Optim.optimize(od, lower, upper, initial_x, Fminbox(GradientDescent()))
results = Optim.optimize(od, lower, upper, initial_x, Fminbox())
VZresults = Optim.optimize(f, lower, upper, initial_x, Fminbox(LBFGS()))

function g!(storage, x)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
    end

    od = OnceDifferentiable(f, g!, initial_x)
    results = optimize(od, lower, upper,initial_x, Fminbox(GradientDescent()))

    results = optimize(od, lower, upper,initial_x, Fminbox(NelderMead()), Optim.Options(g_tol = 10^-5, x_tol = 10^-5,  f_tol = 10^-5, outer_iterations = 50, show_trace = true))

    https://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/#box-constrained-optimization

    https://discourse.julialang.org/t/optim-jl-do-all-methods-allow-box-constraints-should-all-work-without-them/10209/3