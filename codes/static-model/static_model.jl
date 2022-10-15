# Solve the simplest static model (first pass)
cd("/Users/meghanagaur/Dropbox (Princeton)/OptimalContracts_Search/Programs/static-model")

using Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =15, yguidefontsize=15, xtickfontsize=13, ytickfontsize=13,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 9)

using LinearAlgebra, Distributions, Random, Interpolations, ForwardDiff,
BenchmarkTools, LaTeXStrings, Parameters, StatsBase, Parameters

"""
Solve static model with unit mass of workers, exponential utility;
optimal linear contract; y = az + η, where η ∼ N(0, σ^2).
ν = matching elasticity
ψ = matching efficiency 
r = risk aversion
ϕ = disutility of effort
z = labor productivity
b = worker's value from unemployment benefit
σ = st dev of η distribution
κ = vacancy-posting cost
Default parameters from Shimer (2005) (exc. r, ϕ, σ) 
"""
function staticModel(; z = 1, ν = 0.72, ψ = 0.9, κ = 0.213, 
    ϕ = 1, r = 0.8, b = 0.4, σ = 0.2)

    β = z^2 / (z^2 + ϕ*r*σ^2)           # optimal contract
    α = b + β^2 * (ϕ*r*σ^2 - z^2 )/(2ϕ) # optimal contract
    a = β*z/ϕ                           # optimal effort
    y = a*z                             # E[output]
    w = α + β*y                         # E[wage] (optimal linear contract)
    J = y - w                           # E[profit of filled vacancy]
    v = (ψ*J/κ)^(1/ν)                   # vacancies (given by free entry condition)
    f = ψ*v^(1 - ν)                     # job-finding rate
    q = ψ*v^(-ν)                        # job-filling rate
    n = copy(f)                         # employment

    return (α = α, β = β, a = a, w = w, y = y, J = J, f = f,
            q = q, v = v, n = n, ν = ν, ψ = ψ, κ = κ, ϕ = ϕ, 
            z = z, r = r, b = b, σ = σ)
end

# Compute various eq objects as a function of z
W(x)   =  staticModel(z = x).w
A(x)   =  staticModel(z = x).a
Y(x)   =  staticModel(z = x).y
N(x)   =  staticModel(z = x).n
JJ(x)  =  staticModel(z = x).J
W(x)   =  staticModel(z = x).w

# For plotting -- results are quite sensitive to parameter choices
minz  = 0.95 # otherwise, E[profit/worker] < 0
maxz  = 1.05 # plot until n ≈ 1

# Plot E[profit/filled vacancy] as a function of z
plot(JJ, minz, maxz, legend = false)
plot(N, minz, maxz, legend = false)
ylabel!(L"J")
xlabel!(L"z")

# Plot employment as a function of z
plot(N, minz, maxz, legend = false)
ylabel!(L"n")
xlabel!(L"z")

# Plot log employment against log z
plot(x -> log(N.(exp(x))), log(minz), log(maxz), legend = false)
ylabel!(L"\log n")
xlabel!(L"\log z")

""" 
Compute dlogn/dlog z, following analytical formula.
"""
function derivAnalytical(x)

    @unpack ν, w, y, a, J, r, σ, ϕ  = staticModel(z = x)

    extra_term = (x^3)*(r*σ^2)/(x^2 + ϕ*r*σ^2)^2
    return ((1 - ν)*ν^(-1))*(a + extra_term)*x/J

    # should match the above
    #g  = ForwardDiff.derivative(JJ, x) 
    #return ((1 - ν)*ν^(-1))*g*x/JJ(x)

    # expressions from static_model.pdf
    #return ((1 - ν)*ν^(-1))*a*x/JJ(x)
    #return ((1 - ν)*ν^(-1))/(1 - w/y) 
end

"""
Compute dlogn/dlog z, using numerical derivative.
"""
function derivNumerical(x)
    g = ForwardDiff.derivative(N, x)  
    return g*x/N(x)
end 

# Compute first order approximation around z = z0
z0   = 1
δ    = 0.05

# Update plotting window
minz  = z0 - δ
maxz  = z0 + δ

# Plot the comparison
plot(derivAnalytical, minz, maxz, legend =:topright, label = "Analytical Elasticity")
plot!(derivNumerical, minz, maxz, label = "Numerical Elasticity")
ylabel!(L"d \log n/d \log z")
xlabel!(L"z")

#= Plot the first order approximations

# n as a function of z -- first order approx
firstOrderApprox(z, z0)     = N(z0) + derivAnalytical(z0)*(N(z0)/z0)*(z - z0)

# log n as a function of z -- first order approx 
firstOrderApproxLogs(z, z0) = log(N(z0)) + (derivAnalytical(z0))*(z - log(z0))

# Plot the first order approximation -- on n, z space
plot(x -> N.(x), minz, maxz, legend =:bottomright, label= "Actual")
plot!(x-> firstOrderApprox(x, z0), minz, maxz, label = "First Order Approximation")
ylabel!(L"n")
xlabel!(L"z")

# Plot the first order approximation -- in logn, logz space
plot(x -> log(N.(exp(x))), log(minz), log(maxz), legend =:bottomright, label= "Actual")
plot!(x-> firstOrderApproxLogs(x, z0), log(minz), log(maxz), label = "First Order Approximation")
ylabel!(L"\log n")
xlabel!(L"\log z")
=#

"""
Compute J, N, D as a function of z = z, holding fixed w at w*(x0), a at a*(x0)
"""
function globalApproxFixed(x, x0)

    @unpack ν, w, a, ψ, κ = staticModel(z = x0) # solve model at z = x0

    y = a*x
    J = y - w           # get profits at z = x, holding fixed w, a at z = x0
    v = (ψ*J/κ)^(1/ν)   # get vacancies from free entry condition
    n = ψ*v^(1 - ν)     # get job-finding/employment
   
    d = ((1 - ν)*ν^(-1))/(1 - w/y) # emp fluctuations

    return (J = J, n = n, d = d, w = w, a = a)
end

"""
Compute J, N, D as a function of z,  holding fixed effort a at a*(z0)
Set w = Nash Bargained wage for z = x, where β = worker's bargaining power.
h(a) = a shift term captures disutility of effort a in worker's surplus
"""
function globalApproxNash(x, x0; β = staticModel().ν)
   
    @unpack ν, a, ψ, κ, b, w       = staticModel(z = x0) # solve model at z = x0
    
    wN(x) = β*a*x + (1 - β)*(b)   # compute Nash-Bargained wage at y = a*(x0)x

    y = a*x                       # E[output]
    w = wN(x) + (w  - wN(x0))     # add a shift term (for now) to capture disuility of a(x0)
    J = y - w                     # get profits at z = x, holding fixed a
    v = (ψ*J/κ)^(1/ν)             # get vacancies from free entry condition
    n = ψ*v^(1 - ν)               # get job-finding/employment
    
    d = ((1 - ν)*ν^(-1))*(1-β)*a*x/J # emp fluctuations

    return (J = J, n = n, d = d)
end

"""
Compute J, N, D as a function of z = z, holding fixed a at a*(x0)
"""
function globalApproxFixedA(x, x0)
    @unpack ν, a, ψ, κ, ϕ, r, σ = staticModel(z = x0) # solve model at z = x0
    @unpack w                   = staticModel(z = x) # solve model at z = x
    
    y = a*x             # get output
    J = y - w           # get profits at z = x, holding fixed a at x0
    v = (ψ*J/κ)^(1/ν)   # get vacancies from free entry condition
    n = ψ*v^(1 - ν)     # get job-finding/employment

    dw_dz = (x^5 + (x^3)*2ϕ*r*σ^2)/((ϕ)*(x^2 +ϕ*r*σ^2)^2)
    d     = ((1 - ν)*ν^(-1))*(a - dw_dz)*x/J # emp fluctuations

    return (J = J, n = n, d = d)
end

"""
Compute J, N, D as a function of z = z, holding fixed w at w*(x0)
"""
function globalApproxFixedW(x, x0)
    @unpack ν, w, ψ, κ, ϕ, σ, r = staticModel(z = x0) # solve model at z = x0
    @unpack a                   = staticModel(z = x)  # solve model at z = x

    y = a*x             # get output
    J = y - w           # get profits at z = x, holding fixed w at x0
    v = (ψ*J/κ)^(1/ν)   # get vacancies from free entry condition
    n = ψ*v^(1 - ν)     # get job-finding/employment

    da_dz = (x^4 + (x^2)*3ϕ*r*σ^2)/((ϕ)*(x^2 +ϕ*r*σ^2)^2)
    d      = ((1 - ν)*ν^(-1))*(a + x*da_dz)*x/J # emp fluctuations

    return (J = J, n = n, d = d)
end

# Update plotting window -- decrease δ
δ     = 0.01
minz  = z0 - δ
maxz  = z0 + δ

# Plot expected profit / filled vacancy as a function of z for the different models
plot(x -> globalApproxNash(x, z0).J, minz, maxz, label = "Shimer: Nash wage, Fixed a", linecolor=:black)
plot!(x -> globalApproxFixed(x, z0).J, minz, maxz, legend =:topleft, label = "Hall: Fixed w and a", linecolor=:blue)
plot!(x -> globalApproxFixedA(x, z0).J, minz, maxz, label = "Fixed a, Variable w", linecolor=:green)
plot!(x -> globalApproxFixedW(x, z0).J, minz, maxz, label = "Fixed w, Variable a", linecolor=:cyan)
plot!(x -> JJ(x), minz, maxz, label = "Bonus Economy: Variable w and a", linecolor=:red)
plot!(x -> globalApproxFixedW(x, z0).J + globalApproxFixedA(x, z0).J -  globalApproxFixed(x, z0).J, minz, maxz, 
label = "Fixed w + Fixed a - Fixed w and a (= Bonus)", linecolor=:magenta)
ylabel!(L"J")
xlabel!(L"z")
savefig("static-figs/J_diff_models.pdf")

# Plot n as a function of z for the different models
plot(x -> globalApproxNash(x, z0).n, minz, maxz, label = "Shimer: Nash wage, Fixed a", linecolor=:black)
plot!(x -> globalApproxFixed(x, z0).n, minz, maxz, legend =:topleft, label = "Hall: Fixed w and a", linecolor=:blue)
plot!(x -> globalApproxFixedA(x, z0).n, minz, maxz, label = "Fixed a, Variable w", linecolor=:green)
plot!(x -> globalApproxFixedW(x, z0).n, minz, maxz, label = "Fixed w, Variable a", linecolor=:cyan)
plot!(x -> N(x), minz, maxz, label = "Bonus Economy: Variable w and a", linecolor=:red)
ylabel!(L"n")
xlabel!(L"z")
savefig("static-figs/n_diff_models.pdf")

# Compute the numerical counterparts for the above elasticities
function derivNumericalFixed(x, x0)
    NN(x) = globalApproxFixed(x, x0).n
    g     = ForwardDiff.derivative(NN, x)  
    return g*x/NN(x)
end 

function derivNumericalNash(x, x0)
    NN(x) = globalApproxNash(x, x0).n
    g     = ForwardDiff.derivative(NN, x)  
    return g*x/NN(x)
end 

function derivNumericalFixedA(x, x0)
    NN(x) = globalApproxFixedA(x, x0).n
    g     = ForwardDiff.derivative(NN, x)  
    return g*x/NN(x)
end 

function derivNumericalFixedW(x, x0)
    NN(x) = globalApproxFixedW(x, x0).n
    g     = ForwardDiff.derivative(NN, x)  
    return g*x/NN(x)
end 

# Plot the numerical dlogn/dlogz as a function of z for the different models
plot(x -> derivNumericalNash(x, z0), minz, maxz, label = "Shimer: Nash wage, Fixed a", linecolor=:black, legend =:topright)
plot!(x -> derivNumericalFixed(x, z0), minz, maxz, label = "Hall: Fixed w and a", linecolor=:blue)
plot!(x -> derivNumericalFixedA(x, z0), minz, maxz, label = "Fixed a, Variable w", linecolor=:green)
plot!(x -> derivNumericalFixedW(x, z0), minz, maxz, label = "Fixed w, Variable a", linecolor=:cyan)
plot!(derivNumerical, minz, maxz, label = "Bonus Economy: Variable w and a", linecolor=:red)
ylabel!(L"d \log n/d \log z")
xlabel!(L"z")
savefig("static-figs/dlogn_dlogz_diff_models_numerical.pdf")

# Plot the analytical dlogn/dlogz as a function of z for the different models
plot(x -> globalApproxNash(x, z0).d, minz, maxz, label = "Shimer: Nash wage, Fixed a", linecolor=:black, legend = :topright)
plot!(x -> globalApproxFixed(x, z0).d, minz, maxz, label = "Hall: Fixed w and a", linecolor=:blue)
plot!(x -> globalApproxFixedA(x, z0).d, minz, maxz, label = "Fixed a, Variable w", linecolor=:green)
plot!(x -> globalApproxFixedW(x, z0).d, minz, maxz, label = "Fixed w, Variable a", linecolor=:cyan)
plot!(derivAnalytical, minz, maxz, label = "Bonus Economy: Variable w and a", linecolor=:red)
ylabel!(L"d \log n/d \log z")
xlabel!(L"z")
savefig("static-figs/dlogn_dlogz_diff_models_analytical.pdf")

## Sequential figures for slides

#str = ["Hall: Fixed w and a"; "Fixed a, Variable w"; "Fixed w, Variable a"; "Bonus Economy: Variable w and a"]
#str = ["Hall: Fixed w and a"; "Bonus Economy: Variable w and a"]
str  = ["Rigid Wage: Fixed w and a"; "Incentive Pay: Variable w and a"]

#col = [:blue, :green, :cyan, :red]
col = [:blue, :red]

# Plot J as a function of z for the different models
#F   = [x -> globalApproxFixed(x, z0).J ; x -> globalApproxFixedA(x, z0).J ; x -> globalApproxFixedW(x, z0).J ; JJ]
F   = [x -> globalApproxFixed(x, z0).J; JJ]

for i = 1:length(F)
#for i = 0:length(F)
    #local p1 = plot(x -> globalApproxNash(x, z0).J, minz, maxz, label = "Shimer: Nash wage, Fixed a", legend=:topleft, linecolor = :black)
    local p1 = plot()
    ylabel!(L"J")
    xlabel!(L"z")
    local ymin = globalApproxFixedW(minz, z0).J
    local ymax = globalApproxFixedW(maxz, z0).J
    ylims!(p1,(ymin, ymax))
    #=if i == 0
       savefig("static-figs/slides/J_0.pdf")
    end =#
    for j = 1:i 
        plot!(p1, F[j], minz, maxz, label = str[j], linecolor = col[j])
    end
    savefig("static-figs/slides/J_"*string(i)*".pdf")
end

# Plot n as a function of z for the different models
#F   = [x -> globalApproxFixed(x, z0).n ; x -> globalApproxFixedA(x, z0).n ; x -> globalApproxFixedW(x, z0).n ;  N]
F   = [x -> globalApproxFixed(x, z0).n  ; N]

for i = 1:length(F)
#for i = 0:length(F)
    #local p1 = plot(x -> globalApproxNash(x, z0).n, minz, maxz, label = "Shimer: Nash wage, Fixed a", legend=:topleft, linecolor = :black)
    local p1 = plot()
    ylabel!(L"n")
    xlabel!(L"z")
    local ymin = globalApproxFixedW(minz, z0).n
    local ymax = globalApproxFixedW(maxz, z0).n
    ylims!(p1,(ymin, ymax))
   #=if i == 0
        savefig("static-figs/slides/n_0.pdf")
    end =#
    for j = 1:i 
        plot!(p1, F[j], minz, maxz, label = str[j], linecolor = col[j])
    end
    savefig("static-figs/slides/n_"*string(i)*".pdf")
end


# Plot dlogn/dlogz as a function of z for the different models
#F   = [x -> globalApproxFixed(x, z0).d ; x -> globalApproxFixedA(x, z0).d ; x -> globalApproxFixedW(x, z0).d ; derivAnalytical]
F   = [x -> globalApproxFixed(x, z0).d;  derivAnalytical]

for i = 1:length(F)
#for i = 0:length(F)
    #local p1 = plot(x -> globalApproxNash(x, z0).d, minz, maxz, label = "Shimer: Nash wage, Fixed a", legend=:topright, linecolor = :black)
    local p1 = plot()
    ylabel!(L"d \log n/d \log z")
    xlabel!(L"z")
    local ymin = globalApproxFixedA(maxz, z0).d - 0.2
    local ymax = globalApproxFixedW(minz, z0).d + 0.2
    ylims!(p1,(ymin, ymax))
    #= if i == 0
        savefig("static-figs/slides/dlogn_dlogz_0.pdf")
    end=#
    for j = 1:i 
        plot!(p1, F[j], minz, maxz, label = str[j], linecolor = col[j])
    end
    savefig("static-figs/slides/dlogn_dlogz_"*string(i)*".pdf")
end

#Plot with just Hall and Bonus; Update plotting window -- increase δ
δ     = 0.1
minz  = z0 - δ
maxz  = z0 + δ


rigid   = "Rigid Wage: Fixed w and a"
perfpay = "Incentive Pay: Variable w and a"

plot(x -> globalApproxFixed(x, z0).J, minz, maxz, legend =:topleft, label = rigid, linecolor=:blue)
plot!(x -> JJ(x), minz, maxz, label = perfpay, linecolor=:red)
ylabel!(L"J")
xlabel!(L"z")
savefig("static-figs/slides/J_Hall_vs_Bonus.pdf")

plot(x -> globalApproxFixed(x, z0).n, minz, maxz, legend =:topleft, label = rigid, linecolor=:blue)
plot!(x -> N(x), minz, maxz, label = perfpay, linecolor=:red)
ylabel!(L"n(z)")
xlabel!(L"z")
savefig("static-figs/slides/n_Hall_vs_Bonus.pdf")

plot(x -> globalApproxFixed(x, z0).w, minz, maxz, legend =:topleft, label = rigid, linecolor=:blue)
plot!(x -> W(x), minz, maxz, label = perfpay, linecolor=:red)
ylabel!(L"\mathbb{E}[w|z]")
xlabel!(L"z")
savefig("static-figs/slides/w_Hall_vs_Bonus.pdf")

plot(x -> globalApproxFixed(x, z0).a, minz, maxz, legend =:topleft, label = rigid, linecolor=:blue)
plot!(x -> A(x), minz, maxz, label = perfpay, linecolor=:red)
ylabel!(L"a(z)")
xlabel!(L"z")
savefig("static-figs/slides/a_Hall_vs_Bonus.pdf")

δ     = 0.05
minz  = z0 - δ
maxz  = z0 + δ

plot(x ->  globalApproxFixed(x, z0).d, minz, maxz, legend =:topleft, label = "Hall: Fixed w and a", linecolor=:blue)
plot!(derivAnalytical, minz, maxz, label = perfpay, linecolor=:red)
ylabel!(L"d \log n/d \log z")
xlabel!(L"z")
savefig("static-figs/slides/dlogn_dlogz_Hall_vs_Bonus.pdf")

#= Compare the analytical and numerical elasticities

δ     = 0.01
minz  = z0 - δ
maxz  = z0 + δ

# fixed wage and effort
plot(x -> globalApproxFixed(x, z0).d, minz, maxz, legend =:topright, label = "Fixed w, a", linecolor=:blue)
plot!(x -> derivNumericalFixed(x, z0), minz, maxz, legend =:topright, label = "Fixed w, a", linecolor=:yellow)

# Nash wage with fixed effort
plot(x -> globalApproxNash(x, z0).d, minz, maxz, label = "Nash wage, Fixed a", linecolor=:black)
plot!(x -> derivNumericalNash(x, z0), minz, maxz, label = "Nash wage, Fixed a", linecolor=:yellow)

# fixed effort
plot(x -> globalApproxFixedA(x, z0).d, minz, maxz, label = "Fixed a", linecolor=:green)
plot!(x -> derivNumericalFixedA(x, z0), minz, maxz, label = "Fixed a", linecolor=:yellow)

plot(x -> globalApproxFixedW(x, z0).d, minz, maxz, label = "Fixed w", linecolor=:red)
plot!(x -> derivNumericalFixedW(x, z0), minz, maxz, label = "Fixed w", linecolor=:yellow)

plot(derivAnalytical, minz, maxz, label ="Flexible w, a", linecolor=:cyan)
plot!(derivNumerical, minz, maxz, label = "Flexible w, a", linecolor=:yellow)

## Compute labor share as we vary b, fix z at z0
function laborShare(x, z0)
    @unpack ν, w, y = staticModel(b = x, z = z0)
    return w/y
end

plot(b -> laborShare(b, z0), 0, 0.5, legend = false) 
ylabel!(L"\theta")
xlabel!(L"b")
=#
