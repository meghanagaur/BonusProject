using LinearAlgebra, Distributions, Random, Interpolations, ForwardDiff,
BenchmarkTools, LaTeXStrings, Parameters, StatsBase, Parameters

"""
Holmstrom & Milgrom (1987) 
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
γ = level of unemployment benefit
χ = elasticity of unemployment benefit wrt z
Default parameters from Shimer (2005) (exc. r, ϕ, σ) 
"""
function staticModel(; z = 1, ν = 0.72, ψ = 0.9, κ = 0.213, 
    ϕ = 1, r = 0.8, γ = 0.4, χ = 0.0, σ = 0.2)

    b = γ*z^χ                           # unemployment benefit
    β = z^2 / (z^2 + ϕ*r*σ^2)           # optimal linear contract - constant
    α = b + β^2 * (ϕ*r*σ^2 - z^2 )/(2ϕ) # optimal linear contract - slope
    a = β*z/ϕ                           # optimal effort
    y = a*z                             # E[output]
    w = α + β*y                         # E[wage] (optimal linear contract)
    J = y - w                           # E[profit of filled vacancy]
    v = (ψ*J/κ)^(1/ν)                   # vacancies (given by free entry condition)
    f = ψ*v^(1 - ν)                     # job-finding rate
    q = ψ*v^(-ν)                        # job-filling rate
    n = copy(f)                         # employment, since u = 1

    return (α = α, β = β, a = a, w = w, y = y, J = J, f = f,
            q = q, v = v, n = n, ν = ν, ψ = ψ, κ = κ, ϕ = ϕ, 
            z = z, r = r, b = b, σ = σ)
end

# Compute various eq objects as a function of z, at baseline calibration
W(x)   =  staticModel(z = x).w
A(x)   =  staticModel(z = x).a
Y(x)   =  staticModel(z = x).y
N(x)   =  staticModel(z = x).n
J(x)   =  staticModel(z = x).J
W(x)   =  staticModel(z = x).w
V(x)   =  staticModel(z = x).v

""" 
Compute dlogn/dlog z, following analytical formula.
"""
function derivAnalytical(x)

    @unpack ν, w, y, a, J, r, σ, ϕ  = staticModel(z = x)

    extra_term = (x^3)*(r*σ^2)/(x^2 + ϕ*r*σ^2)^2
    return ((1 - ν)*ν^(-1))*(a + extra_term)*x/J

    # should match the above
    #g  = ForwardDiff.derivative(J, x) 
    #return ((1 - ν)*ν^(-1))*g*x/J(x)

    # expressions from static_model.pdf
    #return ((1 - ν)*ν^(-1))*a*x/J(x)
    #return ((1 - ν)*ν^(-1))/(1 - w/y) 
end

"""
Compute dlogn/dlog z, using numerical derivative.
"""
function derivNumerical(x)
    g = ForwardDiff.derivative(N, x)  
    return g*x/N(x)
end 

"""
Compute J, N, D as a function of z = z, holding fixed w at w*(x0), a at a*(x0)
"""
function globalApproxFixed(x, x0; χ = 0.0)

    @unpack ν, w, a, ψ, κ = staticModel(z = x0, χ = χ) # solve model at z = x0

    y = a*x
    J = y - w           # get profits at z = x, holding fixed w, a at z = x0
    v = (ψ*J/κ)^(1/ν)   # get vacancies from free entry condition
    n = ψ*v^(1 - ν)     # get job-finding/employment
   
    d = ((1 - ν)*ν^(-1))/(1 - w/y) # emp fluctuations

    return (J = J, n = n, d = d, w = w, a = a, v = v)
end

"""
Compute J, N, D as a function of z,  holding fixed effort a at a*(z0)
Set w = Nash Bargained wage for z = x, where β = worker's bargaining power.
h(a) = a shift term captures disutility of effort a in worker's surplus
"""
function globalApproxNash(x, x0; β = staticModel().ν, χ = 0.0)
   
    @unpack ν, a, ψ, κ, b, w       = staticModel(z = x0, χ = χ) # solve model at z = x0
    
    wN(x) = β*a*x + (1 - β)*(b)   # compute Nash-Bargained wage at y = a*(x0)x

    y = a*x                       # E[output]
    w = wN(x) + (w  - wN(x0))     # add a shift term (for now) to capture disuility of a(x0)
    J = y - w                     # get profits at z = x, holding fixed a
    v = (ψ*J/κ)^(1/ν)             # get vacancies from free entry condition
    n = ψ*v^(1 - ν)               # get job-finding/employment
    
    d = ((1 - ν)*ν^(-1))*(1-β)*a*x/J # emp fluctuations

    return (J = J, n = n, d = d, v = v)
end

"""
Compute J, N, D as a function of z = z, holding fixed a at a*(x0)
"""
function globalApproxFixedA(x, x0; χ = 0.0)
    @unpack ν, a, ψ, κ, ϕ, r, σ = staticModel(z = x0, χ = χ) # solve model at z = x0
    @unpack w                   = staticModel(z = x, χ = χ)  # solve model at z = x
    
    y = a*x             # get output
    J = y - w           # get profits at z = x, holding fixed a at x0
    v = (ψ*J/κ)^(1/ν)   # get vacancies from free entry condition
    n = ψ*v^(1 - ν)     # get job-finding/employment

    dw_dz = (x^5 + (x^3)*2ϕ*r*σ^2)/((ϕ)*(x^2 +ϕ*r*σ^2)^2)
    d     = ((1 - ν)*ν^(-1))*(a - dw_dz)*x/J # emp fluctuations

    return (J = J, n = n, d = d, v = v)
end

"""
Compute J, N, D as a function of z = z, holding fixed w at w*(x0)
"""
function globalApproxFixedW(x, x0; χ = 0.0)
    @unpack ν, w, ψ, κ, ϕ, σ, r = staticModel(z = x0, χ = χ) # solve model at z = x0
    @unpack a                   = staticModel(z = x, χ = χ)  # solve model at z = x

    y = a*x             # get output
    J = y - w           # get profits at z = x, holding fixed w at x0
    v = (ψ*J/κ)^(1/ν)   # get vacancies from free entry condition
    n = ψ*v^(1 - ν)     # get job-finding/employment

    da_dz = (x^4 + (x^2)*3ϕ*r*σ^2)/((ϕ)*(x^2 +ϕ*r*σ^2)^2)
    d      = ((1 - ν)*ν^(-1))*(a + x*da_dz)*x/J # emp fluctuations

    return (J = J, n = n, d = d, v = v)
end

# Compute the numerical elasticities
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

"""
Compute J, N, D as a function of z = z,
holding fixed effort. Exogenously impose 
w(z) = w(z_0) + w(z | χ = χ) - w(z | χ = 0) 
"""
function exogCycWage(x; x0 = 1.0, χ = 0.0)

    # solve model at z = x0, to get fixed effort
    @unpack ν, a, ψ, κ, ϕ, σ, r, w = staticModel(z = x0, χ = χ) 
    w_cyc      = staticModel(z = x, χ = χ).w    # procyclical wage
    w_acyc     = staticModel(z = x, χ = 0.0).w  # acyclical wage
    
    w += (w_cyc - w_acyc) # construct cyclical wage
    y  = a*x              # get expected output
    J  = y - w            # get expected profits at z = x
    v  = (ψ*J/κ)^(1/ν)    # get vacancies from free entry condition
    n  = ψ*v^(1 - ν)      # get job-finding/employment

    return (J = J, n = n, a = a, w = w, v = v)
end

"""
Compare dJ/dz in the procyclical incentive pay 
and fixed effort, procyclical wage economy
"""
function dJexogCycWage(z; χ = 0.0)
   
    # difference between procyclical b incentive pay + exogenous wage
    g_1    = ForwardDiff.derivative(x-> staticModel(z = x, χ = χ).J, z)  
    g_2    = ForwardDiff.derivative(x -> exogCycWage(x; χ = χ).J, z)  

    # difference between acyclical b incentive pay + rigid wage
    g_3    = ForwardDiff.derivative(x -> staticModel(z = x, χ = 0.0).J, z)  
    g_4    = ForwardDiff.derivative(x -> globalApproxFixed(x, z0).J , z)  

    # direct effect of z on constraints in incentive pay model 
    @unpack r, σ, ϕ  = staticModel()
    extra_term = (z^3)*(r*σ^2)/(z^2 + ϕ*r*σ^2)^2

    return (diff_1 = g_3 - g_4, diff_2 = g_1 - g_2, check = extra_term)

end 

