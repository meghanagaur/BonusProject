using LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =10, yguidefontsize=10, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 9)

using DynamicModel, BenchmarkTools, DataStructures, Distributions, Optim, Sobol,
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase, JLD2

## Load the saved output
output         = load("jld/pre-testing.jld2", "output")
sob_seq        = load("jld/pre-testing.jld2", "sob_seq")
baseline_model = load("jld/pre-testing.jld2", "baseline_model")

# Retain the valid indices
N_old   = length(output)
indices = [output[i][3] == 0 for i =1:N_old]
out_new = output[indices]
N       = length(out_new)

# Record the function values
fvals   = [out_new[i][1] for i = 1:N]
# Record the moments
moms   = reduce(hcat, out_new[i][2] for i = 1:N)'
# Record the parameters
pars    = sob_seq[indices,:] 

#= Note:
var_Δlw      = 1st moment (variance of log wage changes)
dlw1_du      = 2nd moment (dlog w_1 / d u)
dΔlw_dy      = 3rd moment (d Δ log w_it / y_it)
ε            = 1st param
σ_η          = 2nd param
χ            = 3rd param
=#

ε_vals   = pars[:,1]
σ_η_vals = pars[:,2]
χ_vals   = pars[:,3]

var_Δlw   = moms[:,1]
dlw1_du   = moms[:,2]
dΔlw_dy   = moms[:,3]

## Plot the function values
p1=plot(ε_vals, fvals, xlabel=L"\varepsilon")
p2=plot(σ_η_vals, fvals, xlabel=L"\sigma_\eta")
p3=plot(χ_vals, fvals, xlabel=L"\chi")
plot(p1, p2, p3, layout = (3, 1),legend=:false, ylabel=L"f")

## Plot model moments

# 1) var_Δlw
p1=plot(ε_vals, var_Δlw, xlabel=L"\varepsilon")
p2=plot(σ_η_vals, var_Δlw, xlabel=L"\sigma_\eta")
p3=plot(χ_vals, var_Δlw, xlabel=L"\chi")
plot(p1, p2, p3, layout = (3, 1),legend=:false, ylabel=L"Var(\Delta \log w)")

# 2) dlw1_du
p1=plot(ε_vals, dlw1_du, xlabel=L"\varepsilon")
p2=plot(σ_η_vals, dlw1_du, xlabel=L"\sigma_\eta")
p3=plot(χ_vals, dlw1_du, xlabel=L"\chi")
plot(p1, p2, p3, layout = (3, 1),legend=:false, ylabel=L"\frac{ d E[ \log w_1 | z_t ]}{ d u_t}")

# 3) dΔlw_dy
p1=plot(ε_vals, dΔlw_dy, xlabel=L"\varepsilon")
p2=plot(σ_η_vals, dΔlw_dy, xlabel=L"\sigma_\eta")
p3=plot(χ_vals, dΔlw_dy, xlabel=L"\chi")
plot(p1, p2, p3, layout = (3, 1),legend=:false, ylabel=L"\frac{d \Delta \log w_{it} }{ d \log y_{it}}")

