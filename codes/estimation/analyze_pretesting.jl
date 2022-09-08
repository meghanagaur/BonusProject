# Produce preliminary binscatters of pretesting output.

using LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =10, yguidefontsize=10, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 9)

using DynamicModel, BenchmarkTools, DataStructures, Distributions, Optim, Sobol, DataFrames,
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase, JLD2, Binscatters

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
params = sob_seq[indices,:] 

#= Note:
var_Δlw      = 1st moment (variance of log wage changes)
dlw1_du      = 2nd moment (dlog w_1 / d u)
dΔlw_dy      = 3rd moment (d Δ log w_it / y_it)
ε            = 1st param
σ_η          = 2nd param
χ            = 3rd param
=#

ε_vals   = params[:,1]
σ_η_vals = params[:,2]
χ_vals   = params[:,3]

var_Δlw   = moms[:,1]
dlw1_du   = moms[:,2]
dΔlw_dy   = moms[:,3]

## Plot the function values
df = DataFrame(y = fvals, x = ε_vals )
p1 = binscatter(df, @formula(y ~ x),xlabel=L"\varepsilon")
df = DataFrame(y = fvals, x = σ_η_vals )
p2 = binscatter(df, @formula(y ~ x),xlabel=L"\sigma_\eta")
df = DataFrame(y = fvals, x = χ_vals )
p3 = binscatter(df, @formula(y ~ x),xlabel=L"\chi")
plot(p1, p2, p3, layout = (3, 1),legend=:false, ylabel=L"f")
savefig("figs/fvals.png")

## Plot model moments

# 1) var_Δlw
df = DataFrame(y = var_Δlw, x = ε_vals )
p1 = binscatter(df, @formula(y ~ x),xlabel=L"\varepsilon")
df = DataFrame(y = var_Δlw, x = σ_η_vals )
p2 = binscatter(df, @formula(y ~ x),xlabel=L"\sigma_\eta")
df = DataFrame(y = var_Δlw, x = χ_vals )
p3 = binscatter(df, @formula(y ~ x),xlabel=L"\chi")
plot(p1, p2, p3, layout = (3, 1),legend=:false,  ylabel=L"Var(\Delta \log w)")
savefig("figs/var_dlw.png")

# 2) dlw1_du
df = DataFrame(y = dlw1_du, x = ε_vals )
p1 = binscatter(df, @formula(y ~ x),xlabel=L"\varepsilon")
df = DataFrame(y = dlw1_du, x = σ_η_vals )
p2 = binscatter(df, @formula(y ~ x),xlabel=L"\sigma_\eta")
df = DataFrame(y = dlw1_du, x = χ_vals )
p3 = binscatter(df, @formula(y ~ x),xlabel=L"\chi")
plot(p1, p2, p3, layout = (3, 1),legend=:false, ylabel=L"\frac{ d E[ \log w_1 | z_t ]}{ d u_t}")
savefig("figs/dlw1_du.png")

# 3) dΔlw_dy
df = DataFrame(y = dΔlw_dy, x = ε_vals )
p1 = binscatter(df, @formula(y ~ x),xlabel=L"\varepsilon")
df = DataFrame(y = dΔlw_dy, x = σ_η_vals )
p2 = binscatter(df, @formula(y ~ x),xlabel=L"\sigma_\eta")
df = DataFrame(y = dΔlw_dy, x = χ_vals )
p3 = binscatter(df, @formula(y ~ x),xlabel=L"\chi")
plot(p1, p2, p3, layout = (3, 1),legend=:false, ylabel=L"\frac{d \Delta \log w_{it} }{ d \log y_{it}}")
savefig("figs/ddlw_dy.png")
