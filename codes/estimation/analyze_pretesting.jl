# Produce preliminary binscatters of the pretesting output.

using LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =10, yguidefontsize=10, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 9)

using DynamicModel, BenchmarkTools, DataStructures, Distributions, Optim, Sobol, DataFrames,
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase, JLD2, Binscatters

loc = "/Users/meghanagaur/BonusProject/codes/estimation/"
@unpack moms, fvals, pars = load(loc*"jld/pretesting_clean.jld2") 

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
γ_vals   = params[:,4]

std_Δlw   = moms[:,1]
dlw1_du   = moms[:,2]
dΔlw_dy   = moms[:,3]
u_ss      = moms[:,4]

## Plot the function values
df = DataFrame(y = fvals, x = ε_vals )
p1 = binscatter(df, @formula(y ~ x),xlabel=L"\varepsilon")
df = DataFrame(y = fvals, x = σ_η_vals )
p2 = binscatter(df, @formula(y ~ x),xlabel=L"\sigma_\eta")
df = DataFrame(y = fvals, x = χ_vals )
p3 = binscatter(df, @formula(y ~ x),xlabel=L"\chi")
df = DataFrame(y = fvals, x = χ_vals )
p4 = binscatter(df, @formula(y ~ x),xlabel=L"\gamma")
plot(p1, p2, p3, p4, layout = (2, 2),legend=:false, ylabel=L"f")
savefig(loc*"figs/fvals.png")

## Plot model moments

# 1) std_Δlw
df = DataFrame(y = var_Δlw, x = ε_vals )
p1 = binscatter(df, @formula(y ~ x),xlabel=L"\varepsilon")
df = DataFrame(y = var_Δlw, x = σ_η_vals )
p2 = binscatter(df, @formula(y ~ x),xlabel=L"\sigma_\eta")
df = DataFrame(y = var_Δlw, x = χ_vals )
p3 = binscatter(df, @formula(y ~ x),xlabel=L"\chi")
df = DataFrame(y = var_Δlw, x = γ_vals )
p4 = binscatter(df, @formula(y ~ x),xlabel=L"\gamma")
plot(p1, p2, p3, p4, layout = (2, 2),legend=:false,  ylabel=L"Std(\Delta \log w)")
savefig(loc*"figs/var_dlw.png")

# 2) dlw1_du
df = DataFrame(y = dlw1_du, x = ε_vals )
p1 = binscatter(df, @formula(y ~ x),xlabel=L"\varepsilon")
df = DataFrame(y = dlw1_du, x = σ_η_vals )
p2 = binscatter(df, @formula(y ~ x),xlabel=L"\sigma_\eta")
df = DataFrame(y = dlw1_du, x = χ_vals )
p3 = binscatter(df, @formula(y ~ x),xlabel=L"\chi")
df = DataFrame(y = dlw1_du, x = γ_vals )
p4 = binscatter(df, @formula(y ~ x),xlabel=L"\gamma")
plot(p1, p2, p3, p4, layout = (2, 2),legend=:false, ylabel=L"\frac{ d E[ \log w_1 | z_t ]}{ d u_t}")
savefig(loc*"figs/dlw1_du.png")

# 3) dΔlw_dy
df = DataFrame(y = dΔlw_dy, x = ε_vals )
p1 = binscatter(df, @formula(y ~ x),xlabel=L"\varepsilon")
df = DataFrame(y = dΔlw_dy, x = σ_η_vals )
p2 = binscatter(df, @formula(y ~ x),xlabel=L"\sigma_\eta")
df = DataFrame(y = dΔlw_dy, x = χ_vals )
p3 = binscatter(df, @formula(y ~ x),xlabel=L"\chi")
df = DataFrame(y = dΔlw_dy, x = γ_vals )
p4 = binscatter(df, @formula(y ~ x),xlabel=L"\gamma")
plot(p1, p2, p3, p4, layout = (2, 2),legend=:false, ylabel=L"\frac{d \Delta \log w_{it} }{ d \log y_{it}}")
savefig(loc*"figs/ddlw_dy.png")

# 4) u_ss
df = DataFrame(y = u_ss, x = ε_vals )
p1 = binscatter(df, @formula(y ~ x),xlabel=L"\varepsilon")
df = DataFrame(y = u_ss, x = σ_η_vals )
p2 = binscatter(df, @formula(y ~ x),xlabel=L"\sigma_\eta")
df = DataFrame(y = u_ss, x = χ_vals )
p3 = binscatter(df, @formula(y ~ x),xlabel=L"\chi")
df = DataFrame(y = u_ss, x = γ_vals )
p4 = binscatter(df, @formula(y ~ x),xlabel=L"\gamma")
plot(p1, p2, p3, p4, layout = (2, 2),legend=:false, ylabel=L"u_ss")
savefig(loc*"figs/u_ss.png")