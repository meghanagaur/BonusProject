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
exp_Δlw      = 2nd moment (expectation of log wage changes)
dlw1_du      = 2nd moment (dlog w_1 / d u)
dlw_dly      = 3rd moment (d log w_it / d log y_it)
ε            = 1st param
σ_η          = 2nd param
χ            = 3rd param
γ            = 4th param
hbar         = 5th param
=#

# Unpack parameters
ε_vals    = pars[:,1]
σ_η_vals  = pars[:,2]
χ_vals    = pars[:,3]
γ_vals    = pars[:,4]
hbar_vals = pars[:,5]

# Unpack moments
std_Δlw   = moms[:,1]
avg_Δlw   = moms[:,2]
dlw1_du   = moms[:,3]
dlw_dly   = moms[:,4]
u_ss      = moms[:,5]

# make bins
nbins    = 100

## Plot the function values
idx = (isnan.(fvals) .== 0).*(fvals .< 1000)
df = DataFrame(y = fvals[idx], x = ε_vals[idx])
p1 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\varepsilon")
df = DataFrame(y = fvals[idx], x = σ_η_vals[idx] )
p2 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\sigma_\eta")
df = DataFrame(y = fvals[idx], x = χ_vals[idx])
p3 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\chi")
plot(p1, p2, p3, layout = (3, 1),legend=:false, ylabel=L"f")
savefig(loc*"figs/pretesting/fvals_1.png")

df = DataFrame(y = fvals, x = γ_vals)
p4 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\gamma")
df = DataFrame(y = fvals, x = hbar_vals)
p4 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\bar{h}")
plot(p1, p2, layout = (2, 1),legend=:false, ylabel=L"f")
savefig(loc*"figs/pretesting/fvals_2.png")

## Plot model moments

## 1) std_Δlw
df = DataFrame(y = std_Δlw, x = ε_vals )
p1 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\varepsilon")
df = DataFrame(y = std_Δlw, x = σ_η_vals )
p2 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\sigma_\eta")
df = DataFrame(y = std_Δlw, x = χ_vals )
p3 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\chi")
plot(p1, p2, p3, layout = (3, 1),legend=:false,  ylabel=L"Std(\Delta \log w)")
savefig(loc*"figs/pretesting/var_dlw_1.png")

df = DataFrame(y = std_Δlw, x = γ_vals )
p1 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\gamma")
df = DataFrame(y = std_Δlw, x = hbar_vals )
p2 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\bar{h}")
plot(p1, p2, layout = (1, 2),legend=:false,  ylabel=L"Std(\Delta \log w)")
savefig(loc*"figs/pretesting/var_dlw_2.png")

## 2) avg_Δlw
df = DataFrame(y = avg_Δlw, x = ε_vals )
p1 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\varepsilon")
df = DataFrame(y = avg_Δlw, x = σ_η_vals )
p2 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\sigma_\eta")
df = DataFrame(y = avg_Δlw, x = χ_vals )
p3 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\chi")
plot(p1, p2, p3, layout = (3, 1),legend=:false,  ylabel=L"\mathbb{E}[\Delta \log w]")
savefig(loc*"figs/pretesting/avg_dlw_1.png")

df = DataFrame(y = avg_Δlw, x = γ_vals )
p1 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\gamma")
df = DataFrame(y = avg_Δlw, x = hbar_vals )
p2 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\bar{h}")
plot(p1, p2, layout = (1, 2),legend=:false,  ylabel=L"\mathbb{E}[\Delta \log w]")
savefig(loc*"figs/pretesting/avg_dlw_2.png")

## 3) dlw1_du
idx = (isnan.(dlw1_du) .== 0)
df = DataFrame(y = dlw1_du[idx], x = ε_vals[idx] )
p1 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\varepsilon")
df = DataFrame(y = dlw1_du[idx], x = σ_η_vals[idx] )
p2 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\sigma_\eta")
df = DataFrame(y = dlw1_du[idx], x = χ_vals[idx])
p3 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\chi")
plot(p1, p2, p3, layout = (3, 1),legend=:false, ylabel=L"\frac{ d E[ \log w_1 | z_t ]}{ d u_t}")
savefig(loc*"figs/pretesting/dlw1_du_1.png")

df = DataFrame(y = dlw1_du[idx], x = γ_vals[idx] )
p1 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\gamma")
df = DataFrame(y = dlw1_du[idx], x = hbar_vals[idx] )
p2 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\bar{h}")
plot(p1, p2,layout = (2, 1),legend=:false, ylabel=L"\frac{ d E[ \log w_1 | z_t ]}{ d u_t}")
savefig(loc*"figs/pretesting/dlw1_du_2.png")

## 4) dlw_dly
idx = (isnan.(dlw_dly) .== 0)
df = DataFrame(y = dlw_dly[idx], x = ε_vals[idx] )
p1 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\varepsilon")
df = DataFrame(y = dlw_dly[idx], x = σ_η_vals[idx] )
p2 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\sigma_\eta")
df = DataFrame(y = dlw_dly[idx], x = χ_vals[idx] )
p3 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\chi")
plot(p1, p2, p3, layout = (3, 1),legend=:false, ylabel=L"\frac{d \log w_{it} }{ d \log y_{it} }")
savefig(loc*"figs/pretesting/dlw_dly_1.png")


df = DataFrame(y = dlw_dly, x = γ_vals )
p1 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\gamma")
df = DataFrame(y = dlw_dly, x = hbar_vals )
p2 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\bar{h}")
plot(p1, p2, layout = (2, 1),legend=:false, ylabel=L"\frac{d \log w_{it} }{ d \log y_{it} }")
savefig(loc*"figs/pretesting/dlw_dly_2.png")

## 5) u_ss
idx = (isnan.(u_ss) .== 0).*(u_ss .>0).*(u_ss .< 1)
df  = DataFrame(y = u_ss[idx], x = ε_vals[idx] )
p1  = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\varepsilon")
df  = DataFrame(y = u_ss[idx], x = σ_η_vals[idx] )
p2  = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\sigma_\eta")
df  = DataFrame(y = u_ss[idx], x = χ_vals[idx] )
p3  = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\chi")
plot(p1, p2, p3, layout = (3, 1),legend=:false, ylabel=L"u_{ss}")
savefig(loc*"figs/pretesting/u_ss_1.png")

df  = DataFrame(y = u_ss[idx], x = γ_vals[idx] )
p1  = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\gamma")
df  = DataFrame(y = u_ss[idx], x = hbar_vals[idx] )
p2  = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\bar{h}")
plot(p1, p2,layout = (2, 1),legend=:false, ylabel=L"u_{ss}")
savefig(loc*"figs/pretesting/u_ss.png")