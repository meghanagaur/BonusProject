# Produce preliminary binscatters of the pretesting output.

using LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =10, yguidefontsize=10, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 9)

using DynamicModel, BenchmarkTools, DataStructures, Distributions, Optim, Sobol, DataFrames,
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase, JLD2, Binscatters

cd(dirname(@__FILE__))

@unpack moms, fvals, pars, IR_flag, IR_err = load("../runs/jld/pretesting_fix_eps03.jld2") 
dir = "figs/pretesting/"

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
#ε_vals    = pars[:,1]
σ_η_vals  = pars[:,1]
χ_vals    = pars[:,2]
γ_vals    = pars[:,3]
hbar_vals = pars[:,4]

# Unpack moments
std_Δlw   = moms[:,1]
#avg_Δlw   = moms[:,2]
dlw1_du   = moms[:,2]
dlw_dly   = moms[:,3]
u_ss      = moms[:,4]

# IR constraint error
IR_err    = vec(IR_err)
IR_flag   = vec(IR_flag)

# Number of bins
nbins    = 100

## 0) Plot the function values
#df = DataFrame(y = fvals, x = ε_vals)
#p1 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\varepsilon")
df = DataFrame(y = fvals, x = σ_η_vals )
p2 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\sigma_\eta")
df = DataFrame(y = fvals, x = χ_vals)
p3 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\chi")
df = DataFrame(y = fvals, x = γ_vals)
p4 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\gamma")
df = DataFrame(y = fvals, x = hbar_vals)
p5 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\bar{h}")
plot(p2, p3, p4, p5, layout = (2, 2),legend=:false, ylabel=L"f")
savefig(dir*"fvals.png")

## Plot model moments

## 1) std_Δlw
#df = DataFrame(y = std_Δlw, x = ε_vals)
#p1 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\varepsilon")
df = DataFrame(y = std_Δlw, x = σ_η_vals )
p2 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\sigma_\eta")
df = DataFrame(y = std_Δlw, x = χ_vals)
p3 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\chi")
df = DataFrame(y = std_Δlw, x = γ_vals)
p4 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\gamma")
df = DataFrame(y = std_Δlw, x = hbar_vals)
p5 = binscatter(df, @formula(y ~ x), nbins,  xlabel=L"\bar{h}")
plot(p2, p3, p4, p5, layout = (2, 2),legend=:false,ylabel=L"Std(\Delta \log w)")
savefig(dir*"std_dlw.png")

## 2) dlw1_du
#df = DataFrame(y = std_Δlw, x = ε_vals)
#p1 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\varepsilon")
df = DataFrame(y = dlw1_du, x = σ_η_vals )
p2 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\sigma_\eta")
df = DataFrame(y = dlw1_du, x = χ_vals)
p3 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\chi")
df = DataFrame(y = dlw1_du, x = γ_vals)
p4 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\gamma")
df = DataFrame(y = dlw1_du, x = hbar_vals)
p5 = binscatter(df, @formula(y ~ x), nbins,  xlabel=L"\bar{h}")
plot(p2, p3, p4, p5, layout = (2, 2),legend=:false, ylabel=L"\frac{ d E[ \log w_1 | z_t ]}{ d u_t}")
savefig(dir*"dlw1_du.png")

## 3) dlw_dly
#df = DataFrame(y = dlw_dly, x = ε_vals)
#p1 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\varepsilon")
df = DataFrame(y = dlw_dly, x = σ_η_vals )
p2 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\sigma_\eta")
df = DataFrame(y = dlw_dly, x = χ_vals)
p3 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\chi")
df = DataFrame(y = dlw_dly, x = γ_vals)
p4 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\gamma")
df = DataFrame(y = dlw_dly, x = hbar_vals)
p5 = binscatter(df, @formula(y ~ x), nbins,  xlabel=L"\bar{h}")
plot(p2, p3, p4, p5, layout = (2, 2),legend=:false,ylabel=L"\frac{d \log w_{it} }{ d \log y_{it} }")
savefig(dir*"dlw_dly.png")

## 4) u_ss
#df = DataFrame(y = u_ss, x = ε_vals)
#p1 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\varepsilon")
df = DataFrame(y = u_ss, x = σ_η_vals )
p2 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\sigma_\eta")
df = DataFrame(y = u_ss, x = χ_vals)
p3 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\chi")
df = DataFrame(y = u_ss, x = γ_vals)
p4 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\gamma")
df = DataFrame(y = u_ss, x = hbar_vals)
p5 = binscatter(df, @formula(y ~ x), nbins,  xlabel=L"\bar{h}")
plot(p2, p3, p4, p5, layout = (2, 2),legend=:false, ylabel=L"u_{ss}")
savefig(dir*"u_ss.png")

## 5) Plot IR_err
#df = DataFrame(y = IR_err, x = ε_vals)
#p1 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\varepsilon")
df = DataFrame(y = IR_err, x = σ_η_vals )
p2 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\sigma_\eta")
df = DataFrame(y = IR_err, x = χ_vals)
p3 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\chi")
df = DataFrame(y = IR_err, x = γ_vals)
p4 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\gamma")
df = DataFrame(y = IR_err, x = hbar_vals)
p5 = binscatter(df, @formula(y ~ x), nbins,  xlabel=L"\bar{h}")
plot(p2, p3, p4, p5, layout = (2, 2),legend=:false, ylabel="IR Error")
savefig(dir*"ir_err.png")

df = DataFrame(y = IR_flag, x = σ_η_vals )
p2 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\sigma_\eta")
df = DataFrame(y = IR_flag, x = χ_vals)
p3 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\chi")
df = DataFrame(y = IR_flag, x = γ_vals)
p4 = binscatter(df, @formula(y ~ x), nbins, xlabel=L"\gamma")
df = DataFrame(y = IR_flag, x = hbar_vals)
p5 = binscatter(df, @formula(y ~ x), nbins,  xlabel=L"\bar{h}")
plot(p2, p3, p4, p5, layout = (2, 2),legend=:false, ylabel="IR flag")
savefig(dir*"ir_flag.png")