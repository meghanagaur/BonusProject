# Produce preliminary heatmaps of calibration of σ_η and ε.
using LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =13, yguidefontsize=13, xtickfontsize=8, ytickfontsize=8, titlefontsize =12,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 12*Plots.px, legendfontsize = 9)

using DynamicModel, BenchmarkTools, DataStructures, Distributions, Optim, Sobol, DataFrames,
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase, JLD2, Binscatters

cd(dirname(@__FILE__))

# file path and parameters
file     = "vary_sigma_hbar"

if file == "vary_eps_hbar"
    par1_str = L"\varepsilon"
    par2_str = L"\bar{h}"

elseif file == "vary_sigma_hbar"
    par1_str = L"\sigma_\eta"
    par2_str = L"\bar{h}"

elseif file == "vary_eps_chi"
    par1_str = L"\varepsilon"
    par2_str = L"\chi"
end

@unpack output, par_grid, baseline_model = load("jld/calibration_"*file*".jld2") 

# figure file path
dir  = "figs/heatmaps/"*file*"/"
mkpath(dir)

# Export data to make heatmaps
N            = length(output)
df           = DataFrame()
df.par1      = par_grid[1,:]
df.par2      = par_grid[2,:]

df.var_dlw       = [output[i][1][1] for i = 1:N]
df.dlw1_du       = [output[i][1][2] for i = 1:N]
df.dlw_dly       = [output[i][1][3] for i = 1:N]
df.u_ss          = [output[i][1][4] for i = 1:N]
df.dlogθ_dlogz   = [output[i][1][5] for i = 1:N]
df.ir_flag       = [output[i][3] for i = 1:N]
df.flag          = [output[i][2] for i = 1:N]
df.ir_err        = [output[i][4] for i = 1:N]

# Get back the parameter grids
par1_grid    = unique(df.par1)
par2_grid    = unique(df.par2)

# Reshape moments into a conformable matrix
var_dlw      = reshape(df.var_dlw, length(par1_grid), length(par2_grid) )
dlw1_du      = reshape(df.dlw1_du, length(par1_grid), length(par2_grid) )
dlw_dly      = reshape(df.dlw_dly, length(par1_grid), length(par2_grid) )
u_ss         = reshape(df.u_ss,  length(par1_grid), length(par2_grid) )
dlogθ_dlogz  = reshape(df.dlogθ_dlogz, length(par1_grid), length(par2_grid) )
ir_flag      = reshape(df.ir_flag, length(par1_grid), length(par2_grid) )
ir_err       = reshape(df.ir_err, length(par1_grid), length(par2_grid) )
flag         = reshape(df.flag, length(par1_grid), length(par2_grid) )

# Plot var_dlw
p1 = heatmap(par1_grid, par2_grid, var_dlw, title="\n"*L"std(\Delta \log w)")
xlabel!(par1_str)
ylabel!(par2_str)

# Plot dlw1_du
p2 = heatmap(par1_grid, par2_grid, dlw1_du, title=L"\frac{ \partial E[ \log w_1 | z_t ]}{ \partial u_t}")
xlabel!(par1_str)
ylabel!(par2_str)

# Plot dlw_dly
p3 = heatmap(par1_grid, par2_grid, dlw_dly,title=L"\frac{\partial \log w_{it} }{ \partial \log y_{it} }")
xlabel!(par1_str)
ylabel!(par2_str)

# Plot u_ss
p4 = heatmap(par1_grid, par2_grid, u_ss, title="\n"*L"u_{ss}")
xlabel!(par1_str)
ylabel!(par2_str)

plot(p1, p2, p3, p4, layout = (2,2))
savefig(dir*"moments.pdf")

## Plot IR_err
p1 = heatmap(par1_grid, par2_grid, ir_err, title="IR err")
xlabel!(par1_str)
ylabel!(par2_str)

## Plot IR flag
p2 = heatmap(par1_grid, par2_grid, ir_flag, title="IR flag")
xlabel!(par1_str)
ylabel!(par2_str)

plot(p1, p2, layout = (1,2),  size = (600,200))
savefig(dir*"ir_error.pdf")

## Plot dlogθ/dlogz at steady state
dlogθ_dlogz[ir_flag.==1] .= NaN
p1 = heatmap(par1_grid, par2_grid, dlogθ_dlogz)
xlabel!(par1_str)
ylabel!(par2_str)
savefig(dir*"dlogtheta_dlogz.pdf")

## Compare alternative moment definitions
df.dlw_dly_2 = [output[i][1][6] for i = 1:N]
df.u_ss_2    = [output[i][1][7] for i = 1:N]
dlw_dly_2    = reshape(df.dlw_dly_2, length(par1_grid), length(par2_grid) )
u_ss_2       = reshape(df.u_ss_2,  length(par1_grid), length(par2_grid) )

# Plot u_ss_1
p1 = heatmap(par1_grid, par2_grid, u_ss, title="\n"*L"u_{ss}^1")
xlabel!(par1_str)
ylabel!(par2_str)

# Plot u_ss_2
p2 = heatmap(par1_grid, par2_grid, u_ss_2, title ="\n"*L"u_{ss}^2") 
xlabel!(par1_str)
ylabel!(par2_str)

# Plot dlw_dly_1
p3 = heatmap(par1_grid, par2_grid, dlw_dly, title=L"\frac{\partial \log w_{it}^1 }{\partial \log y_{it} }")
xlabel!(par1_str)
ylabel!(par2_str)

# Plot dlw_dly_1
p4 = heatmap(par1_grid, par2_grid, dlw_dly_2, title=L"\frac{\partial \log w_{it}^2 }{ \partial \log y_{it} }")
xlabel!(par1_str)
ylabel!(par2_str)

plot(p1, p2, p3, p4, layout = (2,2))
savefig(dir*"moments_check.pdf")

#= Zoom into u_ss where IR flag = 0

if maximum(ir_flag) >=1
    cc = 0
    i = 1
    while (cc == 0) && (i <= size(ir_flag,1))
        global i +=1
        global cc = i*(sum(ir_flag[i,:])>1)
    end

    # Plot u_ss
    p1 = heatmap(par1_grid, par2_grid, u_ss, title=L"u_{ss}")
    xlabel!(par1_str)
    ylabel!(par2_str)

    p2 = heatmap(par1_grid, par2_grid[1:cc], u_ss[1:cc,:], title=L"u_{ss}")
    xlabel!(par1_str)
    ylabel!(par2_str)

    plot(p1, p2, layout = (1,2),size = (600,200))
    savefig(dir*"u_zoom.pdf")
end
=#