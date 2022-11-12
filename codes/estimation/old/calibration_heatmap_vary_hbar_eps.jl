# Produce preliminary heatmaps of calibration of σ_η and ε.
using LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =13, yguidefontsize=13, xtickfontsize=8, ytickfontsize=8, titlefontsize =12,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 12*Plots.px, legendfontsize = 9)

using DynamicModel, BenchmarkTools, DataStructures, Distributions, Optim, Sobol, DataFrames,
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase, JLD2, Binscatters

cd(dirname(@__FILE__))

@unpack output, par_grid, baseline_model = load("jld/calibration_data_vary_hbar_epsilon.jld2") 

# Export data to make heatmaps
N            = length(output)
df           = DataFrame()
df.ε         = par_grid[1,:]
df.hbar      = par_grid[2,:]

df.var_dlw   = [output[i][1][1] for i = 1:N]
df.dlw1_du   = [output[i][1][2] for i = 1:N]
df.dlw_dly   = [output[i][1][3] for i = 1:N]
df.u_ss      = [output[i][1][4] for i = 1:N]
df.ir_flag   = [output[i][3] for i = 1:N]
df.flag      = [output[i][2] for i = 1:N]
df.ir_err    = [output[i][4] for i = 1:N]

# Get back the parameter grids
ε_grid       = unique(df.ε)
hbar_grid    = unique(df.hbar)

# Reshape moments into a conformable N_hbar by N_σ_η matrix
var_dlw      = reshape(df.var_dlw, length(ε_grid), length(hbar_grid) )
dlw1_du      = reshape(df.dlw1_du, length(ε_grid), length(hbar_grid) )
dlw_dly      = reshape(df.dlw_dly, length(ε_grid), length(hbar_grid) )
u_ss         = reshape(df.u_ss,  length(ε_grid), length(hbar_grid) )
ir_flag      = reshape(df.ir_flag, length(ε_grid), length(hbar_grid) )
ir_err       = reshape(df.ir_err, length(ε_grid), length(hbar_grid) )

# Plot var_dlw
p1 = heatmap(ε_grid, hbar_grid, var_dlw, title=L"Std(\Delta \log w)")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

# Plot dlw1_du
p2 = heatmap(ε_grid, hbar_grid, dlw1_du, title=L"\frac{ d E[ \log w_1 | z_t ]}{ d u_t}")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

# Plot dlw_dly
p3 = heatmap(ε_grid, hbar_grid, dlw_dly,title=L"\frac{d \log w_{it} }{ d \log y_{it} }")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

# Plot u_ss
p4 = heatmap(ε_grid, hbar_grid, u_ss, title=L"u_{ss}")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

plot(p1, p2, p3, p4, layout = (2,2))
savefig("figs/calibration/moments_vary_hbar_epsilon.pdf")

## Plot IR_err
p1 = heatmap(ε_grid, hbar_grid, ir_err, title="IR err")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

## Plot IR flag
p2 = heatmap(ε_grid, hbar_grid, ir_flag, title="IR flag")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

plot(p1, p2, layout = (1,2),  size = (600,200))
savefig("figs/calibration/ir_vary_hbar_epsilon.pdf")

## Zoom into u_ss where IR flag = 0
cc = 0
i = 1
while (cc == 0) && (i <= size(ir_flag,1))
    global i +=1
    global cc = i*(sum(ir_flag[i,:])>1)
end

# Plot u_ss
p1 = heatmap(ε_grid, hbar_grid, u_ss, title=L"u_{ss}")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

p2 = heatmap(ε_grid, hbar_grid[1:cc], u_ss[1:cc,:], title=L"u_{ss}")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

plot(p1, p2, layout = (1,2),size = (600,200))
savefig("figs/calibration/u_zoom_vary_hbar_epsilon.pdf")

## Compare alternative moment definitions
df.dlw_dly_2 = [output[i][5] for i = 1:N]
df.u_ss_2    = [output[i][6] for i = 1:N]
dlw_dly_2    = reshape(df.dlw_dly_2, length(ε_grid), length(hbar_grid) )
u_ss_2       = reshape(df.u_ss_2,  length(ε_grid), length(hbar_grid) )

# Plot u_ss_1
p1 = heatmap(ε_grid, hbar_grid, u_ss, title=L"u_{ss}^1")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

# Plot u_ss_2
p2 = heatmap(ε_grid, hbar_grid, u_ss_2, title =L"u_{ss}^2") 
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

# Plot dlw_dly_1
p3 = heatmap(ε_grid, hbar_grid, dlw_dly, title=L"\frac{\partial \log w_{it} }{\partial \log y_{it} }^1")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

# Plot dlw_dly_1
p4 = heatmap(ε_grid, hbar_grid, dlw_dly, title=L"\frac{\partial \log w_{it} }{ \partial \log y_{it} }^2")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

plot(p1, p2, p3, p4, layout = (2,2))
savefig("figs/calibration/mom_comp_vary_hbar_epsilon.pdf")
