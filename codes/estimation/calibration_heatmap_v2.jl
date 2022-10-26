# Produce preliminary heatmaps of calibration of σ_η and ε.

using LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =13, yguidefontsize=13, xtickfontsize=8, ytickfontsize=8, titlefontsize =12,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 12*Plots.px, legendfontsize = 9)

using DynamicModel, BenchmarkTools, DataStructures, Distributions, Optim, Sobol, DataFrames,
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase, JLD2, Binscatters

loc = "/Users/meghanagaur/BonusProject/codes/estimation/"
@unpack output, par_grid, baseline_model = load(loc*"jld/calibration_data.jld2") 

# Export data to make heatmaps
N            = length(output)
df           = DataFrame()
df.epsilon   = par_grid[1,:]
df.sigma_eta = par_grid[2,:]
df.hbar      = par_grid[3,:]

df.var_dlw   = [output[i][1][1] for i = 1:N]
df.dlw1_du   = [output[i][1][2] for i = 1:N]
df.dlw_dly   = [output[i][1][3] for i = 1:N]
df.u_ss      = [output[i][1][4] for i = 1:N]
df.avg_Δlw   = [output[i][1][5] for i = 1:N]
df.dlw1_dlz  = [output[i][1][6] for i = 1:N]
df.dlY_dlz   = [output[i][1][7] for i = 1:N]
df.dlu_dlz   = [output[i][1][8] for i = 1:N]
df.std_u     = [output[i][1][9] for i = 1:N]
df.std_z     = [output[i][1][10] for i = 1:N]
df.std_Y     = [output[i][1][11] for i = 1:N]
df.std_w0    = [output[i][1][12] for i = 1:N]

# Get back the parameter grids
ε_grid       = unique(df.epsilon)
σ_η_grid     = unique(df.sigma_eta)
hbar_grid    = unique(df.hbar)

# Reshape moments into a conformable N_ε by N_σ_η matrix
var_dlw      = reshape(df.var_dlw, length(ε_grid), length(σ_η_grid), length(hbar_grid) )
dlw1_du      = reshape(df.dlw1_du, length(ε_grid), length(σ_η_grid), length(hbar_grid) )
dlw_dly      = reshape(df.dlw_dly, length(ε_grid), length(σ_η_grid), length(hbar_grid) )
u_ss         = reshape(df.u_ss, length(ε_grid), length(σ_η_grid), length(hbar_grid) )
avg_Δlw      = reshape(df.avg_Δlw, length(ε_grid), length(σ_η_grid), length(hbar_grid))
dlw1_dlz     = reshape(df.dlw1_dlz, length(ε_grid), length(σ_η_grid), length(hbar_grid) )
dlY_dlz      = reshape(df.dlY_dlz, length(ε_grid), length(σ_η_grid), length(hbar_grid) )
dlu_dlz      = reshape(df.dlu_dlz, length(ε_grid), length(σ_η_grid), length(hbar_grid) )
std_u        = reshape(df.std_u, length(ε_grid), length(σ_η_grid), length(hbar_grid) )
std_z        = reshape(df.std_z, length(ε_grid), length(σ_η_grid), length(hbar_grid) )
std_Y        = reshape(df.std_Y, length(ε_grid), length(σ_η_grid), length(hbar_grid) )
std_w0       = reshape(df.std_w0, length(ε_grid), length(σ_η_grid), length(hbar_grid) )

############## FIX HBAR AT 1 ##########################
idx = findfirst(x -> x >= 1, hbar_grid)
############## FIRST SET OF MOMENTS ###################

# Plot var_dlw
p1 = heatmap(ε_grid, σ_η_grid, var_dlw[idx,:,:], title=L"Std(\Delta \log w)")
xlabel!(L"\varepsilon")
ylabel!(L"\sigma_\eta")

# Plot dlw1_du
p2 = heatmap(ε_grid, σ_η_grid, dlw1_du[idx,:,:], title=L"\frac{ d E[ \log w_1 | z_t ]}{ d u_t}")
xlabel!(L"\varepsilon")
ylabel!(L"\sigma_\eta")

# Plot dlw_dly
p3 = heatmap(ε_grid, σ_η_grid, dlw_dly[idx,:,:],title=L"\frac{d \log w_{it} }{ d \log y_{it} }")
xlabel!(L"\varepsilon")
ylabel!(L"\sigma_\eta")

# Plot u_ss
p4 = heatmap(ε_grid, σ_η_grid, u_ss[idx,:,:], title=L"u_{ss}")
xlabel!(L"\varepsilon")
ylabel!(L"\sigma_\eta")

plot(p1, p2, p3, p4, layout = (2,2), plot_title=L"\bar{h}="*string(round(hbar_grid[idx], digits=2)))
savefig(loc*"figs/calibration/fix_hbar_1.pdf")

############## SECOND SET OF MOMENTS ###################
#=
# Plot E[dlw]
p1 = heatmap(ε_grid, σ_η_grid, avg_Δlw[idx,:,:], title=L"E[\Delta \log w]")
xlabel!(L"\varepsilon")
ylabel!(L"\sigma_\eta")

# Plot dlw1_dlz
p2 = heatmap(ε_grid, σ_η_grid, dlw1_dlz[idx,:,:], title=L"\frac{ d E[ \log w_1 | z ]}{ d \log z }")
xlabel!(L"\varepsilon")
ylabel!(L"\sigma_\eta")

# Plot dlY_dlz
p3 = heatmap(ε_grid, σ_η_grid, dlY_dlz[idx,:,:],title=L"\frac{d \log Y }{ d \log z }")
xlabel!(L"\varepsilon")
ylabel!(L"\sigma_\eta")

# Plot dlu_dlz
p4 = heatmap(ε_grid, σ_η_grid, dlu_dlz[idx,:,:], title=L"\frac{d \log u }{ d \log z }")
xlabel!(L"\varepsilon")
ylabel!(L"\sigma_\eta")

plot(p1, p2, p3, p4, layout = (2,2), plot_title=L"\bar{h}="*string(round(hbar_grid[idx], digits=2)))
savefig(loc*"figs/calibration/fix_hbar_2.pdf")

############## THIRD SET OF MOMENTS ###################

# Std(u_t)
p1 = heatmap(ε_grid, σ_η_grid, std_u[idx,:,:], title=L"Std(u_t)")
xlabel!(L"\varepsilon")
ylabel!(L"\sigma_\eta")

# Std(z_t)
p2 = heatmap(ε_grid, σ_η_grid, std_z[idx,:,:], title=L"Std(z_t)")
xlabel!(L"\varepsilon")
ylabel!(L"\sigma_\eta")

# Std(Y_t)
p3 = heatmap(ε_grid, σ_η_grid, std_Y[idx,:,:], title=L"Std(y_t)")
xlabel!(L"\varepsilon")
ylabel!(L"\sigma_\eta")

# Std(w_0)
p4 = heatmap(ε_grid, σ_η_grid, std_w0[idx,:,:], title=L"Std(w_0)")
xlabel!(L"\varepsilon")
ylabel!(L"\sigma_\eta")

plot(p1, p2, p3, p4, layout = (2,2), plot_title=L"\bar{h}="*string(round(hbar_grid[idx], digits=2)))
savefig(loc*"figs/calibration/fix_hbar_3.pdf")
=#
############## FIX SIGMA_ETA at 0.5 ###################
idx = length(σ_η_grid)
############## FIRST SET OF MOMENTS ###################

# Plot var_dlw
p1 = heatmap(ε_grid, hbar_grid, var_dlw[:,idx,:], title=L"Std(\Delta \log w)")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

# Plot dlw1_du
p2 = heatmap(ε_grid, hbar_grid, dlw1_du[:,idx,:], title=L"\frac{ d E[ \log w_1 | z_t ]}{ d u_t}")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

# Plot dlw_dly
p3 = heatmap(ε_grid, hbar_grid, dlw_dly[:,idx,:],title=L"\frac{d \log w_{it} }{ d \log y_{it} }")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

# Plot u_ss
p3 = heatmap(ε_grid, hbar_grid, u_ss[:,idx,:], title=L"u_{ss}")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

plot(p1, p2, p3, p4, layout = (2,2), plot_title=L"\sigma_{\eta}="*string(round(σ_η_grid[idx], digits=2)))
savefig(loc*"figs/calibration/fix_sigma_eta_1.pdf")

############## SECOND SET OF MOMENTS ###################
#=
# Plot E[dlw]
p1 = heatmap(ε_grid, hbar_grid, avg_Δlw[:,idx,:], title=L"E[\Delta \log w]")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

# Plot dlw1_dlz
p2 = heatmap(ε_grid, hbar_grid, dlw1_dlz[:,idx,:], title=L"\frac{ d E[ \log w_1 | z ]}{ d \log z }")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

# Plot dlY_dlz
p3 = heatmap(ε_grid, hbar_grid, dlY_dlz[:,idx,:],title=L"\frac{d \log Y }{ d \log z }")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

# Plot dlu_dlz
p4 = heatmap(ε_grid, hbar_grid, dlu_dlz[:,idx,:], title=L"\frac{d \log u }{ d \log z }")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

plot(p1, p2, p3, p4, layout = (2,2), plot_title=L"\sigma_{\eta}="*string(round(σ_η_grid[idx], digits=2)))
savefig(loc*"figs/calibration/fix_sigma_eta_2.pdf")

############## THIRD SET OF MOMENTS ###################

# Std(u_t)
p1 = heatmap(ε_grid, σ_η_grid, std_u[:,idx,:], title=L"Std(u_t)")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

# Std(z_t)
p2 = heatmap(ε_grid, σ_η_grid, std_z[:,idx,:], title=L"Std(z_t)")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

# Std(Y_t)
p3 = heatmap(ε_grid, σ_η_grid, std_Y[:,idx,:], title=L"Std(y_t)")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

# Std(w_0)
p4 = heatmap(ε_grid, σ_η_grid, std_w0[:,idx,:], title=L"Std(w_0)")
xlabel!(L"\varepsilon")
ylabel!(L"\bar{h}")

plot(p1, p2, p3, p4, layout = (2,2), plot_title=L"\sigma_{\eta}="*string(σ_η_grid[idx]))
savefig(loc*"figs/calibration/fix_sigma_eta_3.pdf")
=#
############## FIX EPSILON at 0.3 #####################
idx = findfirst(x -> x >= 0.3, ε_grid)
############## FIRST SET OF MOMENTS ###################

# Plot var_dlw
p1 = heatmap(σ_η_grid, hbar_grid, var_dlw[:,:,idx], title=L"Std(\Delta \log w)")
xlabel!(L"\sigma_\eta")
ylabel!(L"\bar{h}")

# Plot dlw1_du
p2 = heatmap(σ_η_grid, hbar_grid, dlw1_du[:,:,idx], title=L"\frac{ d E[ \log w_1 | z_t ]}{ d u_t}")
xlabel!(L"\sigma_\eta")
ylabel!(L"\bar{h}")

# Plot dlw_dly
p3 = heatmap(σ_η_grid, hbar_grid, dlw_dly[:,:,idx],title=L"\frac{d \log w_{it} }{ d \log y_{it} }")
xlabel!(L"\sigma_\eta")
ylabel!(L"\bar{h}")

# Plot u_ss
p4 = heatmap(σ_η_grid, hbar_grid, u_ss[:,:,idx], title=L"u_{ss}")
xlabel!(L"\sigma_\eta")
ylabel!(L"\bar{h}")

plot(p1, p2, p3, p4, layout = (2,2), plot_title=L"\varepsilon="*string(round(ε_grid[idx], digits=2)))
savefig(loc*"figs/calibration/fix_epsilon_1.pdf")

############## SECOND SET OF MOMENTS ###################

# Plot E[dlw]
p1 = heatmap(σ_η_grid, hbar_grid, avg_Δlw[:,:,idx], title=L"E[\Delta \log w]")
xlabel!(L"\sigma_\eta")
ylabel!(L"\bar{h}")

# Plot dlw1_dlz
p2 = heatmap(σ_η_grid, hbar_grid, dlw1_dlz[:,:,idx], title=L"\frac{ d E[ \log w_1 | z ]}{ d \log z }")
xlabel!(L"\sigma_\eta")
ylabel!(L"\bar{h}")

# Plot dlY_dlz
p3 = heatmap(σ_η_grid, hbar_grid, dlY_dlz[:,:,idx],title=L"\frac{d \log Y }{ d \log z }")
xlabel!(L"\sigma_\eta")
ylabel!(L"\bar{h}")

# Plot dlu_dlz
p4 = heatmap(σ_η_grid, hbar_grid, dlu_dlz[:,:,idx], title=L"\frac{d \log u }{ d \log z }")
xlabel!(L"\sigma_\eta")
ylabel!(L"\bar{h}")

plot(p1, p2, p3, p4, layout = (2,2), plot_title=L"\varepsilon="*string(round(ε_grid[idx], digits=2)))
savefig(loc*"figs/calibration/fix_epsilon_2.pdf")

############## THIRD SET OF MOMENTS ###################

# Std(u_t)
p1 = heatmap(σ_η_grid, hbar_grid, std_u[:,:,idx], title=L"Std(u_t)")
xlabel!(L"\sigma_\eta")
ylabel!(L"\bar{h}")

# Std(z_t)
p2 = heatmap(σ_η_grid, hbar_grid, std_z[:,:,idx], title=L"Std(z_t)")
xlabel!(L"\sigma_\eta")
ylabel!(L"\bar{h}")

# Std(Y_t)
p3 = heatmap(σ_η_grid, hbar_grid, std_Y[:,:,idx], title=L"Std(y_t)")
xlabel!(L"\sigma_\eta")
ylabel!(L"\bar{h}")

# Std(w_0)
p4 = heatmap(σ_η_grid, hbar_grid, std_w0[:,:,idx], title=L"Std(w_0)")
xlabel!(L"\sigma_\eta")
ylabel!(L"\bar{h}")

plot(p1, p2, p3, p4, layout = (2,2), plot_title=L"\varepsilon="*string(round(ε_grid[idx], digits=2)))
savefig(loc*"figs/calibration/fix_epsilon_3.pdf")


#= check on the grids 
N_grid    = 5
ε_grid    = LinRange(param_bounds[1][1],param_bounds[1][2], N_grid)
σ_η_grid  = LinRange(param_bounds[2][1],param_bounds[2][2], N_grid)
hbar_grid = LinRange(param_bounds[5][1],param_bounds[5][2], N_grid)

# Stack the parameter vectors for parallel computation
p_grid = zeros(3, N_grid^3)
t = 1
for i =1:N_grid
    for j = 1:N_grid
        for k = 1:N_grid
            p_grid[1,t] = ε_grid[i]
            p_grid[2,t] = σ_η_grid[j]
            p_grid[3,t] = hbar_grid[k]
            global t+=1
        end
    end
end

reshape(p_grid[1,:],5, 5, 5)
reshape(p_grid[2,:],5, 5, 5)
reshape(p_grid[3,:],5, 5, 5)

=#


#=
# Remove indices
function rm_nans(mat)
    nan_bool = isnan.(mat).==0
    num_cols = maximum([findfirst(isequal(0), nan_bool[i,:] ) for i = 1:size(mat,2)])
    return num_cols
end
=#
