# Produce preliminary heatmaps of calibration of σ_η and ε.

using LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =10, yguidefontsize=10, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 9)

using DynamicModel, BenchmarkTools, DataStructures, Distributions, Optim, Sobol, DataFrames,
ForwardDiff, Interpolations, LinearAlgebra, Parameters, Random, Roots, StatsBase, JLD2, Binscatters

loc = "/Users/meghanagaur/BonusProject/codes/estimation/"
@unpack output, par_grid, baseline_model = load(loc*"jld/calibration_data.jld2") 

#= Note:
var_Δlw      = 1st moment (variance of log wage changes)
dlw1_du      = 2nd moment (dlog w_1 / d u)
dly_dlw      = 3rd moment (d log y_it / d log w_it)
u_ss         = 4th moment
dW_du        = 5th moment (PV wages / d unemployment)
dY_du        = 6th moment (d Y(z) / d u(z))

ε            = 1st param
σ_η          = 2nd param
=#


# Export data to make heatmaps of 6 moments
N            = length(output)
df           = DataFrame()
df.epsilon   = par_grid[1,:]
df.sigma_eta = par_grid[2,:]
df.var_dlw   = [output[i][1][1] for i = 1:N]
df.dlw1_du   = [output[i][1][2] for i = 1:N]
df.dlw_dly   = [output[i][1][3] for i = 1:N]
df.u_ss      = [output[i][1][4] for i = 1:N]
df.dW_du     = [output[i][1][5] for i = 1:N]
df.dY_du     = [output[i][1][6] for i = 1:N]


# Get back the parameter grids
ε_grid   = unique(df.epsilon)
σ_η_grid = unique(df.sigma_eta)

# Reshape moments into a conformable N_ε by N_σ_η matrix
var_dlw = reshape(df.var_dlw, length(ε_grid), length(σ_η_grid) )
dlw1_du = reshape(df.dlw1_du, length(ε_grid), length(σ_η_grid) )
dlw_dly = reshape(df.dlw_dly, length(ε_grid), length(σ_η_grid) )
u_ss    = reshape(df.u_ss, length(ε_grid), length(σ_η_grid) )
dW_du   = reshape(df.dW_du, length(ε_grid), length(σ_η_grid) )
dY_du   = reshape(df.dY_du, length(ε_grid), length(σ_η_grid) )

# Remove indices
function rm_nans(mat)
    nan_bool = isnan.(u_ss).==0
    num_cols = maximum([findfirst(isequal(0), nan_bool[i,:] ) for i = 1:size(mat,2)])
    return num_cols
end

# Plot var_dlw
num_cols = rm_nans(var_dlw)
heatmap(ε_grid[1:num_cols], σ_η_grid, var_dlw[:,1:num_cols])

# Plot dlw1_du
num_cols = rm_nans(dlw1_du)
heatmap(ε_grid[1:num_cols], σ_η_grid, dlw1_du[:,1:num_cols])

# Plot dly_dlw
num_cols = rm_nans(dly_dlw)
heatmap(ε_grid[1:num_cols], σ_η_grid, dlw_dly[:,1:num_cols])

# Plot u_ss
num_cols = rm_nans(u_ss)
heatmap(ε_grid[1:num_cols], σ_η_grid, u_ss[:,1:num_cols])

# Plot dW_du
num_cols = rm_nans(dW_du)
heatmap(ε_grid[1:num_cols], σ_η_grid, dW_du[:,1:num_cols])

# Plot dY_du
num_cols = rm_nans(dY_du)
heatmap(ε_grid[1:num_cols], σ_η_grid, dY_du[:,1:num_cols])


#= check on the grids 
N_grid   = 5
ε_grid   = LinRange(param_bounds[1][1],param_bounds[1][2], N_grid)
σ_η_grid = LinRange(param_bounds[2][1],param_bounds[2][2], N_grid)


# Stack the parameter vectors for parallel computation
par_grid = zeros(2, N_grid^2)
t = 1
@inbounds for i =1:N_grid
    @inbounds for j = 1:N_grid
        par_grid[1,t] = ε_grid[i]
        par_grid[2,t] = σ_η_grid[j]
        global t+=1
    end
end

var_dlw = reshape(par_grid[1,:],5, 5)

=#