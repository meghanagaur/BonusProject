cd(dirname(@__FILE__))

include("functions/smm_settings.jl")        # SMM inputs, settings, packages, etc.

using DataFrames, Binscatters, DelimitedFiles, LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =13, yguidefontsize=13, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 10)

## Logistics
vary_z_N     = 251
file_str     = "fix_eps03_dlogw1_du_05"
file_pre     = "smm/jld/pretesting_"*file_str*".jld2"   # pretesting data location
file_est     = "smm/jld/estimation_"*file_str*".txt"    # estimation output location
file_save    = "figs/vary-z1/"*file_str*"/"             # file to-save 
mkpath(file_save)
println("File name: "*file_str)

# Load output
est_output = readdlm(file_est, ',', Float64)            # open output across all jobs
@unpack moms, fvals, pars, mom_key, param_bounds, param_est, param_vals, data_mom, J, W = load(file_pre) 

# Get the final minimum 
idx        = argmin(est_output[:,1])                    # check for the lowest function value across processes 
pstar      = est_output[idx, 2:(2+J-1)]                 # get parameters 

# Get the relevant parameters
Params =  OrderedDict{Symbol, Float64}()
for (k, v) in param_vals
    if haskey(param_est, k)
        Params[k]  = pstar[param_est[k]]
    else
        Params[k]  = v
    end
end

@unpack σ_η, χ, γ, hbar, ε = Params
baseline = model(σ_η = σ_η, χ = χ, γ = γ, hbar = hbar, ε = ε) 

# Get all of the relevant parameters for the model
@unpack hp, zgrid, N_z, ψ, f, s, σ_η, χ, γ, hbar, ε = baseline 

# Unpack the relevant shocks
@unpack η_shocks, z_shocks, z_shocks_idx, λ_N_z, zstring, burnin, z_ss_dist, indices, indices_y, T_sim = shocks

# Generate model data for every point on zgrid:

iz        = 1
N_sim     = 100000

# Results from panel simulation
lw        = zeros(N_sim)        # log w_it, given z_1 and z_t
ly        = zeros(N_sim)        # log y_it, given z_1 and z_t
η_bar     = 0.3                 # selection criterion for η
η_idx     = zeros(N_sim)        # index for selecting based on η
y_idx     = zeros(N_sim)         # index for selecting based on η
pt        = zeros(N_sim)        # direct computation of pass-through moment

# Get z, η-shocks to compute wages and output for continuing hires
@views z_shocks_z        = z_shocks[iz][1,:]'
@views z_shocks_idx_z    = z_shocks_idx[iz][1,:]'
η_shocks_z               = σ_η*rand(Normal(0,σ_η), N_sim, T_sim)
η_shocks_demean          = vec(η_shocks_z .- mean(η_shocks_z, dims=1))
η_idx                    = vec(abs.(η_shocks_z) .<= η_bar) # limit η range for log y computation

# solve the model for z_1 = zgrid[iz]
sol           = solveModel(baseline; z_1 = zgrid[iz],  noisy = false, check_mult = false)

@unpack conv_flag1, conv_flag2, conv_flag3, wage_flag, effort_flag, IR_err, flag_IR, az, yz, w_0, θ, Y = sol

# Expectation of the log wage of new hires, given z_1 = z
hpz_z1       = hp.(az)  # h'(a(z|z_1))

# Compute relevant terms for log wages and log output
@views hp_az             = hpz_z1[z_shocks_idx_z]
t1                       = ψ*hp_az.*η_shocks_z
t2                       = 0.5*(ψ*hp_az*σ_η).^2 
lw_mat                   = log(w_0) .+ cumsum(t1, dims = 2) .- cumsum(t2, dims = 2)
lw_demean                = vec(lw_mat .- mean(lw_mat,dims=1))
lw                       = vec(lw_mat)

# Compute log individual output
ηz                        = z_shocks_z.*η_shocks_z                # truncate η_t
@views y                  = yz[z_shocks_idx_z] .+ ηz              # a_t(z_t|z_1)*z_t
y_idx                     = vec(y .> 0)                           # y < 0
ly_mat                    = log.(max.(y, 10^-5))                  # nudge up to avoid any runtime errors 
ly_demean                 = vec(ly_mat .- mean(ly_mat, dims=1))   # demean 
ly                        = vec(ly_mat)

# Compute directly pass-through for comparison
@views pt                 =  ψ*hbar*az[z_shocks_idx_z].^(1 + 1/ε)

# Make some adjustments to compute annual wage changes
@views Δlw_y = vec([lw_mat[i,t+12] - lw_mat[i,t] for  i = 1:size(η_shocks_z,1), t = 1:T_sim-12])

# direct
dlw_dly      = mean(pt)

# OLS regression
dlw_dly_2    = ols(lw[(y_idx.==1).*(η_idx.==1)], ly[(y_idx.==1).*(η_idx.==1)] )[2]
cov(lw[(y_idx.==1).*(η_idx.==1)], ly[(y_idx.==1).*(η_idx.==1)])/var(ly[(y_idx.==1).*(η_idx.==1)] )

# add time fixed effects 
dlw_dly_4    = ols(lw_demean[(y_idx.==1).*(η_idx.==1)], ly_demean[(y_idx.==1).*(η_idx.==1)],intercept=false)

# iv estimate
η_shocks_vec = vec(η_shocks_z)
dlw_dly_3    = ols(lw[(y_idx.==1).*(η_idx.==1)], η_shocks_vec[(y_idx.==1).*(η_idx.==1)] )[2]/ols(ly[(y_idx.==1).*(η_idx.==1)], η_shocks_vec[(y_idx.==1).*(η_idx.==1)] )[2]

# iv estimate with time fixed effects
dlw_dly_3    = ols(lw_demean[(y_idx.==1).*(η_idx.==1)], η_shocks_demean[(y_idx.==1).*(η_idx.==1)] )[2] / ols(ly_demean[(y_idx.==1).*(η_idx.==1)], η_shocks_demean[(y_idx.==1).*(η_idx.==1)] )[2]

## explore dlw/dη
ols(lw[(y_idx.==1).*(η_idx.==1)], η_shocks_vec[(y_idx.==1).*(η_idx.==1)] )[2]
cov(lw[(y_idx.==1).*(η_idx.==1)], η_shocks_vec[(y_idx.==1).*(η_idx.==1)])/var(η_shocks_vec[(y_idx.==1).*(η_idx.==1)])
mean(ψ*hp_az) # check

## explore dly/dη
ols(ly[(y_idx.==1).*(η_idx.==1)], η_shocks_vec[(y_idx.==1).*(η_idx.==1)] )[2]
cov(ly[(y_idx.==1).*(η_idx.==1)], η_shocks_vec[(y_idx.==1).*(η_idx.==1)])/var(η_shocks_vec[(y_idx.==1).*(η_idx.==1)])
mean( vec(1 ./ (az[z_shocks_idx_z] .+ η_shocks_z))[(y_idx.==1).*(η_idx.==1)]) # check

## explore dlw/dly 
term1 = vec(ψ*hbar*(hp_az).*(az[z_shocks_idx_z] .+ η_shocks_z).*ly_mat)

cov(term1[(y_idx.==1).*(η_idx.==1)], ly[(y_idx.==1).*(η_idx.==1)])/var(ly[(y_idx.==1).*(η_idx.==1)])
cov(lw[(y_idx.==1).*(η_idx.==1)], ly[(y_idx.==1).*(η_idx.==1)])


cov(term1[(y_idx.==1).*(η_idx.==1)], lw[(y_idx.==1).*(η_idx.==1)])
cov(term1[(y_idx.==1).*(η_idx.==1)], ly[(y_idx.==1).*(η_idx.==1)])
