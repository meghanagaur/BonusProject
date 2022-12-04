cd(dirname(@__FILE__))

include("functions/smm_settings.jl")        # SMM inputs, settings, packages, etc.
include("functions/calibration_vary_z1.jl") # vary z1 functions

using DelimitedFiles, LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =13, yguidefontsize=13, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 9)

## Logistics
file_str     = "fix_eps03"
file_pre     = "runs/jld/pretesting_"*file_str*".jld2"  # pretesting data location
file_est     = "runs/jld/estimation_"*file_str*".txt"   # estimation output location
file_save    = "figs/vary-z1/"*file_str*"/"             # file to-save 
mkpath(file_save)

# Simulate moments
est_output = readdlm(file_est, ',', Float64)   # open output across all jobs
@unpack moms, fvals, pars, mom_key, param_bounds, param_est, param_vals, data_mom, J, W = load(file_pre) 

idx        = argmin(est_output[:,1])           # check for the lowest function value across processes 
pstar      = est_output[idx, 2:(2+J-1)]        # get parameters 

# Get the relevant parameters
Params =  OrderedDict{Symbol, Float64}()
for (k, v) in param_vals
    if haskey(param_est, k)
        Params[k]  = pstar[param_est[k]]
    else
        Params[k]  = v
    end
end

""" 
Function to simulate moments at estimated parameter values
"""
function simulate_moments(Params; check_mult = false)
    @unpack σ_η, χ, γ, hbar, ε = Params
    baseline = model(σ_η = σ_η, χ = χ, γ = γ, hbar = hbar, ε = ε) 
    out      = simulate(baseline, shocks; check_mult = check_mult)
    return out
end

# Get moments
output     = simulate_moments(Params; check_mult = true)
@unpack std_Δlw, dlw1_du, dlw_dly, u_ss, u_ss_2, avg_Δlw, dlw1_dlz, dlY_dlz, dlu_dlz, std_u, std_z, std_Y, std_w0, flag, flag_IR, IR_err  = output

# Estimated parameters
round.(Params[:σ_η], digits=4)
round.(Params[:χ], digits=4)
round.(Params[:γ], digits=4)
round.(Params[:hbar], digits=4)
round.(Params[:ε], digits=4)

# Targeted moments
round(std_Δlw,digits=4)
round(dlw1_du,digits=4)
round(dlw_dly,digits=4)
round(u_ss,digits=4)

# Addiitonal moments
round(u_ss_2,digits=4)
round(dlu_dlz,digits=4)
round(dlY_dlz,digits=4)
round(dlw1_dlz,digits=4)
round(std_u,digits=4)
round(std_z,digits=4)
round(std_Y,digits=4)
round(std_w0,digits=4)

## Vary z1 experiments

# Get the Bonus model aggregates
@unpack σ_η, χ, γ, hbar, ε = Params
modd  = model(N_z = 21, σ_η = σ_η, χ = χ, γ = γ, hbar = hbar, ε = ε)
#modd  = model(N_z = 21, σ_η = 0.5, χ = 0.0, γ = 0.6, hbar = 1.0, ε = 0.5)
@unpack w_0_B, θ_B, W_B, Y_B, ω_B, J_B, a_B, z_ss_idx, zgrid = vary_z1(modd)

# Get the Hall analogues
a_H, W_H, J_H, Y_H, θ_H = solveHall(modd, z_ss_idx, Y_B, W_B, J_B);

# Plot labels
rigid   = "Rigid Wage: Fixed w and a"
bonus   = "Incentive Pay: Variable w and a"
logz    = log.(zgrid)

# Plot profits
plot(logz, J_B, linecolor=:red, label=bonus, legend=:topleft)
plot!(logz, J_H, linecolor=:blue,label=rigid)
#hline!([0],linecolor=:black,linestyle=:dash, label="")
xaxis!(L"\log z")
yaxis!(L"J")
savefig(file_save*"profits.pdf")

# Plot effort 
plot(logz, a_B, linecolor=:red, label=bonus, legend=:topleft)
hline!([a_H], linecolor=:blue, label=rigid)
xaxis!(L"\log z")
yaxis!(L"a")
savefig(file_save*"efforts.pdf")

# Plot tightness
plot(logz, θ_B, linecolor=:red, label=bonus, legend=:topleft)
plot!(logz, θ_H, linecolor=:blue, label=rigid)
xaxis!(L"\log z")
yaxis!(L"\theta")
savefig(file_save*"tightness.pdf")

# Plot wages
plot(logz, W_B, linecolor=:red, label=bonus, legend=:topleft)
hline!([W_H], linecolor=:blue, label=rigid)
xaxis!(L"\log z")
yaxis!(L"W")
savefig(file_save*"wages.pdf")

# Plot dlog θ / d log z
tt_B   = slope(θ_B, zgrid).*zgrid[1:end-1]./θ_B[1:end-1]
tt_H   = slope(θ_H, zgrid).*zgrid[1:end-1]./θ_H[1:end-1]
idx    = findfirst(x -> ~isnan(x) && x<50, tt_H)

plot(logz[idx:end-1], tt_B[idx:end], linecolor=:red, label=bonus, legend=:topright)
plot!(logz[idx:end-1], tt_H[idx:end], linecolor=:blue,label=rigid)
xaxis!(L" \log z")
yaxis!(L"\frac{d \log \theta }{d \log z}")
savefig(file_save*"dlogtheta.pdf")



