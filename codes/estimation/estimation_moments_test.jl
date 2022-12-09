cd(dirname(@__FILE__))

include("functions/smm_settings.jl")        # SMM inputs, settings, packages, etc.
include("functions/calibration_vary_z1.jl") # vary z1 functions

using DelimitedFiles, LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =13, yguidefontsize=13, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 9)

## Logistics
ε    = 1.63051 #0.5
σ_η  = 0.288581 #0.3
χ    = 0.0 #0.0
γ    = 0.279864 #0.65
hbar = 1.2 #1.0

# Define the parameter values
param_vals  = OrderedDict{Symbol, Real}([ 
                (:ε,   ε),       
                (:σ_η, σ_η),     
                (:χ, χ),         
                (:γ, γ),         
                (:hbar, hbar) ]) 

""" 
Function to simulate moments at estimated parameter values
"""
function simulate_moments(params; check_mult = false)
    @unpack σ_η, χ, γ, hbar, ε = params
    baseline = model(σ_η = σ_η, χ = χ, γ = γ, hbar = hbar, ε = ε) 
    out      = simulate(baseline, shocks; check_mult = check_mult)
    return out
end

# Get moments
output     = simulate_moments(param_vals; check_mult = true)
@unpack std_Δlw, dlw1_du, dlw_dly, u_ss, u_ss_2, avg_Δlw, dlw1_dlz, dlY_dlz, dlu_dlz, std_u, std_z, std_Y, std_w0, flag, flag_IR, IR_err  = output

## Vary z1 experiments

# Get the Bonus model aggregates
@unpack σ_η, χ, γ, hbar, ε = param_vals
modd  = model(N_z = 11, σ_η = σ_η, χ = χ, γ = γ, hbar = hbar, ε = ε)
@unpack w_0_B, θ_B, W_B, Y_B, ω_B, J_B, a_B, z_ss_idx, zgrid, aflag = vary_z1(modd; check_mult = false)

# Get the Hall analogues
a_H, W_H, J_H, Y_H, θ_H = solveHall(modd, z_ss_idx, Y_B, W_B);

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
idx    = findfirst(x -> ~isnan(x) && x<60, tt_H)

plot(logz[idx:end-1], tt_B[idx:end], linecolor=:red, label=bonus, legend=:topright)
plot!(logz[idx:end-1], tt_H[idx:end], linecolor=:blue,label=rigid)
xaxis!(L" \log z")
yaxis!(L"\frac{d \log \theta }{d \log z}")
savefig(file_save*"dlogtheta.pdf")

# isolate effort/wage movements
p1 = plot( zgrid, Y_B , label="Variable a", linecolor=:red, linewidth=3)
plot!(p1, zgrid, Y_H, label="Fixed a", linecolor=:blue)
ylabel!(L"Y")
xlabel!(L"z_1")
p2= plot(W_B, label="Variable w",linecolor=:red)
hline!(p2, [W_H], label="Fixed w",linecolor=:blue)
ylabel!(L"W")
xlabel!(L"z_0")

plot(p1, p2, layout = (2, 1), legend=:topleft)
savefig(file_save*"y_w_movements.pdf")

# Plot dJ / d z
JJ_B   = slope(J_B, zgrid)
JJ_H   = slope(J_H, zgrid)
plot(logz[1:end-1], JJ_B, linecolor=:red, label=bonus, legend=:topright)
plot!(logz[1:end-1], JJ_H, linecolor=:blue,label=rigid)
xaxis!(L" \log z")
yaxis!(L"\frac{d J }{d z}")
