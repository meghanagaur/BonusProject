cd(dirname(@__FILE__))

include("functions/smm_settings.jl")        # SMM inputs, settings, packages, etc.
include("functions/calibration_vary_z1.jl") # vary z1 functions

using DelimitedFiles, LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =13, yguidefontsize=13, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 9)

## Logistics
file_str     = "fix_eps03_highu"
file_pre     = "runs/jld/pretesting_"*file_str*".jld2"  # pretesting data location
file_est     = "runs/jld/estimation_"*file_str*".txt"   # estimation output location
file_save    = "figs/vary-z1/"*file_str*"/"             # file to-save 
mkpath(file_save)

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

""" 
Function to simulate moments at estimated parameter values
"""
function simulate_moments(Params; check_mult = false)
    @unpack σ_η, χ, γ, hbar, ε = Params
    baseline = model(σ_η = σ_η, χ = χ, γ = γ, hbar = hbar, ε = ε) 
    out      = simulate(baseline, shocks; check_mult = check_mult)
    return out
end

# Get moments (check multiplicity)
output2    = simulate_moments(Params; check_mult = true)
output     = simulate_moments(Params; check_mult = false)

@unpack std_Δlw, dlw1_du, dlw_dly, u_ss, u_ss_2, avg_Δlw, dlw1_dlz, dlw_dly_2, dlY_dlz, dlu_dlz, std_u, std_z, std_Y, flag, flag_IR, IR_err  = output

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
round(dlw_dly_2,digits=4)
round(u_ss,digits=4)
round(dlu_dlz,digits=4)
round(std_u,digits=4)

# Addiitonal moments
round(u_ss_2, digits=4)
round(dlu_dlz, digits=4)
round(dlY_dlz, digits=4)
round(dlw1_dlz, digits=4)
round(std_z, digits=4)
round(std_Y, digits=4)
round(std_w0, digits=4)

## Vary z1 experiments

# Get the Bonus model aggregates
@unpack σ_η, χ, γ, hbar, ε = Params
modd  = model(N_z = 51, χ = χ, γ = γ, hbar = hbar, ε = ε, σ_η = σ_η)
@unpack w_0_B, θ_B, W_B, Y_B, ω_B, J_B, a_B, z_ss_idx, zgrid, aflag = vary_z1(modd)

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
idx    = findfirst(x -> ~isnan(x) && x<100, tt_H)

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

# check dJ/dz1
@unpack P_z, zgrid, ρ, β, s = modd

# Solve for expected PV of sum of the z_t's
sol          = solveModel(modd)
@unpack az   = sol
exp_az       = zeros(length(zgrid)) 
@inbounds for (iz, z1) in enumerate(zgrid)

    z0_idx  = findfirst(isequal(z1), zgrid)  # index of z0 on zgrid
    
    # initialize guesses
    v0     = zgrid./(1-ρ*β*(1-s))
    v0_new = zeros(N_z)
    iter   = 1
    err    = 10
    
    # solve via simple value function iteration
    @inbounds while err > 10^-10 && iter < 500
        v0_new = az.*zgrid + ρ*β*(1-s)*P_z*v0
        err    = maximum(abs.(v0_new - v0))
        v0     = copy(v0_new)
        iter +=1
    end

    exp_az[iz]   = v0[z0_idx]

end

JJ_EVT = exp_az/zgrid[z_ss_idx]
JJ_B   = slope(J_B, zgrid)
JJ_H   = slope(J_H, zgrid)

plot( logz, JJ_EVT, legend=:bottomright)
plot!(logz[1:end-1], JJ_B)
plot!(logz[1:end-1], JJ_H)

# Scatter plot of log emploment
T_sim     = 5000
burnin    = 10000
minz_idx  = max(findfirst(x -> x >= 10^-5, θ_H), findfirst(x -> x >= 10^-5, θ_B))

N_B       = simulate_employment(modd, T_sim, burnin, θ_B, minz_idx; u0 = 0.067).nt
N_H       = simulate_employment(modd, T_sim, burnin, θ_H, minz_idx; u0 = 0.067).nt

zt_B      = zgrid[simulate_employment(modd, T_sim, burnin, θ_H, minz_idx; u0 = 0.067).zt_idx]
zt_H      = zgrid[simulate_employment(modd, T_sim, burnin, θ_H, minz_idx; u0 = 0.067).zt_idx]
@assert(zt_B == zt_H)

plot(log.(zt_B), log.(N_B), seriestype=:scatter, label=bonus, legend=:bottomright)
plot!(log.(zt_B), log.(N_H), seriestype=:scatter, label=rigid)

savefig(file_save*"logn.pdf")

#=simulate N = 10000 paths and compute average
N_sim          = 10000
T_sim          = 1000
exp_az                              = zeros(N_sim, 1) 
@unpack z_shocks, z_shocks_idx      = simulateZShocks(P_z, zgrid; N = N_sim, T = T_sim)

Threads.@threads for n = 1:N_sim

    for t =1:T_sim
        #rt   = mapreduce(x -> ρ^x, +, [0:t-1;]) 
        rt    = ρ^(t-1) #(1-ρ^(t-1) )/(1-ρ)
        exp_az[n] += rt*((β*(1-s))^(t-1))*az[z_shocks_idx[n,t]]*z_shocks[n,t]
    end

end

z1   = unique(z_shocks[:,1])
dJdz = mean(exp_az)/z1
=#

