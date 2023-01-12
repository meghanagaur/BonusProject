cd(dirname(@__FILE__))

# turn off for cluster
ENV["GKSwstype"] = "nul"

include("functions/smm_settings.jl")        # SMM inputs, settings, packages, etc.
include("functions/calibration_vary_z1.jl") # vary z1 functions

using DataFrames, Binscatters, DelimitedFiles, LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =13, yguidefontsize=13, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 10)

## Logistics
vary_z_N     = 251
file_str     = "fix_chi0"
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

""" 
Function to simulate moments at estimated parameter values
"""
function simulate_moments(Params; check_mult = false)
    @unpack σ_η, χ, γ, hbar, ε = Params
    baseline = model(σ_η = σ_η, χ = χ, γ = γ, hbar = hbar, ε = ε) 
    out      = simulate(baseline, shocks; check_mult = check_mult)
    return out
end

# Get moments (check for multiplicity and verfiy solutions are the same)
output     = simulate_moments(Params; check_mult = false)   # skip multiplicity check
output2    = simulate_moments(Params; check_mult = true)    # check multiplicity of roots (slow)
@assert(isapprox(output.std_Δlw, output2.std_Δlw), 10^-8)   # check on multiplicity
@assert(isapprox(output.dlw1_du, output2.dlw1_du), 10^-8)   # check on multiplicity
@assert(isapprox(output.u_ss, output2.u_ss), 10^-8)         # check on multiplicity
@assert(isapprox(output.dlw_dly, output2.dlw_dly), 10^-8)   # check on multiplicity

# Unpack parameters
@unpack std_Δlw, dlw1_du, dlw_dly, u_ss, u_ss_2, avg_Δlw, dlw1_dlz, dlw_dly_2, dlY_dlz, dlu_dlz, std_u, std_z, std_Y, flag, flag_IR, IR_err  = output

# Estimated parameters
println("------------------------")
println("ESTIMATED PARAMETERS")
println("------------------------")
println("σ_η: \t"*string(round.(Params[:σ_η], digits=4)))
println("χ: \t"*string(round.(Params[:χ], digits=4)))
println("γ: \t"*string(round.(Params[:γ], digits=4)))
println("hbar: \t"*string(round.(Params[:hbar], digits=4)))
println("ε: \t"*string(round.(Params[:ε], digits=4)))

# Targeted moments
println("------------------------")
println("TARGETED MOMENTS")
println("------------------------")
println("std_Δlw: \t"*string(round.(std_Δlw, digits=4)))
println("dlw1_du: \t"*string(round.(dlw1_du, digits=4)))
println("dlw_dly: \t"*string(round.(dlw_dly, digits=4)))
println("u_ss: \t\t"*string(round.(u_ss, digits=4)))

# Untargeted, alternative moments
println("------------------------")
println("UNTARGETED MOMENTS")
println("------------------------")
println("dlu_dlz: \t"*string(round.(dlu_dlz, digits=4)))
println("std_logu: \t"*string(round.(std_u, digits=4)))
println("dlw_dly_2: \t"*string(round.(dlw_dly_2, digits=4)))
println("u_ss_2: \t"*string(round.(u_ss_2, digits=4)))
println("dlu_dlz: \t"*string(round.(dlu_dlz, digits=4)))
println("dlY_dlz: \t"*string(round.(dlY_dlz, digits=4)))
println("dlw1_dlz: \t"*string(round.(dlw1_dlz, digits=4)))
println("std_logz: \t"*string(round.(std_z, digits=4)))
println("std_logY: \t"*string(round.(std_Y, digits=4)))

## Vary initial productivity z_1 experiments

# Get the Bonus model aggregates
@unpack σ_η, χ, γ, hbar, ε = Params
modd  = model(N_z = vary_z_N, χ = χ, γ = γ, hbar = hbar, ε = ε, σ_η = σ_η);
@unpack w_0_B, θ_B, W_B, Y_B, ω_B, J_B, a_B, z_ss_idx, zgrid, aflag, modds = vary_z1(modd);
# Get the Hall aggregates
a_H, W_H, J_H, Y_H, θ_H = solveHall(modd, z_ss_idx, Y_B, W_B)

# Plot labels
rigid    = "Rigid Wage: Fixed w and a"
bonus    = "Incentive Pay: Variable w and a"
logz     = log.(zgrid)
minz_idx = max( findfirst(x -> x >=  -0.1, logz)  , max(findfirst(x -> x > 10^-6, θ_H), findfirst(x -> x > 10^-6, θ_B)))
maxz_idx = findfirst(x -> x >=  0.1, logz) 

# Plot profits
p1 = plot(logz, J_B, linecolor=:red, label=bonus, legend=:topleft);
plot!(logz, J_H, linecolor=:blue,label=rigid);
xaxis!(L"\log z_1");
yaxis!(L"J(z_1)");
xlims!((-0.1,0.1));
savefig(p1, file_save*"profits.pdf")

# Plot effort 
p2 = plot(logz, a_B, linecolor=:red, label=bonus, legend=:topleft);
hline!([a_H], linecolor=:blue, label=rigid);
xaxis!(L"\log z_1");
yaxis!(L"a(z_1|z_1)");
xlims!((-0.1,0.1));
savefig(p2, file_save*"efforts.pdf")

# Plot wages
p3 = plot(logz, W_B, linecolor=:red, label=bonus, legend=:topleft);
hline!([W_H], linecolor=:blue, label=rigid);
xaxis!(L"\log z_1");
yaxis!(L"W(z_1)");
xlims!((-0.1,0.1));
savefig(p3, file_save*"wages.pdf")

# Plot tightness
p4 = plot(logz, θ_B, linecolor=:red, label=bonus, legend=:topleft);
plot!(logz, θ_H, linecolor=:blue, label=rigid);
xaxis!(L"\log z_1");
yaxis!(L"\theta(z_1)");
xlims!((-0.1,0.1));
savefig(p4, file_save*"tightness.pdf")

# Plot tightness fluctuations: dlog θ / d log z
tt_B   = slope(θ_B, zgrid).*zgrid[1:end-1]./θ_B[1:end-1]
tt_H   = slope(θ_H, zgrid).*zgrid[1:end-1]./θ_H[1:end-1]
idx1   = findfirst(x -> ~isnan(x) && x<100, tt_H)

p5 = plot(logz[idx1:maxz_idx], tt_B[idx1:maxz_idx], linecolor=:red, label=bonus, legend=:topright);
plot!(logz[idx1:maxz_idx], tt_H[idx1:maxz_idx], linecolor=:blue,label=rigid);
xaxis!(L" \log z_1");
yaxis!(L"\frac{d \log \theta(z_1) }{d \log z_1}");
savefig(p5, file_save*"dlogtheta.pdf")

# Isolate effort/wage movements
p1 = plot( zgrid, Y_B , label="Variable a", linecolor=:red, linewidth=3);
plot!(p1, zgrid, Y_H, label="Fixed a", linecolor=:blue);
ylabel!(L"Y");
xlabel!(L"z_1");
p2 = plot(zgrid, W_B, label="Variable w",linecolor=:red);
hline!(p2, [W_H], label="Fixed w",linecolor=:blue);
ylabel!(L"W");
xlabel!(L"z_1");
plot(p1, p2, layout = (2, 1), legend=:topleft);
savefig(file_save*"y_w_movements.pdf")

# Compute dJ/dz when C term is 0
@unpack P_z, zgrid, N_z, ρ, β, s = modd;

# Solve for dJ/dz when C term = 0 (direct effect)
JJ_EVT   = zeros(length(zgrid)) 
Threads.@threads for iz = 1:N_z

    # Initialize guess of direct effect
    v0     = zgrid./(1-ρ*β*(1-s))
    v0_new = zeros(N_z)
    iter   = 1
    err    = 10
    
    # solve via simple value function iteration
    @inbounds while err > 10^-10 && iter < 1000
        v0_new = modds[iz].az.*zgrid + ρ*β*(1-s)*P_z*v0
        err    = maximum(abs.(v0_new - v0))
        v0     = copy(v0_new)
        iter +=1
    end

    JJ_EVT[iz]   = v0[iz]/zgrid[iz]

end

# Solve for dJ/dz in Hall 
JJ_H_2   = zeros(length(zgrid)) 
Threads.@threads for iz = 1:N_z

    # Initialize guess of direct effect
    v0     = zgrid./(1-ρ*β*(1-s))
    v0_new = zeros(N_z)
    iter   = 1
    err    = 10
    
    # solve via simple value function iteration
    @inbounds while err > 10^-10 && iter < 1000
        v0_new = zgrid + ρ*β*(1-s)*P_z*v0
        err    = maximum(abs.(v0_new - v0))
        v0     = copy(v0_new)
        iter +=1
    end

    JJ_H_2[iz]   = a_H*v0[iz]/zgrid[iz]

end

# Compute C term, Bonus, Hall
JJ_B      = slope(J_B, zgrid; diff = "central");
JJ_H      = slope(J_H, zgrid; diff = "central");
minz_idx  = minz_idx+1 # move up to account for backwards differencing 

plot(zgrid[minz_idx+1:maxz_idx], JJ_B[minz_idx+1:maxz_idx]-JJ_H[minz_idx+1:maxz_idx], linecolor=:red, label = bonus)
plot(zgrid[minz_idx+1:maxz_idx], JJ_B[minz_idx+1:maxz_idx]-JJ_EVT[minz_idx+1:maxz_idx], linecolor=:red, label = bonus)
plot(zgrid[minz_idx+1:maxz_idx], JJ_H[minz_idx+1:maxz_idx]-JJ_H_2[minz_idx+1:maxz_idx], linecolor=:red)

JJ_EVT[z_ss_idx]-JJ_B[z_ss_idx]     # c term
JJ_EVT[z_ss_idx]-JJ_H_2[z_ss_idx]   # c term
JJ_EVT[z_ss_idx]-JJ_H[z_ss_idx]     # c term

## C term graphs
p1 = plot(logz[minz_idx:maxz_idx], JJ_EVT[minz_idx:maxz_idx], linecolor=:black, label = "Incentive Pay: No C term", legend =:bottomright);
plot!(logz[minz_idx:maxz_idx], JJ_B[minz_idx:maxz_idx], linecolor=:red,  label = bonus);
plot!(logz[minz_idx:maxz_idx], JJ_H[minz_idx:maxz_idx], linecolor=:blue, label = rigid);
plot!(logz[minz_idx:maxz_idx], JJ_H_2[minz_idx:maxz_idx], linecolor=:yellow, label = "Rigid wage: Analytical", legend =:bottom)
vline!([logz[z_ss_idx]], label="");
xaxis!(L"z_1");
yaxis!(L"\frac{d J(z_1) }{d z_1}");
savefig(p1, file_save*"hall_bonus_cterm.pdf")

# Plot the C term
c_term = JJ_EVT -  JJ_B
p2 = plot(logz[minz_idx:maxz_idx], c_term[minz_idx:maxz_idx], legend =:false);
xaxis!(L"\log z_1");
yaxis!("C term");
savefig(p2, file_save*"cterm.pdf")

# Print the C term at steadty state
println("C term at μ_z: \t"*string(round(c_term[z_ss_idx],digits=5)))

## Scatter plot of log emploment
T_sim     = 5000
burnin    = 10000
bonus_sim = simulate_employment(modd, T_sim, burnin, θ_B; minz_idx = 1, u0 = 0.067)
hall_sim  = simulate_employment(modd, T_sim, burnin, θ_H; minz_idx = 1, u0 = 0.067)

# n_t and z_t 
N_B       = bonus_sim.nt
N_H       = hall_sim.nt
zt_B      = zgrid[bonus_sim.zt_idx]
zt_H      = zgrid[hall_sim.zt_idx]
@assert(zt_B == zt_H)

# raw scatter plot
p1 = plot(log.(zt_B), log.(N_B), seriestype=:scatter, label=bonus, legend=:bottomright, ms=:3, mc=:red);
plot!(log.(zt_B), log.(N_H), seriestype=:scatter, label=rigid, ms=:3, mc=:blue);
xlabel!(L"\log z_t");
ylabel!(L"\log n_t");
savefig(p1, file_save*"scatter_logn.pdf")

# Produce binscatter of log employment against log z
nbins  = 100
df     = DataFrame(n = log.(N_B), z = log.(zt_B), model = "bonus")  # bonus
df_H   = DataFrame(n = log.(N_H), z = log.(zt_B), model = "hall" )  # hall
append!(df, df_H)

p2 = binscatter(groupby(df, :model), @formula(n ~ z), nbins; markersize = 4, seriestype = :linearfit, 
        labels=[bonus rigid], markercolor= [:red :blue], linecolor = [:red :blue], legend=:bottomright);
ylabel!(L"\log n_t");
xlabel!(L"\log z_t");
savefig(file_save*"binscatter_logn.pdf")

#= simulate N = 10000 paths and compute average
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

