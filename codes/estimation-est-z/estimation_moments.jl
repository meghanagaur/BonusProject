cd(dirname(@__FILE__))

# Produce main figures/moments for the paper

# turn off for cluster
ENV["GKSwstype"] = "nul"

# Load helper files
include("functions/smm_settings.jl")                    # SMM inputs, settings, packages, etc.
include("functions/moments.jl")                         # vary z1 functions

using DataFrames, Binscatters, DelimitedFiles, LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =13, yguidefontsize=13, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 12)

## Logistics
vary_z_N             = 21 #251
N_sim_macro          = 10^4
N_sim_macro_workers  = 5*10^3

file_str     = "fix_rho_eps03_iota08"
file_pre     = "smm/jld/pretesting_"*file_str*".jld2"   # pretesting data location
file_est     = "smm/jld/estimation_"*file_str*".txt"    # estimation output location
file_save    = "figs/vary-z1/"*file_str*"/"             # file to-save 
mkpath(file_save)
println("File name: "*file_str)

# Load output
est_output = readdlm(file_est, ',', Float64)            # open estimation output across all jobs
@unpack moms, fvals, pars, mom_key, param_bounds, param_est, param_vals, data_mom, J, W = load(file_pre) # pretesting output

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
function simulate_moments(modd, shocks; check_mult = false)
    out      = simulate(modd, shocks; check_mult = check_mult)
    return out
end

# Unpack parameters
@unpack σ_η, χ, γ, hbar, ε, ρ, σ_ϵ, ι = Params

# Get moments (check for multiplicity and verfiy solutions are the same)
modd       = model(σ_η = σ_η, χ = χ, γ = γ, hbar = hbar, ε = ε, ρ = ρ, σ_ϵ = σ_ϵ, ι = ι) 
shocks     = rand_shocks(; N_sim_macro = N_sim_macro, N_sim_macro_workers = N_sim_macro_workers)
output     = simulate_moments(modd, shocks; check_mult = false)   # skip multiplicity check
output2    = simulate_moments(modd, shocks; check_mult = true)    # check multiplicity of roots (slow)

@assert(isapprox(output.std_Δlw, output2.std_Δlw), 10^-8)   # check on multiplicity of effort 
@assert(isapprox(output.dlw1_du, output2.dlw1_du), 10^-8)   # check on multiplicity of effort
@assert(isapprox(output.u_ss, output2.u_ss), 10^-8)         # check on multiplicity of effort
@assert(isapprox(output.dlw_dly, output2.dlw_dly), 10^-8)   # check on multiplicity of effort 
@assert(isapprox(output.alp_ρ, output2.alp_ρ), 10^-8)       # check on multiplicity of effort 
@assert(isapprox(output.alp_σ, output2.alp_σ), 10^-8)       # check on multiplicity of effort 

# Unpack parameters
@unpack std_Δlw, dlw1_du, dlw_dly, u_ss, alp_ρ, alp_σ, u_ss_2, dlu_dlz, std_u, flag, flag_IR, IR_err  = output

# Estimated parameters
println("------------------------")
println("ESTIMATED PARAMETERS")
println("------------------------")
println("σ_η: \t\t"*string(round.(Params[:σ_η], digits=4)))
println("χ: \t\t"*string(round.(Params[:χ], digits=4)))
println("γ: \t\t"*string(round.(Params[:γ], digits=4)))
println("hbar: \t\t"*string(round.(Params[:hbar], digits=4)))
println("ε: \t\t"*string(round.(Params[:ε], digits=4)))
println("ρ: \t\t"*string(round.(Params[:ρ], digits=4)))
println("σ_ϵ: \t\t"*string(round.(Params[:σ_ϵ], digits=4)))
println("ι: \t\t"*string(round.(Params[:ι], digits=4)))

# Targeted moments
println("------------------------")
println("TARGETED MOMENTS")
println("------------------------")
println("std_Δlw: \t"*string(round.(std_Δlw, digits=4)))
println("dlw1_du: \t"*string(round.(dlw1_du, digits=4)))
println("dlw_dly: \t"*string(round.(dlw_dly, digits=4)))
println("u_ss: \t\t"*string(round.(u_ss, digits=4)))
println("alp_ρ: \t\t"*string(round.(alp_ρ, digits=4)))
println("alp_σ: \t\t"*string(round.(alp_σ, digits=4)))

# Untargeted moments
println("------------------------")
println("UNTARGETED MOMENTS")
println("------------------------")
println("u_ss_2: \t"*string(round.(u_ss_2, digits=4)))
println("dlu_dlz: \t"*string(round.(dlu_dlz, digits=4)))
println("std_logu: \t"*string(round.(std_u, digits=4)))
println("θ(μ_z): \t"*string(round.(solveModel(modd).θ, digits=4)))

## Vary initial productivity z_1 experiments

# Get the Bonus model aggregates
modd       = model(N_z = vary_z_N, χ = χ, γ = γ, hbar = hbar, ε = ε, σ_η = σ_η, ι = ι);
modd_chi0  = model(N_z = vary_z_N, χ = 0, γ = γ, hbar = hbar, ε = ε, σ_η = σ_η, ι = ι);
bonus      = vary_z1(modd);
bonus_chi0 = vary_z1(modd_chi0);

# Get the Hall aggregates
hall       = solveHall(modd, bonus.Y, bonus.W)

# Plot labels
rigid      = "Rigid Wage: Fixed w and a"
fip        = "Incentive Pay: Variable w and a"
ip         = "Incentive pay: No bargaining"
zgrid      = modd.zgrid
logz       = log.(zgrid)
minz_idx   = max( findfirst(x -> x >=  -0.05, logz)  , max(findfirst(x -> x > 10^-6, hall.θ), findfirst(x -> x > 10^-6, bonus.θ)))
maxz_idx   = findlast(x -> x <=  0.05, logz)
maxz_idx   = isnothing(maxz_idx) ? vary_z_N : maxz_idx
range_1    = minz_idx:maxz_idx
dz         = 0.05

# Plot EPDV of profits
p1 = plot(logz, bonus.J, linecolor=:red, label=fip, legend=:topleft);
plot!(logz, hall.J, linecolor=:blue,label=rigid);
plot!(logz, bonus_chi0.J, linecolor=:cyan,label=ip, linestyle=:dash);
xaxis!(L"\log z_1");
yaxis!(L"J(z_1)");
xlims!((-dz, dz));

savefig(p1, file_save*"profits.pdf")

# Plot effort 
p2 = plot(logz, bonus.a, linecolor=:red, label=fip, legend=:topleft);
hline!([hall.a], linecolor=:blue, label=rigid);
plot!(logz, bonus_chi0.a, linecolor=:cyan,label=ip, linestyle=:dash);
xaxis!(L"\log z_1");
yaxis!(L"a(z_1|z_1)");
xlims!((-dz, dz));

savefig(p2, file_save*"efforts.pdf")

# Plot EPDV of wages
p3 = plot(logz, bonus.W, linecolor=:red, label=fip, legend=:topleft);
hline!([hall.W], linecolor=:blue, label=rigid);
plot!(logz, bonus_chi0.W, linecolor=:cyan,label=ip, linestyle=:dash);
xaxis!(L"\log z_0");
yaxis!(L"W(z_0)");
xlims!((-dz, dz));

savefig(p3, file_save*"wages.pdf")

# Plot tightness
p4 = plot(logz, bonus.θ, linecolor=:red, label=fip, legend=:topleft);
plot!(logz, hall.θ, linecolor=:blue, label=rigid);
plot!(logz, bonus_chi0.θ, linecolor=:cyan,label=ip, linestyle=:dash);
xaxis!(L"\log z_0");
yaxis!(L"\theta(z_0)");
xlims!((-dz, dz))

savefig(p4, file_save*"tightness.pdf")

# Plot omega (value of unemployment)
p5 = plot(logz, bonus.ω, linecolor=:red, label=fip, legend=:topleft);
plot!(logz, bonus_chi0.ω, linecolor=:cyan,label=ip, linestyle=:dash);
xaxis!(L"\log z_0");
yaxis!(L"\omega(z_0)");
xlims!((-dz, dz));

savefig(p5, file_save*"omega.pdf")

# Plot tightness fluctuations: dlog θ / d log z
tt_B    = slope(bonus.θ, zgrid).*zgrid./bonus.θ
tt_H    = slope(hall.θ, zgrid).*zgrid./hall.θ
tt_B0   = slope(bonus_chi0.θ, zgrid).*zgrid./bonus_chi0.θ
idx1    = findfirst(x -> ~isnan(x) && x < 120, tt_H) # start at reasonable scale
range_2 = idx1:maxz_idx

p6 = plot(logz[range_2], tt_B[range_2], linecolor=:red, label=fip, legend=:topright);
plot!(logz[range_2], tt_H[range_2], linecolor=:blue,label=rigid);
plot!(logz[range_2], tt_B0[range_2], linecolor=:cyan,label=ip, linestyle=:dash);
xaxis!(L" \log z_0");
yaxis!(L"\frac{d \log \theta(z_0) }{d \log z_1}");

savefig(p6, file_save*"dlogtheta.pdf")

# Compute the C term via the IR constraint
ω_B  = bonus.ω
dω_B = slope(ω_B, modd.zgrid)
@unpack χ, β, ρ, s, μ_z = modd 
B    = (χ/(1-β*ρ))
A    = (log(γ) + β*B*(1-ρ)*μ_z)/(1-β)
ω_2  = A .+ B*logz
@assert(maximum(abs.(ω_B - ω_2)) < 10^-5 )
dω_2 = B./zgrid

@assert( minimum( max.(abs.(dω_B - dω_2) .< 10^-3, isnan.(dω_B)) )  == 1 )

# Compute the direct effect on the IR constraint
dIR  = dω_2.*(ρ*β - 1)/(1-ρ*β*(1-s))

#= Isolate effort/wage movements
p1 = plot( zgrid, bonus.Y , label="Variable a", linecolor=:red, linewidth=3);
plot!(p1, zgrid, hall.Y, label="Fixed a", linecolor=:blue);
ylabel!(L"Y");
xlabel!(L"z_1");
p2 = plot(zgrid, bonus.W, label="Variable w",linecolor=:red);
hline!(p2, [hall.W], label="Fixed w",linecolor=:blue);
ylabel!(L"W");
xlabel!(L"z_1");
plot(p1, p2, layout = (2, 1), legend=:topleft);
savefig(file_save*"y_w_movements.pdf")
=#

# Compute dJ/dz when C term is 0
@unpack P_z, zgrid, N_z, ρ, β, s, z_ss_idx = modd

# Solve for dJ/dz when C term = 0 (direct effect), conditional on initial z_1
JJ_EVT   = zeros(length(zgrid)) 
Threads.@threads for iz = 1:N_z

    # Initialize guess of direct effect
    v0     = zgrid./(1-ρ*β*(1-s))
    v0_new = zeros(N_z)
    iter   = 1
    err    = 10
    
    # solve via simple value function iteration
    @inbounds while err > 10^-10 && iter < 1000
        v0_new = bonus.modds[iz].az.*zgrid + ρ*β*(1-s)*P_z*v0
        err    = maximum(abs.(v0_new - v0))
        v0     = copy(v0_new)
        iter +=1
    end

    JJ_EVT[iz]   = v0[iz]/zgrid[iz]

end

# Solve for dJ/dz in Hall directly using above formula
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

    JJ_H_2[iz]   = hall.a*v0[iz]/zgrid[iz]

end

# Compute C term, Bonus, Hall
JJ_B      = slope(bonus.J, zgrid; diff = "central")
JJ_H      = slope(hall.J, zgrid; diff = "central")
JJ_B0     = slope(bonus_chi0.J, zgrid; diff = "central")

plot(zgrid[range], JJ_B[range] - JJ_H[range], linecolor=:red, label = fip)
plot(zgrid[range], JJ_B[range] - JJ_EVT[range], linecolor=:red, label = fip)
plot(zgrid[range], JJ_H[range] - JJ_H_2[range], linecolor=:red, label = fip)
plot(zgrid[range], JJ_B0[range] - JJ_EVT[range], linecolor=:red, label = fip)

JJ_EVT[z_ss_idx]-JJ_B[z_ss_idx]     # c term
JJ_EVT[z_ss_idx]-JJ_H_2[z_ss_idx]   # c term
JJ_EVT[z_ss_idx]-JJ_H[z_ss_idx]     # c term
JJ_EVT[z_ss_idx]-JJ_B0[z_ss_idx]    # c term

## C term graphs
p1 = plot(logz[minz_idx:maxz_idx], JJ_EVT[minz_idx:maxz_idx], linecolor=:black, label = "Incentive Pay: No C term", legend =:bottomright);
plot!(logz[minz_idx:maxz_idx], JJ_B[minz_idx:maxz_idx], linecolor=:red,  label = fip);
plot!(logz[minz_idx:maxz_idx], JJ_H[minz_idx:maxz_idx], linecolor=:blue, label = rigid);
plot!(logz[minz_idx+1:maxz_idx], JJ_B0[minz_idx+1:maxz_idx], linecolor=:yellow, label = ip, legend =:bottom)
xaxis!(L"z_0");
yaxis!(L"\frac{d J(z_0) }{d z_0}");
xlims!((-dz, dz));

savefig(p1, file_save*"hall_bonus_cterm.pdf")

# Plot the C term
c_term = JJ_B - JJ_EVT 
plot(logz[minz_idx:maxz_idx], c_term[minz_idx:maxz_idx], legend=:false);
xaxis!(L"\log z_1");
yaxis!("C term");
savefig(file_save*"cterm.pdf")

# Print the C term at steadty state
println("C term at μ_z: \t"*string(round(c_term[z_ss_idx],digits=5)))

# plot the lower bound
plot(logz[range_1], c_term[range_1]./dIR[range_1], legend=:bottomright, label=L"\mu(z)" )
plot!(logz[range_1], bonus.w_0[range_1], linestyle=:dash, label=L"w_{-1}(z)")
xaxis!(L"\log z_0");

savefig(file_save*"cterm_multiplier.pdf")

## Scatter plot of log employment
T_sim     = 5000
burnin    = 10000
bonus_sim = simulate_employment(modd, T_sim, burnin, bonus.θ; minz_idx = minz_idx)
hall_sim  = simulate_employment(modd, T_sim, burnin, hall.θ; minz_idx = minz_idx)

# n_t and z_t 
N_B       = bonus_sim.nt
N_H       = hall_sim.nt
zt_B      = zgrid[bonus_sim.zt_idx]
zt_H      = zgrid[hall_sim.zt_idx]
@assert(zt_B == zt_H)

# raw scatter plot
p1 = plot(log.(zt_B), log.(N_B), seriestype=:scatter, label=fip, legend=:bottomright, ms=:3, mc=:red);
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
        labels=[fip rigid], markercolor= [:red :blue], linecolor = [:red :blue], legend=:bottomright);
ylabel!(L"\log n_t");
xlabel!(L"\log z_t");
savefig(file_save*"binscatter_logn.pdf")

