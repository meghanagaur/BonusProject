cd(dirname(@__FILE__))

# Produce main figures/moments for the paper

# turn off for cluster
ENV["GKSwstype"] = "nul"

# Load helper files
include("functions/smm_settings.jl")                    # SMM inputs, settings, packages, etc.
include("functions/moments.jl")                         # vary z1 functions

using DataFrames, Binscatters, DelimitedFiles, LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize = 13, yguidefontsize = 13, xtickfontsize=10, ytickfontsize=10,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px, legendfontsize = 12)

## Logistics
big_run      = false #true        
file_str     ="fix_hbar10_chi0" #ARGS[1]                              
file_pre     = "smm/jld/pretesting_"*file_str*".jld2"   # pretesting data location
file_est     = "smm/jld/estimation_"*file_str*".txt"    # estimation output location
file_save    = "figs/vary-z1/"*file_str*"/"             # file to-save 

mkpath(file_save)
println("File name: "*file_str)

# Settings for simulation
if big_run == false
    vary_z_N             = 51          # number of gridpoints for first-order 
    N_sim_macro          = 10^4        # number of panels for macro stats exc. ALP
    N_sim_macro_workers  = 1000        # number of workers for ALP simulation
    N_sim_macro_est_alp  = 1000        # number of panels for ALP simulation
else
    vary_z_N             = 201         # number of gridpoints for first-order 
    N_sim_macro          = 10^4        # number of panels for macro stats exc. ALP
    N_sim_macro_workers  = 5*10^3      # number of workers for ALP simulation
    N_sim_macro_est_alp  = 10^4        # number of panels for ALP simulation
end

# Load output
est_output = readdlm(file_est, ',', Float64)   # estimation output       
@unpack moms, fvals, pars, mom_key, param_bounds, param_est, param_vals, data_mom, J, W, fix_a = load(file_pre) # pretesting output

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

# Unpack parameters
@unpack σ_η, χ, γ, hbar, ε, ρ, σ_ϵ, ι = Params

# Get moments (check for multiplicity and verfiy solutions are the same)
modd       = model(σ_η = σ_η, χ = χ, γ = γ, hbar = hbar, ε = ε, ρ = ρ, σ_ϵ = σ_ϵ, ι = ι) 
shocks     = rand_shocks(; N_sim_macro = N_sim_macro, N_sim_macro_workers = N_sim_macro_workers, N_sim_macro_est_alp = N_sim_macro_est_alp)

if fix_a == false

    @time output     = simulate(modd, shocks; check_mult = false)         # skip multiplicity check
    @time output2    = simulate(modd, shocks; check_mult = true)          # check multiplicity of roots (slow)

    if (output2.flag == 0)
        @assert(isapprox(output.std_Δlw, output2.std_Δlw), 10^-8)   # check on multiplicity of effort 
        @assert(isapprox(output.dlw1_du, output2.dlw1_du), 10^-8)   # check on multiplicity of effort
        @assert(isapprox(output.u_ss, output2.u_ss), 10^-8)         # check on multiplicity of effort
        @assert(isapprox(output.dlw_dly, output2.dlw_dly), 10^-8)   # check on multiplicity of effort 
        @assert(isapprox(output.alp_ρ, output2.alp_ρ), 10^-8)       # check on multiplicity of effort 
        @assert(isapprox(output.alp_σ, output2.alp_σ), 10^-8)       # check on multiplicity of effort 
        println("effort multiplicity check passed")
    else
        println("effort multiplicity check violated")
    end
else 
    @time output = simulateFixedEffort(modd, shocks; a = Params[:a])    
end

# Unpack parameters
@unpack std_Δlw, dlw1_du, dlw_dly, u_ss, alp_ρ, alp_σ, u_ss_2, dlu_dly, std_u, flag, flag_IR, IR_err, std_z  = output

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
println("dlu_dly: \t"*string(round.(dlu_dly, digits=4)))
println("std_logu: \t"*string(round.(std_u, digits=4)))
println("std_logz: \t"*string(round.(std_z, digits=4)))

# Compute some extra moments
if fix_a == false
    @unpack θ, w_0, Y, az = solveModel(modd; noisy = false);
else 
    @unpack θ, w_0, Y, az = solveModelFixedEffort(modd; a = Params[:a], noisy = false);
end

println("a(μ_z): \t"*string(round.(az[modd.z_ss_idx], digits=4)))
println("θ(μ_z): \t"*string(round.(θ, digits=4)))
println("W (ss): \t"*string(round.(w_0/modd.ψ, digits=4)))
println("Y (ss): \t"*string(round.(Y, digits=4)))
println("W/Y (ss): \t"*string(round.(w_0/(modd.ψ*Y), digits=4)))

## Vary initial productivity z_0 

# Get the Bonus model aggregates
modd       = model(N_z = vary_z_N, χ = χ, γ = γ, hbar = hbar, ε = ε, σ_η = σ_η, ι = ι, ρ = ρ, σ_ϵ = σ_ϵ)
modd_chi0  = model(N_z = vary_z_N, χ = 0.0, γ = γ, hbar = hbar, ε = ε, σ_η = σ_η, ι = ι, ρ = ρ, σ_ϵ = σ_ϵ)

if fix_a == true
    bonus      = vary_z1(modd; fix_a = fix_a, a = Params[:a])
    bonus_chi0 = vary_z1(modd_chi0; fix_a = fix_a, a = Params[:a])
else 
    bonus      = vary_z1(modd; fix_a = fix_a)
    bonus_chi0 = vary_z1(modd_chi0; fix_a = fix_a)
end

@unpack P_z, zgrid, N_z, ρ, β, s, z_ss_idx, q, ι, κ, χ, μ_z, ψ, hp, logz = modd

# Get Hall aggregates
hall       = solveHall(modd, bonus.Y, bonus.W)

# Print out some cyclical fluctuations
dlY_dlz      = slopeFD(log.(max.(eps(), bonus.Y)), logz; diff = "central")
dlW_dlz      = slopeFD(log.(max.(eps(), bonus.W)), logz; diff = "central")
dla0_dlz     = slopeFD(log.(max.(eps(), bonus.a1)), logz; diff = "central")
tt_B         = slopeFD(bonus.θ, zgrid).*zgrid./bonus.θ
tt_H         = slopeFD(hall.θ, zgrid).*zgrid./hall.θ
tt_B0        = slopeFD(bonus_chi0.θ, zgrid).*zgrid./bonus_chi0.θ

println("dlY_dlz: \t"*string(round.(dlY_dlz[z_ss_idx], digits=4)))
println("dlW_dlz: \t"*string(round.(dlW_dlz[z_ss_idx], digits=4)))
println("dla_dlz: \t"*string(round.(dla0_dlz[z_ss_idx], digits=4)))
println("dlθ_dlz: \t"*string(round.(tt_B[z_ss_idx], digits=4)))

# Plot labels
rigid      = "Rigid Wage: fixed w and a"
fip        = "Incentive Pay: variable w and a"
ip         = "Incentive Pay, setting "*L"\chi = 0"
minz_idx   = max( findfirst(x -> x >=  -0.05, logz), findfirst(x -> x > 10^-8, bonus.θ))
maxz_idx   = findlast(x -> x <=  0.05, logz)
maxz_idx   = isnothing(maxz_idx) ? vary_z_N : maxz_idx
range_1    = minz_idx:maxz_idx

# Get decomposition components
@unpack JJ_EVT, WF, BWF, IWF, resid, total_resid = decomposition(modd, bonus; fix_a = fix_a)

## Compute C term and dJ/dz in Bonus, Hall
JJ_B      = slopeFD(bonus.J, zgrid; diff = "central")
JJ_H      = slopeFD(hall.J, zgrid; diff = "central")
JJ_B0     = slopeFD(bonus_chi0.J, zgrid; diff = "central")
c_term    = JJ_B - JJ_EVT

## Print the C term at steady state
println("C term at μ_z: \t"*string(round(c_term[z_ss_idx], digits=3)))

## Share of Incentive Wage Flexibility
println("------------------------")
println("WAGE FLEXIBILITY")
println("------------------------")

## Share of bargained wage flexibility
WF_chi0  = slopeFD(bonus_chi0.W, zgrid) # total wage flexibility
BWF_2    = -(c_term./WF)                # primary IWF measure

println("IWF Share #1: \t\t"*string(round(1 - BWF_2[z_ss_idx], digits = 3)))
println("IWF Share #2: \t\t"*string(round((IWF./WF)[z_ss_idx], digits = 3)))
println("WF with χ = 0/WF \t"*string(round((WF_chi0./WF)[z_ss_idx], digits = 3)))

if fix_a == false
   
    # Plot EPDV of profits
    p1 = plot(logz[range_1], bonus.J[range_1], linecolor=:red, label=fip, legend=:topleft)
    plot!(logz[range_1], hall.J[range_1], linecolor=:blue,label=rigid)
    plot!(logz[range_1], bonus_chi0.J[range_1], linecolor=:cyan,label=ip, linestyle=:dash)
    xaxis!(L"\log z_0")
    yaxis!(L"J(z_0)")

    savefig(p1, file_save*"profits.pdf")

    # Plot effort 
    p2 = plot(logz[range_1], bonus.a1[range_1], linecolor=:red, label=fip, legend=:topleft)
    hline!([hall.a], linecolor=:blue, label=rigid)
    plot!(logz[range_1], bonus_chi0.a1[range_1], linecolor=:cyan,label=ip, linestyle=:dash)
    xaxis!(L"\log z_0")
    yaxis!(L"a(z_0|z_0)")

    savefig(p2, file_save*"efforts.pdf")

    # Plot EPDV of wages
    p3 = plot(logz[range_1], bonus.W[range_1], linecolor=:red, label=fip, legend=:topleft)
    hline!([hall.W], linecolor=:blue, label=rigid)
    plot!(logz[range_1], bonus_chi0.W[range_1], linecolor=:cyan,label=ip, linestyle=:dash)
    xaxis!(L"\log z_0")
    yaxis!(L"W(z_0)")

    savefig(p3, file_save*"wages.pdf")

    # Plot tightness
    p4 = plot(logz[range_1], bonus.θ[range_1], linecolor=:red, label=fip, legend=:topleft)
    plot!(logz[range_1], hall.θ[range_1], linecolor=:blue, label=rigid)
    plot!(logz[range_1], bonus_chi0.θ[range_1], linecolor=:cyan,label=ip, linestyle=:dash)
    xaxis!(L"\log z_0")
    yaxis!(L"\theta(z_0)")

    savefig(p4, file_save*"tightness.pdf")

    # Plot omega (value of unemployment)
    p5 = plot(logz, bonus.ω, linecolor=:red, label=fip, legend=:topleft)
    plot!(logz, bonus_chi0.ω, linecolor=:cyan,label=ip, linestyle=:dash)
    xaxis!(L"\log z_0")
    yaxis!(L"\omega(z_0)")

    savefig(p5, file_save*"omega.pdf")

    # Plot tightness fluctuations: dlog θ / d log z
    idx1    = max( max( findfirst(x -> ~isnan(x) && ~iszero(x) && x < 120, tt_B)) , 
                    max( findfirst(x -> ~isnan(x)  && ~iszero(x)  && x < 120, tt_H))) # start at reasonable scale
    range_2 = idx1:maxz_idx

    p6 = plot(logz[range_2], tt_B[range_2], linecolor=:red, label=fip, legend=:topright)
    plot!(logz[range_2], tt_H[range_2], linecolor=:blue,label=rigid)
    plot!(logz[range_2], tt_B0[range_2], linecolor=:cyan,label=ip, linestyle=:dash)
    xaxis!(L" \log z_0")
    yaxis!(L"\frac{d \log \theta(z_0) }{d \log z_0}")

    savefig(p6, file_save*"dlogtheta.pdf")

    # Compute the direct effect on the IR constraint
    ω_B  = bonus.ω
    dω_B = slopeFD(ω_B, modd.zgrid)
    B    = (χ/(1-β*ρ))
    A    = (log(γ) + β*B*(1-ρ)*μ_z)/(1-β)
    ω_2  = A .+ B*logz

    @assert(maximum(abs.(ω_B - ω_2)) < 10^-4 )

    dω_2 = B./zgrid

    @assert( minimum( max.(abs.(dω_B - dω_2) .< 10^-3, isnan.(dω_B)) )  == 1 )

    dIR  = dω_2.*(ρ*β - 1)/(1-ρ*β*(1-s))

    plot(zgrid[range_2], JJ_B[range_2] , linecolor=:red, label = "bonus", linestyle=:dot)
    plot!(zgrid[range_2], JJ_EVT[range_2], linecolor=:black, label = "EVT")
    plot!(zgrid[range_2], JJ_B0[range_2], linecolor=:blue, label = "chi0 ", linestyle=:dash)
    plot!(zgrid[range_2], JJ_H[range_2] , linecolor=:green, label = "hall", linestyle=:dashdot)

    JJ_EVT[z_ss_idx] - JJ_B[z_ss_idx]     # c term
    JJ_EVT[z_ss_idx] - JJ_H[z_ss_idx]     # c term
    JJ_EVT[z_ss_idx] - JJ_B0[z_ss_idx]    # should be close to 0

    ## C term graphs
    p1 = plot(logz[range_2], JJ_EVT[range_2], linecolor=:black, label = "Incentive Pay: No C term", legend =:bottomright)
    #plot!(logz[range_2], JJ_B[range_2], linecolor=:red,  label = fip)
    plot!(logz[range_2], JJ_H[range_2], linecolor=:blue, label = rigid)
    plot!(logz[range_2], JJ_B0[range_2], linecolor=:yellow, label = ip, legend =:bottom)
    xaxis!(L"z_0")
    yaxis!(L"\frac{d J(z_0) }{d z_0}")

    savefig(p1, file_save*"dJ_dz_0_c_term.pdf")

    # plot the lower bound
    plot(logz[range_2], c_term[range_2]./dIR[range_2], legend=:bottomright, label=L"\mu(z)" )
    plot!(logz[range_2], bonus.w_0[range_2], linestyle=:dash, label=L"w_{-1}(z)")
    xaxis!(L"\log z_0")

    savefig(file_save*"cterm_multiplier.pdf")

    lm = c_term[range_1]./dIR[range_1]
    #plot(logz[range_1], slopeFD(bonus.w_0[range_1], zgrid[range_1]))
    #plot!(logz[range_1], slopeFD(lm, zgrid[range_1]),linestyle=:dash)

    ## Scatter plot of log employment
    T_sim     = 5000
    burnin    = 10000
    minz_idx  = max(findfirst(x -> x > 10^-6, hall.θ), findfirst(x -> x > 10^-6, bonus.θ))
    bonus_sim = simulate_employment(modd, T_sim, burnin, bonus.θ; minz_idx = minz_idx)
    hall_sim  = simulate_employment(modd, T_sim, burnin, hall.θ; minz_idx = minz_idx)

    # n_t and z_t 
    N_B       = bonus_sim.nt
    N_H       = hall_sim.nt
    zt_B      = zgrid[bonus_sim.zt_idx]
    zt_H      = zgrid[hall_sim.zt_idx]

    @assert(zt_B == zt_H)

    # raw scatter plot
    p1 = plot(log.(zt_B), log.(N_B), seriestype=:scatter, label=fip, legend=:bottomright, ms=:3, mc=:red)
    plot!(log.(zt_B), log.(N_H), seriestype=:scatter, label=rigid, ms=:3, mc=:blue)
    xlabel!(L"\log z_t")
    ylabel!(L"\log n_t")

    savefig(p1, file_save*"scatter_logn.pdf")

    ## Produce binscatter of log employment against log z
    nbins  = 100
    df     = DataFrame(n = log.(N_B), z = log.(zt_B), model = "bonus")  # bonus
    df_H   = DataFrame(n = log.(N_H), z = log.(zt_B), model = "hall" )  # hall
    append!(df, df_H)

    p2 = binscatter(groupby(df, :model), @formula(n ~ z), nbins; markersize = 4, seriestype = :linearfit, 
            labels=[fip rigid], markercolor= [:red :blue], linecolor = [:red :blue], legend=:false)
    ylabel!(L"\log n_t")
    xlabel!(L"\log z_t")

    savefig(file_save*"binscatter_logn.pdf")

    # plot the comparison of partial, total derivative of the residual + dJ/dz in Bonus model
    plot(logz[range_2], JJ_B[range_2], label=L"\frac{d J(z_0) }{d z_0}", legend=:bottomright)
    plot!(logz[range_2], -resid[range_2], label=L"\frac{\partial \kappa/q(z_0) }{\partial z_0}"*" (method 1 )", linestyle=:dashdot)
    plot!(logz[range_2],total_resid[range_2], label=L"\frac{d \kappa/q(z_0) }{d z_0}"*" (direct)", linestyle=:dash)

    savefig(file_save*"dJ_dz_resids.pdf")

    plot(logz[range_2], (BWF./WF)[range_2], label="BWF #1")
    plot!(logz[range_2], (BWF_2)[range_2],label="BWF #2", linestyle=:dash)
    ylabel!("BWF share")
    xlabel!(L"\log z_0")

    savefig(file_save*"WF_shares.pdf")

end

# Check convergence 
indices    = sortperm(est_output[:,1])
stop       = findlast(x -> x < 1, est_output[indices,1])
indices    = reverse(indices[1:stop])

p1 = plot()
p2 = plot()
p3 = plot()
p4 = plot()

for (k,v) in param_est
    if k == :σ_ϵ
        plot!(p1, est_output[indices,v+1], label = string(k))
    elseif k == :ε || k == :hbar
        plot!(p2, est_output[indices,v+1], label = string(k))
    elseif k == :χ
        plot!(p3, est_output[indices,v+1], label = string(k))
    else
        plot!(p4, est_output[indices,v+1], label = string(k))
    end
end

p5 = plot(p1, p2, p3, p4, layout = (2,2), legend=:topleft)
savefig(p5, file_save*"params_converge.pdf")

# function values
p6 = plot(est_output[indices,1], title="function values", legend=:false)
savefig(p6, file_save*"fvals_converge.pdf")

#=
# Vary params and compute BWF
if vary_params
    
    function bwf(modd)
        bonus           = vary_z1(modd)
        @unpack WF, BWF = decomposition(modd, bonus; fix_a = fix_a)
        return (BWF./WF)[modd.z_ss_idx], maximum(bonus.IR_flag*1.0)
    end

    # vary χ
    N_χ      = 30
    χ_grid   = LinRange(0.0, 1.0, N_χ)
    bwf_grid = zeros(2, N_χ)

    Threads.@threads for i = 1:N_χ
        bwf_grid[:, i] .=  bwf(model(χ = χ_grid[i], γ = γ, hbar = hbar, ε = ε, σ_η = σ_η, 
                             ι = ι, ρ = ρ, σ_ϵ = σ_ϵ))
    end

    plot(χ_grid, bwf_grid[1,:], label = "BWF share", legend=:right)
    xlabel!(L"\chi")
    ylabel!("BWF share")

    savefig(file_save*"bwf_vary_chi.pdf")

    # vary ε
    N_ε      = 30
    ε_grid   = LinRange(0.3, 3.0, N_ε)

    Threads.@threads for i = 1:N_ε
        bwf_grid[:,i] .=  bwf(model(χ = χ, γ = γ, hbar = hbar, ε = ε_grid[i], σ_η = σ_η, 
                                            ι = ι, ρ = ρ, σ_ϵ = σ_ϵ))
    end

    # check IR constraint
    max_idx = findfirst(isequal(1), bwf_grid[2,:])
    max_idx = isnothing(max_idx)  ? N_ε : max_idx
    range   = 1:max_idx

    plot(ε_grid[range], bwf_grid[1,range], label = "BWF share", legend=:right)
    xlabel!(L"\varepsilon")
    ylabel!("BWF share")

    savefig(file_save*"bwf_vary_eps.pdf")

    # vary hbar
    N_h        = 30
    hbar_grid  = LinRange(1.0, 4.0, N_h)

    Threads.@threads for i = 1:N_h
        bwf_grid[:,i] .=  bwf(model(χ = χ, γ = γ, hbar = hbar_grid[i], ε = ε, σ_η = σ_η, 
                             ι = ι, ρ = ρ, σ_ϵ = σ_ϵ))
    end

    # check IR constraint
    max_idx = findfirst(isequal(1), bwf_grid[2,:])
    max_idx = isnothing(max_idx)  ? N_h : max_idx
    range   = 1:max_idx

    plot(hbar_grid[range], bwf_grid[1,range], label = "BWF share #1", legend=:right)
    xlabel!(L"\bar{h}")
    ylabel!("BWF share")

    savefig(file_save*"bwf_vary_hbar.pdf")

    # vary σ_η
    N_s     = 30
    σ_grid  = LinRange(0.0, 0.5, N_s)

    Threads.@threads for i = 1:N_s
        bwf_grid[:,i] .=  bwf(model(χ = χ, γ = γ, hbar = hbar, ε = ε, σ_η = σ_grid[i], 
                                ι = ι, ρ = ρ, σ_ϵ = σ_ϵ))
    end

    # check IR constraint
    max_idx = findfirst(isequal(1), bwf_grid[2,:])
    max_idx = isnothing(max_idx)  ? N_s : max_idx
    range   = 1:max_idx

    p1 = plot(σ_grid[range], bwf_grid[1,range], ylabel = "BWF share #1")
    p2 = plot(σ_grid[range], bwf_grid[2,range], ylabel = "BWF share #2")
    xlabel!(L"\sigma_\eta")
    plot(p1, p2, layout = (2,1), legend=:false)

    savefig(file_save*"bwf_vary_sigma_eta.pdf")
end
=#

#=
# effort as a function of z, w_0
a_z(x) = effort(x, bonus.modds[z_ss_idx].w_0,  ψ, ε, hp, σ_η, hbar )
a_w(w) = effort(zgrid[z_ss_idx], w,  ψ, ε, hp, σ_η, hbar)

dd(z)  = dadz(z, bonus.modds[z_ss_idx].w_0,  ψ, ε, hp, σ_η, hbar)[1]
dd2(x) = central_fdm(5,1)(a_z, x)
dd3(x) = central_fdm(5,1)(a_w, x)
dd4(x) = dadz(zgrid[z_ss_idx], x,  ψ, ε, hp, σ_η, hbar)[2]

# check derivatives 
plot(dd,0.9,1.1)
plot!(dd2, 0.9, 1.1)

plot(dd3, 0.9,1.1)
plot!(dd4, 0.9, 1.1)
=#


#=
# Solve for dJ/dz in Hall directly
JJ_H_2   = zeros(N_z) 
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

# Solve for dJ/dz in Hall directly
JJ_H_3   = zeros(N_z) 
Threads.@threads for iz = 1:N_z

    # Initialize guess of direct effect
    v0     = zgrid./(1- β*(1-s))
    v0_new = zeros(N_z)
    iter   = 1
    err    = 10
    
    # solve via simple value function iteration
    @inbounds while err > 10^-10 && iter < 1000
        v0_new = hall.a*zgrid + β*(1-s)*P_z*v0
        err    = maximum(abs.(v0_new - v0))
        v0     = copy(v0_new)
        iter +=1
    end

    JJ_H_3[iz]   = v0[iz]

end

plot(JJ_H_2)
plot!(slopeFD(JJ_H_3,zgrid), linestyle=:dash)
=#