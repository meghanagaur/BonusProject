# Produce main figures/moments for the paper

cd(dirname(@__FILE__))

# turn off for cluster
ENV["GKSwstype"] = "nul"

# Load helper files
include("functions/smm_settings.jl")                    # SMM inputs, settings, packages, etc.
include("functions/moments.jl")                         # vary z_0 functions

using DataFrames, Binscatters, DelimitedFiles, LaTeXStrings, IntervalArithmetic, IntervalRootFinding,
Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize = 13, yguidefontsize = 13, xtickfontsize=10, ytickfontsize=10,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px, legendfontsize = 12)

## Logistics
files        = ["fix_a_bwc10"] #["baseline" "fix_a_bwc10" "fix_chi0"]
big_run      = false        
file_idx     = big_run ? parse(Int64, ENV["SLURM_ARRAY_TASK_ID"]) : 1
file_str     = files[file_idx]                              
file_pre     = "smm/jld-original/pretesting_"*file_str*".jld2"   # pretesting data location
file_est     = "smm/jld-original/estimation_"*file_str*".txt"    # estimation output location
file_save    = "figs/vary-z0/"*file_str*"/"                      # file to-save 
λ            = 10^5                                              # smoothing parameter for HP filter

# Make directory for figures
mkpath(file_save)
println("File name: "*file_str)

# Settings for simulation
if big_run == false
    fix_wages                = false 
    N_vary_z                 = 51            # # of gridpoints when taking numerical derivatives
    N_sim_alp_workers        = 1
    N_sim_alp                = 1
else
    N_vary_z                 = 201           # number of gridpoints when taking numerical derivatives
    N_sim_alp_workers        = 10^4          # number of workers for endogneous ALP simulation
    N_sim_alp                = 1000          # number of panels for endogneous ALP simulation
end

# Load output
est_output = readdlm(file_est, ',', Float64) # estimation output       
@unpack moms, fvals, pars, mom_key, param_bounds, param_est, param_vals, data_mom, J, W, fix_a = load(file_pre) # pretesting output

# Get the final minimum 
idx        = argmin(est_output[:,1])         # check for the lowest function value across processes 
pstar      = est_output[idx, 2:(2+J-1)]      # get parameters 

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

# Model set-up
modd                       = model(σ_η = σ_η, χ = χ, γ = γ, hbar = hbar, ε = ε, ρ = ρ, σ_ϵ = σ_ϵ, ι = ι) 
shocks                     = rand_shocks(modd.P_z, modd.p_z; z0_idx = modd.z_ss_idx, N_sim_alp_workers = N_sim_alp_workers, N_sim_alp = N_sim_alp)

# Compute simulated model moments 
if fix_a == false
   
    sol                     = getModel(modd)
    @unpack a_z, θ_z, W, Y  = sol
    a_z                     = a_z[:, modd.z_ss_idx]
    @time output            = simulate(modd, shocks; check_mult = false, smm = false, λ = λ, sd_cut = 3.0) # get output

    #= Check for multiplicity roots
    println("Checking for multiplicity of roots..")
   
    a_min = 10^-8
    a_max = 10

    @unpack ψ, ε, q, κ, hp, σ_η, hbar = modd
    a_gap(x, z) = x - ((z*x/sol.w_0 - (ψ/ε)*(hp(x)*σ_η)^2)/hbar)^(ε/(1+ε))

    p1 = plot()
    p2 = plot()
    for (iz,z) in enumerate(modd.zgrid)
        println(roots( x -> a_gap(x, z)  ,  a_min..a_max))
        plot!(p1, x -> a_gap(x, z) , 0, 10^-3)
        plot!(p2, x -> a_gap(x, z), a_min, 2.0)
    end

    plot(p1, p2, layout = (1,2), size = (800,400), title = "Implicit Effort Gap", legend=:false)
    savefig(file_save*"effort_error.pdf")
    =#
else
    @unpack a_z, θ_z, W, Y = getFixedEffort(modd; a = 1.0, fixed_wages = fixed_wages) 
    @time output = simulateFixedEffort(modd, shocks; a = Params[:a], sd_cut = 3.0, fixed_wages = fixed_wages, smm = false)  
end

# Unpack parameters
@unpack std_Δlw, dlw1_du, dlw_dly, u_ss, flag, flag_IR, IR_err, rho_lx, std_lx, corr_lx, macro_vars = output

# CHANGE ROUNDING MODE TO ROUND NEAREST AWAY 

println("Min fval: \t"*string(round(minimum(est_output[:,1]), RoundNearestTiesAway, digits = 10 )) )

# Estimated parameters
println("------------------------------------------------")
println("ESTIMATED PARAMETERS")
println("------------------------------------------------")
println("σ_η: \t\t"*string(round.(Params[:σ_η], RoundNearestTiesAway, digits = 3)))
println("χ: \t\t"*string(round.(Params[:χ], RoundNearestTiesAway, digits = 3)))
println("γ: \t\t"*string(round.(Params[:γ], RoundNearestTiesAway, digits = 3)))
println("hbar: \t\t"*string(round.(Params[:hbar], RoundNearestTiesAway, digits = 3)))
println("ε: \t\t"*string(round.(Params[:ε], RoundNearestTiesAway, digits = 3)))
println("ρ: \t\t"*string(round.(Params[:ρ], RoundNearestTiesAway, digits = 3)))
println("σ_ϵ: \t\t"*string(round.(Params[:σ_ϵ], RoundNearestTiesAway, digits = 3)))
println("ι: \t\t"*string(round.(Params[:ι], RoundNearestTiesAway, digits = 3)))

# Targeted moments
println("\n------------------------------------------------")
println("TARGETED MOMENTS")
println("------------------------------------------------")
println("std_Δlw: \t"*string(round.(std_Δlw, RoundNearestTiesAway, digits = 3)))
println("dlw0_du: \t"*string(round.(dlw1_du, RoundNearestTiesAway, digits = 3)))
println("dlw_dly: \t"*string(round.(dlw_dly, RoundNearestTiesAway, digits = 3)))
println("u_ss: \t\t"*string(round.(u_ss, RoundNearestTiesAway, digits = 3)))

# Untargeted moments
println("\n------------------------------------------------")
println("UNTARGETED MOMENTS, UNCONDITIONAL")
println("------------------------------------------------")

println("\nSTANDARD DEVIATION")
println("------------------------------------------------")
for i = 1:length(macro_vars) 
    println(string(macro_vars[i])*": \t"*string(round(std_lx[i],RoundNearestTiesAway, digits = 3)))
end

println("\nAUTOCORRELATION")
println("------------------------------------------------")
for i = 1:length(macro_vars) 
    println(string(macro_vars[i])*": \t"*string(round(rho_lx[i],RoundNearestTiesAway, digits = 3)))
end

println("\nCORRELATIONS")
println("------------------------------------------------")
for var in macro_vars 
    print("\t"*string(var))
end
println()
for i = 1:length(macro_vars) 
    str = "\n"*string(macro_vars[i])
    for j = 1:length(macro_vars)
        str = str*"\t"*string(round(corr_lx[i,j],RoundNearestTiesAway, digits = 3))
    end 
    println(str)
end

# Compute conditional moments
println("------------------------------------------------")
println("CONDITIONAL ON z = μ_z")
println("------------------------------------------------")

println("a(μ_z): \t"*string(round.(a_z[modd.z_ss_idx], RoundNearestTiesAway, digits = 3)))
println("θ(μ_z): \t"*string(round.(θ_z[modd.z_ss_idx], RoundNearestTiesAway, digits = 3)))
println("Y (μ_z): \t"*string(round.(Y[modd.z_ss_idx], RoundNearestTiesAway, digits = 3)))
println("W (μ_z): \t"*string(round.(W[modd.z_ss_idx], RoundNearestTiesAway, digits = 3)))
println("W/Y (μ_z): \t"*string(round.(W[modd.z_ss_idx]./Y[modd.z_ss_idx], RoundNearestTiesAway, digits = 3)))

## Vary initial productivity z_0 

# Get the Bonus model aggregates
modd_big    = model(N_z = N_vary_z, χ = χ, γ = γ, hbar = hbar, ε = ε, σ_η = σ_η, ι = ι, ρ = ρ, σ_ϵ = σ_ϵ)
modd_chi0   = model(N_z = N_vary_z, χ = 0.0, γ = γ, hbar = hbar, ε = ε, σ_η = σ_η, ι = ι, ρ = ρ, σ_ϵ = σ_ϵ)

if fix_a == true
    bonus      = vary_z0(modd_big; fix_a = fix_a, a = Params[:a])
    bonus_chi0 = vary_z0(modd_chi0; fix_a = fix_a, a = Params[:a])
else 
    bonus      = vary_z0(modd_big; fix_a = fix_a)
    bonus_chi0 = vary_z0(modd_chi0; fix_a = fix_a)
end

# Get primitives
@unpack P_z, zgrid, N_z, ρ, β, s, z_ss_idx, q, ι, κ, χ, μ_z, ψ, hp, logz = modd_big

# Get Hall aggregates
hall         = solveHall(modd_big, bonus.Y, bonus.W)

# Plot labels
rigid      = "Rigid Wage: fixed w and a"
fip        = "Incentive Pay: variable w and a"
ip         = "Incentive Pay, setting "*L"\chi = 0"
minz_idx   = max( findfirst(x -> x >=  -0.05, logz), findfirst(x -> x > 10^-8, bonus.θ))
maxz_idx   = findlast(x -> x <=  0.05, logz)
maxz_idx   = isnothing(maxz_idx) ? N_vary_z : maxz_idx
range_1    = minz_idx:maxz_idx

# Compute some numerical derivatives
dlY_dlz      = slopeFD(log.(max.(eps(), bonus.Y)), logz; diff = "central")
dlW_dlz      = slopeFD(log.(max.(eps(), bonus.W)), logz; diff = "central")
dlW_dlY      = slopeFD(log.(max.(eps(), bonus.W)), log.(max.(eps(), bonus.Y)); diff = "central")
dla0_dlz     = slopeFD(log.(max.(eps(), bonus.a1)), logz; diff = "central")
dlθ_dlY      = slopeFD(log.(max.(eps(), bonus.θ)), log.(max.(eps(), bonus.Y)); diff = "central")
dlθ_dlz      = slopeFD(log.(max.(eps(), bonus.θ)), logz; diff = "central")
dlθ_dlz_H    = slopeFD(log.(max.(eps(), hall.θ)), logz; diff = "central")
dlθ_dlz0     = slopeFD(log.(max.(eps(), bonus_chi0.θ)), logz; diff = "central")
dJ_dz        = slopeFD(bonus.J, zgrid; diff = "central")
dJ_dz_H      = slopeFD(hall.J, zgrid; diff = "central")
dJ_dz0       = slopeFD(bonus_chi0.J, zgrid; diff = "central")

println("dla_dlz: \t"*string(round.(dla0_dlz[z_ss_idx], RoundNearestTiesAway, digits = 3)))
println("dlY_dlz: \t"*string(round.(dlY_dlz[z_ss_idx], RoundNearestTiesAway, digits = 3)))
println("dlW_dlz: \t"*string(round.(dlW_dlz[z_ss_idx], RoundNearestTiesAway, digits = 3)))
println("dlW_dlY: \t"*string(round.(dlW_dlY[z_ss_idx], RoundNearestTiesAway, digits = 3)))
println("dlθ_dlz: \t"*string(round.(dlθ_dlz[z_ss_idx], RoundNearestTiesAway, digits = 3)))

# Get decomposition components
@unpack JJ_EVT, WC, BWC_resid, IWC_resid, BWC_share, c_term = decomposition(modd_big, bonus; fix_a = fix_a)

## Share of Incentive Wage Flexibility
println("------------------------")
println("WAGE CYCLICALITY MOMENTS")
println("------------------------")

## Share of bargained wage flexibility

println("BWC Share #1: \t"*string(round(BWC_share[z_ss_idx], RoundNearestTiesAway, digits = 3)))
println("BWC Share #2: \t"*string(round((BWC_resid./WC)[z_ss_idx], RoundNearestTiesAway, digits = 3)))
println("IWC: \t\t"*string(round((1-BWC_share[z_ss_idx])*dlw1_du , RoundNearestTiesAway, digits = 3)))
#println(IWC_resid[z_ss_idx])
#println(((1-BWC_share[z_ss_idx])*WC)[z_ss_idx])
println("C term at μ_z: \t"*string(round(c_term[z_ss_idx], RoundNearestTiesAway, digits = 3)))

# Make some plots of conditional moments
p1 = plot(logz[range_1], dlW_dlz[range_1], title=L"d \log W_0/d \log z_0")
p2 = plot(logz[range_1], dlW_dlY[range_1], title=L"d \log W_0/d \log Y_0")
p3 = plot(logz[range_1], dla0_dlz[range_1], title=L"d \log a(z_0,z_0) /d \log z_0")
p4 = plot(logz[range_1], dlY_dlz[range_1], title=L"d \log Y_0 /d \log z_0")
plot(p1,p2,p3,p4, legend=:false)
xaxis!(L"\log z")
savefig(file_save*"cond_moments.pdf")

plot(logz[range_1], dlθ_dlz[range_1],  label=L"d \log  \theta_0/d \log z_0")
plot!(logz[range_1], dlθ_dlY[range_1],  label=L"d \log \theta_0/d \log Y_0")
xaxis!(L"\log z")
savefig(file_save*"dlθ_dlY_dlz.pdf")

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

    p2 = plot(logz[range_1], bonus.az[range_1,range_1[1]], legend=:topleft, label=L"\log z_0="*string(round(logz[range_1[1]], digits=3)))
    plot!(p2, logz[range_1], bonus.az[range_1, Int64(median(range_1))], label=L"\log z_0="*string(round(logz[Int64(median(range_1))], digits=3)))
    plot!(p2, logz[range_1], bonus.az[range_1, range_1[end]], label=L"\log z_0="*string(round(logz[range_1[end]], digits=3)))
    xaxis!(L"\log z_t")
    yaxis!(L"a(z_t|z_0)")

    savefig(p2, file_save*"efforts_shifts.pdf")

    pt(x) = x^(1+1/ε)
    p2 = plot(logz[range_1], ψ*pt.(bonus.az[range_1,range_1[1]]), legend=:topleft, label=L"\log z_0="*string(round(logz[range_1[1]], digits=3)))
    plot!(p2, logz[range_1], ψ*pt.(bonus.az[range_1, Int64(median(range_1))]), label=L"\log z_0="*string(round(logz[Int64(median(range_1))], digits=3)))
    plot!(p2, logz[range_1], ψ*pt.(bonus.az[range_1, range_1[end]]), label=L"\log z_0="*string(round(logz[range_1[end]], digits=3)))
    xaxis!(L"\log z_t")
    yaxis!(L"\psi a(z_t|z_0)^{1 + 1/\epsilon}")

    savefig(p2, file_save*"passthrough_shifts.pdf")

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

    # BWC share
    p6 = plot(logz[range_1], (BWC_resid./WC)[range_1], legend=:false, title="BWC Share")
    plot!(p6, logz[range_1], BWC_share[range_1])
    xaxis!(L"\log z_0")
    savefig(p6, file_save*"BWC_share.pdf")

    p7 = plot(logz[range_1], (bonus.W./bonus.Y)[range_1], title=L"W_0(z_0)/Y_0(z_0)", legend=:false)
    xaxis!(L"\log z_0")
    savefig(p7, file_save*"labor_share.pdf")

    # Plot tightness fluctuations: dlog θ / d log z
    idx1    = max( max( findfirst(x -> ~isnan(x) && ~iszero(x) && x < 120, dlθ_dlz)) , 
                    max( findfirst(x -> ~isnan(x)  && ~iszero(x)  && x < 120, dlθ_dlz_H))) # start at reasonable scale
    range_2 = idx1:maxz_idx

    p8 = plot(logz[range_2], dlθ_dlz[range_2], linecolor=:red, label=fip, legend=:topright)
    plot!(p8, logz[range_2], dlθ_dlz_H[range_2], linecolor=:blue,label=rigid)
    plot!(p8, logz[range_2], dlθ_dlz0[range_2], linecolor=:cyan,label=ip, linestyle=:dash)
    xaxis!(L" \log z_0")
    yaxis!(L"\frac{d \log \theta(z_0) }{d \log z_0}")

    savefig(p8, file_save*"dlogtheta.pdf")

    # Compute the direct effect on the IR constraint
    ω_B  = bonus.ω
    dω_B = slopeFD(ω_B, modd_big.zgrid)
    B    = (χ/(1-β*ρ))
    A    = (log(γ) + β*B*(1-ρ)*μ_z)/(1-β)
    ω_2  = A .+ B*logz

    @assert(maximum(abs.(ω_B - ω_2)) < 10^-4 )

    dω_2 = B./zgrid

    @assert( minimum( max.(abs.(dω_B - dω_2) .< 10^-3, isnan.(dω_B)) )  == 1 )

    dIR  = dω_2.*(ρ*β - 1)/(1-ρ*β*(1-s))

    plot(zgrid[range_2], dJ_dz[range_2] , linecolor=:red, label = "bonus", linestyle=:dot)
    plot!(zgrid[range_2], JJ_EVT[range_2], linecolor=:black, label = "EVT")
    plot!(zgrid[range_2], dJ_dz0[range_2], linecolor=:blue, label = "chi0 ", linestyle=:dash)
    plot!(zgrid[range_2], dJ_dz_H[range_2] , linecolor=:green, label = "hall", linestyle=:dashdot)

    JJ_EVT[z_ss_idx] - dJ_dz[z_ss_idx]     # C term
    JJ_EVT[z_ss_idx] - dJ_dz_H[z_ss_idx]     # C term
    JJ_EVT[z_ss_idx] - dJ_dz0[z_ss_idx]    # should be close to 0

    ## C term graphs
    p1 = plot(logz[range_2], JJ_EVT[range_2], linecolor=:black, label = "Incentive Pay: No C term", legend =:bottomright)
    plot!(logz[range_2], dJ_dz_H[range_2], linecolor=:blue, label = rigid)
    plot!(logz[range_2], dJ_dz0[range_2], linecolor=:yellow, label = ip, legend =:bottom)
    xaxis!(L"z_0")
    yaxis!(L"\frac{d J(z_0) }{d z_0}")

    savefig(p1, file_save*"dJ_dz_0_c_term.pdf")

    # plot the lower bound
    plot(logz[range_2], c_term[range_2]./dIR[range_2], legend=:bottomright, label=L"\mu(z)" )
    plot!(logz[range_2], bonus.w_0[range_2], linestyle=:dash, label=L"w_{-1}(z)")
    xaxis!(L"\log z_0")

    savefig(file_save*"cterm_multiplier.pdf")
end

# Check convergence 
indices    = sortperm(est_output[:,1])
stop       = findlast(x -> x < 0.05, est_output[indices,1])
indices    = reverse(indices[1:stop])

p1 = plot()
p2 = plot()
p3 = plot()
p4 = plot()

for (k,v) in param_est
    if k == :σ_η
        plot!(p1, est_output[indices, v+1], label = L"\sigma_\eta")
    elseif k == :ε
        plot!(p2, est_output[indices, v+1], label = L"\epsilon")
    elseif k == :χ
        plot!(p3, est_output[indices, v+1], label = L"\chi")
    elseif k == :γ
        plot!(p4, est_output[indices, v+1], label = L"\gamma")
    end
end

p5 = plot(p1, p2, p3, p4, layout = (2,2), legend=:topleft, ytitle= "Parameter values", xtitle = "Sorted iterations")
savefig(p5, file_save*"params_converge.pdf")

# Function values
p6 = plot(est_output[indices,1], title = "Function values", legend=:false, xlabel = "Sorted iterations")
savefig(p6, file_save*"fvals_converge.pdf")

# Re-name log file
if big_run == true
    mkpath("logs")
    job_id  = parse(Int64, ENV["SLURM_ARRAY_JOB_ID"])
    mv("slurm-"*string(job_id)*"."*string(file_idx)*".out", "logs/"*file_str*".txt", force = true)
end
