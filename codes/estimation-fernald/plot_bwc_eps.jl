cd(dirname(@__FILE__))

# Produce main figures/moments for the paper

# turn off for cluster
ENV["GKSwstype"] = "nul"

# Load helper files
include("functions/smm_settings.jl")                    # SMM inputs, settings, packages, etc.
include("functions/moments.jl")                         # vary z1 functions

using DataFrames, Binscatters, DelimitedFiles, LaTeXStrings, IntervalArithmetic, IntervalRootFinding,
Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize = 13, yguidefontsize = 13, xtickfontsize=10, ytickfontsize=10,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px, legendfontsize = 12)

## Logistics
files        = ["fix_eps05" "fix_eps10" "fix_eps15" "fix_eps20" "fix_eps25" "fix_eps30" "fix_eps35" "fix_eps40" "fix_eps45" "fix_eps50" "baseline"]
big_run      = false       

# Initialize vectors
bwc          = zeros(length(files))
epsilon      = copy(bwc)
chi          = copy(bwc)

# Settings for simulation
if big_run == false
    vary_z_N                 = 51           # lower # of gridpoints when taking numerical derivatives
else
    vary_z_N                 = 101          # increase # of gridpoints when taking numerical derivatives
end

Threads.@threads for file_idx = 1:length(files)

    file_str     = files[file_idx]                              
    file_pre     = "smm/jld-original/pretesting_"*file_str*".jld2"   # pretesting data location
    file_est     = "smm/jld-original/estimation_"*file_str*".txt"    # estimation output location

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
    epsilon[file_idx]  = ε
    chi[file_idx]      = χ

    ## Vary initial productivity z_0 

    # Get the Bonus model aggregates
    modd       = model(N_z = vary_z_N, χ = χ, γ = γ, hbar = hbar, ε = ε, σ_η = σ_η, ι = ι, ρ = ρ, σ_ϵ = σ_ϵ)

    if fix_a == true
        bonus      = vary_z0(modd; fix_a = fix_a, a = Params[:a])
    else 
        bonus      = vary_z0(modd; fix_a = fix_a)
    end

    # Get decomposition components
    @unpack JJ_EVT, WC, BWC_resid, IWC_resid, BWC_share, c_term = decomposition(modd, bonus; fix_a = fix_a)

    @unpack z_ss_idx = modd 
    
    bwc[file_idx] = BWC_share[z_ss_idx]
end

# Plot BWC as a function of epsilon

i = sortperm(epsilon)
plot(sort(epsilon[i]), bwc[i], legend=:false)
annotate!([epsilon[end]], [bwc[end]], "X", annotationcolor=:red)
xlabel!(L"\epsilon")
ylabel!(L"\textrm{BWC\ Share}")
savefig("figs/vary_eps_bwc.pdf")

# Plot calibrated χ as a function of epsilon

i = sortperm(epsilon)
plot(sort(epsilon[i]), chi[i], legend=:false)
annotate!([epsilon[end]], [chi[end]], "X", annotationcolor=:red)
xlabel!(L"\epsilon")
ylabel!(L"\chi")
savefig("figs/vary_eps_chi.pdf")
