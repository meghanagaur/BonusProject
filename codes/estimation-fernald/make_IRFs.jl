cd(dirname(@__FILE__))

# Produce main figures/moments for the paper

# turn off for cluster
ENV["GKSwstype"] = "nul"

# Load helper files
include("functions/smm_settings.jl")                    # SMM inputs, settings, packages, etc.
include("functions/finite-horizon-non-stoch-egss.jl")   # SMM inputs, settings, packages, etc.

using DataFrames, Binscatters, DelimitedFiles, LaTeXStrings, IntervalArithmetic, IntervalRootFinding,
Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize = 20, yguidefontsize = 20, xtickfontsize=12, ytickfontsize=12,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px, legendfontsize = 12)

## Logistics
files        = [ "baseline" "fix_a_bwc10" "fix_a_bwc0543"]
labels       = ["Incentives + Bargaining" "Bargaining" "User Guide"]
big_run      = true        

# Settings for IRFs
N = big_run ? 200 : 10

# Initilize the Plots 
p1 = plot()
p2 = plot()
p3 = plot()

# Make directory for figures
file_save = "figs/irfs/"      
mkpath(file_save)

# Loop through files
for file_idx = 1:length(files)

    file_str     = files[file_idx]                              
    file_pre     = "smm/jld/pretesting_"*file_str*".jld2"   # pretesting data location
    file_est     = "smm/jld/estimation_"*file_str*".txt"    # estimation output location
    println("File name: "*file_str)

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
    @unpack σ_η, χ, γ, ε, ρ, ι = Params

    ## Vary initial productivity z_0 
    if fix_a == false
        ss  = IRFs(; ρ = ρ, T = 12*20, N = N, lz1 = 0.0, u1 = 0.06, ε = ε, σ_η = σ_η, χ = χ, γ = γ, ι = ι, fix_a = false)
        @assert(abs(ss.u_t[end]- ss.u_ss)<eps())
        irf = IRFs(; ρ = ρ, T = 12*20, N = N, lz1 = 0.01, u1 = ss.u_ss, ε = ε, σ_η = σ_η, χ = χ, γ = γ, ι = ι, fix_a = false)
    elseif fix_a == true
        ss  = IRFs(; ρ = ρ, T = 12*20, N = N, lz1 = 0.0, u1 = 0.06, ε = ε, σ_η = σ_η, χ = χ, γ = γ, ι = ι, fix_a = true)
        @assert(abs(ss.u_t[end]-ss.u_ss)<eps())
        irf = IRFs(; ρ = ρ, T = 12*20, N = N, lz1 = 0.01, u1 = ss.u_ss, ε = ε, σ_η = σ_η, χ = χ, γ = γ, ι = ι, fix_a = true)
    end
    
    # Plot series in % deviations from SS 
    dθ_t  = (irf.θ_t - ss.θ_t)./(ss.θ_t)       # % deviations 
    du_t  = (irf.u_t .- ss.u_ss)./(ss.u_ss)    # % deviations
    dlz_t = (irf.lz_t - ss.lz_t)               # log deviations 

    # re-index to start at 0
    plot!(p1, 0:N-1, 100*dθ_t, label = labels[file_idx], ylabel = L"\theta", xlabel = L"t")
    plot!(p2, 0:N, 100*du_t, label = labels[file_idx], ylabel = L"u", xlabel = L"t", legend=:bottomright)
    plot!(p3, 0:N-1, 100*dlz_t[1:N], label = labels[file_idx], ylabel = L"z", xlabel = L"t")
end

savefig(p1, file_save*"θ.pdf")
savefig(p2, file_save*"u.pdf")
savefig(p3, file_save*"z.pdf")