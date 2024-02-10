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
labels       = ["Incentives + Bargaining" "Bargaining" "Bargaining: user guide"]

# Settings for IRFs (in years)
T        = 20   # horizon of contract (in years)
N        = 30   # length of IRF (in years)
N_plot   = 5    # number of years to plot
z_ss     = 1.0  # steady state z
sd       = 3

# initialize series
θ_t      = zeros(N*12, length(files))
u_t      = zeros(N*12, length(files))
lz_t     = zeros(N*12, length(files))
u_ss     = zeros(length(files))
θ_ss     = zeros(length(files))

# Make directory for figures
file_save = "figs/irfs/sd"*string(sd)*"/"      
mkpath(file_save)

# Loop through files
for file_idx = 1:length(files)

    file_str     = files[file_idx]                              
    file_pre     = "smm/jld-original/pretesting_"*file_str*".jld2"   # pretesting data location
    file_est     = "smm/jld-original/estimation_"*file_str*".txt"    # estimation output location
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
    @unpack σ_η, χ, γ, ε, ρ, ι, σ_ϵ = Params
    dlz     = sd*σ_ϵ   # size of shock 
    lz1     = log(z_ss) + dlz 

    ## Vary initial productivity z_0 
    if fix_a == false
        ss  = IRFs(; ρ = ρ, T = 12*T, N = 12*N, lz1 = log(z_ss), u1 = 0.06, ε = ε, σ_η = σ_η, χ = χ, γ = γ, ι = ι, fix_a = false)
        @assert(abs(ss.u_t[end]- ss.u_ss) < eps())
        irf = IRFs(; ρ = ρ, T = 12*T, N = 12*N, lz1 = lz1, u1 = ss.u_ss, ε = ε, σ_η = σ_η, χ = χ, γ = γ, ι = ι, fix_a = false)
    elseif fix_a == true
        ss  = IRFs(; ρ = ρ, T = 12*T, N = 12*N, lz1 = log(z_ss), u1 = 0.06, ε = ε, σ_η = σ_η, χ = χ, γ = γ, ι = ι, fix_a = true)
        @assert(abs(ss.u_t[end]-ss.u_ss) < eps())
        irf = IRFs(; ρ = ρ, T = 12*T, N = 12*N, lz1 = lz1, u1 = ss.u_ss, ε = ε, σ_η = σ_η, χ = χ, γ = γ, ι = ι, fix_a = true)
    end
    
    # Plot series in % deviations from SS 
    θ_t[:, file_idx]  = irf.θ_t
    u_t[:, file_idx]  = irf.u_t
    lz_t[:, file_idx] = irf.lz_t[1:N*12]
    θ_ss[file_idx]    = first(unique(ss.θ_t))
    u_ss[file_idx]    = ss.u_ss
end

# Plot nonlinear IRFs
NN    = N_plot*12 
dlθ_t = [(θ_t[:,i]  .- θ_ss[i])./θ_ss[i] for i = 1:length(files)]  # % deviations 
dθ_t  = [(θ_t[:,i]  .- θ_ss[i])  for i = 1:length(files)]          # deviations (levels)
du_t  = [(u_t[:,i]  .- u_ss[i]) for i = 1:length(files)]           # deviations (levels)
dlz_t = [(lz_t[:,i] .- log(z_ss)) for i = 1:length(files)]         # log deviations  

# Deviations from SS 

# dlogθ
if sd < 0
    p1 = plot(legend=:bottomright)
else 
    p1 = plot(legend=:topright)
end
for i = 1:length(files)
    plot!(p1, 0:NN-1, 100*dlθ_t[i][1:NN], label = labels[i], ylabel = L"\theta", xlabel = L"t")
end

# du
p2 = plot()
for i = 1:length(files)
    plot!(p2, 0:NN-1, 100*du_t[i][1:NN], label = labels[i], ylabel = L"u", xlabel = L"t", legend=:false)
end

# dlogz
p3 = plot()
for i = 1:length(files)
    plot!(p3, 0:NN-1, 100*dlz_t[i][1:NN], label = labels[i], ylabel = L"z", xlabel = L"t", legend=:false)
end

# dθ
if sd < 0
    p4 = plot(legend=:bottomright)
else 
    p4 = plot(legend=:topright)
end

for i = 1:length(files)
    plot!(p4, 0:NN-1, dθ_t[i][1:NN], label = labels[i], ylabel = L"θ", xlabel = L"t", legend=:false)
end

# Plot dampening due to bargaining 
θ_correct = θ_ss[2]/θ_ss[1]
p5        = plot(0:NN-1, (θ_t[1:NN,2]./θ_t[1:NN,1])./θ_correct,  legend=:false, ylabel=L"\theta")
#p5        = plot(0:NN-1, (dθ_t[2][1:NN]./dθ_t[1][1:NN]),  legend=:false, ylabel=L"\theta")

u_correct = u_ss[2]/u_ss[1]
p6        = plot(0:NN-1, (u_t[1:NN,2]./u_t[1:NN,1])./u_correct, legend=:false, ylabel=L"u")
#p6        = plot(0:NN-1, (du_t[2][1:NN]./du_t[1][1:NN]), legend=:false, ylabel=L"u")

# for the sake of being exact
dlθ_t     = [(log.(θ_t[:,i])  .- log.(θ_ss[i])) for i = 1:length(files)]  # % deviations 
p7        = plot(0:NN-1, dlθ_t[2][1:NN]./dlθ_t[1][1:NN], legend=:false, ylabel=L"\theta")

savefig(p1, file_save*"dlθ.pdf")
savefig(p2, file_save*"du.pdf")
savefig(p3, file_save*"dlz.pdf")
savefig(p4, file_save*"dθ.pdf")
savefig(p5, file_save*"θdamp.pdf")
savefig(p6, file_save*"udamp.pdf")
savefig(p7, file_save*"dlθdamp.pdf")


