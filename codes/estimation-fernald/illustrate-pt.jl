cd(dirname(@__FILE__))

using DataFrames, Binscatters, DelimitedFiles, LaTeXStrings, IntervalArithmetic, IntervalRootFinding
Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize = 13, yguidefontsize = 13, xtickfontsize=10, ytickfontsize=10,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px, legendfontsize = 12)

# Load helper files
include("functions/smm_settings.jl")                    # SMM inputs, settings, packages, etc.
include("functions/finite-horizon-non-stoch-egss.jl")   # SMM inputs, settings, packages, etc.

# Define a pass-through function
pt(x,y; ψ = model().ψ) = ψ*x^(1+1/y)

p1 = plot(xlabel=L"\epsilon", yaxis=L"\psi a^{1 + 1/\epsilon}")
for x = 0.9:0.05:1.1
    plot!(p1, y-> pt(x,y), 0.3, 5.0, label=L"a="*string(x))
end

savefig(p1,"figs/pass-through/illustration.pdf")

## Logistics
T            = 10  # horizon of contract (in years)
sd           = 0
file         = "baseline" 
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
TT      = T*12
lz_t    = [lz1*ρ^(t-1) for t = 1:TT+1] # lz_t path
modd    = model_FH(; ρ =  ρ, ε = ε, σ_η = σ_η, χ = χ, γ = γ, T = TT, ι = ι) 
@unpack ψ, β, s = modd

# Settings for IRFs (in years)
sol   = solveModel_FH(modd, lz_t; noisy = false)
pt_fh = (ψ).*sol.az.^(1+1/ε)
p1 = plot(pt_fh, legend=:false)
yaxis!(p1, L"\psi a^{1 + 1/\epsilon}")
xaxis!(p1, L"t")

# Simulate worker separations and average pass-through
f_fh           = modd.f(sol.θ)
N_sim          = 5*10^4
T_sim          = 1000
burnin         = 500

# Information about job/employment spells
T_sim          = T_sim + burnin
pt_sim         = fill(NaN, N_sim, T_sim)           # N x T panel of pass-through
tenure         = zeros(Int64, N_sim, T_sim)        # N x T panel of tenure
s_shocks       = rand(Uniform(0,1), N_sim, T_sim)  # separation shocks
jf_shocks      = rand(Uniform(0,1), N_sim, T_sim)  # job-finding shocks
pt_sim[:,1]   .= pt_fh[1]                          # period pass-through
tenure[:,1]   .= 1                                 # within-job tenure
unemp          = zeros(N_sim)                      # current unemployment status

@views @inbounds for t = 2:T_sim

    Threads.@threads for n = 1:N_sim

        if unemp[n] == false  
            
            # separation shock at the end of t-1 
            if s_shocks[n, t-1] < s              
                
                # find a job and produce within the period t
                if  jf_shocks[n, t] < f_fh          
                    tenure[n,t]  = 1
                    pt_sim[n, t] = pt_fh[1]         
                else
                    unemp[n]     = true
                end

            # no separation shock, remain employed    
            else     
                tenure[n,t] = tenure[n,t-1] + 1
                
                # separate for sure if have survived to this point with no separation shock
                if tenure[n,t] <= TT                                   
                    pt_sim[n, t] = pt_fh[tenure[n,t]]         
                else
                    unemp[n]     = true
                end   
            end

        elseif unemp[n] == true

            # job-finding shock
            if jf_shocks[n, t] < f_fh       
                unemp[n]     = false              
                tenure[n,t]  = 1                
                pt_sim[n, t] = pt_fh[1]          
            end 
        end

    end
end

@views pt_pb   = pt_sim[:, burnin+1:end]  
@views avg_pt  = mean(pt_pb[isnan.(pt_pb).==0])

idx = findfirst(x -> x >= avg_pt, pt_fh)
annotate!(p1,[idx], [avg_pt], "X", annotationcolor=:red)
title!("Average pass-through: "*string(round(avg_pt,digits=3)))

savefig("figs/pass-through/finite_horizon_perfect_foresight_"*string(T)*"yr.pdf")
