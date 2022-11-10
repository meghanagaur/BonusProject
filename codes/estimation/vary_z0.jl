using LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =13, yguidefontsize=13, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 9)

using DataStructures, Distributions, ForwardDiff, Interpolations, DelimitedFiles
 LinearAlgebra, Parameters, Random, Roots, StatsBase, DynamicModel

cd(dirname(@__FILE__))

include("smm_settings.jl")

"""
Vary z_1, and compute relevant aggregate variables
"""
function vary_z1(xx)
    modds    = OrderedDict{Int64, Any}()
    modd     =  model(σ_η = xx[1], χ = xx[2], γ = xx[3],  hbar = xx[4])
    @unpack β,s,ψ,ρ,σ_ϵ,hp,σ_η,q,κ,ι,ε,zgrid,N_z,P_z, z_1_idx  = modd
    dz      = zgrid[2:end] - zgrid[1:end-1]

    # Solve the model for different z_0
    @time Threads.@threads for iz = 1:length(zgrid)
        modds[iz] =  solveModel(model(σ_η = xx[1], χ = xx[2], γ = xx[3],  hbar = xx[4], z_1 = zgrid[iz]), noisy = false)
    end

    ## Store series of interest
    w_0    = [modds[i].w_0 for i = 1:length(zgrid)]      # w0 (constant)
    θ_0    = [modds[i].θ for i = 1:length(zgrid)]        # tightness
    W_0    = [modds[i].w_0/ψ[1] for i = 1:length(zgrid)] # PV of wages
    Y_0    = [modds[i].Y for i = 1:length(zgrid)]        # PV of output
    ω_0    = [modds[i].ω_0 for i = 1:length(zgrid)]      # PV of unemployment at z0
    J_0    = Y_0 - W_0

    return w_0, θ_0, W_0, Y_0, ω_0, J_0
end

"""
Solve Hall model
"""
function solveHall(model, Y, W)
    # Solve for expected PV of z_t's
    exp_z = zeros(length(zgrid)) 
    @inbounds for (iz,z0) in enumerate(zgrid)
        z0_idx  = findfirst(isequal(z0), zgrid)  # index of z0 on zgrid
        
        # initialize guesses
        v0     = zgrid./(1-β*(1-s))
        v0_new = zeros(N_z)
        iter   = 1
        err    = 10
        
        # solve via simple value function iteration
        @inbounds while err > 10^-8 && iter < 500
            v0_new = zgrid + β*(1-s)*P_z*v0
            err    = maximum(abs.(v0_new - v0))
            v0     = copy(v0_new)
            iter +=1
        end
        exp_z[iz]   = v0[z0_idx]
    end

    a_opt    = Y[idx]./exp_z[idx]  # exactly match SS PV of output in the 2 models
    w        = W[idx]              # match SS PV of wages (E_0[w_t] = w_0 from martingale property)
    JJ       = a_opt.*exp_z .- w   # Hall economy profits
    YY       = a_opt.*exp_z        # Hall economy output 

    return a_opt, w, JJ, YY
end


# Load estimation output
est_output = readdlm("jld/estimation_3.txt", ',', Float64)     # open output across all jobs
idx        = argmin(est_output[:,1])                         # check for the lowest function value across processes 
pstar      = est_output[idx, 2:(2+J-1)]                      # get parameters 
