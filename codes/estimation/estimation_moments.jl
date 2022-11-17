cd(dirname(@__FILE__))

include("functions/smm_settings.jl") # SMM inputs, settings, packages, etc.

using DelimitedFiles, LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =13, yguidefontsize=13, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 9)

## Logistics
file_str     = "fix_eps05"
file_pre     = "runs/jld/pretesting_"*file_str*".jld2"  # pretesting data location
file_est     = "runs/jld/estimation_"*file_str*".txt"   # estimation output location
file_save    = "figs/vary-z1/"*file_str*"/"             # file to-save 
mkpath(file_save)

# Simulate moments
est_output = readdlm(file_est, ',', Float64)   # open output across all jobs
@unpack moms, fvals, pars, mom_key, param_bounds, param_est, param_vals, data_mom, J = load(file_pre) 

idx        = argmin(est_output[:,1])           # check for the lowest function value across processes 
pstar      = est_output[idx, 2:(2+J-1)]        # get parameters 

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
function simulate_moments(Params)
    @unpack σ_η, χ, γ, hbar, ε = Params
    baseline = model(σ_η = σ_η, χ =  χ, γ =  γ,  hbar = hbar, ε = ε) 
    out      = simulate(baseline, shocks)
    return out
end

# Get moments
output     = simulate_moments(Params)
@unpack std_Δlw, dlw1_du, dlw_dly, u_ss, u_ss_2, avg_Δlw, dlw1_dlz, dlY_dlz, dlu_dlz, std_u, std_z, std_Y, std_w0, flag, flag_IR, IR_err  = output

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
round(u_ss,digits=4)

# Addiitonal moments
round(u_ss_2,digits=4)
round(dlu_dlz,digits=4)
round(dlY_dlz,digits=4)
round(dlw1_dlz,digits=4)
round(std_u,digits=4)
round(std_z,digits=4)
round(std_Y,digits=4)
round(std_w0,digits=4)

## Vary z1 xperiments

"""
Vary z_1, and compute relevant aggregate variables
"""
function vary_z1(Params)

    @unpack σ_η, χ, γ, hbar, ε = Params
    modds      = OrderedDict{Int64, Any}()
    modd       = model(σ_η = σ_η, χ =  χ, γ =  γ,  hbar = hbar, ε = ε) 
    @unpack β, s, ψ, zgrid ,N_z, z_1_idx  = modd
    z_ss_idx   = modd.z_1_idx # z_SS index

    # Solve the model for different z_0
    @time Threads.@threads for iz = 1:N_z
        modds[iz] =  solveModel(model(σ_η = σ_η, χ =  χ, γ =  γ,  hbar = hbar, ε = ε, z_1 = zgrid[iz]), noisy = false)
    end

    ## Store series of interest
    w_0    = [modds[i].w_0 for i = 1:N_z]      # w0 (constant)
    θ_1    = [modds[i].θ for i = 1:N_z]        # tightness
    W_1    = w_0/ψ                             # EPV of wages
    Y_1    = [modds[i].Y for i = 1:N_z]        # EPV of output
    ω_1    = [modds[i].ω_0 for i = 1:N_z]      # EPV value of unemployment at z0
    J_1    = Y_1 - W_1                         # EPV profits
    a_1    = [modds[i].az[i] for i = 1:N_z]    # optimal effort @ start of contract

    return w_0, θ_1, W_1, Y_1, ω_1, J_1, z_ss_idx, a_1
end

"""
Solve Hall model, Y,W from Bonus
"""
function solveHall(Y_B, W_B, J_B)

    @unpack zgrid, z_1_idx, κ, ι, s, β, N_z, P_z = model()
    
    # Solve for expected PV of sum of the z_t's
    exp_z = zeros(length(zgrid)) 
    @inbounds for (iz,z0) in enumerate(zgrid)
        z0_idx  = findfirst(isequal(z0), zgrid)  # index of z0 on zgrid
        
        # initialize guesses
        v0     = zgrid./(1-β*(1-s))
        v0_new = zeros(N_z)
        iter   = 1
        err    = 10
        
        # solve via simple value function iteration
        @inbounds while err > 10^-10 && iter < 500
            v0_new = zgrid + β*(1-s)*P_z*v0
            err    = maximum(abs.(v0_new - v0))
            v0     = copy(v0_new)
            iter +=1
        end
        exp_z[iz]   = v0[z0_idx]

    end

    aa       = Y_B[z_1_idx]./exp_z[z_1_idx]  # exactly match SS PV of output in the 2 models
    WW       = W_B[z_ss_idx]                 # match SS PV of wages (E_0[w_t] = w_0 from martingale property)
    YY       = aa.*exp_z                     # Hall economy output 
    JJ       = YY .- WW                      # Hall economy profits
    qθ       = min.(1, max.(0, κ./JJ))       # job-filling rate
    θ        = (qθ.^(-ι) .- 1).^(1/ι).*(qθ .!= 0) # implied tightness

    return aa, WW, JJ, YY, θ
end

# Get the Bonus model aggregates
w_0_B, θ_B, W_B, Y_B, ω_B, J_B, z_ss_idx, a_B = vary_z1(Params)

# Get the Hall analogues
a_H, W_H, J_H, Y_H, θ_H = solveHall(Y_B, W_B, J_B);

# Plot labels
rigid   = "Rigid Wage: Fixed w and a"
bonus   = "Incentive Pay: Variable w and a"

# Plot profits
logz = log.(model().zgrid)
plot(logz,J_B, linecolor=:red, label=bonus, legend=:topleft)
plot!(logz,J_H, linecolor=:blue,label=rigid)
#hline!([0],linecolor=:black,linestyle=:dash, label="")
xaxis!(L"\log z")
yaxis!(L"J")
savefig(file_save*"profits.pdf")

# Plot effort 
plot(logz,a_B, linecolor=:red, label=bonus, legend=:topleft)
hline!([a_H], linecolor=:blue, label=rigid)
xaxis!(L"\log z")
yaxis!(L"a")
savefig(file_save*"efforts.pdf")

# Plot tightness
plot(logz,θ_B, linecolor=:red, label=bonus, legend=:topleft)
plot!(logz,θ_H, linecolor=:blue, label=rigid)
xaxis!(L"\log z")
yaxis!(L"\theta")
savefig(file_save*"tightness.pdf")

# Plot wages
plot(logz,W_B, linecolor=:red, label=bonus, legend=:topleft)
hline!([W_H], linecolor=:blue, label=rigid)
xaxis!(L"\log z")
yaxis!(L"W")
savefig(file_save*"wages.pdf")

""" 
Approx slope using forward finite differences 
"""
function slope(xx, dz)
    return (xx[2:end]-xx[1:end-1])./dz
end

@unpack zgrid = model()
lzgrid        = log.(zgrid)
dz   = zgrid[2:end] - zgrid[1:end-1]
tt_B = slope(θ_B, dz).*zgrid[1:end-1]./θ_B[1:end-1]
tt_H = slope(θ_H, dz).*zgrid[1:end-1]./θ_H[1:end-1]

plot(lzgrid[5:end-1],tt_B[5:end], linecolor=:red, label=bonus, legend=:topleft)
plot!(lzgrid[5:end-1],tt_H[5:end], linecolor=:blue,label=rigid)
xaxis!(L" \log z")
yaxis!(L"\frac{d \theta }{dz}\frac{z}{\theta}")
savefig(file_save*"dtheta.pdf")


