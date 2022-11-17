using LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =13, yguidefontsize=13, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 9)

using DataStructures, Distributions, ForwardDiff, Interpolations, DelimitedFiles,
 LinearAlgebra, Parameters, Random, Roots, StatsBase, DynamicModel

cd(dirname(@__FILE__))

#dir = "figs/vary-z1/chi/"
dir = "figs/vary-z1/chi0/"

"""
Vary z_1, and compute relevant aggregate variables
"""
function vary_z1(xx)

    modds      = OrderedDict{Int64, Any}()
    modd       =  model(σ_η = xx[1], χ = xx[2], γ = xx[3],  hbar = xx[4])
    @unpack β, s, ψ, zgrid ,N_z, z_1_idx  = modd
    z_ss_idx   = modd.z_1_idx # z_SS index

    # Solve the model for different z_0
    @time Threads.@threads for iz = 1:N_z
        #modds[iz] =  solveModel(model(σ_η = xx[1], χ = xx[2], γ = xx[3],  hbar = xx[4], z_1 = zgrid[iz]), noisy = false)
        modds[iz] =  solveModel(model(σ_η = xx[1], χ = 0.3, γ = xx[3],  hbar = xx[4], z_1 = zgrid[iz]), noisy = false)
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

# Load estimation output
est_output = readdlm("jld/estimation_3.txt", ',', Float64)    # open output across all jobs
idx        = argmin(est_output[:,1])                          # check for the lowest function value across processes 
pstar      = est_output[idx, 2:end]                           # get parameters 

# Get the Bonus model aggregates
w_0_B, θ_B, W_B, Y_B, ω_B, J_B, z_ss_idx, a_B = vary_z1(pstar)

# Get the Hall analogues
a_H, W_H, J_H, Y_H, θ_H = solveHall(Y_B, W_B, J_B);

rigid   = "Rigid Wage: Fixed w and a"
bonus   = "Incentive Pay: Variable w and a"

# plot profit results
logz = log.(model().zgrid)
plot(logz,J_B, linecolor=:red, label=bonus, legend=:topleft)
plot!(logz,J_H, linecolor=:blue,label=rigid)
#hline!([0],linecolor=:black,linestyle=:dash, label="")
xaxis!(L"\log z")
yaxis!(L"J")
savefig(dir*"profits.pdf")

# plot effort results
plot(logz,a_B, linecolor=:red, label=bonus, legend=:topleft)
hline!([a_H], linecolor=:blue, label=rigid)
xaxis!(L"\log z")
yaxis!(L"a(z)")
savefig(dir*"efforts.pdf")

# plot tightness results
plot(logz,θ_B, linecolor=:red, label=bonus, legend=:topleft)
plot!(logz,θ_H, linecolor=:blue, label=rigid)
xaxis!(L"\log z")
yaxis!(L"\theta(z)")
savefig(dir*"tightness.pdf")

# plot wage results
plot(logz,W_B, linecolor=:red, label=bonus, legend=:topleft)
hline!([W_H], linecolor=:blue, label=rigid)
xaxis!(L"\log z")
yaxis!(L"W(z)")
savefig(dir*"wages.pdf")

# approx slope using forward finite differences 
function slope(xx, dz)
    return (xx[2:end]-xx[1:end-1])./dz
end

@unpack zgrid =model()
lzgrid = log.(zgrid)
dz   = zgrid[2:end] - zgrid[1:end-1]
tt_B = slope(θ_B, dz).*zgrid[1:end-1]./θ_B[1:end-1]
tt_H = slope(θ_H, dz).*zgrid[1:end-1]./θ_H[1:end-1]

plot(lzgrid[5:end-1],tt_B[5:end], linecolor=:red, label=bonus, legend=:topleft)
plot!(lzgrid[5:end-1],tt_H[5:end], linecolor=:blue,label=rigid)
xaxis!(L" \log z")
yaxis!(L"\frac{d \theta }{dz}\frac{z}{\theta}")
savefig(dir*"dtheta.pdf")


#=
# w_0  = ψ*(Y_0[z_1_idx] - κ/q_0) # constant for wage difference equation
@unpack q,κ,ψ,ι=model()

ψ*(Y_B - κ./q.(θ_B)) -w_0
κ./q.(θ_B) - J_B
1 ./q.(θ_B) - J_B/κ
q.(θ_B) .- κ./J_B

κ./J_B


(x.^(-ι) .- 1).^(1/ι)

θ_H.^(-ι) .- 1
=#