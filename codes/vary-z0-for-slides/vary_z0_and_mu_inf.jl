 # vary z_0 <-> vary μ_z

using LaTeXStrings, Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =13, yguidefontsize=13, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 9)

using DataStructures, Distributions, ForwardDiff, Interpolations,
 LinearAlgebra, Parameters, Random, Roots, StatsBase, DynamicModel

#=
Set up the dynamic EGSS model, where m(u,v) = (uv)/(u^ι + v^⟦)^(1/ι),
η ∼ N(0, σ_η^2), log(z_t) = μ_z + ρ*log(z_t-1) + u_t, u_t ∼ N(0, σ_z^2),
and y_t = z_t(a_t + η_t).
β    = discount factor
r    = interest rate
s    = exogenous separation rate
ι    = matching elasticity
κ    = vacancy-posting cost
ω    = worker's PV from unemployment (infinite horizon)
χ    = prop. of unemp benefit to z / actual unemp benefit
γ    = intercept for unemp benefit w/ procyclical benefit
z_ss = steady state of productivity (this is a definition)
z_1  = initial prod. (= z_ss by default)
μ_z  = unconditional mean of log prod. process (= log(z_1) by default)
ρ    = persistence of log prod. process
σ_ϵ  = variance of innovation in log prod. process
σ_η  = st dev of η distribution
ε    = Frisch elasticity: disutility of effort
ψ    = pass-through parameters
procyclical == (procyclical unemployment benefit) <- can also set χ = 0
=#
function model(; β = 0.99, s = 0.1, κ = 0.474, ι = 1.67, ε = 0.5, σ_η = 0.05, z_ss = 1.0,
    ρ =  0.995, σ_ϵ = 0.001, χ = 0.0, γ = 0.658, z_1 = z_ss, μ_z = log(z_1), N_z = 11, procyclical = false)

    # Basic parameterization
    q(θ)    = 1/(1 + θ^ι)^(1/ι)                     # vacancy-filling rate
    f(θ)    = 1/(1 + θ^-ι)^(1/ι)                    # job-filling rate
    u(c)    = log(max(c, eps()))                    # utility from consumption                
    h(a)    = (a^(1 + 1/ε))/(1 + 1/ε)               # disutility from effort  
    u(c, a) = u(c) - h(a)                           # utility function
    hp(a)   = a^(1/ε)                               # h'(a)
    r       = 1/β -1                                # interest rate        

    # Define productivity grid
    if (iseven(N_z)) error("N_z must be odd") end 
    logz, P_z = rouwenhorst(μ_z, ρ, σ_ϵ, N_z)        # discretized logz grid & transition probabilties
    zgrid     = exp.(logz)                           # actual productivity grid
    z_1_idx   = findfirst(isapprox(z_1), zgrid)       # index of z0 on zgrid

    # Pass-through parameter
    ψ    = 1 - β*(1-s)

    # Unemployment benefit given aggregate state: (z) 
    if procyclical == true
        ξ(z) = (γ)*(z/z_ss)^χ 
    elseif procyclical == false
        ξ    = γ
    end

    # PV of unemp = PV of utility from consuming unemployment benefit forever
    if procyclical == false
        ω = log(ξ)/(1-β) # scalar
    elseif procyclical == true
        println("Solving for value of unemployment...")
        ω = unemploymentValue(β, ξ, u, zgrid, P_z).v0 # N_z x 1
    end
    
    return (β = β, r = r, s = s, κ = κ, ι = ι, ε = ε, σ_η = σ_η, ρ = ρ, σ_ϵ = σ_ϵ, 
    ω = ω, μ_z = μ_z, N_z = N_z, q = q, f = f, ψ = ψ, z_1 = z_1, h = h, u = u, hp = hp, 
    z_1_idx = z_1_idx, zgrid = zgrid, P_z = P_z, ξ = ξ, χ = χ, γ = γ, procyclical = procyclical)
end

#=
modd1 = model(procyclical = false)
modd2 = model(procyclical = true)
sol1 = solveModel(modd1)
sol2 = solveModel(modd2)
=#

@unpack β,s,ψ,ρ,σ_ϵ,hp,σ_η,q,κ,ι,ε,zgrid  = model()

# Define the z1_grid 
zmin      = 0.97
zmax      = 1.03
dz        = 0.001
z1_grid   = collect(zmin:dz:zmax)
idx       = floor(Int64,median(1:length(z1_grid))) # SS
modd      = OrderedDict{Int64, Any}()

# Solve the model for different z_0
#@time @inbounds for (iz,z0) in enumerate(z1_grid)
Threads.@threads for iz = 1:length(z1_grid)
    modd[iz] =  solveModel(model(z_1 = z1_grid[iz] , procyclical = false), noisy = false)
    #modd[iz] =  solveModel(model(z_1 = zgrid[iz] , procyclical = false), noisy = false)
end


## Store series of interest
w0    = [modd[i].w_0 for i = 1:length(z1_grid)]      # w0 (constant)
theta = [modd[i].θ for i = 1:length(z1_grid)]        # tightness
W     = [modd[i].w_0/ψ[1] for i = 1:length(z1_grid)] # PV of wages
Y     = [modd[i].Y for i = 1:length(z1_grid)]        # PV of output
ω0    = [modd[i].ω_0 for i = 1:length(z1_grid)]      # PV of unemployment at z0
J     = Y .- W

# Approx elasticity using forward finite differences for derivatives
function elasticity(yy, zgrid, dz) #, dlz)
    e1 = zgrid[1:end-1].*(yy[2:end]-yy[1:end-1])./(dz.*yy[1:end-1])
    return e1 
end

# Approx slope using forward finite differences 
function slope(xx, dz)
    return (xx[2:end]-xx[1:end-1])./dz
end

# plot series vs z0
p1 = plot(z1_grid, theta, ylabel=L"\theta_0", xlabel=L" z_0")
p2 = plot(z1_grid, W, ylabel=L"W_0", xlabel=L" z_0")
p3 = plot(z1_grid, Y, ylabel=L"Y_0",xlabel=L" z_0")
p4 = plot(z1_grid, J, ylabel=L"J_0",xlabel=L" z_0")
plot(p1, p2, p3, p4, layout = (2, 2), legend=:false)

# Plot unemployment value at z0 vs z0
plot(z1_grid, ω0, ylabel=L"\omega(z_0)", xlabel=L" z_0",  linewidth=4, linecolor=:cyan, label="actual benefit")

# Plot elasticities
t1 = elasticity(theta, z1_grid, dz)
w1 = elasticity(W, z1_grid, dz)
y1 = elasticity(Y, z1_grid, dz)
j1 = elasticity(J, z1_grid, dz)

p1 = plot(z1_grid[1:end-1], t1, ylabel=L"d\log \theta_0 / d \log z_0", xlabel=L" z_0")
p2 = plot(z1_grid[1:end-1], w1, ylabel=L"d \log W_0 d / \log z_0", xlabel=L" z_0")
p3 = plot(z1_grid[1:end-1], y1, ylabel=L"d \log Y_0 d / \log z_0",xlabel=L" z_0")
p4 = plot(z1_grid[1:end-1], j1, ylabel=L"d \log J_0 d / \log z_0",xlabel=L" z_0")
plot(p1, p2, p3, p4, layout = (2, 2), legend=:false)

# plot slopes 
tt  = slope(theta, dz)
ww  = slope(W, dz)
yy  = slope(Y, dz)
jj  = slope(J, dz)

p1 = plot(z1_grid[1:end-1], tt, ylabel=L"d \theta_0 / d  z_0", xlabel=L" z_0")
p2 = plot(z1_grid[1:end-1], ww, ylabel=L"d  W_0 d /  z_0", xlabel=L" z_0")
p3 = plot(z1_grid[1:end-1], yy, ylabel=L"d  Y_0 d / z_0",xlabel=L" z_0")
p4 = plot(z1_grid[1:end-1], jj, ylabel=L"d J_0 d / z_0",xlabel=L" z_0")
plot(p1, p2, p3, p4, layout = (2, 2), legend=:false)

# double-check slopes
qq(x) =  -(x^(-1+ι))*(1+x^ι)^(-1 -1/ι) # q'(θ)
xx    = theta[1:end-1]
ww2   = yy + (κ./(q.(xx)).^2).*tt.*qq.(xx) # dW/dz0 = dy/dz0 - d ( k/q(θ) ) / dz0

# check on slopes
plot(slope(q.(theta[1:end]), dz))
plot!(qq.(xx).*tt)

plot(slope(1 ./q.(theta[1:end]), dz))
plot!( (-qq.(xx).*tt) ./ (q.(theta[1:end-1]).^2) )

# Plot dY/dz0, dW/z0, and dJ/dz0 and check that these make sense.
plot(z1_grid[1:end-1],  ww2, label=L"d W_0 d / z_0 *",xlabel=L" z_0",linecolor=:yellow, linewidth=3) #check 
plot!(z1_grid[1:end-1], ww, label=L"d W_0 d / z_0", xlabel=L" z_0", linecolor=:orange)
plot!(z1_grid[1:end-1], yy, label=L"d  Y_0 d /  z_0",xlabel=L" z_0", linecolor=:red)
plot!(z1_grid[1:end-1], yy-ww2, label=L"d  J_0 d /  z_0 *",xlabel=L" z_0", linecolor=:cyan, linewidth=3) # check
plot!(z1_grid[1:end-1], jj, label=L"d  J_0 d / z_0",xlabel=L" z_0", linecolor=:blue)
plot!(z1_grid[1:end-1], yy -ww, label=L"d  J_0 d / z_0",xlabel=L" z_0", linecolor=:green)
plot!(z1_grid[1:end-1], -(κ./(q.(xx)).^2).*tt.*qq.(xx), label=L"d  J_0 d / z_0 *",xlabel=L" z_0", linecolor=:purple) # check

# Now consider a Hall vs Bonus Economy comparison

# Solve for expected PV of z_t's
exp_z = zeros(length(z1_grid)) 
Threads.@threads for iz = 1:length(z1_grid)
#@inbounds for (iz,z0) in enumerate(z1_grid)
    @unpack zgrid, P_z, N_z = modd[iz].mod
    z_1_idx  = findfirst(isequal(z1_grid[iz]), zgrid)  # index of z0 on zgrid
    
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
    exp_z[iz]   = v0[z_1_idx]
end

a_opt    = Y[idx]./exp_z[idx]  # exactly match SS PV of output in the 2 models
w        = W[idx]              # match SS PV of wages (E_0[w_t] = w_0 from martingale property)
JJ       = a_opt.*exp_z .- w   # Hall economy profits
YY       = a_opt.*exp_z        # Hall economy output 

# plot profits in the different economies
p4= plot(z1_grid, JJ, label="Hall: Fixed w and fixed a", linecolor=:blue)#,linewidth=3)
#plot!(p4, z1_grid, Y .- w, label="Fixed w and variable a", linecolor=:red)
#plot!(p4,z1_grid, YY.- W, label="Fixed a and variable w", linecolor=:blue)
plot!(p4,z1_grid, J, label="Bonus economy: Variable w and variable a", linecolor=:red, legend=:topleft)
xlabel!(L"z_1")
ylabel!(L"J_1")
savefig("figs/vary_z_1.pdf")

#=
# isolate effort/wage movements
p1 = plot( z1_grid, Y , label="Variable a", linecolor=:red, linewidth=3)
plot!(p1, z1_grid, YY, label="Fixed a", linecolor=:blue)
ylabel!(L"Y_0")
xlabel!(L"z_0")
p2= plot(W, label="Variable w",linecolor=:red)
hline!(p2, [w], label="Fixed w",linecolor=:blue)
ylabel!(L"W_0")
xlabel!(L"z_0")
p3 = plot(z1_grid./w0, ylabel=L"z_0/w_0", label="") # super flat 
plot(p1, p2, p3, layout = (3, 1), legend=:topleft)

# Zoom in
plot(z1_grid[idx-3:idx+3], JJ[idx-3:idx+3], label="Hall: Fixed w and fixed a", linecolor=:cyan,linewidth=3)
xlabel!(L"z_0")
ylabel!(L"J_0")
plot!(z1_grid[idx-3:idx+3], Y[idx-3:idx+3] .-w, label="Fixed w and variable a", linecolor=:red,legend=:topleft)
plot!(z1_grid[idx-3:idx+3], YY[idx-3:idx+3].- W[idx-3:idx+3], label="Fixed a and variable w", linecolor=:blue,linewidth=3)
plot!(z1_grid[idx-3:idx+3], Y[idx-3:idx+3] .-W[idx-3:idx+3], label="Bonus economy: Variable w and variable a", linecolor=:black,legend=:topleft)

# Compute slopes
JJ_B = slope(J,dz)     # Bonus 
JJ_H = slope(JJ,dz)    # Hall
ZZ   = slope(exp_z,dz) # slope of ∑ z_t (β(1-s))^(t-1)

# Check to compute ZZ analytically
exp_zz = zeros(length(z1_grid)) 
@inbounds for (iz,z0) in enumerate(z1_grid)
    @unpack zgrid, P_z, N_z = modd[iz].mod
    z0_idx  = findfirst(isequal(z0), zgrid)  # index of z0 on zgrid
    
    # initialize guesses
    v0     = zgrid./(1-β*(1-s))
    v0_new = zeros(N_z)
    iter   = 1
    err    = 10
    
    # solve via simple value function iteration
    @inbounds while err > 10^-8 && iter < 500
        v0_new = zgrid./z0 + β*(1-s)*P_z*v0
        err    = maximum(abs.(v0_new - v0))
        v0     = copy(v0_new)
        iter +=1
    end
    exp_zz[iz]   = v0[z0_idx]
end

plot(z1_grid[1:end-1], JJ_H, label="Hall: Fixed w and fixed a", linecolor=:cyan, linewidth=3,legend=:outerbottom, ylabel=L"dJ_0/ dz_0")
plot!(z1_grid[1:end-1], JJ_B, label="Bonus economy: Variable w and variable a", linecolor=:black)
plot!(z1_grid[1:end-1], a_opt*ZZ, label="Check 1")
plot!(z1_grid, a_opt*exp_zz, label="Check 2")
xlabel!(L"z_0")

# try approximating total derivative of value function with the partial derivative
# take dz_t/d_z0 from analytical AR1 process
exp_az = zeros(length(z1_grid)) 
@inbounds for (iz,z0) in enumerate(z1_grid)
    @unpack zgrid, P_z, N_z = modd[iz].mod
    @unpack az = modd[iz]
    z0_idx  = findfirst(isequal(z0), zgrid)  # index of z0 on zgrid
    
    # initialize guesses
    v0     = zgrid./(1-β*(1-s))
    v0_new = zeros(N_z)
    iter   = 1
    err    = 10
    # solve via simple value function iteration
    @inbounds while err > 10^-8 && iter < 500
        v0_new = az.*zgrid/z0 + β*(1-s)*P_z*v0
        err    = maximum(abs.(v0_new - v0))
        v0     = copy(v0_new)
        iter +=1
    end
    exp_az[iz]   = v0[z0_idx]
end


# try central finite differences for this plot
JJ_B = (J[3:end] - J[1:end-2])/2dz # re-compute Bonus slopes
plot(z1_grid, exp_az, label="Analytical Bonus w/ EVT", ylabel=L"dJ_0/ dz_0",linecolor=:red)
plot!(z1_grid,  a_opt*exp_zz,label="Hall Numerical", linecolor=:cyan)
plot!(z1_grid[2:end-1], JJ_B,label = "Numerical Bonus",linecolor=:black, legend=:topleft)

=#

#= try with a continuous process (take expectations first but shouldn't matter)
TT      = 20000
N_sim   = 10000
exp_zz2 = zeros(length(z1_grid))
Threads.@threads for iz = 1:length(z1_grid)
    z0 = log(z1_grid[iz])
    zt = zeros(N_sim, TT)
    zt[:,1] .= z0
    @inbounds for t=2:TT
        zt[:,t] = z0*(1-ρ) .+ ρ*zt[:,t-1] 
    end 
    zt = exp.(zt)
    ZT = mapreduce(t -> zt[:,t] *(β*(1-s))^(t-1), +, 1:TT)  # compute PV of Y_i
    exp_zz2[iz] = mean(ZT)
end

slope(exp_zz2,dz) 
exp_zz
=#
