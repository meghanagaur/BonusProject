# Solve the simplest static model (first pass)
cd(dirname(@__FILE__))

using Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =15, yguidefontsize=15, xtickfontsize=13, ytickfontsize=13,
linewidth = 3, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 13)

# Load helper functions
include("staticModel.jl")

# For plotting -- results are quite sensitive to parameter choices
minz  = 0.95 # otherwise, E[profit/worker] < 0
maxz  = 1.05 # plot until n ≈ 1

# Plot E[profit/filled vacancy] as a function of z
plot(J, minz, maxz, legend = false)
plot(N, minz, maxz, legend = false)
ylabel!(L"J")
xlabel!(L"z")

# Plot employment as a function of z
plot(N, minz, maxz, legend = false)
ylabel!(L"n")
xlabel!(L"z")

# Plot log employment against log z
plot(x -> log(N.(exp(x))), log(minz), log(maxz), legend = false)
ylabel!(L"\log n")
xlabel!(L"\log z")

# Compute first order approximation around z = z0
z0   = 1
δ    = 0.05

# Update plotting window
minz  = z0 - δ
maxz  = z0 + δ

# Plot the comparison
plot(derivAnalytical, minz, maxz, legend =:topright, label = "Analytical Elasticity")
plot!(derivNumerical, minz, maxz, label = "Numerical Elasticity", linestyle=:dash)
ylabel!(L"d \log n/d \log z")
xlabel!(L"z")

## Update plotting window -- decrease δ
δ     = 0.01
minz  = z0 - δ
maxz  = z0 + δ

# Plot expected profit / filled vacancy as a function of z for the different models
plot(x -> globalApproxNash(x, z0).J, minz, maxz, label = "Shimer: Nash wage, Fixed a", linecolor=:black)
plot!(x -> globalApproxFixed(x, z0).J, minz, maxz, legend =:topleft, label = "Hall: Fixed w and a", linecolor=:blue)
plot!(x -> globalApproxFixedA(x, z0).J, minz, maxz, label = "Fixed a, Variable w", linecolor=:green)
plot!(x -> globalApproxFixedW(x, z0).J, minz, maxz, label = "Fixed w, Variable a", linecolor=:cyan)
plot!(x -> J(x), minz, maxz, label = "Bonus Economy: Variable w and a", linecolor=:red)
plot!(x -> globalApproxFixedW(x, z0).J + globalApproxFixedA(x, z0).J -  globalApproxFixed(x, z0).J, minz, maxz, 
label = "Fixed w + Fixed a - Fixed w and a (= Bonus)", linecolor=:magenta)
ylabel!(L"J")
xlabel!(L"z")
savefig("static-figs/J_diff_models.pdf")

# Plot n as a function of z for the different models
plot(x -> globalApproxNash(x, z0).n, minz, maxz, label = "Shimer: Nash wage, Fixed a", linecolor=:black)
plot!(x -> globalApproxFixed(x, z0).n, minz, maxz, legend =:topleft, label = "Hall: Fixed w and a", linecolor=:blue)
plot!(x -> globalApproxFixedA(x, z0).n, minz, maxz, label = "Fixed a, Variable w", linecolor=:green)
plot!(x -> globalApproxFixedW(x, z0).n, minz, maxz, label = "Fixed w, Variable a", linecolor=:cyan)
plot!(x -> N(x), minz, maxz, label = "Bonus Economy: Variable w and a", linecolor=:red)
ylabel!(L"n")
xlabel!(L"z")
savefig("static-figs/n_diff_models.pdf")

# Plot the numerical dlogn/dlogz as a function of z for the different models
plot(x -> derivNumericalNash(x, z0), minz, maxz, label = "Shimer: Nash wage, Fixed a", linecolor=:black, legend =:topright)
plot!(x -> derivNumericalFixed(x, z0), minz, maxz, label = "Hall: Fixed w and a", linecolor=:blue)
plot!(x -> derivNumericalFixedA(x, z0), minz, maxz, label = "Fixed a, Variable w", linecolor=:green)
plot!(x -> derivNumericalFixedW(x, z0), minz, maxz, label = "Fixed w, Variable a", linecolor=:cyan)
plot!(derivNumerical, minz, maxz, label = "Bonus Economy: Variable w and a", linecolor=:red)
ylabel!(L"d \log n/d \log z")
xlabel!(L"z")
savefig("static-figs/dlogn_dlogz_diff_models_numerical.pdf")

# Plot the analytical dlogn/dlogz as a function of z for the different models
plot(x -> globalApproxNash(x, z0).d, minz, maxz, label = "Shimer: Nash wage, Fixed a", linecolor=:black, legend = :topright)
plot!(x -> globalApproxFixed(x, z0).d, minz, maxz, label = "Hall: Fixed w and a", linecolor=:blue)
plot!(x -> globalApproxFixedA(x, z0).d, minz, maxz, label = "Fixed a, Variable w", linecolor=:green)
plot!(x -> globalApproxFixedW(x, z0).d, minz, maxz, label = "Fixed w, Variable a", linecolor=:cyan)
plot!(derivAnalytical, minz, maxz, label = "Bonus Economy: Variable w and a", linecolor=:red)
ylabel!(L"d \log n/d \log z")
xlabel!(L"z")
savefig("static-figs/dlogn_dlogz_diff_models_analytical.pdf")

## Figures for slides

#str  = ["Rigid Wage: Fixed w and a"; "Incentive Pay: Variable w and a"]
str  = ["Rigid Wage"; "Incentive Pay"]

#col = [:blue, :green, :cyan, :red]
col = [:blue, :red]

# Plot J as a function of z for the different models

#F   = [x -> globalApproxFixed(x, z0).J ; x -> globalApproxFixedA(x, z0).J ; x -> globalApproxFixedW(x, z0).J ; J]
F   = [x -> globalApproxFixed(x, z0).J; J]

for i = 1:length(F)
#for i = 0:length(F)
    #local p1 = plot(x -> globalApproxNash(x, z0).J, minz, maxz, label = "Shimer: Nash wage, Fixed a", legend=:topleft, linecolor = :black)
    local p1 = plot()
    ylabel!(L"J")
    xlabel!(L"z")
    local ymin = globalApproxFixedW(minz, z0).J
    local ymax = globalApproxFixedW(maxz, z0).J
    ylims!(p1,(ymin, ymax))
    #=if i == 0
       savefig("static-figs/slides/J_0.pdf")
    end =#
    for j = 1:i 
        plot!(p1, F[j], minz, maxz, label = str[j], linecolor = col[j])
    end
    savefig("static-figs/slides/J_"*string(i)*".pdf")
end

# Plot n as a function of z for the different models

#F   = [x -> globalApproxFixed(x, z0).n ; x -> globalApproxFixedA(x, z0).n ; x -> globalApproxFixedW(x, z0).n ;  N]
F   = [x -> globalApproxFixed(x, z0).n  ; N]

for i = 1:length(F)
#for i = 0:length(F)
    #local p1 = plot(x -> globalApproxNash(x, z0).n, minz, maxz, label = "Shimer: Nash wage, Fixed a", legend=:topleft, linecolor = :black)
    local p1 = plot()
    ylabel!(L"n")
    xlabel!(L"z")
    local ymin = globalApproxFixedW(minz, z0).n
    local ymax = globalApproxFixedW(maxz, z0).n
    ylims!(p1,(ymin, ymax))
   #=if i == 0
        savefig("static-figs/slides/n_0.pdf")
    end =#
    for j = 1:i 
        plot!(p1, F[j], minz, maxz, label = str[j], linecolor = col[j])
    end
    savefig("static-figs/slides/n_"*string(i)*".pdf")
end


# Plot dlogn/dlogz as a function of z for the different models

#F   = [x -> globalApproxFixed(x, z0).d ; x -> globalApproxFixedA(x, z0).d ; x -> globalApproxFixedW(x, z0).d ; derivAnalytical]
F   = [x -> globalApproxFixed(x, z0).d;  derivAnalytical]

for i = 1:length(F)
#for i = 0:length(F)
    #local p1 = plot(x -> globalApproxNash(x, z0).d, minz, maxz, label = "Shimer: Nash wage, Fixed a", legend=:topright, linecolor = :black)
    local p1 = plot()
    ylabel!(L"d \log n/d \log z")
    xlabel!(L"z")
    local ymin = globalApproxFixedA(maxz, z0).d - 0.2
    local ymax = globalApproxFixedW(minz, z0).d + 0.2
    ylims!(p1,(ymin, ymax))
    #= if i == 0
        savefig("static-figs/slides/dlogn_dlogz_0.pdf")
    end=#
    for j = 1:i 
        plot!(p1, F[j], minz, maxz, label = str[j], linecolor = col[j])
    end
    savefig("static-figs/slides/dlogn_dlogz_"*string(i)*".pdf")
end

## Plots with just Hall and Bonus; Update plotting window -- increase δ
δ          = 0.05
minz       = z0 - δ
maxz       = z0 + δ
rigid      = "Rigid Wage" #: Fixed w and a"
perfpay    = "Flexible Incentive Pay" #: Variable w and a"

plot(x -> globalApproxFixed(x, z0).J, minz, maxz, legend =:topleft, label = rigid, linecolor=:blue)
plot!(x -> J(x), minz, maxz, label = perfpay, linecolor=:red)
ylabel!(L"\mathbb{E}[J|z]")
xlabel!(L"z")
savefig("static-figs/slides/J_Hall_vs_Bonus.pdf")

plot(x -> globalApproxFixed(x, z0).n, minz, maxz, legend =:topleft, label = rigid, linecolor=:blue)
plot!(x -> N(x), minz, maxz, label = perfpay, linecolor=:red)
ylabel!(L"n(z)")
xlabel!(L"z")
savefig("static-figs/slides/n_Hall_vs_Bonus.pdf")

plot(x -> globalApproxFixed(x, z0).w, minz, maxz, legend =:topleft, label = rigid, linecolor=:blue)
plot!(x -> W(x), minz, maxz, label = perfpay, linecolor=:red)
ylabel!(L"\mathbb{E}[w|z]")
xlabel!(L"z")
savefig("static-figs/slides/w_Hall_vs_Bonus.pdf")

plot(x -> globalApproxFixed(x, z0).a, minz, maxz, legend =:topleft, label = rigid, linecolor=:blue)
plot!(x -> A(x), minz, maxz, label = perfpay, linecolor=:red)
ylabel!(L"a(z)")
xlabel!(L"z")
savefig("static-figs/slides/a_Hall_vs_Bonus.pdf")

plot(x -> globalApproxFixed(x, z0).v, minz, maxz, legend =:topleft, label = rigid, linecolor=:blue)
plot!(x -> V(x), minz, maxz, label = perfpay, linecolor=:red)
ylabel!(L"v(z)")
xlabel!(L"z")
savefig("static-figs/slides/v_Hall_vs_Bonus.pdf")

## Plot some other derivatives
δ     = 0.05
minz  = z0 - δ
maxz  = z0 + δ

plot(x ->  globalApproxFixed(x, z0).d, minz, maxz, legend =:topleft, label = "Hall: Fixed w and a", linecolor=:blue)
plot!(derivAnalytical, minz, maxz, label = perfpay, linecolor=:red)
ylabel!(L"d \log n/d \log z")
xlabel!(L"z")
#savefig("static-figs/slides/dlogn_dlogz_Hall_vs_Bonus.pdf")

## Adding in cyclical unemployment benefit: b = γ + χ*log(γ)
δ           = 0.1
minz        = z0 - δ
maxz        = z0 + δ

# set χ value
χ           = 0.3
rigid       = "Rigid Wage: Fixed w and a"
perfpay     = "Incentive Pay: Acyclical b"
perfpay_chi = "Incentive Pay: Procyclical b"
exogWage    = "Fixed effort: Exogenously procyclical w"

# profits
plot(x -> staticModel(z = x, χ = 0.0).J, minz, maxz, label = perfpay, linecolor=:red, legend=:bottomright)
plot!(x -> globalApproxFixed(x, z0).J, minz, maxz,  label = rigid, linecolor=:blue, linestyle=:dash)
plot!(x -> staticModel(z = x, χ = χ).J, minz, maxz, label = perfpay_chi, linecolor=:black, linestyle=:dashdot)
plot!(x -> exogCycWage(x; χ = χ).J, minz, maxz, label = exogWage, linecolor=:green, linestyle=:dot)
ylabel!(L"\mathbb{E}[J|z]")
xlabel!(L"z")

savefig("static-figs/slides/J_Hall_vs_Bonus_cyc_b.pdf")

# employment
plot(x -> staticModel(z = x, χ = 0.0).n, minz, maxz, label = perfpay, linecolor=:red, legend=:bottomright)
plot!(x -> globalApproxFixed(x, z0).n, minz, maxz, label = rigid, linecolor=:blue, linestyle=:dash)
plot!(x -> staticModel(z = x, χ = χ).n, minz, maxz, label = perfpay_chi, linecolor=:black,linestyle=:dashdot)
plot!(x -> exogCycWage(x; χ = χ).n, minz, maxz, label = exogWage, linecolor=:green, linestyle=:dot)
ylabel!(L"n(z)")
xlabel!(L"z")

savefig("static-figs/slides/n_Hall_vs_Bonus_cyc_b.pdf")

# wages
plot(x -> staticModel(z = x, χ = 0.0).w, minz, maxz, label = perfpay, linecolor=:red, legend=:bottomright)
plot!(x -> globalApproxFixed(x, z0).w, minz, maxz, label = rigid, linecolor=:blue, linestyle=:dash)
plot!(x -> staticModel(z = x, χ = χ).w, minz, maxz, label = perfpay_chi, linecolor=:black,linestyle=:dashdot)
plot!(x -> exogCycWage(x; χ = χ).w, minz, maxz, label = exogWage, linecolor=:green, linestyle=:dot)
ylabel!(L"\mathbb{E}[w|z]")
xlabel!(L"z")

savefig("static-figs/slides/w_Hall_vs_Bonus_cyc_b.pdf")

# effort
plot(x -> staticModel(z = x, χ = 0.0).a, minz, maxz, label = perfpay, linecolor=:red, legend=:bottomright)
plot!(x -> globalApproxFixed(x, z0).a, minz, maxz, label = rigid, linecolor=:blue, linestyle=:dash)
plot!(x -> staticModel(z = x, χ = χ).a, minz, maxz, label = perfpay_chi, linecolor=:black,linestyle=:dashdot)
plot!(x -> exogCycWage(x; χ = χ).a, minz, maxz, label = exogWage, linecolor=:green, linestyle=:dot)
ylabel!(L"a(z)")
xlabel!(L"z")

savefig("static-figs/slides/a_Hall_vs_Bonus_cyc_b.pdf")

# unemployment benefit
plot(x -> staticModel(z = x, χ = 0.0).b, minz, maxz, label = "Constant b", linecolor=:red, legend=:bottomright)
plot!(x -> staticModel(z = x, χ = χ).b, minz, maxz, label = "Procyclical b", linecolor=:black,linestyle=:dash)
ylabel!(L"b(z)")
xlabel!(L"z")

savefig("static-figs/slides/b_Hall_vs_Bonus_cyc_b.pdf")

# vacancies
plot(x -> staticModel(z = x, χ = 0.0).v, minz, maxz, label = perfpay, linecolor=:red, legend=:bottomright)
plot!(x -> globalApproxFixed(x, z0).v, minz, maxz, label = rigid, linecolor=:blue, linestyle=:dash)
plot!(x -> staticModel(z = x, χ = χ).v, minz, maxz, label = perfpay_chi, linecolor=:black,linestyle=:dashdot)
plot!(x -> exogCycWage(x; χ = χ).v, minz, maxz, label = exogWage, linecolor=:green, linestyle=:dot)
ylabel!(L"v(z)")
xlabel!(L"z")

savefig("static-figs/slides/v_Hall_vs_Bonus_cyc_b.pdf")

# additional check
plot(x -> dJexogCycWage(x, χ = χ).check, minz, maxz,  linecolor=:black, legend=:bottomright, label ="direct effect of z on linearity constraint")
plot!(x -> dJexogCycWage(x, χ = χ).diff_1, minz, maxz, linecolor=:red, legend=:bottomright, label ="dJ/dz  procyc b - dJ/dz  exog w")
plot!(x -> dJexogCycWage(x, χ = χ).diff_2, minz, maxz,  linecolor=:blue, legend=:bottomright, linestyle=:dash,  label ="dJ/dz acyc b - dJ/dz  rigid w")
