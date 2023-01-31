
using ForwardDiff, LaTeXStrings
cd(dirname(@__FILE__))

using Plots; gr(border = :box, grid = true, minorgrid = true, gridalpha=0.2,
xguidefontsize =13, yguidefontsize=15, xtickfontsize=8, ytickfontsize=8,
linewidth = 2, gridstyle = :dash, gridlinewidth = 1.2, margin = 10* Plots.px,legendfontsize = 12)

## playing around with the matching function

# job-filling rates
q(θ,ι)     = 1/(1 + θ^ι)^(1/ι)                # h&m mf
q_cd(θ)    =  max(min(1.355*θ^(-0.72),1), 0)  # shimer mf

p1 = plot(x-> q(x,0.8),0,2,label=L"\iota = 0.8")
plot!(p1, x-> q(x,1.25),0,2, label=L"\iota = 1.25")
ylabel!(p1, L"q(\theta)")
xlabel!(p2, L"\theta")

# job-finding rates
f(θ,ι)  = 1/(1 + θ^-ι)^(1/ι)               # h&m mf
f_cd(θ) = max(min(1.355*θ^(1-0.72),1),0)   # shimer mf

p2 = plot(x-> f(x,0.8),0,2, legend=:false)
plot!(p2, x-> f(x,1.25),0,2)
ylabel!(p2, L"f(\theta)")
xlabel!(p2, L"\theta")


plot(p1, p2, layout = (2,1))
savefig("figs/mf_rates.pdf")

# numerical derivatives of job-filling rate
function dqdθ(x,ι)
    g = ForwardDiff.derivative(y -> q(y, ι), x)  
    return g*x/q(x,ι)
end 

## plot the job-filling elasticities
p1 = plot(legend=:false)

ι_vals = [0.8 1.25]
for i = 1:length(ι_vals)
    plot!(x->dqdθ(x, ι_vals[i]), 0, 2)
end
xlabel!(p1, L"\theta")
ylabel!(p1, L"\varepsilon_{q,\theta}")

# numerical derivatives of job-finding rate
function dfdθ(x,ι)
    g = ForwardDiff.derivative(y -> f(y, ι), x)  
    return g*x/f(x,ι)
end 

## plot the job-finding elasticities
p2 = plot(legend=:false) 

ι_vals = [0.8 1.25]
for i = 1:length(ι_vals)
    plot!(p2, x->dfdθ(x,ι_vals[i]), 0, 2)
end
xlabel!(p2, L"\theta")
ylabel!(p2, L"\varepsilon_{f,\theta}")

plot(p1, p2, layout = (2,1))
savefig("figs/mf_elasticities.pdf")