
using Plots, ForwardDiff, LaTeXStrings
cd(dirname(@__FILE__))

## playing around with the matching function
q1(θ,ι)  = 1/(1 + θ^ι)^(1/ι)                # h&m mf
q2(θ)    =  max(min(1.355*θ^(-0.72),1), 0)  # shimer mf

plot(x->q1(x,0.5),0,5)
plot!(x->q1(x,1),0,5)
plot!(x->q1(x,2),0,5)
plot!(x->q1(x,3),0,5)
plot!(x->q1(x,4),0,5)
plot!(x->q1(x,5),0,5)
plot!(q2,0,5)
ylabel!(L"q(\theta)")
xlabel!(L"\theta")

f1(θ,ι)  = 1/(1 + θ^-ι)^(1/ι)               # h&m mf
f2(θ)    = max(min(1.355*θ^(1-0.72),1),0)   # shimer mf

plot(x->f1(x,1),0,5)
plot!(x->f1(x,2),0,5)
plot!(x->f1(x,3),0,5)
plot!(x->f1(x,4),0,5)
plot!(x->f1(x,5),0,5)
plot!(f2,0,5)
ylabel!(L"f(\theta)")
xlabel!(L"\theta")

#numerical derivatives of job-filling rate
function derivNumerical1(x,ι)
    g = ForwardDiff.derivative(y -> q1(y, ι), x)  
    return g*x/q1(x,ι)
end 

function derivNumerical2(x)
    g = ForwardDiff.derivative(q2, x)  
    return g*x/q2(x)
end 

# plot the job-filling rate
p1 = plot(q2, 0, 5,)

ι_vals = [0.8 0.9 1 1.1 1.2 1.3]
for i = 1:length(ι_vals)

    plot!(x->q1(x,ι_vals[i]), 0, 5, label=L"\iota = "*string(ι_vals[i]))

end

xlabel!(L"\theta")
ylabel!(L"\varepsilon_{q,\theta}")


## plot the elasticities
p1 = plot(derivNumerical2, 0, 5,)
xlabel!(L"\theta")
ylabel!(L"\varepsilon_{q,\theta}")

ι_vals = [0.8 0.9 1 1.1 1.2 1.3]
for i = 1:length(ι_vals)

    plot!(x->derivNumerical1(x,ι_vals[i]), 0, 5, label=L"\iota = "*string(ι_vals[i]))

end

savefig("matching_function_elasticities.pdf")