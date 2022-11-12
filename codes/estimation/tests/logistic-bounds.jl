

# how steep is search
g(x) = ForwardDiff.derivative(u->logit(u,λ=0.5), x)  
plot(g, -1,1)