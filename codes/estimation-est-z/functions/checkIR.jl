
# Solve for the worker's continuation value upon separation 
iter3 = 10^5
tol3  = 10^-5
err3  = 10
iter3 = 1  
W_0   = copy(ω) # initial guess
flow  = β*s*(P_z*ω)

@inbounds while err3 > tol3 && iter3 <= max_iter3
    W_1  = flow + β*(1-s)*(P_z*W_0)
    err3 = maximum(abs.(W_1 - W_0))
    if (err3 > tol3) 
        iter3 += 1
        if (iter3 < max_iter3) 
            W_0  = W_1
        end
    end
    println(err3)
end

d1 = ω - W_0
d2 = (1-β)*A/(1-β*(1-s)) .+ B*logz*(1 - β*s/(1 - ρ*β*(1-s)) )  .- β*s*(1-ρ)*μ_z/(1 - ρ*β*(1-s))

plot(d1)
plot!(d2)