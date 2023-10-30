function test(; μ = 0.0, β = 0.99^(1/3), s = 0.031, ρ = 0.99, σ_ϵ = 0.001, N_z =51)
        
        logz, P_z, p_z  = rouwenhorst(μ, ρ, σ_ϵ, N_z)     # log z grid, transition matrix, invariant distribution
        zgrid           = exp.(logz)                      # z grid in levels

        v0     = zgrid./(1-ρ*β*(1-s))
        v0_new = zeros(N_z)
        iter   = 1
        err    = 10
        
        # solve via simple value function iteration
        @inbounds while err > 10^-10 && iter < 1000
            v0_new = zgrid + β*(1-s)*P_z*v0            
            err    = maximum(abs.(v0_new - v0))
            v0     = copy(v0_new)
            iter +=1
            println(err)
        end

        v1     = zgrid./(1-ρ*β*(1-s))
        v1_new = zeros(N_z)
        iter   = 1
        err    = 10
        
        # solve via simple value function iteration
        @inbounds while err > 10^-10 && iter < 1000
            v1_new = zgrid + ρ*β*(1-s)*P_z*v1          
            err    = maximum(abs.(v1_new - v1))
            v1     = copy(v1_new)
            iter +=1
            println(err)
        end

        idx = Int64(median(1:N_z))
        
        return v0_new, zgrid, v1_new./zgrid, idx
end

out, zgrid, out2, idx = test()

out1 = slopeFD(out, zgrid)
(out[2:end]-out[1:end-1])./(zgrid[2:end] - zgrid[1:end-1])
 
#plot(out, label = "actual function")
plot(out1, label ="actual deriv")
plot!(out2, label ="hopeful deriv", linestyle=:dash)

out1[idx] - out2[idx]


# Solve for dJ/dz directly (direct effect)
JJ_2   = zeros(N_z) 
Threads.@threads for iz = 1:N_z

    # Initialize guess of direct effect
    v0     = zgrid./(1-ρ*β*(1-s))
    v0_new = zeros(N_z)
    iter   = 1
    err    = 10
    
        # solve via simple value function iteration
        @inbounds while err > 10^-10 && iter < 1000
            v0_new = bonus.modds[iz].az.*zgrid + ρ*β*(1-s)*P_z*v0  
            #v0_new = zgrid + ρ*β*(1-s)*P_z*v0            
          
            err    = maximum(abs.(v0_new - v0))
            v0     = copy(v0_new)
            iter +=1
        end

    JJ_2[iz]   = v0[iz]/zgrid[iz]

end

# Solve for dJ/dz in Hall directly
JJ_3   = zeros(N_z) 

# Initialize guess of direct effect
v0     = zgrid./(1- β*(1-s))
v0_new = zeros(N_z)
iter   = 1
err    = 10

# solve via simple value function iteration
@inbounds while err > 10^-10 && iter < 1000
    v0_new = bonus.modds[z_ss_idx].az.*zgrid+ β*(1-s)*P_z*v0

    err    = maximum(abs.(v0_new - v0))
    v0     = copy(v0_new)
    iter +=1
end

JJ_3  = slopeFD(v0, zgrid)

plot(JJ_2[range_2])
plot!(JJ_EVT[range_2], linestyle=:dash)
plot!(JJ_3[range_2])
