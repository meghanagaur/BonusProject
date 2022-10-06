include("smm_settings.jl") # SMM inputs, settings, packages, etc.

# Simulate moments at given parameter values
function simulate_moments(xx)
    baseline = model(ε = xx[1] , σ_η = xx[2], hbar = xx[3], χ = 0, γ = .66) 
    out      = simulate(baseline, shocks)
    return out
end

xx      = [0.3; 0.5; 1]
output  = simulate_moments(xx)

@unpack std_Δlw, dlw1_du, dlw_dly, u_ss, avg_Δlw, dlw1_dlz, dlY_dlz, dlu_dlz, std_u, std_z, std_Y, std_w0, flag  = output


# targeted moments
round(std_Δlw,digits=4)
round(dlw1_du,digits=4)
round(dlw_dly,digits=4)
round(u_ss,digits=4)
round(avg_Δlw,digits=4)

# addiitonal simulation moments
round(dlu_dlz,digits=4)
round(dlY_dlz,digits=4)
round(dlw1_dlz,digits=4)
round(std_u,digits=4)
round(std_z,digits=4)
round(std_Y,digits=4)
round(std_w0,digits=4)
