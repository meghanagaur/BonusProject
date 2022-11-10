cd(dirname(@__FILE__))

include("smm_settings.jl") # SMM inputs, settings, packages, etc.
using DelimitedFiles

# Simulate moments at given parameter values
function simulate_moments(xx)
    baseline = model(σ_η = xx[1], χ = xx[2], γ = xx[3],  hbar = xx[4]) 
    out      = simulate(baseline, shocks)
    return out
end

est_output = readdlm("jld/estimation_3.txt", ',', Float64)     # open output across all jobs
idx        = argmin(est_output[:,1])                         # check for the lowest function value across processes 
pstar      = est_output[idx, 2:(2+J-1)]                      # get parameters 

output     = simulate_moments(pstar)

@unpack std_Δlw, dlw1_du, dlw_dly, u_ss, avg_Δlw, dlw1_dlz, dlY_dlz, dlu_dlz, std_u, std_z, std_Y, std_w0, flag, flag_IR  = output

# estimated parameters
round.(pstar, digits=4)

# targeted moments
round(std_Δlw,digits=4)
round(dlw1_du,digits=4)
round(dlw_dly,digits=4)
round(u_ss,digits=4)

# addiitonal simulation moments
round(dlu_dlz,digits=4)
round(dlY_dlz,digits=4)
round(dlw1_dlz,digits=4)
round(std_u,digits=4)
round(std_z,digits=4)
round(std_Y,digits=4)
round(std_w0,digits=4)
