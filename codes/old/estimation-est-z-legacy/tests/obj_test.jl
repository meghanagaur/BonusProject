@unpack σ_η, χ, γ, hbar, ε, ρ, σ_ϵ = Params

xx  =  [ε, σ_η, χ, γ, hbar, ρ, σ_ϵ]

shocks = rand_shocks()

baseline    = model(σ_η = σ_η, χ = χ, γ = γ, hbar = hbar, ε = ε, ρ = ρ, σ_ϵ = σ_ϵ) 

out        = simulate(baseline, shocks)

W          = Matrix(1.0I, 6, 6)
mod_mom    = [out.std_Δlw, out.dlw1_du, out.dlw_dly, out.u_ss, out.alp_ρ, out.alp_σ]
d1         = (mod_mom - data_mom)./abs.(data_mom) 
d2         = (mod_mom - data_mom) 
d3         = (mod_mom - data_mom)./0.5(abs.(mod_mom) + abs.(data_mom)) # arc % differences

d1'*W*d1
d2'*W*d2
d3'*W*d3


# old results
mod_mom2   = [0.0631; -1.0001; 0.0339; 0.0658]
data_mom2  = [0.064; -1; 0.05; 0.067]
d4         = (mod_mom2 - data_mom2)./abs.(data_mom2) 
W2         = Matrix(1.0I, 4, 4)
d4'*W2*d4 

hcat(d1[1:4],d4)