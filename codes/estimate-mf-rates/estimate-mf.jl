using CSV, DataFrames, Optim, StatsBase

cd("/Users/meghanagaur/BonusProject/code/estimate-mf-rates")

# Load the vacancy, unemploymentdatda
df = DataFrame(CSV.File(pwd()*"/data/mf_rates.csv"))

# get relevant data
idx     = findfirst(!ismissing, df.q_frate)
#frate   = 1 .- (1 .- df.frate[idx:end-1]).^3
frate   = df.q_frate[idx]
θ       = df.tightness[idx:end-1]
f(θ, ι) = 1/(1 + θ^(-ι))^(1/ι)

# define objective function to be minimized
function ff(ι, θ, frate1)
    frate2 = f.(θ, ι)
    return (abs.(mean(frate2) - frate1))
end

# let's just match the quarterly job-finding rate for θ = 1
opt     = optimize(ι -> ff(ι, 1, frate), 0.2, 20)
ι_opt   = Optim.minimizer(opt)

# ≈ 1.67