using CSV, DataFrames, Optim, StatsBase

cd("/Users/meghanagaur/BonusProject/codes/estimate-mf-rates")

# Load the vacancy, unemploymentdatda
df = DataFrame(CSV.File(pwd()*"/data/mf_rates.csv"))

# get relevant data
idx       = findfirst(!ismissing, df.tightness)
frates    = df.q_frate[idx:end]
tightness = df.tightness[idx:end]
fr(θ, ι) = 1/(1 + θ^(-ι))^(1/ι)

# objective function to be minimized,
# compares implied frate to actual frate
function obj(ι, θ, frate_d)
    frate_m = fr.(θ, ι) # compute implied job-finding rate, given θ and ι
    return (mean(abs.(frate_m - frate_d)))
end


# match the quarterly job-finding rate data, given tightness in the data
opt     = optimize(ι -> obj(ι, tightness, frates), 0.2, 20)
ι_opt   = Optim.minimizer(opt)
#3.5249128585043454

# match average quarterly job-finding rate, for θ=1
opt     = optimize(ι -> obj(ι, 1, df.avg_qfrate[1]), 0.2, 20)
ι_opt   = Optim.minimizer(opt)
#2.7960718186676154