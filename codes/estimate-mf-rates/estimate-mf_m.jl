using CSV, DataFrames, Optim, StatsBase

cd("/Users/meghanagaur/BonusProject/codes/estimate-mf-rates")

# Load the vacancy, unemploymentdatda
df = DataFrame(CSV.File(pwd()*"/data/mf_rates_monthly.csv"))

# get relevant data
idx       = findfirst(!ismissing, df.vacancies)
frates    = df.frate[idx:end]
tightness = df.vacancies[idx:end]./df.unemp[idx:end]
fr(θ, ι) = 1/(1 + θ^(-ι))^(1/ι)

# objective function to be minimized,
# compares implied frate to actual frate
function obj(ι, θ, frate_d)
    frate_m = fr.(θ, ι) # compute implied job-finding rate, given θ and ι
    return (maximum(abs.(frate_m - frate_d)))
end


# match the quarterly job-finding rate data, given tightness in the data
opt     = optimize(ι -> obj(ι, tightness[1:end-1], frates[1:end-1]), 0.2, 20)
ι_opt   = Optim.minimizer(opt)
#0.733462873251313

# match average quarterly job-finding rate, for θ=1
opt     = optimize(ι -> obj(ι, 1, df.avg_frate[1]), 0.2, 20)
ι_opt   = Optim.minimizer(opt)
#0.7814827601364035

# match average quarterly job-finding rate, for θ=1.5
opt     = optimize(ι -> obj(ι, 1.5, df.avg_frate[1]), 0.2, 20)
ι_opt   = Optim.minimizer(opt)
#0.643888401555918