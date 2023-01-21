using CSV, DataFrames, Optim, StatsBase

cd(dirname(@__FILE__))

# Load the vacancy, unemploymentdatda
df = DataFrame(CSV.File(pwd()*"/data/mf_rates_monthly.csv"))

# get relevant data
idx       = findfirst(!ismissing, df.vacancies)
frates    = df.frate[idx:end]
tightness = df.vacancies[idx:end]./df.unemp[idx:end]
fr(θ, ι) = 1/(1 + θ^(-ι))^(1/ι)

"""
objective function to be minimized, compares implied frate to actual frate
obj = sum of squared residuals
"""
function obj(ι, θ, frate_d)
    frate_m = fr.(θ, ι) # compute implied job-finding rate, given θ and ι
    return sum((frate_m .- frate_d).^2)
end

"""
objective function to be minimized, compares implied frate to actual frate
obj=max
"""
function obj2(ι, θ, frate_d)
    frate_m = fr.(θ, ι) # compute implied job-finding rate, given θ and ι
    return (maximum(abs.(frate_m - frate_d)))
end

# match monthly job-finding rates in the data, given tightness in the data
opt     = optimize(ι -> obj(ι, tightness[1:end-1], frates[1:end-1]), 0.2, 20)
ι_opt   = Optim.minimizer(opt)
#0.7963342768320739
 
# match monthly job-finding rates in the data, given tightness in the data - MAX obj
opt     = optimize(ι -> obj2(ι, tightness[1:end-1], frates[1:end-1]), 0.2, 20)
ι_opt   = Optim.minimizer(opt)
#0.74483547685439

# match the average job-finding rate data over ENTIRE series, given tightness in the data
opt     = optimize(ι -> obj(ι, tightness[1:end-1], df.avg_frate[1]), 0.2, 20)
ι_opt   = Optim.minimizer(opt)
#1.0894360621077501

# match the average job-finding rate over ENTIRE SERIES, for average tightness in shorter series <- gives 1.27
opt       = optimize(ι -> obj(ι, mean(tightness),  df.avg_frate[1]), 0.2, 20)
ι_opt     = Optim.minimizer(opt)
#1.2511388983710665

# match average job-finding rate, for average tightness in shorter series
opt       = optimize(ι -> obj(ι, mean(tightness), mean(frates[1:end-1])), 0.2, 20)
ι_opt     = Optim.minimizer(opt)
#0.7938970591115313





# match average monthly job-finding rate, for θ=1
opt       = optimize(ι -> obj(ι, 1, df.avg_frate[1]), 0.2, 20)
ι_opt     = Optim.minimizer(opt)
#0.7814827601364035

# match average monthly job-finding rate, for θ=1.5
opt     = optimize(ι -> obj(ι, 1.5, df.avg_frate[1]), 0.2, 20)
ι_opt   = Optim.minimizer(opt)
#0.643888401555918

# match average monthly job-finding rate, for θ=2
opt     = optimize(ι -> obj(ι, 2, df.avg_frate[1]), 0.2, 20)
ι_opt   = Optim.minimizer(opt)
#0.5780803082372136


# match average monthly job-finding rate, for θ=3
opt     = optimize(ι -> obj(ι, 3, df.avg_frate[1]), 0.2, 20)
ι_opt   = Optim.minimizer(opt)
#0.5095238249429132