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
    return (maximum(abs.(frate_m .- frate_d)))
end

# match monthly job-finding rates in the data, given tightness in the data
opt     = optimize(ι -> obj(ι, tightness[1:end-1], frates[1:end-1]), 0.2, 20, Brent())
ι_opt   = Optim.minimizer(opt)
#0.8956788079165414
 
# match monthly job-finding rates in the data, given tightness in the data - MAX obj
opt     = optimize(ι -> obj2(ι, tightness[1:end-1], frates[1:end-1]), 0.2, 20, Brent())
ι_opt   = Optim.minimizer(opt)
#0.835656603805031

# match the average job-finding rate data over shorter series, given tightness in the data
opt     = optimize(ι -> obj(ι, tightness[1:end-1],  mean(frates[1:end-1])), 0.2, 20, Brent())
ι_opt   = Optim.minimizer(opt)
#0.8299221156411114

# match the average job-finding rate data over ENTIRE series, given tightness in the data
opt     = optimize(ι -> obj(ι, tightness[1:end-1], df.avg_frate[1]), 0.2, 20, Brent())
ι_opt   = Optim.minimizer(opt)
#1.1399219041519928

# match the average job-finding rate over ENTIRE SERIES, for average tightness in shorter series <- gives 1.27
opt       = optimize(ι -> obj(ι, mean(tightness),  df.avg_frate[1]), 0.2, 20, Brent())
ι_opt     = Optim.minimizer(opt)
#1.3297469539291658

# match average job-finding rate, for average tightness in shorter series
opt       = optimize(ι -> obj(ι, mean(tightness), mean(frates[1:end-1])), 0.2, 20, Brent())
ι_opt     = Optim.minimizer(opt)
#0.8961947884957904

##################################

# match average monthly job-finding rate, for θ=0.6
opt       = optimize(ι -> obj(ι, 0.6, df.avg_frate[1]), 0.2, 20,  Brent())
ι_opt     = Optim.minimizer(opt)
#1.2113749876993996
fr(0.6,ι_opt)

# match average monthly job-finding rate, for θ=1
opt       = optimize(ι -> obj(ι, 1, df.avg_frate[1]), 0.2, 20,  Brent())
ι_opt     = Optim.minimizer(opt)
#0.7999305738095385
fr(1,ι_opt)

# match average monthly job-finding rate, for θ=1.5
opt     = optimize(ι -> obj(ι, 1.5, df.avg_frate[1]), 0.2, 20,  Brent())
ι_opt   = Optim.minimizer(opt)
#0.6565200993752364
fr(1.5,ι_opt)

