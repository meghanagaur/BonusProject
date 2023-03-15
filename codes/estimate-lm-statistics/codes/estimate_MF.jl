using CSV, DataFrames, Optim, StatsBase

cd(dirname(@__FILE__))

# Load the vacancy, unemploymentdatda
df = DataFrame(CSV.File("../data/lm_monthly.csv"))

# get relevant data
idx       = findfirst(!ismissing, df.vacancies)
frates    = df.frate[idx:end]
tightness = df.tightness[idx:end]
fr(θ, ι) = 1/(1 + θ^(-ι))^(1/ι)

"""
objective function to be minimized, compares implied frate to actual frate
norm = L2
"""
function obj(ι, θ, frate_d)
    frate_m = fr.(θ, ι) # compute implied job-finding rate, given θ and ι
    return sum((frate_m .- frate_d).^2)
end

"""
objective function to be minimized, compares implied frate to actual frate
sup norm
"""
function obj2(ι, θ, frate_d)
    frate_m = fr.(θ, ι) # compute implied job-finding rate, given θ and ι
    return (maximum(abs.(frate_m .- frate_d)))
end

# match monthly job-finding rates in the data, given tightness in the data - SSR obj
opt     = Optim.optimize(ι -> obj(ι, tightness[1:end-1], frates[1:end-1]), 0.2, 2, Brent())
ι_opt   = Optim.minimizer(opt)
#0.8956788079165414
 
# match monthly job-finding rates in the data, given tightness in the data - MAX obj
opt     = Optim.optimize(ι -> obj2(ι, tightness[1:end-1], frates[1:end-1]), 0.2, 2, Brent())
ι_opt   = Optim.minimizer(opt)
#0.835656603805031

# match the average job-finding rate data over shorter series, given tightness in the data
opt     = Optim.optimize(ι -> obj(ι, tightness[1:end-1],  mean(frates[1:end-1])), 0.2, 2, Brent())
ι_opt   = Optim.minimizer(opt)
#0.8299221156411114

# match the average job-finding rate data over shorter series, given average tightness in the data
opt       = Optim.optimize(ι -> obj(ι, mean(tightness), mean(frates[1:end-1])), 0.2, 2, Brent())
ι_opt     = Optim.minimizer(opt)
#0.8961947884957904

# match the average job-finding rate data over ENTIRE series, given tightness in the data
opt     = Optim.optimize(ι -> obj(ι, tightness[1:end-1], df.avg_frate[1]), 0.2, 2, Brent())
ι_opt   = Optim.minimizer(opt)
#1.1399219041519928

# match the average job-finding rate over ENTIRE SERIES, for average tightness in shorter series 
opt       = Optim.optimize(ι -> obj(ι, mean(tightness),  df.avg_frate[1]), 0.2, 2, Brent())
ι_opt     = Optim.minimizer(opt)
#1.3297469539291658

##################################

# match average monthly job-finding rate, for θ=0.6
opt       = Optim.optimize(ι -> obj(ι, 0.6, df.avg_frate[1]), 0.2, 20,  Brent())
ι_opt     = Optim.minimizer(opt)
#1.2113749876993996
fr(0.6,ι_opt)

# match average monthly job-finding rate, for θ=1
opt       = Optim.optimize(ι -> obj(ι, 1, df.avg_frate[1]), 0.2, 20,  Brent())
ι_opt     = Optim.minimizer(opt)
#0.7999305738095385
fr(1,ι_opt)

# switch to job-finding rate of 0.47 for θ =1, u_ss = 0.06
opt       = Optim.optimize(ι -> obj(ι, 1, 0.47), 0.2, 20,  Brent())
ι_opt     = Optim.minimizer(opt)
#0.9180482789470614

fr(1,1.25)
fr(1,0.92)
fr(1,0.8)
s/(s+f)  = 0.06
