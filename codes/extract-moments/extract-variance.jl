using CSV, DataFrames, Optim, StatsBase

cd("/Users/meghanagaur/BonusProject/codes/extract-moments")

# Compute std from the extracted picture data
function get_std(file_str)
    df = DataFrame(CSV.File(pwd()*"/data/csv/"*file_str*".csv",header=false))
    rename!(df,:Column1 => :wchange)
    rename!(df,:Column2 => :freq)


    # correct for manual errors during extraction
    df.freq_norm  = abs.(df.freq)
    df.freq_norm  = df.freq_norm./sum(df.freq_norm )
    exp_wchange   = sum(df.freq_norm.*df.wchange)
    exp_wchange2  = sum(df.freq_norm.*(df.wchange.^2))

    std_wchange   = sqrt(exp_wchange2 - exp_wchange.^2)
    return std_wchange/100, 
end

get_std("qtrly_wage_weighted")       # quarterly wage changes
get_std("qtrly_wage4q_weighted")     # annual wage changes
get_std("sal_qtrly_ern_weighted")    # salary - earnings
get_std("sal_qtrly_wage_weighted")   # salary - wage
get_std("hrly_qtrly_wage_weighted")  # quarterly wage changes - hourly workers



