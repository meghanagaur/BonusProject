using CSV, DataFrames, StatsBase

# website used for extraction: https://automeris.io/WebPlotDigitizer/
cd("/Users/meghanagaur/BonusProject/codes/extract-moments")

# Compute std and average of wage growth from the extracted picture data
function get_stat(file_str)
    
    # load the data
    df = DataFrame(CSV.File(pwd()*"/data/csv/"*file_str*".csv", header=false))
    rename!(df,:Column1 => :wchange)
    rename!(df,:Column2 => :freq)

    # correct for manual errors during extraction
    df.freq_norm  = abs.(df.freq)                       # frequency should be nonnegative
    df.freq_norm  = df.freq_norm./sum(df.freq_norm )    # frequency should sum to 1

    # Compute expectations
    exp_wchange   = sum(df.freq_norm.*df.wchange)       
    exp_wchange2  = sum(df.freq_norm.*(df.wchange.^2))

    # Compute std = sqrt(var) = E[x^2] - E[x]^2 
    std_wchange   = sqrt(exp_wchange2 - exp_wchange.^2)

    # return std, exp 
    return std_wchange/100, exp_wchange/100
end

get_stat("qtrly_wage_weighted_old")   # quarterly wage changes
get_stat("qtrly_wage4q_weighted")     # annual wage changes
get_stat("sal_qtrly_ern_weighted")    # salary - earnings changes
get_stat("sal_qtrly_wage_weighted")   # salary - wage changes
get_stat("hrly_qtrly_wage_weighted")  # quarterly wage changes - hourly workers



