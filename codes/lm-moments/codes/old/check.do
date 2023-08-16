clear all

cd "/Users/meghanagaur/BonusProject/codes/lm-moments/codes"

insheet using "../data/mf_rates_monthly.csv", clear 

* Take quarterly averages
collapse (mean) urate frate srate vacancies unemp , by(qdate)

gen yq = quarterly(qdate, "YQ")
format %tq yq

tsset yq 

gen log_urate                = log(urate)
tsfilter hp urate_hp         = log_urate, smooth(100000) trend(urate_trend)
tsfilter hp urate_hp_1600    = log_urate, smooth(1600) 

summ urate_hp urate_trend urate_hp_1600


