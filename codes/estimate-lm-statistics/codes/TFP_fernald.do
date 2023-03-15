clear all

cd "/Users/meghanagaur/BonusProject/codes/estimate-lm-statistics/codes"

*set graphics off

* sample period
global start_year = 1951
global end_year   = 2019

* Load the Fernald data series
import excel date = A dtfp_util = N using "../data/fernald-tfp.xlsx", sheet("quarterly") clear

* Clean the data
drop if _n == 1 | _n ==2

gen year    = substr(date, 1,4)
gen quarter = substr(date, 7,7)

destring dtfp_util year quarter, force replace

drop if missing(year) | missing(quarter)

keep if (year <= $end_year) & (year >= $start_year)

recast int quarter

gen yq = yq(year, quarter)
drop date
format yq %tq

tsset yq

* set base year
gen ltfp      = log(100) if _n == 1

local N = _N
forvalues i = 2/`N' {
	replace ltfp = l.ltfp + dtfp/400 in `i'
}

* HP-filter with smoothing parameter 
tsfilter hp ltfp_hp     = ltfp, smooth(100000)
tsfilter hp ltfp_hp1600 = ltfp, smooth(1600)

* Standard deviations
summ ltfp_hp ltfp_hp1600

* .0160301 for 10^5
* .0096429 for 1600

* Compute autocorrelations for 10^5
ac ltfp_hp, lags(1) generate(ac)
list ac  in 1, clean
pwcorr ltfp_hp l.ltfp_hp

* Compute autocorrelations for 1600
ac ltfp_hp1600, lags(1) generate(ac_low)
list ac_low in 1, clean
list ltfp_hp1600 l.ltfp_hp1600

* 0.87 for 10^5
* 0.65 for 1600

merge 1:1 yq using "../data/lm_quarterly", keepusing (l*) nogen

tsset yq 

* summarize and produce correlations
summ lalp_hp ltfp_hp lurate_hp lfrate_hp  
pwcorr lalp_hp ltfp_hp lurate_hp lfrate_hp  

summ lalp_hp1600 ltfp_hp1600 lurate_hp1600 lfrate_hp1600 
pwcorr lalp_hp1600 ltfp_hp1600 lurate_hp1600 lfrate_hp1600 

save "../data/quarterly_data", replace

* do some extra checks
tsset 
reg lurate ltfp
reg lurate lalp

reg f1.lurate ltfp
reg f1.lurate lalp
