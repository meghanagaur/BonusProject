clear all

cd "/Users/meghanagaur/BonusProject/codes/lm-moments/codes"

set graphics off

* sample period
global start_year = 1951
global end_year   = 2019

* Load the Fernald data series; dtfp_util = utilization-adjusted TFP
import excel date = A dtfp_util = N using "../data/fernald-tfp.xlsx", sheet("quarterly") clear

* Clean the data
drop if _n == 1 | _n ==2

* Generate date variables
gen year    = substr(date, 1, 4)
gen quarter = substr(date, 7, 7)

* Destring relevant variables
destring dtfp_util year quarter, force replace

drop if missing(year) | missing(quarter)

* Apply date restrictions
keep if (year <= $end_year) & (year >= $start_year)

recast int quarter

* Generate time series for log quarterly TFP
gen yq = yq(year, quarter)
drop date
format yq %tq

tsset yq

* Set base year
gen ltfp      = log(100) if _n == 1

* cumulate TFP changes
* from fernald data sheet: All variables are percent change at an annual rate (=400 * change in natural log). 
local N = _N
forvalues i = 2/`N' {
	replace ltfp = l.ltfp + dtfp/400 in `i'
}

* HP-filter with smoothing parameter 
tsfilter hp ltfp_hp     = ltfp, smooth(100000) // as in Shimer (2005)
tsfilter hp ltfp_hp1600 = ltfp, smooth(1600)  // more common choice 

* Standard deviations
summ ltfp_hp ltfp_hp1600

* sigma = .0160301 for 10^5
* sigma = .0096429 for 1600

* Compute autocorrelations for lambda = 10^5
ac ltfp_hp, lags(1) generate(ac)
list ac  in 1, clean
pwcorr ltfp_hp l.ltfp_hp

* Compute autocorrelations for lambda = 1600
ac ltfp_hp1600, lags(1) generate(ac_low)
list ac_low in 1, clean
list ltfp_hp1600 l.ltfp_hp1600

* rho = 0.87 for 10^5
* rho = 0.65 for 1600

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

reg ltightness ltfp
reg lurate ltfp
reg lurate ltfp
reg lfrate ltightness
reg f1.lurate ltfp
reg f1.lurate lalp
