cd "/Users/meghanagaur/BonusProject/codes/estimate-prod-process/codes"

set graphics off

* Load the Fernald data series
import excel  date=A dtfp_util=N using "../data/quarterly_tfp.xlsx", sheet("quarterly") clear

* Clean the data

drop if _n == 1 | _n ==2

gen year    = substr(date, 1,4)
gen quarter = substr(date, 7,7)

destring dtfp_util year quarter, force replace

drop if missing(year) | missing(quarter)

recast int quarter

gen yq = yq(year, quarter)
drop date
format yq %tq

tsset yq

* set base year
keep if year <= 2019 & year >= 1951
gen tfp     = 100 if _n == 1

local N = _N
forvalues i = 2/`N' {
	replace tfp = exp(log(l.tfp) + dtfp/400) in `i'
}

* Compute logs 
gen ltfp = log(tfp)

* HP-filter with smoothing parameter 
tsfilter hp ltfp_hp     = ltfp, smooth(100000)
tsfilter hp ltfp_hp_low = ltfp, smooth(1600)

* Standard deviations
summ ltfp_hp  ltfp_hp_low

* Compute autocorrelations for 10^5
ac ltfp_hp, lags(1) generate(ac)
list ac  in 1, clean

reg ltfp_hp l.ltfp_hp

* Compute autocorrelations for 1600
ac ltfp_hp_low, lags(1) generate(ac_low)
list ac_low in 1, clean

reg ltfp_hp_low l.ltfp_hp_low
