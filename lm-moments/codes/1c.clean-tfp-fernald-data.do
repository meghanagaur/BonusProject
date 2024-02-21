/*******************************************************************************
Clean the Ferrnald TFP data and compute the autocorrelation
and unconditional standard deviation of log quarterly TFP. This code  uses
TFP (utilization and non-utilization adjusted) data, which was downloaded
from Fernald's websit on Feburary 14th, 2023 data release. 
*******************************************************************************/
clear all
set more off 
macro drop _all
set graphics off

* Set local directory
cd "/Users/meghanagaur/BonusProject/lm-moments/"
global base       = c(pwd)
global download   = 1
global data 	  = "${base}/data"

* Sample period
global start_year = 1951
global end_year   = 2019

*******************************************************************************/
* Load the Fernald data series; dtfp_util = utilization-adjusted TFP
import excel date = A dtfp = N dtfp_unadj = L using "$data/raw/fernald-tfp.xlsx", sheet("quarterly") clear

* Clean the data
drop if _n == 1 | _n ==2

* Generate date variables
gen year    = substr(date, 1, 4)
gen quarter = substr(date, 7, 7)

* Destring relevant variables
* dtfp_unadj = business sector tfp 
* dtfp       = utilization-adjusted tfp
destring dtfp dtfp_unadj year quarter, force replace
drop if missing(year) | missing(quarter)

* Generate time series for log quarterly TFP
recast int quarter
gen yq = yq(year, quarter)
drop date 
format yq %tq
tsset yq

* Set base year/quarter to 1947
foreach var in tfp tfp_unadj {

	* Cumulate TFP changes. 
	gen l`var'      = log(100) if _n == 1

	/* From Fernald data sheet: All variables are percent changes at an 
	annual rate (=400 * change in natural log).*/
	local N = _N
	forvalues i = 2/`N' {
		replace l`var' = l.l`var' + d`var'/400 in `i'
	}
	
	* drop original variable
	drop d`var'
}

* Apply date restrictions
keep if (year <= $end_year) & (year >= $start_year)

* Try seasonally adjusting by regressing on quarterly dummies
foreach var in ltfp ltfp_unadj {
	reghdfe `var', absorb(quarter) resid(`var'_sa)
}

* HP-filter with smoothing parameter 
foreach var in ltfp ltfp_unadj ltfp_sa ltfp_unadj_sa {
	tsfilter hp `var'_hp     = `var', smooth(100000) // lambda from Shimer (2005)
	tsfilter hp `var'_hp1600 = `var', smooth(1600)   //  standard choice of lambda
}

* Save the data 
drop quarter 
save "data/clean/fernald_tfp", replace 

*******************************************************************************
* Unconditional standard deviation of utilization-adjusted TFP
summ ltfp_hp ltfp_hp1600 ltfp_sa_hp ltfp_sa_hp1600

* results for non-SA data:
* sigma = .0160301 for 10^5
* sigma = .0096429 for 1600

* Compute autocorrelations for lambda = 10^5
pwcorr ltfp_hp l.ltfp_hp
pwcorr ltfp_hp1600 l.ltfp_hp1600

* results for non-SA data:
* rho = 0.87 for lambda = 10^5
* rho = 0.65 for lambda = 1600

pwcorr ltfp_sa_hp l.ltfp_sa_hp
pwcorr ltfp_sa_hp1600 l.ltfp_sa_hp1600

* results for non-SA data:
* rho = 0.83 for lambda = 10^5
* rho = 0.57 for lambda = 1600

* Look at correlations
pwcorr ltfp ltfp_unadj
pwcorr ltfp_hp1600 ltfp_unadj_hp1600
pwcorr ltfp_hp ltfp_unadj_hp

pwcorr ltfp_sa ltfp_unadj_sa
pwcorr ltfp_sa_hp1600 ltfp_unadj_sa_hp1600
pwcorr ltfp_sa_hp ltfp_unadj_sa_hp1600 



