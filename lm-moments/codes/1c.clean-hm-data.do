/*******************************************************************************
Get the Shimer data
*******************************************************************************/
clear all
set more off 
macro drop _all

* Set local director
cd "/Users/meghanagaur/BonusProject/lm-moments/"
global base       = c(pwd)
global download   = 1
global data 	  = "${base}/data"

* Sample period
global start_year = 1951
global end_year   = 2019
global lambda 	  = 100000
*******************************************************************************/
* Load the HM replication data 
 use "$data/raw/hm-data", clear
 
sort year quarter
drop if year < 1951
drop if year > 2004

* Re-name variables
rename output_pp alp
rename vacancies v_hwi
gen wages         = labor_share*alp
gen unemp	      = uirate/100
gen theta_hwi     = v_hwi/unemp
drop labor_share uirate

* HP-filter the data
gen yq = yq(year, quarter)
tsset  yq 
format yq %tq
foreach var in  v_hwi theta_hwi unemp alp wages {
	
	* Take logs and HP-filter
	gen l`var'_hm                = log(`var')
	tsfilter hp l`var'_hp_hm     = l`var'_hm, smooth($lambda) 
	drop `var'
}

drop if missing(yq)

* Save the data
save "${data}/clean/hm", replace
