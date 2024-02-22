/*******************************************************************************
Clean the unemployment/employment, vacancy (JOLTS), and wage data. 
*******************************************************************************/
clear 
set more off 
macro drop _all

* Set local director
cd "/Users/meghanagaur/BonusProject/lm-moments/"
global base       = c(pwd)
global download   = 0
global data 	  = "${base}/data"

* Sample period
global start_year = 1951
global end_year   = 2019
global lambda 	  = 100000
*******************************************************************************/
if $download == 1 {


	* Download monthly, SA data from FRED (all in thousands)
	* Note: COMPRNFB and OPHNFB are quarterly 
	import fred UEMPLT5 CE16OV UNEMPLOY JTSJOL UNRATE AHETPI COMPRNFB OPHNFB, clear
	
	* Rename variables 
	rename UEMPLT5 st_unemp
	rename UNEMPLOY unemp
	rename CE16OV emp
	rename JTSJOL v_jolts
	rename UNRATE urate 
	rename COMPRNFB wages
	rename OPHNFB alp

	* Monthly date
	gen mdate = mofd(daten)
	format mdate %tm
	tsset mdate
	drop date*
	order mdate 
	sort mdate
	
	save "${data}/clean/freddata_lm", replace 
	
	* Barnichon vacancy data (monthly)
	import excel using "${data}/raw/CompositeHWI.xlsx", clear cellrange(A8:C856) firstrow
	
	* date formatting
	rename year date
	gen year 			   = real(substr(string(date),1,4))
	bysort year: gen month = _n
	
	rename V_hwi v_hwi
	rename VLF v_hwi_rate 
	
	* Monthly date
	gen mdate = ym(year, month)
	format mdate %tm
	sort mdate

	* Save cleaned data
	save "${data}/clean/CompositeHWI", replace 

}

*******************************************************************************/
* Load the data
use "${data}/clean/freddata_lm", clear

* Merge in HWI composte data 
merge 1:1 mdate using "${data}/clean/CompositeHWI", keepusing(year month v*)
tab mdate if _merge == 1 | _merge == 2
drop _merge

* Restrict dates
keep if (year <= $end_year) & (year >= $start_year) 

* Convert unemployment rate from % to rate
replace urate        = urate*0.01

* Shimer adjustment for change in CPS structure
gen st_unemp_adj     = st_unemp
replace st_unemp_adj = st_unemp*1.1 if mdate > ym(1994, 1)

* Compute mothly job-finding and separation rate, following Shimer
gen frate  		     = 1 - (f.unemp - f.st_unemp_adj)/unemp
gen srate   		 = f.st_unemp_adj/(emp*(1 - 0.5*frate))

* Market tightness
gen theta_jolts		 = v_jolts/unemp
gen theta_hwi		 = v_hwi/unemp

* save monthly data
export delimited "${data}/clean/lm_monthly.csv", replace

* Quarterly date
gen yq = qofd(dofm(mdate))
format %tq yq

* Only one month of data in 2000
replace v_jolts = . if year(mdate) == 2000

* Take quarterly averages
collapse (mean) alp urate frate srate unemp v_jolts v_hwi wages, by(year yq)

* Generate quarterly tightness using quarterly averages of v and u
gen theta_jolts		 = v_jolts/unemp
gen theta_hwi		 = v_hwi/unemp

* HP-filter the data
tsset yq 

foreach var of varlist alp urate frate srate v_* theta_* unemp wages {
	
	* Take logs and HP-filter
	gen l`var'                = log(`var')
	tsfilter hp l`var'_hp     = l`var', smooth($lambda)
	drop `var'
}

save  "${data}/clean/lm_quarterly", replace
