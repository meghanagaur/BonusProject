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
* Load the Shimer replication data 
import excel "${data}/raw/shimer-data.xls", sheet("Monthly Data") firstrow clear

* Basic cleaning
drop if _n < 6

* Rename variables
rename UnemploymentRate urate 
rename HelpWantedAdvertising hwol 
rename JobOpenings jolts 
rename JobFindingRate  frate 
rename SeparationRate srate
rename UnemploymentLevel unemp
destring urate frate srate hwol jolts unemp, force replace

* Format the date variable 
gen date = date(Title, "DMY")
drop Title
format date %td
gen yq = qofd(date)
format yq %tq

order yq* 
sort yq* 

* Take quarterly averages 
collapse (mean) urate frate srate hwol jolts unemp, by(yq)

* Save a temp file with the cleaned data 
tempfile shimer 
save `shimer', replace

* Load the labor productivity data
import excel "${data}/raw/shimer-data.xls", sheet("Quarterly Data") firstrow clear

* Basic formatting
drop if _n < 7
rename PR alp
destring alp, force replace

* Quarterly date
rename Title date
gen year = substr(date, 1, 4)
gen qtr  = substr(date, 8, 8)
destring year qtr, force replace
gen yq  = yq(year, qtr)
format %tq yq
drop qtr date 
order yq 
sort yq 

* Merge back in other data
merge 1:1 yq using `shimer'
tab _merge 
drop _merge 

* Compute quarterly tightness using quarterly averages of unemployment and vacancies.
rename hwol v_hwi 
rename jolts v_jolts 
gen theta_hwi = v_hwi/unemp 

* HP-filter the data
tsset yq 
foreach var in urate frate srate v_hwi v_jolts theta_hwi unemp alp {
	
	* Take logs and HP-filter
	gen l`var'_sh                = log(`var')
	tsfilter hp l`var'_hp_sh     = l`var'_sh, smooth($lambda) 
	drop `var'
}

drop if missing(yq)

* Save the data
save "${data}/clean/shimer", replace



