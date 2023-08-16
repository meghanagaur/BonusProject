global dir = "/Users/meghanagaur/BonusProject/codes/lm-moments/codes"
cd $dir 
clear all 

* Load the Shimer replication data 
import excel "../data/shimer-data.xls", sheet("Monthly Data") firstrow clear

drop if _n < 6

* Take quarterly averages to smooth data
rename UnemploymentRate urate 
rename HelpWantedAdvertising hwol 
rename JobOpenings jolts 
rename JobFindingRate  frate 
rename SeparationRate srate
rename UnemploymentLevel unemp

* Format the date variable 
gen date = date(Title, "DMY")
drop Title

format date %td

gen yq = qofd(date)
format yq %tq

destring urate frate srate hwol jolts unemp, force replace

* Take quarterly avaerages 
collapse (mean) urate frate srate hwol jolts unemp, by(yq)

* Compute quarterly tightness using quarterly averages of unemployment and vacancies.
gen tightness = hwol/unemp

* Save a temp file with the cleaned monthly data 
tempfile shimer 
save `shimer', replace

import excel "../data/shimer-data.xls", sheet("Quarterly Data") firstrow clear

rename PR alp
rename Title date
destring alp, force replace

drop if _n < 7

* Quarterly date
gen year = substr(date, 1, 4)
gen qtr  = substr(date, 8, 8)
destring year qtr, force replace

gen yq  = yq(year, qtr)
format %tq yq

merge 1:1 yq using `shimer'

tsset yq 

* HP filter the data 
foreach var in urate frate srate tightness hwol jolts unemp alp {
	
	gen l`var'_sh                = log(`var')
	tsfilter hp l`var'_hp_sh     = l`var'_sh, smooth(100000) 
	tsfilter hp l`var'_hp1600_sh = l`var'_sh, smooth(1600) 

	reg l`var'_hp_sh  l.l`var'_hp_sh
	reg l`var'_hp1600_sh  l.l`var'_hp1600_sh 
}

summ l*_hp_sh
pwcorr l*_hp_sh

keep  yq ltightness* lalp* lurate* lhwol* lunemp*
 
drop if missing(yq)

save "../data/shimer", replace
