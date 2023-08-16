/* Clean the unemployment/employment, vacancy (JOLTS), and wage data. */

global dir = "/Users/meghanagaur/BonusProject/codes/lm-moments/"

cd $dir 

clear all

global download = 1

if $download == 1 {

	* Download monthly, SA data from FRED (all in thousands, exc. urate, wages, PCE)
	freduse UEMPLT5 CE16OV UNEMPLOY JTSJOL UNRATE AHETPI PCEPILFE, clear
	save "data/freddata_lm", replace 
}

use "data/freddata_lm", clear

* Sample period
global start_year = 1951
global end_year   = 2019

keep if (year(daten) <= $end_year) & (year(daten) >= $start_year)

* Monthly date
gen mdate = mofd(daten)
format mdate %tm

* Quarterly date
gen qdate = qofd(dofm(mdate))
format %tq qdate

tsset mdate

* Rename variables 
rename UEMPLT5 st_unemp
rename UNEMPLOY unemp
rename CE16OV emp
rename JTSJOL vacancies
rename UNRATE urate 
rename AHETPI wages
rename PCEPILFE pce

* Deflate wages
replace wages        = wages/pce

* Convert unemployment from % to rate
replace urate        = urate*0.01

* Shimer adjustment for change in CPS structure
gen st_unemp_adj     = st_unemp
replace st_unemp_adj = st_unemp*1.1 if mdate > ym(1994, 1)

* Compute mothly job-finding and separation rate, following Shimer
gen frate  		     = 1 - (f.unemp - f.st_unemp_adj)/unemp
gen srate   		 = f.st_unemp_adj/(emp*(1 - 0.5*frate))
egen avg_frate       = mean(frate)
egen avg_srate       = mean(srate)
gen tightness 		 = vacancies/unemp

summ urate srate frate tightness 

/* 
Variable |        Obs        Mean    Std. dev.       Min        Max
-------------+---------------------------------------------------------
       urate |        828    .0576606     .016538       .025       .108
       srate |        827    .0313995    .0067716    .015902   .0496018
       frate |        827    .4204166    .0892433   .1817443   .6911365
	   tightness |    229    .5594207    .2841447   .1528662   1.241864
*/ 

export delimited "data/lm_monthly.csv", replace

* Only one month of data in 2000
replace vacancies = . if year(daten) == 2000

* Take quarterly averages
collapse (mean) urate frate srate vacancies unemp wages, by(qdate)

* Generate quarterly tightness using quarterly averages of v and u
gen tightness = vacancies/unemp 

rename qdate yq
tsset yq 

foreach var in urate frate srate tightness vacancies unemp wages {
	
	gen l`var'                = log(`var')
	tsfilter hp l`var'_hp     = l`var', smooth(100000)
	tsfilter hp l`var'_hp1600 = l`var', smooth(1600) 
}

save  "data/lm_quarterly", replace
