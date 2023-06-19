clear all

cd "/Users/meghanagaur/BonusProject/codes/estimate-lm-statistics/codes"

* Download monthly, SA data from FRED (all in thousands, except for urate + ALP)
freduse UEMPLT5 CE16OV UNEMPLOY JTSJOL UNRATE OPHNFB, clear
 
* Sample period
global start_year = 1951
global end_year   = 2019

keep if (year(daten) <= $end_year) & (year(daten) >= $start_year)

* monthly date
gen mdate = mofd(daten)
format mdate %tm

* quarterly date
gen qdate = qofd(dofm(mdate))
format %tq qdate

tsset mdate

* rename variables 
rename UEMPLT5 st_unemp
rename UNEMPLOY unemp
rename CE16OV emp
rename JTSJOL vacancies
rename UNRATE urate 
rename OPHNFB alp

replace urate = urate*0.01

* Shimer adjustment for change in CPS structure
gen st_unemp_adj     = st_unemp
replace st_unemp_adj = st_unemp*1.1 if mdate > ym(1994,1)

* get mothly job-finding and separation rate, following Shimer
gen frate  		     = 1 - (f.unemp - f.st_unemp_adj)/unemp
gen srate   		 = f.st_unemp_adj/(emp*(1 - 0.5*frate))

*gen u_ss            = avg_srate/(avg_srate + avg_frate)
gen tightness 		 = vacancies/unemp

summ urate srate frate tightness 

export delimited "../data/lm_monthly.csv", replace

* Take quarterly averages
collapse (mean) urate frate srate tightness vacancies unemp alp, by(qdate)

rename qdate yq
tsset yq 

foreach var in urate frate srate tightness vacancies unemp alp {
	
	gen l`var'                = log(`var')
	tsfilter hp l`var'_hp     = l`var', smooth(100000)
	tsfilter hp l`var'_hp1600 = l`var', smooth(1600) 

	pwcorr l`var'_hp l.l`var'_hp
	pwcorr l`var'_hp1600 l.l`var'_hp1600
}


summ *hp 
summ *hp1600 

pwcorr lurate_hp lfrate_hp  
pwcorr lurate_hp1600 lfrate_hp1600 

save  "../data/lm_quarterly", replace
