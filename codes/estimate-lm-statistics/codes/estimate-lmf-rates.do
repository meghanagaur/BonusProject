cd "/Users/meghanagaur/BonusProject/codes/estimate-lm-statistics/codes"

* Download monthly SA unemp + emp data from CPS and vacancies from JOLTS (all in thousands)
freduse UEMPLT5 CE16OV UNEMPLOY JTSJOL UNRATE, clear
 
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

rename UEMPLT5 st_unemp
rename UNEMPLOY unemp
rename CE16OV emp
rename JTSJOL vacancies
rename UNRATE urate 

* Convert unemployment rate
replace urate = urate*.01

* Shimer adjustment for change in CPS structure
gen st_unemp_adj     = st_unemp
replace st_unemp_adj = st_unemp*1.1 if mdate > ym(1994,1)

* get mothly job-finding and separation rate, following Shimer
gen frate  		     = 1 - (f.unemp - f.st_unemp_adj)/unemp
gen srate   		 = f.st_unemp_adj/(emp*(1 - 0.5*frate))

*gen u_ss            = avg_srate/(avg_srate + avg_frate)
gen tightness 		 = vacancies/unemp

summ urate srate frate

export delimited "../data/mf_rates_monthly.csv", replace

* take quarterly averages to smooth data
collapse (mean) urate frate srate vacancies unemp, by(qdate)

tsset qdate 

gen log_urate        = log(urate)
tsfilter hp urate_hp = log_urate, smooth(100000) trend(urate_trend)

summ urate_trend


/*
* hp-filter the two series
tsset qdate
tsfilter hp frate_hp = frate, smooth(100000) trend(frate_trend)
tsfilter hp srate_hp = srate, smooth(100000)  trend(srate_trend)
*/

* compute implied quarterly probabilities
gen q_frate       = 1 - (1 - frate)^3
gen q_srate       = 1 - (1 - srate)^3

* compute the avg frate + sep rate
egen avg_qfrate    = mean(q_frate) // if !missing(vacancies)
egen avg_qsrate    = mean(q_srate) // if !missing(vacancies)
egen avg_urate     = mean(urate)

* generate tightness
gen tightness     = vacancies/unemp

export delimited "../data/mf_rates_quarterly.csv", replace
