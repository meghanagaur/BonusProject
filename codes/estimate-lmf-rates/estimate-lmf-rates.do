cd "/Users/meghanagaur/BonusProject/codes/estimate-lmf-rates"

* Download monthly SA unemp + emp data from CPS and vacancies from JOLTS (all in thousands)
freduse UEMPLT5 CE16OV UNEMPLOY JTSJOL UNRATE, clear
 
* truncate covid for nows
keep if year(daten) < 2020

gen mdate = mofd(daten)
format mdate %tm

tsset mdate

rename UEMPLT5 st_unemp
rename UNEMPLOY unemp
rename CE16OV emp
rename JTSJOL vacancies
rename UNRATE urate 

* get mothly job-finding and separation rate, following Shimer
gen frate   = 1 - (f.unemp - f.st_unemp)/unemp
gen srate   = f.st_unemp/(emp*(1 - 0.5*frate))

* generate quarters
gen qdate   = qofd(daten)
format qdate %tq


egen avg_frate    = mean(frate) 
egen avg_srate    = mean(srate) 

gen u_ss          = avg_srate/(avg_srate + avg_frate)


tsfilter hp frate_hp = frate, smooth(100000) trend(frate_trend)
tsfilter hp srate_hp = srate, smooth(100000)  trend(srate_trend)


*egen avg_frate_hp    = mean(frate_trend) 
*egen avg_srate_hp    = mean(srate_trend) 

export delimited "data/mf_rates_monthly.csv", replace

* take quarterly averages to smooth data
collapse (mean) urate frate srate vacancies unemp, by(qdate)

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

export delimited "data/mf_rates_quarterly.csv", replace
