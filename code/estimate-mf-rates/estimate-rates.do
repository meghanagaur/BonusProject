cd "/Users/meghanagaur/BonusProject/code/estimate-mf-rates"

* Download monthly SA unemp + emp data from CPS and vacancies from JOLTS (all in thousands)
freduse UEMPLT5 CE16OV UNEMPLOY JTSJOL, clear
 
* truncate covid for nows
keep if year(daten) < 2020

gen mdate = mofd(daten)
format mdate %tm

tsset mdate

rename UEMPLT5 st_unemp
rename UNEMPLOY unemp
rename CE16OV emp
rename JTSJOL vacancies

* gen mothly job-finding and separation rate, following Shimer
gen frate   = 1 - (f.unemp - f.st_unemp)/unemp
gen srate   = f.st_unemp/(emp*(1 - 0.5*frate))

/* generate quarters
gen qdate   = qofd(daten)

* take quarterly averages 
collapse (mean) frate srate, by(qdate)

* hp-filter the two series
tsset qdate
tsfilter hp frate_hp = frate, smooth(100000) trend(frate_trend)
tsfilter hp srate_hp = srate, smooth(100000)  trend(srate_trend)
*/

* compute the avg frate + sep rate
egen avg_frate    = mean(frate) // if !missing(vacancies)
egen avg_srate    = mean(srate) // if !missing(vacancies)

* compute quarterly probabilities
gen q_frate       = 1 - (1 - avg_frate)^3
gen q_srate       = 1 - (1 - avg_srate)^3

* generate tightness
gen tightness     = vacancies/unemp

export delimited "data/mf_rates.csv", replace
