freduse OPHNFB, clear

gen year = year(daten)
keep if year >= 1970 & year < 2020

rename OPHNFB alp
 
* generate annual labor productivity by averaging quarterly
collapse (mean) alp, by(year) 

* generate log alp
gen lalp = log(alp)

* detrend log avg labor productivity using linear time trend
sort year
gen t = _n
reg lalp t
predict resid, residuals
 
* estimate AR coefficients for log alp
tsset t
reg resid l.resid, nocons

* get variance of residuals
predict resid2, residuals
summ resid2
