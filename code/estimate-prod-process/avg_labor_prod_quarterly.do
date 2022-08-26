
freduse OPHNFB, clear

gen year = year(daten)
keep if year < 2020

rename OPHNFB alp
 
* generate log alp
gen lalp = log(alp)



gen qdate = qofd(daten)
format qdate %tq
tsset qdate
tsfilter hp resid = lalp, smooth(100000)

/*
* detrend log avg labor productivity using linear time trend and a constant
sort daten
gen t = _n
reg lalp t
predict resid, residuals
tsset
*/
 
* estimate AR coefficients for log alp
gen lresid = L1.resid

reg resid l.resid, nocons
reg resid l.resid

summ resid


* get variance of residuals
predict resid2, residuals
summ resid2

bootstrap std=r(sd), reps(1000): sum resid2
bootstrap  _b[lresid], reps(1000): reg resid lresid, nocons
