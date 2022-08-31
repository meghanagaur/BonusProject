
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

* useful check
reg resid l.resid
predict resid2, residuals
reg resid l.resid, nocons
bootstrap  _b[lresid], reps(10000): reg resid lresid

* get variance of filtered log productivity
bootstrap std=r(sd), reps(10000): sum resid

* get variance of residuals
bootstrap std=r(sd), reps(10000): sum resid2
