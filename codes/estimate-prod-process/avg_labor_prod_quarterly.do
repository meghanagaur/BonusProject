
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



* useful check
reg resid l.resid
predict resid2, residuals
reg resid l.resid, nocons

* try boostrapping
gen lresid = L.resid

bootstrap  _b[lresid], nodots reps(10000): reg resid lresid
gen rho = e(b)[1,1]

* get variance of filtered log productivity
bootstrap std=r(sd), nodots reps(10000): sum resid
gen std = e(b)[1,1]

* get implied variance of innovation
gen sigma_e = sqrt((std^2)*(1-rho^2))

* get variance of residuals
bootstrap std=r(sd), nodots reps(10000): sum resid2
