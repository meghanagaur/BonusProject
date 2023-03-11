
* Download the seasonally adjusted series of ALP
freduse OPHNFB UNRATE, clear

* sample period
global start_year = 1951
global end_year   = 2019

keep if (year(daten) <= $end_year) & (year(daten) >= $start_year)

* quarterly date
gen qdate = qofd(daten)
format %tq qdate

rename OPHNFB alp
rename UNRATE urate 

collapse (mean) urate alp, by(qdate)

* Take logs
gen lalp   = log(alp)
gen lurate = log(urate)

* HP Filter the series
tsset qdate
tsfilter hp lalp_hp = lalp, smooth(100000)

ac lalp_hp, lags(1) generate(ac)
list ac  in 1, clean

* Estimate the AR(1) process
reg lalp_hp l.lalp_hp
predict resid, residuals

* extra check
reg lalp_hp l.lalp_hp, nocons

* Bootstrap the estimate

* create a new variable for the lag
gen lalp_hp_l = L.lalp_hp

bootstrap _b[lalp_hp_l], nodots reps(10000): reg lalp_hp lalp_hp_l
gen rho = e(b)[1,1]

* get standard deviation of HP-filtered log ALP
bootstrap std=r(sd), nodots reps(10000): sum lalp_hp
gen std = e(b)[1,1]

* get implied standard deviation of innovation
gen sigma_e = sqrt((std^2)*(1-rho^2))

* compute standard deviation of residuals (estimate for innovation)
bootstrap std=r(sd), nodots reps(10000): sum resid

