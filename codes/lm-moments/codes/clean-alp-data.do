cd "/Users/meghanagaur/BonusProject/codes/lm-moments/codes"

clear all

global download = 1

if $download == 1 {

	* Download the quarterly seasonally adjusted series of ALP from FRED
	freduse OPHNFB, clear
	
	* Restrict the sample period
	global start_year = 1951
	global end_year   = 2019

	keep if (year(daten) <= $end_year) & (year(daten) >= $start_year)

	* Quarterly date
	gen yq = qofd(daten)
	format %tq yq

	rename OPHNFB alp

	* Take logs
	gen lalp   = log(alp)

	* HP Filter the series
	tsset yq
	tsfilter hp lalp_hp     = lalp, smooth(100000)
	tsfilter hp lalp_hp1600 = lalp, smooth(1600)
	
	* Save the data
	save "../data/freddata_alp", replace 
}

* Load the data
use "../data/freddata_alp", clear

* Compute autocorrelations
ac lalp_hp1600, lags(1) generate(ac_hp)
list ac_hp  in 1, clean

ac lalp_hp, lags(1) generate(ac)
list ac  in 1, clean

pwcorr lalp_hp1600 l.lalp_hp1600
pwcorr lalp_hp l.lalp_hp

* rho = 0.72 with lambda = 1600
* rho = 0.89 with lambda = 10^5

* Get the unconditional standard deviation
summ lalp_hp lalp_hp1600

* sigma = 0.017 with lambda = 1600
* sigma = 0.0103 with lambda = 10^5

/* Bootstrap the estimates

* create a new variable for the lag
gen lalp_hp_l = L.lalp_hp

bootstrap _b[lalp_hp_l], nodots reps(10000): reg lalp_hp lalp_hp_l
gen rho = e(b)[1,1]

* get standard deviation of HP-filtered log ALP
bootstrap std=r(sd), nodots reps(10000): sum lalp_hp
gen std = e(b)[1,1]

* get implied standard deviation of innovation (conditional st dev)
gen sigma_e = sqrt((std^2)*(1-rho^2))

* compute standard deviation of residuals (estimate for innovation; conditional st dev)
bootstrap std=r(sd), nodots reps(10000): sum resid

* merge with labor market data 
merge 1:1 yq using "../../estimate-lm-statistics/data/quarterly_lm", keepusing (l*) nogen 
merge 1:1 yq using "../../estimate-lm-statistics/data/shimer", keepusing (*sh) nogen

summ lalp_hp lurate_hp lfrate_hp  
pwcorr lalp_hp lurate_hp lfrate_hp  

summ lalp_hp1600 lurate_hp1600 lfrate_hp1600
pwcorr lalp_hp1600 lurate_hp1600 lfrate_hp1600
*/


