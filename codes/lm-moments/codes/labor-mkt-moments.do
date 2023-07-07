clear all

cd "/Users/meghanagaur/BonusProject/codes/lm-moments/codes"

global download = 0

if $download == 1 {

	* Download monthly, SA data from FRED (all in thousands, except for urate + ALP)
	freduse UEMPLT5 CE16OV UNEMPLOY JTSJOL UNRATE OPHNFB, clear
	save "../data/freddata", replace 
}


use "../data/freddata", clear

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

egen avg_frate       = mean(frate)
*gen u_ss            = avg_srate/(avg_srate + avg_frate)
gen tightness 		 = vacancies/unemp

summ urate srate frate tightness 

/* 
Variable |        Obs        Mean    Std. dev.       Min        Max
-------------+---------------------------------------------------------
       urate |        828    .0576606     .016538       .025       .108
       srate |        827    .0313995    .0067716    .015902   .0496018
       frate |        827    .4204166    .0892433   .1817443   .6911365
	   tightness |    229    .5594207    .2841447   .1528662   1.241864
*/ 

gen lurate              = log(urate)
tsfilter hp lurate_hp   = lurate, smooth(129600)

summ lurate_hp

export delimited "../data/lm_monthly.csv", replace

* Take quarterly averages
collapse (mean) urate frate srate tightness vacancies unemp alp, by(qdate)

rename qdate yq
tsset yq 

foreach var in urate frate srate tightness vacancies unemp alp {
	
	gen l`var'                = log(`var')
	tsfilter hp l`var'_hp     = l`var', smooth(100000)
	tsfilter hp l`var'_hp1600 = l`var', smooth(1600) 

	*pwcorr l`var'_hp l.l`var'_hp
	*pwcorr l`var'_hp1600 l.l`var'_hp1600
}


summ *hp 

/* 
    Variable |        Obs        Mean    Std. dev.       Min        Max
-------------+---------------------------------------------------------
   lurate_hp |        276    1.90e-10    .2034675  -.4171211   .4732535
   lfrate_hp |        276   -1.02e-10     .140245  -.4482797   .2523849
   lsrate_hp |        276   -1.75e-11    .0714271  -.2197686   .2316698
ltightness~p |         77    1.26e-09    .3955577  -.9267228   .6405027
lvacancies~p |         77    1.05e-09    .1818125  -.5046699   .3450061
-------------+---------------------------------------------------------
   lunemp_hp |        276    6.12e-10     .202573  -.4170066    .471583
     lalp_hp |        276    4.92e-12    .0169176  -.0466661   .0426264
*/

summ *hp1600 

pwcorr lurate_hp lfrate_hp  
pwcorr lurate_hp1600 lfrate_hp1600 

save  "../data/lm_quarterly", replace
