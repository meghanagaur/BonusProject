/*******************************************************************************
Clean the unemployment/employment, vacancy (JOLTS), and wage data. 
*******************************************************************************/
clear all
set more off 
macro drop _all

* Set local director
cd "/Users/meghanagaur/BonusProject/lm-moments/"
global base       = c(pwd)
global download   = 1
global data 	  = "${base}/data"
global tables 	  = "${base}/tables"

*******************************************************************************/
* Load tightness and wage data 
use "$data/clean/lm_quarterly", clear

* Merge in TFP Fernald data 
merge 1:1 yq using "${data}/clean/fernald_tfp", keepusing (ltfp*) 
tab year if _merge != 3
drop _merge 

* Merge in the Shimer data 
merge 1:1 yq using "${data}/clean/shimer", keepusing (*sh*)
tab year if _merge != 3
drop _merge 
	
tsset yq 

* Variable labels 
foreach var of varlist *theta* {
	label var `var' "$\theta$"
}

foreach var of varlist *alp* {
	label var `var' "$ p $"
}

foreach var of varlist *tfp* {
	label var `var' "$z$ (adj.)"
}

foreach var of varlist ltfp_unadj* {
	label var `var' "$z$ (unadj.)"
}

foreach var of varlist *unemp* *urate* {
	label var `var' "$ u $"
}

foreach var of varlist *wage* {
	label var `var' "$ w $"
}


foreach var of varlist *hwi* {
	label var `var' "$ v $"
}

* Standard deviation with lambda = 10^5 <- longer time period
global vars 	   = "lalp_hp lunemp_hp lv_hwi_hp  ltheta_hwi_hp lwages_hp"
global shimer_vars = "lalp_hp_sh lunemp_hp_sh lv_hwi_hp_sh ltheta_hwi_hp_sh"

* 1951- 2019
estpost tabstat $vars, statistics(sd) 
esttab . using "${tables}/std.tex", replace label nostar booktabs not cells("lalp_hp(fmt(a3)) lunemp_hp(fmt(a3)) lv_hwi_hp(fmt(a3))  ltheta_hwi_hp(fmt(a3)) lwages_hp(fmt(a3))") noobs substitute("sd" "S.D." "lalp_hp" "$  p$" "lunemp_hp" "$ u $" "lv_hwi_hp" "$ v $" "ltheta_hwi_hp" "$ \theta $" "lwages_hp" "$ w $") nonumber compress  

* Shimer data
estpost tabstat $shimer_vars, statistics(sd) 
esttab . using "${tables}/std_sh.tex", label replace nostar booktabs not cells("lalp_hp_sh(fmt(a3)) lunemp_hp_sh(fmt(a3)) lv_hwi_hp_sh(fmt(a3))  ltheta_hwi_hp_sh(fmt(a3))") noobs substitute("sd" "S.D." "lalp_hp_sh" "$  p$" "lunemp_hp_sh" "$ u $" "lv_hwi_hp_sh" "$ v $" "ltheta_hwi_hp_sh" "$ \theta $") nonumber compress  

* Autocorrelations 

* 1951 - 2019
local tables ""
foreach var of global vars {
	
	gen l_`var' = l.`var'	
	local lab: variable label `var'
	label variable l_`var' "`lab' lag"
	
	eststo tab`var': estpost correlate `var' l_`var' 
	local tables = "`tables' tab`var'"
}

esttab `tables' using "${tables}/autocorr.tex", unstack not compress replace label nonumbers booktabs nostar noobs

* Shimer data
local tables_sh ""
foreach var of global shimer_vars {
	
	gen l_`var' = l.`var'	
	local lab: variable label `var'
	label variable l_`var' "`lab' lag"
	
	eststo tab`var': estpost correlate `var' l_`var' 
	local tables_sh = "`tables_sh' tab`var'"
}
esttab `tables_sh' using "${tables}/autocorr_sh.tex", unstack not compress replace label nonumbers booktabs nostar noobs

drop l_*

* Correlations 

* 1951 - 2019
estpost correlate $vars, matrix 
esttab . using "${tables}/corr.tex", unstack not compress replace label nonumbers booktabs nostar noobs

* Shimer data
estpost correlate $shimer_vars, matrix listwise
esttab . using "${tables}/corr_sh.tex", unstack not compress replace label nonumbers booktabs nostar noobs

* Reg wages on ALP
reg lwages_hp lalp_hp 
