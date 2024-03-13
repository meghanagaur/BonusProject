/*******************************************************************************
Clean the unemployment/employment, vacancy (JOLTS), and wage data. 
*******************************************************************************/
clear all
set more off 
macro drop _all

* Set local directory
cd "/Users/meghanagaur/BonusProject/lm-moments/"
global base       = c(pwd)
global download   = 1
global data 	  = "${base}/data"
global tables 	  = "${base}/tables"

* Formatting 
global fmt fmt(%12.3f) 
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

* Merge in the HM data 
merge 1:1 yq using "${data}/clean/hm", keepusing (*hm*)
tab year if _merge != 3
drop _merge 
		
tsset yq 

* Variable labels 
foreach var of varlist *gdp* {
	label var `var' "$ y $"
}

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
	gen `var'_sh = `var'  
	label var `var'_sh "$ w $"
}

foreach var of varlist *v_*hwi* {
	label var `var' "$ v $"
}


foreach var of varlist *theta* {
	label var `var' "$ \theta $"
}

***** Standard deviation *****
global vars 	   = "lgdp_hp lalp_hp lunemp_hp lv_hwi_hp ltheta_hwi_hp lwages_hp"


* 1951- 2019
estpost tabstat $vars, statistics(sd) 
esttab . using "${tables}/std.tex", replace label nostar booktabs not cells("lgdp_hp($fmt) lalp_hp($fmt) lunemp_hp($fmt) lv_hwi_hp($fmt)  ltheta_hwi_hp($fmt) lwages_hp($fmt)") noobs substitute("sd" "S.D." "lgdp_hp" "$  y $" "lalp_hp" "$  p$" "lunemp_hp" "$ u $" "lv_hwi_hp" "$ v $" "ltheta_hwi_hp" "$ \theta $" "lwages_hp" "$ w $") nonumber compress  

* Shimer data

global shimer_vars = ""
foreach var of global vars {
	if "`var'" != "lgdp_hp" {
		global shimer_vars = "$shimer_vars `var'_sh"
		replace `var'_sh   = . if year >= 2004
	}
}	

estpost tabstat $shimer_vars, statistics(sd) 
esttab . using "${tables}/std_sh.tex", label replace nostar booktabs not cells("lalp_hp_sh($fmt) lunemp_hp_sh($fmt) lv_hwi_hp_sh($fmt)  ltheta_hwi_hp_sh($fmt) lwages_hp_sh($fmt)") noobs substitute("sd" "S.D." "lalp_hp_sh" "$  p$" "lunemp_hp_sh" "$ u $" "lv_hwi_hp_sh" "$ v $" "ltheta_hwi_hp_sh" "$ \theta $" "lwages_hp_sh" "$ w $") nonumber compress  

* HM data
global hm_vars = ""
foreach var of global vars {
	if "`var'" != "lgdp_hp" {
		global hm_vars = "$hm_vars `var'_hm"
	}
}	

estpost tabstat $hm_vars, statistics(sd) 
esttab . using "${tables}/std_hm.tex", label replace nostar booktabs not cells("lalp_hp_hm($fmt) lunemp_hp_hm($fmt) lv_hwi_hp_hm($fmt)  ltheta_hwi_hp_hm($fmt) lwages_hp_hm($fmt)") noobs substitute("sd" "S.D." "lalp_hp_hm" "$  p$" "lunemp_hp_hm" "$ u $" "lv_hwi_hp_hm" "$ v $" "ltheta_hwi_hp_hm" "$ \theta $" "lwages_hp_hm" "$ w $") nonumber compress  
  
***** Autocorrelations *****

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

* HM data
local tables_hm ""
foreach var of global hm_vars {
	
	gen l_`var' = l.`var'	
	local lab: variable label `var'
	label variable l_`var' "`lab' lag"
	
	eststo tab`var': estpost correlate `var' l_`var' 
	local hm_vars = "`hm_vars' tab`var'"
}
esttab `hm_vars' using "${tables}/autocorr_hm.tex", unstack not compress replace label nonumbers booktabs nostar noobs

drop l_*

* Correlations 

* 1951 - 2019
estpost correlate $vars, matrix  
esttab . using "${tables}/corr.tex", unstack not compress replace label nonumbers booktabs nostar noobs cells(b($fmt))  

* Shimer data
estpost correlate $shimer_vars, matrix listwise
esttab . using "${tables}/corr_sh.tex", unstack not compress replace label nonumbers booktabs nostar noobs cells(b($fmt))  

* HM data
estpost correlate $hm_vars, matrix listwise 
esttab . using "${tables}/corr_hm.tex", unstack not compress replace label nonumbers booktabs nostar noobs cells(b($fmt))  

* Reg wages on ALP
reg lwages_hp lalp_hp 
reg lwages_hp_sh lalp_hp_sh
reg lwages_hp_hm lalp_hp_hm

