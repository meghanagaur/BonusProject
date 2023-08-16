global dir = "/Users/meghanagaur/BonusProject/codes/lm-moments"
cd $dir 

clear all

* Load the ALP data
use "data/freddata_alp", clear

* Merge in tightness and wage data 
merge 1:1 yq using "data/lm_quarterly", keepusing (*wages* *tightness* *urate* *unemp*) nogen

* Merge in TFP Fernald data 
merge 1:1 yq using "data/fernald_tfp", keepusing (ltfp*) nogen

* Merge in the Shimer data 
merge 1:1 yq using "data/shimer", keepusing (ltightness* lalp* lurate*) nogen

* Restrict the sample period
global start_year = 1951
global end_year   = 2019

keep if (year(daten) <= $end_year) & (year(daten) >= $start_year)
	
tsset yq 

* Variable labels 
foreach var of varlist ltightness* {
	label var `var' "$\theta$"
}

foreach var of varlist lalp* {
	label var `var' "ALP"
}

foreach var of varlist ltfp ltfp_sa* ltfp_hp* {
	label var `var' "TFP"
}

foreach var of varlist ltfp_unadj* {
	label var `var' "TFP (unadj.)"
}

foreach var of varlist lurate* {
	label var `var' "$ u_t $"
}

* First, standard deviation of TPF/ALP
estpost tabstat lalp_hp ltfp_hp ltfp_unadj_hp lurate_hp, statistics(sd) 
esttab . using "tables/std_hp100000.tex", replace label nostar booktabs not cells("lalp_hp(fmt(a3)) ltfp_hp(fmt(a3)) ltfp_unadj_hp(fmt(a3))  lurate_hp(fmt(a3))") noobs substitute("sd" "S.D." "lalp_hp" "ALP" "ltfp_hp" "TFP" "ltfp_unadj_hp" "TFP (unadj.)" "lurate_hp" "$ u_t $") nonumber compress 

estpost tabstat lalp_hp1600 ltfp_hp1600 ltfp_unadj_hp1600 lurate_hp1600, statistics(sd)  
esttab . using "tables/std_hp1600.tex", replace label nostar booktabs not cells("lalp_hp1600(fmt(a3)) ltfp_hp1600(fmt(a3)) ltfp_unadj_hp1600(fmt(a3))  lurate_hp1600(fmt(a3))") noobs substitute("sd" "S.D." "lalp_hp1600" "ALP" "ltfp_hp1600" "TFP" "ltfp_unadj_hp1600" "TFP (unadj.)" "lurate_hp1600" "$ u_t $") nonumber compress 

* Summarize and produce correlations
pwcorr ltightness_hp1600 ltfp_sa_hp1600

estpost correlate ltightness_hp1600 ltfp_hp1600 ltfp_unadj_hp1600 lalp_hp1600, matrix listwise 
esttab . using "tables/ltightness_lprod_hp1600_corr.tex", unstack not compress replace label nonumbers booktabs

pwcorr ltightness_hp ltfp_sa_hp

estpost correlate ltightness_hp ltfp_hp ltfp_unadj_hp lalp_hp, matrix listwise
esttab . using "tables/ltightness_lprod_hp100000_corr.tex", unstack not compress replace label nonumbers booktabs
	
local prod = "alp tfp tfp_unadj" 
foreach var of local prod {
	eststo hp100000_`var': reg ltightness_hp l`var'_hp, robust
	eststo hp1600_`var':   reg ltightness_hp1600 l`var'_hp1600, robust
}

esttab hp1600_tfp hp1600_tfp_unadj hp1600_alp hp100000_tfp hp100000_tfp_unadj hp100000_alp using "tables/regs_prod_tightness.tex", label se  mgroups("$\lambda = 1600$" "$\lambda = 10^5", pattern( 1 0 0 1 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) drop(_cons) ///
	s(r2 N, label("R^2" "Observations")) replace booktabs substitute( "Standard errors in parentheses" "") star(* 0.10 ** 0.05 *** 0.01) nonotes mlabels( "TFP" "TFP" "ALP" "TFP" "TFP" "ALP", notitles lhs("$\log \theta $")) rename(lalp_hp alp ltfp_hp tfp lalp_hp1600 alp ltfp_hp1600 tfp ltfp_unadj_hp1600 tfp_unadj ltfp_unadj_hp tfp_unadj) coeflabels(alp "$\log$ ALP" tfp "$\log$ TFP"tfp_unadj "$\log$ TFP (unadj.)")  order(tfp tfp_unadj alp) nonumbers
 
**** Using Shimer HWOL/ALP data  ****
*keep if ltightness_hp_sh != . 

local prod = "alp tfp tfp_unadj" 

foreach var of local prod {
	
	*drop l`var'_hp l`var'_hp1600
	*tsfilter hp l`var'_hp     = l`var', smooth(100000)
	*tsfilter hp l`var'_hp1600 = l`var', smooth(1600)
	
	eststo hp100000_`var'_sh: reg ltightness_hp_sh l`var'_hp, robust
	eststo hp1600_`var'_sh:   reg ltightness_hp1600_sh l`var'_hp1600, robust
}

* use the original ALP data from Shimer 
eststo hp1600_alpsh_sh:   reg ltightness_hp1600_sh lalp_hp1600_sh, robust
eststo hp100000_alpsh_sh: reg ltightness_hp_sh lalp_hp_sh, robust

esttab hp1600_tfp_sh hp1600_tfp_unadj_sh hp1600_alp_sh hp1600_alpsh_sh hp100000_tfp_sh hp100000_tfp_unadj_sh hp100000_alp_sh hp100000_alpsh_sh using "tables/regs_prod_tightness_sh.tex", label se  mgroups( "$\lambda = 1600$" "$\lambda = 10^5", pattern( 1 0 0 0 1 0 0 0 0) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) drop(_cons) ///
	s(r2 N, label("R^2" "Observations")) replace booktabs substitute( "Standard errors in parentheses" "") star(* 0.10 ** 0.05 *** 0.01) nonotes mlabels( "TFP" "TFP" "ALP" "ALP" "TFP" "TFP" "ALP" "ALP", notitles lhs("$\log \theta $")) rename(lalp_hp alp ltfp_hp tfp ltfp_unadj_hp tfp_unadj lalp_hp1600 alp ltfp_hp1600 tfp ltfp_unadj_hp1600 tfp_unadj lalp_hp1600_sh alp_sh lalp_hp_sh alp_sh) coeflabels(alp "$\log$ ALP" alp_sh "$\log$ ALP (Shimer)" tfp "$\log$ TFP" tfp_unadj "$\log$ TFP (unadj.)")  order(tfp tfp_unadj alp alp_sh) nonumbers
  
* Summarize and produce correlations from Shimer data
label var lalp_hp1600_sh "ALP (Shimer)"
label var lalp_hp_sh "ALP (Shimer)"

estpost correlate ltightness_hp1600_sh ltfp_hp1600 ltfp_unadj_hp1600 lalp_hp1600 lalp_hp1600_sh, matrix listwise
esttab . using "tables/ltightness_lprod__hp1600_corr_sh.tex", unstack not compress replace label nonumbers booktabs

estpost correlate ltightness_hp_sh ltfp_hp ltfp_unadj_hp lalp_hp lalp_hp_sh, matrix listwise
esttab . using "tables/ltightness_lprod_hp100000_corr_sh.tex", unstack not compress replace label nonumbers booktabs

	