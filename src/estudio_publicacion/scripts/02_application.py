"""ENDES district application, fixed comparisons and untouched province test.

Run with the bundled Python and single-threaded BLAS. Input microdata are never
modified. Only domain-level summaries leave the data-preparation component.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key,"1")
import argparse, json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import multivariate_normal, norm
from models import fit_model, spatial_basis, prepare_observations, bootstrap_intervals, METHODS

ROOT=Path(__file__).resolve().parents[1]
FEATURES=["altitude","poverty","illiteracy","electricity","spanish","biomass"]
SEED=20261005

def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda x: x.tolist() if isinstance(x,np.ndarray) else float(x) if isinstance(x,np.floating) else int(x) if isinstance(x,np.integer) else str(x)))

def dataset(year,rule):
    path=ROOT/f"datos/processed/areas_{year}_6_59.csv"
    auxpath=ROOT/"datos/processed/aux_all_districts.csv"
    areas=pd.read_csv(path,dtype={"ubigeo":str,"prov_code":str,"dept_code":str})
    aux=pd.read_csv(auxpath,dtype={"ubigeo":str,"prov_code":str,"dept_code":str})
    # Keep one common census population for all four models. New or ambiguous
    # codes with no verified district covariates are retained as unsupported
    # direct-only domains in a separate appendix, never assigned invented X.
    good=np.all(np.isfinite(aux[FEATURES+["latitude","longitude"]]),axis=1)
    excluded_aux=aux.loc[~good].copy()
    aux=aux.loc[good].reset_index(drop=True)
    columns=[c for c in areas.columns if c not in aux.columns or c=="ubigeo"]
    allareas=aux.merge(areas[columns],on="ubigeo",how="left",validate="one_to_one")
    unsupported=areas.loc[~areas.ubigeo.isin(aux.ubigeo)].copy()
    xraw=allareas[FEATURES].to_numpy(float)
    center=xraw.mean(axis=0);scale=xraw.std(axis=0)
    X=np.c_[np.ones(len(xraw)),(xraw-center)/scale]
    a="anemia_old" if rule=="old" else "anemia_who2024"
    yc=[f"y_{a}","y_stunting"]
    yall=allareas[yc].to_numpy(float)
    idx=np.flatnonzero(np.any(np.isfinite(yall),axis=1))
    observed=allareas.iloc[idx]
    y=yall[idx]
    D=np.zeros((len(idx),2,2))
    D[:,0,0]=observed[f"var_{a}"]
    D[:,1,1]=observed.var_stunting
    D[:,0,1]=D[:,1,0]=observed["cov_old_stunting" if rule=="old" else "cov_who_stunting"]
    ne=observed[[f"n_eff_{a}","n_eff_stunting"]].to_numpy(float)
    # Within-domain PSU minus represented strata is a stability proxy; these
    # are not the degrees of freedom of the full-design Taylor variance.
    df=np.column_stack([np.maximum(observed[f"n_psu_{a}"].to_numpy(float)-observed[f"n_strata_{a}"].to_numpy(float),1),
                        np.maximum(observed.n_psu_stunting.to_numpy(float)-observed.n_strata_stunting.to_numpy(float),1)])
    manifest={"year":year,"rule":rule,"age_months":[6,59],"census_areas":len(aux),"sampled_areas":len(idx),
              "partially_observed_areas":int(np.sum(np.any(np.isfinite(y),axis=1)&~np.all(np.isfinite(y),axis=1))),
              "unsupported_survey_codes":unsupported.ubigeo.tolist(),"excluded_auxiliary_rows":excluded_aux.ubigeo.tolist(),
              "features":FEATURES,"X_center":center.tolist(),"X_scale":scale.tolist(),
              "design_condition_number":float(np.linalg.cond(X)),
              "X_scaling":"all known census covariates, no response-dependent screening",
              "df_proxy":"max(outcome-specific domain PSU minus represented domain strata,1); not an exact chi-square survey df",
              "auxiliary_source":"2017 census compilation supplied by authors; original extraction provenance still requires author verification",
              "geography_scope":"common 2017 district covariates; new/ambiguous unsplit domains are excluded from model fits",
              "sampling_D":"Taylor design covariances from full survey design; supplied domain summaries preserve marginal measurement availability",
              "input_sha256":{str(path):hashlib.sha256(path.read_bytes()).hexdigest(),str(auxpath):hashlib.sha256(auxpath.read_bytes()).hexdigest()}}
    return allareas,X,y,D,ne,idx,df,unsupported,manifest

def province_folds(provinces,nfold=5):
    rng=np.random.default_rng(SEED)
    groups=np.unique(provinces)
    shuffled=rng.permutation(groups)
    # Greedy assignment balances sampled district count. Geography and counts
    # alone define folds, with no outcomes used in the locked assignment.
    weights={g:int(np.sum(provinces==g)) for g in groups}
    tie={g:i for i,g in enumerate(shuffled)}
    ordered=sorted(groups,key=lambda g:(-weights[g],tie[g]))
    size=np.zeros(nfold,int);assign={}
    for g in ordered:
        f=int(np.argmin(size));assign[g]=f;size[f]+=weights[g]
    return np.array([assign[g] for g in provinces]),assign

def score_fit(fit,y,D,ne,hold_idx,df,label,fold,year,rule):
    pred=fit.predict()
    # Only training-fitted GVF parameters enter held-out regularization. Direct
    # y enters its own Jeffreys transform and delta variance at scoring time.
    ob=prepare_observations(y,D,ne,transform="logit",D_mode="smooth_biascorrected",df=df,pooling=fit.preparation)
    z,ds=ob["z"],ob["D"]
    mu=pred["eta_mean"][hold_idx]
    cov=pred["eta_cov"][hold_idx]+ds
    rows=[]
    for j,name in enumerate(("anemia","stunting")):
        ok=np.isfinite(y[:,j])
        err=pred["p_mean"][hold_idx,j][ok]-y[ok,j]
        sigma=np.sqrt(cov[ok,j,j])
        ll=norm.logpdf(z[ok,j],mu[ok,j],sigma)
        covered=np.abs(z[ok,j]-mu[ok,j])<=norm.ppf(.975)*sigma
        rows.append({"year":year,"rule":rule,"validation":label,"fold":fold,"method":fit.method,
                     "outcome":name,"n":int(ok.sum()),"rmse_noisy_direct":float(np.sqrt(np.mean(err**2))),
                     "mae_noisy_direct":float(np.mean(abs(err))),"mean_negative_log_score_logit":float(np.mean(-ll)),
                     "predictive_coverage95_noisy_direct":float(np.mean(covered)),
                     "mean_predictive_width_logit":float(np.mean(2*norm.ppf(.975)*sigma)),
                     "converged":fit.status["success"]})
    ok=np.all(np.isfinite(z),axis=1)
    nll=[]
    nll_area=np.full(len(y),np.nan)
    for i in np.flatnonzero(ok):
        value=-multivariate_normal.logpdf(z[i],mean=mu[i],cov=cov[i])
        nll.append(value);nll_area[i]=value
    rows.append({"year":year,"rule":rule,"validation":label,"fold":fold,"method":fit.method,"outcome":"joint",
                 "n":int(ok.sum()),"mean_negative_log_score_logit":float(np.mean(nll)),"converged":fit.status["success"]})
    detail=pd.DataFrame({"all_area_index":hold_idx,"fold":fold,"validation":label,"method":fit.method,
                         "direct_anemia":y[:,0],"direct_stunting":y[:,1],
                         "pred_anemia":pred["p_mean"][hold_idx,0],"pred_stunting":pred["p_mean"][hold_idx,1]})
    detail["negative_log_score_joint_logit"]=nll_area
    for j,name in enumerate(("anemia","stunting")):
        detail[f"negative_log_score_{name}_logit"]=-norm.logpdf(z[:,j],mu[:,j],np.sqrt(cov[:,j,j]))
        detail[f"predictive_sd_{name}_logit"]=np.sqrt(cov[:,j,j])
    return rows,detail

def locked_test_comparison(allareas,X,y,D,ne,idx,df,basis,out,year,rule):
    """Paired province-bootstrap uncertainty for prespecified test contrasts.

    It compares methods on exactly the same held-out noisy direct estimates;
    its intervals concern comparative predictive scores, not prevalence.
    """
    folds=pd.read_csv(out/"validation_folds.csv",dtype={"ubigeo":str,"province":str})
    f=folds.set_index("ubigeo").loc[allareas.iloc[idx].ubigeo]
    test=np.flatnonzero(f.locked_test.to_numpy())
    tr=np.flatnonzero(~f.locked_test.to_numpy())
    details=[]
    for method in METHODS:
        fit=fit_model(X,y[tr],D[tr],ne[tr],idx[tr],basis,method,D_mode="smooth_biascorrected",df=df[tr],maxiter=160,multistart=True)
        _,detail=score_fit(fit,y[test],D[test],ne[test],idx[test],df[test],"province_locked_test",0,year,rule)
        detail["ubigeo"]=allareas.iloc[idx[test]].ubigeo.to_numpy()
        detail["province"]=allareas.iloc[idx[test]].prov_code.to_numpy()
        details.append(detail)
    dd=pd.concat(details,ignore_index=True)
    dd.to_csv(out/"locked_test_area_scores.csv",index=False)
    ref=dd[dd.method=="FH"].set_index("ubigeo")
    provinces=ref.province.to_numpy();groups=np.unique(provinces)
    rng=np.random.default_rng(SEED)
    draws=[rng.choice(groups,size=len(groups),replace=True) for _ in range(2000)]
    comparisons=[]
    for method in METHODS[1:]:
        cur=dd[dd.method==method].set_index("ubigeo").loc[ref.index]
        delta=cur.negative_log_score_joint_logit.to_numpy()-ref.negative_log_score_joint_logit.to_numpy()
        results=[]
        for draw in draws:
            values=np.concatenate([delta[provinces==g] for g in draw])
            results.append(np.nanmean(values))
        comparisons.append({"method":method,"reference":"FH","mean_joint_NLL_difference":float(np.nanmean(delta)),
                            "province_cluster_bootstrap_lo95":float(np.quantile(results,.025)),
                            "province_cluster_bootstrap_hi95":float(np.quantile(results,.975)),
                            "B":2000,"test_provinces":len(groups),"test_districts":len(ref),
                            "direction":"negative difference favours method; noisy-direct predictive score only"})
    write_json(out/"locked_test_paired_score_comparison.json",comparisons)

def validate(allareas,X,y,D,ne,idx,df,basis,out,year,rule):
    provinces=allareas.iloc[idx].prov_code.astype(str).to_numpy()
    blocks,assignment=province_folds(provinces)
    development=np.flatnonzero(blocks!=0);test=np.flatnonzero(blocks==0)
    rng=np.random.default_rng(SEED+1)
    rassign=np.empty(len(development),int)
    rassign[rng.permutation(len(development))]=np.arange(len(development))%5
    foldtable=pd.DataFrame({"ubigeo":allareas.iloc[idx].ubigeo.to_numpy(),"province":provinces,"province_fold":blocks,"locked_test":blocks==0})
    foldtable.to_csv(out/"validation_folds.csv",index=False)
    write_json(out/"validation_protocol.json",{"seed":SEED,"province_assignment":assignment,
        "locked_test":"province fold zero; excluded from all development fits and selection",
        "development":"four province folds and five disjoint random folds within remaining 80%",
        "selection":"smallest area-weighted joint negative Gaussian log score on development province folds; failed fits ineligible",
        "test":"each prespecified model trained on all development areas and evaluated once on locked provinces",
        "sampling_D_mode":"smooth_biascorrected; D regularization fitted anew inside every training subset",
        "scoring":"latent predictive covariance plus held-out sampling D; noisy direct is not latent truth"})
    schedules=[]
    for f in range(1,5):
        ho=np.flatnonzero(blocks==f);tr=np.flatnonzero((blocks!=f)&(blocks!=0))
        schedules.append(("province_development",f,tr,ho))
    for f in range(5):
        ho=development[rassign==f];tr=development[rassign!=f]
        schedules.append(("random_development",f,tr,ho))
    rows=[];details=[];status=[]
    def run(label,fold,tr,ho):
        for method in METHODS:
            t=time.time()
            fit=fit_model(X,y[tr],D[tr],ne[tr],idx[tr],basis,method,D_mode="smooth_biascorrected",df=df[tr],maxiter=160,multistart=True)
            status.append({"validation":label,"fold":fold,**fit.status,"seconds":time.time()-t})
            rr,detail=score_fit(fit,y[ho],D[ho],ne[ho],idx[ho],df[ho],label,fold,year,rule)
            rows.extend(rr);details.append(detail)
            print(f"{year}/{rule} {label} fold{fold} {method} success={fit.status['success']} sec={time.time()-t:.1f}",flush=True)
            pd.DataFrame(rows).to_csv(out/"validation_metrics.csv",index=False)
            write_json(out/"validation_fit_status.json",status)
    for schedule in schedules:
        run(*schedule)
    dev=pd.DataFrame(rows)
    criterion=dev[(dev.validation=="province_development")&(dev.outcome=="joint")]
    scores={}
    for method in METHODS:
        sub=criterion[criterion.method==method]
        if len(sub)==4 and sub.converged.all():
            scores[method]=float(np.average(sub.mean_negative_log_score_logit,weights=sub.n))
    if not scores:
        raise RuntimeError("No model converged in all development province folds")
    selected=min(scores,key=scores.get)
    write_json(out/"selected_model_before_test.json",{"method":selected,"criterion_scores":scores,"test_used_for_selection":False})
    run("province_locked_test",0,development,test)
    predall=pd.concat(details,ignore_index=True)
    predall["ubigeo"]=allareas.ubigeo.to_numpy()[predall.all_area_index]
    predall.to_csv(out/"validation_predictions.csv",index=False)
    return selected

def priority_probabilities(pred,thresholds=(.30,.15)):
    mu,cov=pred["eta_mean"],pred["eta_cov"]
    sd=np.sqrt(np.stack((cov[:,0,0],cov[:,1,1]),axis=1))
    cut=logit(np.array(thresholds))
    marginal=norm.sf((cut-mu)/sd)
    joint=[]
    rng=np.random.default_rng(SEED)
    for i in range(len(mu)):
        bottom=multivariate_normal.cdf(cut,mean=mu[i],cov=cov[i],maxpts=20000,abseps=1e-7,releps=1e-7,rng=rng)
        joint.append(np.clip(marginal[i].sum()-1+bottom,0,1))
    return marginal,np.array(joint)

def fullfit(allareas,X,y,D,ne,idx,df,basis,out,selected,B,rule="who2024"):
    fits={};rows=[];base=allareas.copy()
    base["sampled_any"]=False;base.loc[idx,"sampled_any"]=True
    outcome_name="anemia_old" if rule=="old" else "anemia_who2024"
    minimum_ne=base[["n_eff_"+outcome_name,"n_eff_stunting"]].min(axis=1)
    minimum_psu=base[["n_psu_"+outcome_name,"n_psu_stunting"]].min(axis=1)
    base["data_support_unsampled"]=~base.sampled_any
    base["data_support_at_most_one_measured_PSU"]=base.sampled_any&(minimum_psu<=1)
    base["data_support_min_Kish_neff_at_most5"]=base.sampled_any&(minimum_ne<=5)
    base["data_support_min_Kish_neff_at_most10"]=base.sampled_any&(minimum_ne<=10)
    base["decision_probability_scope"]="experimental model-conditional prevalence-threshold probability; not validated policy priority or individual co-occurrence"
    for method in METHODS:
        t=time.time()
        fit=fit_model(X,y,D,ne,idx,basis,method,D_mode="smooth_biascorrected",df=df,maxiter=200,multistart=True)
        fits[method]=fit
        pred=fit.predict();marginal,joint=priority_probabilities(pred)
        tab=base.copy()
        for j,name in enumerate(("anemia","stunting")):
            tab[f"p_{name}"]=pred["p_mean"][:,j]
            tab[f"p_{name}_lo95_plugin"]=pred["p_lo"][:,j]
            tab[f"p_{name}_hi95_plugin"]=pred["p_hi"][:,j]
            tab[f"eta_{name}_sd_plugin"]=np.sqrt(pred["eta_cov"][:,j,j])
            tab[f"prob_exceed_{name}_plugin"]=marginal[:,j]
        tab["prob_both_area_marginal_thresholds_plugin"]=joint
        tab["method"]=method
        tab.to_csv(out/f"district_predictions_{method}.csv",index=False)
        np.savez_compressed(out/f"fit_{method}.npz",params=fit.params,beta=fit.beta,beta_cov=fit.beta_cov,A=fit.A,S=fit.S if fit.S is not None else np.zeros((2,2)),sample_idx=idx,D_logit=fit.D,
                            eta_mean=pred["eta_mean"],eta_cov=pred["eta_cov"])
        preparation_metadata=dict(fit.preparation)
        preparation_metadata["df_source"]="outcome-specific domain PSU-minus-represented-strata stability proxy, lower bounded at 1; not full-survey residual degrees of freedom"
        rows.append({"method":method,"fit_status":fit.status,"preparation":preparation_metadata,"seconds":time.time()-t})
        print(f"full {method} success={fit.status['success']} sec={time.time()-t:.1f}",flush=True)
    write_json(out/"full_fit_status.json",rows)
    if not fits[selected].status["success"]:
        raise RuntimeError("Selected full-data model failed optimizer checks")
    selected_csv=out/f"district_predictions_{selected}.csv"
    table=pd.read_csv(selected_csv,dtype={"ubigeo":str,"prov_code":str,"dept_code":str})
    if B:
        write_json(out/"bootstrap_in_progress.json",{"B_requested":B,"method":selected,"status":"running"})
        boot=bootstrap_intervals(fits[selected],ne,B=B,seed=SEED,df=df,resample_D=False,maxiter=140)
        for j,name in enumerate(("anemia","stunting")):
            table[f"p_{name}_lo95_bootstrap_conditional_D"]=boot["p_lo"][:,j]
            table[f"p_{name}_hi95_bootstrap_conditional_D"]=boot["p_hi"][:,j]
            table[f"eta_{name}_bootstrap_MSE"]=boot["mse_eta"][:,j]
        small={k:v for k,v in boot.items() if k not in ("p_lo","p_hi","eta_lo","eta_hi","mse_eta","bias_eta")}
        write_json(out/"bootstrap_status.json",small)
        (out/"bootstrap_in_progress.json").unlink(missing_ok=True)
    table.to_csv(out/"district_predictions_SELECTED.csv",index=False)
    sensitivity=[]
    pbase=fits[selected].predict()["p_mean"]
    for mode,strength,rank in [("raw",20,40),("smooth",20,40),("smooth_biascorrected",5,40),("smooth_biascorrected",50,40),("smooth_biascorrected",20,20),("smooth_biascorrected",20,80)]:
        bs=basis if rank==40 or selected in ("FH","MFH") else spatial_basis(allareas[["latitude","longitude"]].to_numpy(float),rank=rank)
        fit=fit_model(X,y,D,ne,idx,bs,selected,D_mode=mode,df=df,prior_df=strength,maxiter=200,multistart=True)
        pp=fit.predict()["p_mean"]
        sensitivity.append({"D_mode":mode,"prior_df":strength,"spatial_rank":rank if selected in ("SFH","MSFH") else 0,
                            "fit_status":fit.status,"mean_abs_change_anemia":float(np.mean(abs(pp[:,0]-pbase[:,0]))),
                            "mean_abs_change_stunting":float(np.mean(abs(pp[:,1]-pbase[:,1]))),
                            "max_abs_change":float(np.max(abs(pp-pbase)))})
        pd.DataFrame({"ubigeo":allareas.ubigeo,"p_anemia":pp[:,0],"p_stunting":pp[:,1]}).to_csv(out/f"sensitivity_{mode}_df{strength}_q{rank}.csv",index=False)
    write_json(out/"sensitivity_summary.json",sensitivity)
    return fits

def modelled_D_uncertainty_sensitivity(allareas,X,y,D,ne,idx,df,basis,out,B=199):
    """A distributional stress test, explicitly not a survey bootstrap.

    Wishart perturbations use the domain stability df proxy around the fitted
    transformed D. The perturbed matrices enter refits as regularized raw D,
    rather than refitting the original probability-scale GVF. Thus this is a
    modelled sampling-covariance/estimation sensitivity, not isolated exact
    propagation of the original estimator's sampling-variance uncertainty.
    """
    selected=json.loads((out/"selected_model_before_test.json").read_text())["method"]
    fit=fit_model(X,y,D,ne,idx,basis,selected,D_mode="smooth_biascorrected",df=df,maxiter=200,multistart=True)
    write_json(out/"modelled_D_sensitivity_in_progress.json",{"B":B,"method":selected,"status":"running"})
    boot=bootstrap_intervals(fit,ne,B=B,seed=SEED+50,df=df,resample_D=True,maxiter=160)
    pred=fit.predict()
    table=pd.DataFrame({"ubigeo":allareas.ubigeo,"sampled_any":np.isin(np.arange(len(allareas)),idx)})
    for j,name in enumerate(("anemia","stunting")):
        table[f"p_{name}_lo95_modelled_D_sensitivity"]=boot["p_lo"][:,j]
        table[f"p_{name}_hi95_modelled_D_sensitivity"]=boot["p_hi"][:,j]
        table[f"eta_{name}_MSE_modelled_D_sensitivity"]=boot["mse_eta"][:,j]
    table.to_csv(out/"modelled_D_uncertainty_sensitivity.csv",index=False)
    meta={k:v for k,v in boot.items() if k not in ("p_lo","p_hi","eta_lo","eta_hi","mse_eta","bias_eta")}
    meta["method"]=selected
    meta["scope"]="modelled covariance uncertainty sensitivity using domain df proxy, not a design bootstrap or validated nominal coverage"
    meta["refit_sampling_D"]="regularized raw perturbed transformed D; original probability-scale GVF is not resmoothed"
    write_json(out/"modelled_D_uncertainty_sensitivity_status.json",meta)
    (out/"modelled_D_sensitivity_in_progress.json").unlink(missing_ok=True)

def summarize_existing():
    """Recreate publication appendices from completed district-level outputs."""
    r=ROOT/"resultados";support={};decision={};qa=[]
    for year in (2019,2025):
        p=r/f"aplicacion_{year}_who2024"
        if not (p/"selected_model_before_test.json").exists():
            continue
        method=json.loads((p/"selected_model_before_test.json").read_text())["method"]
        f=p/"district_predictions_SELECTED.csv"
        if not f.exists():
            f=p/f"district_predictions_{method}.csv"
        d=pd.read_csv(f,dtype={"ubigeo":str})
        sampled=d.sampled_any.to_numpy()
        ne=d[["n_eff_anemia_who2024","n_eff_stunting"]].min(axis=1).to_numpy()
        psu=d[["n_psu_anemia_who2024","n_psu_stunting"]].min(axis=1).to_numpy()
        high=d.prob_both_area_marginal_thresholds_plugin.to_numpy()>=.8
        s={"year":year,"districts":len(d),"sampled_districts":int(sampled.sum()),"unsampled_districts":int((~sampled).sum()),
           "sampled_minKish_neff_median":float(np.nanmedian(ne[sampled])),"sampled_minKish_neff_le5":int(np.sum(sampled&(ne<=5))),
           "sampled_minKish_neff_le10":int(np.sum(sampled&(ne<=10))),"sampled_at_most_one_measurementPSU":int(np.sum(sampled&(psu<=1))),
           "conditional_plugin_joint_probability_ge80_total":int(high.sum()),"conditional_plugin_joint_probability_ge80_unsampled":int(np.sum(high&~sampled)),
           "conditional_plugin_joint_probability_ge80_sampled_psu_le1":int(np.sum(high&sampled&(psu<=1))),
           "conditional_plugin_joint_probability_ge80_sampled_neff_le5":int(np.sum(high&sampled&(ne<=5))),
           "interpretation":"model-conditional experimental probabilities, not validated coverage or policy priority recommendations"}
        write_json(p/"application_data_support_summary.json",s);support[str(year)]=s
        oldpath=r/f"aplicacion_{year}_old/district_predictions_SELECTED.csv"
        if oldpath.exists():
            old=pd.read_csv(oldpath,dtype={"ubigeo":str}).set_index("ubigeo").loc[d.ubigeo].reset_index()
            points=pd.DataFrame({"ubigeo":d.ubigeo,"department":d.department,"province":d.province,"district":d.district,
                "p_anemia_who2024":d.p_anemia,"p_anemia_old":old.p_anemia,"delta_p_anemia_who_minus_old":d.p_anemia-old.p_anemia,
                "p_stunting_who_jointfit":d.p_stunting,"p_stunting_old_jointfit":old.p_stunting,
                "prob_both_thresholds_who_plugin":d.prob_both_area_marginal_thresholds_plugin,
                "prob_both_thresholds_old_plugin":old.prob_both_area_marginal_thresholds_plugin,
                "point_anemia_threshold30_who":d.p_anemia>.30,"point_anemia_threshold30_old":old.p_anemia>.30})
            points.to_csv(p/"paired_rule_model_point_comparison.csv",index=False)
            q={"year":year,"model_fixed":method+", selected in principal development data before reserved test", "districts":len(d),
               "mean_model_anemia_point_difference":float(points.delta_p_anemia_who_minus_old.mean()),
               "min_model_anemia_point_difference":float(points.delta_p_anemia_who_minus_old.min()),"max_model_anemia_point_difference":float(points.delta_p_anemia_who_minus_old.max()),
               "districts_above_30_who":int(points.point_anemia_threshold30_who.sum()),"districts_above_30_old":int(points.point_anemia_threshold30_old.sum()),
               "point_classification_switch":int((points.point_anemia_threshold30_who!=points.point_anemia_threshold30_old).sum()),
               "joint_threshold_plugin_probability_ge80_who":int((points.prob_both_thresholds_who_plugin>=.8).sum()),
               "joint_threshold_plugin_probability_ge80_old":int((points.prob_both_thresholds_old_plugin>=.8).sum()),
               "limitations":"unweighted district point comparisons; no paired posterior difference probability, no individual co-occurrence, no population temporal change",
               "sensitivity_point_decisions":[]}
            main=(d.p_anemia.to_numpy()>.30)&(d.p_stunting.to_numpy()>.15)
            for path in sorted(p.glob("sensitivity_*.csv")):
                alt=pd.read_csv(path,dtype={"ubigeo":str}).set_index("ubigeo").loc[d.ubigeo]
                cls=(alt.p_anemia.to_numpy()>.30)&(alt.p_stunting.to_numpy()>.15)
                q["sensitivity_point_decisions"].append({"file":path.name,"n_point_double_threshold_main":int(main.sum()),
                    "n_point_double_threshold_alternative":int(cls.sum()),"point_classification_disagreement":int(np.sum(main!=cls)),
                    "point_threshold_definition":"anemia prevalence>.30 AND stunting prevalence>.15"})
            write_json(p/"paired_rule_and_decision_summary.json",q);decision[str(year)]=q
        for rule in ("who2024","old"):
            pp=r/f"aplicacion_{year}_{rule}";path=pp/"district_predictions_SELECTED.csv"
            if not path.exists():
                continue
            tab=pd.read_csv(path,dtype={"ubigeo":str});rec={"year":year,"rule":rule,"file":path.name,"n":len(tab),"unique_ubigeo":tab.ubigeo.nunique()}
            for name in ("anemia","stunting"):
                arr=tab[["p_"+name,"p_"+name+"_lo95_plugin","p_"+name+"_hi95_plugin"]].to_numpy()
                rec[name+"_finite"]=bool(np.isfinite(arr).all());rec[name+"_range01"]=bool(np.all((arr>=0)&(arr<=1)))
                rec[name+"_interval_order"]=bool(np.all(arr[:,1]<=arr[:,2]));rec[name+"_positive_width_count"]=int(np.sum(arr[:,2]>arr[:,1]))
                lo="p_"+name+"_lo95_bootstrap_conditional_D";hi="p_"+name+"_hi95_bootstrap_conditional_D"
                if lo in tab:
                    rec[name+"_bootstrap_interval_order"]=bool(np.all(tab[lo]<tab[hi]))
            rec["valid_joint_probability"]=bool(np.all((tab.prob_both_area_marginal_thresholds_plugin>=0)&(tab.prob_both_area_marginal_thresholds_plugin<=1)))
            rec["optimizer_success"]=all(x["fit_status"]["success"] for x in json.loads((pp/"full_fit_status.json").read_text()))
            if (pp/"bootstrap_status.json").exists():
                rec["bootstrap"]=json.loads((pp/"bootstrap_status.json").read_text())["B_success"]
            qa.append(rec)
    write_json(r/"application_data_support_summary.json",support)
    write_json(r/"application_rule_decision_summary.json",decision)
    write_json(r/"application_output_validation.json",qa)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--year",type=int,default=2025)
    ap.add_argument("--rule",choices=["old","who2024"],default="who2024")
    ap.add_argument("--bootstrap",type=int,default=100)
    ap.add_argument("--skip-cv",action="store_true")
    ap.add_argument("--resume-bootstrap",action="store_true")
    ap.add_argument("--D-uncertainty-sensitivity",type=int,default=0)
    ap.add_argument("--summarize-existing",action="store_true")
    ap.add_argument("--selected",choices=METHODS,default="MSFH")
    args=ap.parse_args()
    if args.summarize_existing:
        summarize_existing()
        return
    out=ROOT/f"resultados/aplicacion_{args.year}_{args.rule}"
    out.mkdir(parents=True,exist_ok=True)
    aa,X,y,D,ne,idx,df,unsupported,manifest=dataset(args.year,args.rule)
    np.savez_compressed(out/"model_input_values_used.npz",Xall=X,y=y,D=D,n_eff=ne,sample_idx=idx,df_proxy=df)
    aa.to_csv(out/"model_input_all_districts_used.csv",index=False)
    write_json(out/"input_and_method_manifest.json",manifest)
    unsupported.to_csv(out/"unsupported_direct_only_domains.csv",index=False)
    basis=spatial_basis(aa[["latitude","longitude"]].to_numpy(float),rank=40)
    np.save(out/"fixed_graph_spectral_basis_q40.npy",basis)
    if args.D_uncertainty_sensitivity:
        modelled_D_uncertainty_sensitivity(aa,X,y,D,ne,idx,df,basis,out,args.D_uncertainty_sensitivity)
        return
    if args.resume_bootstrap and args.skip_cv:
        raise ValueError("Use either a principal bootstrap resume or a secondary skip-CV run")
    if args.skip_cv:
        write_json(out/"prespecified_secondary_model.json",{"method":args.selected,
            "reason":"secondary anemia-rule analysis uses the model selected in the WHO2024 principal analysis of the same year, no new response-based model selection",
            "CV_performed":False})
    if args.resume_bootstrap:
        selected=json.loads((out/"selected_model_before_test.json").read_text())["method"]
        write_json(out/"bootstrap_resume.json",{"method":selected,"CV_unchanged":True,"selection_unchanged":True,
            "reason":"add the prespecified full conditional-D predictive-error bootstrap after the completed, frozen validation"})
    else:
        selected=args.selected if args.skip_cv else validate(aa,X,y,D,ne,idx,df,basis,out,args.year,args.rule)
    if not args.skip_cv and not args.resume_bootstrap:
        locked_test_comparison(aa,X,y,D,ne,idx,df,basis,out,args.year,args.rule)
    fullfit(aa,X,y,D,ne,idx,df,basis,out,selected,args.bootstrap,args.rule)
    write_json(out/"completed.json",{"year":args.year,"rule":args.rule,"selected":selected,"bootstrap_requested":args.bootstrap,
         "claim_scope":"district marginal prevalences under specified Gaussian area model, census geography and sampling-variance approximation; neither individual cooccurrence nor guaranteed calibration"})

if __name__=="__main__":
    main()
