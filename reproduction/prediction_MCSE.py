"""Training-only Monte Carlo uncertainty diagnostics for the first four worlds.

Reconstructs E[p_i|hyperparameters] from retained production chains. Uses
nonoverlapping batch means, not Stan's rank-normalized ESS implementation.
No population truth, prediction accuracy score, or hypothesis test is read/run.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta as beta_distribution
import cmdstanpy
import sikov_worker as worker


def batch_mcse(values, chains, batches=20):
    """Equal-chain batch-means variance of a grand posterior mean."""
    x = np.asarray(values, float)
    if x.ndim != 2 or len(x) % chains or batches < 20:
        raise ValueError("Aligned draws x areas and >=20 batches per chain required")
    per_chain = len(x)//chains; batch_length = per_chain//batches
    if batch_length < 10:
        raise ValueError("Batch length must be >=10 draws")
    draws = x.reshape(chains, per_chain, x.shape[1])
    used = batch_length*batches
    batch = draws[:, :used].reshape(chains, batches, batch_length, x.shape[1]).mean(axis=2)
    variance_batch = batch.var(axis=1, ddof=1)
    variance_grand_mean = variance_batch.sum(axis=0)/(chains**2*batches)
    mcse = np.sqrt(np.maximum(variance_grand_mean, 0.))
    marginal_variance = draws[:, :used].reshape(chains*used, x.shape[1]).var(axis=0, ddof=1)
    ess = np.divide(marginal_variance, variance_grand_mean,
                    out=np.full(x.shape[1],np.nan),where=variance_grand_mean>0)
    # Classic split Rhat is supplementary and is not a rank-normalized gate.
    half = per_chain//2
    split = np.concatenate((draws[:, :half], draws[:, -half:]), axis=0)
    within = split.var(axis=1,ddof=1).mean(axis=0)
    between = half*split.mean(axis=1).var(axis=0,ddof=1)
    variance_hat = (half-1)/half*within + between/half
    rhat = np.sqrt(np.divide(variance_hat,within,out=np.ones_like(within),where=within>0))
    return {"mcse":mcse,"ess_batch_mean_uncapped":ess,"classic_split_Rhat":rhat,
            "chains":chains,"draws_per_chain":per_chain,"batches_per_chain":batches,
            "batch_length":batch_length,"draws_used":chains*used,
            "within_chain_batch_means":batch,"chain_means":draws.mean(axis=1)}


def conditional_arrays(ctx, context, replication, endpoint, source, folder):
    from beta_power import EndpointPosterior, working_icc, cluster_effective_n
    d=dict(np.load(source/"model_inputs/inputs.npz",allow_pickle=False));u=pd.read_csv(source/"model_inputs/sample.csv")
    j=("anemia","stunting").index(endpoint)
    icc,_=working_icc(u,endpoint)
    effective,_=cluster_effective_n(u,endpoint,d["sample_idx"],d["n_eff"][:,j],icc)
    post=EndpointPosterior(d["X"],d["y"][:,j],effective,d["sample_idx"],d["basis"])
    status=worker.read_json(folder/"ICC_status.json")["endpoints"][j]
    attempt=status["sampler_trials"][-1]["attempt"]
    chain_folder=folder/"hmc_chains"/endpoint/("attempt_"+str(attempt))
    files=sorted(chain_folder.glob("*.csv"))
    if len(files)!=4:
        raise RuntimeError("Four complete production chain CSVs required")
    fit=cmdstanpy.from_csv([str(p) for p in files])
    beta=fit.stan_variable("beta");gamma=fit.stan_variable("gamma");log_kappa=fit.stan_variable("log_kappa")
    eta=beta@post.X.T+gamma@post.B.T;kappa=np.exp(log_kappa)[:,None]
    alpha=np.maximum(kappa*expit(eta),1e-10);other=np.maximum(kappa*expit(-eta),1e-10)
    alpha[:,post.idx]+=post.k;other[:,post.idx]+=post.n-post.k
    conditional=alpha/(alpha+other)
    stored=np.load(folder/"ICC_conditional_draws.npz",allow_pickle=False)[endpoint+"_marginal"]
    if stored.shape!=conditional.shape:
        raise RuntimeError("Saved conditional draws and hyperparameter chain supports differ")
    return conditional,alpha,other,stored,status,{str(p):worker.sha(p) for p in files}


def diagnose(ctx,batches=20):
    if batches < 20:
        raise ValueError("At least 20 batches per chain required")
    run=ctx["run"];destination=run/"validacion/prediction_MCSE_first4.json"
    if destination.exists():
        raise FileExistsError("Preserve completed MCSE diagnostics")
    rows=[];endpoints=[];bound_inputs={};raw_status=[]
    for context in worker.CONTEXTS:
        rep=30000;source=run/"evaluacion/inputs"/context/f"rep_{rep:05d}"
        folder=run/"evaluacion/fits"/context/f"rep_{rep:05d}"
        completion=worker.validate_completion(folder,worker.METHODS[:9])
        if completion["truth_read_by_fits"] is not False:
            raise RuntimeError("Wrong fit role")
        reference=ctx["reference"]/"evaluacion/fits"/context/f"rep_{rep:05d}"
        actual=dict(np.load(folder/"ICC.npz",allow_pickle=False));old=dict(np.load(reference/"ICC.npz",allow_pickle=False))
        for j,endpoint in enumerate(("anemia","stunting")):
            cond,alpha,other,pdraw,status,bindings=conditional_arrays(ctx,context,rep,endpoint,source,folder)
            bound_inputs.update(bindings)
            mean=cond.mean(axis=0);mean_difference=float(np.max(abs(mean-actual["mean"][:,j])))
            if mean_difference>1e-10:
                raise RuntimeError("Reconstructed conditional means differ from stored production point predictions")
            diagnostic=batch_mcse(cond,4,batches)
            alternative=batch_mcse(cond,4,40) if len(cond)//4>=400 else None
            endpoint_summary={"context":context,"replication":rep,"endpoint":endpoint,
                "production_NUTS_passed":status["success"],"production_attempt":status["sampler_trials"][-1]["attempt"],
                "saved_mean_reconstruction_max_error":mean_difference,"chain_count":4,
                **{k:diagnostic[k] for k in ("draws_per_chain","batches_per_chain","batch_length","draws_used")},
                "maximum_conditional_mean_mcse":float(diagnostic["mcse"].max()),
                "median_conditional_mean_mcse":float(np.median(diagnostic["mcse"])),
                "minimum_derived_mean_ESS_batch":float(np.nanmin(diagnostic["ess_batch_mean_uncapped"])),
                "maximum_derived_classic_split_Rhat":float(diagnostic["classic_split_Rhat"].max()),
                "largest_legacy_mean_absolute_difference":float(np.max(abs(mean-old["mean"][:,j])))}
            quantile_details={}
            for quantile,key in ((.025,"lower"),(.975,"upper")):
                q=np.quantile(pdraw,quantile,axis=0)
                if np.max(abs(q-actual[key][:,j]))>1e-12:
                    raise RuntimeError("Stored predictive quantile cannot be reconstructed")
                indicator=(pdraw<=q[None,:]).astype(float)
                cdf=batch_mcse(indicator,4,batches)
                density=beta_distribution.pdf(q[None,:],alpha,other).mean(axis=0)
                valid=np.isfinite(density)&(density>0)&(q>0)&(q<1)
                q_mcse=np.divide(cdf["mcse"],density,out=np.full(len(q),np.nan),where=valid)
                quantile_details[key]=(q,q_mcse,cdf,density,valid)
            for i in range(cond.shape[1]):
                difference=float(mean[i]-old["mean"][i,j]);se=float(diagnostic["mcse"][i])
                cell={"context":context,"replication":rep,"endpoint":endpoint,"synthetic_area_index":i,
                    "fresh_conditional_mean":float(mean[i]),"archived_conditional_mean":float(old["mean"][i,j]),
                    "mean_difference_fresh_minus_archive":difference,"fresh_mean_MCSE_batch":se,
                    "fresh_mean_ESS_batch_uncapped":float(diagnostic["ess_batch_mean_uncapped"][i]),
                    "fresh_mean_classic_split_Rhat":float(diagnostic["classic_split_Rhat"][i]),
                    "absolute_difference_divided_by_fresh_MCSE_only":abs(difference)/se if se>0 else np.nan,
                    "sqrt2_fresh_MCSE_assumption_equal_independent_old_precision":np.sqrt(2)*se,
                    "ratio_using_sqrt2_assumption_NOT_a_test":abs(difference)/(np.sqrt(2)*se) if se>0 else np.nan,
                    "archived_MCSE_unknown":True}
                if alternative is not None:
                    cell["fresh_mean_MCSE_40_batches_sensitivity"]=float(alternative["mcse"][i])
                for key,(q,q_mcse,cdf,density,valid) in quantile_details.items():
                    cell[key+"_fresh_quantile"]=float(q[i]);cell[key+"_archived_quantile"]=float(old[key][i,j])
                    cell[key+"_quantile_difference"]=float(q[i]-old[key][i,j])
                    cell[key+"_fresh_quantile_MCSE_batch_delta"]=float(q_mcse[i])
                    cell[key+"_fresh_CDF_MCSE_batch"]=float(cdf["mcse"][i])
                    cell[key+"_posterior_mixture_density_at_quantile"]=float(density[i])
                    cell[key+"_quantile_MCSE_defined"]=bool(valid[i])
                    cell[key+"_sqrt2_current_quantile_MCSE_ASSUMPTION"]=float(np.sqrt(2)*q_mcse[i])
                    cell[key+"_ratio_sqrt2_ASSUMPTION_NOT_test"]=float(abs(q[i]-old[key][i,j])/(np.sqrt(2)*q_mcse[i])) if valid[i] and q_mcse[i]>0 else np.nan
                rows.append(cell)
            endpoints.append(endpoint_summary)
            raw_status.append({"context":context,"endpoint":endpoint,"sampler_trials":status["sampler_trials"]})
        for path in (source/"model_inputs/inputs.npz",source/"model_inputs/sample.csv",folder/"completion.json",folder/"ICC.npz",folder/"ICC_conditional_draws.npz",reference/"ICC.npz"):
            bound_inputs[str(path)]=worker.sha(path)
    frame=pd.DataFrame(rows);csv=run/"diagnostics/prediction_MCSE_first4_cells.csv";csv.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(csv,index=False)
    result={"status":"COMPLETED_TRAINING_ONLY_FIRST4_PREDICTION_MCSE_DIAGNOSTIC",
        "principal_or_accuracy_result_certificate":False,"truth_or_accuracy_scores_read":False,
        "world_count":4,"replication":30000,"endpoints":endpoints,"production_sampler_trials":raw_status,
        "cell_count":len(frame),"output_csv":str(csv),"output_csv_sha256":worker.sha(csv),
        "source_hashes_sha256":worker.source_gate(),"adapter_sha256":worker.sha(__file__),
        "input_bindings_sha256":bound_inputs,"archived_MCSE_available":False,
        "methods":{"point":"Equal-chain nonoverlapping batch-means MCSE for Rao-Blackwellized conditional probabilities",
            "ESS":"Posterior sample variance divided by estimated variance of the posterior mean; batch approximation, uncapped",
            "quantile":"Batch-means MCSE of empirical CDF indicators divided by conditional-beta mixture density at the saved quantile",
            "Rhat":"Classic split Rhat of derived conditional means; not rank-normalized and not used as a production gate"},
        "limitations":["Only four worlds and the retained finite chain lengths are assessed",
            "20 batches per chain may not capture all slow dependence; 40-batch point-MCSE sensitivity is retained",
            "Quantile MCSE is a local asymptotic delta approximation with an estimated density",
            "Old Monte Carlo uncertainty is unknown; sqrt(2) times current MCSE assumes equal independent precision",
            "Standardized differences are descriptive diagnostics, without p values or pass/fail thresholds",
            "No diagnostic replaces exact printed-cell agreement, full-cohort closure, model gates, or human review"]}
    worker.write_json(destination,result)
    print(json.dumps({"status":result["status"],"cells":len(frame),"endpoints":endpoints}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,required=True);p.add_argument("--batches",type=int,default=20)
    a=p.parse_args();cfg=json.loads(a.config.read_text());ctx=worker.initialize(cfg["source_root"],cfg["reference_root"],cfg["sikov_run_root"],cfg["beta_exe"],cfg["cmdstan"])
    diagnose(ctx,a.batches)


if __name__=="__main__":
    main()
