"""Replication-level interval evaluation; finite population truth stays here."""
from pathlib import Path
import numpy as np
import pandas as pd

OUTCOMES = ("anemia", "stunting")
ALPHA = .05


def interval_score(truth, lower, upper, alpha=ALPHA):
    truth, lower, upper = map(np.asarray, (truth, lower, upper))
    return upper-lower + 2/alpha * np.maximum(lower-truth,0) + 2/alpha * np.maximum(truth-upper,0)


def replication_metrics(input_folder, predictions, method, protocol, converged, elapsed_seconds):
    folder=Path(input_folder)
    truth=np.load(folder/"evaluation_only"/"truth.npz")
    direct=np.load(folder/"model_inputs"/"direct.npz")
    theta=truth["theta"];idx=truth["sample_idx"]
    p=np.asarray(predictions["mean"]);lo=np.asarray(predictions["lower"]);hi=np.asarray(predictions["upper"])
    if p.shape!=theta.shape or lo.shape!=theta.shape or hi.shape!=theta.shape:
        raise ValueError("Predictions must contain both outcomes in the 80-area prediction universe")
    if np.any(np.isfinite(lo)&np.isfinite(hi)&(lo>hi)):
        raise ValueError("Reversed interval")
    sampled=np.isin(np.arange(len(theta)),idx)
    observed=np.zeros_like(theta,dtype=bool);observed[idx]=np.isfinite(direct["y"])
    rows=[]
    for t,outcome in enumerate(OUTCOMES):
        available=np.isfinite(p[:,t])&np.isfinite(lo[:,t])&np.isfinite(hi[:,t])
        if np.any(available&((p[:,t]<0)|(p[:,t]>1)|(lo[:,t]<0)|(hi[:,t]>1))):
            raise ValueError("Proportion prediction outside [0,1]")
        groups={"all":np.ones(len(theta),bool),"sampled_common":sampled,
                "observed_endpoint":observed[:,t],"unsampled_common":~sampled}
        for group,mask in groups.items():
            # A direct estimate has no prediction for unavailable endpoints.
            # Its valid common comparison is observed_endpoint only.
            if method.startswith("Direct") and group!="observed_endpoint":continue
            target=mask&available
            if not target.any():continue
            e=p[target,t]-theta[target,t];scores=interval_score(theta[target,t],lo[target,t],hi[target,t])
            rows.append(dict(method=method,protocol=protocol,group=group,outcome=outcome,
                bias=float(e.mean()),mse=float(np.mean(e**2)),mae=float(np.mean(abs(e))),
                coverage=float(np.mean((lo[target,t]<=theta[target,t])&(theta[target,t]<=hi[target,t]))),
                width=float(np.mean(hi[target,t]-lo[target,t])),interval_score=float(scores.mean()),
                areas=int(target.sum()),areas_requested=int(mask.sum()),
                availability_rate=float(target.sum()/mask.sum()),
                converged=bool(converged[outcome] if isinstance(converged,dict) else converged),
                fit_seconds=float(elapsed_seconds)))
    return rows


def summarize(records):
    data=pd.DataFrame(records);rows=[]
    keys=["scenario","method","protocol","group","outcome"]
    for key,p in data.groupby(keys,dropna=False):
        R=len(p)
        row=dict(zip(keys,key),replications=R,convergence_rate=float(p.converged.mean()),
                 mean_areas=float(p.areas.mean()),mean_fit_seconds=float(p.fit_seconds.mean()))
        for column in ("areas_requested","availability_rate"):
            if column in p:row["mean_"+column]=float(p[column].mean())
        mse=p.mse.mean();row["rmse"]=float(np.sqrt(mse))
        row["rmse_mcse"]=float(p.mse.std(ddof=1)/np.sqrt(R)/(2*np.sqrt(mse))) if R>1 and mse>0 else np.nan
        for metric in ("bias","mae","coverage","width","interval_score"):
            row[metric]=float(p[metric].mean());row[metric+"_mcse"]=float(p[metric].std(ddof=1)/np.sqrt(R)) if R>1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_contrasts(candidate,reference,label):
    keys=["scenario","replication","group","outcome"]
    if candidate.duplicated(keys).any() or reference.duplicated(keys).any():
        raise ValueError("Select one method/protocol before forming paired contrasts")
    paired=candidate.merge(reference,on=keys,suffixes=("_candidate","_reference"),validate="one_to_one")
    rows=[]
    for key,p in paired.groupby(["scenario","group","outcome"]):
        R=len(p);row=dict(zip(["scenario","group","outcome"],key),contrast=label,paired_replications=R)
        M1=p.mse_candidate.mean();M0=p.mse_reference.mean()
        row["rmse_relative_change_pct"]=float(100*(np.sqrt(M1/M0)-1))
        influence=.5*(p.mse_candidate/M1-p.mse_reference/M0)
        row["log_rmse_ratio_mcse"]=float(influence.std(ddof=1)/np.sqrt(R)) if R>1 else np.nan
        for metric in ("bias","coverage","width","interval_score"):
            delta=p[metric+"_candidate"]-p[metric+"_reference"]
            row[metric+"_change"]=float(delta.mean());row[metric+"_change_mcse"]=float(delta.std(ddof=1)/np.sqrt(R)) if R>1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
