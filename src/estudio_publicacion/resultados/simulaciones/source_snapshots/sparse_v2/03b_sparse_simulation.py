#!/usr/bin/env python3
"""ENDES-sized finite populations: SRS boundary and single-PSU stress.

Domain sample counts/PSU counts come from aggregate 2025 data, never individual
records. Binary finite-population truths are generated independently. The PSU
scenario deliberately reproduces the undefined single-PSU first-stage variance
as zero; it is a stress test, not a claim of unbiased full-design variance.
"""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import traceback
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np
import pandas as pd
from scipy.special import expit, ndtr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"resultados"/"simulaciones"
DATA=ROOT/"datos"/"processed"/"areas_2025_6_59.csv"
spec=importlib.util.spec_from_file_location("sparse_helpers",Path(__file__).with_name("03_simulation.py"))
sim=importlib.util.module_from_spec(spec);spec.loader.exec_module(sim)
SCENARIOS=("sparse_srs_2025","sparse_psu_2025")
SEED=2026100517


def generate(rep,scenario):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,SCENARIOS.index(scenario),rep]))
    X,coords,basis=sim.geometry()
    m=len(X)
    empirical=pd.read_csv(os.environ.get("ENDES_SPARSE_TEMPLATE",str(DATA)))
    rows=empirical.iloc[rng.integers(0,len(empirical),m)]
    k=rows.n_eligible.to_numpy(int)
    target_ns=rows[["n_anemia_who2024","n_stunting"]].to_numpy(float)
    response_probability=np.clip(target_ns/k[:,None],0,1)
    planned_psu=rows.n_psu.to_numpy(int)
    proxy_df=np.maximum((rows.n_psu-rows.n_strata).to_numpy(float),1.)
    eta=sim.latent_truth(rng,X,basis,dict(rho=.6,spatial=.5))
    truth=[];ys=[];Ds=[];neffs=[];idx=[];oracle_complete=[]
    counts=[];observed_psus=[];zero_psu_areas=partial=empty=0
    sample_latent_rho=.4
    chol=np.linalg.cholesky(np.array([[1.,sample_latent_rho],[sample_latent_rho,1.]]))
    for area in range(m):
        if scenario=="sparse_srs_2025":
            N=max(1000,2*int(k[area]))
            person_prob=np.repeat(expit(eta[area])[None,:],N,axis=0)
            outcomes=(ndtr(rng.normal(size=(N,2))@chol.T)<person_prob).astype(float)
            truth.append(outcomes.mean(axis=0))
            take=rng.choice(N,int(k[area]),replace=False)
            values=outcomes[take]
            observed=rng.uniform(size=values.shape)<response_probability[area]
            n=observed.sum(axis=0)
            y=np.divide((values*observed).sum(axis=0),n,
                        out=np.full(2,np.nan),where=n>0)
            D=np.zeros((2,2))
            for t in range(2):
                if n[t]>1:
                    D[t,t]=(1-n[t]/N)*np.var(values[observed[:,t],t],ddof=1)/n[t]
            both=observed.all(axis=1)
            if np.all(n>0) and both.sum()>1:
                # Conditional observation patterns are fixed subsets of a common
                # SRS. Their overlap identifies the first-order covariance.
                D[0,1]=D[1,0]=(1-k[area]/N)*np.sum(
                    (values[both,0]-y[0])*(values[both,1]-y[1]))/(n[0]*n[1])
            ev,U=np.linalg.eigh(D)
            D=(U*np.maximum(ev,0))@U.T
            neff=n.astype(float)
            measured_psu=np.where(n>0,planned_psu[area],0)
            oracle_complete.append((1-k[area]/N)*np.cov(outcomes.T,ddof=1)/k[area])
        else:
            G=max(100,int(planned_psu.max())+10)
            J=int(planned_psu[area])
            L=max(25,math.ceil(k[area]/J))
            psu_effect=(rng.normal(size=(G,2))@np.linalg.cholesky(
                np.array([[1.,.6],[.6,1.]])).T)*np.array([.9,.8])
            psu_probability=expit(eta[area][None,:]+psu_effect)
            latent=rng.normal(size=(G,L,2))@chol.T
            outcomes=(ndtr(latent)<psu_probability[:,None,:]).astype(float)
            truth.append(outcomes.mean(axis=(0,1)))
            selected=rng.choice(G,J,replace=False)
            allocations=np.full(J,int(k[area])//J,int)
            allocations[:int(k[area])%J]+=1
            rng.shuffle(allocations)
            numerator=np.zeros((J,2));denominator=np.zeros((J,2))
            unit_weights=[];unit_outcomes=[];unit_observed=[]
            for j,(psu,nj) in enumerate(zip(selected,allocations)):
                children=rng.choice(L,int(nj),replace=False)
                values=outcomes[psu,children]
                observed=rng.uniform(size=values.shape)<response_probability[area]
                w=np.where(observed,(L/nj)/np.maximum(response_probability[area],1e-10),0.)
                numerator[j]=(w*values).sum(axis=0)
                denominator[j]=w.sum(axis=0)
                unit_weights.append(w);unit_outcomes.append(values);unit_observed.append(observed)
            w=np.concatenate(unit_weights);seen=np.concatenate(unit_observed)
            n=seen.sum(axis=0)
            den=denominator.sum(axis=0)
            y=np.divide(numerator.sum(axis=0),den,out=np.full(2,np.nan),where=den>0)
            residual=numerator-np.nan_to_num(y)*denominator
            D=np.zeros((2,2))
            if J>1:
                D=(1-J/G)*J/(J-1)*(residual.T@residual)/np.maximum(den[:,None]*den[None,:],1e-20)
            else:
                zero_psu_areas+=1
            neff=np.divide(den**2,(w*w).sum(axis=0),out=np.zeros(2),where=(w*w).sum(axis=0)>0)
            measured_psu=(denominator>0).sum(axis=0)
            # Exact complete-response design covariance for the balanced two-
            # stage equal-PSU mean. The first term describes selected PSUs;
            # the second is the expected within-PSU sampling contribution.
            psu_means=outcomes.mean(axis=1)
            first=(1-J/G)*np.cov(psu_means.T,ddof=1)/J
            within=np.mean([np.cov(group.T,ddof=1) for group in outcomes],axis=0)
            second=within*np.sum((1-allocations/L)/allocations)/J**2
            oracle_complete.append(first+second)
        counts.append(n);observed_psus.append(measured_psu)
        if not np.any(n>0):
            empty+=1
            continue
        partial+=int(np.any(n==0))
        D[n==0,:]=0;D[:,n==0]=0
        ys.append(y);Ds.append(D);neffs.append(neff);idx.append(area)
    truth=np.asarray(truth)
    # Independently mark 20% of entire districts as unsampled, while preserving
    # all available endpoints in the remaining districts, including NA pairs.
    selected_areas=set(rng.choice(m,int(.8*m),replace=False).tolist())
    keep=np.array([i in selected_areas for i in idx])
    idx=np.asarray(idx)[keep];y=np.asarray(ys)[keep];D=np.asarray(Ds)[keep]
    ne=np.asarray(neffs)[keep]
    counts=np.asarray(counts);observed_psus=np.asarray(observed_psus)
    return dict(X=X,coords=coords,basis=basis,theta=truth,eta=np.log(
        np.clip(truth,1e-10,1-1e-10)/np.clip(1-truth,1e-10,1)),
        y=y,D=D,n_eff=ne,sample_idx=idx,df=proxy_df[idx],
        oracle_complete_D=np.asarray(oracle_complete),
        metadata=dict(scenario=scenario,replication=rep,seed=[SEED,SCENARIOS.index(scenario),rep],
            observed_areas=len(idx),missing_endpoint_areas=partial,empty_areas=empty,
            zero_first_stage_psu_areas=zero_psu_areas,
            empirical_n_median=float(np.median(k)),empirical_psu_median=float(np.median(planned_psu)),
            planned_one_psu_fraction=float(np.mean(planned_psu==1)),
            realised_n_anemia_median=float(np.median(counts[idx,0])),
            realised_n_stunting_median=float(np.median(counts[idx,1])),
            boundary_anemia=float(np.mean((y[:,0]==0)|(y[:,0]==1))),
            boundary_stunting=float(np.mean((y[:,1]==0)|(y[:,1]==1))),
            zero_variance_anemia=float(np.mean(D[:,0,0]<=1e-12)),
            zero_variance_stunting=float(np.mean(D[:,1,1]<=1e-12)),
            singular_fraction=float(np.mean(np.linalg.det(D)<=1e-12)),
            covariance_estimator="SRS first-order covariance" if scenario=="sparse_srs_2025" else
                "Ultimate-cluster linearization; m=1 undefined first-stage variance stored as zero; within-stage contribution omitted",
            proxy_df="max(empirical PSU minus strata,1); not exact chi-square degrees of freedom"))


def run(job):
    scenario,rep,maxiter=job
    data=generate(rep,scenario)
    models=sim.simulation_models()
    rows=[];failures=[];statuses=[]
    folder=OUT/"inputs_sparse_v2"/scenario;folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/f"rep_{rep:04d}.npz",theta=data["theta"],y=data["y"],D=data["D"],
                        n_eff=data["n_eff"],sample_idx=data["sample_idx"],df=data["df"],
                        oracle_complete_D=data["oracle_complete_D"])
    for mode in ("raw","smooth_biascorrected"):
        for method in sim.METHODS:
            tick=time.perf_counter()
            try:
                fit=models.fit_model(data["X"],data["y"],data["D"],data["n_eff"],
                    data["sample_idx"],data["basis"],method,transform="logit",D_mode=mode,
                    df=data["df"],maxiter=maxiter)
                pred=fit.predict()
                if not np.isfinite(pred["eta_mean"]).all() or not np.isfinite(pred["eta_cov"]).all():
                    raise ValueError("Nonfinite predictions")
                rows.extend(sim.replication_metrics(data,pred,method,mode,scenario,rep,
                            time.perf_counter()-tick,fit.status["success"]))
                statuses.append(dict(D_mode=mode,**fit.status))
                if not fit.status["success"]:
                    failures.append(dict(scenario=scenario,replication=rep,method=method,D_mode=mode,
                        status="optimization_not_converged",message=fit.status["message"]))
            except Exception as exc:
                failures.append(dict(scenario=scenario,replication=rep,method=method,D_mode=mode,
                    status="exception",message=repr(exc),traceback=traceback.format_exc()))
    data["metadata"]["fit_statuses"]=statuses
    return rows,failures,data["metadata"]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--reps",type=int,default=500)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--maxiter",type=int,default=120)
    parser.add_argument("--tag",default="sparse_v2")
    parser.add_argument("--after-full",action="store_true")
    args=parser.parse_args()
    if args.after_full:
        while not (OUT/"completion_full_v1.json").exists():
            time.sleep(5)
    snapshot=OUT/"source_snapshots"/args.tag
    snapshot.mkdir(parents=True,exist_ok=True)
    model=snapshot/"models.py"
    if model.exists():
        raise FileExistsError(f"Duplicate experiment tag {args.tag}")
    for source in (Path(__file__),Path(__file__).with_name("03_simulation.py"),ROOT/"scripts"/"models.py",DATA):
        shutil.copy2(source,snapshot/source.name)
    os.environ["ENDES_SIM_MODEL_PATH"]=str(model)
    os.environ["ENDES_SPARSE_TEMPLATE"]=str(snapshot/DATA.name)
    empirical=pd.read_csv(DATA)
    config=dict(seed=SEED,scenarios=SCENARIOS,replications_each=args.reps,
        methods=sim.METHODS,D_modes=["raw","smooth_biascorrected"],m=80,q=12,
        models_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        aggregate_template_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        empirical_districts=len(empirical),
        empirical_n_median=float(empirical.n_eligible.median()),
        empirical_one_psu_fraction=float((empirical.n_psu==1).mean()),
        empirical_zero_variance_anemia=float((empirical.var_anemia_who2024<=1e-12).mean()),
        empirical_zero_variance_stunting=float((empirical.var_stunting<=1e-12).mean()),
        empirical_singular_fraction=float(empirical.D_who_singular.mean()),
        response="Independent MAR with probabilities equal to observed marginal counts/eligible counts; known weights",
        estimand="Realized finite-population district prevalence, not nominal probability",
        oracle_complete_D="Exact complete-response design covariance; not exact under simulated MAR",
        limitation="PSU stress ultimate-cluster variance omits within-stage term and stores undefined m=1 as zero; no design-unbiasedness claim",
        args=vars(args))
    (OUT/f"config_{args.tag}.json").write_text(json.dumps(config,indent=2))
    jobs=[(scenario,rep,args.maxiter) for scenario in SCENARIOS for rep in range(args.reps)]
    tick=time.perf_counter()
    rowpath=OUT/f"replications_{args.tag}.jsonl"
    if rowpath.exists():raise FileExistsError(rowpath)
    with (rowpath.open("w") as rf,
          (OUT/f"failures_{args.tag}.jsonl").open("w") as ff,
          (OUT/f"metadata_{args.tag}.jsonl").open("w") as mf):
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            for done,(rows,failures,meta) in enumerate(executor.map(run,jobs),1):
                for row in rows:rf.write(json.dumps(row)+"\n")
                for fail in failures:ff.write(json.dumps(fail)+"\n")
                mf.write(json.dumps(meta)+"\n");rf.flush();ff.flush();mf.flush()
                if done%10==0 or done==len(jobs):
                    print(json.dumps(dict(completed=done,total=len(jobs),seconds=round(time.perf_counter()-tick,1))),flush=True)
    df=pd.read_json(rowpath,lines=True)
    df.to_csv(OUT/f"replications_{args.tag}.csv",index=False)
    sim.summarize(df).to_csv(OUT/f"summary_{args.tag}.csv",index=False)
    (OUT/f"completion_{args.tag}.json").write_text(json.dumps(dict(completed=True,
        replications=len(jobs),metric_rows=len(df),elapsed_seconds=time.perf_counter()-tick,
        failures=sum(1 for _ in (OUT/f"failures_{args.tag}.jsonl").open())),indent=2))


if __name__=="__main__":main()
