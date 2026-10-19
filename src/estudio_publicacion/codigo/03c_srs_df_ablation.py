#!/usr/bin/env python3
"""Matched SRS sensitivity: measured n-1 rather than a PSU/stratum proxy.

Uses the first 200 saved SRS populations, not 200 new populations. The usual
Normal-theory n-1 variance degrees of freedom are a working approximation for
binary observations; this script does not assume an exact chi-square law.
"""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import shutil
import time
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"resultados"/"simulaciones"
TAG="srs_df_ablation"
spec=importlib.util.spec_from_file_location("srs_ablation_helpers",Path(__file__).with_name("03_simulation.py"))
sim=importlib.util.module_from_spec(spec);spec.loader.exec_module(sim)


def run(rep):
    saved=np.load(OUT/"inputs_sparse_v2"/"sparse_srs_2025"/f"rep_{rep:04d}.npz")
    X,coords,basis=sim.geometry()
    neff=saved["n_eff"]
    data=dict(X=X,coords=coords,basis=basis,theta=saved["theta"],y=saved["y"],D=saved["D"],
              n_eff=neff,sample_idx=saved["sample_idx"])
    df=np.maximum(neff-1,1.)
    models=sim.simulation_models()
    records=[];failures=[];statuses=[]
    for method in sim.METHODS:
        start=time.perf_counter()
        try:
            fit=models.fit_model(X,data["y"],data["D"],neff,data["sample_idx"],basis,method,
                                  transform="logit",D_mode="smooth_biascorrected",df=df,maxiter=120)
            pred=fit.predict()
            if not np.isfinite(pred["eta_mean"]).all() or not np.isfinite(pred["eta_cov"]).all():
                raise ValueError("Nonfinite fitted moments")
            records.extend(sim.replication_metrics(data,pred,method,"smooth_biascorrected",
                           "sparse_srs_2025",rep,time.perf_counter()-start,fit.status["success"]))
            statuses.append(fit.status)
            if not fit.status["success"]:
                failures.append(dict(replication=rep,method=method,status="optimization_not_converged",message=fit.status["message"]))
        except Exception as exc:
            failures.append(dict(replication=rep,method=method,status="exception",message=repr(exc)))
    return records,failures,dict(replication=rep,df_alternative="max(measured n-1,1), outcome-specific",
                               missing_endpoints=int(np.isnan(data["y"]).sum()),fit_statuses=statuses)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    if args.workers<1:parser.error("workers must be positive")
    snapshot=OUT/"source_snapshots"/TAG;snapshot.mkdir(parents=True,exist_ok=True)
    model=snapshot/"models.py"
    if model.exists():raise FileExistsError("Ablation tag already exists")
    shutil.copy2(OUT/"source_snapshots"/"sparse_v2"/"models.py",model)
    shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    os.environ["ENDES_SIM_MODEL_PATH"]=str(model)
    config=dict(tag=TAG,matched_existing_populations=200,new_populations=0,
        parent_scenario="sparse_srs_2025",parent_experiment="sparse_v2",replications=list(range(200)),
        alternative_df="max(measured outcome-specific n-1,1)",
        assumption="Normal-theory approximation; binary sample variance is not exactly chi-square",
        models_source_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        modes=["smooth_biascorrected"],methods=sim.METHODS)
    (OUT/f"config_{TAG}.json").write_text(json.dumps(config,indent=2))
    tick=time.perf_counter()
    with ((OUT/f"replications_{TAG}.jsonl").open("w") as rf,
          (OUT/f"failures_{TAG}.jsonl").open("w") as ff,
          (OUT/f"metadata_{TAG}.jsonl").open("w") as mf):
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for done,(rows,failures,meta) in enumerate(pool.map(run,range(200)),1):
                for row in rows:rf.write(json.dumps(row)+"\n")
                for fail in failures:ff.write(json.dumps(fail)+"\n")
                mf.write(json.dumps(meta)+"\n");rf.flush();ff.flush();mf.flush()
                if done%10==0:print(json.dumps(dict(completed=done,total=200,seconds=round(time.perf_counter()-tick,1))),flush=True)
    records=pd.read_json(OUT/f"replications_{TAG}.jsonl",lines=True)
    records.to_csv(OUT/f"replications_{TAG}.csv",index=False)
    summary=sim.summarize(records);summary["experiment"]=TAG
    summary.to_csv(OUT/f"summary_{TAG}.csv",index=False)
    (OUT/f"completion_{TAG}.json").write_text(json.dumps(dict(completed=True,
        matched_replications=200,new_populations=0,elapsed_seconds=time.perf_counter()-tick,
        failures=sum(1 for _ in (OUT/f"failures_{TAG}.jsonl").open())),indent=2))


if __name__=="__main__":main()
