#!/usr/bin/env python3
"""Development benchmark using saved common samples, with frozen fit sources.

Final seed IDs (>=10000) require a separate, explicit protocol freeze. This
runner never overwrites the validated phase-one simulation results.
"""
from pathlib import Path
import argparse
import concurrent.futures
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
import numpy as np
import pandas as pd
from benchmark_inputs import ROOT, SIM, OUT, SCENARIOS, PRIMARY_SCENARIOS, export_replication, sha256, add_coordinate_fit_basis
from benchmark_metrics import replication_metrics, summarize

METHODS=("Direct","DirectWilsonKish","DirectWilsonCluster","FH","MFH","SFH","MSFH",
         "BetaPowerSpatial","BetaPowerSpatialKish","BetaPowerIndependent","BetaPowerHMCSpatial","BetaPowerHMCGraph")
HMC_METHODS=("BetaPowerHMCSpatial","BetaPowerHMCGraph")


def load(name,path):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def run_replication(job):
    scenario,rep,tag,methods,basis_kind=job
    folder=OUT/"inputs"/scenario/f"rep_{rep:05d}"
    if not folder.exists():export_replication(scenario,rep)
    fitdir=folder/"model_inputs";direct=np.load(fitdir/"direct.npz")
    areas=pd.read_csv(fitdir/"area.csv");units=pd.read_csv(fitdir/"sample.csv")
    X=np.c_[np.ones(len(areas)),areas[["X1","X2"]]]
    if basis_kind=="graph":
        if not (fitdir/"fit_basis.npy").exists():add_coordinate_fit_basis(folder)
        basis=np.load(fitdir/"fit_basis.npy")
    else:
        basis=np.load(fitdir/"spatial_basis.npy")
    model=load("benchmark_frozen_area_models",OUT/"source_snapshots"/tag/"models.py")
    stable=load("benchmark_frozen_beta_power_stable",OUT/"source_snapshots"/tag/"beta_power_stable.py")
    sys.modules["beta_power_stable"]=stable
    beta=load("benchmark_frozen_beta_power",OUT/"source_snapshots"/tag/"beta_power.py")
    # HMC imports the selected likelihood/ICC implementation by this name.
    # Resolve it to the run snapshot rather than any live candidate source.
    sys.modules["beta_power"]=beta
    os.environ["ENDES_STUDY_ROOT"]=str(ROOT)
    helpers=load("benchmark_prediction_helpers",SIM/"source_snapshots"/"finite_v2"/"03_simulation.py")
    rows=[];failures=[]
    for method in methods:
        tick=time.perf_counter();destination=OUT/tag/"fit_results"/method/scenario/f"rep_{rep:05d}"
        destination.mkdir(parents=True,exist_ok=True)
        try:
            if method.startswith("Direct"):
                p=np.full((len(areas),2),np.nan);p[direct["sample_idx"]]=direct["y"]
                se=np.sqrt(np.maximum(np.diagonal(direct["D"],axis1=1,axis2=2),0))
                lower=p.copy();upper=p.copy();lower[direct["sample_idx"]]=np.maximum(direct["y"]-1.959963984540054*se,0)
                upper[direct["sample_idx"]]=np.minimum(direct["y"]+1.959963984540054*se,1)
                predictions=dict(mean=p,lower=lower,upper=upper);status=dict(success=True,interval="unadjusted direct Wald")
                if method!="Direct":
                    ne=direct["n_eff"].copy();icc_status=[]
                    if method=="DirectWilsonCluster":
                        for t,name in enumerate(("anemia","stunting")):
                            icc,info=beta.working_icc(units,name)
                            ne[:,t],_=beta.cluster_effective_n(units,name,direct["sample_idx"],ne[:,t],icc)
                            icc_status.append(info)
                    z=1.959963984540054;n=np.maximum(ne,1e-12);y=direct["y"]
                    centre=(y+z*z/(2*n))/(1+z*z/n)
                    half=z*np.sqrt(y*(1-y)/n+z*z/(4*n*n))/(1+z*z/n)
                    lower[direct["sample_idx"]]=np.maximum(centre-half,0)
                    upper[direct["sample_idx"]]=np.minimum(centre+half,1)
                    status=dict(success=True,interval="Wilson score with working effective n; not exact survey coverage",ICC=icc_status)
            elif method in ("FH","MFH","SFH","MSFH"):
                fitted=model.fit_model(X,direct["y"],direct["D"],direct["n_eff"],direct["sample_idx"],
                                        basis,method,transform="logit",D_mode="smooth_biascorrected",
                                        df=direct["df_primary"],maxiter=120)
                p,lower,upper,_=helpers.prediction_values(fitted.predict())
                predictions=dict(mean=p,lower=lower,upper=upper);status=fitted.status
            else:
                seed=int(np.random.SeedSequence([20261005,8117,SCENARIOS.index(scenario),rep]).generate_state(1)[0])
                candidate=load("benchmark_frozen_beta_power_hmc",OUT/"source_snapshots"/tag/"beta_power_hmc.py") if method in HMC_METHODS else beta
                fitted=candidate.fit(X,direct["y"],direct["n_eff"],direct["sample_idx"],units,
                                basis if method in ("BetaPowerSpatial","BetaPowerSpatialKish",*HMC_METHODS) else None,seed=seed,
                                config={"cluster_adjustment":False} if method=="BetaPowerSpatialKish" else None)
                predictions=dict(mean=fitted["p_mean"],lower=fitted["lower95"],upper=fitted["upper95"])
                status=dict(success=all(x["success"] for x in fitted["status"]),endpoints=fitted["status"])
            elapsed=time.perf_counter()-tick
            np.savez_compressed(destination/"predictions.npz",**predictions)
            pd.DataFrame({"area":np.arange(len(areas)),**{
                f"{label}_{name}":predictions[key][:,t]
                for t,name in enumerate(("anemia","stunting"))
                for label,key in (("p_mean","mean"),("lo95","lower"),("hi95","upper"))}}).to_csv(destination/"predictions.csv",index=False)
            status["fit_seconds"]=elapsed;(destination/"status.json").write_text(json.dumps(status,indent=2))
            endpoint_status={name:status["endpoints"][j]["success"] for j,name in enumerate(("anemia","stunting"))} if "endpoints" in status else status["success"]
            metrics=replication_metrics(folder,predictions,method,"common_info_areaX_primary_df_"+basis_kind,endpoint_status,elapsed)
            for row in metrics:row.update(scenario=scenario,replication=rep)
            rows.extend(metrics)
            if not status["success"]:
                failures.append(dict(scenario=scenario,replication=rep,method=method,type="nonconverged_finite_predictions_retained",status=status))
        except Exception as exc:
            failures.append(dict(scenario=scenario,replication=rep,method=method,type="exception_no_fallback",message=repr(exc),traceback=traceback.format_exc()))
    return rows,failures


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tag",required=True)
    parser.add_argument("--replications",nargs="+",type=int,default=[0])
    parser.add_argument("--scenarios",nargs="+",choices=SCENARIOS,default=list(PRIMARY_SCENARIOS))
    parser.add_argument("--methods",nargs="+",choices=METHODS,
                        default=["Direct","DirectWilsonKish","DirectWilsonCluster","FH","MFH","SFH","MSFH","BetaPowerHMCGraph"])
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--worker-limit-file",type=Path,
                        help="Runtime-only JSON {active_workers:n}; limits submissions without changing the frozen estimator")
    parser.add_argument("--basis-kind",choices=("graph","oracle"),default="graph")
    args=parser.parse_args()
    if "BetaPowerHMCGraph" in args.methods and args.basis_kind!="graph":raise ValueError("Graph-labelled candidate requires coordinate-only graph basis")
    if "BetaPowerHMCSpatial" in args.methods and args.basis_kind!="oracle":raise ValueError("Historical HMCSpatial is oracle diagnostic; select HMCGraph for new evaluation")
    if any(rep>=10000 for rep in args.replications):
        if args.basis_kind!="graph":raise RuntimeError("Oracle generating covariance is forbidden in independent evaluation")
        freeze=OUT/"protocol_freeze.json"
        if not freeze.exists() or not json.loads(freeze.read_text()).get("approved_for_independent_evaluation"):
            raise RuntimeError("Independent final seeds require approved protocol_freeze.json")
    destination=OUT/args.tag
    if destination.exists():raise FileExistsError("A run tag already exists; use a fresh version")
    destination.mkdir(parents=True)
    snapshot=OUT/"source_snapshots"/args.tag;snapshot.mkdir(parents=True)
    sources={"models.py":SIM/"source_snapshots"/"sparse_v2"/"models.py",
             "beta_power.py":Path(__file__).with_name("beta_power.py"),
             "beta_power_stable.py":Path(__file__).with_name("beta_power_stable.py"),
             "benchmark_run.py":Path(__file__),
             "benchmark_inputs.py":Path(__file__).with_name("benchmark_inputs.py"),
             "benchmark_structural.py":Path(__file__).with_name("benchmark_structural.py"),
             "benchmark_metrics.py":Path(__file__).with_name("benchmark_metrics.py"),
             "benchmark_collect.py":Path(__file__).with_name("benchmark_collect.py"),
             "benchmark_compare.py":Path(__file__).with_name("benchmark_compare.py"),
             "benchmark_primary_score.py":Path(__file__).with_name("benchmark_primary_score.py"),
             "benchmark_validate_inputs.py":Path(__file__).with_name("benchmark_validate_inputs.py")}
    if any(m in HMC_METHODS for m in args.methods):
        sources.update({name:Path(__file__).with_name(name) for name in ("beta_power_hmc.py","beta_power_hmc.stan","beta_power_vectorized.stan","beta_power_stable_functions.stan","beta_power_hybrid.stan")})
    if any(rep>=10000 for rep in args.replications):
        frozen=json.loads((OUT/"protocol_freeze.json").read_text())
        for scenario in args.scenarios:
            if scenario not in frozen.get("independent_seed_ids",{}):
                raise RuntimeError(f"Scenario {scenario} is not in the approved independent protocol")
            if not set(args.replications).issubset(frozen["independent_seed_ids"][scenario]):
                raise RuntimeError(f"Unplanned independent seed IDs for {scenario}")
        if not set(args.methods).issubset(frozen.get("methods",[])):
            raise RuntimeError("A method outside the approved independent protocol was requested")
        expected=frozen.get("source_sha256",{})
        for name,source in sources.items():
            if name not in expected or sha256(source)!=expected[name]:
                raise RuntimeError(f"Independent evaluation source {name} differs from approved freeze")
    for name,source in sources.items():shutil.copy2(source,snapshot/name)
    if any(m in HMC_METHODS for m in args.methods):
        # Precompile once before workers to avoid concurrent writes to Stan cache.
        # A binary copied from the same checked Stan source avoids a redundant build.
        for name in ("beta_power_hmc.stan","beta_power_vectorized.stan","beta_power_hybrid.stan"):
            executable=Path(__file__).with_name(Path(name).stem)
            latest_source=max(sources[name].stat().st_mtime,sources["beta_power_stable_functions.stan"].stat().st_mtime)
            if executable.exists() and executable.stat().st_mtime>=latest_source:
                shutil.copy2(executable,snapshot/executable.name)
        sys.modules["beta_power_stable"]=load("benchmark_frozen_beta_power_stable",snapshot/"beta_power_stable.py")
        sys.modules["beta_power"]=load("benchmark_frozen_beta_power",snapshot/"beta_power.py")
        os.environ["ENDES_STUDY_ROOT"]=str(ROOT)
        hmc=load("benchmark_frozen_beta_power_hmc",snapshot/"beta_power_hmc.py")
        compiled=hmc.get_model()
    config=dict(tag=args.tag,phase="independent_evaluation" if min(args.replications)>=10000 else "development_existing_populations",
                replications=args.replications,scenarios=args.scenarios,methods=args.methods,
                source_sha256={name:sha256(source) for name,source in sources.items()},
                final_performance_not_yet_claimed=True,finite_support="archived common support",
                SRS_df="max(outcome-specific measured n-1,1)",PSU_df="archived application proxy",
                information="area X equal; cluster working effective-n candidate uses observed training-unit PSU information",
                finite_nonconvergence_predictions_retained=True,basis_kind=args.basis_kind,
                worker_limit_file=str(args.worker_limit_file) if args.worker_limit_file else None,
                generating_covariance_available_to_estimator=args.basis_kind=="oracle",
                prediction_helper_sha256=sha256(SIM/"source_snapshots"/"finite_v2"/"03_simulation.py"))
    if any(m in HMC_METHODS for m in args.methods):
        config["HMC_sampler"]={**hmc.HMC_CONFIG}
        config["compiled_HMC_executable_sha256"]=sha256(Path(compiled.exe_file))
    (destination/"config.json").write_text(json.dumps(config,indent=2))
    tick=time.perf_counter();jobs=[(s,r,args.tag,args.methods,args.basis_kind) for s in args.scenarios for r in args.replications]
    with (destination/"replications.jsonl").open("w") as rf,(destination/"failures.jsonl").open("w") as ff:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            iterator=iter(jobs);pending=set();exhausted=False;completed=0;last_limit=None
            while pending or not exhausted:
                limit=args.workers
                if args.worker_limit_file:
                    limit=int(json.loads(args.worker_limit_file.read_text())["active_workers"])
                    if not 1<=limit<=args.workers:raise ValueError("Runtime worker limit outside1..workers")
                if limit!=last_limit:
                    event=dict(active_worker_limit=limit,elapsed_seconds=time.perf_counter()-tick)
                    with (destination/"runtime_parallelism.jsonl").open("a") as pl:pl.write(json.dumps(event)+"\n")
                    last_limit=limit
                while len(pending)<limit and not exhausted:
                    job=next(iterator,None)
                    if job is None:exhausted=True;break
                    pending.add(pool.submit(run_replication,job))
                if not pending:break
                ready,pending=concurrent.futures.wait(pending,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in ready:
                    records,failures=future.result();completed+=1
                    for row in records:rf.write(json.dumps(row)+"\n")
                    for failure in failures:ff.write(json.dumps(failure)+"\n")
                    rf.flush();ff.flush()
                    print(json.dumps(dict(completed=completed,total=len(jobs),elapsed_seconds=round(time.perf_counter()-tick,1),recent_failures=len(failures))),flush=True)
    records=pd.read_json(destination/"replications.jsonl",lines=True)
    records.to_csv(destination/"replications.csv",index=False);summarize(records).to_csv(destination/"summary.csv",index=False)
    (destination/"completion.json").write_text(json.dumps(dict(completed=True,populations=len(jobs),
        rows=len(records),elapsed_seconds=time.perf_counter()-tick,failures=sum(1 for _ in (destination/"failures.jsonl").open())),indent=2))


if __name__=="__main__":main()
