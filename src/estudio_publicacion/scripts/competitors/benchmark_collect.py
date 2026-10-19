#!/usr/bin/env python3
"""Evaluate externally fitted Gao/Hogg predictions without exposing truth to fits.

The task CSV supplies scenario, replication, method, protocol, output_dir and
diagnostic_style (gao/hogg). Methods/variants must have distinct labels. Existing
prediction artifacts are read; this program does not fit or repair any model.
"""
from pathlib import Path
import argparse,json,time,traceback
import numpy as np
import pandas as pd
from benchmark_inputs import OUT,SCENARIOS,sha256
from benchmark_metrics import replication_metrics,summarize,OUTCOMES


def diagnosis(folder,style):
    path=folder/("diagnostics.json" if style=="gao" else "status.json")
    d=json.loads(path.read_text())
    if style=="gao":
        endpoints=d["endpoints"]
        ok={name:bool(endpoints[name]["INLA_ok"] and endpoints[name]["working_model_converged"]) for name in OUTCOMES}
    elif style=="hogg":
        endpoints=d["outcomes"]
        ok={name:bool(endpoints[name]["success"]) for name in OUTCOMES}
    else:raise ValueError("Unknown diagnostic style")
    elapsed=float(d.get("elapsed_seconds",d.get("fit_seconds",d.get("seconds_total",0))))
    return d,ok,elapsed,path


def canonical_predictions(path):
    data=pd.read_csv(path);ids=pd.to_numeric(data.area,errors="raise").astype(int)
    if ids.duplicated().any() or set(ids)!=set(range(80)):
        raise ValueError("External predictions require each of the same80 areas exactly once")
    data=data.assign(area=ids).sort_values("area")
    return {key:data[[f"{prefix}_{name}" for name in OUTCOMES]].to_numpy(float)
            for key,prefix in (("mean","p_mean"),("lower","lo95"),("upper","hi95"))}


def main():
    p=argparse.ArgumentParser();p.add_argument("--tasks",type=Path,required=True)
    p.add_argument("--tag",required=True);args=p.parse_args()
    destination=OUT/args.tag
    if destination.exists():raise FileExistsError("Use a new immutable evaluation tag")
    destination.mkdir(parents=True);tasks=pd.read_csv(args.tasks)
    required={"scenario","replication","method","protocol","output_dir","diagnostic_style"}
    if not required.issubset(tasks):raise ValueError("Missing task columns: "+str(required-set(tasks)))
    if tasks.duplicated(["scenario","replication","method","protocol"]).any():raise ValueError("Duplicate fit tasks")
    records=[];failures=[];fits=[];tick=time.perf_counter()
    for task in tasks.to_dict("records"):
        scenario=task["scenario"];rep=int(task["replication"])
        if scenario not in SCENARIOS:raise ValueError("Unknown scenario")
        folder=OUT/"inputs"/scenario/f"rep_{rep:05d}";fitfolder=Path(task["output_dir"])
        fitrecord={**task,"replication":rep,"completed_prediction_artifact":False}
        try:
            predictions=canonical_predictions(fitfolder/"predictions.csv")
            d,ok,elapsed,diagnostic_path=diagnosis(fitfolder,task["diagnostic_style"])
            rows=replication_metrics(folder,predictions,task["method"],task["protocol"],ok,elapsed)
            for row in rows:row.update(scenario=scenario,replication=rep)
            records.extend(rows)
            fitrecord.update(completed_prediction_artifact=True,endpoint_convergence=ok,
                prediction_sha256=sha256(fitfolder/"predictions.csv"),diagnostic_sha256=sha256(diagnostic_path))
            if not all(ok.values()):failures.append({**task,"type":"nonconverged_finite_predictions_retained","diagnostics":d})
        except Exception as exc:
            failures.append({**task,"type":"exception_no_fallback","message":repr(exc),"traceback":traceback.format_exc()})
        fits.append(fitrecord)
    data=pd.DataFrame(records);data.to_csv(destination/"replications.csv",index=False)
    if len(data):summarize(data).to_csv(destination/"summary.csv",index=False)
    (destination/"fits.json").write_text(json.dumps(fits,indent=2))
    (destination/"failures.json").write_text(json.dumps(failures,indent=2))
    (destination/"completion.json").write_text(json.dumps(dict(completed=True,requested_fits=len(tasks),
        completed_prediction_artifacts=sum(f["completed_prediction_artifact"] for f in fits),
        failures=len(failures),rows=len(data),elapsed_seconds=time.perf_counter()-tick,
        task_sha256=sha256(args.tasks),metrics_source_sha256=sha256(Path(__file__).with_name("benchmark_metrics.py")),
        truth_access="Evaluator only, after prediction artifacts exist",finite_nonconvergence_predictions_retained=True),indent=2))


if __name__=="__main__":main()
