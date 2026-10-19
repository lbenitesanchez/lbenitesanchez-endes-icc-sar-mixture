#!/usr/bin/env python3
"""Paired evaluation and explicitly separated development selection.

The balanced development score averages each endpoint/context equally within
each independent population-seed replicate. It never tunes final-seed results.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from benchmark_inputs import OUT,SCENARIOS,PRIMARY_SCENARIOS
from benchmark_metrics import paired_contrasts,summarize


def balanced_development(records,candidates):
    data=records[(records.group=="all")&records.method.isin(candidates)].copy()
    expected=len(PRIMARY_SCENARIOS)*2
    if (data.replication>=10000).any():
        raise ValueError("Independent final seeds cannot select a candidate")
    counts=data.groupby(["method","replication"]).size()
    if not (counts==expected).all():
        raise ValueError("Balanced selection requires all six context/outcome cells")
    if data.duplicated(["method","replication","scenario","outcome"]).any():
        raise ValueError("A development cell occurs more than once")
    perrep=data.groupby(["method","replication"]).interval_score.mean().reset_index()
    table=perrep.groupby("method").interval_score.agg(["mean","std","count"]).reset_index()
    table["mcse"]=table["std"]/np.sqrt(table["count"])
    table=table.rename(columns={"mean":"balanced_interval_score","count":"replications"})
    table=table.sort_values("balanced_interval_score").drop(columns="std")
    contrasts=[];chosen=table.iloc[0].method
    a=perrep[perrep.method==chosen].set_index("replication").interval_score
    for reference in candidates:
        if reference==chosen:continue
        b=perrep[perrep.method==reference].set_index("replication").interval_score
        delta=a-b
        contrasts.append(dict(candidate=chosen,reference=reference,replications=len(delta),
            balanced_interval_score_change=float(delta.mean()),
            paired_mcse=float(delta.std(ddof=1)/np.sqrt(len(delta))),
            relative_score_change_pct=float(100*(a.mean()/b.mean()-1))))
    return table,pd.DataFrame(contrasts),chosen


def main():
    p=argparse.ArgumentParser();p.add_argument("--tag",required=True)
    p.add_argument("--additional-tags",nargs="*",default=[])
    p.add_argument("--candidate",default="BetaPowerSpatial")
    p.add_argument("--select-development",action="store_true")
    args=p.parse_args();folder=OUT/args.tag
    frames=[pd.read_csv(OUT/tag/"replications.csv") for tag in [args.tag,*args.additional_tags]]
    records=pd.concat(frames,ignore_index=True)
    keys=["scenario","replication","method","protocol","group","outcome"]
    if records.duplicated(keys).any():raise ValueError("Combined runs contain duplicate evaluation cells")
    summarize(records).to_csv(folder/"combined_summary.csv",index=False)
    candidate=records[records.method==args.candidate]
    if candidate.empty:raise ValueError("No candidate predictions")
    contrasts=[]
    for method in records.method.unique():
        if method==args.candidate:continue
        for protocol,reference in records[records.method==method].groupby("protocol"):
            contrasts.append(paired_contrasts(candidate,reference,args.candidate+" minus "+method+" ["+protocol+"]"))
    if contrasts:pd.concat(contrasts,ignore_index=True).to_csv(folder/"paired_contrasts.csv",index=False)
    report=dict(selection_permitted=False,candidate=args.candidate,
                paired_missingness="Each contrast reports actual common population-seed count; exceptions are separately logged",
                final_performance_claimed=False)
    if args.select_development:
        table,paired,chosen=balanced_development(records,["BetaPowerSpatial","BetaPowerSpatialKish","BetaPowerIndependent"])
        table.to_csv(folder/"development_selection.csv",index=False)
        paired.to_csv(folder/"development_selection_paired.csv",index=False)
        report.update(selection_permitted=True,selected_for_HMC_development=chosen,
            criterion="Equal mean 95% interval score over all80 domains, both outcomes and all3 contexts; old development seeds only",
            independent_validation_pending=True)
    (folder/"comparison_manifest.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":main()
