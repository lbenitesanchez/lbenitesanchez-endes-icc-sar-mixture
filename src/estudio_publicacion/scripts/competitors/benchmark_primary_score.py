#!/usr/bin/env python3
"""Locked primary balanced interval-score contrasts and Holm correction.

Each primary-context replicate contributes the average of two endpoint scores.
The three context means receive equal weight. Native structural sensitivities,
extra-H arms and historical development results cannot enter this confirmatory
table. This is Monte Carlo evidence for the simulation experiment only.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from benchmark_inputs import OUT,PRIMARY_SCENARIOS


def holm(pvalues):
    pvalues=np.asarray(pvalues,float);order=np.argsort(pvalues);adjusted=np.empty(len(order))
    running=0.
    for rank,index in enumerate(order):
        running=max(running,min(1.,(len(order)-rank)*pvalues[index]));adjusted[index]=running
    return adjusted


def primary_contrast(data,candidate,reference,expected_reps):
    data=data[(data.group=="all")&data.scenario.isin(PRIMARY_SCENARIOS)
              &data.method.isin([candidate,reference])].copy()
    if (data.replication<10000).any():raise ValueError("Development cannot enter independent primary contrasts")
    if not (data.areas==80).all():raise ValueError("Primary score requires complete predictions for the same80-area target")
    keys=["method","scenario","replication","outcome"]
    if data.duplicated(keys).any():raise ValueError("Each method must have exactly one primary protocol per cell")
    common=data[data.method==candidate].merge(data[data.method==reference],
        on=["scenario","replication","outcome"],suffixes=("_candidate","_reference"),validate="one_to_one")
    common["delta"]=common.interval_score_candidate-common.interval_score_reference
    per=common.groupby(["scenario","replication"]).agg(
        score_candidate=("interval_score_candidate","mean"),score_reference=("interval_score_reference","mean"),
        delta=("delta","mean"),cells=("outcome","count"),
        convergence_candidate=("converged_candidate","all"),convergence_reference=("converged_reference","all")).reset_index()
    if not (per.cells==2).all():raise ValueError("Missing endpoint in a primary paired population")
    counts=per.groupby("scenario").size().reindex(PRIMARY_SCENARIOS,fill_value=0)
    requested=set(range(10000,10000+expected_reps))
    missing={s:sorted(requested-set(per.loc[per.scenario==s,"replication"])) for s in PRIMARY_SCENARIOS}
    extra={s:sorted(set(per.loc[per.scenario==s,"replication"])-requested) for s in PRIMARY_SCENARIOS}
    if not (counts==expected_reps).all() or any(missing.values()) or any(extra.values()):
        return dict(candidate=candidate,reference=reference,primary_test_available=False,
                    paired_counts=counts.to_dict(),missing_requested_seeds=missing,extra_unplanned_seeds=extra,
                    reason="Missing execution/artifact pairs; do not silently reduce or replace the primary cohort"),per
    contexts=per.groupby("scenario").delta.agg(["mean","var","count"]).reindex(PRIMARY_SCENARIOS)
    change=float(contexts["mean"].mean());mcse=float(np.sqrt(np.sum(contexts["var"]/contexts["count"])/9))
    c=float(per.groupby("scenario").score_candidate.mean().mean());r=float(per.groupby("scenario").score_reference.mean().mean())
    z=change/mcse if mcse>0 else np.nan
    row=dict(candidate=candidate,reference=reference,primary_test_available=True,
        populations_per_context=expected_reps,independent_paired_populations=3*expected_reps,
        balanced_interval_score_candidate=c,balanced_interval_score_reference=r,
        interval_score_change=change,paired_mcse=mcse,
        change_ci95_lower=change-1.959963984540054*mcse,change_ci95_upper=change+1.959963984540054*mcse,
        relative_score_change_pct=100*(c/r-1),z=z,p_two_sided=float(2*norm.sf(abs(z))),
        candidate_all_endpoint_convergence=float(per.convergence_candidate.mean()),
        reference_all_endpoint_convergence=float(per.convergence_reference.mean()))
    return row,per


def main():
    p=argparse.ArgumentParser();p.add_argument("--tags",nargs="+",required=True)
    p.add_argument("--output-tag",required=True);p.add_argument("--candidate",default="BetaPowerHMCGraph")
    p.add_argument("--references",nargs=3,required=True);p.add_argument("--expected-replications",type=int,default=200)
    args=p.parse_args();folder=OUT/args.output_tag
    if folder.exists():raise FileExistsError("Use an immutable new score-evaluation tag")
    folder.mkdir(parents=True)
    data=pd.concat([pd.read_csv(OUT/tag/"replications.csv") for tag in args.tags],ignore_index=True)
    rows=[]
    for reference in args.references:
        row,paired=primary_contrast(data,args.candidate,reference,args.expected_replications)
        rows.append(row);paired.to_csv(folder/f"paired_population_scores_{reference}.csv",index=False)
    table=pd.DataFrame(rows)
    available=table.primary_test_available
    # The family always has the three predefined competitors. If any comparison
    # is unavailable, the whole confirmatory family remains unavailable.
    if available.all():table["p_Holm3"]=holm(table.p_two_sided)
    else:table["p_Holm3"]=np.nan
    table.to_csv(folder/"primary_balanced_interval_score.csv",index=False)
    (folder/"score_protocol.json").write_text(json.dumps(dict(primary_contexts=list(PRIMARY_SCENARIOS),
        candidate=args.candidate,references=args.references,weighting="Equal context means and equal endpoint means, all80 domains",
        uncertainty="Between-population Monte Carlo variance; endpoint covariance preserved within population",
        multiplicity="Two-sided normal Monte Carlo contrast; Holm family of3 locked competitors",
        structural_sensitivities="Not part of the primary family; exploratory confidence intervals",
        all_finite_nonconverged_predictions_retained=True,no_universal_superiority_claim=True),indent=2))


if __name__=="__main__":main()
