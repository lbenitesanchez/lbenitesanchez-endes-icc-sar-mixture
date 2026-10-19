#!/usr/bin/env python3
"""Consolidate locked independent predictions after all dispatches complete.

This evaluator preserves failed convergence flags and never substitutes seeds.
All final performance calculations reuse the frozen metric and score functions.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_inputs import OUT,sha256
from benchmark_metrics import summarize,paired_contrasts
from benchmark_primary_score import primary_contrast,holm
from benchmark_secondary_accounting import write_metadata

PRIMARY=['binary_informative_mar','sparse_srs_2025','sparse_psu_2025']
STRUCTURAL=['sparse_srs_iid','sparse_psu_bym2']
REFERENCES=['GaoAuthorStrict','GaoCommonContinuityX','HoggTSLN']
CANDIDATE='BetaPowerHMCGraph'
EXTRA=['GaoAuthorStrictFullH','HoggTSLNFullH','HoggDesignPsiPSU']
PRIMARY_METHODS=['BetaPowerHMCGraph','FH','MFH','SFH','MSFH',*REFERENCES]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tags',nargs='+',default=None)
    p.add_argument('--output-tag',default='final_evaluation');args=p.parse_args()
    if args.tags is None:
        args.tags=json.loads((OUT/'final_result_contract.json').read_text())['final_tags']
    destination=OUT/args.output_tag
    if destination.exists():raise FileExistsError('Use an immutable fresh final-evaluation tag')
    protocol=json.loads((OUT/'protocol_freeze.json').read_text())
    input_validation=json.loads((OUT/'input_validation_final800.json').read_text())
    assert input_validation['passed'] and input_validation['independent_populations']==800
    frames=[];artifacts={}
    for tag in args.tags:
        folder=OUT/tag;path=folder/'replications.csv'
        if (folder/'invalidated.json').exists():raise ValueError(f'Numerically invalidated run cannot be evaluated: {tag}')
        frame=pd.read_csv(path);frame['run_tag']=tag;frames.append(frame)
        artifacts[tag]={'replications_sha256':sha256(path),
            'completion':json.loads((folder/'completion.json').read_text())}
    data=pd.concat(frames,ignore_index=True)
    keys=['scenario','replication','method','protocol','group','outcome']
    if data.duplicated(keys).any():raise ValueError('Duplicated final evaluation cells')
    for s,group in data.groupby('scenario'):
        if s not in protocol['independent_seed_ids']:raise ValueError('Unplanned scenario')
        if not set(group.replication).issubset(protocol['independent_seed_ids'][s]):
            raise ValueError('Unplanned or development seed in final evaluation')
    destination.mkdir(parents=True)
    data['experiment']=np.where(data.method.isin(EXTRA),'paired_additional_arm',
        np.where(data.scenario.isin(STRUCTURAL),'structural_sensitivity','primary'))
    data.to_csv(destination/'combined_replications.csv',index=False)
    table=summarize(data)
    table['experiment']=np.where(table.method.isin(EXTRA),'paired_additional_arm',
        np.where(table.scenario.isin(STRUCTURAL),'structural_sensitivity','primary'))
    table.to_csv(destination/'combined_summary.csv',index=False)
    table[(table.experiment=='primary')&(table.group=='all')&table.method.isin(PRIMARY_METHODS)].to_csv(
        destination/'primary_publication_table.csv',index=False)
    table[table.experiment=='structural_sensitivity'].to_csv(destination/'structural_summary.csv',index=False)
    table[table.experiment=='paired_additional_arm'].to_csv(destination/'additional_arms_summary.csv',index=False)
    table[(table.experiment=='primary')&(table.group=='observed_endpoint')].to_csv(
        destination/'direct_and_SAE_observed_summary.csv',index=False)
    # Sensitivity only; unconditional finite predictions above remain primary.
    summarize(data[data.converged]).to_csv(destination/'converged_only_summary.csv',index=False)
    candidate=data[data.method==CANDIDATE];contrasts=[]
    for method in sorted(set(data.method)-{CANDIDATE}):
        ref=data[data.method==method]
        contrasts.append(paired_contrasts(candidate,ref,f'{CANDIDATE}_minus_{method}'))
    contrasts=pd.concat(contrasts,ignore_index=True)
    contrasts.to_csv(destination/'paired_contrasts.csv',index=False)
    secondary_accounting=write_metadata(data,contrasts,destination)
    primary=[]
    for reference in REFERENCES:
        row,paired=primary_contrast(data,CANDIDATE,reference,200)
        primary.append(row);paired.to_csv(destination/f'paired_population_scores_{reference}.csv',index=False)
    scores=pd.DataFrame(primary)
    scores['p_Holm3']=holm(scores.p_two_sided) if scores.primary_test_available.all() else np.nan
    scores.to_csv(destination/'primary_balanced_interval_score.csv',index=False)
    cohorts=data[['scenario','replication']].drop_duplicates()
    counts={s:int(sum(cohorts.scenario==s))for s in protocol['independent_seed_ids']}
    all_cohorts_complete=all(counts[s]==len(ids)for s,ids in protocol['independent_seed_ids'].items())
    completion=dict(evaluation_completed=True,finished_utc=datetime.now(timezone.utc).isoformat(),
        unique_requested_populations=800,unique_observed_populations=len(cohorts),
        population_counts=counts,all_requested_populations_represented=all_cohorts_complete,
        primary_Holm_family_available=bool(scores.primary_test_available.all()),
        prediction_artifact_runs=artifacts,finite_nonconverged_predictions_retained=True,
        converged_only_table='Sensitivity; never replaces unconditional prediction table',
        additional_arms_are_paired_not_new_populations=True,
        additional_arm_accounting=secondary_accounting,
        additional_arm_failure_reasons_sha256=sha256(destination/'additional_arm_failures.json'),
        protocol_sha256=sha256(OUT/'protocol_freeze.json'),
        input_validation_sha256=sha256(OUT/'input_validation_final800.json'),
        evaluator_sha256=sha256(Path(__file__)),
        output_sha256={f.name:sha256(f)for f in destination.glob('*.csv')},
        unavailable_predictions='Execution failures retain requested-fit denominators in source completion manifests; no implicit seed replacement',
        interpretation='Comparative Monte Carlo evidence for these scenarios only; no universal superiority or empirical ENDES coverage claim')
    (destination/'final_completion.json').write_text(json.dumps(completion,indent=2))
    print(json.dumps({k:v for k,v in completion.items()if k not in ['prediction_artifact_runs','output_sha256']},indent=2))


if __name__=='__main__':main()
