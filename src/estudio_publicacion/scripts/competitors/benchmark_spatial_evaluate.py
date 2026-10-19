#!/usr/bin/env python3
"""Evaluate the fixed Hogg S2 BYM2 sensitivity independently of primary Holm3.

Only finished prediction artifacts are evaluated. The 100 populations are the
existing structural BYM2 worlds; no fit, replacement seed or new truth is made.
The original frozen replication/MCSE functions are reused unchanged.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import numpy as np
import pandas as pd
from benchmark_inputs import ROOT, OUT, sha256
from benchmark_metrics import summarize, paired_contrasts, OUTCOMES

SCENARIO = 'sparse_psu_bym2'
IDS = set(range(20000,20100))
METHOD = 'HoggSpatial'
CANDIDATE = 'BetaPowerHMCGraph'
CONTRAST = CANDIDATE+'_minus_'+METHOD
PROTOCOL = 'separate_S2_BYM2_component_sensitivity_v1'
TASKS = OUT/'task_plan/hogg_spatial100.csv'
CONTRACT = OUT/'spatial_result_contract.json'
SOURCE_TAG = 'final_hogg_spatial_source_v1'
CANDIDATE_TAG = 'final_structural_v2'
OUTPUT_TAG = 'spatial_evaluation'


def numerical_execution_amendment(freeze_path):
    path=ROOT/'resultados/competidores_hogg/hogg_spatial_numerical_amendment.json'
    amendment=json.loads(path.read_text());freeze=json.loads(freeze_path.read_text())
    if not amendment['approved_after_independent_audit_and_pilot']:
        raise ValueError('Separate spatial numerical initialization is not approved')
    if not amendment['original_prior_target_data_support_graph_unchanged']:
        raise ValueError('Spatial numerical amendment changed statistical support')
    if amendment['original_spatial_freeze_sha256']!=sha256(freeze_path):
        raise ValueError('Spatial numerical amendment references a different original freeze')
    if amendment['original_source_sha256']!=freeze['source_sha256']:
        raise ValueError('Spatial numerical amendment changed original source hashes')
    for key in ['numerical_gates','fixed_retry']:
        if amendment[key]!=freeze[key]:raise ValueError('Spatial numerical amendment changed '+key)
    if amendment['sampler']!=freeze['initial_sampler']:
        raise ValueError('Spatial numerical amendment changed sampler settings')
    adapter=Path(__file__).with_name('hogg_spatial_numeric.py')
    if sha256(adapter)!=amendment['adapter_sha256']:
        raise ValueError('Spatial numerical adapter source hash changed')
    for prefix in ['independent_audit','numerical_pilot']:
        if sha256(ROOT/amendment[prefix+'_path'])!=amendment[prefix+'_sha256']:
            raise ValueError('Spatial numerical amendment evidence changed: '+prefix)
    archive=OUT/'spatial_execution_contract_archive/v1_before_initializer_adapter/spatial_result_contract.json'
    return dict(amendment_file=str(path.relative_to(ROOT)),amendment_sha256=sha256(path),
        adapter_file=str(adapter.relative_to(ROOT)),adapter_sha256=sha256(adapter),
        original_result_contract_sha256=sha256(archive)if archive.exists()else None,
        target_priors_sampler_gates_data_graph_and100_IDs_unchanged=True,
        change='Data-derived chain initial values and positive initial kinetic scaling only; no score or statistical model change',
        independent_audit_sha256=amendment['independent_audit_sha256'],
        numerical_pilot_sha256=amendment['numerical_pilot_sha256'])


def expected_tasks():
    base=ROOT/'resultados/competidores_hogg'
    return pd.DataFrame([dict(scenario=SCENARIO,replication=r,method=METHOD,
        protocol=PROTOCOL,diagnostic_style='hogg',
        output_dir=str(base/f'spatial_sensitivity_{(r-20000)//25:02d}'/'structural_bym2'/f'rep_{r:05d}'))
        for r in sorted(IDS)])


def validate_tasks(tasks):
    required=['scenario','replication','method','protocol','diagnostic_style','output_dir']
    if not set(required).issubset(tasks):raise ValueError('Spatial task columns are incomplete')
    actual=tasks[required].sort_values('replication').reset_index(drop=True)
    if len(actual)!=100 or actual.replication.duplicated().any():
        raise ValueError('Spatial sensitivity requires exactly100 distinct existing worlds')
    pd.testing.assert_frame_equal(actual,expected_tasks()[required],check_dtype=False)
    return actual


def prepare_contract():
    TASKS.parent.mkdir(parents=True,exist_ok=True)
    if TASKS.exists():validate_tasks(pd.read_csv(TASKS))
    else:expected_tasks().to_csv(TASKS,index=False)
    freeze=ROOT/'resultados/competidores_hogg/hogg_spatial_freeze.json'
    contract=dict(execution_contract_version=2,method=METHOD,contrast=CONTRAST,scenario=SCENARIO,
        fixed_population_ids=sorted(IDS),requested_populations=100,
        additional_independent_populations=0,
        candidate_tag=CANDIDATE_TAG,source_tag=SOURCE_TAG,final_directory=OUTPUT_TAG,
        task_file=str(TASKS.relative_to(ROOT)),task_sha256=sha256(TASKS),
        spatial_freeze=str(freeze.relative_to(ROOT)),spatial_freeze_sha256=sha256(freeze),
        numerical_execution_amendment=numerical_execution_amendment(freeze),
        files=dict(replications='replications.csv',summary='summary.csv',
                   paired='paired_contrasts.csv',completion='completion.json'),
        methods_source_sha256=sha256(Path(__file__).with_name('benchmark_metrics.py')),
        evaluator_sha256=sha256(Path(__file__)),
        inference='Separate exploratory paired sensitivity; unadjusted MC intervals, outside primary Holm3',
        failure_policy='Retain finite nonconverged predictions; explicit requested100 denominators; no seed replacement',
        evaluation_guard='evaluation_completed AND all_requested_populations_represented AND all_requested_candidate_pairs_available')
    if CONTRACT.exists():
        if json.loads(CONTRACT.read_text())!=contract:
            raise ValueError('Existing spatial result contract differs; preserve and document any amendment')
    else:CONTRACT.write_text(json.dumps(contract,indent=2))
    return contract


def task_accounted(task):
    folder=Path(task['output_dir'])
    finished=any((folder/name).exists()for name in ('task_execution.json','worker_failure.json','execution_failure.json'))
    artifacts=(folder/'predictions.csv').exists()and(folder/'status.json').exists()
    explicit_failure=any((folder/name).exists()for name in ('worker_failure.json','execution_failure.json'))
    return finished and (artifacts or explicit_failure)


def checked_frame(data,method):
    if data.empty:return data
    if set(data.method)!={method} or set(data.scenario)!={SCENARIO}:
        raise ValueError('Unplanned method or scenario in spatial evaluation')
    if not set(data.replication.astype(int)).issubset(IDS):
        raise ValueError('Unplanned or development population in spatial evaluation')
    if data.duplicated(['scenario','replication','method','protocol','group','outcome']).any():
        raise ValueError('Duplicated spatial replication cells')
    return data


def endpoint_counts(data):
    counts={}
    for name in OUTCOMES:
        allrows=data[(data.group=='all')&(data.outcome==name)] if len(data) else pd.DataFrame()
        complete=allrows[(allrows.areas==80)&(allrows.areas_requested==80)] if len(allrows) else allrows
        counts[name]=dict(requested_replications=100,
            prediction_replications=int(len(allrows)),
            complete80_area_replications=int(len(complete)),
            converged_prediction_replications=int(allrows.converged.sum()) if len(allrows) else 0,
            finite_nonconverged_prediction_replications=int((~allrows.converged).sum()) if len(allrows) else 0,
            complete_population_ids=sorted(complete.replication.astype(int).tolist()) if len(complete) else [])
    return counts


def evaluate():
    contract=prepare_contract();tasks=validate_tasks(pd.read_csv(TASKS))
    if not all(task_accounted(t)for t in tasks.to_dict('records')):
        raise RuntimeError('Some requested spatial fits are still pending; no incomplete evaluation is written')
    freeze=ROOT/contract['spatial_freeze'];frozen=json.loads(freeze.read_text())
    if not frozen['approved_after_numerical_pilot'] or sha256(freeze)!=contract['spatial_freeze_sha256']:
        raise ValueError('Spatial freeze changed')
    for name,expected in frozen['source_sha256'].items():
        if sha256(Path(__file__).with_name(name))!=expected:raise ValueError('Spatial source changed: '+name)
    candidate_folder=OUT/CANDIDATE_TAG
    if (candidate_folder/'invalidated.json').exists():raise ValueError('Invalidated candidate predictions cannot be used')
    candidate_completion=candidate_folder/'completion.json'
    if not candidate_completion.exists():raise RuntimeError('Candidate v2 structural run is still pending')
    amendment=json.loads((OUT/'numerical_amendment_v2.json').read_text())
    if not amendment.get('approved_for_dispatch'):
        raise ValueError('Candidate numerical amendment v2 is not approved')
    destination=OUT/OUTPUT_TAG
    if destination.exists():raise FileExistsError('Spatial final evaluation is immutable; inspect existing completion')
    source=OUT/SOURCE_TAG
    if not (source/'completion.json').exists():
        if source.exists():raise FileExistsError('Partial immutable spatial collection exists')
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('benchmark_collect.py')),
            '--tasks',str(TASKS),'--tag',SOURCE_TAG],check=False)
        if result.returncode:raise RuntimeError('Spatial prediction collector failed; preserve its partial output')
    collection=json.loads((source/'completion.json').read_text())
    if collection['requested_fits']!=100 or collection['task_sha256']!=sha256(TASKS):
        raise ValueError('Spatial source collection differs from fixed100 tasks')
    try:reference=pd.read_csv(source/'replications.csv')
    except pd.errors.EmptyDataError:reference=pd.DataFrame()
    reference=checked_frame(reference,METHOD)
    candidate_all=pd.read_csv(candidate_folder/'replications.csv')
    if reference.empty:reference=pd.DataFrame(columns=candidate_all.columns)
    candidate=candidate_all[(candidate_all.method==CANDIDATE)&
        (candidate_all.scenario==SCENARIO)&candidate_all.replication.isin(IDS)].copy()
    candidate=checked_frame(candidate,CANDIDATE)
    if candidate.protocol.nunique()>1:raise ValueError('Candidate protocols must not be pooled')
    destination.mkdir(parents=True)
    reference.to_csv(destination/'replications.csv',index=False)
    summary=summarize(reference)if len(reference)else pd.DataFrame(columns=[
        'scenario','method','protocol','group','outcome','replications','convergence_rate',
        'mean_areas','rmse','rmse_mcse','bias','bias_mcse','coverage','coverage_mcse',
        'width','width_mcse','interval_score','interval_score_mcse'])
    summary['experiment']='paired_spatial_component_sensitivity'
    summary['requested_replications']=100
    summary['unavailable_prediction_replications']=100-summary.replications
    summary['converged_requested_rate']=summary.convergence_rate*summary.replications/100
    summary.to_csv(destination/'summary.csv',index=False)
    paired=paired_contrasts(candidate,reference,CONTRAST)if len(reference) and len(candidate) else pd.DataFrame(columns=[
        'scenario','group','outcome','contrast','paired_replications','interval_score_change','interval_score_change_mcse'])
    paired['requested_paired_replications']=100
    paired['unavailable_pair_replications']=100-paired.paired_replications
    paired['inference']='Exploratory unadjusted paired Monte Carlo contrast; outside primary Holm3'
    paired.to_csv(destination/'paired_contrasts.csv',index=False)
    counts=endpoint_counts(reference);candidate_counts=endpoint_counts(candidate)
    complete=all(set(counts[name]['complete_population_ids'])==IDS for name in OUTCOMES)
    candidate_complete=all(set(candidate_counts[name]['complete_population_ids'])==IDS for name in OUTCOMES)
    pairs_all=paired[paired.group=='all'] if len(paired)else paired
    pairs_complete=len(pairs_all)==2 and set(pairs_all.outcome)==set(OUTCOMES) and bool((pairs_all.paired_replications==100).all())
    failures=json.loads((source/'failures.json').read_text())
    fits=json.loads((source/'fits.json').read_text())
    (destination/'failures.json').write_text(json.dumps(failures,indent=2))
    (destination/'fits.json').write_text(json.dumps(fits,indent=2))
    completion=dict(evaluation_completed=True,finished_utc=datetime.now(timezone.utc).isoformat(),
        scenario=SCENARIO,method=METHOD,contrast=CONTRAST,unique_requested_populations=100,
        population_ids=sorted(IDS),additional_independent_populations=0,
        all_requested_fits_accounted=True,
        completed_prediction_artifacts=collection['completed_prediction_artifacts'],
        all_requested_populations_represented=complete,
        all_requested_candidate_pairs_available=complete and candidate_complete and pairs_complete,
        reference_endpoint_counts=counts,candidate_endpoint_counts=candidate_counts,
        failure_records=len(failures),finite_nonconverged_predictions_retained=True,
        metric_means='Available prediction artifacts only; requested100 and missing counts explicit; no silent replacement',
        primary_800_population_evaluation_unchanged=True,primary_Holm3_unchanged=True,
        inference='Separate exploratory sensitivity; unadjusted paired MCSE; no universal superiority claim',
        source_collection_completion=collection,
        candidate_completion_sha256=sha256(candidate_completion),
        candidate_replications_sha256=sha256(candidate_folder/'replications.csv'),
        task_sha256=sha256(TASKS),contract_sha256=sha256(CONTRACT),
        spatial_freeze_sha256=sha256(freeze),numerical_amendment_v2_sha256=sha256(OUT/'numerical_amendment_v2.json'),
        spatial_numerical_execution_amendment=contract['numerical_execution_amendment'],
        metrics_source_sha256=contract['methods_source_sha256'],evaluator_sha256=sha256(Path(__file__)),
        output_sha256={p.name:sha256(p)for p in destination.iterdir()if p.is_file()})
    (destination/'completion.json').write_text(json.dumps(completion,indent=2))
    print(json.dumps({k:v for k,v in completion.items()if k not in
        ['source_collection_completion','output_sha256','reference_endpoint_counts','candidate_endpoint_counts']},indent=2))
    return completion


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    if args.prepare_only:print(json.dumps(prepare_contract(),indent=2))
    else:evaluate()


if __name__=='__main__':main()
