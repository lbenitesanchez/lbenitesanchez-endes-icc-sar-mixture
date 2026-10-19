#!/usr/bin/env python3
"""Failure-aware metadata for the three locked paired additional arms.

No new score calculation or model selection is performed. Existing finite
prediction rows and paired counts are reported alongside every requested fit.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_inputs import ROOT,OUT,sha256

CANDIDATE='BetaPowerHMCGraph'
METHODS=['GaoAuthorStrictFullH','HoggTSLNFullH','HoggDesignPsiPSU']
CONDITION='Available finite candidate/reference predictions, conditional on comparator computational and eligibility availability; exploratory and unadjusted'
FIELDS=['method','scenario','outcome','contrast','requested_population_attempts',
    'attempted_populations','completed_prediction_artifacts',
    'available_endpoint_prediction_populations','candidate_available_endpoint_prediction_populations',
    'converged_endpoint_prediction_populations','finite_nonconverged_endpoint_prediction_populations',
    'unavailable_endpoint_prediction_populations','usable_endpoint_pairs','unavailable_pair_populations',
    'all_requested_attempts_completed','all_requested_endpoint_predictions_available',
    'all_requested_endpoint_pairs_available','contrast_condition']


def prepare_contract():
    protocol=json.loads((OUT/'protocol_freeze.json').read_text())
    arms={name:protocol['paired_additional_arms'][name]for name in METHODS}
    contract=dict(files=dict(availability='final_evaluation/additional_arm_availability.csv',
        failure_reasons='final_evaluation/additional_arm_failures.json',
        contrasts='final_evaluation/paired_contrasts.csv',
        requested_ledger=dict(Gao='final_gao_evaluation_v1/fits.json',Hogg='final_hogg_evaluation_v1/fits.json')),
        methods=arms,availability_header=FIELDS,
        failed_attempts='All requested IDs are attempted and kept in ledgers, with explicit failure reasons; never replaced',
        retained_predictions='Finite nonconverged predictions remain in scores and paired contrasts',
        contrast_condition=CONDITION,
        inference='Secondary exploratory contrast on the actual usable pairs; MCSE unchanged; no new hypothesis family',
        primary_600_and_Holm3_unchanged=True,
        publication_guard='all_requested_attempts_completed; display requested versus usable counts and availability conditioning',
        protocol_sha256=sha256(OUT/'protocol_freeze.json'),
        metadata_source_sha256=sha256(Path(__file__)))
    path=OUT/'secondary_arm_result_contract.json'
    if path.exists()and json.loads(path.read_text())!=contract:
        raise ValueError('Existing secondary result contract differs; preserve and document amendment')
    if not path.exists():path.write_text(json.dumps(contract,indent=2))
    return contract


def attempt_finished(task):
    folder=Path(task['output_dir'])
    if bool(task.get('completed_prediction_artifact',False)):return True
    names=('task_execution.json','execution_failure.json','worker_failure.json','failure.json')
    return any((folder/name).exists()for name in names)


def underlying_failure(task,unavailable_outcomes):
    folder=Path(task['output_dir']);record=dict(method=task['method'],scenario=task['scenario'],
        replication=int(task['replication']),unavailable_outcomes=unavailable_outcomes,
        completed_prediction_artifact=bool(task.get('completed_prediction_artifact',False)),
        output_dir=str(folder.relative_to(ROOT))if folder.is_relative_to(ROOT)else str(folder))
    evidence=[]
    for name in ['execution_failure.json','worker_failure.json','failure.json','task_execution.json']:
        path=folder/name
        if path.exists():
            obj=json.loads(path.read_text())
            if name=='task_execution.json'and obj.get('completed',True):continue
            evidence.append(dict(file=str(path.relative_to(ROOT))if path.is_relative_to(ROOT)else str(path),
                sha256=sha256(path),error_type=obj.get('error_type',obj.get('type')),
                reason=obj.get('error',obj.get('message',obj.get('reason'))),
                full_record=obj))
    record['failure_evidence']=evidence
    record['fallback_reason']='Prediction artifact or finite endpoint output unavailable; inspect source collector failures'if not evidence else None
    return record


def report(data,pairs,ledgers,arms):
    rows=[];failures=[]
    for method in METHODS:
        spec=arms[method];scenario=spec['scenario'];ids=set(map(int,spec['seed_ids']))
        tasks=[task for task in ledgers if task['method']==method and task['scenario']==scenario]
        if len(tasks)!=len(ids)or{int(task['replication'])for task in tasks}!=ids:
            raise ValueError('Secondary requested-fit ledger differs from locked cohort: '+method)
        attempts=sum(attempt_finished(task)for task in tasks)
        usable_by_outcome={}
        for outcome in ['anemia','stunting']:
            reference=data[(data.method==method)&(data.scenario==scenario)&(data.group=='all')&(data.outcome==outcome)]
            candidate=data[(data.method==CANDIDATE)&(data.scenario==scenario)&(data.group=='all')&(data.outcome==outcome)&data.replication.isin(ids)]
            if reference.replication.duplicated().any()or candidate.replication.duplicated().any():
                raise ValueError('Duplicate secondary endpoint prediction cell')
            if not set(reference.replication).issubset(ids):raise ValueError('Unplanned secondary prediction')
            available=set(reference.replication.astype(int));candidate_ids=set(candidate.replication.astype(int))
            usable=available&candidate_ids;usable_by_outcome[outcome]=available
            contrast=CANDIDATE+'_minus_'+method
            pair=pairs[(pairs.contrast==contrast)&(pairs.scenario==scenario)&(pairs.group=='all')&(pairs.outcome==outcome)]
            count=int(pair.paired_replications.iloc[0])if len(pair)==1 else 0
            if len(pair)>1 or count!=len(usable):raise ValueError('Paired contrast count differs from actual finite rows')
            rows.append(dict(method=method,scenario=scenario,outcome=outcome,contrast=contrast,
                requested_population_attempts=len(ids),attempted_populations=attempts,
                completed_prediction_artifacts=sum(bool(task.get('completed_prediction_artifact',False))for task in tasks),
                available_endpoint_prediction_populations=len(available),
                candidate_available_endpoint_prediction_populations=len(candidate_ids),
                converged_endpoint_prediction_populations=int(reference.converged.sum()),
                finite_nonconverged_endpoint_prediction_populations=int((~reference.converged).sum()),
                unavailable_endpoint_prediction_populations=len(ids)-len(available),
                usable_endpoint_pairs=len(usable),unavailable_pair_populations=len(ids)-len(usable),
                all_requested_attempts_completed=attempts==len(ids),
                all_requested_endpoint_predictions_available=available==ids,
                all_requested_endpoint_pairs_available=usable==ids,contrast_condition=CONDITION))
        for task in tasks:
            missing=[outcome for outcome in ['anemia','stunting']if int(task['replication'])not in usable_by_outcome[outcome]]
            if missing:failures.append(underlying_failure(task,missing))
    return pd.DataFrame(rows,columns=FIELDS),failures


def write_metadata(data,pairs,destination):
    contract=prepare_contract();ledgers=[]
    for tag in ['final_gao_evaluation_v1','final_hogg_evaluation_v1']:
        ledgers.extend(json.loads((OUT/tag/'fits.json').read_text()))
    table,failures=report(data,pairs,ledgers,contract['methods'])
    table.to_csv(destination/'additional_arm_availability.csv',index=False)
    (destination/'additional_arm_failures.json').write_text(json.dumps(failures,indent=2))
    return dict(all_requested_attempts_completed=bool(table.all_requested_attempts_completed.all()),
        all_requested_secondary_pairs_available=bool(table.all_requested_endpoint_pairs_available.all()),
        availability_file='additional_arm_availability.csv',failure_reasons_file='additional_arm_failures.json',
        contrast_condition=CONDITION,primary_600_and_Holm3_unchanged=True,
        source_sha256=sha256(Path(__file__)),contract_sha256=sha256(OUT/'secondary_arm_result_contract.json'))


def self_check():
    arms={method:dict(scenario='control',seed_ids=[1,2,3])for method in METHODS}
    records=[];ledgers=[];pairs=[]
    for r in [1,2,3]:
        for name in ['anemia','stunting']:
            records.append(dict(method=CANDIDATE,scenario='control',replication=r,group='all',outcome=name,converged=True))
    for method in METHODS:
        for r in [1,2,3]:
            available=not(method=='HoggDesignPsiPSU'and r==3)
            ledgers.append(dict(method=method,scenario='control',replication=r,
                completed_prediction_artifact=available,output_dir='/nonexistent',
                test_attempt_complete=True))
            if available:
                for name in ['anemia','stunting']:
                    records.append(dict(method=method,scenario='control',replication=r,group='all',outcome=name,converged=r!=1))
        for name in ['anemia','stunting']:
            pairs.append(dict(contrast=CANDIDATE+'_minus_'+method,scenario='control',group='all',outcome=name,
                paired_replications=2 if method=='HoggDesignPsiPSU' else 3))
    original=globals()['attempt_finished'];globals()['attempt_finished']=lambda task:bool(task['test_attempt_complete'])
    try:table,failures=report(pd.DataFrame(records),pd.DataFrame(pairs),ledgers,arms)
    finally:globals()['attempt_finished']=original
    design=table[table.method=='HoggDesignPsiPSU']
    assert (design.requested_population_attempts==3).all() and (design.attempted_populations==3).all()
    assert (design.usable_endpoint_pairs==2).all() and (design.unavailable_endpoint_prediction_populations==1).all()
    assert design.all_requested_attempts_completed.all()and not design.all_requested_endpoint_pairs_available.any()
    assert (design.finite_nonconverged_endpoint_prediction_populations==1).all()and len(failures)==1
    return dict(passed=True,legitimate_failure_keeps_requested_attempts=True,
        available_pair_count_not_inflated=True,finite_nonconvergence_retained=True,
        primary_scores_or_true_population_not_accessed=True,source_sha256=sha256(Path(__file__)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--self-check',action='store_true');args=p.parse_args()
    if args.self_check:
        proof=self_check();path=OUT/'validation/secondary_accounting_qa.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
    elif args.prepare_only:print(json.dumps(prepare_contract(),indent=2))
    else:raise ValueError('Metadata is written by benchmark_finalize after all requested fits are accounted for')


if __name__=='__main__':main()
