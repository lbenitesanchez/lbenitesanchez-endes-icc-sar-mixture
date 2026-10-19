#!/usr/bin/env python3
"""Dispatch-only preparation of the approved independent cohorts.

No estimator, population generator, seed or frozen statistical source is edited.
Every exported cohort is validated before its readiness marker is written.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import pandas as pd
from benchmark_inputs import ROOT, OUT, SIM, export_replication, sha256
from benchmark_validate_inputs import validate


def check_protocol():
    protocol=json.loads((OUT/'protocol_freeze.json').read_text())
    if not protocol['approved_for_independent_evaluation']:
        raise RuntimeError('Approved independent protocol required')
    for name,expected in protocol['source_sha256'].items():
        path=SIM/'source_snapshots/sparse_v2/models.py' if name=='models.py' else Path(__file__).with_name(name)
        if sha256(path)!=expected:raise RuntimeError(f'Frozen source mismatch: {name}')
    return protocol


def task_plan(protocol):
    plan=OUT/'task_plan';plan.mkdir(exist_ok=True)
    primary=[(s,r)for r in range(10000,10200)for s in protocol['primary_scenarios']]
    structural=[(s,r)for r in range(20000,20100)for s in protocol['structural_scenarios']]
    assert len(primary)==600 and len(structural)==200
    all_cohorts=primary+structural
    assert len(set(all_cohorts))==800
    for s,r in all_cohorts:
        assert r in protocol['independent_seed_ids'][s]
    cohorts=pd.DataFrame(all_cohorts,columns=['scenario','replication'])
    cohorts.to_csv(plan/'cohorts800.csv',index=False)
    hogg_shards={
        'hogg_primary_first300.csv':[(s,r,'primary')for s,r in primary if r<10100],
        'hogg_primary_remaining300.csv':[(s,r,'primary')for s,r in primary if r>=10100],
        'hogg_structural200.csv':[(s,r,'primary')for s,r in structural],
        'hogg_finiteH200.csv':[('binary_informative_mar',r,'finite_plusH')for r in range(10000,10200)],
        'hogg_designPSU100.csv':[('sparse_psu_2025',r,'design_PSU')for r in range(10000,10100)]}
    for name,tasks in hogg_shards.items():
        pd.DataFrame(tasks,columns=['scenario','replication','arm']).to_csv(plan/name,index=False)
    pd.concat([pd.read_csv(plan/name)for name in hogg_shards],ignore_index=True).to_csv(plan/'hogg_all1100.csv',index=False)
    gao=[]
    for s,r in all_cohorts:
        for method,mode in [('GaoAuthorStrict','author_strict'),('GaoCommonContinuityX','common_continuity')]:
            indir=OUT/'inputs'/s/f'rep_{r:05d}'/'model_inputs'
            gao.append(dict(scenario=s,replication=r,method=method,protocol='frozen_independent_v1',
                diagnostic_style='gao',model_inputs=str(indir),
                output_dir=str(OUT/'external_predictions/gao'/method/s/f'rep_{r:05d}'),
                variant='area_only',stage2_mode=mode,
                extension_file=str(OUT/'extension_inputs'/s/f'rep_{r:05d}.csv') if mode=='common_continuity' else ''))
    for r in range(10000,10200):
        s='binary_informative_mar';method='GaoAuthorStrictFullH'
        gao.append(dict(scenario=s,replication=r,method=method,protocol='extra_population_auxiliary_H_v1',
            diagnostic_style='gao',model_inputs=str(OUT/'inputs'/s/f'rep_{r:05d}'/'model_inputs'),
            output_dir=str(OUT/'external_predictions/gao'/method/s/f'rep_{r:05d}'),
            variant='full_population_X',stage2_mode='author_strict',extension_file=''))
    gao=pd.DataFrame(gao);assert len(gao)==1800
    gao.to_csv(plan/'gao_all1800.csv',index=False)
    gao[(gao.replication<10100)&gao.scenario.isin(protocol['primary_scenarios'])].to_csv(plan/'gao_first700.csv',index=False)
    contract=json.loads((OUT/'final_result_contract.json').read_text())
    candidate_tags=contract['final_tags'][:2]
    manifest=dict(prepared_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha256(OUT/'protocol_freeze.json'),dispatcher_sha256=sha256(Path(__file__)),
        unique_populations=800,primary_populations=600,structural_populations=200,
        Hogg_requested_fits=1100,Gao_requested_fits=1800,additional_arms_paired=True,
        Hogg_design_PSU_subset='sparse_psu_2025, IDs10000..10099',
        output_names=dict(candidate_primary=candidate_tags[0],candidate_structural=candidate_tags[1],
            final_directory='final_evaluation',summary='combined_summary.csv',
            replications='combined_replications.csv',paired_contrasts='paired_contrasts.csv',
            primary_score='primary_balanced_interval_score.csv',structural='structural_summary.csv',
            additional_arms='additional_arms_summary.csv',completion='final_completion.json'),
        csv_sha256={p.name:sha256(p)for p in plan.glob('*.csv')})
    (plan/'task_plan_manifest.json').write_text(json.dumps(manifest,indent=2))
    return [('primary_first300',primary[:300]),('primary_remaining300',primary[300:]),('structural200',structural)]


def prepare_one(task):
    s,r=task;folder=OUT/'inputs'/s/f'rep_{r:05d}'
    if not folder.exists():export_replication(s,r)
    result=validate(folder)
    result['manifest_sha256']=sha256(folder/'manifest.json')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();protocol=check_protocol();stages=task_plan(protocol)
    started=time.monotonic();checks=[]
    for label,tasks in stages:
        stage=[]
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'))as pool:
            futures={pool.submit(prepare_one,t):t for t in tasks}
            for future in as_completed(futures):
                try:row=future.result()
                except Exception as error:
                    failed=dict(stage=label,task=futures[future],error_type=type(error).__name__,error=str(error))
                    (OUT/'task_plan/preparation_failure.json').write_text(json.dumps(failed,indent=2))
                    raise
                stage.append(row);checks.append(row)
                if len(stage)%25==0 or len(stage)==len(tasks):
                    print(json.dumps(dict(stage=label,stage_completed=len(stage),stage_requested=len(tasks),
                        total_validated=len(checks),elapsed_seconds=time.monotonic()-started)),flush=True)
        marker=dict(ready=True,stage=label,validated_populations=len(stage),checks=stage,
            protocol_sha256=sha256(OUT/'protocol_freeze.json'),finished_utc=datetime.now(timezone.utc).isoformat())
        (OUT/'task_plan'/f'{label}_READY.json').write_text(json.dumps(marker,indent=2))
    assert len(checks)==800
    result=dict(passed=True,independent_populations=800,checks=checks,
        protocol_sha256=sha256(OUT/'protocol_freeze.json'),seconds_total=time.monotonic()-started,
        dispatcher_sha256=sha256(Path(__file__)),validation_source_sha256=sha256(Path(__file__).with_name('benchmark_validate_inputs.py')))
    (OUT/'input_validation_final800.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items()if k!='checks'}),flush=True)


if __name__=='__main__':main()
