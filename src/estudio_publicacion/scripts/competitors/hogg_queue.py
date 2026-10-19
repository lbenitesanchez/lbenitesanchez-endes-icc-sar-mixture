"""Resource-limited dispatch of disjoint frozen Hogg shards.

Each invocation occupies one CPU worker. Additional lanes claim unstarted
shards atomically; source, seeds, sampler and all scientific denominators stay
fixed. No score is read and no completed failure is replaced.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime,timezone
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--lane',required=True)
    p.add_argument('--plan',type=Path,default=ROOT/'resultados/competidores_hogg/final_source_20261005/shard_plan.csv')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--max-shards',type=int,default=0,help='Optional runtime resource boundary; zero dispatches all available shards')
    args=p.parse_args()
    plan=pd.read_csv(args.plan)
    source=args.plan.parent/'source_snapshot'
    frozen=json.loads((source/'snapshot_manifest.json').read_text())
    runner=ROOT/'scripts/competitors/hogg_benchmark_batch.py'
    if sha(runner)!=frozen['source_sha256']['hogg_benchmark_batch.py']:
        raise ValueError('Dispatch runner differs from the immutable source snapshot')
    for row in plan.to_dict('records'):
        if sha(row['task_csv'])!=row['sha256']:raise ValueError('Shard task identity changed')
    state=args.plan.parent/'queue_dispatch';state.mkdir(exist_ok=True)
    priority_path=state/'runtime_priority.json'
    priority_sha=None
    if priority_path.exists():
        priority=json.loads(priority_path.read_text())
        order=priority['family_order']
        if len(order)!=len(set(order))or set(order)!=set(plan.family):
            raise ValueError('Runtime priority must retain every shard family exactly once')
        priority_sha=sha(priority_path)
        ranks={family:i for i,family in enumerate(order)}
        plan=plan.assign(_priority=plan.family.map(ranks)).sort_values('_priority',kind='stable').drop(columns='_priority')
    records=[];started_shards=0
    for row in plan.to_dict('records'):
        if not args.dry_run and args.max_shards and started_shards>=args.max_shards:break
        if not args.dry_run and(state/('pause_'+args.lane)).exists():break
        output=ROOT/'resultados/competidores_hogg'/row['tag']
        manifest=output/'batch_manifest.json'
        if manifest.exists()and json.loads(manifest.read_text()).get('all_requested_tasks_accounted_for'):
            records.append({'tag':row['tag'],'state':'already_accounted_for'})
            continue
        claim=state/(row['tag']+'.claim.json')
        if args.dry_run:
            records.append({'tag':row['tag'],'state':'claimed'if claim.exists()else'available'})
            continue
        try:
            fd=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            records.append({'tag':row['tag'],'state':'claimed_by_other_lane'})
            continue
        claim_record={'lane':args.lane,'queue_pid':os.getpid(),'tag':row['tag'],
            'claimed_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(args.plan),
            'tasks_sha256':row['sha256'],'source_snapshot_manifest_sha256':sha(source/'snapshot_manifest.json')}
        claim_record['runtime_priority_sha256']=priority_sha
        with os.fdopen(fd,'w')as f:json.dump(claim_record,f,indent=2)
        if output.exists():
            # An interrupted batch is never silently overwritten. Its complete
            # requested denominator remains in the collector for manual audit.
            claim_record.update(state='existing_incomplete_batch_requires_audit',success=False)
            claim.write_text(json.dumps(claim_record,indent=2))
            records.append(claim_record)
            continue
        cmd=[sys.executable,str(runner),'--tasks',row['task_csv'],'--tag',row['tag'],
             '--workers','1','--source-snapshot',str(source)]
        log=state/(row['tag']+'.log')
        print(json.dumps({'lane':args.lane,'starting':row['tag'],'requested':int(row['tasks'])}),flush=True)
        with log.open('w')as f:
            result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=False)
        started_shards+=1
        finished=json.loads(manifest.read_text())if manifest.exists()else{}
        claim_record.update(state='finished',returncode=result.returncode,
            all_requested_tasks_accounted_for=finished.get('all_requested_tasks_accounted_for',False),
            completed_fits=finished.get('completed_fits'),converged=finished.get('converged'),
            finished_utc=datetime.now(timezone.utc).isoformat())
        claim.write_text(json.dumps(claim_record,indent=2))
        records.append(claim_record)
        print(json.dumps(claim_record),flush=True)
    (state/(args.lane+'.summary.json')).write_text(json.dumps(records,indent=2))
    if args.dry_run:print(json.dumps(records),flush=True)


if __name__=='__main__':main()
