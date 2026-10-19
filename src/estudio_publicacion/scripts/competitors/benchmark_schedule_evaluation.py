#!/usr/bin/env python3
"""Collect approved outputs once every requested independent fit is accounted for."""
import argparse,json,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from benchmark_inputs import ROOT,OUT,sha256


def accounted(task):
    folder=Path(task['output_dir']);style=task['diagnostic_style']
    diagnostic='diagnostics.json' if style=='gao' else 'status.json'
    if (folder/'predictions.csv').exists()and(folder/diagnostic).exists():return True
    failures=['failure.json'] if style=='gao' else ['execution_failure.json','worker_failure.json']
    return any((folder/f).exists()for f in failures)


def command(script,args,log):
    with log.open('w')as handle:
        run=subprocess.run([sys.executable,str(Path(__file__).with_name(script)),*args],stdout=handle,stderr=subprocess.STDOUT)
    if run.returncode:raise RuntimeError(f'{script} returned{run.returncode}; inspect {log}; do not overwrite partial evaluation')


def main():
    p=argparse.ArgumentParser();p.add_argument('--poll-seconds',type=int,default=30)
    p.add_argument('--candidate-version',type=int,choices=[1,2],default=1);args=p.parse_args()
    candidate_tags=[f'final_primary_v{args.candidate_version}',f'final_structural_v{args.candidate_version}']
    tables=[('gao',OUT/'task_plan/gao_all1800.csv','final_gao_evaluation_v1'),
            ('hogg',OUT/'task_plan/hogg_collector1100.csv','final_hogg_evaluation_v1')]
    manifest=OUT/'evaluation_queue_manifest.json'
    state=dict(state='waiting_for_requested_fits',created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha256(Path(__file__)),external_requested_fits={'gao':1800,'hogg':1100},
        additional_arms_paired=True,unique_populations=800,primary_statistical_code_unchanged=True,
        candidate_tags=candidate_tags,candidate_numerical_version=args.candidate_version)
    manifest.write_text(json.dumps(state,indent=2))
    tasks={name:pd.read_csv(path).to_dict('records')for name,path,_ in tables}
    collected=set()
    while True:
        progress={}
        for name,path,tag in tables:
            if name in collected:continue
            count=sum(accounted(t)for t in tasks[name]);progress[name]=[count,len(tasks[name])]
            if count==len(tasks[name]):
                completion=OUT/tag/'completion.json'
                if not completion.exists():
                    command('benchmark_collect.py',['--tasks',str(path),'--tag',tag],OUT/f'{tag}.log')
                collected.add(name)
        candidate_complete=all((OUT/t/'completion.json').exists() and not (OUT/t/'invalidated.json').exists()
                               for t in candidate_tags)
        state.update(external_accounted_progress=progress,candidate_completed=candidate_complete,
            collected_external_tags=sorted(collected),checked_utc=datetime.now(timezone.utc).isoformat())
        manifest.write_text(json.dumps(state,indent=2))
        if len(collected)==2 and candidate_complete:break
        time.sleep(args.poll_seconds)
    final=OUT/'final_evaluation/final_completion.json'
    if not final.exists():
        command('benchmark_finalize.py',['--tags',*candidate_tags,'final_gao_evaluation_v1','final_hogg_evaluation_v1'],OUT/'final_evaluation.log')
    state.update(state='completed',final_completion_sha256=sha256(final),
        finished_utc=datetime.now(timezone.utc).isoformat())
    manifest.write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)


if __name__=='__main__':main()
