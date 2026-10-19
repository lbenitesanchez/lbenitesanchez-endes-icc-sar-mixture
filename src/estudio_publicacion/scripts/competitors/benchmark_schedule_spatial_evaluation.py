#!/usr/bin/env python3
"""Wait for all fixed spatial predictions, then run the isolated evaluator."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from benchmark_spatial_evaluate import (OUT,TASKS,CANDIDATE_TAG,OUTPUT_TAG,
    prepare_contract,validate_tasks,task_accounted,sha256)
import pandas as pd


def main():
    p=argparse.ArgumentParser();p.add_argument('--poll-seconds',type=int,default=30);args=p.parse_args()
    if args.poll_seconds<10:raise ValueError('Use a low-frequency observer, at least10 seconds')
    prepare_contract();tasks=validate_tasks(pd.read_csv(TASKS)).to_dict('records')
    manifest=OUT/'spatial_evaluation_queue_manifest.json'
    final=OUT/OUTPUT_TAG/'completion.json'
    state=dict(state='waiting_for_spatial_predictions',created_utc=datetime.now(timezone.utc).isoformat(),
        requested_spatial_fits=100,additional_independent_populations=0,
        candidate_tag=CANDIDATE_TAG,main800_and_Holm3_unchanged=True,
        task_sha256=sha256(TASKS),observer_source_sha256=sha256(Path(__file__)))
    while not final.exists():
        count=sum(task_accounted(task)for task in tasks)
        candidate_complete=(OUT/CANDIDATE_TAG/'completion.json').exists()and not (OUT/CANDIDATE_TAG/'invalidated.json').exists()
        state.update(accounted_spatial_fits=count,candidate_structural_v2_completed=candidate_complete,
            checked_utc=datetime.now(timezone.utc).isoformat())
        manifest.write_text(json.dumps(state,indent=2))
        if count==100 and candidate_complete:
            log=OUT/'spatial_evaluation.log'
            with log.open('w')as handle:
                result=subprocess.run([sys.executable,str(Path(__file__).with_name('benchmark_spatial_evaluate.py'))],
                    stdout=handle,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('Spatial evaluation failed; preserve artifacts and inspect '+str(log))
            break
        time.sleep(args.poll_seconds)
    state.update(state='completed',completion_sha256=sha256(final),
        finished_utc=datetime.now(timezone.utc).isoformat())
    manifest.write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)


if __name__=='__main__':main()
