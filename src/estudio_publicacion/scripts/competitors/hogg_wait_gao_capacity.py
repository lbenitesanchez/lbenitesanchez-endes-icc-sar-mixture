"""Dispatch one additional frozen Hogg lane after both Gao queues finish.

Resource scheduling only. Reads execution-accounting JSON, never predictions,
diagnostic acceptance, evaluation truth or scores. Explicit failures preserve
the requested denominator and count as accounted artifacts. The first freed
Gao task is reserved by the parent for candidate sensitivities; only the
second released task permits this third Hogg lane.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT/'resultados/competidores_hogg'
LOGS = ROOT/'resultados/resultados_comparadores/dispatch_logs'


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None  # Missing or currently being written: keep waiting.


def capacity_proof(label):
    marker = LOGS/f'continuation_gao_remaining550_{label}.json'
    execution = LOGS/f'gao_remaining550_{label}/execution.json'
    a, b = read_json(marker), read_json(execution)
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False, {'label': label, 'state': 'waiting_for_complete_JSON'}
    count_keys = ['requested_fits', 'missing_artifacts',
                  'completed_prediction_artifacts', 'explicit_failure_artifacts']
    valid_counts = all(type(b.get(k)) is int for k in count_keys)
    codes = b.get('worker_exit_codes')
    valid_codes = isinstance(codes, list) and len(codes)==1 and type(codes[0]) is int and codes[0]==0
    ready = bool(a.get('state')=='finished' and type(a.get('returncode')) is int
        and a['returncode']==0 and isinstance(a.get('finished_utc'),str)
        and bool(a['finished_utc']) and b.get('completed_execution') is True
        and valid_counts and b['requested_fits']==550 and b['missing_artifacts']==0
        and valid_codes and b['completed_prediction_artifacts']+b['explicit_failure_artifacts']==550)
    return ready, {'label': label, 'ready': ready,
        'marker': str(marker.relative_to(ROOT)), 'execution': str(execution.relative_to(ROOT)),
        'marker_state': a.get('state'), 'returncode': a.get('returncode'),
        'requested_fits': b.get('requested_fits'), 'missing_artifacts': b.get('missing_artifacts'),
        'prediction_artifacts': b.get('completed_prediction_artifacts'),
        'explicit_failures': b.get('explicit_failure_artifacts'), 'worker_exit_codes': codes,
        'completed_execution': b.get('completed_execution')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', default='lane03')
    parser.add_argument('--poll-seconds', type=float, default=15.)
    parser.add_argument('--snapshot-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        raise ValueError('Polling interval must be between 1 and 60 seconds')
    RESULT.mkdir(exist_ok=True)
    state = RESULT/f'{args.lane}_capacity_observer.json'
    if not args.snapshot_only:
        claim = RESULT/f'{args.lane}_capacity_observer.claim.json'
        try:
            fd = os.open(claim, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError('An observer already claimed this lane; audit its PID before dispatching another')
        with os.fdopen(fd, 'w') as stream:
            json.dump({'observer_pid':os.getpid(), 'lane':args.lane,
                'started_utc':datetime.now(timezone.utc).isoformat(),
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},stream,indent=2)
    previous = None
    while True:
        proofs = [capacity_proof(label) for label in ['C','D']]
        ready = all(ok for ok,_ in proofs)
        report = {'state': 'capacity_proved' if ready else 'waiting',
            'utc': datetime.now(timezone.utc).isoformat(), 'observer_pid': os.getpid(),
            'lane': args.lane, 'required_both_queues': [p for _,p in proofs],
            'parent_authorization': 'Third Hogg lane only after both Gao C/D550 continuations finish. First freed task reserved for candidate sensitivities. Effective global maximum is 8 ENDES fitting tasks; this remains a one-for-one replacement. No other project is touched.',
            'maximum_concurrent_ENDES_tasks': 8,
            'capacity_authority': 'resultados/competidores_hogg/runtime_capacity_limit.json',
            'data_access': 'Execution accounting only; no scores, truth, predictions or numerical gate selection.'}
        key=json.dumps([p for _,p in proofs],sort_keys=True)
        if key!=previous or ready:
            temporary=state.with_suffix('.tmp')
            temporary.write_text(json.dumps(report,indent=2));temporary.replace(state)
            print(json.dumps(report),flush=True);previous=key
        if args.snapshot_only:
            return
        if ready:
            # Atomic shard claims in the existing queue prevent duplicate IDs.
            command=[sys.executable,str(ROOT/'scripts/competitors/hogg_queue.py'),
                     '--lane',args.lane]
            report['dispatch_command']='python scripts/competitors/hogg_queue.py --lane '+args.lane
            report['state']='dispatching_one_additional_Hogg_lane'
            state.write_text(json.dumps(report,indent=2))
            result=subprocess.run(command,cwd=ROOT,check=False)
            report.update(state='lane_finished',returncode=result.returncode,
                          finished_utc=datetime.now(timezone.utc).isoformat())
            state.write_text(json.dumps(report,indent=2))
            return
        time.sleep(args.poll_seconds)


if __name__=='__main__':
    main()
