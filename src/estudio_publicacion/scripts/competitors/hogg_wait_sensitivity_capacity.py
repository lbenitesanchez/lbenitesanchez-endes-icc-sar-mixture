"""Replace the fully closed seven-sensitivity lane with Hogg lane06.

Resource accounting alone triggers this one-for-one handoff. No fit's
convergence, predictions, truth or comparative performance selects a job.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'resultados/competidores_hogg'
BASE = ROOT / 'resultados/resultados_comparadores/application_sensitivity_v2'
NAMES = {'historical_2019', 'historical_2025', 'rank20', 'rank80',
         'spatial_sd025', 'spatial_sd1', 'no_cluster'}


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def write_state(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def planned_outputs(plan):
    if not isinstance(plan, dict) or not isinstance(plan.get('jobs'), list):
        raise ValueError('Sensitivity plan is unavailable')
    jobs = plan['jobs']
    if len(jobs) != 7 or {job.get('name') for job in jobs} != NAMES:
        raise ValueError('Declared seven sensitivity identities changed')
    paths = {}
    for job in jobs:
        arguments = job['args']
        tag = arguments[arguments.index('--tag') + 1]
        year = arguments[arguments.index('--year') + 1]
        rule = arguments[arguments.index('--rule') + 1] if '--rule' in arguments else 'who2024'
        paths[job['name']] = 'resultados/resultados_comparadores/' + tag + '/aplicacion_' + year + '_' + rule
    return paths


def accounting_ready(completion, states, outputs):
    if not isinstance(completion, dict) or completion.get('completed') is not True:
        return False
    if type(completion.get('jobs')) is not int or completion['jobs'] != 7:
        return False
    if completion.get('all_requested_jobs_recorded') is not True:
        return False
    if not isinstance(states, dict) or set(states) != NAMES:
        return False
    for name, record in states.items():
        if not isinstance(record, dict) or type(record.get('returncode')) is not int or record['returncode'] != 0:
            return False
        if record.get('output') != outputs[name]:
            return False
        status = read_json(ROOT / outputs[name] / 'fit_status.json')
        if not isinstance(status, dict) or not isinstance(status.get('endpoints'), list) or len(status['endpoints']) != 2:
            return False
        for endpoint in status['endpoints']:
            if not isinstance(endpoint, dict) or type(endpoint.get('success')) is not bool:
                return False
            if not isinstance(endpoint.get('sampler_trials'), list) or not endpoint['sampler_trials']:
                return False
    return True


def process_snapshot():
    result = subprocess.run(['ps', '-axo', 'pid=,ppid=,command='], capture_output=True, text=True, check=True)
    records = {}
    for line in result.stdout.splitlines():
        columns = line.strip().split(None, 2)
        if len(columns) != 3:
            continue
        records[int(columns[0])] = {'ppid': int(columns[1]), 'command': columns[2]}
    return records


def live_fitting_processes(processes, controller_pid, outputs, known_descendants):
    controller = processes.get(controller_pid)
    tracked = {controller_pid} if controller and 'beta_power_application_sensitivities.py' in controller['command'] else set()
    changed = True
    while changed:
        changed = False
        for pid, record in processes.items():
            if record['ppid'] in tracked and pid not in tracked:
                tracked.add(pid)
                changed = True
    for pid in tracked:
        known_descendants[pid] = processes[pid]['command']
    live = {pid for pid, command in known_descendants.items()
            if pid in processes and processes[pid]['command'] == command}
    for pid, record in processes.items():
        command = record['command']
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = []
        if any(Path(token).name == 'beta_power_application_sensitivities.py' for token in tokens):
            live.add(pid)
        if any(output in command for output in outputs.values()) and 'beta_power_hybrid' in command:
            live.add(pid)
    return sorted(live)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--after-pid', type=int, required=True)
    parser.add_argument('--poll-seconds', type=float, default=15)
    args = parser.parse_args()
    if args.after_pid <= 0 or args.poll_seconds <= 0:
        raise ValueError('Positive controller PID and interval required')
    claim = RESULTS / 'lane06_capacity_observer.claim.json'
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with os.fdopen(descriptor, 'w') as stream:
        json.dump({'observer_pid': os.getpid(), 'sensitivity_controller_pid': args.after_pid,
                   'source_sha256': source_hash, 'created_utc': datetime.now(timezone.utc).isoformat()}, stream, indent=2)
    plan_path = BASE / 'declared_plan.json'
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    outputs = planned_outputs(read_json(plan_path))
    known_descendants = {}
    state_path = RESULTS / 'lane06_capacity_observer.json'
    base = {'observer_pid': os.getpid(), 'sensitivity_controller_pid': args.after_pid,
        'source_sha256': source_hash, 'sensitivity_plan_sha256': plan_hash, 'lane': 'lane06',
        'authorization': 'Parent: replace exactly one closed sensitivity lane with one Hogg lane; effective global maximum is 8 ENDES fitting tasks.',
        'maximum_concurrent_ENDES_tasks': 8,
        'capacity_authority': 'resultados/competidores_hogg/runtime_capacity_limit.json',
        'data_access': 'Seven job and fourteen endpoint execution accounting plus process liveness; no truth, scores or convergence-based selection.'}
    while True:
        if hashlib.sha256(plan_path.read_bytes()).hexdigest() != plan_hash:
            raise ValueError('Declared sensitivity plan changed; preserve dispatch for audit')
        ready = accounting_ready(read_json(BASE / 'completion.json'), read_json(BASE / 'queue_status.json'), outputs)
        live = live_fitting_processes(process_snapshot(), args.after_pid, outputs, known_descendants)
        if ready and not live:
            break
        write_state(state_path, {**base, 'state': 'waiting', 'execution_accounted': ready,
            'live_sensitivity_controller_or_descendants': live, 'utc': datetime.now(timezone.utc).isoformat()})
        time.sleep(args.poll_seconds)
    evidence = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [BASE / 'completion.json', BASE / 'queue_status.json']}
    write_state(state_path, {**base, 'state': 'running_hogg_queue', 'released_active_tasks': 1,
        'allocated_active_tasks': 1, 'completion_evidence_sha256': evidence,
        'utc': datetime.now(timezone.utc).isoformat()})
    with (RESULTS / 'final_hogg_lane06.log').open('w') as stream:
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/competitors/hogg_queue.py'),
            '--lane', 'lane06'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    write_state(state_path, {**base, 'state': 'finished', 'returncode': result.returncode,
        'completion_evidence_sha256': evidence, 'utc': datetime.now(timezone.utc).isoformat()})
    if result.returncode:
        raise RuntimeError('Hogg lane06 queue requires audit; no completed task is replaced')


if __name__ == '__main__':
    main()
