"""Resource-only handoff from spatial applications to the fixed 100 worlds.

This observer dispatches no extra concurrent fitting task: it waits for the
application process to exit, then uses that released lane serially. Source S1
accounting determines readiness; posterior scores, truth and success rates do
not determine task selection. Completed statistical failures remain reported.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'resultados/competidores_hogg'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def write_state(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(path)


def application_accounted(summary):
    if not isinstance(summary, list) or len(summary) != 2:
        return False
    modes = set()
    for record in summary:
        if not isinstance(record, dict):
            return False
        if record.get('year') != 2025 or record.get('prepared_only', False):
            return False
        if not isinstance(record.get('success'), bool):
            return False
        if 'seconds_spatial_fit' not in record or record.get('all_prediction_districts') != 1874:
            return False
        modes.add(record.get('mode'))
    return modes == {'full', 'locked_train'}


def process_exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def source_readiness(rows, start, stop):
    selected = [row for row in rows if row['scenario'] == 'sparse_psu_bym2'
                and start <= int(row.get('replication', row.get('rep_ID'))) < stop]
    identifiers = sorted(int(row.get('replication', row.get('rep_ID'))) for row in selected)
    if identifiers != list(range(start, stop)):
        raise ValueError('Canonical source set must contain every fixed ID exactly once')
    pending = []
    for row in selected:
        directory = Path(row['output_dir'])
        if not directory.is_absolute():
            directory = ROOT / directory
        if not any((directory / name).is_file() for name in
                   ['task_execution.json', 'worker_failure.json', 'execution_failure.json']):
            pending.append(int(row.get('replication', row.get('rep_ID'))))
    return pending


def chunk_accounted(manifest, start, stop):
    return (isinstance(manifest, dict)
            and manifest.get('requested_tasks') == stop - start
            and manifest.get('accounted_for') == stop - start
            and manifest.get('population_IDs') == list(range(start, stop)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--after-pid', type=int, required=True)
    parser.add_argument('--poll-seconds', type=float, default=15)
    args = parser.parse_args()
    if args.after_pid <= 0 or args.poll_seconds <= 0:
        raise ValueError('Positive process ID and polling interval required')
    claim = RESULTS / 'spatial_after_application_observer.claim.json'
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump({'observer_pid': os.getpid(), 'after_pid': args.after_pid,
                   'source_sha256': sha(__file__),
                   'created_utc': datetime.now(timezone.utc).isoformat()}, stream, indent=2)
    state_path = RESULTS / 'spatial_after_application_observer.json'
    plan_path = ROOT / 'resultados/resultados_comparadores/task_plan/hogg_collector1100.csv'
    plan_hash = sha(plan_path)
    with plan_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    dispatch = read_json(RESULTS / 'spatial_secondary_dispatch_plan.json')
    if [chunk['start'] for chunk in dispatch['chunks']] != [20000, 20025, 20050, 20075]:
        raise ValueError('Fixed four-chunk partition changed')
    if [chunk['stop_exclusive'] for chunk in dispatch['chunks']] != [20025, 20050, 20075, 20100]:
        raise ValueError('Fixed 100-population denominator changed')
    base = {'observer_pid': os.getpid(), 'after_application_pid': args.after_pid,
            'source_plan_sha256': plan_hash, 'observer_source_sha256': sha(__file__),
            'purpose': 'Resource-only serial handoff; same approved adapter, original sources, seeds, gates and all100 fixed worlds.',
            'data_access': 'Application/source execution accounting only; no truth, scores or success-rate based dispatch.'}
    while process_exists(args.after_pid) or not application_accounted(read_json(RESULTS / 'spatial_age_sex_application_summary.json')):
        write_state(state_path, {**base, 'state': 'waiting_for_application_exit_and_accounting',
                                 'utc': datetime.now(timezone.utc).isoformat()})
        time.sleep(args.poll_seconds)
    records = []
    for chunk in dispatch['chunks']:
        start, stop, tag = chunk['start'], chunk['stop_exclusive'], chunk['tag']
        output = RESULTS / tag
        manifest_path = output / 'batch_manifest.json'
        if chunk_accounted(read_json(manifest_path), start, stop):
            records.append({'tag': tag, 'state': 'already_accounted_for'})
            continue
        if output.exists():
            raise FileExistsError('Existing incomplete spatial chunk requires audit: ' + tag)
        while True:
            if sha(plan_path) != plan_hash:
                raise ValueError('Canonical S1 source plan changed during waiting')
            pending = source_readiness(rows, start, stop)
            if not pending:
                break
            write_state(state_path, {**base, 'state': 'waiting_for_accounted_source_S1',
                'tag': tag, 'pending_source_IDs': pending, 'records': records,
                'utc': datetime.now(timezone.utc).isoformat()})
            time.sleep(args.poll_seconds)
        command = [sys.executable, str(ROOT / 'scripts/competitors/hogg_spatial_numeric.py'),
                   '--driver', 'batch', '--start', str(start), '--stop', str(stop), '--tag', tag]
        write_state(state_path, {**base, 'state': 'running_spatial_chunk', 'tag': tag,
            'records': records, 'utc': datetime.now(timezone.utc).isoformat()})
        log_path = RESULTS / (tag + '_dispatch.log')
        with log_path.open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
        record = {'tag': tag, 'returncode': result.returncode,
                  'accounted_for': chunk_accounted(read_json(manifest_path), start, stop)}
        records.append(record)
        if result.returncode or not record['accounted_for']:
            write_state(state_path, {**base, 'state': 'dispatch_requires_audit', 'records': records,
                'utc': datetime.now(timezone.utc).isoformat()})
            raise RuntimeError('Spatial chunk has incomplete execution accounting; no replacement or silent retry')
    write_state(state_path, {**base, 'state': 'finished', 'requested_populations': 100,
        'records': records, 'utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
