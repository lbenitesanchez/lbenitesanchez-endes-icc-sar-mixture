"""Return the same released spatial lane to the unchanged generic queue."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'resultados/competidores_hogg'


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def all_spatial_accounted(state):
    if not isinstance(state, dict) or state.get('state') != 'finished' or state.get('requested_populations') != 100:
        return False
    records = state.get('records')
    if not isinstance(records, list) or len(records) != 4:
        return False
    if any(not isinstance(record, dict) for record in records):
        return False
    expected = {f'spatial_sensitivity_{i:02d}' for i in range(4)}
    if {record.get('tag') for record in records} != expected:
        return False
    for record in records:
        if record.get('state') == 'already_accounted_for':
            continue
        if type(record.get('returncode')) is not int or record['returncode'] != 0 or record.get('accounted_for') is not True:
            return False
    for i in range(4):
        start = 20000 + 25 * i
        manifest = read_json(RESULTS / f'spatial_sensitivity_{i:02d}/batch_manifest.json')
        if not isinstance(manifest, dict):
            return False
        if manifest.get('requested_tasks') != 25 or manifest.get('accounted_for') != 25:
            return False
        if manifest.get('population_IDs') != list(range(start, start + 25)):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poll-seconds', type=float, default=15)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError('Positive polling interval required')
    claim = RESULTS / 'generic_after_spatial_observer.claim.json'
    descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with os.fdopen(descriptor, 'w') as stream:
        json.dump({'observer_pid': os.getpid(), 'source_sha256': source_hash,
                   'created_utc': datetime.now(timezone.utc).isoformat()}, stream, indent=2)
    state_path = RESULTS / 'generic_after_spatial_observer.json'
    base = {'observer_pid': os.getpid(), 'source_sha256': source_hash, 'lane': 'lane01',
            'authorization': 'Parent: return same spatial lane to generic queue after all four fixed chunks; remove only pause_lane01. No additional concurrent fitting lane.',
            'data_access': 'Execution accounting only; no truth, scores or convergence-based selection.'}
    while not all_spatial_accounted(read_json(RESULTS / 'spatial_after_application_observer.json')):
        state_path.write_text(json.dumps({**base, 'state': 'waiting_for_all100_spatial_accounting',
            'utc': datetime.now(timezone.utc).isoformat()}, indent=2))
        time.sleep(args.poll_seconds)
    pause = RESULTS / 'final_source_20261005/queue_dispatch/pause_lane01'
    removed = pause.exists()
    if removed:
        pause.unlink()
    state_path.write_text(json.dumps({**base, 'state': 'running_generic_queue',
        'removed_only_pause_lane01': removed, 'utc': datetime.now(timezone.utc).isoformat()}, indent=2))
    with (RESULTS / 'final_hogg_lane01_resumed_after_spatial.log').open('w') as stream:
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/competitors/hogg_queue.py'),
            '--lane', 'lane01'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    state_path.write_text(json.dumps({**base, 'state': 'finished', 'returncode': result.returncode,
        'removed_only_pause_lane01': removed, 'utc': datetime.now(timezone.utc).isoformat()}, indent=2))
    if result.returncode:
        raise RuntimeError('Generic queue requires audit; no completed task is replaced')


if __name__ == '__main__':
    main()
