#!/usr/bin/env python3
"""Fit sampled-graph worlds with the frozen pre-closure two-worker budget.

This optional operational dispatcher leaves run_all.py untouched. It yields
new dispatches well before that controller reaches the sampled-graph stage.
It never reads population truth or changes any fitting seed or model setting.
"""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--yield-at-primary', type=int, default=200)
    a = p.parse_args()
    if not 1 <= a.yield_at_primary <= 200:
        p.error('Yield no later than 200 primary closures to avoid controller overlap')
    cfg = json.loads(a.config.read_text())
    base, run = Path(cfg['run_base']).resolve(), Path(cfg['sikov_run_root']).resolve()
    if base.is_relative_to(HERE.parent) or run.is_relative_to(HERE.parent):
        raise ValueError('Private execution must remain outside the repository')
    ledger = base / 'sampled_prefit'
    if ledger.exists():
        raise FileExistsError('Preserve the existing dispatch ledger: ' + str(ledger))
    ledger.mkdir(parents=True)
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    env['SDKROOT'] = cfg['sdkroot']
    env['MACOSX_DEPLOYMENT_TARGET'] = cfg['deployment_target']
    common = ['--source-root', cfg['source_root'], '--reference-root', cfg['reference_root'],
              '--run-root', str(run), '--exe', cfg['beta_exe'], '--cmdstan', cfg['cmdstan']]
    state = dict(status='RUNNING', workers=2, cohort='sampled_graph', expected_worlds=64,
                 accuracy_or_truth_read=False, kernel_or_seed_changes=False,
                 adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 yield_at_primary=a.yield_at_primary, commands=[])

    def primary_closed():
        progress = json.loads((base / 'progress.json').read_text())
        return int(progress.get('completed', {}).get('eval-one', 0))

    def command(rep):
        cmd = [cfg['python_new'], str(HERE / 'sikov_worker.py'), '--stage', 'sampled-one',
               *common, '--rep', str(rep)]
        start = time.time(); log = ledger / f'rep_{rep}.log'
        with log.open('x') as out:
            result = subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT)
        record = dict(replication=rep, command=cmd, exit_code=result.returncode,
                      elapsed_seconds=time.time()-start,
                      log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                      finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        save(ledger / f'rep_{rep}.json', record)
        return record

    pending = iter(range(30000, 30064)); active = {}; exhausted = False; yielding = False
    save(ledger / 'progress.json', state)
    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        while active or not exhausted:
            count = primary_closed()
            if count >= a.yield_at_primary:
                yielding = True; exhausted = True
            while not exhausted and len(active) < 2:
                rep = next(pending, None)
                if rep is None:
                    exhausted = True; break
                folder = run / 'sensibilidad_sampled_graph/fits' / f'rep_{rep:05d}'
                if folder.exists():
                    raise FileExistsError('Unexpected existing sampled fit: ' + str(folder))
                active[pool.submit(command, rep)] = rep
            if not active:
                break
            done, _ = futures.wait(active, timeout=10, return_when=futures.FIRST_COMPLETED)
            for future in done:
                active.pop(future)
                record = future.result(); state['commands'].append(record)
                # A failure is preserved. Stop new jobs without omitting it.
                if record['exit_code']:
                    exhausted = True; state['failure_retained'] = True
                print(json.dumps(record), flush=True)
            state['primary_worlds_closed'] = count
            state['active_replications'] = sorted(active.values())
            save(ledger / 'progress.json', state)
    state['status'] = ('FAILED_OUTPUTS_RETAINED' if state.get('failure_retained') else
                       'YIELDED_TO_PRIMARY_CONTROLLER' if yielding else 'ALL_64_NEW_FITS_CLOSED')
    state['completed_worlds'] = len(state['commands'])
    state['updated_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    save(ledger / 'progress.json', state)
    print(json.dumps({k: state[k] for k in ('status', 'completed_worlds', 'updated_utc')}), flush=True)
    if state.get('failure_retained'):
        raise RuntimeError('Investigate the retained sampled-graph failure before continuing')


if __name__ == '__main__':
    main()
