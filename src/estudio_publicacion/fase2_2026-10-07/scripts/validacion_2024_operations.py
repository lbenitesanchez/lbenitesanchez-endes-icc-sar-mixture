"""Operational dispatcher only: run the unchanged frozen 2024 CLI targets.

No model, prior, initialization, seed, diagnostic or scoring decision is made
here. The root's append-only access approval and the exact temporal prediction
lock must exist before prepare/fit/lock actions. Dry-run never opens children.
"""
from pathlib import Path
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import uuid

from validacion_fit_models import ROOT, gate, sha

MAX_CHAINS = 6
HOGG_CHAINS = 4
MAX_CUT_WORKERS = 2
METHODS = {'BetaPowerExisting', 'BetaDesignUncertainty', 'HoggAgeSexSpatial'}
PREPARED = ROOT / 'datos/private_external_prepared_2024_training'
UNITS = ROOT / 'datos/private_external/children_2024_training.csv'
BASELINE = ROOT / 'resultados/externa_territorial_2024_baseline'
HOGG = ROOT / 'resultados/externa_territorial_2024_hogg'
DESIGN = ROOT / 'resultados/externa_territorial_2024_designcut8'


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def frozen_auxiliary():
    proof = json.loads((ROOT / 'protocol_freeze.json').read_text())
    candidates = [ROOT / key for key in proof['input_sha256']
                  if key.startswith('datos/geografia/aux_verified_') and key.endswith('.csv')]
    if len(candidates) != 1:
        raise RuntimeError('Expected one uniquely frozen official auxiliary CSV route')
    path = candidates[0]
    if sha(path) != proof['input_sha256'][str(path.relative_to(ROOT))]:
        raise RuntimeError('Frozen official auxiliary input changed')
    return path


def check_temporal_lock():
    protocol_sha = gate(2024)
    path = ROOT / 'validacion/externa_prediction_lock_temporal.json'
    lock = json.loads(path.read_text())
    if (lock.get('scope') != 'temporal' or not lock.get('approved_for_scoring')
            or lock.get('protocol_sha256') != protocol_sha
            or {r['method'] for r in lock['predictions']} != METHODS
            or len(lock['predictions']) != 3):
        raise RuntimeError('Exact three-method temporal prediction lock unavailable')
    for item in lock['predictions']:
        prediction = ROOT / item['path']
        status = ROOT / item['status_path']
        if sha(prediction) != item['sha256'] or sha(status) != item['status_sha256']:
            raise RuntimeError('Locked temporal prediction/status changed')
        if not json.loads(status.read_text()).get('success'):
            raise RuntimeError('Locked temporal model no longer has a successful status')
    return {'protocol_sha256': protocol_sha, 'temporal_lock_sha256': sha(path),
            'access_approval_sha256': sha(ROOT / 'validacion/externa_reserved_access_approval.json')}


def verify_preparation_classification(journal):
    path = ROOT / 'validacion/externa_reserved_preparation.json'
    proof = json.loads(path.read_text())
    required = ('WHO2024_HC57A_classification_match', 'child_keys_unique',
                'household_keys_unique', 'roster_keys_unique',
                'full_household_frame_preserved')
    if (proof.get('official_classification_compared_records', 0) <= 0
            or any(proof.get(key) is not True for key in required)):
        raise RuntimeError('Reserved parser classification/key/full-frame operational guard failed')
    journal.event('classification_and_key_guards_passed',
                  preparation_validation_sha256=sha(path),
                  official_classification_compared_records=proof['official_classification_compared_records'],
                  outcome_summaries_computed=False)


def active_phase2_fits():
    rows = subprocess.run(['/bin/ps', '-axo', 'pid,ppid,command'], check=True,
                          capture_output=True, text=True).stdout.splitlines()
    found = []
    own = os.getpid()
    for row in rows[1:]:
        fields = row.strip().split(None, 2)
        if len(fields) != 3:
            continue
        pid, ppid, command = int(fields[0]), int(fields[1]), fields[2]
        if pid == own:
            continue
        # Match existing phase2 Python fit controllers even between chains.
        python_fit = (Path(command.split(None, 1)[0]).name.startswith('python')
                      and any(token in command for token in
                      ('scripts/diseno_fit_cut.py', 'scripts/validacion_fit_models.py',
                       'scripts/validacion_fit_design_reserved.py',
                       'scripts/root_run_simulation_cohort.py', 'scripts/sim_design_diagnostics.py',
                       'scripts/validacion_2024_operations.py')))
        stan_fit = ('method=sample' in command and str(ROOT.parent) in command)
        if python_fit or stan_fit:
            found.append({'pid': pid, 'ppid': ppid, 'command': command})
    return found


def schedule_capacity(running):
    reserved = sum(item['chain_budget'] for item in running)
    cut_workers = sum(item['name'].startswith('cut_state_') for item in running)
    return reserved < MAX_CHAINS and cut_workers < MAX_CUT_WORKERS


class Journal:
    def __init__(self, action, gates):
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.path = ROOT / 'validacion/externa_2024_operations' / (stamp + '_' + uuid.uuid4().hex[:8])
        self.path.mkdir(parents=True, exist_ok=False)
        self.event('controller_started', action=action, controller_pid=os.getpid(),
                   controller_source_sha256=sha(Path(__file__)), **gates)

    def event(self, event, **fields):
        entry = {'event': event, 'time_utc': utc(), **fields}
        with (self.path / 'events.jsonl').open('a') as f:
            f.write(json.dumps(entry) + '\n')
        print(json.dumps(entry), flush=True)

    def serial(self, name, command):
        output = self.path / (name + '.log')
        self.event('serial_started', name=name, command=command, log=str(output))
        with output.open('x') as handle:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        self.event('serial_completed', name=name, exit_code=result.returncode,
                   log_sha256=sha(output))
        if result.returncode:
            raise RuntimeError('Frozen CLI failed: ' + name)


def model_command(method, output, auxiliary):
    return [sys.executable, '-u', str(ROOT / 'scripts/validacion_fit_models.py'),
            '--method', method, '--year', '2024', '--units', str(UNITS),
            '--aux', str(auxiliary), '--prepared', str(PREPARED), '--output', str(output),
            '--parallel-chains', '4' if method == 'hogg' else '1']


def state_command(state, auxiliary):
    return [sys.executable, '-u', str(ROOT / 'scripts/validacion_fit_design_reserved.py'),
            '--prepared', str(PREPARED), '--aux', str(auxiliary), '--output', str(DESIGN),
            '--state', str(state)]


def fit(journal, auxiliary):
    others = active_phase2_fits()
    if others:
        journal.event('dispatch_refused_other_phase2_fits', processes=others)
        raise RuntimeError('Root must release other phase2 fitting workers before 2024 dispatch')
    preparation = json.loads((PREPARED / 'preparation_manifest.json').read_text())
    if (preparation.get('year') != 2024 or preparation.get('scope') != 'territorial_training'
            or preparation.get('reserved_test_outcomes_read', True)):
        raise RuntimeError('2024 common training-only preparation unavailable')
    if any(path.exists() for path in (BASELINE, HOGG, DESIGN)):
        raise FileExistsError('Preserve all existing fit outputs; no automatic operational restart')
    running, finished = [], []

    def launch(name, command, chain_budget):
        if sum(x['chain_budget'] for x in running) + chain_budget > MAX_CHAINS:
            raise RuntimeError('Active reserved CmdStan chain budget would exceed six')
        log = journal.path / (name + '.log')
        handle = log.open('x')
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        item = {'name': name, 'process': process, 'chain_budget': chain_budget,
                'handle': handle, 'log': log}
        running.append(item)
        journal.event('fit_worker_started', name=name, pid=process.pid, command=command,
                      chain_budget=chain_budget,
                      total_reserved_chain_budget=sum(x['chain_budget'] for x in running),
                      log=str(log))

    # Fixed ordering requested by root: Hogg4, baseline1, CUT state1.
    launch('hogg', model_command('hogg', HOGG, auxiliary), HOGG_CHAINS)
    launch('baseline', model_command('baseline', BASELINE, auxiliary), 1)
    launch('cut_state_01', state_command(1, auxiliary), 1)
    next_state = 2
    while running or next_state <= 8:
        changed = False
        for item in list(running):
            result = item['process'].poll()
            if result is None:
                continue
            item['handle'].close()
            running.remove(item)
            finished.append({'name': item['name'], 'pid': item['process'].pid,
                             'exit_code': result, 'chain_budget': item['chain_budget'],
                             'log_sha256': sha(item['log'])})
            journal.event('fit_worker_completed', **finished[-1])
            changed = True
        # With Hogg4 and baseline1 still running there is at most one CUT
        # worker. When baseline ends, two CUT workers reserve the other slots.
        while next_state <= 8 and schedule_capacity(running):
            launch(f'cut_state_{next_state:02d}', state_command(next_state, auxiliary), 1)
            next_state += 1
            changed = True
        if not changed:
            time.sleep(2)
    hogg_result = next(x for x in finished if x['name'] == 'hogg')
    if hogg_result['exit_code'] == 1:
        # The frozen wrapper's known final JSON type error is repaired only
        # after every actual fit has completed. The helper refuses other
        # exceptions, failed diagnostics or mismatched chains/maps/inputs.
        journal.serial('recover_hogg_metadata_only', [sys.executable, '-u',
                       str(ROOT / 'scripts/validacion_recover_hogg_status.py'),
                       '--year', '2024', '--units', str(UNITS), '--aux', str(auxiliary),
                       '--output', str(HOGG), '--original-log', str(journal.path/'hogg.log')])
        hogg_result['metadata_recovered_after_original_exit_1'] = True
        journal.event('hogg_metadata_recovery_completed', original_wrapper_exit_code=1,
                      recovered_status_sha256=sha(HOGG/'status_recovered.json'),
                      recovery_manifest_sha256=sha(HOGG/'serialization_recovery_manifest.json'),
                      frozen_scientific_sources_modified=False, model_refitted=False)
    success = all(x['exit_code'] == 0 or x.get('metadata_recovered_after_original_exit_1', False)
                  for x in finished)
    journal.event('all_prescribed_workers_completed', workers=finished, success=success)
    if not success:
        raise RuntimeError('At least one prescribed model/state failed; preserve failure and withhold locks/scores')
    journal.serial('cut_collect', state_command('collect', auxiliary))


def lock(journal):
    hogg_status = HOGG / ('status.json' if (HOGG/'status.json').exists() else 'status_recovered.json')
    specifications = [
        {'method': 'BetaPowerExisting', 'path': str(BASELINE.relative_to(ROOT) / 'predictions.csv'),
         'status_path': str(BASELINE.relative_to(ROOT) / 'status.json')},
        {'method': 'BetaDesignUncertainty', 'path': str(DESIGN.relative_to(ROOT) / 'predictions.csv'),
         'status_path': str(DESIGN.relative_to(ROOT) / 'completion.json')},
        {'method': 'HoggAgeSexSpatial', 'path': str(HOGG.relative_to(ROOT) / 'predictions.csv'),
         'status_path': str(hogg_status.relative_to(ROOT))},
    ]
    path = journal.path / 'territorial_prediction_specifications.json'
    path.write_text(json.dumps(specifications, indent=2))
    journal.serial('territorial_prediction_lock', [sys.executable, '-u',
                   str(ROOT / 'scripts/validacion_lock_predictions.py'), '--scope', 'territorial',
                   '--specifications', str(path)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['prepare', 'fit', 'lock'], default='fit')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        gate(2019)
        print(json.dumps({'status': 'DRY_RUN_RESERVED_RESPONSES_UNREAD',
              'auxiliary': str(frozen_auxiliary()), 'prepared': str(PREPARED),
              'outputs': [str(x) for x in (BASELINE, HOGG, DESIGN)],
              'maximum_active_reserved_chains': MAX_CHAINS,
              'hogg_parallel_chains': HOGG_CHAINS, 'baseline_parallel_chains': 1,
              'maximum_serial_cut_state_workers': MAX_CUT_WORKERS,
              'initial_order': ['hogg4', 'baseline1', 'cut_state01'],
              'scientific_sources_modified': False, 'fits_dispatched': False}), flush=True)
        return
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    gates = check_temporal_lock()
    journal = Journal(args.action, gates)
    auxiliary = frozen_auxiliary()
    try:
        if args.action == 'prepare':
            journal.serial('extract_and_parse_reserved', [sys.executable, '-u',
                           str(ROOT / 'scripts/validacion_prepare_reserved.py')])
            verify_preparation_classification(journal)
            journal.serial('prepare_training_only', [sys.executable, '-u',
                           str(ROOT / 'scripts/validacion_prepare_training.py'), '--year', '2024',
                           '--aux', str(auxiliary), '--output', str(PREPARED)])
        elif args.action == 'fit':
            fit(journal, auxiliary)
        else:
            lock(journal)
        journal.event('controller_completed', action=args.action, success=True,
                      scores_computed=False)
    except Exception as error:
        journal.event('controller_failed', action=args.action, success=False,
                      error_type=type(error).__name__, error=str(error), scores_computed=False)
        raise


if __name__ == '__main__':
    main()
