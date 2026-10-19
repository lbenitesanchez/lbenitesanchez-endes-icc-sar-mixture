#!/usr/bin/env python3
"""Transfer two released candidate slots to Hogg, only after real v2 closure.

Runtime orchestration only: no truth, predictions, scores, fits, seeds, sampler
settings or statistical sources are read or changed. One exclusive observer
may launch lane04/lane05; Hogg's existing O_EXCL shard claims remain authoritative.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pandas as pd

from benchmark_inputs import ROOT, OUT, sha256

TAGS = ('final_primary_v2', 'final_structural_v2')
LANES = ('lane04', 'lane05')
DESTINATION = OUT / 'dispatch_logs/candidate_to_hogg'
PLAN = ROOT / 'resultados/competidores_hogg/final_source_20261005/shard_plan.csv'
HOGG_QUEUE = ROOT / 'scripts/competitors/hogg_queue.py'


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def process_table():
    result = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,lstart=,command='], text=True)
    records = {}
    for line in result.splitlines():
        fields = line.split(None, 7)
        if len(fields) != 8:
            continue
        pid = int(fields[0])
        records[pid] = dict(pid=pid, ppid=int(fields[1]), started=' '.join(fields[2:7]),
                            command=fields[7])
    return records


def candidate_root(record):
    words = record['command'].split()
    budget_names = {str(OUT / 'candidate_runtime_budget.json'),
                    'resultados/resultados_comparadores/candidate_runtime_budget.json',
                    'estudio_publicacion/resultados/resultados_comparadores/candidate_runtime_budget.json'}
    def owns_budget(option):
        if option not in words:
            return False
        at = words.index(option) + 1
        return at < len(words) and words[at] in budget_names
    if len(words) > 1:
        program = Path(words[1]).name
        if program == 'benchmark_schedule_candidate.py' and '--numerical-version' in words and owns_budget('--budget-file'):
            at = words.index('--numerical-version') + 1
            return at < len(words) and words[at] == '2'
        if program == 'benchmark_run.py' and '--tag' in words and owns_budget('--worker-limit-file'):
            at = words.index('--tag') + 1
            return at < len(words) and words[at] in TAGS
    executable = words[0] if words else ''
    return any(executable.startswith(str(OUT / 'source_snapshots' / tag) + '/') for tag in TAGS)


def track_candidate(table, known):
    selected = {pid for pid, record in table.items() if candidate_root(record)}
    selected.update(pid for pid, record in table.items()
                    if (pid, record['started']) in known)
    while True:
        children = {pid for pid, record in table.items() if record['ppid'] in selected}
        before = len(selected)
        selected.update(children)
        if len(selected) == before:
            break
    live = []
    for pid in sorted(selected):
        record = table[pid]
        key = (pid, record['started'])
        words = record['command'].split()
        program = Path(words[1]).name if len(words) > 1 else Path(words[0]).name
        if words and any(words[0].startswith(str(OUT / 'source_snapshots' / tag) + '/') for tag in TAGS):
            program = Path(words[0]).name
        if program == '-c':
            program = 'multiprocessing_tracker_or_worker'
        safe = {key: record[key] for key in ('pid', 'ppid', 'started')}
        safe['program'] = program
        known[key] = safe
        live.append(safe)
    return live


def completion_errors(queue, completions, configs, protocol):
    errors = []
    if not queue or queue.get('state') != 'completed' or queue.get('numerical_version') != 2 or queue.get('completed_populations') != 800:
        errors.append('candidate_v2_controller_not_completed_800')
    expected = protocol['independent_seed_ids']
    for tag, count, scenarios in (
        (TAGS[0], 600, ['binary_informative_mar', 'sparse_srs_2025', 'sparse_psu_2025']),
        (TAGS[1], 200, ['sparse_srs_iid', 'sparse_psu_bym2'])):
        completion, config = completions.get(tag), configs.get(tag)
        if not completion or not completion.get('completed') or completion.get('populations') != count:
            errors.append(tag + '_not_completed_all_requested_populations')
        if not config or config.get('scenarios') != scenarios:
            errors.append(tag + '_configuration_missing_or_changed')
        elif any(list(config.get('replications', [])) != list(expected[scenario]) for scenario in scenarios):
            errors.append(tag + '_seed_ids_changed')
    return errors


def frozen_checks():
    manifest = load(OUT / 'source_snapshots/numerical_v2_master/source_manifest.json')
    for name, expected in manifest['source_sha256'].items():
        path = ROOT / 'scripts' / name if name == 'models.py' else ROOT / 'scripts/competitors' / name
        if sha256(path) != expected:
            raise ValueError('Candidate statistical source differs: ' + name)
    if sha256(OUT / 'protocol_freeze.json') != manifest['protocol_sha256']:
        raise ValueError('Approved candidate protocol differs')
    amendment = load(OUT / 'numerical_amendment_v2.json')
    if not amendment.get('approved_for_dispatch') or amendment.get('statistical_configuration_changed') is not False:
        raise ValueError('Numerical v2 amendment is not approved')
    snapshot = PLAN.parent / 'source_snapshot/snapshot_manifest.json'
    frozen = load(snapshot)
    runner = ROOT / 'scripts/competitors/hogg_benchmark_batch.py'
    if sha256(runner) != frozen['source_sha256']['hogg_benchmark_batch.py']:
        raise ValueError('Hogg dispatcher differs from immutable source snapshot')
    frame = pd.read_csv(PLAN)
    if len(frame) != 22 or int(frame.tasks.sum()) != 1100 or frame.tag.duplicated().any():
        raise ValueError('Hogg plan must preserve22 disjoint shards/1100 existing fits')
    for row in frame.to_dict('records'):
        if sha256(Path(row['task_csv'])) != row['sha256']:
            raise ValueError('Hogg shard identities differ')
    return dict(candidate_sources=manifest['source_sha256'],
                protocol_sha256=manifest['protocol_sha256'],
                numerical_amendment_sha256=sha256(OUT / 'numerical_amendment_v2.json'),
                hogg_queue_sha256=sha256(HOGG_QUEUE), hogg_plan_sha256=sha256(PLAN),
                hogg_source_snapshot_manifest_sha256=sha256(snapshot))


def existing_lanes(table):
    present = []
    for record in table.values():
        words = record['command'].split()
        if len(words) > 1 and Path(words[1]).name == 'hogg_queue.py' and '--lane' in words:
            at = words.index('--lane') + 1
            if at < len(words) and words[at] in LANES:
                present.append(dict(pid=record['pid'], lane=words[at]))
    state = PLAN.parent / 'queue_dispatch'
    for lane in LANES:
        if (state / (lane + '.summary.json')).exists():
            present.append(dict(lane=lane, evidence='existing_lane_summary'))
    for path in state.glob('*.claim.json'):
        claim = load(path)
        if claim and claim.get('lane') in LANES:
            present.append(dict(lane=claim['lane'], evidence=path.name))
    return present


def self_check():
    protocol = dict(independent_seed_ids={scenario: list(range(10000, 10200))
        for scenario in ['binary_informative_mar', 'sparse_srs_2025', 'sparse_psu_2025']})
    protocol['independent_seed_ids'].update({scenario: list(range(20000, 20100))
        for scenario in ['sparse_srs_iid', 'sparse_psu_bym2']})
    queue = dict(state='completed', numerical_version=2, completed_populations=800)
    completions = {TAGS[0]: dict(completed=True, populations=600, failures=1),
                   TAGS[1]: dict(completed=True, populations=200, failures=0)}
    configs = {TAGS[0]: dict(scenarios=list(protocol['independent_seed_ids'])[:3], replications=list(range(10000, 10200))),
               TAGS[1]: dict(scenarios=list(protocol['independent_seed_ids'])[3:], replications=list(range(20000, 20100)))}
    assert not completion_errors(queue, completions, configs, protocol)
    assert completion_errors(dict(queue, state='running'), completions, configs, protocol)
    assert completion_errors(queue, {TAGS[0]: completions[TAGS[0]]}, configs, protocol)
    changed = dict(configs)
    changed[TAGS[1]] = dict(configs[TAGS[1]], replications=[20000])
    assert completion_errors(queue, completions, changed, protocol)
    known = {}
    table = {10: dict(pid=10, ppid=1, started='original', command='python benchmark_schedule_candidate.py --numerical-version 2 --budget-file resultados/resultados_comparadores/candidate_runtime_budget.json'),
             11: dict(pid=11, ppid=10, started='worker', command='python -c worker'),
             12: dict(pid=12, ppid=1, started='unrelated', command='python other_project.py')}
    assert len(track_candidate(table, known)) == 2
    assert len(track_candidate({11: dict(table[11], ppid=1)}, known)) == 1
    assert not track_candidate({10: dict(table[10], started='reused', command='python unrelated.py')}, known)
    with tempfile.TemporaryDirectory() as temporary:
        lease = Path(temporary) / 'exclusive.json'
        descriptor = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(descriptor)
        try:
            descriptor = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
            raise AssertionError('Duplicate lease was admitted')
    return dict(passed=True, partial_completion_rejected=True, controller_and_orphan_worker_closure_required=True,
                pid_reuse_distinguished_by_start_time=True, unrelated_projects_ignored=True,
                explicit_completed_failures_preserved=True, duplicate_observer_lease_rejected=True,
                no_real_process_launched=True, no_truth_or_performance_read=True,
                source_sha256=sha256(Path(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if args.self_check:
        proof = self_check()
        path = OUT / 'validation/candidate_to_hogg_observer_qa.json'
        write(path, proof)
        print(json.dumps(proof, indent=2))
        return
    if not 10 <= args.poll_seconds <= 60:
        raise ValueError('Use a10..60second observation interval')
    DESTINATION.mkdir(parents=True, exist_ok=True)
    lease = DESTINATION / 'lease.json'
    descriptor = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, 'w') as handle:
        json.dump(dict(observer_pid=os.getpid(), created_utc=now(), lanes=list(LANES),
                       source_sha256=sha256(Path(__file__)), authorization='Root explicit transfer of two candidate slots after800 v2/real closure'), handle, indent=2)
    frozen = frozen_checks()
    protocol = load(OUT / 'protocol_freeze.json')
    budget_path = OUT / 'candidate_runtime_budget.json'
    if int(load(budget_path)['active_workers']) != 2:
        raise ValueError('The authorized transfer is exactly the current two candidate slots')
    known = {}
    closed_observations = 0
    state = dict(created_utc=now(), state='waiting_for800_v2_and_process_closure', lanes=list(LANES),
                 released_candidate_slots=2, allocated_hogg_slots=2, global_ENDES_budget=8,
                 additional_fits=0, replacement_seed_ids=0, statistical_sources_changed=False,
                 truth_or_performance_read=False, provenance=frozen)
    while True:
        table = process_table()
        live = track_candidate(table, known)
        errors = completion_errors(load(OUT / 'candidate_queue_manifest.json'),
            {tag: load(OUT / tag / 'completion.json') for tag in TAGS},
            {tag: load(OUT / tag / 'config.json') for tag in TAGS}, protocol)
        if any((OUT / tag / 'invalidated.json').exists() for tag in TAGS):
            raise ValueError('A canonical v2 run was invalidated; no resource transfer')
        closed_observations = closed_observations + 1 if not errors and not live else 0
        state.update(checked_utc=now(), pending_completion_guards=errors,
                     live_candidate_processes=live, tracked_candidate_process_identities=len(known),
                     consecutive_closed_process_observations=closed_observations)
        write(DESTINATION / 'observer.json', state)
        write(DESTINATION / 'candidate_process_identities.json', list(known.values()))
        if closed_observations >= 2:
            break
        time.sleep(args.poll_seconds)
    if frozen_checks() != frozen:
        raise ValueError('Source/plan provenance changed while waiting')
    if track_candidate(process_table(), known):
        raise ValueError('Candidate processes reappeared; no handoff')
    if existing_lanes(process_table()):
        raise ValueError('Lane04/lane05 already exist; never duplicate a dispatcher')
    budget = load(budget_path)
    if int(budget['active_workers']) != 2:
        raise ValueError('Current runtime allocation no longer provides the two authorized slots')
    state.update(candidate_closure_confirmed_utc=now(), candidate_completed_populations=800,
        candidate_failure_log_bytes=sum((OUT / tag / 'failures.jsonl').stat().st_size for tag in TAGS),
        candidate_completion_sha256={tag: sha256(OUT / tag / 'completion.json') for tag in TAGS})
    budget.update(active_workers=0, updated_utc=now(),
        reason='Completed800 candidate v2 and confirmed real controller/worker closure; root-approved transfer of both slots to Hogg lane04/lane05 within total8')
    write(budget_path, budget)
    with (OUT / 'candidate_runtime_budget_history.jsonl').open('a') as handle:
        handle.write(json.dumps(budget) + '\n')
    launches = []
    environment = os.environ.copy()
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        environment[variable] = '1'
    for lane in LANES:
        command = [sys.executable, str(HOGG_QUEUE), '--lane', lane, '--plan', str(PLAN)]
        log = DESTINATION / (lane + '.log')
        with log.open('x') as handle:
            child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        launches.append(dict(lane=lane, queue_pid=child.pid, launched_utc=now(),
                             log=str(log.relative_to(ROOT)), command=command))
        state.update(state='dispatching_released_slots', launches=launches)
        write(DESTINATION / 'observer.json', state)
    state.update(state='dispatch_completed', transfer_completed=True, launched_lanes=2,
                 hogg_fit_completion_not_asserted=True, finished_utc=now())
    write(DESTINATION / 'observer.json', state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == '__main__':
    main()
