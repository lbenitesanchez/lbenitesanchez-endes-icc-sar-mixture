#!/usr/bin/env python3
"""Read-only execution progress; deliberately never opens truth or performance.

This observer does not dispatch tasks, fit models, change frozen sources or
decide statistical eligibility. It records only artifacts, gates and hashes.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pandas as pd

from benchmark_inputs import ROOT, OUT, sha256


def readable_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def artifact_counts(task_file):
    if not task_file.exists():
        return None
    tasks = pd.read_csv(task_file).to_dict('records')
    artifacts = failures = 0
    failure_files = ('failure.json', 'execution_failure.json', 'worker_failure.json')
    for task in tasks:
        folder = Path(task['output_dir'])
        style = task.get('diagnostic_style', 'hogg')
        status = 'diagnostics.json' if style == 'gao' else 'status.json'
        if (folder / 'predictions.csv').exists() and readable_json(folder / status) is not None:
            artifacts += 1
        elif any(readable_json(folder / name) is not None for name in failure_files):
            failures += 1
    return dict(requested=len(tasks), completed_prediction_artifacts=artifacts,
                explicit_execution_failures=failures,
                accounted=artifacts + failures, pending=len(tasks) - artifacts - failures)


def candidate_counts():
    planned = readable_json(OUT / 'protocol_freeze.json')['independent_seed_ids']
    planned_keys = {(scenario, int(rep)) for scenario, ids in planned.items() for rep in ids}
    statuses = []
    racing = []
    observed = set()
    per_scenario = {scenario: 0 for scenario in planned}
    failure_bytes = 0
    for tag in ('final_primary_v2', 'final_structural_v2'):
        base = OUT / tag
        failure_log = base / 'failures.jsonl'
        if failure_log.exists():
            failure_bytes += failure_log.stat().st_size
        for path in (base / 'fit_results/BetaPowerHMCGraph').glob('*/rep_*/status.json'):
            status = readable_json(path)
            if status is None:
                racing.append(str(path.relative_to(OUT)))
                continue
            key = (path.parent.parent.name, int(path.parent.name.removeprefix('rep_')))
            if key not in planned_keys or key in observed:
                raise ValueError('Unplanned or duplicated candidate execution artifact: ' + str(key))
            observed.add(key)
            per_scenario[key[0]] += 1
            statuses.append(status)
    trials = [endpoint['sampler_trials'][-1]
              for status in statuses for endpoint in status['endpoints']]
    all_trials = [trial for status in statuses for endpoint in status['endpoints']
                  for trial in endpoint['sampler_trials']]
    gates = None
    if trials:
        gates = dict(endpoints=len(trials), all_final_gates_pass=all(t['success'] for t in trials),
                     all_finite_diagnostics=all(t['finite_diagnostics'] for t in trials),
                     max_Rhat=max(t['max_Rhat'] for t in trials),
                     min_bulk_ESS=min(t['min_bulk_ESS'] for t in trials),
                     min_tail_ESS=min(t['min_tail_ESS'] for t in trials),
                     min_E_BFMI=min(min(t['E_BFMI']) for t in trials),
                     final_divergences=sum(t['divergences'] for t in trials),
                     final_depth_hits=sum(t['max_treedepth_hits'] for t in trials),
                     all_attempt_divergences=sum(t['divergences'] for t in all_trials),
                     all_attempt_depth_hits=sum(t['max_treedepth_hits'] for t in all_trials),
                     retained_attempts=len(all_trials),
                     retried_endpoints=sum(len(e['sampler_trials']) > 1
                         for s in statuses for e in s['endpoints']),
                     beta_shape_guard_cells=sum(e.get('beta_shape_guard_cells', 0)
                         for s in statuses for e in s['endpoints']))
    return dict(requested_populations=len(planned_keys), completed_status_populations=len(statuses),
                passed_status_populations=sum(s['success'] for s in statuses),
                completed_by_scenario=per_scenario, pending_populations=len(planned_keys) - len(statuses),
                execution_failure_log_bytes=failure_bytes,
                incomplete_status_reads=racing, numerical_diagnostics=gates)


def progress():
    manifest = readable_json(OUT / 'source_snapshots/numerical_v2_master/source_manifest.json')
    source_checks = {}
    for name, expected in manifest['source_sha256'].items():
        path = ROOT / 'scripts' / name if name == 'models.py' else ROOT / 'scripts/competitors' / name
        source_checks[name] = sha256(path) == expected
    return dict(checked_utc=datetime.now(timezone.utc).isoformat(),
                scope='Execution artifacts, sampler gates and provenance only; no truth, predictions or performance examined',
                candidate=candidate_counts(),
                external=dict(gao=artifact_counts(OUT / 'task_plan/gao_all1800.csv'),
                              hogg=artifact_counts(OUT / 'task_plan/hogg_collector1100.csv'),
                              hogg_spatial=artifact_counts(OUT / 'task_plan/hogg_spatial100.csv')),
                all16_frozen_statistical_sources_match=all(source_checks.values()),
                source_checks=source_checks,
                runtime_budget=readable_json(OUT / 'candidate_runtime_budget.json'),
                main_evaluation_completed=(OUT / 'final_evaluation/final_completion.json').exists(),
                spatial_evaluation_completed=(OUT / 'spatial_evaluation/completion.json').exists(),
                observer_source_sha256=sha256(Path(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--watch', action='store_true', help='Observe until both evaluations finish; never dispatch fits')
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    if not 30 <= args.poll_seconds <= 60:
        raise ValueError('Use a low-frequency observer with at most60 seconds per wait')
    while True:
        record = progress()
        if args.save:
            temporary = OUT / 'runtime_progress.tmp'
            temporary.write_text(json.dumps(record, indent=2))
            temporary.replace(OUT / 'runtime_progress.json')
            with (OUT / 'runtime_progress_history.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
        print(json.dumps({key: value for key, value in record.items() if key != 'source_checks'}, indent=2), flush=True)
        if not args.watch or (record['main_evaluation_completed'] and record['spatial_evaluation_completed']):
            break
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
