"""Read-only audit of the fixed 1,100 Hogg tasks, without truth or scores.

This reader checks accounting, numerical diagnostics and source/input provenance.
It does not fit, retry, replace a seed, or read predictions/posterior draws.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'resultados/competidores_hogg'
PLAN_ROOT = RESULTS / 'final_source_20261005'
SOURCE = PLAN_ROOT / 'source_snapshot'
EXCLUDED = ('lp__', 'eta[', 'effect_area[', 'linking_mean[', 'sampling_sd[',
            'log_sampling_sd_mean[', 'observation_variance[', 'theta[', 'prevalence[')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def ready(plan):
    for shard in plan:
        manifest = read_json(RESULTS / shard['tag'] / 'batch_manifest.json')
        if (not manifest or not manifest.get('all_requested_tasks_accounted_for')
                or manifest.get('completed_tasks') != int(shard['tasks'])):
            return False
    return True


def check_stage(path, diagnostic):
    rows = read_csv(path)
    name = next(iter(rows[0]))
    values = [[float(row[k]) if row[k].strip() else math.nan for k in ('R_hat', 'ESS_bulk', 'ESS_tail')]
              for row in rows if not row[name].startswith(EXCLUDED)]
    assert values
    finite = all(math.isfinite(x) for row in values for x in row)
    # Match the retained pandas diagnostic reductions, which skip NaN values.
    # A NaN in any parameter still makes the convergence gate fail below.
    def reduce_column(index, operation):
        column = [r[index] for r in values if not math.isnan(r[index])]
        return operation(column) if column else math.nan
    extrema = [reduce_column(0, max), reduce_column(1, min), reduce_column(2, min)]
    recorded = [diagnostic['max_parameter_Rhat'], diagnostic['min_parameter_ESS_bulk'],
                diagnostic['min_parameter_ESS_tail']]
    assert len(values) == diagnostic['sampled_parameter_count']
    assert diagnostic['all_parameter_diagnostics_finite'] == finite
    def matches_json_number(actual, recorded_value):
        # The sampler's JSON encoder maps both NaN and +/-Inf to null.
        if not math.isfinite(actual):
            return recorded_value is None or (not math.isfinite(recorded_value)
                and ((math.isnan(actual) and math.isnan(recorded_value)) or actual == recorded_value))
        return recorded_value is not None and math.isfinite(recorded_value) and math.isclose(
            actual, recorded_value, rel_tol=1e-11, abs_tol=1e-12)
    assert all(matches_json_number(a, b) for a, b in zip(extrema, recorded))
    bfmi = diagnostic['BFMI_per_chain']
    assert len(bfmi) == 4
    bfmi_finite = all(v is not None and math.isfinite(v) for v in bfmi)
    bfmi_min = min(bfmi) if bfmi_finite else math.nan
    passed = (finite and bfmi_finite and extrema[0] <= 1.01 and extrema[1] >= 400 and extrema[2] >= 400
              and bfmi_min >= .3 and diagnostic['divergences'] == 0
              and diagnostic['max_treedepth_hits'] == 0)
    assert passed == diagnostic['success']
    return extrema + [bfmi_min]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    plan = read_csv(PLAN_ROOT / 'shard_plan.csv')
    while not ready(plan):
        if not args.wait:
            raise FileNotFoundError('The fixed 1,100-task execution is still incomplete')
        time.sleep(30)
    frozen = read_json(SOURCE / 'snapshot_manifest.json')
    assert all(sha(SOURCE / name) == expected for name, expected in frozen['source_sha256'].items())
    assert all(sha(RESULTS / name) == expected
               for name, expected in frozen['core_scientific_frozen_records'].items())
    keys = set()
    output_directories = set()
    counts = Counter()
    final_diagnostics = []
    failures = []
    retries = 0
    for shard in plan:
        batch = RESULTS / shard['tag']
        requested = read_csv(batch / 'requested_tasks.csv')
        assert sha(Path(shard['task_csv'])) == sha(batch / 'requested_tasks.csv') == shard['sha256']
        contract = read_csv(batch / 'task_output_contract.csv')
        assert len(contract) == len(requested) == int(shard['tasks']) == 50
        assert {(r['arm'], r['scenario'], int(r['replication'])) for r in requested} == {
            (r['arm'], r['scenario'], int(r['replication'])) for r in contract}
        for task in contract:
            key = (task['arm'], task['scenario'], int(task['replication']))
            assert key not in keys
            keys.add(key)
            counts['requested/' + '/'.join(key[:2])] += 1
            output = Path(task['output'])
            output_directories.add(str(output.resolve()))
            execution = read_json(output / 'task_execution.json') or read_json(output / 'worker_failure.json')
            assert execution is not None, str(output)
            assert (execution['arm'], execution['scenario'], int(execution['replication'])) == key
            inputs = ROOT / 'resultados/resultados_comparadores/inputs' / key[1] / f'rep_{key[2]:05d}' / 'model_inputs'
            if 'source_snapshot_manifest_sha256' in execution:
                assert execution['source_snapshot_manifest_sha256'] == sha(SOURCE / 'snapshot_manifest.json')
                assert execution['source_python_sha256'] == frozen['source_sha256']['hogg_tsln.py']
            if 'input_sha256' in execution:
                assert set(execution['input_sha256']) == {'sample.csv', 'area.csv', 'direct.npz'}
                assert all(sha(inputs / name) == expected for name, expected in execution['input_sha256'].items())
            if not execution.get('completed'):
                assert (output / 'execution_failure.json').exists() or (output / 'worker_failure.json').exists()
                ineligible = (key[0] == 'design_PSU' and execution.get('error_type') == 'ValueError'
                    and execution.get('error') == 'Fewer than3 reliable stable design-variance domains; no silently prior-only GVF')
                failures.append({'arm': key[0], 'scenario': key[1], 'replication': key[2],
                    'category': 'prespecified_design_ineligibility' if ineligible else 'fitting_or_worker_execution_failure',
                    'error_type': execution.get('error_type'), 'error': execution.get('error'),
                    'available_input_and_source_metadata_verified': 'input_sha256' in execution and 'source_snapshot_manifest_sha256' in execution})
                counts[('ineligible/' if ineligible else 'execution_failures/') + '/'.join(key[:2])] += 1
                continue
            assert 'source_snapshot_manifest_sha256' in execution and 'input_sha256' in execution
            status = read_json(output / 'status.json')
            assert status['completed'] and execution['success'] == status['success']
            assert status['success'] == all(status['outcomes'][endpoint]['success'] for endpoint in ('anemia', 'stunting'))
            counts['completed/' + '/'.join(key[:2])] += 1
            counts[('converged/' if status['success'] else 'diagnostic_failures/') + '/'.join(key[:2])] += 1
            for j, endpoint in enumerate(('anemia', 'stunting')):
                attempts = status['all_attempts'][endpoint]
                assert 1 <= len(attempts) <= 2 and status['outcomes'][endpoint] == attempts[-1]
                assert len(attempts) == 1 or attempts[0]['success'] is False
                retries += len(attempts) - 1
                for number, attempt in enumerate(attempts):
                    cfg = attempt['config']
                    assert cfg['seed'] == 20261005 + key[2] + 100*j
                    assert cfg['chains'] == 4 and cfg['parallel_chains'] == 1 and cfg['max_treedepth'] == 12
                    assert cfg['min_ess'] == 400 and cfg['model_rhat_limit'] == 1.01
                    assert (cfg['warmup'], cfg['samples'], cfg['adapt_delta']) == ((500,1000,.95) if number == 0 else (1000,2000,.99))
                    assert all(expected == frozen['source_sha256'][name]
                               for name, expected in attempt['source_sha256'].items())
                    measured = [check_stage(output / endpoint / f'attempt_{number}' / f'{stage}_parameter_summary.csv', attempt[stage])
                                for stage in ('stage1', 'stage2')]
                    assert attempt['success'] == (attempt['stage1']['success'] and attempt['stage2']['success'])
                    if number == len(attempts)-1:
                        for stage, values in zip(('stage1', 'stage2'), measured):
                            metric_names = ('Rhat', 'ESS_bulk', 'ESS_tail', 'BFMI_min')
                            final_diagnostics.append(dict(arm=key[0], scenario=key[1], replication=key[2],
                                endpoint=endpoint, stage=stage, attempt=number, success=attempt[stage]['success'],
                                all_parameter_diagnostics_finite=attempt[stage]['all_parameter_diagnostics_finite'],
                                nonfinite_metric_columns=';'.join(name for name, value in zip(metric_names, values) if not math.isfinite(value)),
                                **{name: value if math.isfinite(value) else None for name, value in zip(metric_names, values)}))
    assert len(keys) == 1100
    expected = read_csv(ROOT / 'resultados/resultados_comparadores/task_plan/hogg_collector1100.csv')
    assert len(expected) == 1100
    assert keys == {(r['arm'], r['scenario'], int(r['replication'])) for r in expected}
    assert output_directories == {str(Path(r['output_dir']).resolve()) for r in expected}
    with (RESULTS / 'final_execution_endpoint_diagnostics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(final_diagnostics[0]))
        writer.writeheader(); writer.writerows(final_diagnostics)
    result = {'status':'PASS', 'utc':datetime.now(timezone.utc).isoformat(), 'requested_pairs':1100,
        'accounted_pairs':len(keys), 'requested_endpoint_denominator':2200,
        'counts':dict(sorted(counts.items())), 'fixed_diagnostic_retries':retries,
        'explicit_execution_failures':failures,
        'prespecified_design_ineligible_pairs':sum(r['category']=='prespecified_design_ineligibility' for r in failures),
        'fitting_or_worker_execution_failure_pairs':sum(r['category']=='fitting_or_worker_execution_failure' for r in failures),
        'all_frozen_sources_unchanged':True, 'all_available_execution_input_hashes_match':True,
        'sampler_seed_and_retry_rules_match':True,
        'data_access':'Execution, provenance and diagnostic summaries only; no truth, predictions, draws or scores.',
        'diagnostic_limit':'Sampler CSVs were fingerprinted and removed by the frozen runner. Parameter Rhat/ESS were independently reread; BFMI, divergence and depth counts are checked from retained recorded diagnostics, not recalculated from deleted chain files.',
        'interpretation':'PASS certifies fixed-task accounting/provenance and diagnostic consistency; it does not imply calibrated coverage or predictive superiority.',
        'source_sha256':sha(__file__), 'shard_plan_sha256':sha(PLAN_ROOT/'shard_plan.csv'),
        'diagnostic_table_sha256':sha(RESULTS/'final_execution_endpoint_diagnostics.csv')}
    (RESULTS / 'final_execution_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
