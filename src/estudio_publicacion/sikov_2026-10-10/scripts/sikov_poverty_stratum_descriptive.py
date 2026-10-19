"""Separately prespecified descriptive poverty-stratum supplement.

Uses closed predictions and the same finite-population targets and domain masks.
No fitting, tuning, hypothesis tests, exclusions or changes to either Holm family.
Real truth bytes are only accessed after both public release gates pass.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import inspect
import json
import math
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
PLAN = ROOT / 'validacion/sikov_poverty_stratum_descriptive_plan.json'
QA = ROOT / 'validacion/sikov_poverty_stratum_descriptive_qa.json'
STATUS = ROOT / 'validacion/sikov_poverty_stratum_descriptive_status.json'
OUT = ROOT / 'resultados/poverty_stratum_descriptive'
ENDPOINTS = ('anemia', 'stunting')
GROUPS = ('all', 'observed_endpoint', 'unsampled_common')
METRICS = ('mse', 'rmse', 'mae', 'bias', 'coverage', 'width', 'interval_score')
REFERENCES = ('Sikov_FH', 'Sikov_SAR', 'Sikov_SAR_Published')
ADDITIONAL = ('Sikov_SAR_Published', 'Sikov_SAR_Published_BC')
STRATUM_LABELS = {1: 'poverty_below_30_percent', 2: 'poverty_30_to_55_percent',
                  3: 'poverty_above_55_percent'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def gate_function_digests():
    # Package inventories are operational. Bind the two mandatory pre-truth
    # gate bodies without preventing additive verification or inventory work.
    from public_verify import prediction_preflight, completed_result_gates
    return {fn.__name__: hashlib.sha256(inspect.getsource(fn).encode('utf-8')).hexdigest()
            for fn in (prediction_preflight, completed_result_gates)}


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve existing certificate: ' + str(path))
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def score_array(theta, mean, lower, upper):
    """Native probability-scale scores for one complete population/domain cell."""
    theta, mean, lower, upper = map(np.asarray, (theta, mean, lower, upper))
    if theta.ndim != 1 or not len(theta) or any(x.shape != theta.shape for x in (mean, lower, upper)):
        raise ValueError('Nonempty aligned one-dimensional target/prediction cell required')
    if not np.isfinite(np.stack((theta, mean, lower, upper))).all() or np.any(lower > upper):
        raise ValueError('Finite ordered sealed predictions required')
    error = mean - theta
    mse = float(np.mean(error ** 2))
    return {'mse': mse, 'rmse': math.sqrt(mse),
            'mae': float(np.mean(np.abs(error))), 'bias': float(np.mean(error)),
            'coverage': float(np.mean((lower <= theta) & (theta <= upper))),
            'width': float(np.mean(upper - lower)),
            'interval_score': float(np.mean(upper - lower + 40 * np.maximum(lower - theta, 0)
                                            + 40 * np.maximum(theta - upper, 0)))}


def cells(theta, inputs, predictions, convergence, context, replication):
    theta = np.asarray(theta, float)
    n = len(inputs['X_raw'])
    if theta.shape != (n, 2) or not np.isfinite(theta).all() or np.any((theta < 0) | (theta > 1)):
        raise ValueError('Same bounded finite-population target required')
    sampled = np.asarray(inputs['sample_idx'], int)
    y = np.asarray(inputs['y'], float)
    strata = np.asarray(inputs['strata'], int)
    if (y.shape != (len(sampled), 2) or strata.shape != (n,) or set(strata) != {1, 2, 3}
            or len(np.unique(sampled)) != len(sampled) or np.any((sampled < 0) | (sampled >= n))):
        raise ValueError('Aligned three-stratum sample frame required')
    observed = np.zeros((n, 2), bool)
    observed[sampled] = np.isfinite(y)
    unobserved_common = ~np.isin(np.arange(n), sampled)
    rows = []
    for method, prediction in predictions.items():
        if any(np.asarray(prediction[key]).shape != (n, 2) for key in ('mean', 'lower', 'upper')):
            raise ValueError('Same full prediction universe required: ' + method)
        for j, endpoint in enumerate(ENDPOINTS):
            for stratum in (1, 2, 3):
                for group, group_mask in (('all', np.ones(n, bool)),
                                          ('observed_endpoint', observed[:, j]),
                                          ('unsampled_common', unobserved_common)):
                    mask = (strata == stratum) & group_mask
                    values = score_array(theta[mask, j], *(prediction[key][mask, j]
                                                         for key in ('mean', 'lower', 'upper')))
                    rows.append({'scenario': context, 'replication': int(replication), 'method': method,
                                 'outcome': endpoint, 'poverty_stratum': stratum,
                                 'poverty_stratum_label': STRATUM_LABELS[stratum], 'group': group,
                                 'domains': int(mask.sum()), 'converged': bool(convergence[method]),
                                 **values})
    return rows


def with_endpoint_balance(frame):
    keys = ['scenario', 'replication', 'method', 'poverty_stratum', 'poverty_stratum_label', 'group']
    if frame.duplicated(keys + ['outcome']).any():
        raise ValueError('Duplicate endpoint/population score cell')
    paired = []
    for identity, block in frame.groupby(keys, sort=True):
        if set(block.outcome) != set(ENDPOINTS) or len(block) != 2:
            raise ValueError('Both endpoints required for population-balanced scores')
        paired.append({**dict(zip(keys, identity)), 'outcome': 'endpoint_balanced',
                       **{m: float(block[m].mean()) for m in METRICS},
                       'domains': float(block.domains.mean()), 'converged': bool(block.converged.all())})
    return pd.concat((frame, pd.DataFrame(paired)), ignore_index=True)


def summarize_populations(frame, contexts, replications):
    """MCSE uses independent populations; never treats districts as replications."""
    contexts, replications = tuple(contexts), tuple(replications)
    keys = ['method', 'outcome', 'poverty_stratum', 'poverty_stratum_label', 'group']
    rows = []
    for identity, block in frame.groupby(keys, sort=True):
        statistics = []
        for context in contexts:
            population = block[block.scenario == context].sort_values('replication')
            if (len(population) != len(replications) or tuple(population.replication) != replications
                    or len(replications) < 2 or not np.isfinite(population[list(METRICS)]).all().all()):
                raise ValueError('Every planned population must be retained in every summary cell')
            mean = population[list(METRICS)].mean().to_numpy(float)
            variance = population[list(METRICS)].var(ddof=1).to_numpy(float) / len(replications)
            statistics.append((mean, variance))
            for m, estimate, variance_of_mean in zip(METRICS, mean, variance):
                rows.append({**dict(zip(keys, identity)), 'scenario': context, 'metric': m,
                             'estimate': float(estimate), 'mcse': math.sqrt(float(variance_of_mean)),
                             'population_worlds': len(replications), 'context_count': 1})
        means = np.asarray([s[0] for s in statistics]); variances = np.asarray([s[1] for s in statistics])
        for m, estimate, variance_of_mean in zip(METRICS, means.mean(axis=0),
                                               variances.sum(axis=0) / len(contexts) ** 2):
            rows.append({**dict(zip(keys, identity)), 'scenario': 'context_balanced', 'metric': m,
                         'estimate': float(estimate), 'mcse': math.sqrt(float(variance_of_mean)),
                         'population_worlds': len(contexts) * len(replications), 'context_count': len(contexts)})
    return pd.DataFrame(rows)


def paired_descriptive(frame, contexts, replications):
    keys = ['scenario', 'replication', 'outcome', 'poverty_stratum', 'poverty_stratum_label', 'group']
    candidate = frame[frame.method == 'Selected']
    rows = []
    for reference in REFERENCES:
        baseline = frame[frame.method == reference]
        paired = candidate.merge(baseline, on=keys, validate='one_to_one', suffixes=('_candidate', '_reference'))
        if len(paired) != len(candidate) or len(paired) != len(baseline):
            raise ValueError('Same paired population/domain cells required')
        differences = paired[keys].copy()
        differences['method'] = 'Selected_minus_' + reference
        for m in METRICS:
            differences[m] = paired[m + '_candidate'] - paired[m + '_reference']
        summary = summarize_populations(differences, contexts, replications)
        summary['reference'] = reference
        summary = summary.rename(columns={'estimate': 'difference_Selected_minus_reference'})
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def fake_qa():
    checks = []
    from sikov_model import poverty_strata
    checks.append(('labels_match_frozen_30_55_model_convention',
                   STRATUM_LABELS[1] == 'poverty_below_30_percent'
                   and STRATUM_LABELS[2] == 'poverty_30_to_55_percent'
                   and STRATUM_LABELS[3] == 'poverty_above_55_percent'
                   and np.array_equal(poverty_strata(np.array([0., .2999, .30, .55, .5501, 1.])),
                                      np.array([1, 1, 2, 2, 3, 3]))))
    target = np.array([.2, .6]); exact = score_array(target, target, target - .1, target + .1)
    checks.append(('perfect_points', all(abs(exact[k]) < 1e-14 for k in ('mse', 'rmse', 'mae', 'bias'))))
    checks.append(('inclusive_coverage_width_score', exact['coverage'] == 1 and abs(exact['width'] - .2) < 1e-14
                   and abs(exact['interval_score'] - .2) < 1e-14))
    calculated = score_array(target, np.array([.4, .5]), np.array([.35, .45]), np.array([.45, .55]))
    oracle = {'mse': .025, 'rmse': math.sqrt(.025), 'mae': .15, 'bias': .05,
              'coverage': 0., 'width': .1, 'interval_score': 4.1}
    checks.append(('hand_calculated_scores', all(abs(calculated[k] - oracle[k]) < 1e-13 for k in oracle)))
    inputs = {'X_raw': np.ones((9, 1)), 'strata': np.repeat([1, 2, 3], 3),
              'sample_idx': np.array([0, 1, 3, 4, 6, 7]), 'y': np.full((6, 2), .5)}
    inputs['y'][0, 0] = np.nan; inputs['y'][3, 1] = np.nan
    truth = np.full((9, 2), .5)
    predictions = {'Selected': {'mean': truth, 'lower': truth - .1, 'upper': truth + .1}}
    example = pd.DataFrame(cells(truth, inputs, predictions, {'Selected': False}, 'fake', 0))
    counts = example.set_index(['outcome', 'poverty_stratum', 'group']).domains
    checks.append(('same_endpoint_missingness', counts['anemia', 1, 'observed_endpoint'] == 1
                   and counts['stunting', 2, 'observed_endpoint'] == 1))
    checks.append(('same_common_unsampled_mask', (example[example.group == 'unsampled_common'].domains == 1).all()))
    checks.append(('all_domains_retained_and_flags_retained', len(example) == 18 and not example.converged.any()))
    try:
        score_array(np.array([]), np.array([]), np.array([]), np.array([]))
        checks.append(('empty_cell_fails_without_exclusion', False))
    except ValueError:
        checks.append(('empty_cell_fails_without_exclusion', True))
    synthetic = []
    contexts = ('fake_a', 'fake_b'); reps = tuple(range(4))
    for c, context in enumerate(contexts):
        for r in reps:
            for method, shift in (('Selected', .02 * r), ('Sikov_FH', .01 * r),
                                  ('Sikov_SAR', 0.), ('Sikov_SAR_Published', -.01 * r)):
                for j, outcome in enumerate(ENDPOINTS):
                    synthetic.append({'scenario': context, 'replication': r, 'method': method,
                                      'outcome': outcome, 'poverty_stratum': 3,
                                      'poverty_stratum_label': STRATUM_LABELS[3], 'group': 'all',
                                      'domains': 40, 'converged': True,
                                      **{m: .2 + .1 * c + (r - 1.5) * (.03 if j == 0 else -.02) + shift
                                         for m in METRICS}})
    fixture = pd.DataFrame(synthetic); balanced = with_endpoint_balance(fixture)
    summary = summarize_populations(balanced, contexts, reps)
    selected = summary[(summary.method == 'Selected') & (summary.outcome == 'endpoint_balanced')
                       & (summary.scenario == 'context_balanced') & (summary.metric == 'mse')].iloc[0]
    oracle_pop = np.array([.2 + .1 * c + (r - 1.5) * .005 + .02 * r for c in range(2) for r in reps]).reshape(2, 4)
    expected_mcse = math.sqrt(np.var(oracle_pop, axis=1, ddof=1).sum() / 4 / 4)
    checks.append(('endpoint_covariance_preserved_before_mcse', abs(selected.mcse - expected_mcse) < 1e-14))
    checks.append(('context_equal_weight_mean', abs(selected.estimate - oracle_pop.mean()) < 1e-14))
    paired = paired_descriptive(balanced, contexts, reps)
    row = paired[(paired.reference == 'Sikov_FH') & (paired.outcome == 'endpoint_balanced')
                 & (paired.scenario == 'context_balanced') & (paired.metric == 'mse')].iloc[0]
    checks.append(('paired_population_covariance_mcse', abs(row.difference_Selected_minus_reference - .015) < 1e-14
                   and abs(row.mcse - math.sqrt(np.var(np.array(reps) * .01, ddof=1) / 4 / 2)) < 1e-14))
    checks.append(('no_p_or_selection_columns', not any(x.startswith('p_') or 'signif' in x or 'Holm' in x
                                                       for x in paired.columns)))
    try:
        summarize_populations(balanced.iloc[:-1], contexts, reps)
        checks.append(('incomplete_population_fails_without_omission', False))
    except ValueError:
        checks.append(('incomplete_population_fails_without_omission', True))
    result = {'status': 'PASS_SYNTHETIC_POVERTY_STRATUM_DESCRIPTIVE_QA' if all(x[1] for x in checks) else 'FAIL',
              'checks_passed': sum(bool(x[1]) for x in checks), 'checks': {k: bool(v) for k, v in checks},
              'synthetic_arrays_only': True, 'actual_accuracy_or_truth_read': False,
              'source_sha256': sha(SOURCE), 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    write(QA, result)
    if result['status'] == 'FAIL':
        raise RuntimeError('Synthetic QA failed')
    return result


def freeze_plan():
    if PLAN.exists():
        raise FileExistsError('Preserve the frozen descriptive plan')
    proof = read(ROOT / 'protocol_freeze.json')
    add = read(ROOT / 'validacion/sikov_published_addendum_freeze.json')
    qa = read(QA)
    if qa['status'] != 'PASS_SYNTHETIC_POVERTY_STRATUM_DESCRIPTIVE_QA' or qa['source_sha256'] != sha(SOURCE):
        raise RuntimeError('Current source must pass synthetic-only QA first')
    forbidden = [ROOT / 'validacion/joint_prediction_lock.json', ROOT / 'validacion/fresh_evaluation_status.json',
                 ROOT / 'validacion/sikov_published_evaluation_status.json', ROOT / 'validacion/sikov_final_independent_audit.json']
    if any(p.exists() for p in forbidden):
        raise RuntimeError('Plan must be frozen before any jointly closed accuracy scoring')
    identities = {**proof['statistical_sources_sha256'], **add['additional_sources_sha256']}
    if len(identities) != 37 or any(sha(path) != digest for path, digest in identities.items()):
        raise RuntimeError('The 37 original/additional statistical identities must remain intact')
    result = {'status': 'FROZEN_DESCRIPTIVE_POVERTY_STRATUM_PLAN_BEFORE_ACCURACY',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'protocol_sha256': sha(ROOT / 'protocol_freeze.json'),
              'addendum_sha256': sha(ROOT / 'validacion/sikov_published_addendum_freeze.json'),
              'source_sha256': sha(SOURCE), 'synthetic_QA_sha256': sha(QA),
              'gate_function_text_sha256': gate_function_digests(),
              'initial_gate_module_snapshot_sha256': sha(ROOT / 'scripts/public_verify.py'),
              'gate_module_snapshot_is_provenance_only': True,
              'operational_package_inventory_is_not_frozen_by_this_plan': True,
              'statistical_source_identities_sha256': identities,
              'original_statistical_sources_verified_intact': 37,
              'methods': list(proof['methods']) + list(ADDITIONAL), 'contexts': proof['contexts'],
              'evaluation_ids': proof['evaluation_ids'], 'fresh_worlds': proof['fresh_worlds'],
              'poverty_strata': STRATUM_LABELS, 'primary_interest_stratum': 3,
              'groups': list(GROUPS), 'metrics': list(METRICS),
              'group_definitions': {'all': 'all districts within the stratum',
                                    'observed_endpoint': 'sample_idx with finite y for the endpoint',
                                    'unsampled_common': 'districts absent from sample_idx, common to both endpoints'},
              'target': 'same finite-population latent prevalence theta as the locked primary experiment',
              'interval_score_alpha': .05, 'predictions_and_intervals_clipped': False,
              'rmse_summary': 'mean of per-population RMSE; distinct from sqrt of pooled mean MSE',
              'mcse_unit': 'independent population/world, with district averages computed within each world',
              'context_balancing': 'four equally weighted contexts; sum of context mean variances divided by 16',
              'endpoint_balancing': 'average both endpoint scores within each world before variance estimation',
              'paired_descriptive_references': list(REFERENCES), 'hypothesis_tests_or_p_values': False,
              'confidence_intervals_or_significance_claims': False, 'Holm_families_modified': False,
              'selection_or_tuning_performed': False, 'models_or_predictions_modified': False,
              'population_or_method_exclusion_permitted': False,
              'empty_group_policy': 'fail and preserve the cell; never omit a population silently',
              'required_pre_truth_gates': ['prediction_preflight', 'completed_result_gates'],
              'required_independent_audit_status': 'PASS_COMPLETE_COMPARISON_INDEPENDENT_AUDIT',
              'truth_or_accuracy_read_by_this_postprocess_before_plan': False,
              'accuracy_result_certificates_present_before_plan': False,
              'addendum_selected_recipe_unchanged': add['selected_recipe'] == proof['selected_recipe']}
    if len(result['methods']) != 11 or result['fresh_worlds'] != 256 or not result['addendum_selected_recipe_unchanged']:
        raise RuntimeError('Wrong complete descriptive cohort')
    write(PLAN, result)
    return result


def run():
    plan = read(PLAN)
    if (plan['status'] != 'FROZEN_DESCRIPTIVE_POVERTY_STRATUM_PLAN_BEFORE_ACCURACY'
            or plan['source_sha256'] != sha(SOURCE) or plan['synthetic_QA_sha256'] != sha(QA)):
        raise RuntimeError('Changed frozen descriptive source/QA')
    if gate_function_digests() != plan['gate_function_text_sha256']:
        raise RuntimeError('Changed mandatory pre-truth gate function body')
    from public_verify import prediction_preflight, completed_result_gates
    meta = prediction_preflight()
    gates = completed_result_gates(meta)
    if (plan['protocol_sha256'] != meta['protocol_sha256'] or plan['addendum_sha256'] != meta['addendum_sha256']
            or plan['methods'] != meta['methods'] or plan['contexts'] != meta['proof']['contexts']
            or plan['evaluation_ids'] != meta['proof']['evaluation_ids']):
        raise RuntimeError('Changed descriptive cohort/protocol')
    identities = {**meta['proof']['statistical_sources_sha256'], **meta['addendum']['additional_sources_sha256']}
    if identities != plan['statistical_source_identities_sha256'] or len(identities) != 37:
        raise RuntimeError('Changed original statistical identity inventory')
    if OUT.exists() or STATUS.exists():
        raise FileExistsError('Preserve completed descriptive outputs')
    records = []; truths = {}; completions = {}
    # Both complete gates above succeed before the first truth hash/load below.
    for context, replication, source, original, manifest, completion in meta['closed']:
        truth_path = source / 'evaluation_only/truth.npz'
        truth_hash = sha(truth_path)
        if truth_hash != manifest['truth_sha256']:
            raise RuntimeError('Changed locked finite-population truth')
        truths[str(truth_path)] = truth_hash
        with np.load(truth_path, allow_pickle=False) as z:
            theta = z['theta'].copy()
        with np.load(source / 'model_inputs/inputs.npz', allow_pickle=False) as z:
            inputs = dict(z)
        additional = ROOT / 'evaluacion/published_sar' / context / f'rep_{replication:05d}'
        pub_completion = read(additional / 'completion.json')
        predictions = {}
        convergence = {**completion['convergence'], **pub_completion['convergence']}
        for method in plan['methods']:
            folder = additional if method in ADDITIONAL else original
            with np.load(folder / (method + '.npz'), allow_pickle=False) as z:
                predictions[method] = {key: z[key].copy() for key in ('mean', 'lower', 'upper')}
        records.extend(cells(theta, inputs, predictions, convergence, context, replication))
        for folder in (original, additional):
            completions[str(folder / 'completion.json')] = sha(folder / 'completion.json')
    frame = pd.DataFrame(records)
    expected = 256 * 11 * 2 * 3 * 3
    keys = ['scenario', 'replication', 'method', 'outcome', 'poverty_stratum', 'group']
    if len(frame) != expected or frame.duplicated(keys).any():
        raise RuntimeError('Incomplete descriptive population/method/endpoint/stratum/group universe')
    balanced = with_endpoint_balance(frame)
    summary = summarize_populations(balanced, plan['contexts'], plan['evaluation_ids'])
    differences = paired_descriptive(balanced, plan['contexts'], plan['evaluation_ids'])
    OUT.mkdir(parents=True)
    frame.to_csv(OUT / 'per_population_stratum_scores.csv', index=False)
    summary.to_csv(OUT / 'stratum_descriptive_summary.csv', index=False)
    differences.to_csv(OUT / 'selected_reference_descriptive_differences.csv', index=False)
    result = {'status': 'PASS_FROZEN_POVERTY_STRATUM_DESCRIPTIVE_SUPPLEMENT',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'protocol_sha256': meta['protocol_sha256'], 'addendum_sha256': meta['addendum_sha256'],
              'joint_prediction_lock_sha256': meta['joint_sha256'],
              'independent_audit_sha256': sha(ROOT / 'validacion/sikov_final_independent_audit.json'),
              'plan_sha256': sha(PLAN), 'source_sha256': sha(SOURCE), 'synthetic_QA_sha256': sha(QA),
              'fresh_worlds': 256, 'methods': plan['methods'], 'poverty_strata': [1, 2, 3],
              'per_population_rows': len(frame), 'summary_rows': len(summary),
              'paired_descriptive_rows': len(differences), 'all_planned_cells_retained': True,
              'all_method_convergence_flags_preserved': True, 'convergence_failures_retained': int((~frame.converged).sum()),
              'models_predictions_weights_or_Holm_families_modified': False,
              'tests_p_values_reselection_or_accuracy_based_tuning': False,
              'truth_access_only_after_prediction_preflight_and_completed_result_gates': True,
              'descriptive_nonconfirmatory_sensitivity': True,
              'rmse_summary_is_mean_population_rmse': True,
              'inputs_sha256': {**truths, **completions},
              'outputs_sha256': {str(p): sha(p) for p in OUT.iterdir() if p.is_file()}}
    write(STATUS, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--qa', action='store_true'); action.add_argument('--freeze-plan', action='store_true')
    action.add_argument('--run', action='store_true')
    action.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    if args.wait:
        plan = read(PLAN)
        if plan['source_sha256'] != sha(SOURCE) or plan['synthetic_QA_sha256'] != sha(QA):
            raise RuntimeError('Changed frozen descriptive source/QA before waiting')
        audit = ROOT / 'validacion/sikov_final_independent_audit.json'
        print('Waiting for the complete independent comparison audit; no truth is accessed.', flush=True)
        while True:
            if not audit.is_file():
                time.sleep(30)
                continue
            try:
                certificate = read(audit)
            except json.JSONDecodeError:
                time.sleep(30)
                continue
            if certificate['status'] != plan['required_independent_audit_status']:
                raise RuntimeError('Wrong final independent audit status')
            break
        args.run = True
    result = fake_qa() if args.qa else freeze_plan() if args.freeze_plan else run()
    print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'inputs_sha256', 'outputs_sha256')}, indent=2))


if __name__ == '__main__':
    main()
