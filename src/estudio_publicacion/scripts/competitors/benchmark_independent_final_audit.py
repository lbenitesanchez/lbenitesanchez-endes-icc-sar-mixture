#!/usr/bin/env python3
"""Independent arithmetic audit after BOTH locked evaluations finish.

No frozen metric, summary, contrast or Holm function is imported. Calculations
start from exported predictions and realized finite populations; no fit, model
selection, seed replacement or new sensitivity is performed.
"""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_inputs import ROOT, OUT, sha256

OUTCOMES = ('anemia', 'stunting')
KEYS = ['scenario', 'replication', 'method', 'group', 'outcome']
SUMMARY_KEYS = ['scenario', 'method', 'protocol', 'group', 'outcome']
POINT_FIELDS = ['bias', 'mse', 'mae', 'coverage', 'width', 'interval_score',
                'areas', 'areas_requested', 'availability_rate', 'converged', 'fit_seconds']
METRICS = ('bias', 'mae', 'coverage', 'width', 'interval_score')
CHECKS = []
ERRORS = {}


def require(condition, message):
    if not condition:
        raise AssertionError(message)
    CHECKS.append(message)


def finished():
    for relative in ('final_evaluation/final_completion.json', 'spatial_evaluation/completion.json'):
        path = OUT / relative
        if not path.exists() or not json.loads(path.read_text()).get('evaluation_completed'):
            raise RuntimeError('Both completed evaluations are required before any performance read')


def compare(actual, expected, keys, fields, label):
    require(not actual.duplicated(keys).any() and not expected.duplicated(keys).any(), label + ': unique keys')
    left = actual.set_index(keys).sort_index()
    right = expected.set_index(keys).sort_index()
    require(set(left.index) == set(right.index), label + ': identical requested/available cells')
    right = right.reindex(left.index)
    maxima = {}
    for field in fields:
        a = pd.to_numeric(left[field], errors='raise').to_numpy(float)
        b = pd.to_numeric(right[field], errors='raise').to_numpy(float)
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=2e-12, equal_nan=True,
                                   err_msg=label + '/' + field)
        finite = np.isfinite(a) & np.isfinite(b)
        maxima[field] = float(np.max(abs(a[finite] - b[finite]))) if finite.any() else None
    ERRORS[label] = maxima
    CHECKS.append(label + ': independently reconstructed arithmetic agrees')


def read_prediction_csv(folder):
    data = pd.read_csv(Path(folder) / 'predictions.csv')
    numeric_area = pd.to_numeric(data.area, errors='raise').to_numpy(float)
    if not np.all(numeric_area == np.round(numeric_area)):
        raise ValueError('Area identifiers must be integral')
    area = numeric_area.astype(int)
    if len(area) != 80 or set(area) != set(range(80)):
        raise ValueError('Prediction universe must contain80 distinct area IDs')
    data = data.assign(area=area).sort_values('area')
    return {key: data[[prefix + '_' + name for name in OUTCOMES]].to_numpy(float)
        for key, prefix in [('mean', 'p_mean'), ('lower', 'lo95'), ('upper', 'hi95')]}


def read_prediction(task):
    folder = Path(task['output_dir'])
    prediction = read_prediction_csv(folder)
    style = task.get('diagnostic_style', 'native')
    path = folder / ('diagnostics.json' if style == 'gao' else 'status.json')
    status = json.loads(path.read_text())
    if style == 'gao':
        converged = [bool(status['endpoints'][name]['INLA_ok'] and
                          status['endpoints'][name]['working_model_converged']) for name in OUTCOMES]
    elif style == 'hogg':
        converged = [bool(status['outcomes'][name]['success']) for name in OUTCOMES]
    elif 'endpoints' in status:
        converged = [bool(value['success']) for value in status['endpoints']]
    else:
        converged = [bool(status['success'])] * 2
    seconds = next((float(status[key]) for key in ['elapsed_seconds', 'fit_seconds', 'seconds_total']
                    if key in status), 0.)
    return prediction, converged, seconds


def independent_rows(task, theta, masks, prediction, converged, seconds):
    point, low, high = [prediction[name] for name in ['mean', 'lower', 'upper']]
    if any(array.shape != (80, 2) for array in (point, low, high)):
        raise ValueError('Both endpoints must use the same80 areas')
    if np.any(np.isfinite(low) & np.isfinite(high) & (low > high)):
        raise ValueError('Reversed prediction interval')
    rows = []
    for column, outcome in enumerate(OUTCOMES):
        finite = np.isfinite(point[:, column]) & np.isfinite(low[:, column]) & np.isfinite(high[:, column])
        if np.any(finite & ((point[:, column] < 0) | (point[:, column] > 1) |
                           (low[:, column] < 0) | (high[:, column] > 1))):
            raise ValueError('Prediction outside proportion support')
        for group, mask in masks.items():
            if task['method'].startswith('Direct') and group != 'observed_endpoint':
                continue
            target = mask[:, column] & finite
            count = int(np.count_nonzero(target))
            if not count:
                continue
            true = theta[target, column]
            estimate, lower, upper = point[target, column], low[target, column], high[target, column]
            difference = estimate - true
            width = upper - lower
            loss = width + 40 * (np.clip(lower - true, 0, None) + np.clip(true - upper, 0, None))
            requested = int(np.count_nonzero(mask[:, column]))
            rows.append(dict(scenario=task['scenario'], replication=int(task['replication']),
                method=task['method'], group=group, outcome=outcome,
                bias=float(np.sum(difference) / count), mse=float(np.dot(difference, difference) / count),
                mae=float(np.sum(np.abs(difference)) / count),
                coverage=float(np.count_nonzero((lower <= true) & (true <= upper)) / count),
                width=float(np.sum(width) / count), interval_score=float(np.sum(loss) / count),
                areas=count, areas_requested=requested, availability_rate=count / requested,
                converged=converged[column], fit_seconds=seconds))
    return rows


def summaries(data):
    rows = []
    for key, part in data.groupby(SUMMARY_KEYS, dropna=False):
        values = dict(zip(SUMMARY_KEYS, key)); count = len(part)
        values.update(replications=count, convergence_rate=part.converged.to_numpy(bool).mean(),
                      mean_areas=part.areas.mean(), mean_fit_seconds=part.fit_seconds.mean(),
                      mean_areas_requested=part.areas_requested.mean(), mean_availability_rate=part.availability_rate.mean())
        mean_square = np.mean(part.mse.to_numpy(float))
        values['rmse'] = math.sqrt(mean_square)
        values['rmse_mcse'] = math.sqrt(np.var(part.mse, ddof=1) / count / (4 * mean_square)) if count > 1 and mean_square > 0 else np.nan
        for field in METRICS:
            observations = part[field].to_numpy(float)
            values[field] = np.sum(observations) / count
            values[field + '_mcse'] = math.sqrt(np.var(observations, ddof=1) / count) if count > 1 else np.nan
        rows.append(values)
    return pd.DataFrame(rows)


def contrasts(data, references):
    candidate = data[data.method == 'BetaPowerHMCGraph']
    rows = []
    keys = ['scenario', 'replication', 'group', 'outcome']
    for reference in references:
        other = data[data.method == reference]
        paired = candidate.merge(other, on=keys, suffixes=('_C', '_R'), validate='one_to_one')
        for key, part in paired.groupby(['scenario', 'group', 'outcome']):
            count = len(part)
            row = dict(zip(['scenario', 'group', 'outcome'], key))
            row.update(contrast='BetaPowerHMCGraph_minus_' + reference, paired_replications=count)
            c, r = part.mse_C.to_numpy(float), part.mse_R.to_numpy(float)
            mean_c, mean_r = c.mean(), r.mean()
            row['rmse_relative_change_pct'] = 100 * (math.sqrt(mean_c / mean_r) - 1)
            influence = c / (2 * mean_c) - r / (2 * mean_r)
            row['log_rmse_ratio_mcse'] = math.sqrt(np.var(influence, ddof=1) / count) if count > 1 else np.nan
            for field in ['bias', 'coverage', 'width', 'interval_score']:
                delta = part[field + '_C'].to_numpy(float) - part[field + '_R'].to_numpy(float)
                row[field + '_change'] = delta.mean()
                row[field + '_change_mcse'] = math.sqrt(np.var(delta, ddof=1) / count) if count > 1 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def balanced_statistics(matrix):
    matrix = np.asarray(matrix, float)
    if matrix.ndim != 2 or matrix.shape[0] != 3:
        raise ValueError('Exactly three context strata are required')
    return float(matrix.mean()), math.sqrt(np.var(matrix, axis=1, ddof=1).sum() / matrix.shape[1] / 9)


def adjusted_holm3(pvalues):
    pvalues = np.asarray(pvalues, float)
    if pvalues.shape != (3,) or not np.isfinite(pvalues).all():
        raise ValueError('Three finite predefined p values are required')
    order = np.argsort(pvalues); adjusted = np.empty(3)
    adjusted[order] = np.minimum(1, np.maximum.accumulate((3 - np.arange(3)) * pvalues[order]))
    return adjusted


def prediction_boundaries(task, theta, masks, prediction, completed_artifact):
    rows = []
    for group, mask in masks.items():
        if task['method'].startswith('Direct') and group != 'observed_endpoint':
            continue
        for column, outcome in enumerate(OUTCOMES):
            point, lower, upper = [prediction[key][:, column] for key in ['mean', 'lower', 'upper']]
            target = mask[:, column]
            available = target & np.isfinite(point) & np.isfinite(lower) & np.isfinite(upper)
            zero = theta[:, column] == 0; one = theta[:, column] == 1
            covered = (lower <= theta[:, column]) & (theta[:, column] <= upper)
            row = dict(scenario=task['scenario'], replication=int(task['replication']), method=task['method'],
                group=group, outcome=outcome, completed_prediction_artifact=completed_artifact,
                requested_area_cells=int(target.sum()), available_area_cells=int(available.sum()),
                truth_zero_requested=int(np.sum(target & zero)), truth_one_requested=int(np.sum(target & one)),
                truth_zero_available=int(np.sum(available & zero)), truth_one_available=int(np.sum(available & one)),
                truth_zero_covered=int(np.sum(available & zero & covered)), truth_one_covered=int(np.sum(available & one & covered)),
                interval_degenerate_zero=int(np.sum(available & (lower == 0) & (upper == 0))),
                interval_degenerate_one=int(np.sum(available & (lower == 1) & (upper == 1))),
                lower_zero_at_interior_truth=int(np.sum(available & (lower == 0) & ~(zero | one))),
                upper_one_at_interior_truth=int(np.sum(available & (upper == 1) & ~(zero | one))))
            for label, values in [('mean', point), ('lower', lower), ('upper', upper)]:
                row[label + '_exact_zero'] = int(np.sum(available & (values == 0)))
                row[label + '_exact_one'] = int(np.sum(available & (values == 1)))
            rows.append(row)
    return rows


def audit():
    finished()
    final, spatial = OUT / 'final_evaluation', OUT / 'spatial_evaluation'
    main_done = json.loads((final / 'final_completion.json').read_text())
    spatial_done = json.loads((spatial / 'completion.json').read_text())
    protocol = json.loads((OUT / 'protocol_freeze.json').read_text())
    contract = json.loads((OUT / 'final_result_contract.json').read_text())
    require(contract['final_tags'][:2] == ['final_primary_v2', 'final_structural_v2'], 'Canonical candidate tags are v2')
    master = json.loads((OUT / 'source_snapshots/numerical_v2_master/source_manifest.json').read_text())
    for name, expected in master['source_sha256'].items():
        path = ROOT / 'scripts' / name if name == 'models.py' else ROOT / 'scripts/competitors' / name
        require(sha256(path) == expected, 'Frozen statistical source checksum: ' + name)
    require(sha256(OUT / 'protocol_freeze.json') == master['protocol_sha256'], 'Approved protocol checksum')
    for name, expected in main_done['output_sha256'].items():
        require(sha256(final / name) == expected, 'Consolidated checksum: ' + name)
    for name, expected in spatial_done['output_sha256'].items():
        require(sha256(spatial / name) == expected, 'Separate spatial checksum: ' + name)
    population = {}; population_boundary = []
    for scenario, ids in protocol['independent_seed_ids'].items():
        for rep in ids:
            folder = OUT / 'inputs' / scenario / f'rep_{rep:05d}'
            input_manifest = json.loads((folder / 'manifest.json').read_text())
            require(input_manifest['scenario'] == scenario and int(input_manifest['replication']) == int(rep),
                    f'Locked population input identity {scenario}/{rep}')
            for filename, expected in input_manifest['input_sha256'].items():
                require(sha256(folder / 'model_inputs' / filename) == expected,
                        f'Estimator input checksum {scenario}/{rep}/{filename}')
            stored = np.load(folder / 'evaluation_only/truth.npz')
            theta = stored['theta']; rebuilt = np.empty((80, 2))
            with np.load(folder / 'evaluation_only/population_outcomes.npz') as units:
                require(set(units.files) == {f'area_{area:03d}' for area in range(80)}, f'Finite population80-area frame {scenario}/{rep}')
                for area in range(80):
                    values = units[f'area_{area:03d}']
                    if values.ndim != 2 or values.shape[1] != 2 or not np.all((values == 0) | (values == 1)):
                        raise AssertionError('Finite population must contain observed binary outcomes')
                    rebuilt[area] = np.sum(values, axis=0, dtype=np.int64) / len(values)
            np.testing.assert_array_equal(theta, rebuilt)
            selected = np.isin(np.arange(80), stored['sample_idx'])
            direct = np.load(folder / 'model_inputs/direct.npz')
            observed = np.zeros((80, 2), bool); observed[direct['sample_idx']] = np.isfinite(direct['y'])
            masks = {'all': np.ones((80, 2), bool), 'sampled_common': np.tile(selected[:, None], (1, 2)),
                     'unsampled_common': np.tile((~selected)[:, None], (1, 2)), 'observed_endpoint': observed}
            population[(scenario, int(rep))] = (theta, masks)
            for group, mask in masks.items():
                for column, outcome in enumerate(OUTCOMES):
                    population_boundary.append(dict(scenario=scenario, replication=int(rep), group=group,
                        outcome=outcome, requested_area_cells=int(mask[:, column].sum()),
                        truth_exact_zero=int(np.sum(mask[:, column] & (theta[:, column] == 0))),
                        truth_exact_one=int(np.sum(mask[:, column] & (theta[:, column] == 1)))))
    require(len(population) == 800, 'Exactly800 original realized finite populations reconstructed')
    tasks = []
    for tag in contract['final_tags'][:2]:
        config = json.loads((OUT / tag / 'config.json').read_text())
        for scenario in config['scenarios']:
            for rep in config['replications']:
                for method in config['methods']:
                    tasks.append(dict(scenario=scenario, replication=int(rep), method=method,
                                      output_dir=str(OUT / tag / 'fit_results' / method / scenario / f'rep_{rep:05d}')))
    for name in ['gao_all1800.csv', 'hogg_collector1100.csv', 'hogg_spatial100.csv']:
        tasks.extend(pd.read_csv(OUT / 'task_plan' / name).to_dict('records'))
    task_frame = pd.DataFrame(tasks)
    require(len(tasks) == 9400 and not task_frame.duplicated(['scenario', 'replication', 'method']).any(), 'Exactly9400 requested fits on800 fixed worlds')
    rebuilt_rows = []; artifact_failures = []; artifacts = {}; boundary_rows = []; completed_artifacts = {}
    for task in tasks:
        key = (task['scenario'], int(task['replication']))
        theta, masks = population[key]
        artifact_key = (task['method'], *key)
        try:
            predictions, converged, seconds = read_prediction(task)
            rows = independent_rows(task, theta, masks, predictions, converged, seconds)
            rebuilt_rows.extend(rows); artifacts[artifact_key] = rows; completed_artifacts[artifact_key] = True
        except Exception as error:
            artifacts[artifact_key] = []
            completed_artifacts[artifact_key] = False
            artifact_failures.append(dict(method=task['method'], scenario=key[0], replication=key[1], error=repr(error)))
        # Boundary accounting describes the exported numeric CSV independently
        # of sampler-status eligibility. A CSV can exist while a metric/status
        # artifact is unavailable; that distinction must not be erased here.
        try:
            boundary_prediction = read_prediction_csv(task['output_dir'])
            boundary_artifact = True
        except Exception:
            boundary_prediction = {name: np.full((80, 2), np.nan) for name in ['mean', 'lower', 'upper']}
            boundary_artifact = False
        boundary_rows.extend(prediction_boundaries(task, theta, masks, boundary_prediction, boundary_artifact))
    arithmetic = pd.DataFrame(rebuilt_rows)
    data = pd.read_csv(final / 'combined_replications.csv')
    space_data = pd.read_csv(spatial / 'replications.csv')
    all_data = pd.concat([data, space_data], ignore_index=True)
    compare(all_data, arithmetic, KEYS, POINT_FIELDS, 'All exported point/interval population metrics')
    # Reuse only labels from the locked row table; every numeric field was independently checked above.
    rebuilt_with_protocol = arithmetic.merge(all_data[KEYS + ['protocol']], on=KEYS, validate='one_to_one')
    sum_fields = ['replications', 'convergence_rate', 'mean_areas', 'mean_fit_seconds', 'mean_areas_requested',
                  'mean_availability_rate', 'rmse', 'rmse_mcse'] + [value for field in METRICS for value in (field, field + '_mcse')]
    expected_summary = summaries(rebuilt_with_protocol)
    saved_summary = pd.concat([pd.read_csv(final / 'combined_summary.csv'), pd.read_csv(spatial / 'summary.csv')], ignore_index=True)
    compare(saved_summary, expected_summary, SUMMARY_KEYS, sum_fields, 'Unconditional summary and between-population MCSE')
    main_summary = expected_summary[expected_summary.method != 'HoggSpatial']
    primary = ['binary_informative_mar', 'sparse_srs_2025', 'sparse_psu_2025']
    structural = ['sparse_srs_iid', 'sparse_psu_bym2']
    secondary_methods = ['GaoAuthorStrictFullH', 'HoggTSLNFullH', 'HoggDesignPsiPSU']
    primary_methods = ['BetaPowerHMCGraph', 'FH', 'MFH', 'SFH', 'MSFH',
                       'GaoAuthorStrict', 'GaoCommonContinuityX', 'HoggTSLN']
    subset_tables = {
        'primary_publication_table.csv': main_summary[main_summary.scenario.isin(primary) &
            main_summary.method.isin(primary_methods) & (main_summary.group == 'all')],
        'structural_summary.csv': main_summary[main_summary.scenario.isin(structural) &
            ~main_summary.method.isin(secondary_methods)],
        'additional_arms_summary.csv': main_summary[main_summary.method.isin(secondary_methods)],
        'direct_and_SAE_observed_summary.csv': main_summary[main_summary.scenario.isin(primary) &
            ~main_summary.method.isin(secondary_methods) & (main_summary.group == 'observed_endpoint')],
    }
    for filename, subset in subset_tables.items():
        compare(pd.read_csv(final / filename), subset, SUMMARY_KEYS, sum_fields,
                'Publication subset preserves verified cells: ' + filename)
    require(len(subset_tables['primary_publication_table.csv']) == 48 and
            (subset_tables['primary_publication_table.csv'].replications == 200).all(),
            'Primary publication table has48 fixed method/context/endpoint cells with200 populations each')
    expected_converged = summaries(rebuilt_with_protocol[rebuilt_with_protocol.converged])
    compare(pd.read_csv(final / 'converged_only_summary.csv'), expected_converged[expected_converged.method != 'HoggSpatial'],
            SUMMARY_KEYS, sum_fields, 'Converged-only sensitivity preserves separate denominator')
    references = sorted(set(all_data.method) - {'BetaPowerHMCGraph'})
    expected_pairs = contrasts(rebuilt_with_protocol, references)
    saved_pairs = pd.concat([pd.read_csv(final / 'paired_contrasts.csv'), pd.read_csv(spatial / 'paired_contrasts.csv')], ignore_index=True)
    pair_fields = ['paired_replications', 'rmse_relative_change_pct', 'log_rmse_ratio_mcse'] + [value for field in ['bias', 'coverage', 'width', 'interval_score'] for value in (field + '_change', field + '_change_mcse')]
    compare(saved_pairs, expected_pairs, ['scenario', 'group', 'outcome', 'contrast'], pair_fields, 'Paired same-population contrasts and influence MCSE')
    primary_references = ['GaoAuthorStrict', 'GaoCommonContinuityX', 'HoggTSLN']
    scores = []
    for reference in primary_references:
        matrix = []
        candidate_score = []; reference_score = []
        population_scores = []
        for scenario in primary:
            c = rebuilt_with_protocol[(rebuilt_with_protocol.method == 'BetaPowerHMCGraph') & (rebuilt_with_protocol.scenario == scenario) & (rebuilt_with_protocol.group == 'all')]
            r = rebuilt_with_protocol[(rebuilt_with_protocol.method == reference) & (rebuilt_with_protocol.scenario == scenario) & (rebuilt_with_protocol.group == 'all')]
            require((c.areas == 80).all() and (r.areas == 80).all(),
                    'Primary score retains every area: ' + scenario + '/' + reference)
            cc = c.pivot(index='replication', columns='outcome', values='converged').reindex(range(10000, 10200))[list(OUTCOMES)]
            rc = r.pivot(index='replication', columns='outcome', values='converged').reindex(range(10000, 10200))[list(OUTCOMES)]
            c = c.pivot(index='replication', columns='outcome', values='interval_score').reindex(range(10000, 10200))[list(OUTCOMES)]
            r = r.pivot(index='replication', columns='outcome', values='interval_score').reindex(range(10000, 10200))[list(OUTCOMES)]
            require(c.shape == r.shape == (200, 2) and np.isfinite(c).all().all() and np.isfinite(r).all().all(), 'Locked primary200 populations/both endpoints: ' + scenario + '/' + reference)
            matrix.append((c.to_numpy() - r.to_numpy()).mean(axis=1))
            candidate_score.append(c.to_numpy().mean()); reference_score.append(r.to_numpy().mean())
            population_scores.extend(dict(scenario=scenario, replication=int(rep),
                score_candidate=float(c.loc[rep].mean()), score_reference=float(r.loc[rep].mean()),
                delta=float((c.loc[rep] - r.loc[rep]).mean()), cells=2,
                convergence_candidate=bool(cc.loc[rep].all()), convergence_reference=bool(rc.loc[rep].all()))
                for rep in range(10000, 10200))
        expected_population_scores = pd.DataFrame(population_scores)
        compare(pd.read_csv(final / ('paired_population_scores_' + reference + '.csv')),
                expected_population_scores, ['scenario', 'replication'],
                ['score_candidate', 'score_reference', 'delta', 'cells', 'convergence_candidate', 'convergence_reference'],
                'Two-endpoint population averaging before MCSE: ' + reference)
        matrix = np.stack(matrix)
        difference, se = balanced_statistics(matrix)
        z = difference / se if se > 0 else np.nan
        scores.append(dict(reference=reference, independent_paired_populations=600,
            populations_per_context=200, primary_test_available=True,
            balanced_interval_score_candidate=np.mean(candidate_score), balanced_interval_score_reference=np.mean(reference_score),
            interval_score_change=difference, paired_mcse=se, change_ci95_lower=difference - 1.959963984540054 * se,
            change_ci95_upper=difference + 1.959963984540054 * se, p_two_sided=math.erfc(abs(z) / math.sqrt(2)),
            relative_score_change_pct=100 * (np.mean(candidate_score) / np.mean(reference_score) - 1), z=z,
            candidate_all_endpoint_convergence=expected_population_scores.convergence_candidate.mean(),
            reference_all_endpoint_convergence=expected_population_scores.convergence_reference.mean()))
    expected_scores = pd.DataFrame(scores)
    expected_scores['p_Holm3'] = adjusted_holm3(expected_scores.p_two_sided.to_numpy())
    compare(pd.read_csv(final / 'primary_balanced_interval_score.csv'), expected_scores, ['reference'],
            [name for name in expected_scores if name != 'reference'], 'Locked600-population balanced score/normal CI/Holm3')
    availability = pd.read_csv(final / 'additional_arm_availability.csv')
    require(len(availability) == 6 and not availability.duplicated(['method', 'outcome']).any(), 'Six additional-arm availability cells including zero-available cells')
    secondary = protocol['paired_additional_arms']
    ledgers = []
    for tag in ['final_gao_evaluation_v1', 'final_hogg_evaluation_v1']:
        ledgers.extend(json.loads((OUT / tag / 'fits.json').read_text()))
    for _, cell in availability.iterrows():
        method = cell.method; specification = secondary[method]; ids = set(specification['seed_ids']); scenario = specification['scenario']
        subset = arithmetic[(arithmetic.method == method) & (arithmetic.scenario == scenario) & (arithmetic.group == 'all') & (arithmetic.outcome == cell.outcome)]
        candidates = arithmetic[(arithmetic.method == 'BetaPowerHMCGraph') & (arithmetic.scenario == scenario) & (arithmetic.group == 'all') & (arithmetic.outcome == cell.outcome) & arithmetic.replication.isin(ids)]
        available = set(subset.replication); paired = available & set(candidates.replication)
        requested = [task for task in ledgers if task['method'] == method and task['scenario'] == scenario]
        require(len(requested) == len(ids) and {int(task['replication']) for task in requested} == ids,
                'Additional requested ledger retains fixed IDs: ' + method + '/' + cell.outcome)
        complete_files = sum(completed_artifacts[(method, scenario, int(task['replication']))] for task in requested)
        attempts = sum(bool(task.get('completed_prediction_artifact')) or any(
            (Path(task['output_dir']) / name).exists() for name in ['task_execution.json', 'execution_failure.json', 'worker_failure.json', 'failure.json'])
            for task in requested)
        require(int(cell.attempted_populations) == attempts and int(cell.completed_prediction_artifacts) == complete_files,
                'Additional attempt/artifact accounting reconstructed: ' + method + '/' + cell.outcome)
        require(int(cell.requested_population_attempts) == int(cell.attempted_populations) == len(ids), 'Every additional-arm attempt completed: ' + method + '/' + cell.outcome)
        require(int(cell.available_endpoint_prediction_populations) == len(available) and int(cell.unavailable_endpoint_prediction_populations) == len(ids) - len(available), 'Additional availability denominator: ' + method + '/' + cell.outcome)
        require(int(cell.usable_endpoint_pairs) == len(paired) and int(cell.unavailable_pair_populations) == len(ids) - len(paired), 'Additional actual paired denominator: ' + method + '/' + cell.outcome)
        require(int(cell.candidate_available_endpoint_prediction_populations) == len(candidates),
                'Additional candidate availability is an actual endpoint count: ' + method + '/' + cell.outcome)
        require(int(cell.converged_endpoint_prediction_populations) == int(subset.converged.sum()) and int(cell.finite_nonconverged_endpoint_prediction_populations) == len(subset) - int(subset.converged.sum()), 'Finite nonconvergence retained: ' + method + '/' + cell.outcome)
        require(bool(cell.all_requested_attempts_completed) and
                bool(cell.all_requested_endpoint_predictions_available) == (available == ids) and
                bool(cell.all_requested_endpoint_pairs_available) == (paired == ids),
                'Additional availability flags retain requested versus usable distinction: ' + method + '/' + cell.outcome)
    missing_secondary = {}
    for method, specification in secondary.items():
        scenario = specification['scenario']
        for rep in specification['seed_ids']:
            rows = artifacts[(method, scenario, int(rep))]
            available_outcomes = {row['outcome'] for row in rows if row['group'] == 'all'}
            missing = [name for name in OUTCOMES if name not in available_outcomes]
            if missing:
                missing_secondary[(method, scenario, int(rep))] = missing
    failures = json.loads((final / 'additional_arm_failures.json').read_text())
    saved_missing = {(entry['method'], entry['scenario'], int(entry['replication'])): entry['unavailable_outcomes']
                     for entry in failures}
    require(len(saved_missing) == len(failures) and saved_missing == missing_secondary,
            'Additional failure reasons cover exactly every unavailable endpoint task without duplicates')
    for entry in failures:
        for evidence in entry['failure_evidence']:
            require(sha256(ROOT / evidence['file']) == evidence['sha256'],
                    'Additional failure evidence checksum: ' + evidence['file'])
    boundary = OUT / 'boundary_audit'
    require((boundary / 'completion.json').exists(), 'Descriptive boundary accounting is finished')
    compare(pd.read_csv(boundary / 'population_replications.csv'), pd.DataFrame(population_boundary),
            ['scenario', 'replication', 'group', 'outcome'], ['requested_area_cells', 'truth_exact_zero', 'truth_exact_one'], 'Exact0/1 realized population boundary counts')
    boundary_expected = pd.DataFrame(boundary_rows)
    boundary_fields = [name for name in boundary_expected if name not in KEYS]
    compare(pd.read_csv(boundary / 'prediction_replications.csv'), boundary_expected,
            KEYS, boundary_fields, 'Exact numeric prediction boundary/coverage/failure counts')
    boundary_done = json.loads((boundary / 'completion.json').read_text())
    require(boundary_done['unique_populations'] == 800 and boundary_done['requested_fit_tasks'] == 9400,
            'Boundary audit retains800 worlds/9400 requested fit denominators')
    return dict(passed=True, checked_utc=datetime.now(timezone.utc).isoformat(),
        independent_arithmetic=True, frozen_statistical_functions_imported=False,
        realized_populations_reconstructed=800, requested_fits_represented=9400,
        exported_population_metric_rows_checked=len(arithmetic), unavailable_artifact_tasks=artifact_failures,
        unconditional_finite_nonconverged_predictions_retained=True,
        primary_population_independence='Conditional on fixed auxiliaries/template; distinct population SeedSequence streams',
        primary_MCSE_convention='Frozen context-stratified between-population variance; common Hogg sampler seeds are computational algorithm inputs, without a marginal-independence claim for finite MCMC noise',
        primary_family=3, primary_paired_population_count=600,
        secondary_availability_conditioning_preserved=True, no_new_hypothesis_family_or_model_selection=True,
        publication_guards=dict(primary=bool(main_done['primary_Holm_family_available']),
            all800=bool(main_done['all_requested_populations_represented']),
            spatial100=bool(spatial_done['all_requested_populations_represented']),
            spatial_candidate100_pairs=bool(spatial_done['all_requested_candidate_pairs_available'])),
        checks=CHECKS, arithmetic_maximum_absolute_errors=ERRORS,
        source_sha256=sha256(Path(__file__)), protocol_sha256=sha256(OUT / 'protocol_freeze.json'),
        main_completion_sha256=sha256(final / 'final_completion.json'), spatial_completion_sha256=sha256(spatial / 'completion.json'),
        population_rng_audit_sha256=sha256(ROOT / 'validacion/competitor_population_rng_independence_audit.json'))


if __name__ == '__main__':
    path = ROOT / 'validacion/competitor_final_independent_audit.json'
    try:
        result = audit()
    except Exception as error:
        result = dict(passed=False, checked_utc=datetime.now(timezone.utc).isoformat(),
                      error_type=type(error).__name__, error=str(error), checks_completed=CHECKS,
                      frozen_statistical_functions_imported=False, no_fit_or_source_modified=True,
                      source_sha256=sha256(Path(__file__)))
        path.write_text(json.dumps(result, indent=2))
        raise
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key not in ['checks', 'arithmetic_maximum_absolute_errors', 'unavailable_artifact_tasks']}, indent=2))
