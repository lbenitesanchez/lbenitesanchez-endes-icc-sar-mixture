#!/usr/bin/env python3
"""Independent post-lock audit of corrected IAOS scores.

This program imports no fitting, scoring, adapter, or manuscript-mapping module.
Its default action is a synthetic self-test and reads no study outcome. Actual
evaluation needs --execute-after-locks and the complete native, prediction,
scoring-completion, and editorial-layer seals. Reports contain only aggregates.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import norm, t

CONTEXTS = ('SAR_probability_srs', 'SAR_logit_srs', 'SAR_logit_psu_sparse', 'IID_logit_psu_sparse')
ENDPOINTS = ('anemia', 'stunting')
METRICS = ('mse', 'bias', 'mae', 'interval_score', 'coverage', 'width')
GROUPS = ('all', 'observed_endpoint', 'unsampled_common')
ORIGINAL = ('Sikov_FH', 'Sikov_SAR', 'Sikov_FH_BC', 'Sikov_SAR_BC',
            'Sikov_SAR_k4', 'Sikov_SAR_k8', 'Sikov_SAR_haversine', 'ICC', 'Selected')
PUBLISHED = ('Sikov_SAR_Published', 'Sikov_SAR_Published_BC')
DEVELOPMENT = ('Sikov_FH', 'Sikov_SAR', 'ICC', 'ICC_poverty', 'ICC_SAR_w25', 'ICC_SAR_w50', 'ICC_SAR_w75')
CANDIDATES = ('ICC', 'ICC_poverty', 'ICC_SAR_w25', 'ICC_SAR_w50', 'ICC_SAR_w75')
Z = 1.959963984540054
ATOL = 2e-11


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x:
        x.item() if isinstance(x, np.generic) else str(x)) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_files(root, bindings):
    root = root.resolve()
    require(bool(bindings), 'Empty file seal')
    for name, digest in bindings.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe sealed path')
        require('evaluation_only' not in relative.parts, 'Outcome is unexpectedly in a prediction seal')
        require(sha(root / relative) == digest, 'Changed sealed file: ' + name)


def gate(args):
    """No evaluation path is opened or hashed before this returns successfully."""
    root = args.corrected_root.resolve()
    native = read(args.native_lock)
    require(native.get('complete') is True and native.get('truth_read') is False,
            'Complete truth-blind native lock required before any outcome access')
    require(native.get('planned') == 2780 and len(native.get('items', [])) == 2780,
            'Native lock must contain all 2780 methods')
    inventory = root / 'job_inventory.json'
    require(sha(inventory) == native.get('inventory_sha256'), 'Inventory seal changed')
    jobs = read(inventory)['jobs']
    require(len(jobs) == 2780 and len({x['job_id'] for x in jobs}) == 2780,
            'Inventory is incomplete or duplicated')
    require({x['job_id'] for x in jobs} == {x['job_id'] for x in native['items']},
            'Native method universe differs')
    require(len({x['job_id'] for x in native['items']}) == 2780, 'Duplicate native lock item')
    if 'resolution_manifest_sha256' in native:
        resolution = root / 'resolution_manifest.json'
        require(sha(resolution) == native['resolution_manifest_sha256'], 'Resolution seal changed')
        require(read(resolution)['allowed_stage_roots'] == native['allowed_stage_roots'],
                'Declared stage roots differ')
    bindings = dict(native['outputs_sha256'])
    for item in native['items']:
        name, digest = item['completion_relative_path'], item['completion_sha256']
        require(name not in bindings or bindings[name] == digest, 'Conflicting completion binding')
        bindings[name] = digest
    allowed = [root] + [Path(p).resolve() for p in native.get('allowed_stage_roots', [])]
    for name in bindings:
        path = root / name
        require(path.parent.resolve().is_relative_to(root), 'Mutable native directory link')
        require(any(path.resolve().is_relative_to(p) for p in allowed), 'Undeclared native source')
    bound_files(root, bindings)
    lock_path = root / 'scoring_prediction_lock.json'
    lock = read(lock_path)
    require(lock.get('complete') is True and lock.get('truth_read') is False, 'Prediction seal incomplete')
    require(lock.get('native_lock_sha256') == sha(args.native_lock), 'Native/prediction seal mismatch')
    require(lock.get('fixed_primary_ICC_weight') == .75, 'Primary mixture weight changed')
    require(sha(root / lock['source_snapshot_relative_path']) == lock['source_sha256'],
            'Preparation source snapshot changed')
    require(lock.get('B300_selected_distributions_derived_and_sealed_before_truth') == 8,
            'B300 mixtures were not sealed before evaluation')
    b300 = {f'sikov/evaluacion/b300/{c}/rep_{r:05d}/Selected_B300.npz'
            for c in CONTEXTS for r in (30000, 30001)}
    require(b300 <= set(lock['outputs_sha256']), 'Incomplete B300 prediction seal')
    bc = {f'sikov/evaluacion/b300/{c}/rep_{r:05d}/published_BC.npz'
          for c in CONTEXTS for r in (30000, 30001)}
    require(lock.get('B300_published_BC_distributions_derived_and_sealed_before_truth') == 8 and
            bc <= set(lock['outputs_sha256']), 'Incomplete pre-evaluation B300 BC prediction seal')
    bound_files(root, lock['outputs_sha256'])
    completion = read(args.scoring_output / 'scoring_completion.json')
    require(completion.get('status') == 'COMPLETE_CORRECTED_NUMERICAL_SCORING', 'Scoring incomplete')
    require(completion.get('prediction_lock_sha256') == sha(lock_path), 'Scoring uses another prediction seal')
    layer = read(args.layer / 'layer_manifest.json')
    require(layer.get('status') == 'COMPLETE_CORRECTED_EDITORIAL_LAYER', 'Editorial layer incomplete')
    require(layer.get('prediction_lock_sha256') == sha(lock_path), 'Editorial prediction seal differs')
    require(layer.get('native_prediction_lock_sha256') == sha(args.native_lock), 'Editorial native seal differs')
    require(layer.get('scoring_receipt_sha256') == sha(args.scoring_output / 'scoring_completion.json'),
            'Editorial scoring receipt differs')
    require(layer.get('primary_ICC_weight') == .75, 'Editorial primary mixture weight changed')
    bound_files(args.layer, layer['files_sha256'])
    export_manifest = read(args.layer/'results/reml_numerical_certificates/export_manifest.json')
    require(export_manifest.get('prediction_lock_sha256') == sha(lock_path), 'Aggregate export uses another seal')
    csv_exports = 0
    for item in export_manifest['files']:
        public = args.layer/item['path']
        require(sha(public) == item['public_sha256'], 'Public aggregate export changed')
        relative = Path(item['path']).relative_to('data/sikov_2026-10-10')
        private = root/'sikov'/relative
        require(sha(private) == item['private_source_sha256'], 'Exported private aggregate changed')
        if public.suffix == '.csv':
            require(item['private_source_sha256'] == item['public_sha256'], 'Public CSV differs from scored aggregate')
            csv_exports += 1
    return dict(native_lock_sha256=sha(args.native_lock), prediction_lock_sha256=sha(lock_path),
                scoring_completion_sha256=sha(args.scoring_output / 'scoring_completion.json'),
                editorial_layer_sha256=sha(args.layer / 'layer_manifest.json'),
                native_files=len(bindings), prediction_files=len(lock['outputs_sha256']),
                editorial_files=len(layer['files_sha256']), public_csv_exports_equal_scored_sources=csv_exports)


class Audit:
    def __init__(self):
        self.checks = []
        self.truth_files = {}
        self.poverty_rows = []

    def compare(self, label, expected, path, keys, columns=None, exact_rows=True):
        actual = pd.read_csv(path)
        expected = pd.DataFrame(expected).copy()
        require(not expected.duplicated(keys).any(), label + ': duplicate independent keys')
        require(not actual.duplicated(keys).any(), label + ': duplicate stored keys')
        joined = expected.merge(actual, on=keys, suffixes=('_e', '_a'), how='outer', indicator=True,
                                validate='one_to_one')
        if exact_rows:
            require(joined['_merge'].eq('both').all(), label + ': row universe differs')
        else:
            require(not joined['_merge'].eq('left_only').any(), label + ': expected rows missing')
            joined = joined.loc[joined['_merge'].eq('both')]
        if columns is None:
            columns = [c for c in expected.select_dtypes(include=['number', 'bool']).columns
                       if c not in keys]
        require(bool(columns), label + ': no comparison columns')
        errors = {}
        for col in columns:
            require(col in expected and col in actual, label + ': missing column ' + col)
            x, y = joined[col + '_e'], joined[col + '_a']
            if pd.api.types.is_numeric_dtype(x) or pd.api.types.is_bool_dtype(x):
                xa, ya = np.asarray(x, float), np.asarray(y, float)
                require(np.array_equal(np.isnan(xa), np.isnan(ya)), label + ': missing-value difference ' + col)
                finite = np.isfinite(xa) & np.isfinite(ya)
                require(np.array_equal(np.isfinite(xa), np.isfinite(ya)), label + ': finite-value difference ' + col)
                error = float(np.max(abs(xa[finite] - ya[finite]))) if finite.any() else 0.
                require(np.allclose(xa, ya, atol=ATOL, rtol=2e-10, equal_nan=True),
                        f'{label}: numerical discrepancy in {col}: {error:.6g}')
                errors[col] = error
            else:
                require(x.fillna('<NA>').astype(str).equals(y.fillna('<NA>').astype(str)),
                        label + ': categorical discrepancy ' + col)
        self.checks.append(dict(table=label, rows=len(expected), numeric_columns=len(errors),
                                maximum_absolute_error=max(errors.values(), default=0.),
                                stored_file_sha256=sha(path)))
        return actual

    def note(self, label, **values):
        self.checks.append(dict(table=label, **values))

    def truth(self, folder):
        path = folder / 'evaluation_only/truth.npz'
        manifest = read(folder / 'manifest.json')
        digest = sha(path)
        require(digest == manifest['truth_sha256'], 'Synthetic target generation seal changed')
        with np.load(path, allow_pickle=False) as value:
            theta = value['theta']
        require(theta.shape == (120, 2) and np.isfinite(theta).all(), 'Unexpected finite population')
        self.truth_files[str(path)] = digest
        return theta


def arrays(path):
    with np.load(path, allow_pickle=False) as value:
        return {k: value[k] for k in value.files}


def primitive(theta, direct, prediction, context, rep, method, physical=False, domain_mask=None):
    p, lo, hi = [np.asarray(prediction[k], float) for k in ('mean', 'lower', 'upper')]
    require(p.shape == lo.shape == hi.shape == theta.shape, 'Prediction universe differs')
    require(np.isfinite([p, lo, hi]).all() and (lo <= hi).all(), 'Invalid complete prediction')
    if physical:
        p, lo, hi = [np.clip(x, 0., 1.) for x in (p, lo, hi)]
    observed = np.zeros(theta.shape, bool)
    observed[direct['sample_idx']] = np.isfinite(direct['y'])
    unsampled = ~np.isin(np.arange(len(theta)), direct['sample_idx'])
    rows = []
    for j, endpoint in enumerate(ENDPOINTS):
        for group, mask in zip(GROUPS, (np.ones(len(theta), bool), observed[:, j], unsampled)):
            if domain_mask is not None:
                mask = mask & domain_mask
            require(mask.any(), 'Empty prespecified scoring mask')
            y, v, l, u = [x[mask, j] for x in (theta, p, lo, hi)]
            error = v - y
            rows.append(dict(scenario=context, replication=rep, method=method, group=group,
                outcome=endpoint, mse=float(np.mean(error * error)), bias=float(np.mean(error)),
                mae=float(np.mean(abs(error))), interval_score=float(np.mean(u-l+40*np.maximum(l-y, 0)+40*np.maximum(y-u, 0))),
                coverage=float(np.mean((l <= y) & (y <= u))), width=float(np.mean(u-l)),
                point_outside_0_1=float(np.mean((v < 0) | (v > 1))),
                interval_outside_0_1=float(np.mean((l < 0) | (u > 1))),
                areas=int(mask.sum()), areas_requested=int(mask.sum()), availability_rate=1.))
    return rows


def summarize(frame):
    rows = []
    keys = ['scenario', 'method', 'group', 'outcome']
    for key, d in frame.groupby(keys):
        n = len(d)
        row = dict(zip(keys, key), replications=n, mean_areas=d.areas.mean(),
                   mean_areas_requested=d.areas_requested.mean(), mean_availability_rate=1.)
        if 'converged' in d:
            row['convergence_rate'] = d.converged.mean()
        mse = d.mse.mean()
        row.update(rmse=np.sqrt(mse), rmse_mcse=d.mse.std(ddof=1)/np.sqrt(n)/(2*np.sqrt(mse)))
        for metric in ('bias', 'mae', 'coverage', 'width', 'interval_score'):
            row[metric] = d[metric].mean()
            row[metric + '_mcse'] = d[metric].std(ddof=1)/np.sqrt(n)
        rows.append(row)
    return pd.DataFrame(rows)


def stratified(values, contexts):
    d = pd.DataFrame(dict(v=np.asarray(values), context=np.asarray(contexts)))
    groups = list(d.groupby('context').v)
    require(all(len(v) >= 2 for _, v in groups), 'Insufficient independent populations for MCSE')
    return float(np.sqrt(sum(v.var(ddof=1)/len(v) for _, v in groups))/len(groups))


def average(values, contexts):
    return float(pd.DataFrame(dict(v=np.asarray(values), context=np.asarray(contexts))).groupby('context').v.mean().mean())


def holm(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = sorted(range(len(pvalues)), key=lambda i: (pvalues[i], i))
    answer = np.empty(len(order))
    running = 0.
    for j, i in enumerate(order):
        running = max(running, min(1., (len(order)-j)*pvalues[i]))
        answer[i] = running
    return answer


def balanced_contrasts(frame, references):
    d = frame.loc[frame.group.eq('all')]
    candidate = d.loc[d.method.eq('Selected')]
    rows = []
    for reference in references:
        pair = candidate.merge(d.loc[d.method.eq(reference)], on=['scenario', 'replication', 'outcome'],
                               suffixes=('_a', '_b'), validate='one_to_one')
        for metric in ('interval_score', 'mse', 'mae', 'coverage', 'width'):
            pop = pair.assign(delta=pair[metric+'_a']-pair[metric+'_b']).groupby(['scenario', 'replication']).delta.mean().reset_index()
            value = average(pop.delta, pop.scenario)
            se = stratified(pop.delta, pop.scenario)
            rows.append(dict(candidate='Selected', reference=reference, metric=metric, paired_worlds=len(pop),
                difference=value, mcse=se, MC_CI95_low=value-Z*se, MC_CI95_high=value+Z*se,
                p_two_sided=float(2*norm.sf(abs(value/se))) if se else float(value == 0)))
        m1, m0 = pair.mse_a.mean(), pair.mse_b.mean()
        influence = pair.assign(v=.5*(pair.mse_a/m1-pair.mse_b/m0)).groupby(['scenario', 'replication']).v.mean().reset_index()
        se = stratified(influence.v, influence.scenario)
        value = .5*np.log(m1/m0)
        rows.append(dict(candidate='Selected', reference=reference, metric='RMSE_ratio', paired_worlds=len(influence),
            difference=np.exp(value), mcse=se, mcse_log_ratio=se,
            MC_CI95_low=np.exp(value-Z*se), MC_CI95_high=np.exp(value+Z*se),
            p_two_sided=float(2*norm.sf(abs(value/se))) if se else float(value == 0)))
    result = pd.DataFrame(rows)
    mask = result.metric.eq('interval_score')
    result.loc[mask, 'p_Holm3'] = holm(result.loc[mask, 'p_two_sided'])
    return result


def balanced_endpoints(frame):
    keys = ['scenario', 'replication', 'method', 'group']
    balanced = frame.groupby(keys, as_index=False)[list(METRICS)].mean()
    balanced['outcome'] = 'endpoint_balanced'
    data = pd.concat([frame, balanced], ignore_index=True)
    data['population_rmse'] = np.sqrt(data.mse)
    data['population_coverage_distance95'] = abs(data.coverage-.95)
    return data


def incremental(frame, reference='ICC'):
    data = balanced_endpoints(frame)
    specs = [(k, k, 'linear') for k in METRICS] + [
        ('RMSE_sqrt_mean_MSE', 'mse', 'sqrt'), ('mean_population_RMSE', 'population_rmse', 'linear'),
        ('aggregate_coverage_distance95', 'coverage', 'distance'),
        ('mean_population_coverage_distance95', 'population_coverage_distance95', 'linear')]
    rows = []
    for (outcome, mask), d in data.groupby(['outcome', 'group']):
        p = d.loc[d.method.eq('Selected')].merge(d.loc[d.method.eq(reference)],
            on=['scenario', 'replication', 'outcome', 'group'], suffixes=('_a', '_b'), validate='one_to_one')
        for scope in ('equal_four_contexts', *CONTEXTS):
            q = p if scope == 'equal_four_contexts' else p.loc[p.scenario.eq(scope)]
            c = q.scenario
            for metric, primitive_name, transformation in specs:
                x, y = q[primitive_name+'_a'].to_numpy(), q[primitive_name+'_b'].to_numpy()
                a, b = average(x, c), average(y, c)
                ga = gb = 1.
                if transformation == 'sqrt':
                    a, b = np.sqrt(a), np.sqrt(b)
                    ga, gb = .5/a, .5/b
                elif transformation == 'distance':
                    ga, gb = float(np.sign(a-.95)), float(np.sign(b-.95))
                    a, b = abs(a-.95), abs(b-.95)
                sea, seb, se = stratified(ga*x, c), stratified(gb*y, c), stratified(ga*x-gb*y, c)
                ratio = a/b if abs(b) > 1e-14 and metric != 'bias' else np.nan
                rse = stratified(ga*x/b-a/b**2*gb*y, c) if np.isfinite(ratio) else np.nan
                row = dict(scope=scope, outcome=outcome, mask=mask, metric=metric,
                    population_pairs=len(q), fixed_contexts=c.nunique(), selected=a, icc=b,
                    selected_mcse=sea, icc_mcse=seb, difference_selected_minus_icc=a-b,
                    difference_mcse=se, descriptive95_low=a-b-Z*se, descriptive95_high=a-b+Z*se,
                    ratio_selected_over_icc=ratio, relative_difference_percent=100*(ratio-1), ratio_mcse=rse,
                    relative_difference_percent_mcse=100*rse, relative_descriptive95_low=100*(ratio-1-Z*rse),
                    relative_descriptive95_high=100*(ratio-1+Z*rse))
                if transformation == 'distance':
                    for label, values in (('selected', x), ('icc', y)):
                        cv, cs = average(values, c), stratified(values, c)
                        left, right = cv-Z*cs, cv+Z*cs
                        row[label+'_coverage_CI_spans95'] = left <= .95 <= right
                        row[label+'_distance95_bound_low'] = 0. if left <= .95 <= right else min(abs(left-.95), abs(right-.95))
                        row[label+'_distance95_bound_high'] = max(abs(left-.95), abs(right-.95))
                rows.append(row)
    result = pd.DataFrame(rows)
    if reference != 'ICC':
        result = result.rename(columns={k: k.replace('icc', 'reference') for k in result if 'icc' in k})
    return result


def ranking(frame):
    rank = frame.groupby(['method', 'scenario', 'outcome'])[['interval_score', 'mse', 'coverage']].mean().groupby('method').mean().reset_index()
    rank['balanced_RMSE_sqrt_mean_MSE'] = np.sqrt(rank.pop('mse'))
    rank = rank.sort_values(['interval_score', 'balanced_RMSE_sqrt_mean_MSE', 'method'], kind='stable').reset_index(drop=True)
    rank.insert(0, 'rank', np.arange(1, len(rank)+1))
    rank['gap_IS95_from_best'] = rank.interval_score-rank.interval_score.iloc[0]
    return rank


def development(frame, label, layer, audit):
    d = frame.loc[frame.group.eq('all') & frame.method.isin(CANDIDATES)]
    require(len(d) == 32*5*2, label + ': development grid dimensions differ')
    rank = ranking(d)
    held, ranks = [], []
    for (context, rep), test in d.groupby(['scenario', 'replication']):
        train = d.loc[~(d.scenario.eq(context) & d.replication.eq(rep))]
        r = ranking(train)
        winner, runner = r.iloc[0], r.iloc[1]
        ranks.extend(r.assign(held_context=context, held_replication=int(rep)).to_dict('records'))
        held.append(dict(held_context=context, held_replication=int(rep), selected_method=winner.method,
            runner_up=runner.method, training_IS95=winner.interval_score,
            training_margin_best_to_second=runner.interval_score-winner.interval_score,
            heldout_IS95=test.loc[test.method.eq(winner.method)].interval_score.mean(),
            heldout_ICC_IS95=test.loc[test.method.eq('ICC')].interval_score.mean()))
    base = layer / ('results/devgrid' if label == 'Laplace' else 'results/devgrid_nuts')
    audit.compare(label+' development ranking', rank, base/(label+'_candidate_ranking.csv'), ['method'],
                  ['rank', 'interval_score', 'coverage', 'balanced_RMSE_sqrt_mean_MSE', 'gap_IS95_from_best'])
    held_columns = ['selected_method', 'runner_up', 'training_IS95', 'training_margin_best_to_second']
    held_frame = pd.DataFrame(held)
    if label == 'Laplace':
        held_columns += ['heldout_IS95', 'heldout_ICC_IS95']
    else:
        held_frame = held_frame.rename(columns={'heldout_IS95': 'heldout_selected_IS95'})
        held_columns += ['heldout_selected_IS95']
    audit.compare(label+' LOPO margins', held_frame, base/(label+'_LOPO_selection_and_margins.csv'),
                  ['held_context', 'held_replication'], held_columns)
    audit.compare(label+' LOPO rankings', ranks, base/(label+'_LOPO_all_candidate_rankings.csv'),
                  ['held_context', 'held_replication', 'method'])
    receipt = read(base/(label+'_development_grid_audit.json'))
    winner_key = 'selected_method' if label == 'Laplace' else 'best_method'
    counts = dict(Counter(x['selected_method'] for x in held))
    require(receipt[winner_key] == rank.iloc[0].method and receipt['second_method'] == rank.iloc[1].method,
            label + ': narrative development winner differs')
    require(receipt['LOPO_counts'] == counts, label + ': narrative LOPO counts differ')
    gap = float(rank.interval_score.iloc[1]-rank.interval_score.iloc[0])
    require(abs(receipt['best_second_absolute_margin']-gap) < ATOL, label+': narrative grid margin differs')
    numerical_claims = dict(best_IS95=float(rank.interval_score.iloc[0]),
        best_second_relative_margin_percent=100*(rank.interval_score.iloc[1]/rank.interval_score.iloc[0]-1),
        LOPO_min_margin=min(x['training_margin_best_to_second'] for x in held),
        LOPO_max_margin=max(x['training_margin_best_to_second'] for x in held))
    for key, expected in numerical_claims.items():
        require(np.isclose(receipt[key], expected, atol=ATOL, rtol=2e-10), label+': narrative development claim differs: '+key)
    if label == 'NUTS':
        all_passed = bool(d.converged.all())
        require(receipt['numerically_converged_grid_available'] == all_passed and
                receipt['selection_stability_claim_permitted'] == all_passed,
                'NUTS development numerical-gate interpretation differs')
        require(receipt['numerical_gates']['method_endpoint_world_gate_cells_failed'] == int((~d.converged).sum()),
                'NUTS development failed gate count differs')
    audit.note(label+' narrative numbers', winner=str(rank.iloc[0].method), runner_up=str(rank.iloc[1].method),
               gap=gap, LOPO_counts=counts, populations=32)
    return rank


def synthetic_cohort(audit, root, kind):
    run = root / 'sikov'
    if kind == 'principal':
        reps, methods = range(30000, 30064), ORIGINAL+PUBLISHED
        input_root = run / 'evaluacion/inputs'
        def prediction(c, r, m):
            branch = 'published_sar' if m in PUBLISHED else 'fits'
            return run/'evaluacion'/branch/c/f'rep_{r:05d}'/(m+'.npz')
    elif kind == 'Laplace':
        reps, methods = range(8), DEVELOPMENT
        input_root = run / 'desarrollo/inputs'
        def prediction(c, r, m):
            return run/'desarrollo/recreated_native/fit_results'/c/f'rep_{r:05d}'/m/'predictions.npz'
    elif kind == 'NUTS':
        reps, methods = range(8), DEVELOPMENT
        input_root = run / 'desarrollo/inputs'
        def prediction(c, r, m):
            return root/'numerical_cohorts/development_nuts/fits'/c/f'rep_{r:05d}'/(m+'.npz')
    elif kind == 'sparse57':
        reps, methods = range(50000, 50064), ('Sikov_FH', 'Sikov_SAR', 'Sikov_SAR_Published', 'ICC', 'Selected')
        input_root = root / 'numerical_cohorts/sparse57/evaluacion/inputs'
        def prediction(c, r, m):
            return root/'numerical_cohorts/sparse57/fits'/c/f'rep_{r:05d}'/(m+'.npz')
    else:
        raise ValueError(kind)
    rows, physical = [], []
    for context in CONTEXTS:
        for rep in reps:
            source = input_root/context/f'rep_{rep:05d}'
            theta, direct = audit.truth(source), arrays(source/'model_inputs/inputs.npz')
            expected_sample = 57 if kind == 'sparse57' else 96
            require(len(direct['sample_idx']) == expected_sample, 'Unexpected common sample size')
            for method in methods:
                path = prediction(context, rep, method)
                pred = arrays(path)
                if kind == 'Laplace':
                    status = read(path.parent/'status.json')
                    require('converged' in status or 'success' in status, 'Development status lacks explicit convergence')
                    converged = bool(status['converged']) if 'converged' in status else bool(np.all(status['success']))
                else:
                    converged = bool(read(path.parent/'completion.json')['convergence'][method])
                records = primitive(theta, direct, pred, context, rep, method)
                for row in records:
                    row['converged'] = converged
                rows.extend(records)
                if kind == 'principal':
                    records = primitive(theta, direct, pred, context, rep, method, True)
                    for row in records:
                        row['converged'] = converged
                    physical.extend(records)
                    for stratum in (1, 2, 3):
                        cells = primitive(theta, direct, pred, context, rep, method,
                                          domain_mask=direct['strata'] == stratum)
                        for cell in cells:
                            cell['poverty_stratum'] = stratum
                            cell['domains'] = cell['areas']
                            cell['rmse'] = np.sqrt(cell['mse'])
                            cell['converged'] = converged
                        audit.poverty_rows.extend(cells)
    return pd.DataFrame(rows), pd.DataFrame(physical)


def check_narratives(audit, summary, layer):
    narrative = read(layer/'results/incremental_icc/narrative.json')
    claims = narrative['raw_unrounded_claims']
    flattened = []
    for value in claims.values():
        flattened.extend(value if isinstance(value, list) else [value])
    cols = ['scope', 'outcome', 'mask', 'metric']
    independent = summary.set_index(cols)
    maxerror = 0.
    for row in flattened:
        expected = independent.loc[tuple(row[k] for k in cols)]
        for key in summary.select_dtypes(include=['number', 'bool']).columns:
            if key not in row:
                continue
            x, y = expected[key], row[key]
            if y is None and pd.isna(x):
                continue
            require(np.isclose(float(x), float(y), atol=ATOL, rtol=2e-10, equal_nan=True), 'Narrative numeric claim differs: '+key)
            if np.isfinite(x):
                maxerror = max(maxerror, abs(float(x)-float(y)))
    audit.note('incremental narrative raw claims', claims=len(flattened), maximum_absolute_error=maxerror)
    get = lambda metric: independent.loc[('equal_four_contexts', 'endpoint_balanced', 'all', metric)]
    iv, rm, wd, cv = [get(m) for m in ('interval_score', 'RMSE_sqrt_mean_MSE', 'width', 'coverage')]
    tokens = [f"{iv.relative_difference_percent:+.2f}%", f"{rm.relative_difference_percent:+.2f}%",
        f"{iv.difference_selected_minus_icc:+.6f}", f"{iv.difference_mcse:.6f}",
        f"[{iv.descriptive95_low:+.6f}, {iv.descriptive95_high:+.6f}]",
        f"{wd.relative_difference_percent:+.2f}%", f"{100*cv.selected:.2f}%", f"{100*cv.icc:.2f}%"]
    require(all(token in narrative['English_main_ready'] for token in tokens),
            'Printed incremental narrative differs from independently rounded values')
    for context in CONTEXTS:
        row = independent.loc[(context, 'endpoint_balanced', 'all', 'interval_score')]
        token = f"{row.difference_selected_minus_icc:+.6f} [{row.descriptive95_low:+.6f}, {row.descriptive95_high:+.6f}]"
        require(token in narrative['English_context_ready'], 'Printed context narrative differs')
    audit.note('printed incremental narrative numbers', main_tokens=len(tokens), context_estimates_and_intervals=4)


def synthetic_audit(args, audit):
    root, run, layer = args.corrected_root, args.corrected_root/'sikov', args.layer
    keys = ['scenario', 'replication', 'method', 'group', 'outcome']
    principal, physical = synthetic_cohort(audit, root, 'principal')
    out = run/'resultados/published_sar_comparison'
    audit.compare('principal primitive scores', principal, out/'per_replication_scores.csv', keys)
    audit.compare('principal summary/MCSE', summarize(principal), out/'summary.csv',
                  ['scenario', 'method', 'group', 'outcome'])
    contrast = balanced_contrasts(principal, ('Sikov_FH', 'Sikov_SAR', 'Sikov_SAR_Published'))
    audit.compare('Holm3 and balanced paired contrasts', contrast, out/'paired_balanced_contrasts_Holm3.csv',
                  ['candidate', 'reference', 'metric'])
    historical = balanced_contrasts(principal, ORIGINAL[:-1]).drop(columns='p_Holm3')
    mask2 = historical.metric.eq('interval_score') & historical.reference.isin(('Sikov_FH', 'Sikov_SAR'))
    historical.loc[mask2, 'p_Holm2'] = holm(historical.loc[mask2, 'p_two_sided'])
    audit.compare('historical Holm2 family kept separate', historical,
        run/'resultados/fresh_balanced_paired_contrasts.csv', ['candidate', 'reference', 'metric'])
    audit.compare('physical projection primitive scores', physical, out/'physical_projection_scores.csv', keys)
    audit.compare('physical projection summary/MCSE', summarize(physical), out/'physical_projection_summary.csv',
                  ['scenario', 'method', 'group', 'outcome'])
    physical_contrast = balanced_contrasts(physical, ('Sikov_FH', 'Sikov_SAR', 'Sikov_SAR_Published')).drop(columns='p_Holm3')
    audit.compare('physical projection paired contrasts', physical_contrast, out/'physical_projection_contrasts.csv',
                  ['candidate', 'reference', 'metric'])
    increment = incremental(principal.loc[principal.method.isin(('Selected', 'ICC'))])
    audit.compare('incremental paired nonlinear metrics/MCSE', increment,
                  layer/'results/incremental_icc/incremental_summary.csv', ['scope', 'outcome', 'mask', 'metric'])
    check_narratives(audit, increment, layer)
    ranks = {}
    for kind in ('Laplace', 'NUTS', 'sparse57'):
        frame, _ = synthetic_cohort(audit, root, kind)
        if kind == 'Laplace':
            stored = run/'desarrollo/recreated_native/replications.csv'
        else:
            folder = 'development_nuts' if kind == 'NUTS' else kind
            stored = args.scoring_output/'numerical_cohorts'/folder/'replications.csv'
        audit.compare(kind+' primitive scores', frame, stored, keys)
        if kind in ('Laplace', 'NUTS'):
            ranks[kind] = development(frame, kind, layer, audit)
        else:
            audit.compare('sparse57 summary/MCSE', summarize(frame), stored.parent/'summary.csv',
                          ['scenario', 'method', 'group', 'outcome'])
            records = []
            for reference in ('ICC', 'Sikov_FH', 'Sikov_SAR', 'Sikov_SAR_Published'):
                result = incremental(frame, reference)
                result = result.rename(columns={k: k.replace('icc', 'reference') for k in result if 'icc' in k})
                result['reference_method'] = reference
                records.append(result)
            sparse = pd.concat(records, ignore_index=True)
            audit.compare('sparse57 nonlinear paired metrics/MCSE', sparse,
                layer/'results/sparse57_analysis/sparse57_exploratory_summary.csv',
                ['reference_method', 'scope', 'outcome', 'mask', 'metric'])
            narrative = read(layer/'results/sparse57_analysis/sparse57_summary.json')
            for field in ('all_masks_global', 'global'):
                if field not in narrative:
                    continue
                rows = pd.DataFrame(narrative[field])
                if rows.empty:
                    continue
                numeric = [c for c in sparse.select_dtypes(include=['number', 'bool']) if c in rows]
                merged = rows.merge(sparse, on=['reference_method', 'scope', 'outcome', 'mask', 'metric'],
                                    suffixes=('_a', '_e'), validate='one_to_one')
                require(len(merged) == len(rows), 'Sparse narrative keys differ')
                for col in numeric:
                    require(np.allclose(merged[col+'_a'], merged[col+'_e'], atol=ATOL, rtol=2e-10,
                                        equal_nan=True), 'Sparse narrative value differs: '+col)
            audit.note('sparse57 narrative numeric claims', rows=len(sparse), expected_worlds=256)
            gate_cells = frame.loc[frame.group.eq('all')]
            world_flags = gate_cells.groupby(['scenario', 'replication']).converged.all()
            method_flags = gate_cells.groupby(['scenario', 'replication', 'method']).converged.all()
            expected_gates = dict(population_worlds_total=256, worlds_with_any_failed_gate=int((~world_flags).sum()),
                world_method_pairs_total=len(method_flags), world_method_pairs_with_any_failed_gate=int((~method_flags).sum()),
                method_endpoint_world_gate_cells_total=len(gate_cells),
                method_endpoint_world_gate_cells_failed=int((~gate_cells.converged).sum()))
            for key, expected in expected_gates.items():
                require(narrative['numerical_gates'][key] == expected, 'Sparse narrative gate count differs: '+key)
    comparison = ranks['NUTS'].merge(ranks['Laplace'], on='method', suffixes=('_NUTS', '_Laplace'), validate='one_to_one')
    comparison['rank_change_NUTS_minus_Laplace'] = comparison.rank_NUTS-comparison.rank_Laplace
    comparison['IS95_change_NUTS_minus_Laplace'] = comparison.interval_score_NUTS-comparison.interval_score_Laplace
    audit.compare('NUTS/Laplace grid changes', comparison,
        layer/'results/devgrid_nuts/NUTS_Laplace_grid_comparison.csv', ['method'])
    audit.note('primary weights retained', ICC_weight=.75, SAR_weight=.25, reselected_from_evaluation=False)
    poverty_audit(args, audit)
    return principal


def graph_audit(args, audit):
    root = args.corrected_root/'sikov/sensibilidad_sampled_graph'
    rows = [[], []]
    for rep in range(30000, 30064):
        source = root/'inputs'/f'rep_{rep:05d}'
        theta, direct = audit.truth(source), arrays(source/'model_inputs/inputs.npz')
        for method in ORIGINAL+PUBLISHED:
            branch = 'published' if method in PUBLISHED else 'original'
            pred = arrays(root/'fits'/f'rep_{rep:05d}'/branch/(method+'.npz'))
            for physical in (False, True):
                rows[int(physical)].extend(primitive(theta, direct, pred,
                    'SAR_probability_sampled_graph_srs', rep, method, physical))
    keys = ['scenario', 'replication', 'method', 'group', 'outcome']
    for prefix, records in zip(('descriptive', 'physical_projection'), rows):
        frame = pd.DataFrame(records)
        out = root/'resultados'
        columns = [*METRICS, 'areas', 'point_outside_0_1', 'interval_outside_0_1']
        audit.compare('graph64 '+prefix+' primitives', frame, out/(prefix+'_replications.csv'), keys, columns)
        summary, contrasts = [], []
        for (method, group), d in frame.groupby(['method', 'group']):
            for endpoint in (*ENDPOINTS, 'balanced_anemia_stunting'):
                part = d if endpoint == 'balanced_anemia_stunting' else d.loc[d.outcome.eq(endpoint)]
                worlds = part.groupby('replication')[list(METRICS)].mean()
                require(len(worlds) == 64, 'Incomplete graph64 worlds')
                means, ses = worlds.mean(), worlds.std(ddof=1)/8.
                summary.append(dict(method=method, group=group, outcome=endpoint, worlds=64,
                    rmse=np.sqrt(means.mse), rmse_mcse=ses.mse/(2*np.sqrt(means.mse)) if means.mse else 0.,
                    **means.to_dict(), **{k+'_mcse': ses[k] for k in METRICS}))
        audit.compare('graph64 '+prefix+' summary/MCSE', summary, out/(prefix+'_summary.csv'),
                      ['method', 'group', 'outcome'])
        for group, d in frame.groupby('group'):
            candidate = d.loc[d.method.eq('Selected')].groupby('replication')[list(METRICS)].mean()
            for reference in ORIGINAL+PUBLISHED:
                if reference == 'Selected':
                    continue
                baseline = d.loc[d.method.eq(reference)].groupby('replication')[list(METRICS)].mean()
                for metric in METRICS:
                    delta = candidate[metric]-baseline[metric]
                    value, se = delta.mean(), delta.std(ddof=1)/8.
                    contrasts.append(dict(candidate='Selected', reference=reference, group=group, metric=metric,
                        paired_worlds=64, difference=value, mcse=se, mcse_log_ratio=np.nan,
                        MC_CI95_low=value-Z*se, MC_CI95_high=value+Z*se))
                m1, m0 = candidate.mse.mean(), baseline.mse.mean()
                if m1 == 0 or m0 == 0:
                    ratio = se = lo = hi = logse = np.nan
                else:
                    logratio = .5*np.log(m1/m0)
                    logse = (.5*(candidate.mse/m1-baseline.mse/m0)).std(ddof=1)/8.
                    ratio, se = np.exp(logratio), np.exp(logratio)*logse
                    lo, hi = np.exp(logratio-Z*logse), np.exp(logratio+Z*logse)
                contrasts.append(dict(candidate='Selected', reference=reference, group=group, metric='RMSE_ratio',
                    paired_worlds=64, difference=ratio, mcse=se, mcse_log_ratio=logse,
                    MC_CI95_low=lo, MC_CI95_high=hi))
        actual = audit.compare('graph64 '+prefix+' paired contrasts', contrasts,
            out/(prefix+'_paired_contrasts.csv'), ['candidate', 'reference', 'group', 'metric'])
        require(not any(k.startswith('p_') or 'Holm' in k for k in actual), 'Descriptive graph sensitivity gained formal tests')


def population_long_summary(frame):
    """Separate predeclared mean-population-RMSE summaries, not pooled RMSE."""
    metrics = (*METRICS, 'rmse')
    keys = ['method', 'outcome', 'group']
    if 'poverty_stratum' in frame:
        keys.append('poverty_stratum')
    balanced = frame.groupby(['scenario', 'replication', 'method', 'group'] +
        (['poverty_stratum'] if 'poverty_stratum' in frame else []), as_index=False)[list(metrics)].mean()
    balanced['outcome'] = 'endpoint_balanced'
    data = pd.concat([frame, balanced], ignore_index=True)
    rows = []
    for identity, d in data.groupby(keys):
        for context in (*CONTEXTS, 'context_balanced'):
            part = d if context == 'context_balanced' else d.loc[d.scenario.eq(context)]
            for metric in metrics:
                rows.append(dict(zip(keys, identity), scenario=context, metric=metric,
                    estimate=average(part[metric], part.scenario), mcse=stratified(part[metric], part.scenario),
                    population_worlds=len(part), context_count=part.scenario.nunique()))
    return pd.DataFrame(rows), data


def poverty_audit(args, audit):
    frame = pd.DataFrame(audit.poverty_rows)
    require(len(frame) == 256*11*2*3*3, 'Poverty cells incomplete')
    out = args.corrected_root/'sikov/resultados/poverty_stratum_descriptive'
    keys = ['scenario', 'replication', 'method', 'outcome', 'group', 'poverty_stratum']
    audit.compare('poverty-stratum primitive scores', frame, out/'per_population_stratum_scores.csv', keys,
                  [*METRICS, 'rmse', 'domains'])
    summary, balanced = population_long_summary(frame)
    audit.compare('poverty-stratum summary/MCSE', summary, out/'stratum_descriptive_summary.csv',
        ['scenario', 'method', 'outcome', 'group', 'poverty_stratum', 'metric'])
    records = []
    for reference in ('Sikov_FH', 'Sikov_SAR', 'Sikov_SAR_Published'):
        pairkeys = ['scenario', 'replication', 'outcome', 'group', 'poverty_stratum']
        pair = balanced.loc[balanced.method.eq('Selected')].merge(
            balanced.loc[balanced.method.eq(reference)], on=pairkeys, suffixes=('_a', '_b'), validate='one_to_one')
        delta = pair[pairkeys].copy()
        for metric in (*METRICS, 'rmse'):
            delta[metric] = pair[metric+'_a']-pair[metric+'_b']
        for (endpoint, group, stratum), d in delta.groupby(['outcome', 'group', 'poverty_stratum']):
            for context in (*CONTEXTS, 'context_balanced'):
                q = d if context == 'context_balanced' else d.loc[d.scenario.eq(context)]
                for metric in (*METRICS, 'rmse'):
                    records.append(dict(method='Selected_minus_'+reference, reference=reference,
                        outcome=endpoint, group=group, poverty_stratum=stratum, scenario=context, metric=metric,
                        population_worlds=len(q), context_count=q.scenario.nunique(),
                        difference_Selected_minus_reference=average(q[metric], q.scenario),
                        mcse=stratified(q[metric], q.scenario)))
    actual = audit.compare('poverty-stratum paired descriptive differences', records,
        out/'selected_reference_descriptive_differences.csv',
        ['scenario', 'method', 'reference', 'outcome', 'group', 'poverty_stratum', 'metric'])
    require(not any(k.startswith('p_') or 'Holm' in k for k in actual), 'Poverty sensitivity gained formal tests')


def b300_audit(args, audit):
    run = args.corrected_root/'sikov'
    full, published = [], []
    for context in CONTEXTS:
        for rep in (30000, 30001):
            source = run/'evaluacion/inputs'/context/f'rep_{rep:05d}'
            theta, direct = audit.truth(source), arrays(source/'model_inputs/inputs.npz')
            original = run/'evaluacion/fits'/context/f'rep_{rep:05d}'
            addition = run/'evaluacion/published_sar'/context/f'rep_{rep:05d}'
            sensitivity = run/'evaluacion/b300'/context/f'rep_{rep:05d}'
            for method, path in (('SAR_B100', original/'Sikov_SAR.npz'), ('SAR_B300', sensitivity/'full_graph.npz'),
                ('Selected_B100', original/'Selected.npz'), ('Selected_B300', sensitivity/'Selected_B300.npz')):
                full.extend(primitive(theta, direct, arrays(path), context, rep, method))
            for method, path in (('Selected_B100', original/'Selected.npz'),
                ('Published_B100', addition/'Sikov_SAR_Published.npz'),
                ('Published_B100_BC', addition/'Sikov_SAR_Published_BC.npz'),
                ('Published_B300', sensitivity/'published.npz'),
                ('Published_B300_BC', sensitivity/'published_BC.npz')):
                published.extend(primitive(theta, direct, arrays(path), context, rep, method))
    frame = pd.DataFrame(full)
    frame = frame.loc[frame.group.eq('all')].rename(columns={'scenario': 'context'})
    columns = ['areas', 'interval_score', 'coverage', 'width', 'mse']
    audit.compare('full SAR B100/B300 primitive sensitivity', frame,
        run/'validacion/sikov_B300_descriptive_scores.csv', ['context', 'replication', 'outcome', 'method'], columns)
    summary = frame.groupby(['context', 'method'], as_index=False)[columns[1:]].mean()
    overall = summary.groupby('method', as_index=False)[columns[1:]].mean().assign(context='equal_four_contexts')
    summary = pd.concat([summary, overall], ignore_index=True)
    audit.compare('full SAR B300 balanced sensitivity', summary,
        run/'validacion/sikov_B300_descriptive_balanced.csv', ['context', 'method'])
    contrasts = []
    for context, d in summary.groupby('context'):
        lookup = d.set_index('method')
        for candidate, reference in (('Selected_B100', 'SAR_B100'), ('Selected_B300', 'SAR_B300'),
            ('Selected_B300', 'Selected_B100'), ('SAR_B300', 'SAR_B100')):
            contrasts.append(dict(context=context, candidate=candidate, reference=reference,
                worlds=8 if context == 'equal_four_contexts' else 2,
                **{k: lookup.loc[candidate, k]-lookup.loc[reference, k] for k in columns[1:]}))
    audit.compare('full SAR B300 paired sensitivity', contrasts,
        run/'validacion/sikov_B300_descriptive_differences.csv', ['context', 'candidate', 'reference'])
    published = pd.DataFrame(published).rename(columns={'areas': 'domains'})
    published['rmse'] = np.sqrt(published.mse)
    out = run/'resultados/published_b300_descriptive'
    audit.compare('observed SAR B300 primitive sensitivity', published, out/'per_population_scores.csv',
        ['scenario', 'replication', 'method', 'group', 'outcome'], [*METRICS, 'rmse', 'domains'])
    summary, balanced = population_long_summary(published)
    audit.compare('observed SAR B300 summary/MCSE', summary, out/'summary.csv',
        ['scenario', 'method', 'outcome', 'group', 'metric'])
    differences = []
    for reference in ('Published_B100', 'Published_B100_BC', 'Published_B300', 'Published_B300_BC'):
        keys = ['scenario', 'replication', 'outcome', 'group']
        pair = balanced.loc[balanced.method.eq('Selected_B100')].merge(
            balanced.loc[balanced.method.eq(reference)], on=keys, suffixes=('_a', '_b'), validate='one_to_one')
        delta = pair[keys].copy()
        delta['method'] = 'Selected_B100_minus_'+reference
        for metric in (*METRICS, 'rmse'):
            delta[metric] = pair[metric+'_a']-pair[metric+'_b']
        # Already includes balanced endpoint rows; summarize directly.
        records = []
        for (method, endpoint, group), d in delta.groupby(['method', 'outcome', 'group']):
            for context in (*CONTEXTS, 'context_balanced'):
                q = d if context == 'context_balanced' else d.loc[d.scenario.eq(context)]
                for metric in (*METRICS, 'rmse'):
                    records.append(dict(method=method, outcome=endpoint, group=group, scenario=context,
                        metric=metric, reference=reference, population_worlds=len(q), context_count=q.scenario.nunique(),
                        difference_Selected_B100_minus_reference=average(q[metric], q.scenario),
                        mcse=stratified(q[metric], q.scenario)))
        differences.extend(records)
    actual = audit.compare('observed SAR B300 paired sensitivity/MCSE', differences,
        out/'selected_reference_differences.csv', ['scenario', 'method', 'outcome', 'group', 'metric', 'reference'])
    require(not any(k.startswith('p_') or 'Holm' in k for k in actual), 'Descriptive B300 sensitivity gained formal tests')


def external_targets(path, scope):
    """Called only after all gates; outputs remain private in memory."""
    data = pd.read_csv(path, dtype={'ubigeo': str, 'psu_key': str})
    require(data.year.eq(2024).all() and data.de_facto.eq(1).all() and data.age_months.between(6, 59).all(),
            'External outcome universe changed')
    data['ubigeo'] = data.ubigeo.str.zfill(6)
    if scope == 'territorial':
        reserved = data.ubigeo.str[:4].map(lambda p:
            int(hashlib.sha256(('ENDES-phase2-20261007|'+p).encode()).hexdigest()[:8], 16) % 5 == 0)
        require(reserved.all(), 'Nonreserved territorial province')
    rows = []
    for endpoint, column in (('anemia', 'y_anemia_old'), ('stunting', 'y_stunting')):
        valid = data[column].isin((0, 1)) & data.weight.gt(0) & np.isfinite(data.weight)
        q = data.loc[valid].copy()
        q['_numerator'] = q.weight*q[column]
        grouped = q.groupby('ubigeo').agg(numerator=('_numerator', 'sum'), denominator=('weight', 'sum'), records=('weight', 'size'))
        for code in sorted(data.ubigeo.unique()):
            if code in grouped.index:
                u = grouped.loc[code]
                direct, denominator, records = u.numerator/u.denominator, u.denominator, int(u.records)
            else:
                direct, denominator, records = np.nan, 0., 0
            rows.append(dict(ubigeo=code, province=code[:4], outcome=endpoint, direct=direct,
                weighted_observed_denominator=denominator, valid_positive_weight_records=records))
    return pd.DataFrame(rows)


def external_predictions(run, scope):
    folder = run/'datos'/('real_2019_full' if scope == 'temporal' else 'real_2024_training')
    ids = pd.read_csv(folder/'area_frame.csv', dtype={'ubigeo': str}).ubigeo.str.zfill(6)
    require(len(ids) == 1874 and not ids.duplicated().any(), 'External full district frame changed')
    rows = []
    for method in ('Selected', 'Sikov_FH', 'Sikov_SAR', 'Sikov_SAR_Published'):
        base = run/'evaluacion/published_sar' if method == 'Sikov_SAR_Published' else run/'evaluacion'
        pred = arrays(base/('real_'+scope)/(method+'.npz'))
        require(pred['mean'].shape == pred['lower'].shape == pred['upper'].shape == (1874, 2), 'Real NPZ frame shape')
        require(np.isfinite([pred[k] for k in ('mean', 'lower', 'upper')]).all(), 'Incomplete real prediction')
        for j, endpoint in enumerate(ENDPOINTS):
            rows.append(pd.DataFrame(dict(ubigeo=ids.to_numpy(), method=method, outcome=endpoint,
                prediction=pred['mean'][:, j], lower=pred['lower'][:, j], upper=pred['upper'][:, j])))
    return pd.concat(rows, ignore_index=True)


def external_cluster_pair(delta, provinces, rng):
    labels = sorted(set(provinces))
    sums = np.zeros((len(labels), 2))
    counts = np.zeros_like(sums)
    for i, label in enumerate(labels):
        d = delta[np.asarray(provinces) == label]
        counts[i] = np.isfinite(d).sum(axis=0)
        sums[i] = np.nansum(d, axis=0)
    totals = counts.sum(axis=0)
    require((totals > 0).all() and len(labels) > 1, 'External endpoint or cluster missing')
    means = sums.sum(axis=0)/totals
    estimate = means.mean()
    influence = .5*((sums-counts*means)/totals).sum(axis=1)
    j = len(labels)
    se = np.sqrt(j/(j-1)*sum(influence*influence))
    statistic = estimate/se if se else (0. if estimate == 0 else np.copysign(np.inf, estimate))
    crit = t.ppf(.975, j-1)
    indices = rng.integers(0, j, (3999, j))
    denominators = counts[indices].sum(axis=1)
    require((denominators > 0).all(), 'External bootstrap endpoint denominator vanished')
    bootstrap = (sums[indices].sum(axis=1)/denominators).mean(axis=1)
    lo, hi = np.quantile(bootstrap, (.025, .975))
    return dict(difference=estimate, endpoint_difference_anemia=means[0], endpoint_difference_stunting=means[1],
        cluster_standard_error=se, province_clusters=j, t_df=j-1, p_two_sided=2*t.sf(abs(statistic), j-1),
        unadjusted_t_ci_low=estimate-crit*se, unadjusted_t_ci_high=estimate+crit*se,
        descriptive_bootstrap_ci_low=lo, descriptive_bootstrap_ci_high=hi, bootstrap_repetitions=3999,
        evaluated_anemia_domains=int(totals[0]), evaluated_stunting_domains=int(totals[1]))


def external_audit(args, audit):
    run = args.corrected_root/'sikov'
    receipt = read(run/'real_scoring_receipt.json')
    preparation = read(run/'real_preparation_receipt.json')
    old = Path(preparation['private_study'])/'fase2_2026-10-07/datos/private_external'
    contrasts, endpoints, response_digests = [], [], {}
    core_contrasts = []
    for scope in ('temporal', 'territorial'):
        response = old/('children_2024_scoring_only.csv' if scope == 'temporal' else 'children_2024_test.csv')
        require(sha(response) == receipt['raw2024_reparsed_target_sha256'][scope]['sha256'], 'External response seal changed')
        response_digests[scope] = sha(response)
        target = external_targets(response, scope)
        pred = external_predictions(run, scope)
        codes = set(pred.ubigeo)
        eligible = target.ubigeo.isin(codes) & target.direct.notna() & target.weighted_observed_denominator.gt(0)
        excluded = target.loc[~eligible].copy()
        excluded['reason'] = np.where(excluded.ubigeo.isin(codes),
            'no_valid_positive_weight_outcome', 'unsupported_auxiliary_geography')
        detail = pred.merge(target.loc[eligible], on=['ubigeo', 'outcome'], validate='many_to_one')
        p, y = detail.prediction.to_numpy(), detail.direct.to_numpy()
        q = np.clip(p, 1e-12, 1-1e-12)
        detail['MSE'] = (p-y)**2
        detail['NLL'] = -y*np.log(q)-(1-y)*np.log1p(-q)
        detail['Brier'] = y*(1-p)**2+(1-y)*p**2
        detail['MSE_bounded_sensitivity'] = (np.clip(p, 0, 1)-y)**2
        detail['outside_probability_range'] = (p < 0) | (p > 1)
        detail['NLL_clipped'] = (p < 1e-12) | (p > 1-1e-12)
        detail['noisy_direct_in_interval'] = (y >= detail.lower) & (y <= detail.upper)
        detail['interval_width_pp'] = 100*(detail.upper-detail.lower)
        # Keep district codes as strings for merging; no individual row is exported.
        out = run/'resultados'/('external_extended_'+scope)
        actual = pd.read_csv(out/'district_scores.csv', dtype={'ubigeo': str, 'province': str})
        compare_columns = ['direct', 'weighted_observed_denominator', 'valid_positive_weight_records',
            'MSE', 'NLL', 'Brier', 'MSE_bounded_sensitivity', 'outside_probability_range',
            'NLL_clipped', 'noisy_direct_in_interval', 'interval_width_pp']
        with tempfile.TemporaryDirectory() as name:
            normalized = Path(name)/'district.csv'
            # Comparator normally infers numeric keys; use temporary prefixed keys
            # in memory for exact six-character district identity on both sides.
            actual['district_key'] = 'u'+actual.ubigeo.str.zfill(6)
            expected = detail.assign(district_key='u'+detail.ubigeo)
            actual.to_csv(normalized, index=False)
            audit.compare('external '+scope+' NPZ to weighted-direct scores', expected, normalized,
                          ['district_key', 'method', 'outcome'], compare_columns)
            audit.checks[-1]['stored_file_sha256'] = sha(out/'district_scores.csv')
        excluded_actual = pd.read_csv(out/'excluded_domains.csv', dtype={'ubigeo': str, 'province': str})
        require(set(zip(excluded.ubigeo, excluded.outcome, excluded.reason)) ==
                set(zip(excluded_actual.ubigeo.str.zfill(6), excluded_actual.outcome, excluded_actual.reason)),
                'External excluded domain universe/reasons differ')
        summaries = []
        for (method, endpoint), d in detail.groupby(['method', 'outcome']):
            summaries.append(dict(scope=scope, method=method, outcome=endpoint, districts=len(d),
                provinces=d.province.nunique(), RMSE_noisy_direct_pp=100*np.sqrt(d.MSE.mean()),
                MSE=d.MSE.mean(), Brier=d.Brier.mean(), NLL=d.NLL.mean(),
                RMSE_bounded_sensitivity_pp=100*np.sqrt(d.MSE_bounded_sensitivity.mean()),
                outside_probability_range_count=int(d.outside_probability_range.sum()),
                NLL_clipped_count=int(d.NLL_clipped.sum()), NLL_clipping_epsilon=1e-12,
                noisy_direct_interval_containment_descriptive=d.noisy_direct_in_interval.mean(),
                mean_interval_width_pp=d.interval_width_pp.mean(), latent_interval_coverage_identified=False))
        endpoints.extend(summaries)
        audit.compare('external '+scope+' endpoint aggregates', summaries, out/'endpoint_summary.csv',
                      ['scope', 'method', 'outcome'])
        all_codes = sorted(detail.ubigeo.unique())
        provinces = np.array([x[:4] for x in all_codes])
        for references, collector in ((('Sikov_FH', 'Sikov_SAR', 'Sikov_SAR_Published'), contrasts),
                                      (('Sikov_FH', 'Sikov_SAR'), core_contrasts)):
            rng = np.random.default_rng(202610102024+(10000 if scope == 'territorial' else 0))
            for metric in ('MSE', 'NLL'):
                a = detail.loc[detail.method.eq('Selected')].pivot(index='ubigeo', columns='outcome', values=metric).reindex(
                    index=all_codes, columns=ENDPOINTS).to_numpy()
                for reference in references:
                    b = detail.loc[detail.method.eq(reference)].pivot(index='ubigeo', columns='outcome', values=metric).reindex(
                        index=all_codes, columns=ENDPOINTS).to_numpy()
                    collector.append(dict(scope=scope, candidate='Selected', reference=reference, metric=metric,
                        **external_cluster_pair(a-b, provinces, rng)))
    for family, records in ((12, contrasts), (8, core_contrasts)):
        frame = pd.DataFrame(records)
        frame['p_Holm'+str(family)] = holm(frame.p_two_sided)
        if family == 12:
            frame['classification'] = np.where(frame.p_Holm12 >= .05, 'difference_not_detected',
                np.where(frame.difference < 0, 'candidate_improvement', 'candidate_deterioration'))
        path = run/'resultados'/('external_comparison_Holm12' if family == 12 else 'external_comparison')
        columns = list(frame.select_dtypes(include=['number']).columns)
        if family == 12:
            columns.append('classification')
        audit.compare('external Holm'+str(family)+' province-cluster contrasts', frame,
            path/('paired_balanced_contrasts_Holm'+str(family)+'.csv'),
            ['scope', 'candidate', 'reference', 'metric'], columns)
        if family == 12:
            summary = read(path/'external_final_summary.json')
            require(summary['classification_counts'] == frame.classification.value_counts().to_dict(),
                    'External narrative Holm12 classification counts differ')
            audit.compare('external both-scope endpoint summaries', endpoints, path/'endpoint_summary_both_scopes.csv',
                          ['scope', 'method', 'outcome'])
    audit.note('external response provenance and interpretation', response_sha256=response_digests,
        weighted_noisy_direct_targets=True, province_joint_endpoint_bootstrap=3999,
        new_independent_2024_validation=False, latent_prevalence_or_coverage_identified=False,
        raw_response_records_exported=False)


def self_test():
    theta = np.array([[.2, .6], [.4, .7], [.8, .1]])
    direct = dict(sample_idx=np.array([0, 1]), y=np.array([[.2, .6], [np.nan, .7]]))
    pred = dict(mean=np.array([[.3, .7], [.2, 1.1], [.7, -.2]]),
                lower=np.array([[.1, .5], [0., .5], [.6, -.4]]),
                upper=np.array([[.5, .9], [.5, 1.3], [.9, .2]]))
    rows = pd.DataFrame(primitive(theta, direct, pred, 'fixture', 0, 'method'))
    require(len(rows) == 6, 'Primitive fixture dimensions')
    anemia = rows.loc[rows.group.eq('all') & rows.outcome.eq('anemia')].iloc[0]
    require(np.isclose(anemia.mse, .02) and np.isclose(anemia.coverage, 1.), 'Primitive fixture arithmetic')
    require(rows.loc[rows.group.eq('observed_endpoint') & rows.outcome.eq('anemia'), 'areas'].item() == 1,
            'Endpoint missingness fixture')
    clipped = pd.DataFrame(primitive(theta, direct, pred, 'fixture', 0, 'method', True))
    require(clipped.point_outside_0_1.eq(0).all() and clipped.interval_outside_0_1.eq(0).all(), 'Physical clip fixture')
    require(np.allclose(holm([.01, .04, .03]), [.03, .06, .06]), 'Holm fixture')
    require(np.isclose(stratified([0., 2., 2., 4.], ['a', 'a', 'b', 'b']), np.sqrt(.5)), 'Stratified MCSE fixture')
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name)
        save(folder/'incomplete.json', {'complete': False, 'truth_read': False})
        args = argparse.Namespace(corrected_root=folder, native_lock=folder/'incomplete.json')
        try:
            gate(args)
        except ValueError as error:
            require('before any outcome access' in str(error), 'Incomplete lock must fail before path access')
        else:
            raise AssertionError('Incomplete seal was accepted')
    return dict(status='PASS_SYNTHETIC_AUDITOR_SELF_TEST', evaluation_truth_read=False,
                tests=['primitive metrics', 'endpoint-specific missingness', 'fixed physical projection',
                       'Holm stepdown', 'stratified population MCSE', 'incomplete-seal fail-closed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-after-locks', action='store_true')
    for name in ('repo', 'corrected-root', 'native-lock', 'scoring-output', 'layer', 'report'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    if not args.execute_after_locks:
        result = self_test()
        if args.report:
            save(args.report, result)
        print(json.dumps(result))
        return
    for name in ('repo', 'corrected_root', 'native_lock', 'scoring_output', 'layer', 'report'):
        require(getattr(args, name) is not None, 'Missing --'+name.replace('_', '-'))
    args.corrected_root, args.layer = args.corrected_root.resolve(), args.layer.resolve()
    require(not args.report.resolve().is_relative_to(args.layer), 'Audit report must not mutate the sealed editorial layer')
    seals = gate(args)
    validation_path = args.repo/'revision_20261015/results/reml_correction/independent_scoring_auditor_fixture_validation.json'
    validation = read(validation_path)
    require(validation.get('status') == 'PASS_INDEPENDENT_SCORING_AUDITOR_ARTIFICIAL_FIXTURES' and
            validation.get('auditor_sha256') == sha(__file__) and
            validation.get('evaluation_truth_read') is False and validation.get('raw_external_responses_read') is False,
            'Current auditor requires matching artificial-fixture validation')
    audit = Audit()
    synthetic_audit(args, audit)
    graph_audit(args, audit)
    b300_audit(args, audit)
    external_audit(args, audit)
    require(seals['native_lock_sha256'] == sha(args.native_lock) and
            seals['prediction_lock_sha256'] == sha(args.corrected_root/'scoring_prediction_lock.json') and
            seals['editorial_layer_sha256'] == sha(args.layer/'layer_manifest.json') and
            seals['scoring_completion_sha256'] == sha(args.scoring_output/'scoring_completion.json'),
            'A complete audit seal changed during verification')
    result = dict(status='PASS_INDEPENDENT_CORRECTED_SCORE_AUDIT',
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), auditor_sha256=sha(__file__),
        source_imports='NumPy/Pandas/SciPy only; no scorer, adapter, fitter or mapper imported',
        outcome_access_authorized_only_after_all_seals_verified=True, truth_read_only_after_locks=True, seals=seals,
        artificial_fixture_validation_sha256=sha(validation_path),
        checks=audit.checks, synthetic_truth_files_verified=len(audit.truth_files),
        individual_targets_or_predictions_exported=False,
        caveats=['Historical main RMSE_ratio mcse is log-ratio MCSE.',
                 'External province-cluster SE is sampling error, not simulation MCSE.',
                 'Visual manuscript QA is a separate check.'])
    save(args.report, result)
    print(json.dumps(dict(status=result['status'], checks=len(audit.checks), report=str(args.report))))


if __name__ == '__main__':
    main()
