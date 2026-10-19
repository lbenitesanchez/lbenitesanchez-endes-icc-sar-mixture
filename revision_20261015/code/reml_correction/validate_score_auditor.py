#!/usr/bin/env python3
"""Test the independent auditor using artificial populations only.

Frozen pure arithmetic functions are extracted with AST, without executing their
module imports, runtime initialization, gates, fitting code, or outcome readers.
"""
from __future__ import annotations
import ast
import argparse
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.stats import norm, t

import audit_corrected_scores as independent


def pure_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    independent.require({n.name for n in nodes} == set(names), 'Missing synthetic-test pure function')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def graph_fixture(repo, root, auditor, rng):
    scripts = repo/'src/estudio_publicacion/sikov_2026-10-10/scripts'
    ns = dict(np=np, pd=pd, CONTEXT='SAR_probability_sampled_graph_srs',
        METHODS=independent.ORIGINAL+independent.PUBLISHED, ENDPOINTS=independent.ENDPOINTS,
        METRICS=independent.METRICS, IDS=tuple(range(30000, 30064)), Z=independent.Z)
    pure_functions(scripts/'sensitivity_sampled_graph_score.py',
                   ['score_arrays', 'descriptive_summary', 'descriptive_contrasts'], ns)
    base = root/'sikov/sensibilidad_sampled_graph'
    records = [[], []]
    for rep in range(30000, 30064):
        source = base/'inputs'/f'rep_{rep:05d}'
        (source/'evaluation_only').mkdir(parents=True)
        (source/'model_inputs').mkdir()
        theta = rng.beta(2, 5, (120, 2))
        sample = rng.choice(120, 96, replace=False)
        y = rng.random((96, 2))
        y[rng.random((96, 2)) < .1] = np.nan
        direct = dict(sample_idx=sample, y=y)
        np.savez_compressed(source/'evaluation_only/truth.npz', theta=theta)
        np.savez_compressed(source/'model_inputs/inputs.npz', **direct)
        independent.save(source/'manifest.json', dict(truth_sha256=independent.sha(source/'evaluation_only/truth.npz')))
        for method in independent.ORIGINAL+independent.PUBLISHED:
            mean = theta+rng.normal(0, .07, (120, 2))
            pred = dict(mean=mean, lower=mean-.11, upper=mean+.11)
            folder = base/'fits'/f'rep_{rep:05d}'/('published' if method in independent.PUBLISHED else 'original')
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder/(method+'.npz'), **pred)
            for physical in (False, True):
                records[int(physical)].extend(ns['score_arrays'](theta, direct, pred, method, rep, True, physical))
    output = base/'resultados'
    output.mkdir()
    for prefix, rows in zip(('descriptive', 'physical_projection'), records):
        frame = pd.DataFrame(rows)
        frame.to_csv(output/(prefix+'_replications.csv'), index=False)
        ns['descriptive_summary'](frame).to_csv(output/(prefix+'_summary.csv'), index=False)
        ns['descriptive_contrasts'](frame).to_csv(output/(prefix+'_paired_contrasts.csv'), index=False)
    independent.graph_audit(SimpleNamespace(corrected_root=root), auditor)


def b300_fixture(repo, root, auditor, rng):
    scripts = repo/'src/estudio_publicacion/sikov_2026-10-10/scripts'
    ns = dict(np=np, pd=pd, math=math, ENDPOINTS=independent.ENDPOINTS,
        METRICS=('mse', 'rmse', 'mae', 'bias', 'coverage', 'width', 'interval_score'),
        MODELS=('Selected_B100', 'Published_B100', 'Published_B100_BC', 'Published_B300', 'Published_B300_BC'),
        REFERENCES=('Published_B100', 'Published_B100_BC', 'Published_B300', 'Published_B300_BC'))
    pure_functions(scripts/'sikov_poverty_stratum_descriptive.py',
        ['score_array', 'with_endpoint_balance', 'summarize_populations'], ns)
    pure_functions(scripts/'sikov_published_b300_sensitivity.py', ['score_cells', 'paired_scores'], ns)
    run = root/'sikov'
    full_rows, pub_rows = [], []
    for context in independent.CONTEXTS:
        for rep in (30000, 30001):
            source = run/'evaluacion/inputs'/context/f'rep_{rep:05d}'
            (source/'evaluation_only').mkdir(parents=True)
            (source/'model_inputs').mkdir()
            theta = rng.beta(2, 5, (120, 2))
            direct = dict(sample_idx=rng.choice(120, 96, replace=False), y=rng.random((96, 2)), X_raw=np.zeros((120, 2)))
            np.savez_compressed(source/'evaluation_only/truth.npz', theta=theta)
            np.savez_compressed(source/'model_inputs/inputs.npz', **direct)
            independent.save(source/'manifest.json', dict(truth_sha256=independent.sha(source/'evaluation_only/truth.npz')))
            original = run/'evaluacion/fits'/context/f'rep_{rep:05d}'
            addition = run/'evaluacion/published_sar'/context/f'rep_{rep:05d}'
            sensitivity = run/'evaluacion/b300'/context/f'rep_{rep:05d}'
            models = {}
            for folder, names in ((original, ('Sikov_SAR', 'Selected')),
                (addition, independent.PUBLISHED), (sensitivity, ('full_graph', 'published', 'published_BC', 'Selected_B300'))):
                folder.mkdir(parents=True, exist_ok=True)
                for method in names:
                    mean = theta+rng.normal(0, .08, (120, 2))
                    pred = dict(mean=mean, lower=mean-.13, upper=mean+.13)
                    np.savez_compressed(folder/(method+'.npz'), **pred)
                    models[method] = pred
            full = dict(SAR_B100=models['Sikov_SAR'], SAR_B300=models['full_graph'],
                        Selected_B100=models['Selected'], Selected_B300=models['Selected_B300'])
            for method, pred in full.items():
                for j, outcome in enumerate(independent.ENDPOINTS):
                    values = ns['score_array'](theta[:, j], pred['mean'][:, j], pred['lower'][:, j], pred['upper'][:, j])
                    full_rows.append(dict(context=context, replication=rep, outcome=outcome, method=method, areas=120,
                        **{k: values[k] for k in ('interval_score', 'coverage', 'width', 'mse')}))
            pub = dict(Selected_B100=models['Selected'], Published_B100=models['Sikov_SAR_Published'],
                Published_B100_BC=models['Sikov_SAR_Published_BC'], Published_B300=models['published'],
                Published_B300_BC=models['published_BC'])
            pub_rows.extend(ns['score_cells'](theta, direct, pub, {k: True for k in pub}, context, rep))
    detail = pd.DataFrame(full_rows)
    metrics = ['interval_score', 'coverage', 'width', 'mse']
    summary = detail.groupby(['context', 'method'])[metrics].mean().reset_index()
    overall = summary.groupby('method')[metrics].mean().reset_index()
    overall.insert(0, 'context', 'equal_four_contexts')
    summary = pd.concat([summary, overall], ignore_index=True)
    differences = []
    for context in (*independent.CONTEXTS, 'equal_four_contexts'):
        local = summary.loc[summary.context.eq(context)].set_index('method')
        for candidate, reference in (('Selected_B100', 'SAR_B100'), ('Selected_B300', 'SAR_B300'),
            ('Selected_B300', 'Selected_B100'), ('SAR_B300', 'SAR_B100')):
            differences.append(dict(context=context, candidate=candidate, reference=reference,
                worlds=8 if context == 'equal_four_contexts' else 2,
                **{k: local.loc[candidate, k]-local.loc[reference, k] for k in metrics}))
    (run/'validacion').mkdir()
    for data, name in ((detail, 'scores'), (summary, 'balanced'), (pd.DataFrame(differences), 'differences')):
        data.to_csv(run/'validacion'/('sikov_B300_descriptive_'+name+'.csv'), index=False)
    pub = pd.DataFrame(pub_rows)
    balanced = ns['with_endpoint_balance'](pub)
    out = run/'resultados/published_b300_descriptive'
    out.mkdir(parents=True)
    for data, name in ((pub, 'per_population_scores.csv'),
        (ns['summarize_populations'](balanced, independent.CONTEXTS, (30000, 30001)), 'summary.csv'),
        (ns['paired_scores'](balanced, independent.CONTEXTS, (30000, 30001)), 'selected_reference_differences.csv')):
        data.drop(columns=['poverty_stratum', 'poverty_stratum_label']).to_csv(out/name, index=False)
    independent.b300_audit(SimpleNamespace(corrected_root=root), auditor)


def external_fixture(repo, root, auditor, rng):
    scripts = repo/'src/estudio_publicacion/sikov_2026-10-10/scripts'
    base_ns = dict(np=np, pd=pd, t=t, BOOTSTRAPS=3999)
    pure_functions(scripts/'align_external_score.py', ['balanced_pair'], base_ns)
    methods = ['Selected', 'Sikov_FH', 'Sikov_SAR', 'Sikov_SAR_Published']
    ns = dict(np=np, pd=pd, METHODS=methods, ENDPOINTS=list(independent.ENDPOINTS),
        old=SimpleNamespace(EPS=1e-12, balanced_pair=base_ns['balanced_pair']))
    pure_functions(scripts/'align_external_extended_score.py', ['score_frame', 'contrasts'], ns)
    run = root/'sikov'
    run.mkdir(parents=True)
    private = root/'private_artificial'
    raw_root = private/'fase2_2026-10-07/datos/private_external'
    raw_root.mkdir(parents=True)
    independent.save(run/'real_preparation_receipt.json', dict(private_study=str(private)))
    digests, all_endpoints, all_contrasts, all_core = {}, [], [], []
    provinces = [f'{x:04d}' for x in range(1, 9999) if
        int(independent.hashlib.sha256(('ENDES-phase2-20261007|'+f'{x:04d}').encode()).hexdigest()[:8], 16) % 5 == 0]
    codes = [p+f'{i:02d}' for p in provinces[:20] for i in range(1, 100)][:1874]
    for scope in ('temporal', 'territorial'):
        frame = run/'datos'/('real_2019_full' if scope == 'temporal' else 'real_2024_training')
        frame.mkdir(parents=True)
        pd.DataFrame(dict(ubigeo=codes)).to_csv(frame/'area_frame.csv', index=False)
        wide = []
        for method in methods:
            mean = rng.uniform(-.03, 1.03, (1874, 2))
            pred = dict(mean=mean, lower=mean-.12, upper=mean+.12)
            base = run/'evaluacion/published_sar' if method == 'Sikov_SAR_Published' else run/'evaluacion'
            destination = base/('real_'+scope)
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination/(method+'.npz'), **pred)
            wide.append(pd.DataFrame(dict(ubigeo=codes, method=method,
                p_anemia=mean[:, 0], p_stunting=mean[:, 1],
                lo95_anemia=mean[:, 0]-.12, hi95_anemia=mean[:, 0]+.12,
                lo95_stunting=mean[:, 1]-.12, hi95_stunting=mean[:, 1]+.12)))
        # Distinct endpoint missingness, zero/negative weights and unsupported
        # auxiliary geography exercise domain inclusion and exclusion rules.
        unsupported = provinces[25]+'01'
        used_codes = codes[::3]+[unsupported]
        raw = pd.DataFrame(dict(ubigeo=np.repeat(used_codes, 3), year=2024, de_facto=1,
            age_months=24, psu_key='fixture', weight=rng.uniform(.5, 2., len(used_codes)*3),
            y_anemia_old=rng.integers(0, 2, len(used_codes)*3).astype(float),
            y_stunting=rng.integers(0, 2, len(used_codes)*3).astype(float)))
        raw.loc[raw.ubigeo.eq(used_codes[0]), 'y_anemia_old'] = np.nan
        raw.loc[raw.ubigeo.eq(used_codes[1]), 'weight'] = 0.
        raw.loc[raw.ubigeo.eq(used_codes[2]), 'weight'] = -1.
        path = raw_root/('children_2024_scoring_only.csv' if scope == 'temporal' else 'children_2024_test.csv')
        raw.to_csv(path, index=False)
        digests[scope] = dict(path=str(path), sha256=independent.sha(path))
        target_rows = []
        for endpoint, column in (('anemia', 'y_anemia_old'), ('stunting', 'y_stunting')):
            for code, d in raw.groupby('ubigeo'):
                q = d.loc[d[column].isin((0, 1)) & d.weight.gt(0) & np.isfinite(d.weight)]
                denominator = q.weight.sum()
                target_rows.append(dict(ubigeo=code, province=code[:4], outcome=endpoint,
                    direct=(q.weight*q[column]).sum()/denominator if denominator else np.nan,
                    weighted_observed_denominator=denominator, valid_positive_weight_records=len(q)))
        target = pd.DataFrame(target_rows)
        detail, summary, excluded = ns['score_frame'](pd.concat(wide, ignore_index=True), target)
        out = run/'resultados'/('external_extended_'+scope)
        out.mkdir(parents=True)
        detail.to_csv(out/'district_scores.csv', index=False)
        pd.DataFrame(excluded).to_csv(out/'excluded_domains.csv', index=False)
        endpoint = pd.DataFrame(summary).assign(scope=scope)
        endpoint.to_csv(out/'endpoint_summary.csv', index=False)
        all_endpoints.extend(endpoint.to_dict('records'))
        all_contrasts.extend(ns['contrasts'](detail, scope, methods[1:]))
        all_core.extend(ns['contrasts'](detail.loc[detail.method.isin(methods[:3])], scope, methods[1:3]))
    independent.save(run/'real_scoring_receipt.json', dict(raw2024_reparsed_target_sha256=digests))
    for family, records in ((12, all_contrasts), (8, all_core)):
        frame = pd.DataFrame(records)
        # Literal original stepdown formula, independently of auditor.holm.
        p = frame.p_two_sided.to_numpy()
        order = np.argsort(p, kind='stable')
        adjusted = np.empty(len(p))
        adjusted[order] = np.minimum(1., np.maximum.accumulate(p[order]*np.arange(len(p), 0, -1)))
        frame['p_Holm'+str(family)] = adjusted
        if family == 12:
            frame['classification'] = np.where(adjusted >= .05, 'difference_not_detected',
                np.where(frame.difference < 0, 'candidate_improvement', 'candidate_deterioration'))
        out = run/'resultados'/('external_comparison_Holm12' if family == 12 else 'external_comparison')
        out.mkdir()
        frame.to_csv(out/('paired_balanced_contrasts_Holm'+str(family)+'.csv'), index=False)
        if family == 12:
            pd.DataFrame(all_endpoints).to_csv(out/'endpoint_summary_both_scopes.csv', index=False)
            independent.save(out/'external_final_summary.json', dict(classification_counts=frame.classification.value_counts().to_dict()))
    independent.external_audit(SimpleNamespace(corrected_root=root), auditor)


def run(repo):
    scripts = repo/'src/estudio_publicacion/sikov_2026-10-10/scripts'
    ns = dict(np=np, pd=pd, norm=norm, t=t, math=math, CONTEXTS=independent.CONTEXTS,
        EVALUATION_IDS=tuple(range(30000, 30064)), ENDPOINTS=independent.ENDPOINTS,
        OUTCOMES=independent.ENDPOINTS, MASKS=independent.GROUPS,
        PRIMITIVES=independent.METRICS, Z=independent.Z, BOOTSTRAPS=3999, ALPHA=.05)
    pure_functions(repo/'src/estudio_publicacion/scripts/competitors/benchmark_metrics.py',
                   ['interval_score', 'summarize'], ns)
    ns['interval_score'].__defaults__ = (.05,)
    pure_functions(scripts/'improve_development.py', ['scores'], ns)
    pure_functions(scripts/'comparison_collect.py', ['balanced'], ns)
    pure_functions(repo/'revision_20261015/code/incremental_icc.py',
        ['balanced_mean', 'stratified_se', 'functional_values', 'summarize_increment', 'rank_development'], ns)
    rng = np.random.default_rng(202610181428)
    original, recomputed = [], []
    for c in independent.CONTEXTS:
        for rep in range(30000, 30064):
            theta = rng.beta(2, 5, (120, 2))
            sample = rng.choice(120, 96, replace=False)
            y = rng.random((96, 2))
            y[rng.random((96, 2)) < .1] = np.nan
            direct = dict(sample_idx=sample, y=y)
            for i, method in enumerate(independent.ORIGINAL+independent.PUBLISHED):
                mean = theta+rng.normal(0, .04+i*.002, (120, 2))
                half = rng.uniform(.05, .15, (120, 2))
                pred = dict(mean=mean, lower=mean-half, upper=mean+half)
                original.extend(ns['scores'](dict(theta=theta), direct, pred, method, c, rep, [True, True], 1.))
                recomputed.extend(independent.primitive(theta, direct, pred, c, rep, method))
    original, recomputed = pd.DataFrame(original), pd.DataFrame(recomputed)
    auditor = independent.Audit()
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name)
        def compare(label, expected, actual, keys, cols=None):
            file = folder/(str(len(auditor.checks))+'.csv')
            pd.DataFrame(actual).to_csv(file, index=False)
            auditor.compare(label, expected, file, keys, cols)
        keys = ['scenario', 'replication', 'method', 'outcome', 'group']
        compare('independent primitives vs frozen pure score', recomputed, original, keys)
        compare('independent summary vs frozen pure summarize', independent.summarize(recomputed), ns['summarize'](original),
                ['scenario', 'method', 'outcome', 'group'])
        refs = ('Sikov_FH', 'Sikov_SAR', 'Sikov_SAR_Published')
        expected = independent.balanced_contrasts(recomputed, refs).drop(columns='p_Holm3')
        actual = [row for ref in refs for row in ns['balanced'](original, 'Selected', ref)]
        compare('independent balanced contrasts vs frozen pure contrast', expected, actual,
                ['candidate', 'reference', 'metric'])
        inc = original.loc[original.method.isin(('Selected', 'ICC'))].rename(columns={'scenario': 'context', 'group': 'mask'})
        balanced = inc.groupby(['context', 'replication', 'method', 'mask'], as_index=False)[list(independent.METRICS)].mean()
        balanced['outcome'] = 'endpoint_balanced'
        inc = pd.concat([inc, balanced], ignore_index=True)
        inc['population_rmse'] = np.sqrt(inc.mse)
        inc['population_coverage_distance95'] = abs(inc.coverage-.95)
        compare('independent paired nonlinear MCSE vs frozen pure incremental', independent.incremental(recomputed),
                ns['summarize_increment'](inc), ['scope', 'outcome', 'mask', 'metric'])
        d = original.loc[original.method.isin(('ICC', 'Sikov_FH', 'Sikov_SAR')) & original.group.eq('all')]
        compare('independent ranking vs frozen pure ranking', independent.ranking(d), ns['rank_development'](d), ['method'])
        # Mean population RMSE and endpoint balance must remain separate from pooled RMSE.
        poverty_ns = dict(np=np, pd=pd, math=math, METRICS=(*independent.METRICS, 'rmse'),
                          ENDPOINTS=independent.ENDPOINTS,
                          STRATUM_LABELS={1: 'poverty_below_30_percent', 2: 'poverty_30_to_55_percent', 3: 'poverty_above_55_percent'})
        pure_functions(scripts/'sikov_poverty_stratum_descriptive.py',
                       ['score_array', 'cells', 'with_endpoint_balance', 'summarize_populations'], poverty_ns)
        cell_direct = {**direct, 'X_raw': np.zeros((120, 2)), 'strata': np.tile([1, 2, 3], 40)}
        original_cells = poverty_ns['cells'](theta, cell_direct, {'Selected': pred}, {'Selected': True}, c, rep)
        independent_cells = []
        for stratum in (1, 2, 3):
            values = independent.primitive(theta, cell_direct, pred, c, rep, 'Selected', domain_mask=cell_direct['strata'] == stratum)
            for row in values:
                row.update(poverty_stratum=stratum, domains=row['areas'], rmse=np.sqrt(row['mse']))
            independent_cells.extend(values)
        compare('independent poverty cell masks vs frozen pure cell scorer', independent_cells, original_cells,
                ['scenario', 'replication', 'method', 'outcome', 'group', 'poverty_stratum'],
                [*independent.METRICS, 'rmse', 'domains'])
        p = recomputed.copy().rename(columns={'areas': 'domains'})
        p['rmse'] = np.sqrt(p.mse)
        p['poverty_stratum'] = 0
        p['poverty_stratum_label'] = 'all_strata'
        p['converged'] = True
        expected, _ = independent.population_long_summary(p)
        actual = poverty_ns['summarize_populations'](poverty_ns['with_endpoint_balance'](p), independent.CONTEXTS,
                                                    tuple(range(30000, 30064)))
        compare('independent long population summary vs frozen pure summary', expected, actual,
                ['scenario', 'method', 'outcome', 'group', 'metric', 'poverty_stratum'])
        # Joint endpoint covariance, variable availability and RNG sequence.
        external_ns = dict(np=np, t=t, BOOTSTRAPS=3999)
        pure_functions(scripts/'align_external_score.py', ['balanced_pair'], external_ns)
        candidate = rng.uniform(0, 2, (50, 2))
        reference = rng.uniform(0, 2, (50, 2))
        candidate[::7, 0] = np.nan
        reference[::9, 1] = np.nan
        provinces = np.repeat(np.arange(10), 5).astype(str)
        e = independent.external_cluster_pair(candidate-reference, provinces, np.random.default_rng(983))
        a = external_ns['balanced_pair'](candidate, reference, provinces, np.random.default_rng(983))
        compare('independent cluster Student-t and 3999 bootstrap vs frozen pure inference',
                [dict(key='fixture', **e)], [dict(key='fixture', **a)], ['key'])
        graph_fixture(repo, folder/'graph_artificial', auditor, rng)
        b300_fixture(repo, folder/'b300_artificial', auditor, rng)
        external_fixture(repo, folder/'external_artificial', auditor, rng)
    result = dict(status='PASS_INDEPENDENT_SCORING_AUDITOR_ARTIFICIAL_FIXTURES',
        evaluation_truth_read=False, raw_external_responses_read=False,
        auditor_sha256=independent.sha(Path(independent.__file__)), validator_sha256=independent.sha(__file__),
        tests=auditor.checks, pure_source_functions_extracted_without_module_execution=True,
        source_sha256={str(p.relative_to(repo)): independent.sha(p) for p in (
            scripts/'improve_development.py', scripts/'comparison_collect.py',
            scripts/'sikov_poverty_stratum_descriptive.py', scripts/'align_external_score.py',
            scripts/'sensitivity_sampled_graph_score.py', scripts/'sikov_published_b300_sensitivity.py',
            scripts/'align_external_extended_score.py',
            repo/'revision_20261015/code/incremental_icc.py',
            repo/'src/estudio_publicacion/scripts/competitors/benchmark_metrics.py')})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.repo.resolve())
    independent.save(args.report, result)
    print(json.dumps(dict(status=result['status'], tests=len(result['tests']),
        maximum_absolute_error=max(x.get('maximum_absolute_error', 0.) for x in result['tests']),
        evaluation_truth_read=False)))
