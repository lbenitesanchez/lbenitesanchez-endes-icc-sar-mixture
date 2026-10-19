"""Independent gates and dense/CDF audit for the 64-world sensitivity.

No fitting or scoring functions are called by this audit. Prediction auditing
never opens synthetic truth. Truth may be opened only by --scores, after the
complete supplemental lock, original JOINT lock and main independent audit.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sensitivity_sampled_graph_generate import (ROOT, DEST, CONTEXT, IDS,
    METHODS, PLAN, gate, local, save, sha)
from sikov_final_audit import (graph, dense_gaussian_check, check_hmc_status)

Z = 1.959963984540054
ENDPOINTS = ('anemia', 'stunting')
METRICS = ('mse', 'mae', 'bias', 'interval_score', 'coverage', 'width')
KEYS = ['scenario', 'replication', 'method', 'outcome', 'group']
LOCK = ROOT / 'validacion/sensitivity_sampled_graph_prediction_lock.json'
PREDICTION_AUDIT = ROOT / 'validacion/sensitivity_sampled_graph_prediction_audit.json'
RESULT_AUDIT = ROOT / 'validacion/sensitivity_sampled_graph_result_audit.json'
RESULT_STATUS = ROOT / 'validacion/sensitivity_sampled_graph_evaluation_status.json'


def validate_closed_metadata(records, plan_sha):
    if len(records) != len(IDS) or \
            {r['replication'] for r in records} != set(IDS):
        raise RuntimeError('All 64 unique supplemental worlds must close')
    for r in records:
        if r['context'] != CONTEXT or r['sensitivity_plan_sha256'] != plan_sha or \
                r['truth_read_by_fits'] is not False:
            raise RuntimeError('Invalid prospective supplemental fit seal')
        if set(r['convergence']) != set(METHODS) or \
                r['selected_recipe'] != {'family': 'ICC_SAR_mixture',
                                          'icc_weight': .75}:
            raise RuntimeError('Changed shared eleven-method recipe or universe')
        rep = r['replication']
        seed = int(np.random.SeedSequence([20261010, 973, rep]).generate_state(1)[0]) \
            % (2 ** 32 - 200000)
        pubseed = int(np.random.SeedSequence([20261010, 974, rep]).generate_state(1)[0])
        if r['original_fit_seed_stream'] != [20261010, 973, rep] or \
                r['original_fit_seed'] != seed or \
                r['published_bootstrap_seed_stream'] != [20261010, 974, rep] or \
                r['published_bootstrap_seed'] != pubseed:
            raise RuntimeError('Unplanned supplemental fit/bootstrap stream')


def independent_triangular(d, samples):
    W = graph(d['coords'], d['strata'], 6)
    W[samples, :] = 0.
    W[np.ix_(samples, samples)] = graph(d['coords'][samples],
                                       d['strata'][samples], 6)
    return W


def validate_joint_path_set(joint):
    expected = {ROOT / 'protocol_freeze.json', ROOT /
        'validacion/sikov_published_addendum_freeze.json', ROOT /
        'validacion/fresh_prediction_lock.json', ROOT /
        'validacion/sikov_published_fresh_prediction_lock.json'}
    for scope in ('temporal', 'territorial'):
        for folder in (ROOT / 'evaluacion' / ('real_' + scope),
                       ROOT / 'evaluacion/published_sar' / ('real_' + scope)):
            expected.update((folder / 'completion.json', folder / 'predictions.csv'))
        expected.update(ROOT / 'validacion' / (prefix + scope + '.json')
            for prefix in ('real_prediction_lock_', 'sikov_published_prediction_lock_'))
    paths = [local(item['path']) for item in joint['files']]
    if len(paths) != len(expected) or set(paths) != expected:
        raise RuntimeError('JOINT seal must bind exactly its sixteen expected metadata/prediction files')
    # Path support is checked before hashing: an injected truth path cannot be
    # read merely by listing it in the JOINT JSON.
    return paths


def principal_scoring_gate():
    plan, plansha, _ = gate()
    jointpath = ROOT / 'validacion/joint_prediction_lock.json'
    joint = json.loads(jointpath.read_text())
    if joint.get('approved_for_scoring') is not True or \
            joint.get('fresh_cohort_size') != 256 or \
            joint.get('all_expected_worlds') is not True or \
            joint.get('fresh_truth_read_before_joint') is not False or \
            joint.get('heldout2024responses_used_before_joint') is not False:
        raise RuntimeError('Complete untouched-before-seal principal JOINT lock required')
    if joint['protocol_sha256'] != plan['protocol_sha256'] or \
            joint['addendum_sha256'] != plan['addendum_sha256']:
        raise RuntimeError('Main JOINT and sensitivity refer to different scientific seals')
    paths = validate_joint_path_set(joint)
    for path, item in zip(paths, joint['files']):
        if sha(path) != item['sha256']:
            raise RuntimeError('Changed principal JOINT-sealed file')
    for key, name in (('fresh_original_prediction_lock_sha256',
                      'fresh_prediction_lock.json'),
                     ('fresh_published_prediction_lock_sha256',
                      'sikov_published_fresh_prediction_lock.json')):
        if joint[key] != sha(ROOT / 'validacion' / name):
            raise RuntimeError('Main paired cohort seal differs')
    auditpath = ROOT / 'validacion/sikov_final_independent_audit.json'
    audit = json.loads(auditpath.read_text())
    if audit.get('status') != 'PASS_COMPLETE_COMPARISON_INDEPENDENT_AUDIT' or \
            audit.get('protocol_sha256') != plan['protocol_sha256'] or \
            audit.get('prediction_lock_sha256') != joint['fresh_original_prediction_lock_sha256'] or \
            audit.get('auditor_source_sha256') != sha(ROOT / 'scripts/sikov_final_audit.py') or \
            audit.get('additional_published_prediction_checks', {}).get(
                'joint_prediction_lock_sha256') != sha(jointpath):
        raise RuntimeError('Main independent numerical audit must pass and bind this JOINT/source')
    return plan, plansha, sha(jointpath), sha(auditpath)


def audit_predictions():
    plan, plansha, _ = gate()
    records = []
    for rep in IDS:
        path = DEST / 'fits' / f'rep_{rep:05d}/completion.json'
        if not path.is_file():
            raise RuntimeError('Supplemental prediction cohort incomplete: ' + str(path))
        records.append(json.loads(path.read_text()))
    validate_closed_metadata(records, plansha)
    items, checks, retained = [], [], []
    maxima = {'mean': 0., 'plugin': 0., 'nll': 0., 'directed_formula': 0.,
              'sampled_covariance_relative': 0., 'mixture_mean': 0.}
    requested = failed = recovered = negative = boundaries = 0
    expected_outputs = {prefix + '/' + name + suffix for prefix, methods in
        (('original', METHODS[:9]), ('published', METHODS[9:]))
        for name in methods for suffix in ('.npz', '_status.json')}
    expected_outputs.add('original/ICC_conditional_draws.npz')
    for rep, record in zip(IDS, records):
        source = DEST / 'inputs' / f'rep_{rep:05d}'
        out = DEST / 'fits' / f'rep_{rep:05d}'
        manifest = json.loads((source / 'manifest.json').read_text())
        if record['protocol_sha256'] != plan['protocol_sha256'] or \
                record['addendum_sha256'] != plan['addendum_sha256'] or \
                record['input_manifest_sha256'] != sha(source / 'manifest.json'):
            raise RuntimeError('Supplemental training manifest changed')
        if manifest['source_sha256'] != sha(ROOT / 'scripts/sensitivity_sampled_graph_generate.py') or \
                manifest['sensitivity_plan_sha256'] != plansha or \
                manifest['population_stream'] != [20261010, 971, rep] or \
                manifest['sampling_stream'] != [20261010, 972, rep] or \
                manifest['generative_rho'] != .6 or \
                manifest['truth_excluded_from_model_inputs'] is not True or \
                manifest['generative_graph_excluded_from_model_inputs'] is not True:
            raise RuntimeError('Changed supplemental generation/prospective role')
        for filename, key in (('inputs.npz', 'fit_inputs_sha256'),
                              ('sample.csv', 'sample_sha256')):
            if sha(source / 'model_inputs' / filename) != record[key] or \
                    record[key] != manifest[key]:
                raise RuntimeError('Supplemental shared input changed')
        if set(record['output_sha256']) != expected_outputs:
            raise RuntimeError('Supplemental outputs must be exactly the eleven predictions/statuses and private draws')
        for filename, digest in record['output_sha256'].items():
            if sha(out / filename) != digest:
                raise RuntimeError('Changed sealed supplemental output ' + filename)
        d = dict(np.load(source / 'model_inputs/inputs.npz', allow_pickle=False))
        if d['X_raw'].shape != (120, 15) or d['X'].shape != (120, 15) or \
                d['basis'].shape != (120, 18) or d['y'].shape != (96, 2) or \
                any((d['strata'] == g).sum() != 40 for g in (1, 2, 3)) or \
                any((d['strata'][d['sample_idx']] == g).sum() != 32 for g in (1, 2, 3)):
            raise RuntimeError('Supplemental common dimensions/support differ')
        if not np.array_equal(d['W'], graph(d['coords'], d['strata'], 6)):
            raise RuntimeError('Estimator full-frame input graph changed')
        Wgen = independent_triangular(d, d['sample_idx'])
        import hashlib
        if hashlib.sha256(np.ascontiguousarray(Wgen).tobytes()).hexdigest() != \
                manifest['generative_W_sha256']:
            raise RuntimeError('Generative sampled-only graph differs independently')
        predictions = {name: dict(np.load(out / ('original' if name in METHODS[:9]
                       else 'published') / (name + '.npz'), allow_pickle=False))
                       for name in METHODS}
        for name, p in predictions.items():
            if np.stack([p[key] for key in ('mean', 'lower', 'upper')]).shape != \
                    (3, 120, 2) or not np.isfinite(np.stack([p[key] for key in
                    ('mean', 'lower', 'upper')])).all() or np.any(p['lower'] > p['upper']):
                raise RuntimeError('Non-finite or incomplete supplemental prediction ' + name)
            if not record['convergence'][name]:
                retained.append([rep, name])
        icc_status = json.loads((out / 'original/ICC_status.json').read_text())
        check_hmc_status(icc_status)
        if record['convergence']['ICC'] != icc_status['converged']:
            raise RuntimeError('Supplemental ICC convergence flag differs')
        for name in ('Sikov_FH', 'Sikov_SAR', 'Sikov_SAR_k4',
                     'Sikov_SAR_k8', 'Sikov_SAR_haversine', 'Sikov_SAR_Published'):
            prefix = 'published' if name == 'Sikov_SAR_Published' else 'original'
            status = json.loads((out / prefix / (name + '_status.json')).read_text())
            if record['convergence'][name] != status['converged']:
                raise RuntimeError('Supplemental Gaussian convergence flag differs')
            overrides = None
            if name == 'Sikov_SAR_Published':
                overrides = []
                for j in range(2):
                    mask = np.isfinite(d['y'][:, j]) & np.isfinite(d['D'][:, j, j]) & (d['D'][:, j, j] > 0)
                    overrides.append(independent_triangular(d, d['sample_idx'][mask]))
            check = dense_gaussian_check(d, status, 'Sikov_SAR' if overrides
                        is not None else name, predictions[name], W_override=overrides)
            checks.append({'replication': rep, 'method': name, **check})
            for key in ('mean', 'plugin', 'nll'):
                maxima[key] = max(maxima[key], check['maximum_errors'][key])
            requested += check['bootstrap_requested']; failed += check['bootstrap_failed']
            recovered += check['bootstrap_recovered']; boundaries += check['rho_boundaries']
            p = predictions[name]
            if np.any(p['mse'] < 0) or any(not np.allclose(p[bound],
                    p['mean'] + sign * Z * np.sqrt(p['mse']), atol=1e-12, rtol=0)
                    for bound, sign in (('lower', -1), ('upper', 1))):
                raise RuntimeError('Supplemental Gaussian native interval convention changed')
            if overrides is not None:
                for j, endpointfit in enumerate(status['fits']):
                    mask = np.isfinite(d['y'][:, j]) & np.isfinite(d['D'][:, j, j]) & (d['D'][:, j, j] > 0)
                    sampled = d['sample_idx'][mask]
                    for sf in endpointfit['fit']:
                        ids = np.flatnonzero(d['strata'] == sf['stratum'])
                        s = np.searchsorted(ids, sampled[np.isin(sampled, ids)])
                        u = np.setdiff1d(np.arange(len(ids)), s)
                        W = overrides[j][np.ix_(ids, ids)]; rho = sf['rho']
                        if np.any(W[np.ix_(s, u)]):
                            raise RuntimeError('Published sampled rows link to non-sampled domains')
                        inv = np.linalg.inv(np.eye(len(ids)) - rho * W)
                        C = inv @ inv.T
                        invD = np.linalg.inv(np.eye(len(s)) - rho * W[np.ix_(s, s)])
                        CD = invD @ invD.T
                        relative = float(abs(C[np.ix_(s, s)] - CD).max()) / max(float(abs(CD).max()), 1.)
                        maxima['sampled_covariance_relative'] = max(maxima['sampled_covariance_relative'], relative)
                        cols = [d['covariate_names'].tolist().index(c) for c in sf['covariates']]
                        X = d['X_raw'][ids][:, cols]; beta = np.asarray(sf['beta'])
                        F = np.linalg.solve(np.eye(len(u)) - rho * W[np.ix_(u, u)], rho * W[np.ix_(u, s)])
                        expected = X[u] @ beta + F @ (p['mean'][ids[s], j] - X[s] @ beta)
                        if len(u):
                            maxima['directed_formula'] = max(maxima['directed_formula'],
                                float(abs(expected - p['mean'][ids[u], j]).max()))
        for name in ('Sikov_FH', 'Sikov_SAR', 'Sikov_SAR_Published'):
            p, bc = predictions[name], predictions[name + '_BC']
            prefix = 'published' if name == 'Sikov_SAR_Published' else 'original'
            bc_status = json.loads((out / prefix / (name + '_BC_status.json')).read_text())
            if record['convergence'][name + '_BC'] != record['convergence'][name] or \
                    bc_status['converged'] != record['convergence'][name]:
                raise RuntimeError('Supplemental BC convergence flag differs from shared fit')
            naive = p['mse_naive'] if 'mse_naive' in p else bc['mse_naive']
            if not np.array_equal(p['mean'], bc['mean']) or \
                    not np.array_equal(p['mse'], naive) or \
                    not np.array_equal(bc['mse'], bc['mse_bc']) or \
                    any(not np.allclose(bc[bound], bc['mean'] + sign * Z *
                        np.sqrt(np.maximum(bc['mse_bc'], 0)), atol=1e-12, rtol=0)
                        for bound, sign in (('lower', -1), ('upper', 1))):
                raise RuntimeError('Supplemental BC sensitivity changed estimator or bootstrap law')
            negative += int((bc['mse_bc'] < 0).sum())
        p, sar, icc = predictions['Selected'], predictions['Sikov_SAR'], predictions['ICC']
        expected_mean = .75 * icc['mean'] + .25 * sar['mean']
        maxima['mixture_mean'] = max(maxima['mixture_mean'], float(abs(p['mean'] - expected_mean).max()))
        if not np.allclose(p['mean'], expected_mean, atol=1e-12, rtol=0) or \
                record['convergence']['Selected'] != (record['convergence']['ICC']
                                                       and record['convergence']['Sikov_SAR']):
            raise RuntimeError('Frozen 75/25 mixture or convergence rule changed')
        expected_variance = .75 * icc['mse'] + .25 * sar['mse'] + \
            .75 * .25 * (icc['mean'] - sar['mean']) ** 2
        if not np.allclose(p['p_variance'], expected_variance, atol=1e-12, rtol=0):
            raise RuntimeError('Supplemental mixture variance omitted component disagreement')
        draws = dict(np.load(out / 'original/ICC_conditional_draws.npz', allow_pickle=False))
        for j, endpoint in enumerate(ENDPOINTS):
            samples = draws[endpoint + '_marginal']
            sd = np.sqrt(sar['mse'][:, j]); mu = sar['mean'][:, j]
            if samples.shape != (icc_status['endpoints'][j]['posterior_draws'], 120) or \
                    not np.isfinite(samples).all() or \
                    any(not np.allclose(icc[bound][:, j], np.quantile(samples, target, axis=0),
                        atol=1e-12, rtol=0) for bound, target in (('lower', .025), ('upper', .975))):
                raise RuntimeError('Supplemental private marginal draws malformed')
            for bound, target in (('lower', .025), ('upper', .975)):
                q = p[bound][:, j]
                gaussian = np.where(sd > 0, ndtr((q - mu) / np.maximum(sd,
                    np.finfo(float).tiny)), (q >= mu).astype(float))
                mixture = .75 * (samples <= q).mean(axis=0) + .25 * gaussian
                if np.any(mixture < target - 1e-10) or \
                        np.any(mixture > target + .75 / len(samples) + 1e-9):
                    raise RuntimeError('Supplemental mixture CDF quantile failed independent inversion')
        items.append({'replication': rep, 'context': CONTEXT,
            'completion_sha256': sha(out / 'completion.json'),
            'all_converged': all(record['convergence'].values()),
            'output_sha256': record['output_sha256']})
    if maxima['directed_formula'] > 1e-8 or maxima['sampled_covariance_relative'] > 1e-11:
        raise RuntimeError('Supplemental Published covariance/extension algebra failed')
    lock = {'status': 'PASS_COMPLETE_64_SUPPLEMENTAL_PREDICTION_SEAL',
        'protocol_sha256': plan['protocol_sha256'], 'addendum_sha256': plan['addendum_sha256'],
        'sensitivity_plan_sha256': plansha, 'worlds': 64, 'methods': list(METHODS),
        'all_expected_worlds': True, 'approved_for_scoring': False,
        'requires_principal_JOINT_and_independent_audit': True,
        'fresh_truth_read_before_lock': False, 'prediction_items': items,
        'convergence_failures_retained': retained, 'selected_recipe_unchanged': True}
    if LOCK.exists():
        if json.loads(LOCK.read_text()) != lock:
            raise RuntimeError('Existing supplemental lock differs')
    else:
        save(LOCK, lock)
    result = {'status': 'PASS_INDEPENDENT_64_SUPPLEMENTAL_PREDICTION_AUDIT',
        'protocol_sha256': plan['protocol_sha256'], 'addendum_sha256': plan['addendum_sha256'],
        'sensitivity_plan_sha256': plansha, 'prediction_lock_sha256': sha(LOCK),
        'worlds': 64, 'methods': list(METHODS), 'truth_values_read': False,
        'independent_max_errors': maxima, 'bootstrap_requested': requested,
        'bootstrap_failed': failed, 'bootstrap_stationarity_recovered': recovered,
        'rho_boundary_fits': boundaries, 'negative_BC_MSE_cells': negative,
        'convergence_failures_retained': retained, 'no_worlds_or_methods_excluded': True,
        'auditor_source_sha256': sha(__file__), 'dense_checks': checks}
    save(PREDICTION_AUDIT, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'dense_checks'}, indent=2), flush=True)
    return result


def supplemental_scoring_gate():
    plan, plansha, jointsha, mainauditsha = principal_scoring_gate()
    lock = json.loads(LOCK.read_text())
    audit = json.loads(PREDICTION_AUDIT.read_text())
    if lock.get('status') != 'PASS_COMPLETE_64_SUPPLEMENTAL_PREDICTION_SEAL' or \
            lock.get('worlds') != 64 or lock.get('fresh_truth_read_before_lock') is not False or \
            lock.get('all_expected_worlds') is not True or lock.get('methods') != list(METHODS) or \
            lock.get('sensitivity_plan_sha256') != plansha or \
            len(lock['prediction_items']) != 64 or \
            {i['replication'] for i in lock['prediction_items']} != set(IDS):
        raise RuntimeError('All 64 supplemental predictions must be sealed before truth access')
    if audit.get('status') != 'PASS_INDEPENDENT_64_SUPPLEMENTAL_PREDICTION_AUDIT' or \
            audit.get('prediction_lock_sha256') != sha(LOCK) or \
            audit.get('sensitivity_plan_sha256') != plansha or \
            audit.get('truth_values_read') is not False or \
            audit.get('auditor_source_sha256') != sha(__file__):
        raise RuntimeError('Independent supplemental pretruth audit incomplete')
    for item in lock['prediction_items']:
        out = DEST / 'fits' / f"rep_{item['replication']:05d}"
        if sha(out / 'completion.json') != item['completion_sha256']:
            raise RuntimeError('Supplemental completion changed after closure')
        completion = json.loads((out / 'completion.json').read_text())
        source = DEST / 'inputs' / f"rep_{item['replication']:05d}"
        if sha(source / 'manifest.json') != completion['input_manifest_sha256']:
            raise RuntimeError('Supplemental input manifest changed after pretruth audit')
        manifest = json.loads((source / 'manifest.json').read_text())
        if manifest['context'] != CONTEXT or manifest['replication'] != item['replication'] or \
                manifest['phase'] != 'prospective_descriptive_sensitivity' or \
                manifest['sensitivity_plan_sha256'] != plansha or \
                manifest['truth_excluded_from_model_inputs'] is not True or \
                manifest['population_stream'] != [20261010, 971, item['replication']] or \
                manifest['sampling_stream'] != [20261010, 972, item['replication']]:
            raise RuntimeError('Supplemental input role changed after pretruth audit')
        for name, key in (('inputs.npz', 'fit_inputs_sha256'),
                          ('sample.csv', 'sample_sha256')):
            if sha(source / 'model_inputs' / name) != completion[key] or \
                    completion[key] != manifest[key]:
                raise RuntimeError('Supplemental training arrays changed after pretruth audit')
        expected_outputs = {prefix + '/' + name + suffix for prefix, methods in
            (('original', METHODS[:9]), ('published', METHODS[9:]))
            for name in methods for suffix in ('.npz', '_status.json')}
        expected_outputs.add('original/ICC_conditional_draws.npz')
        if set(item['output_sha256']) != expected_outputs or \
                item['output_sha256'] != completion['output_sha256']:
            raise RuntimeError('Supplemental closed output support changed')
        for filename, digest in item['output_sha256'].items():
            if sha(out / filename) != digest:
                raise RuntimeError('Supplemental output changed after pretruth audit')
    return plan, {'sensitivity_plan_sha256': plansha,
        'prediction_lock_sha256': sha(LOCK), 'prediction_audit_sha256': sha(PREDICTION_AUDIT),
        'joint_prediction_lock_sha256': jointsha, 'principal_independent_audit_sha256': mainauditsha}


def audit_scores():
    plan, bindings = supplemental_scoring_gate()
    proof = json.loads(RESULT_STATUS.read_text())
    if proof.get('status') != 'PASS_COMPLETE_64_DESCRIPTIVE_SUPPLEMENTAL_SCORING' or \
            any(proof.get(k) != value for k, value in bindings.items()) or \
            proof.get('protocol_sha256') != plan['protocol_sha256'] or \
            proof.get('addendum_sha256') != plan['addendum_sha256'] or \
            proof.get('formal_p_values_or_new_Holm_family') is not False:
        raise RuntimeError('Supplemental scoring status/role differs')
    expected_files = {'descriptive_replications.csv', 'descriptive_summary.csv',
        'descriptive_paired_contrasts.csv', 'physical_projection_replications.csv',
        'physical_projection_summary.csv', 'physical_projection_paired_contrasts.csv'}
    if set(proof['outputs_sha256']) != expected_files:
        raise RuntimeError('Supplemental descriptive output support differs')
    results = DEST / 'resultados'
    for name, digest in proof['outputs_sha256'].items():
        if sha(results / name) != digest:
            raise RuntimeError('Supplemental descriptive output changed')
    recomputed = {False: [], True: []}
    for rep in IDS:
        source = DEST / 'inputs' / f'rep_{rep:05d}'
        manifest = json.loads((source / 'manifest.json').read_text())
        truthfile = source / 'evaluation_only/truth.npz'
        if sha(truthfile) != manifest['truth_sha256']:
            raise RuntimeError('Supplemental population truth changed')
        theta = np.load(truthfile, allow_pickle=False)['theta']
        d = dict(np.load(source / 'model_inputs/inputs.npz', allow_pickle=False))
        observed = np.zeros(theta.shape, bool)
        observed[d['sample_idx']] = np.isfinite(d['y'])
        unsampled = ~np.isin(np.arange(120), d['sample_idx'])
        for name in METHODS:
            out = DEST / 'fits' / f'rep_{rep:05d}' / ('original' if name in METHODS[:9] else 'published')
            prediction = dict(np.load(out / (name + '.npz'), allow_pickle=False))
            for j, endpoint in enumerate(ENDPOINTS):
                for group, mask in (('all', np.ones(120, bool)),
                    ('observed_endpoint', observed[:, j]), ('unsampled_common', unsampled)):
                    target = theta[mask, j]
                    for physical in (False, True):
                        mean, lo, hi = (prediction[k][mask, j] for k in ('mean', 'lower', 'upper'))
                        if physical:
                            mean, lo, hi = (np.clip(a, 0, 1) for a in (mean, lo, hi))
                        error = mean - target
                        losses = hi - lo + 40 * np.maximum(lo - target, 0) + 40 * np.maximum(target - hi, 0)
                        recomputed[physical].append(dict(scenario=CONTEXT, replication=rep,
                            method=name, outcome=endpoint, group=group, mse=np.mean(error ** 2),
                            mae=np.mean(abs(error)), bias=np.mean(error), interval_score=np.mean(losses),
                            coverage=np.mean((lo <= target) & (target <= hi)), width=np.mean(hi - lo)))
    errors = {}
    for physical in (False, True):
        prefix = 'physical_projection' if physical else 'descriptive'
        own = pd.DataFrame(recomputed[physical])
        actual = pd.read_csv(results / (prefix + '_replications.csv'))
        if actual[KEYS].duplicated().any() or len(actual) != 64 * 11 * 2 * 3:
            raise RuntimeError('Supplemental score rows missing/duplicated')
        if not np.isfinite(actual[list(METRICS)].to_numpy(float)).all():
            raise RuntimeError('Supplemental scores must be completely finite')
        joined = own.merge(actual, on=KEYS, validate='one_to_one', suffixes=('_own', '_actual'))
        if len(joined) != len(own):
            raise RuntimeError('Supplemental raw/projected support differs')
        errors[prefix + '_scores'] = max(float(abs(joined[k + '_own'] - joined[k + '_actual']).max()) for k in METRICS)
        summary = pd.read_csv(results / (prefix + '_summary.csv'))
        contrasts = pd.read_csv(results / (prefix + '_paired_contrasts.csv'))
        if any('p_' in c or 'Holm' in c for table in (actual, summary, contrasts) for c in table.columns):
            raise RuntimeError('Supplemental sensitivity must remain descriptive without p values')
        if len(summary) != 11 * 3 * 3 or summary[['method', 'outcome', 'group']].duplicated().any():
            raise RuntimeError('Supplemental endpoint/balanced summary support differs')
        expected_summary = {(m, o, g) for m in METHODS for o in
            (*ENDPOINTS, 'balanced_anemia_stunting') for g in
            ('all', 'observed_endpoint', 'unsampled_common')}
        summary_numeric = ['rmse', 'rmse_mcse', *METRICS,
                           *(key + '_mcse' for key in METRICS)]
        if set(summary[['method', 'outcome', 'group']].itertuples(index=False, name=None)) != expected_summary or \
                not np.isfinite(summary[summary_numeric].to_numpy(float)).all():
            raise RuntimeError('Supplemental summary support or finite metrics differ')
        for row in summary.to_dict('records'):
            part = own[(own.method == row['method']) & (own.group == row['group'])]
            if row['outcome'] != 'balanced_anemia_stunting':
                part = part[part.outcome == row['outcome']]
            perworld = part.groupby('replication')[list(METRICS)].mean().reindex(IDS)
            mean = perworld.mean(); se = perworld.std(ddof=1) / np.sqrt(64)
            expected = {'rmse': float(np.sqrt(mean.mse)),
                'rmse_mcse': float(se.mse / (2 * np.sqrt(mean.mse))) if mean.mse > 0 else 0.,
                **{key: float(mean[key]) for key in METRICS},
                **{key + '_mcse': float(se[key]) for key in METRICS}}
            if row['worlds'] != 64:
                raise RuntimeError('Supplemental summary cohort size differs')
            errors[prefix + '_summary'] = max(errors.get(prefix + '_summary', 0.),
                max(abs(float(row[key]) - value) for key, value in expected.items()))
        if len(contrasts) != 10 * 3 * 7 or contrasts[['reference', 'group', 'metric']].duplicated().any():
            raise RuntimeError('Supplemental paired descriptive contrast support differs')
        expected_contrasts = {(m, g, metric) for m in METHODS if m != 'Selected'
            for g in ('all', 'observed_endpoint', 'unsampled_common')
            for metric in (*METRICS, 'RMSE_ratio')}
        if set(contrasts[['reference', 'group', 'metric']].itertuples(index=False, name=None)) != expected_contrasts:
            raise RuntimeError('Supplemental contrast support differs')
        for row in contrasts.to_dict('records'):
            if row['candidate'] != 'Selected' or row['paired_worlds'] != 64 or \
                    not isinstance(row['estimate_defined'], (bool, np.bool_)):
                raise RuntimeError('Supplemental paired descriptive candidate/cohort/definition flag differs')
            part = own[own.group == row['group']]
            candidate = part[part.method == 'Selected'].groupby('replication')[list(METRICS)].mean().reindex(IDS)
            reference = part[part.method == row['reference']].groupby('replication')[list(METRICS)].mean().reindex(IDS)
            if row['metric'] == 'RMSE_ratio':
                a, b = candidate.mse.mean(), reference.mse.mean()
                if a == 0 or b == 0:
                    if bool(row['estimate_defined']) or row['undefined_reason'] != 'zero_candidate_or_reference_mean_MSE' or \
                            not all(np.isnan(float(row[k])) for k in
                                ('difference', 'mcse', 'MC_CI95_low', 'MC_CI95_high', 'mcse_log_ratio')):
                        raise RuntimeError('Undefined zero-MSE ratio must be explicit with no invented estimate or CI')
                    continue
                logratio = .5 * np.log(a / b)
                logse = float((.5 * (candidate.mse / a - reference.mse / b)).std(ddof=1) / np.sqrt(64))
                estimate = float(np.exp(logratio)); se = estimate * logse
                lo = float(np.exp(logratio - Z * logse)); hi = float(np.exp(logratio + Z * logse))
            else:
                delta = candidate[row['metric']] - reference[row['metric']]
                estimate = float(delta.mean()); se = float(delta.std(ddof=1) / np.sqrt(64))
                lo, hi = estimate - Z * se, estimate + Z * se
            expected = {'difference': estimate, 'mcse': se, 'MC_CI95_low': lo, 'MC_CI95_high': hi}
            if row['metric'] == 'RMSE_ratio':
                expected['mcse_log_ratio'] = logse
            if not bool(row['estimate_defined']) or not np.isfinite([float(row[k]) for k in expected]).all():
                raise RuntimeError('Defined supplemental descriptive contrast must be finite')
            errors[prefix + '_contrasts'] = max(errors.get(prefix + '_contrasts', 0.),
                max(abs(float(row[key]) - value) for key, value in expected.items()))
    if max(errors.values()) > 1e-11:
        raise RuntimeError('Independent supplemental descriptive recomputation differs ' + str(errors))
    result = {'status': 'PASS_INDEPENDENT_64_DESCRIPTIVE_SUPPLEMENTAL_RESULT_AUDIT',
        'protocol_sha256': plan['protocol_sha256'], 'addendum_sha256': plan['addendum_sha256'],
        **bindings, 'worlds': 64, 'methods': list(METHODS), 'score_rows_per_scale': 4224,
        'no_worlds_or_methods_excluded': True, 'formal_p_values_or_new_Holm_family': False,
        'original_256_and_Holm3_Holm12_unchanged': True, 'selected_recipe_unchanged': True,
        'independent_max_errors': errors, 'outputs_sha256': proof['outputs_sha256'],
        'auditor_source_sha256': sha(__file__)}
    save(RESULT_AUDIT, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores', action='store_true')
    args = parser.parse_args()
    audit_scores() if args.scores else audit_predictions()
