"""Separately frozen Published-SAR B300 descriptive interval sensitivity.

Eight fixed worlds, unchanged B100 point estimator and unchanged Selected recipe.
Native fits read training arrays only; targets require both public release gates.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sikov_published_reference import fit_sikov_published_bootstrap, METHOD, BC_METHOD
from sikov_poverty_stratum_descriptive import (score_array, with_endpoint_balance,
    summarize_populations, gate_function_digests, METRICS, GROUPS, ENDPOINTS)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
PLAN = ROOT / 'validacion/sikov_published_b300_plan.json'
QA = ROOT / 'validacion/sikov_published_b300_qa.json'
NATIVE = ROOT / 'validacion/sikov_published_b300_native_status.json'
STATUS = ROOT / 'validacion/sikov_published_b300_status.json'
OUT = ROOT / 'resultados/published_b300_descriptive'
DEPENDENCY = ROOT / 'scripts/sikov_poverty_stratum_descriptive.py'
CONTEXTS = ('SAR_probability_srs', 'SAR_logit_srs', 'SAR_logit_psu_sparse', 'IID_logit_psu_sparse')
REPS = (30000, 30001)
JOBS = tuple((c, r) for c in CONTEXTS for r in REPS)
MODELS = ('Selected_B100', 'Published_B100', 'Published_B100_BC', 'Published_B300', 'Published_B300_BC')
REFERENCES = MODELS[1:]
KEYS = ('mean', 'lower', 'upper', 'mse', 'mse_plugin', 'mse_delta', 'mse_naive', 'mse_bc', 'naive_mcse', 'bc_mcse')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve completed evidence ' + str(path))
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item() if isinstance(x, np.generic) else str(x)) + '\n')
    temporary.replace(path)


def source_gate():
    p = read(PLAN)
    if p['status'] != 'FROZEN_PUBLISHED_B300_DESCRIPTIVE_PLAN_BEFORE_ACCURACY':
        raise RuntimeError('Separate prospective plan required')
    if p['source_sha256'] != sha(SOURCE) or p['QA_sha256'] != sha(QA) or p['score_dependency_sha256'] != sha(DEPENDENCY):
        raise RuntimeError('Changed supplementary statistical source or QA')
    if p['gate_function_text_sha256'] != gate_function_digests():
        raise RuntimeError('Changed mandatory pre-truth gate body')
    if p['protocol_sha256'] != sha(ROOT / 'protocol_freeze.json') or p['addendum_sha256'] != sha(ROOT / 'validacion/sikov_published_addendum_freeze.json'):
        raise RuntimeError('Changed base statistical seals')
    for path, digest in p['statistical_source_identities_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('Changed one of 37 frozen statistical sources: ' + path)
    for path, digest in p['frozen_inputs_and_runtime_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('Changed frozen input/runtime: ' + path)
    return p


def folders(context, rep):
    if (context, rep) not in JOBS:
        raise ValueError('Only the eight prospective worlds are allowed')
    return (ROOT / 'evaluacion/inputs' / context / f'rep_{rep:05d}',
            ROOT / 'evaluacion/fits' / context / f'rep_{rep:05d}',
            ROOT / 'evaluacion/published_sar' / context / f'rep_{rep:05d}',
            ROOT / 'evaluacion/published_b300' / context / f'rep_{rep:05d}')


def case_gate(context, rep):
    """Validate closed training/sample/prediction bytes without reading truth."""
    p = source_gate()
    source, original, published, output = folders(context, rep)
    manifest_path = source / 'manifest.json'
    manifest = read(manifest_path)
    old = read(original / 'completion.json')
    pub = read(published / 'completion.json')
    model = source / 'model_inputs/inputs.npz'
    # Fixed filename from the frozen generator; no path from metadata.
    sample = source / 'model_inputs/sample.csv'
    if (manifest['context'] != context or manifest['replication'] != rep or manifest['phase'] != 'evaluation'
            or manifest['population_stream'] != [20261010, 701, CONTEXTS.index(context), rep]
            or manifest['sampling_stream'] != [20261010, 702, CONTEXTS.index(context), rep]
            or manifest['truth_excluded_from_model_inputs'] is not True
            or old['truth_read_by_fits'] is not False or pub['truth_read_by_fits'] is not False):
        raise RuntimeError('Wrong world or training-only separation')
    if not sample.is_file():
        raise RuntimeError('Frozen sample record file missing')
    if (old['protocol_sha256'] != p['protocol_sha256'] or pub['protocol_sha256'] != p['protocol_sha256']
            or pub['addendum_sha256'] != p['addendum_sha256']
            or pub['original_completion_sha256'] != sha(original / 'completion.json')
            or old['input_manifest_sha256'] != sha(manifest_path) or pub['input_manifest_sha256'] != sha(manifest_path)
            or old['fit_inputs_sha256'] != sha(model) or pub['fit_inputs_sha256'] != sha(model)
            or manifest['fit_inputs_sha256'] != sha(model)
            or manifest['sample_sha256'] != sha(sample) or old['sample_sha256'] != sha(sample)):
        raise RuntimeError('Changed closed input, sample or completion')
    for folder, completion in ((original, old), (published, pub)):
        for name, digest in completion['output_sha256'].items():
            if Path(name).name != name or sha(folder / name) != digest:
                raise RuntimeError('Changed closed prediction/status output')
    return p, source, original, published, output, manifest, old, pub


def bc_arrays(primary):
    bc = {key: value.copy() for key, value in primary.items()}
    bc['mse'] = bc['mse_bc'].copy()
    se = np.sqrt(np.maximum(bc['mse'], 0))
    bc['lower'] = bc['mean'] - 1.959963984540054 * se
    bc['upper'] = bc['mean'] + 1.959963984540054 * se
    return bc


def audit_native(d, status, arrays, B):
    from sikov_published_audit import independent_triangular
    from sikov_final_audit import dense_gaussian_check
    # This distinct method label avoids the base auditor's deliberately fixed
    # B100 metadata branch. B300 counts are independently required below.
    check = dense_gaussian_check(d, status, 'Published_B300', arrays,
                                W_override=independent_triangular(d))
    requested = failed = recovered = negative = 0
    for endpoint in status['fits']:
        if endpoint['bootstrap_B'] != B:
            raise RuntimeError('Wrong endpoint bootstrap size')
        for sf in endpoint['fit']:
            if (sf['bootstrap_B'] != B or sf['bootstrap_success'] < .95 * B
                    or sf['bootstrap_success'] + sf['bootstrap_failed'] != B):
                raise RuntimeError('Wrong B300 draw accounting or <95% successful draws')
            requested += B; failed += sf['bootstrap_failed']; recovered += sf['bootstrap_stationarity_recovered']
    if not np.array_equal(arrays['mse'], arrays['mse_naive']):
        raise RuntimeError('Naive primary convention differs')
    if not np.isfinite(np.stack([arrays[k] for k in KEYS])).all():
        raise RuntimeError('Nonfinite native/BC values')
    if not np.allclose(arrays['lower'], arrays['mean'] - 1.959963984540054 * np.sqrt(np.maximum(arrays['mse'], 0)), atol=1e-12, rtol=0):
        raise RuntimeError('Native interval convention differs')
    negative = int((arrays['mse_bc'] < 0).sum())
    return {**check, 'bootstrap_requested_B300': requested, 'bootstrap_failed_B300': failed,
            'bootstrap_recovered_B300': recovered, 'negative_raw_BC_MSE_cells': negative,
            'truth_values_read': False}


def work(job):
    context, rep = job
    p, source, original, published, output, manifest, old, pub = case_gate(context, rep)
    if output.exists():
        raise FileExistsError('Preserve existing B300 case ' + str(output))
    d = dict(np.load(source / 'model_inputs/inputs.npz', allow_pickle=False))
    seed = int(np.random.SeedSequence([20261010, 975, CONTEXTS.index(context), rep]).generate_state(1)[0])
    begin = time.perf_counter(); values = []; masks = []
    for j, endpoint in enumerate(ENDPOINTS):
        mask = np.isfinite(d['y'][:, j]) & np.isfinite(d['D'][:, j, j]) & (d['D'][:, j, j] > 0)
        masks.append(mask)
        values.append(fit_sikov_published_bootstrap(d['X_raw'], d['y'][mask, j], d['D'][mask, j, j],
            d['sample_idx'][mask], d['coords'], d['strata'], outcome=endpoint,
            covariate_names=d['covariate_names'].astype(str).tolist(), B=300,
            seed=seed + 1009 * j, interval_estimator='naive'))
    arrays = {key: np.column_stack([value[key] for value in values]) for key in KEYS}
    status = {'converged': all(v['converged'] for v in values), 'fits': values,
              'endpoint_training_rows': [int(mask.sum()) for mask in masks]}
    closed = dict(np.load(published / (METHOD + '.npz'), allow_pickle=False))
    difference = float(abs(arrays['mean'] - closed['mean']).max())
    if difference > 1e-10:
        raise RuntimeError('B300 changed the closed Published B100 point estimator')
    native_check = audit_native(d, status, arrays, 300)
    case_gate(context, rep)
    output.mkdir(parents=True)
    np.savez_compressed(output / 'predictions.npz', **arrays)
    save(output / 'status.json', status)
    completion = {'context': context, 'replication': rep, 'B': 300, 'bootstrap_seed': seed,
        'protocol_sha256': p['protocol_sha256'], 'addendum_sha256': p['addendum_sha256'],
        'plan_sha256': sha(PLAN), 'source_sha256': sha(SOURCE), 'QA_sha256': sha(QA),
        'input_manifest_sha256': sha(source / 'manifest.json'), 'fit_inputs_sha256': sha(source / 'model_inputs/inputs.npz'),
        'original_completion_sha256': sha(original / 'completion.json'),
        'published_completion_sha256': sha(published / 'completion.json'),
        'published_B100_predictions_sha256': sha(published / (METHOD + '.npz')),
        'truth_read_by_fits': False, 'closed_point_max_absolute_difference': difference,
        'all_converged': status['converged'], 'convergence_failure_preserved': not status['converged'],
        'naive_and_BC_use_same_draws': True, 'raw_negative_BC_kept': True,
        'seconds': time.perf_counter() - begin, 'independent_native_audit': native_check,
        'output_sha256': {name: sha(output / name) for name in ('predictions.npz', 'status.json')}}
    save(output / 'completion.json', completion)
    return completion


def closed_native(job):
    context, rep = job
    p, source, original, published, output, manifest, old, pub = case_gate(context, rep)
    completion = read(output / 'completion.json')
    expected = {'protocol_sha256': p['protocol_sha256'], 'addendum_sha256': p['addendum_sha256'],
        'plan_sha256': sha(PLAN), 'source_sha256': sha(SOURCE), 'QA_sha256': sha(QA),
        'input_manifest_sha256': sha(source / 'manifest.json'), 'fit_inputs_sha256': sha(source / 'model_inputs/inputs.npz'),
        'original_completion_sha256': sha(original / 'completion.json'),
        'published_completion_sha256': sha(published / 'completion.json'),
        'published_B100_predictions_sha256': sha(published / (METHOD + '.npz'))}
    if any(completion[key] != value for key, value in expected.items()) or completion['truth_read_by_fits'] is not False:
        raise RuntimeError('Changed native B300 case binding')
    if completion['context'] != context or completion['replication'] != rep or completion['B'] != 300:
        raise RuntimeError('Wrong native B300 case')
    seed = int(np.random.SeedSequence([20261010, 975, CONTEXTS.index(context), rep]).generate_state(1)[0])
    if completion['bootstrap_seed'] != seed or set(completion['output_sha256']) != {'predictions.npz', 'status.json'}:
        raise RuntimeError('Wrong native B300 random stream or output support')
    for name, digest in completion['output_sha256'].items():
        if Path(name).name != name or sha(output / name) != digest:
            raise RuntimeError('Changed native B300 output')
    arrays = dict(np.load(output / 'predictions.npz', allow_pickle=False))
    d = dict(np.load(source / 'model_inputs/inputs.npz', allow_pickle=False))
    checked = audit_native(d, read(output / 'status.json'), arrays, 300)
    closed = np.load(published / (METHOD + '.npz'), allow_pickle=False)['mean']
    if float(abs(arrays['mean'] - closed).max()) > 1e-10:
        raise RuntimeError('Saved B300 points differ from B100')
    return completion, checked


def native_run(workers=4):
    source_gate()
    if not 1 <= workers <= 4:
        raise ValueError('Workers must be one to four')
    for job in JOBS:
        case_gate(*job)
    jobs = [job for job in JOBS if not (folders(*job)[3] / 'completion.json').is_file()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for completion in pool.map(work, jobs):
            print(json.dumps({key: completion[key] for key in ('context', 'replication', 'seconds', 'all_converged', 'closed_point_max_absolute_difference')}), flush=True)
    cases = []; maximum = 0.; failures = []
    for job in JOBS:
        completion, check = closed_native(job)
        maximum = max(maximum, completion['closed_point_max_absolute_difference'])
        if not completion['all_converged']:
            failures.append(list(job))
        cases.append({'context': job[0], 'replication': job[1], 'completion_sha256': sha(folders(*job)[3] / 'completion.json'),
                      'independent_native_audit': check})
    result = {'status': 'PASS_EIGHT_CLOSED_PUBLISHED_B300_NATIVE_CASES', 'worlds': 8, 'B': 300,
        'plan_sha256': sha(PLAN), 'source_sha256': sha(SOURCE), 'QA_sha256': sha(QA),
        'truth_values_read': False, 'points_unchanged_max_absolute_difference': maximum,
        'convergence_failures_retained': failures, 'all_cases_retained': True, 'cases': cases,
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    if NATIVE.exists():
        existing = read(NATIVE)
        if existing['cases'] != cases or existing['plan_sha256'] != sha(PLAN):
            raise RuntimeError('Existing native certificate differs')
        return existing
    save(NATIVE, result)
    return result


def score_cells(theta, d, predictions, convergence, context, rep):
    n = len(d['X_raw'])
    if theta.shape != (n, 2) or not np.isfinite(theta).all() or np.any((theta < 0) | (theta > 1)):
        raise ValueError('Same bounded finite-population target required')
    observed = np.zeros((n, 2), bool); observed[d['sample_idx']] = np.isfinite(d['y'])
    unsampled = ~np.isin(np.arange(n), d['sample_idx'])
    rows = []
    for method, prediction in predictions.items():
        if any(prediction[key].shape != (n, 2) for key in ('mean', 'lower', 'upper')):
            raise ValueError('Same full prediction universe required')
        for j, endpoint in enumerate(ENDPOINTS):
            for group, mask in (('all', np.ones(n, bool)), ('observed_endpoint', observed[:, j]), ('unsampled_common', unsampled)):
                rows.append({'scenario': context, 'replication': rep, 'method': method, 'outcome': endpoint,
                    'poverty_stratum': 0, 'poverty_stratum_label': 'all_strata', 'group': group,
                    'domains': int(mask.sum()), 'converged': bool(convergence[method]),
                    **score_array(theta[mask, j], *(prediction[key][mask, j] for key in ('mean', 'lower', 'upper')))})
    return rows


def paired_scores(balanced, contexts, reps):
    keys = ['scenario', 'replication', 'outcome', 'poverty_stratum', 'poverty_stratum_label', 'group']
    selected = balanced[balanced.method == MODELS[0]]; rows = []
    for ref in REFERENCES:
        baseline = balanced[balanced.method == ref]
        paired = selected.merge(baseline, on=keys, validate='one_to_one', suffixes=('_selected', '_ref'))
        if len(paired) != len(selected) or len(paired) != len(baseline):
            raise ValueError('Incomplete paired descriptive worlds')
        differences = paired[keys].copy(); differences['method'] = MODELS[0] + '_minus_' + ref
        for metric in METRICS:
            differences[metric] = paired[metric + '_selected'] - paired[metric + '_ref']
        summary = summarize_populations(differences, contexts, reps)
        summary['reference'] = ref
        summary = summary.rename(columns={'estimate': 'difference_Selected_B100_minus_reference'})
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def score():
    p = source_gate()
    from public_verify import prediction_preflight, completed_result_gates
    meta = prediction_preflight(); completed_result_gates(meta)
    if meta['protocol_sha256'] != p['protocol_sha256'] or meta['addendum_sha256'] != p['addendum_sha256']:
        raise RuntimeError('Complete gates refer to different statistical seals')
    native = read(NATIVE)
    if (native['status'] != 'PASS_EIGHT_CLOSED_PUBLISHED_B300_NATIVE_CASES' or native['plan_sha256'] != sha(PLAN)
            or native['source_sha256'] != sha(SOURCE) or native['QA_sha256'] != sha(QA)
            or native['worlds'] != 8 or native['B'] != 300 or native['truth_values_read']
            or len(native['cases']) != 8
            or [(item['context'], item['replication']) for item in native['cases']] != list(JOBS)):
        raise RuntimeError('Eight closed B300 cases required')
    # Revalidate every input and closed prediction before the first truth hash.
    for job, item in zip(JOBS, native['cases']):
        closed_native(job)
        if item['context'] != job[0] or item['replication'] != job[1] or item['completion_sha256'] != sha(folders(*job)[3] / 'completion.json'):
            raise RuntimeError('Native certificate case binding differs')
    if OUT.exists() or STATUS.exists():
        raise FileExistsError('Preserve completed B300 descriptive outputs')
    records = []; bindings = {}
    for context, rep in JOBS:
        _, source, original, published, output, manifest, old, pub = case_gate(context, rep)
        truth = source / 'evaluation_only/truth.npz'
        if sha(truth) != manifest['truth_sha256']:
            raise RuntimeError('Changed jointly closed finite-population truth')
        bindings[str(truth)] = sha(truth)
        theta = np.load(truth, allow_pickle=False)['theta']
        d = dict(np.load(source / 'model_inputs/inputs.npz', allow_pickle=False))
        primary = dict(np.load(output / 'predictions.npz', allow_pickle=False))
        predictions = {MODELS[0]: dict(np.load(original / 'Selected.npz', allow_pickle=False)),
            MODELS[1]: dict(np.load(published / (METHOD + '.npz'), allow_pickle=False)),
            MODELS[2]: dict(np.load(published / (BC_METHOD + '.npz'), allow_pickle=False)),
            MODELS[3]: primary, MODELS[4]: bc_arrays(primary)}
        convergence = {MODELS[0]: old['convergence']['Selected'], MODELS[1]: pub['convergence'][METHOD],
            MODELS[2]: pub['convergence'][BC_METHOD], MODELS[3]: read(output / 'completion.json')['all_converged'],
            MODELS[4]: read(output / 'completion.json')['all_converged']}
        records.extend(score_cells(theta, d, predictions, convergence, context, rep))
        for folder in (original, published, output):
            bindings[str(folder / 'completion.json')] = sha(folder / 'completion.json')
    frame = pd.DataFrame(records)
    keys = ['scenario', 'replication', 'method', 'outcome', 'group']
    if len(frame) != 8 * 5 * 2 * 3 or frame.duplicated(keys).any():
        raise RuntimeError('Incomplete eight-world descriptive support')
    balanced = with_endpoint_balance(frame)
    summary = summarize_populations(balanced, CONTEXTS, REPS)
    differences = paired_scores(balanced, CONTEXTS, REPS)
    OUT.mkdir(parents=True)
    for value, name in ((frame, 'per_population_scores.csv'), (summary, 'summary.csv'),
                        (differences, 'selected_reference_differences.csv')):
        value.drop(columns=['poverty_stratum', 'poverty_stratum_label']).to_csv(OUT / name, index=False)
    result = {'status': 'PASS_FROZEN_PUBLISHED_B300_DESCRIPTIVE_SENSITIVITY',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'protocol_sha256': p['protocol_sha256'],
        'addendum_sha256': p['addendum_sha256'], 'plan_sha256': sha(PLAN), 'source_sha256': sha(SOURCE), 'QA_sha256': sha(QA),
        'joint_prediction_lock_sha256': meta['joint_sha256'],
        'principal_independent_audit_sha256': sha(ROOT / 'validacion/sikov_final_independent_audit.json'),
        'native_status_sha256': sha(NATIVE), 'native_cases': native['cases'], 'worlds': 8, 'methods': list(MODELS),
        'per_population_rows': len(frame), 'summary_rows': len(summary), 'paired_difference_rows': len(differences),
        'truth_access_only_after_both_public_gates_and_principal_audit': True,
        'all_worlds_and_convergence_flags_retained': True, 'convergence_failure_score_rows_retained': int((~frame.converged).sum()),
        'models_points_weights_primary_B100_or_Holm_modified': False, 'hypothesis_tests_or_reselection': False,
        'new_Selected_mixture_with_Published_component': False, 'descriptive_nonconfirmatory_sensitivity': True,
        'input_and_closed_prediction_bindings_sha256': bindings,
        'outputs_sha256': {str(file): sha(file) for file in OUT.iterdir() if file.is_file()}}
    save(STATUS, result)
    return result


def qa():
    checks = []
    fake = {'X_raw': np.ones((6, 1)), 'sample_idx': np.array([0, 1, 3, 4]), 'y': np.full((4, 2), .5)}
    fake['y'][0, 0] = np.nan
    theta = np.full((6, 2), .5)
    pred = {name: {'mean': theta, 'lower': theta - .1, 'upper': theta + .1} for name in MODELS}
    rows = pd.DataFrame(score_cells(theta, fake, pred, {name: False for name in MODELS}, 'fake', 0))
    checks.append(('all_methods_endpoints_groups_retained', len(rows) == 30))
    checks.append(('endpoint_missingness_common_unsampled', rows[(rows.outcome == 'anemia') & (rows.group == 'observed_endpoint')].domains.eq(3).all()
                   and rows[rows.group == 'unsampled_common'].domains.eq(2).all()))
    checks.append(('finite_convergence_failure_not_removed', not rows.converged.any()))
    fake_bc = {key: np.ones((6, 2)) * .01 for key in KEYS}; fake_bc['mean'] = theta.copy(); fake_bc['mse_bc'][0, 0] = -.02
    bc = bc_arrays(fake_bc)
    checks.append(('BC_points_identical_raw_negative_kept', np.array_equal(bc['mean'], fake_bc['mean']) and bc['mse'][0, 0] == -.02))
    checks.append(('BC_floor_only_interval_sqrt', bc['lower'][0, 0] == .5 and bc['upper'][0, 0] == .5))
    fixture = []
    for c, context in enumerate(('fake_a', 'fake_b')):
        for rep in (0, 1):
            for i, method in enumerate(MODELS):
                for j, endpoint in enumerate(ENDPOINTS):
                    fixture.append({'scenario': context, 'replication': rep, 'method': method, 'outcome': endpoint,
                        'poverty_stratum': 0, 'poverty_stratum_label': 'all_strata', 'group': 'all', 'domains': 6, 'converged': True,
                        **{metric: .2 + c * .1 + rep * (.04 if j == 0 else -.02) + i * rep * .01 for metric in METRICS}})
    balanced = with_endpoint_balance(pd.DataFrame(fixture))
    summary = summarize_populations(balanced, ('fake_a', 'fake_b'), (0, 1))
    differences = paired_scores(balanced, ('fake_a', 'fake_b'), (0, 1))
    chosen = summary[(summary.method == MODELS[0]) & (summary.outcome == 'endpoint_balanced') & (summary.scenario == 'context_balanced') & (summary.metric == 'mse')].iloc[0]
    checks.append(('endpoint_covariance_world_MCSE', abs(chosen.mcse - np.sqrt(.000025 / 2)) < 1e-13))
    checks.append(('equal_context_summary_mean', abs(chosen.estimate - .255) < 1e-13))
    difference = differences[(differences.reference == MODELS[1]) & (differences.outcome == 'endpoint_balanced') & (differences.scenario == 'context_balanced') & (differences.metric == 'mse')].iloc[0]
    checks.append(('paired_world_difference_MCSE', abs(difference.difference_Selected_B100_minus_reference + .005) < 1e-13
                   and abs(difference.mcse - np.sqrt(.000025 / 2)) < 1e-13))
    checks.append(('no_p_confidence_or_Holm_columns', not any('Holm' in k or k.startswith('p_') or 'CI' in k for k in differences.columns)))
    try:
        summarize_populations(balanced.iloc[:-1], ('fake_a', 'fake_b'), (0, 1))
        checks.append(('missing_world_fails_without_omission', False))
    except ValueError:
        checks.append(('missing_world_fails_without_omission', True))
    seeds = [int(np.random.SeedSequence([20261010, 975, CONTEXTS.index(c), r]).generate_state(1)[0]) for c, r in JOBS]
    checks.append(('eight_distinct_prospective_native_seeds', len(set(seeds)) == 8))
    # Reuse the previously validated artificial contract frame. This is a
    # numerical contract fixture, never an evaluation population or selector.
    rng = np.random.default_rng(20261010945); n = 60
    artificial = {'X_raw': np.column_stack([np.ones(n), rng.uniform(size=(n, 14))]),
        'strata': np.repeat([1, 2, 3], 20),
        'sample_idx': np.concatenate([np.arange(i * 20, i * 20 + 15) for i in range(3)]),
        'coords': rng.uniform(size=(n, 2)),
        'covariate_names': np.array(['intercept', 'Altitud', 'PisoTierra', 'Cemento', 'Agua', 'Elect',
            'Desague', 'Internet', 'SIS', 'EsSalud', 'NoSeguro', 'Refrig', 'Castellano', 'Analfabet', 'Rural']),
        'y': rng.uniform(.12, .42, size=(45, 2)), 'D': np.zeros((45, 2, 2))}
    artificial['D'][:, 0, 0] = rng.uniform(.003, .008, 45)
    artificial['D'][:, 1, 1] = rng.uniform(.003, .008, 45)
    artificial['y'][[1, 17, 33], 0] = np.nan
    artificial['D'][[4, 21, 35], 1, 1] = np.nan
    values = []; native_masks = []
    for j, endpoint in enumerate(ENDPOINTS):
        mask = np.isfinite(artificial['y'][:, j]) & np.isfinite(artificial['D'][:, j, j]) & (artificial['D'][:, j, j] > 0)
        native_masks.append(mask)
        values.append(fit_sikov_published_bootstrap(artificial['X_raw'], artificial['y'][mask, j], artificial['D'][mask, j, j],
            artificial['sample_idx'][mask], artificial['coords'], artificial['strata'], outcome=endpoint,
            covariate_names=artificial['covariate_names'].tolist(), B=12, seed=20261010977 + j, interval_estimator='naive'))
    native_arrays = {key: np.column_stack([value[key] for value in values]) for key in KEYS}
    native_status = {'converged': all(value['converged'] for value in values), 'fits': values}
    native_proof = audit_native(artificial, native_status, native_arrays, 12)
    mask = native_masks[0]
    other_B = fit_sikov_published_bootstrap(artificial['X_raw'], artificial['y'][mask, 0], artificial['D'][mask, 0, 0],
        artificial['sample_idx'][mask], artificial['coords'], artificial['strata'], outcome='anemia',
        covariate_names=artificial['covariate_names'].tolist(), B=16, seed=20261010979, interval_estimator='naive')
    checks.append(('artificial_native_independent_dense_and_draw_accounting', native_proof['bootstrap_requested_B300'] == 72
                   and max(native_proof['maximum_errors'].values()) < 1e-7))
    checks.append(('different_B_and_seed_preserve_native_point_exactly', np.array_equal(other_B['mean'], values[0]['mean'])))
    checks.append(('native_BC_export_point_and_raw_MSE_contract', np.array_equal(bc_arrays(native_arrays)['mean'], native_arrays['mean'])
                   and np.array_equal(bc_arrays(native_arrays)['mse'], native_arrays['mse_bc'])))
    checks.append(('no_final_accuracy_certificates_present', not any((ROOT / ('validacion/' + name)).exists() for name in
        ('joint_prediction_lock.json', 'fresh_evaluation_status.json', 'sikov_published_evaluation_status.json', 'sikov_final_independent_audit.json'))))
    result = {'status': 'PASS_SYNTHETIC_PUBLISHED_B300_DESCRIPTIVE_QA' if all(value for _, value in checks) else 'FAIL',
        'checks_passed': sum(bool(value) for _, value in checks), 'checks': {name: bool(value) for name, value in checks},
        'source_sha256': sha(SOURCE), 'score_dependency_sha256': sha(DEPENDENCY), 'synthetic_arrays_only': True,
        'truth_or_final_accuracy_read': False, 'artificial_native_QA_B': [12, 16], 'artificial_native_independent_audit': native_proof,
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    save(QA, result)
    if result['status'] == 'FAIL':
        raise RuntimeError('Synthetic B300 QA failed; preserve failure evidence')
    return result


def freeze():
    proof = read(ROOT / 'protocol_freeze.json'); add = read(ROOT / 'validacion/sikov_published_addendum_freeze.json'); test = read(QA)
    if test['status'] != 'PASS_SYNTHETIC_PUBLISHED_B300_DESCRIPTIVE_QA' or test['source_sha256'] != sha(SOURCE):
        raise RuntimeError('Current helper needs synthetic QA before sealing')
    forbidden = ('joint_prediction_lock.json', 'fresh_evaluation_status.json', 'sikov_published_evaluation_status.json', 'sikov_final_independent_audit.json')
    if any((ROOT / 'validacion' / name).exists() for name in forbidden):
        raise RuntimeError('Separate B300 plan must be sealed before final accuracy')
    identities = {**proof['statistical_sources_sha256'], **add['additional_sources_sha256']}
    if len(identities) != 37 or any(sha(path) != digest for path, digest in identities.items()):
        raise RuntimeError('All 37 frozen statistical sources must remain intact')
    runtime = {}
    for frozen in (proof, add):
        for block in ('frozen_inputs_sha256', 'runtime_binary_sha256'):
            runtime.update(frozen.get(block, {}))
    if any(sha(path) != digest for path, digest in runtime.items()):
        raise RuntimeError('Frozen training/template/runtime identity changed')
    result = {'status': 'FROZEN_PUBLISHED_B300_DESCRIPTIVE_PLAN_BEFORE_ACCURACY',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'protocol_sha256': sha(ROOT / 'protocol_freeze.json'),
        'addendum_sha256': sha(ROOT / 'validacion/sikov_published_addendum_freeze.json'),
        'source_sha256': sha(SOURCE), 'QA_sha256': sha(QA), 'score_dependency_sha256': sha(DEPENDENCY),
        'statistical_source_identities_sha256': identities, 'original_frozen_source_count': 37,
        'frozen_inputs_and_runtime_sha256': runtime, 'gate_function_text_sha256': gate_function_digests(),
        'operational_package_inventory_and_gate_module_full_sha_not_frozen': True,
        'contexts': list(CONTEXTS), 'replications': list(REPS), 'worlds': 8, 'B': 300, 'primary_B100_unchanged': True,
        'seed_stream_role': [20261010, 975, 'context_index', 'replication'], 'endpoint_seed_offset': 1009,
        'uncertainty': 'naive parametric bootstrap B300; raw same-draw BC separately retained',
        'point_identity_tolerance_against_Published_B100': 1e-10,
        'models': list(MODELS), 'descriptive_references': list(REFERENCES), 'groups': list(GROUPS), 'metrics': list(METRICS),
        'native_pre_truth_gates': ['37 statistical identities', 'closed original and Published case predictions', 'closed training/sample identities', 'independent dense Gaussian and B300 draw accounting'],
        'score_pre_truth_gates': ['prediction_preflight', 'completed_result_gates', 'eight native completions and input/prediction identities'],
        'required_principal_audit_status': 'PASS_COMPLETE_COMPARISON_INDEPENDENT_AUDIT',
        'target': 'same finite-population prevalence theta as original main experiment',
        'group_masks': 'all; sampled finite-y per endpoint; common areas absent from sample_idx',
        'mcse_unit': 'population/world; endpoints averaged before MCSE; four equally weighted contexts',
        'context_balanced_worlds': 8, 'per_context_worlds': 2,
        'limited_MC_precision_warning': 'Only two worlds per context; descriptive MCSE is imprecise and no confirmatory inference is made',
        'rmse_summary': 'mean of within-population RMSE', 'additional_interval_or_point_clipping': False,
        'Selected_recipe': proof['selected_recipe'], 'new_Selected_with_Published_component': False,
        'reselection_tuning_hypothesis_tests_p_values_or_Holm_changes': False,
        'native_and_BC_share_bootstrap_draws': True, 'raw_negative_BC_kept': True,
        'population_or_finite_convergence_failure_exclusions': False,
        'actual_accuracy_or_truth_read_before_plan': False,
        'existing_fullgraph_B300_eight_world_sensitivity_preserved': True}
    save(PLAN, result)
    return result


def wait(workers):
    source_gate()
    print('Waiting for the eight original and Published case seals; no truth read.', flush=True)
    while not all((folders(c, r)[1] / 'completion.json').is_file() and (folders(c, r)[2] / 'completion.json').is_file() for c, r in JOBS):
        time.sleep(30)
    native_run(workers)
    print('Native B300 closed; waiting for the complete principal independent audit.', flush=True)
    audit = ROOT / 'validacion/sikov_final_independent_audit.json'
    while True:
        if audit.is_file():
            try:
                certificate = read(audit)
            except json.JSONDecodeError:
                time.sleep(30); continue
            if certificate['status'] != 'PASS_COMPLETE_COMPARISON_INDEPENDENT_AUDIT':
                raise RuntimeError('Principal audit is not complete PASS')
            break
        time.sleep(30)
    return score()


def main():
    parser = argparse.ArgumentParser(); action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--qa', action='store_true'); action.add_argument('--freeze-plan', action='store_true')
    action.add_argument('--native', action='store_true'); action.add_argument('--score', action='store_true'); action.add_argument('--wait', action='store_true')
    parser.add_argument('--workers', type=int, default=4); args = parser.parse_args()
    result = qa() if args.qa else freeze() if args.freeze_plan else native_run(args.workers) if args.native else score() if args.score else wait(args.workers)
    print(json.dumps({key: value for key, value in result.items() if key not in ('cases', 'native_cases', 'checks', 'outputs_sha256', 'input_and_closed_prediction_bindings_sha256', 'statistical_source_identities_sha256', 'frozen_inputs_and_runtime_sha256')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
