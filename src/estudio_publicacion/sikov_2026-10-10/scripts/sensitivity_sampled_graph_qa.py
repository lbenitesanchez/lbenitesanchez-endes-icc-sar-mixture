"""Fictitious-array prefit QA and prospective supplemental plan seal.

The generator-copy check runs both generators on a temporary artificial frame,
never on actual final truths, microdata, development scores or fitted outputs.
"""
import argparse, datetime as dt, json, tempfile, types
from pathlib import Path
import numpy as np
import pandas as pd
import comparison_simulation as original
import sensitivity_sampled_graph_generate as supplementary
from sensitivity_sampled_graph_generate import (ROOT, CONTEXT, IDS, METHODS,
    PLAN, SCRIPT_NAMES, frozen_sources_gate, save, sha)
from sensitivity_sampled_graph_audit import (independent_triangular,
    validate_closed_metadata, validate_joint_path_set)
from sikov_published_reference import triangular_weights
from sikov_final_audit import graph
from sensitivity_sampled_graph_score import (score_arrays, descriptive_summary,
                                              descriptive_contrasts)
from improve_development import scores as frozen_scores

QA_PATH = ROOT / 'validacion/sensitivity_sampled_graph_prefit_QA.json'


def fixture_frame():
    rng = np.random.default_rng(202610109170)
    coords = rng.normal(size=(120, 2))
    strata = np.repeat([1, 2, 3], 40)
    Xraw = np.c_[np.ones(120), rng.uniform(.1, .8, (120, 14))]
    Xraw[:, 1] *= 4
    X = np.c_[np.ones(120), (Xraw[:, 1:] - Xraw[:, 1:].mean(0)) /
                                 Xraw[:, 1:].std(0)]
    return dict(coords=coords, strata=strata, X_raw=Xraw, X=X,
        W=graph(coords, strata, 6), basis=rng.normal(0, .1, (120, 18)),
        covariate_names=np.asarray(original.NAMES))


def require_rejection(function, *args):
    try:
        function(*args)
    except (ValueError, RuntimeError):
        return True
    raise AssertionError('Malformed prefit fixture was accepted')


def algebra_qa():
    d = fixture_frame()
    srng = np.random.default_rng(np.random.SeedSequence([20261010, 972, 30000]))
    s = np.sort(np.concatenate([srng.choice(np.flatnonzero(d['strata'] == g),
                                            32, replace=False) for g in (1, 2, 3)]))
    u = np.setdiff1d(np.arange(120), s)
    W = triangular_weights(d['coords'], s, d['strata'])
    independent = independent_triangular(d, s)
    if not np.array_equal(W, independent) or np.any(W[np.ix_(s, u)]) or \
            not np.allclose(W.sum(1), 1, atol=1e-14, rtol=0):
        raise AssertionError('Independent sampled-only graph differs')
    maxima = {'covariance_relative': 0., 'directed_cross_covariance': 0.,
              'scaled_covariance_relative': 0.}
    for rho in (.6, -.75, .999):
        inverse = np.linalg.inv(np.eye(120) - rho * W)
        covariance = inverse @ inverse.T
        invD = np.linalg.inv(np.eye(96) - rho * W[np.ix_(s, s)])
        CD = invD @ invD.T
        relative = float(abs(covariance[np.ix_(s, s)] - CD).max()) / max(float(abs(CD).max()), 1.)
        maxima['covariance_relative'] = max(maxima['covariance_relative'], relative)
        F = np.linalg.solve(np.eye(24) - rho * W[np.ix_(u, u)],
                            rho * W[np.ix_(u, s)])
        cross = covariance[np.ix_(u, s)]
        cross_error = float(abs(cross - F @ CD).max()) / max(float(abs(cross).max()), 1.)
        maxima['directed_cross_covariance'] = max(maxima['directed_cross_covariance'], cross_error)
        normalizer = np.sqrt(np.mean(np.sum(inverse * inverse, axis=1)))
        scaled = (inverse / normalizer) @ (inverse / normalizer).T
        scaled_error = float(abs(scaled[np.ix_(s, s)] - CD / normalizer ** 2).max()) / max(float(abs(scaled).max()), 1.)
        maxima['scaled_covariance_relative'] = max(maxima['scaled_covariance_relative'], scaled_error)
    # Available-endpoint mask is identical to the predetermined 96 domains,
    # because production blocks (rather than regenerates) any missing direct y.
    y = np.linspace(.1, .8, 192).reshape(96, 2)
    D = np.tile(np.eye(2) * .001, (96, 1, 1))
    for j in range(2):
        mask = np.isfinite(y[:, j]) & np.isfinite(D[:, j, j]) & (D[:, j, j] > 0)
        if not np.array_equal(s[mask], s):
            raise AssertionError('Common finite-positive endpoint support changed')
    fullinverse = np.linalg.inv(np.eye(120) - .6 * d['W'])
    fullcovariance = fullinverse @ fullinverse.T
    triangularinverse = np.linalg.inv(np.eye(120) - .6 * W)
    triangularcovariance = triangularinverse @ triangularinverse.T
    scope_difference = float(abs(fullcovariance[np.ix_(s, s)] -
                                triangularcovariance[np.ix_(s, s)]).max())
    if max(maxima.values()) > 1e-10 or scope_difference < 1e-3:
        raise AssertionError('Sampled covariance/extension distinction did not validate')
    return {**maxima, 'full_graph_observed_covariance_difference': scope_difference,
            'sampled_domains': 96, 'non_sampled_domains': 24,
            'endpoint_available_mask_equals_sampled_design_mask': True}


class FixtureRandom:
    """Remap old declared seeds solely for the temporary generator-copy test."""
    def default_rng(self, seed=None):
        if isinstance(seed, np.random.SeedSequence):
            entropy = seed.entropy
            if isinstance(entropy, (list, tuple)) and len(entropy) == 4 and \
                    entropy[0] == 20261010 and entropy[1] in (701, 702):
                seed = np.random.SeedSequence([20261010,
                    971 if entropy[1] == 701 else 972, entropy[3]])
        return np.random.default_rng(seed)
    def __getattr__(self, name):
        return getattr(np.random, name)


class FixtureNumpy:
    random = FixtureRandom()
    def __getattr__(self, name):
        return getattr(np, name)


def generator_copy_qa():
    """Same fictitious triangular population, same RNGs, original versus copy.

    Original generator sees the triangular W in its temporary template; the
    new generator constructs that W from the design but supplies full W to fit.
    Therefore only the estimator W should differ in their model input arrays.
    """
    d = fixture_frame(); rep = 30000
    srng = np.random.default_rng(np.random.SeedSequence([20261010, 972, rep]))
    s = np.sort(np.concatenate([srng.choice(np.flatnonzero(d['strata'] == g),
                         32, replace=False) for g in (1, 2, 3)]))
    Wgen = triangular_weights(d['coords'], s, d['strata'])
    with tempfile.TemporaryDirectory(prefix='sikov_sampled_graph_prefit_') as tmp:
        tmp = Path(tmp); oldroot = tmp / 'old'; newroot = tmp / 'new'
        for root, W in ((oldroot, Wgen), (newroot, d['W'])):
            template = root / 'datos/simulation_template'
            template.mkdir(parents=True)
            np.savez_compressed(template / 'auxiliary.npz', **{**d, 'W': W})
        oldglobals = dict(original.generate.__globals__)
        oldglobals.update(ROOT=oldroot, np=FixtureNumpy())
        oldgenerate = types.FunctionType(original.generate.__code__, oldglobals,
                                         'temporary_original_generate')
        newglobals = dict(supplementary.generate.__globals__)
        newglobals.update(ROOT=newroot, DEST=newroot / 'supplemental',
            gate=lambda: ({'protocol_sha256': 'temporary_fixture',
                            'addendum_sha256': 'temporary_fixture'}, 'fixture', {}))
        newgenerate = types.FunctionType(supplementary.generate.__code__, newglobals,
                                         'temporary_supplemental_generate')
        a = oldgenerate('SAR_probability_srs', rep, 'evaluation')
        b = newgenerate(rep)
        old = dict(np.load(a / 'model_inputs/inputs.npz', allow_pickle=False))
        new = dict(np.load(b / 'model_inputs/inputs.npz', allow_pickle=False))
        if set(old) != set(new):
            raise AssertionError('Copied generator array schema differs')
        for name in old:
            if name == 'W':
                continue
            equal = np.array_equal(old[name], new[name], equal_nan=True) \
                if old[name].dtype.kind in 'fci' else np.array_equal(old[name], new[name])
            if not equal:
                raise AssertionError('Copied frozen population/design/stabilization changed ' + name)
        if not np.array_equal(old['W'], Wgen) or not np.array_equal(new['W'], d['W']):
            raise AssertionError('Estimator graph must remain unchanged full W')
        old_truth = np.load(a / 'evaluation_only/truth.npz', allow_pickle=False)['theta']
        new_truth = np.load(b / 'evaluation_only/truth.npz', allow_pickle=False)['theta']
        if not np.array_equal(old_truth, new_truth) or \
                (a / 'model_inputs/sample.csv').read_bytes() != \
                (b / 'model_inputs/sample.csv').read_bytes():
            raise AssertionError('Copied finite population or sampled child records differ')
        positive_masks = [np.isfinite(new['y'][:, j]) &
            np.isfinite(new['D'][:, j, j]) & (new['D'][:, j, j] > 0) for j in range(2)]
        if not all(mask.all() for mask in positive_masks):
            raise AssertionError('SRS fixture endpoint mask differs from design support')
        return {'all_shared_model_arrays_identical_except_declared_fitter_W': True,
            'sampled_child_records_byte_identical': True,
            'synthetic_fixture_finite_population_targets_identical': True,
            'actual_final_truth_or_accuracy_read': False,
            'fictitious_temporary_fixture_truth_read_for_copy_QA': True,
            'available_endpoint_counts': [int(m.sum()) for m in positive_masks],
            'copied_design': 'J6/M60/L24; q.40/.80; endpoint availability.85/.90; same Taylor/GVF and printed coefficients'}


def contract_qa():
    records = []
    for rep in IDS:
        records.append({'context': CONTEXT, 'replication': rep,
            'sensitivity_plan_sha256': 'fixture', 'truth_read_by_fits': False,
            'convergence': {name: True for name in METHODS},
            'selected_recipe': {'family': 'ICC_SAR_mixture', 'icc_weight': .75},
            'original_fit_seed_stream': [20261010, 973, rep],
            'original_fit_seed': int(np.random.SeedSequence([20261010, 973, rep]).generate_state(1)[0]) % (2 ** 32 - 200000),
            'published_bootstrap_seed_stream': [20261010, 974, rep],
            'published_bootstrap_seed': int(np.random.SeedSequence([20261010, 974, rep]).generate_state(1)[0])})
    validate_closed_metadata(records, 'fixture')
    rejected = {}
    rejected['63_worlds_blocked'] = require_rejection(validate_closed_metadata,
                                                      records[:-1], 'fixture')
    rejected['duplicate_world_blocked'] = require_rejection(validate_closed_metadata,
                                             records[:-1] + [records[0]], 'fixture')
    for name, key, value in (('truth_flag', 'truth_read_by_fits', True),
            ('changed_recipe', 'selected_recipe', {'family': 'ICC_SAR_mixture', 'icc_weight': .5}),
            ('changed_seed', 'original_fit_seed', 12),
            ('missing_method', 'convergence', {k: True for k in METHODS[:-1]})):
        bad = json.loads(json.dumps(records)); bad[0][key] = value
        rejected[name + '_blocked'] = require_rejection(validate_closed_metadata, bad, 'fixture')
    expected = [ROOT / 'protocol_freeze.json', ROOT /
        'validacion/sikov_published_addendum_freeze.json', ROOT /
        'validacion/fresh_prediction_lock.json', ROOT /
        'validacion/sikov_published_fresh_prediction_lock.json']
    for scope in ('temporal', 'territorial'):
        for folder in (ROOT / 'evaluacion' / ('real_' + scope),
                       ROOT / 'evaluacion/published_sar' / ('real_' + scope)):
            expected.extend((folder / 'completion.json', folder / 'predictions.csv'))
        expected.extend(ROOT / 'validacion' / (prefix + scope + '.json') for prefix
            in ('real_prediction_lock_', 'sikov_published_prediction_lock_'))
    joint = {'files': [{'path': str(path), 'sha256': 'fixture'} for path in expected]}
    validate_joint_path_set(joint)
    bad = json.loads(json.dumps(joint)); bad['files'].append({'path': str(ROOT /
        'sensibilidad_sampled_graph/inputs/rep_30000/evaluation_only/truth.npz'), 'sha256': 'fixture'})
    rejected['injected_truth_path_blocked_before_hash'] = require_rejection(validate_joint_path_set, bad)
    bad = json.loads(json.dumps(joint)); bad['files'][-1] = bad['files'][0]
    rejected['duplicated_joint_path_blocked_before_hash'] = require_rejection(validate_joint_path_set, bad)
    streams = [tuple(np.random.SeedSequence([20261010, channel, rep]).generate_state(4))
               for channel in (971, 972, 973, 974) for rep in IDS]
    primary = [tuple(np.random.SeedSequence([20261010, channel, context, rep]).generate_state(4))
               for channel in (701, 702, 811, 941) for context in range(4) for rep in IDS]
    if len(set(streams)) != 256 or set(streams).intersection(primary):
        raise AssertionError('Supplemental streams collide with each other or primary streams')
    return {'accepted_fixture_64_unique_worlds': True, **rejected,
            'new_stream_states_distinct_from_each_other_and_primary': True,
            'actual_final_truth_or_accuracy_read': False}


def scoring_fixture_qa():
    rng = np.random.default_rng(202610109180)
    rows = {False: [], True: []}; errors = []
    direct = {'sample_idx': np.arange(96), 'y': np.full((96, 2), .4)}
    for rep in IDS:
        theta = rng.uniform(.02, .85, (120, 2))
        perturbation = rng.normal(0, .035, (120, 2))
        for mi, method in enumerate(METHODS):
            prediction = {'mean': theta + perturbation + .004 * mi,
                'lower': theta + perturbation - .10 + .003 * mi,
                'upper': theta + perturbation + .10 + .005 * mi}
            for physical in (False, True):
                own = score_arrays(theta, direct, prediction, method, rep, True, physical)
                fixture = {k: np.clip(v, 0, 1) if physical else v
                           for k, v in prediction.items()}
                reference = frozen_scores({'theta': theta}, direct, fixture,
                    method, CONTEXT, rep, [True, True], 0.)
                for a, b in zip(own, reference):
                    if (a['outcome'], a['group'], a['areas']) != \
                            (b['outcome'], b['group'], b['areas']):
                        raise AssertionError('Supplemental evaluation groups differ from frozen criteria')
                    errors.extend(abs(a[k] - b[k]) for k in ('mse', 'mae', 'bias',
                        'interval_score', 'coverage', 'width', 'point_outside_0_1', 'interval_outside_0_1'))
                rows[physical].extend(own)
    contrast_error = 0.
    for physical in (False, True):
        frame = pd.DataFrame(rows[physical])
        summary = descriptive_summary(frame); contrasts = descriptive_contrasts(frame)
        if len(frame) != 4224 or len(summary) != 99 or len(contrasts) != 210 or \
                any('p_' in name or 'Holm' in name for table in (frame, summary, contrasts)
                    for name in table.columns):
            raise AssertionError('Supplemental descriptive universe or no-p contract differs')
        part = frame[frame.group == 'all']
        a = part[part.method == 'Selected'].groupby('replication').interval_score.mean().reindex(IDS)
        b = part[part.method == 'Sikov_SAR_Published'].groupby('replication').interval_score.mean().reindex(IDS)
        difference = a.to_numpy() - b.to_numpy()
        expected_estimate = float(np.sum(difference) / 64)
        expected_se = float(np.sqrt(np.sum((difference - expected_estimate) ** 2) / 63 / 64))
        actual = contrasts[(contrasts.reference == 'Sikov_SAR_Published') &
            (contrasts.group == 'all') & (contrasts.metric == 'interval_score')].iloc[0]
        contrast_error = max(contrast_error, abs(actual.difference - expected_estimate),
                              abs(actual.mcse - expected_se))
    bad = {k: v.copy() for k, v in prediction.items()}; bad['mean'][0, 0] = np.nan
    require_rejection(score_arrays, theta, direct, bad, 'Selected', 30000, True)
    require_rejection(descriptive_summary, pd.DataFrame(rows[False]).iloc[:-1])
    zero = pd.DataFrame(rows[False]).copy(); zero.loc[zero.method == 'Selected', 'mse'] = 0.
    zero_contrasts = descriptive_contrasts(zero)
    zero_ratios = zero_contrasts[zero_contrasts.metric == 'RMSE_ratio']
    if len(zero_ratios) != 30 or zero_ratios.estimate_defined.any() or \
            not zero_ratios[['difference', 'mcse', 'MC_CI95_low', 'MC_CI95_high', 'mcse_log_ratio']].isna().all().all() or \
            not zero_ratios.undefined_reason.eq('zero_candidate_or_reference_mean_MSE').all():
        raise AssertionError('Zero-MSE ratios must remain explicit, undefined, without invented intervals')
    zero['mse'] = 0.
    if descriptive_contrasts(zero).query('metric == "RMSE_ratio"').estimate_defined.any():
        raise AssertionError('Zero divided by zero must not invent an identity RMSE ratio')
    if max(errors) > 1e-12 or contrast_error > 1e-12:
        raise AssertionError('Supplemental criteria/paired-world MCSE differ')
    return {'frozen_raw_and_shared_projection_score_max_error': max(errors),
        'independent_endpoint_paired_MCSE_max_error': contrast_error,
        'rows_per_scale': 4224, 'summary_rows_per_scale': 99,
        'paired_descriptive_rows_per_scale': 210,
        'NaN_predictions_and_missing_summary_worlds_blocked': True,
        'zero_MSE_and_zero_over_zero_ratios_explicitly_undefined': True,
        'no_formal_p_values_or_Holm_family': True,
        'actual_final_truth_or_accuracy_read': False}


def freeze(qa):
    protocol, base, add = frozen_sources_gate()
    if PLAN.exists():
        raise FileExistsError('Do not replace an existing prospective freeze')
    # No supplemental fitting, predictions or scoring can predate this seal.
    destination = ROOT / 'sensibilidad_sampled_graph'
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError('Supplemental population/fits must start after the plan seal')
    if not qa['passed'] or qa['actual_final_truth_or_accuracy_read']:
        raise RuntimeError('Successful fictitious prefit QA required')
    sources = {str((ROOT / 'scripts' / name).relative_to(ROOT)):
        sha(ROOT / 'scripts' / name) for name in SCRIPT_NAMES}
    inputs = {str(path.relative_to(ROOT)): sha(path) for path in
        (ROOT / 'datos/simulation_template/auxiliary.npz',
         ROOT / 'datos/simulation_template/manifest.json',
         ROOT / 'datos/simulation_template/area_ids.csv')}
    result = {'status': 'FROZEN_PROSPECTIVE_DESCRIPTIVE_SENSITIVITY',
        'frozen_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'protocol_sha256': base, 'addendum_sha256': add,
        'context': CONTEXT, 'replications': list(IDS), 'worlds': 64,
        'methods': list(METHODS), 'sources_sha256': sources, 'inputs_sha256': inputs,
        'prefit_QA_path': str(QA_PATH.relative_to(ROOT)), 'prefit_QA_sha256': sha(QA_PATH),
        'accuracy_or_truth_read_before_freeze': False,
        'original_37_sources_256_worlds_and_Holm3_Holm12_unchanged': True,
        'main_primary_family_membership': False,
        'formal_p_values_or_new_Holm_family': False,
        'no_recipe_or_model_selection_on_supplemental_results': True,
        'selected_recipe': protocol['selected_recipe'],
        'streams': {'population': [20261010, 971, 'replication'],
            'sampling': [20261010, 972, 'replication'],
            'original_fits': [20261010, 973, 'replication'],
            'published_bootstrap': [20261010, 974, 'replication']},
        'frame': {'areas': 120, 'raw_covariates_plus_intercept': 15,
            'strata': 3, 'stratum_areas': 40, 'sampled_stratum_areas': 32,
            'sampled_areas': 96, 'non_sampled_areas': 24, 'ICC_basis_rank': 18},
        'population_and_design': {'rho': .6, 'graph_neighbors': 6,
            'graph': 'W11 sampled-only within-stratum k6; W10 zero; W01/W00 from full-frame within-stratum k6',
            'graph_defined_before_population_outcomes': True,
            'gaussian_innovation_endpoint_correlation': .4,
            'same_global_mean_squared_row_norm_as_original_generator': True,
            'printed_Sikov_coefficients_unchanged': True,
            'base_risk_clip': [.03, .75], 'probability_clip': [.005, .90],
            'spatial_scales': [.07, .05], 'population_PSUs_per_area': 60,
            'children_per_PSU': 24, 'sampled_PSUs_per_area': 6,
            'child_sampling_probability': [.40, .80],
            'endpoint_child_availability': [.85, .90],
            'child_availability_is_not_domain_deletion': True,
            'each_endpoint_requires_96_finite_y_and_positive_finite_D_before_fit': True,
            'missing_support_is_blocking_no_resampling_or_exclusion': True,
            'same_frozen_Taylor_GVF_Kish_inputs_and_weights': True,
            'same_full_W_supplied_to_all_estimator_inputs': True},
        'numerical_fits': {'kernels': 'Unchanged comparison_evaluate.fits and sikov_published_fit.fit_arrays',
            'bootstrap_primary_B': 100, 'native_naive_and_BC_same_innovations': True,
            'NUTS': 'Unchanged 4 serial chains, 1000 warmup+1000 retained per chain; predetermined 2000/2000 retry; all frozen Rhat/ESS/BFMI/divergence/depth/shape gates',
            'finite_failed_numeric_worlds_retained_without_selection': True,
            'initial_max_operational_workers': 2,
            'max_operational_workers_after_original_256_fit_closures': 4,
            'worker_expansion_uses_closure_metadata_only_never_accuracy': True},
        'evaluation': {'groups': ['all', 'observed_endpoint', 'unsampled_common'],
            'criteria': ['RMSE', 'MSE', 'MAE', 'bias', 'coverage95', 'width95', 'interval_score95'],
            'balanced_endpoints_equal_weight_with_paired_world_MCSE': True,
            'shared_0_1_projection_secondary_all_11_methods': True,
            'Monte_Carlo_intervals_descriptive_unadjusted': True,
            'RMSE_ratio_MCSE_scale': 'mcse is delta-method SE of RMSE ratio; mcse_log_ratio is SE of log ratio used for the exponentiated MC interval',
            'RMSE_ratio_zero_MSE_policy': 'If either mean MSE is zero, retain the contrast row with estimate_defined false, explicit undefined_reason, and no ratio/SE/CI; never exclude the world or invent 0/0',
            'first_truth_read_requires': ['all_64_supplemental_predictions_sealed',
                'independent_supplemental_prediction_audit_PASS',
                'principal_JOINT_prediction_lock_valid',
                'principal_independent_audit_PASS_COMPLETE_COMPARISON_INDEPENDENT_AUDIT']},
        'scope_limits': [
            'The original 2023 neighbor rule, poverty vector and non-sampled implementation remain unidentified; this is not exact original numeric replication.',
            'The declared directed extension comes from the authors 2024 preprint; the 2023 observed SAR covariance is reconstructed on a specified k6 graph.',
            'The latent Gaussian observed covariance exactly matches C_D up to the shared global variance normalization when both endpoint masks retain the 96 sampled domains.',
            'The generative spatial graph conditions the population mechanism on the predetermined sampled-domain set; it is a comparator-favourable conditional sensitivity rather than a sampling-independent population model.',
            'Probability clipping, binomial finite populations and GVF-stabilized design variances make Gaussian FH an approximate working model even in this favourable spatial mechanism.',
            'This sensitivity may improve or weaken the fixed candidate; it is not a fifth primary context, a formal additional superiority test, or a new selection dataset.',
            'ICC and design/GVF inputs remain working quantities fixed during NUTS; complete survey uncertainty propagation is not claimed.']}
    save(PLAN, result)
    supplementary.gate()
    print(json.dumps({'status': result['status'], 'sensitivity_plan_sha256': sha(PLAN),
        'sources': list(sources), 'prefit_QA_sha256': sha(QA_PATH)}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if PLAN.exists():
        raise RuntimeError('Prefit QA output is sealed; do not overwrite after freeze')
    result = {'status': 'PASS_FICTITIOUS_PREFIT_SAMPLED_GRAPH_SENSITIVITY_QA',
        'passed': True, 'algebra': algebra_qa(), 'generator_copy': generator_copy_qa(),
        'contracts': contract_qa(), 'scoring': scoring_fixture_qa(),
        'actual_final_truth_or_accuracy_read': False,
        'actual_microdata_or_reserved_responses_read': False,
        'QA_source_sha256': sha(__file__)}
    save(QA_PATH, result)
    print(json.dumps(result, indent=2), flush=True)
    if args.freeze:
        freeze(result)


if __name__ == '__main__':
    main()
