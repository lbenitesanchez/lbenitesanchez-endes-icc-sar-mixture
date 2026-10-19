"""Descriptive scores after all 64 worlds and the main comparison close.

No p values, multiplicity family, recipe selection or superiority selection is
performed. Monte Carlo intervals describe a fixed 64-world sensitivity only.
"""
from pathlib import Path
import argparse, json, time
import numpy as np
import pandas as pd
from sensitivity_sampled_graph_generate import (ROOT, DEST, CONTEXT, IDS,
    METHODS, sha, save)
from sensitivity_sampled_graph_audit import (ENDPOINTS, METRICS, Z,
    RESULT_STATUS, supplemental_scoring_gate)


def score_arrays(theta, d, prediction, method, rep, convergence, physical=False):
    theta = np.asarray(theta)
    if theta.shape != (120, 2) or not np.isfinite(theta).all():
        raise ValueError('Synthetic finite-population targets incomplete')
    arrays = np.stack([np.asarray(prediction[key]) for key in ('mean', 'lower', 'upper')])
    if arrays.shape != (3, 120, 2) or not np.isfinite(arrays).all() or \
            np.any(arrays[1] > arrays[2]):
        raise ValueError('Incomplete or non-finite complete prediction; no deletion')
    observed = np.zeros(theta.shape, bool)
    observed[d['sample_idx']] = np.isfinite(d['y'])
    unsampled = ~np.isin(np.arange(120), d['sample_idx'])
    rows = []
    for j, endpoint in enumerate(ENDPOINTS):
        for group, mask in (('all', np.ones(120, bool)),
            ('observed_endpoint', observed[:, j]), ('unsampled_common', unsampled)):
            truth = theta[mask, j]
            mean, lo, hi = (prediction[key][mask, j] for key in ('mean', 'lower', 'upper'))
            if physical:
                mean, lo, hi = (np.clip(a, 0, 1) for a in (mean, lo, hi))
            error = mean - truth
            score = hi - lo + 40 * np.maximum(lo - truth, 0) + 40 * np.maximum(truth - hi, 0)
            rows.append({'scenario': CONTEXT, 'replication': rep, 'method': method,
                'outcome': endpoint, 'group': group, 'areas': int(mask.sum()),
                'mse': float(np.mean(error ** 2)), 'mae': float(np.mean(abs(error))),
                'bias': float(np.mean(error)), 'interval_score': float(np.mean(score)),
                'coverage': float(np.mean((lo <= truth) & (truth <= hi))),
                'width': float(np.mean(hi - lo)), 'converged': bool(convergence),
                'point_outside_0_1': float(np.mean((mean < 0) | (mean > 1))),
                'interval_outside_0_1': float(np.mean((lo < 0) | (hi > 1))),
                'phase': 'prospective_descriptive_sampled_graph_sensitivity',
                'shared_physical_projection': bool(physical)})
    return rows


def descriptive_summary(frame):
    rows = []
    for method in METHODS:
        for group in ('all', 'observed_endpoint', 'unsampled_common'):
            for endpoint in (*ENDPOINTS, 'balanced_anemia_stunting'):
                part = frame[(frame.method == method) & (frame.group == group)]
                if endpoint != 'balanced_anemia_stunting':
                    part = part[part.outcome == endpoint]
                expected = 128 if endpoint == 'balanced_anemia_stunting' else 64
                if len(part) != expected or part[['replication', 'outcome']].duplicated().any():
                    raise ValueError('Incomplete 64-world descriptive summary')
                worlds = part.groupby('replication')[list(METRICS)].mean().reindex(IDS)
                means = worlds.mean(); ses = worlds.std(ddof=1) / np.sqrt(64)
                rows.append({'method': method, 'outcome': endpoint, 'group': group,
                    'worlds': 64, 'rmse': float(np.sqrt(means.mse)),
                    'rmse_mcse': float(ses.mse / (2 * np.sqrt(means.mse)))
                        if means.mse > 0 else 0.,
                    **{key: float(means[key]) for key in METRICS},
                    **{key + '_mcse': float(ses[key]) for key in METRICS},
                    'nonconverged_worlds_retained': int(part.groupby('replication').converged.all().eq(False).sum()),
                    'descriptive_only': True})
    return pd.DataFrame(rows)


def descriptive_contrasts(frame):
    rows = []
    for group in ('all', 'observed_endpoint', 'unsampled_common'):
        part = frame[frame.group == group]
        candidate = part[part.method == 'Selected'].groupby('replication')[list(METRICS)].mean().reindex(IDS)
        for method in METHODS:
            if method == 'Selected':
                continue
            reference = part[part.method == method].groupby('replication')[list(METRICS)].mean().reindex(IDS)
            if candidate.isna().any().any() or reference.isna().any().any():
                raise ValueError('Incomplete paired descriptive world')
            for metric in METRICS:
                delta = candidate[metric] - reference[metric]
                estimate = float(delta.mean()); se = float(delta.std(ddof=1) / np.sqrt(64))
                rows.append({'candidate': 'Selected', 'reference': method, 'group': group,
                    'metric': metric, 'paired_worlds': 64, 'difference': estimate,
                    'mcse': se, 'MC_CI95_low': estimate - Z * se,
                    'MC_CI95_high': estimate + Z * se, 'descriptive_only': True,
                    'estimate_defined': True, 'undefined_reason': '',
                    'mcse_log_ratio': np.nan,
                    'interval_scope': 'Unadjusted Monte Carlo interval; no formal test or multiplicity claim'})
            m1, m0 = float(candidate.mse.mean()), float(reference.mse.mean())
            if m1 == 0 or m0 == 0:
                rows.append({'candidate': 'Selected', 'reference': method, 'group': group,
                    'metric': 'RMSE_ratio', 'paired_worlds': 64, 'difference': np.nan,
                    'mcse': np.nan, 'mcse_log_ratio': np.nan, 'MC_CI95_low': np.nan,
                    'MC_CI95_high': np.nan, 'descriptive_only': True, 'estimate_defined': False,
                    'undefined_reason': 'zero_candidate_or_reference_mean_MSE',
                    'interval_scope': 'Ratio and its uncertainty not reported when either mean MSE is zero'})
                continue
            logratio = .5 * np.log(m1 / m0)
            influence = .5 * (candidate.mse / m1 - reference.mse / m0)
            se = float(influence.std(ddof=1) / np.sqrt(64))
            rows.append({'candidate': 'Selected', 'reference': method, 'group': group,
                'metric': 'RMSE_ratio', 'paired_worlds': 64, 'difference': float(np.exp(logratio)),
                'mcse': float(np.exp(logratio)) * se, 'mcse_log_ratio': se,
                'MC_CI95_low': float(np.exp(logratio - Z * se)),
                'MC_CI95_high': float(np.exp(logratio + Z * se)), 'descriptive_only': True,
                'estimate_defined': True, 'undefined_reason': '',
                'interval_scope': 'Unadjusted delta-method Monte Carlo interval on log RMSE ratio; no formal test'})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args(argv)
    if args.wait:
        required = (ROOT / 'validacion/joint_prediction_lock.json',
                    ROOT / 'validacion/sikov_final_independent_audit.json')
        while not all(path.is_file() for path in required):
            time.sleep(15)
    # The frozen principal auditor writes its JSON directly. A bounded retry
    # covers only a transient partial JSON write; failed scientific gates are
    # never retried or softened automatically.
    for attempt in range(3):
        try:
            plan, bindings = supplemental_scoring_gate()
            break
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(1)
    raw, physical = [], []
    for rep in IDS:
        source = DEST / 'inputs' / f'rep_{rep:05d}'
        out = DEST / 'fits' / f'rep_{rep:05d}'
        manifest = json.loads((source / 'manifest.json').read_text())
        truthpath = source / 'evaluation_only/truth.npz'
        # This is the first target read; all 64 fits, principal JOINT and both
        # independent numerical audits are validated before even its SHA read.
        if sha(truthpath) != manifest['truth_sha256']:
            raise RuntimeError('Changed supplemental finite-population truth')
        theta = np.load(truthpath, allow_pickle=False)['theta']
        d = dict(np.load(source / 'model_inputs/inputs.npz', allow_pickle=False))
        completion = json.loads((out / 'completion.json').read_text())
        for method in METHODS:
            folder = out / ('original' if method in METHODS[:9] else 'published')
            p = dict(np.load(folder / (method + '.npz'), allow_pickle=False))
            raw.extend(score_arrays(theta, d, p, method, rep,
                                    completion['convergence'][method], False))
            physical.extend(score_arrays(theta, d, p, method, rep,
                                         completion['convergence'][method], True))
    result = DEST / 'resultados'; result.mkdir(exist_ok=True)
    for prefix, rows in (('descriptive', raw), ('physical_projection', physical)):
        frame = pd.DataFrame(rows)
        if len(frame) != 4224:
            raise ValueError('Complete eleven-method supplemental score support required')
        frame.to_csv(result / (prefix + '_replications.csv'), index=False)
        descriptive_summary(frame).to_csv(result / (prefix + '_summary.csv'), index=False)
        descriptive_contrasts(frame).to_csv(result / (prefix + '_paired_contrasts.csv'), index=False)
    filenames = tuple(prefix + suffix + '.csv' for prefix in
        ('descriptive', 'physical_projection') for suffix in
        ('_replications', '_summary', '_paired_contrasts'))
    status = {'status': 'PASS_COMPLETE_64_DESCRIPTIVE_SUPPLEMENTAL_SCORING',
        'protocol_sha256': plan['protocol_sha256'], 'addendum_sha256': plan['addendum_sha256'],
        **bindings, 'context': CONTEXT, 'worlds': 64, 'methods': list(METHODS),
        'no_worlds_or_methods_excluded': True,
        'selected_recipe_unchanged': True, 'formal_p_values_or_new_Holm_family': False,
        'original_256_and_Holm3_Holm12_unchanged': True,
        'truth_access_only_after_all_64_and_main_comparison_closure': True,
        'shared_projection_0_1_is_secondary': True,
        'source_sha256': sha(__file__), 'outputs_sha256':
            {name: sha(result / name) for name in filenames}}
    save(RESULT_STATUS, status)
    from sensitivity_sampled_graph_audit import audit_scores
    audit_scores()
    print('PASS_SUPPLEMENTAL_64_DESCRIPTIVE_SCORING_AND_INDEPENDENT_AUDIT', flush=True)


if __name__ == '__main__':
    main()
