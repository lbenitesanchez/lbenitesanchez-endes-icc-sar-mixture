"""Reserved endpoint scoring, after exact predictions have been locked.

Child records remain private. ENDES direct measurements are noisy and do not
reveal latent prevalence or latent interval coverage. Equal-district scores
and province-cluster inference are fixed before reserved outcomes are opened.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
from pathlib import Path
import argparse
import datetime as dt
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('BetaPowerExisting', 'BetaDesignUncertainty', 'HoggAgeSexSpatial')
ENDPOINTS = ('anemia', 'stunting')
BOOTSTRAPS = 3999
FLOOR = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_inference(candidate, reference, provinces, rng):
    """Cluster sandwich ratio-mean SE and paired province percentile bootstrap."""
    delta = np.asarray(candidate) - np.asarray(reference)
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite district contrast')
    labels, inverse = np.unique(provinces, return_inverse=True)
    J = len(labels)
    if J < 2:
        raise ValueError('Province-cluster inference needs at least two clusters')
    n = len(delta)
    counts = np.bincount(inverse).astype(float)
    sums = np.bincount(inverse, weights=delta)
    difference = float(delta.mean())
    centered = sums - counts * difference
    se = float(np.sqrt(J / (J - 1) * np.dot(centered, centered)) / n)
    statistic = difference / se if se else (0. if difference == 0 else np.copysign(np.inf, difference))
    pvalue = float(2 * t.sf(abs(statistic), J - 1))
    critical = t.ppf(.975, J - 1)
    indices = rng.integers(0, J, size=(BOOTSTRAPS, J))
    boot = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    lo, hi = np.quantile(boot, [.025, .975])
    return {'difference': difference, 'cluster_standard_error': se, 'province_clusters': J,
            't_df': J - 1, 'p_two_sided': pvalue,
            'unadjusted_t_ci_low': difference - critical * se,
            'unadjusted_t_ci_high': difference + critical * se,
            'descriptive_bootstrap_ci_low': float(lo), 'descriptive_bootstrap_ci_high': float(hi),
            'bootstrap_repetitions': BOOTSTRAPS}


def holm(values):
    p = np.asarray(values, float)
    if len(p) != 8 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('The complete prespecified Holm family must contain exactly eight finite p-values')
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(8, 0, -1)))
    result = np.empty(8)
    result[order] = adjusted
    return result


def predictions(scope):
    lock_path = ROOT / f'validacion/externa_prediction_lock_{scope}.json'
    lock = json.loads(lock_path.read_text())
    if not lock.get('approved_for_scoring', False) or lock['protocol_sha256'] != sha(ROOT / 'protocol_freeze.json'):
        raise RuntimeError('Prediction lock absent, unapproved or binds another protocol')
    if sorted(v['method'] for v in lock['predictions']) != sorted(METHODS):
        raise ValueError('Fixed comparator family changed')
    output = []
    common = None
    for spec in lock['predictions']:
        path = ROOT / spec['path']
        if sha(path) != spec['sha256']:
            raise RuntimeError('Predictions changed after response lock: ' + spec['method'])
        area = spec.get('area_key', 'ubigeo')
        prefix = spec.get('mean_prefix', 'p_mean_')
        data = pd.read_csv(path, dtype={area: str}).rename(columns={area: 'ubigeo'})
        data['ubigeo'] = data.ubigeo.str.zfill(6)
        if data.ubigeo.duplicated().any():
            raise ValueError('Duplicate prediction districts')
        for endpoint in ENDPOINTS:
            data['pred_' + endpoint] = data[prefix + endpoint]
            v = data['pred_' + endpoint].to_numpy(float)
            if not np.isfinite(v).all() or ((v < 0) | (v > 1)).any():
                raise ValueError('Nonfinite or invalid reserved prediction')
        support = set(data.ubigeo)
        if common is None:
            common = support
        elif support != common:
            raise ValueError('Unequal supplied prediction frames; cannot select a favorable intersection after scoring')
        output.append(data[['ubigeo', 'pred_anemia', 'pred_stunting']].assign(method=spec['method']))
    return pd.concat(output, ignore_index=True), lock, lock_path


def direct_measurements(path):
    data = pd.read_csv(path, dtype={'ubigeo': str, 'psu_key': str})
    if not data.year.eq(2024).all() or not data.de_facto.eq(1).all() or not data.age_months.between(6, 59).all():
        raise ValueError('Reserved scoring population differs from protocol')
    data['ubigeo'] = data.ubigeo.str.zfill(6)
    rows = []
    for endpoint, response in [('anemia', 'y_anemia_who2024'), ('stunting', 'y_stunting')]:
        valid = data[response].isin([0, 1]) & data.weight.gt(0) & np.isfinite(data.weight)
        u = data.loc[valid]
        total = u.weight.groupby(u.ubigeo).sum()
        direct = (u.weight * u[response]).groupby(u.ubigeo).sum() / total
        count = u.groupby('ubigeo').size()
        for code in sorted(data.ubigeo.unique()):
            rows.append({'ubigeo': code, 'province': code[:4], 'outcome': endpoint,
                         'direct': float(direct.get(code, np.nan)),
                         'weighted_observed_denominator': float(total.get(code, 0.)),
                         'valid_positive_weight_records': int(count.get(code, 0))})
    return pd.DataFrame(rows)


def score_scope(scope):
    pred, lock, lock_path = predictions(scope)
    response_path = ROOT / ('datos/private_external/children_2024_scoring_only.csv' if scope == 'temporal'
                            else 'datos/private_external/children_2024_test.csv')
    truth = direct_measurements(response_path)
    if scope == 'territorial':
        held = truth.province.map(lambda code: int(hashlib.sha256(('ENDES-phase2-20261007|' + code).encode()).hexdigest()[:8], 16) % 5 == 0)
        if not held.all():
            raise RuntimeError('Nonreserved province entered territorial test scoring')
    excluded = []
    detail = []
    summary = []
    contrasts = []
    rng = np.random.default_rng(202610072024 + (10000 if scope == 'territorial' else 0))
    for endpoint in ENDPOINTS:
        target = truth.loc[truth.outcome.eq(endpoint)].copy()
        in_frame = target.ubigeo.isin(pred.ubigeo)
        valid = target.direct.notna() & target.weighted_observed_denominator.gt(0)
        for row in target.loc[~in_frame | ~valid].to_dict('records'):
            excluded.append({**row, 'scope': scope, 'reason': 'unverified_auxiliary_geography' if row['ubigeo'] not in set(pred.ubigeo) else 'no_valid_positive_weight_outcome'})
        target = target.loc[in_frame & valid]
        merged = pred.merge(target, on='ubigeo', validate='many_to_one')
        p = merged['pred_' + endpoint].to_numpy(float)
        y = merged.direct.to_numpy(float)
        q = np.clip(p, FLOOR, 1 - FLOOR)
        merged['binary_NLL'] = -(y * np.log(q) + (1 - y) * np.log1p(-q))
        merged['squared_error_against_noisy_direct'] = (p - y) ** 2
        merged['Brier_binary_expectation'] = y * (1 - p) ** 2 + (1 - y) * p ** 2
        merged['numerical_clipping_applied'] = (p < FLOOR) | (p > 1 - FLOOR)
        merged['scope'] = scope
        detail.append(merged)
        for method in METHODS:
            part = merged.loc[merged.method.eq(method)]
            summary.append({'scope': scope, 'outcome': endpoint, 'method': method,
                            'districts': len(part), 'provinces': part.province.nunique(),
                            'binary_NLL': float(part.binary_NLL.mean()),
                            'RMSE_noisy_direct_pp': float(100 * np.sqrt(part.squared_error_against_noisy_direct.mean())),
                            'Brier_binary_expectation': float(part.Brier_binary_expectation.mean()),
                            'clipped_cells': int(part.numerical_clipping_applied.sum()),
                            'latent_coverage_observed': False})
        candidate = merged.loc[merged.method.eq('BetaDesignUncertainty')].sort_values('ubigeo')
        for reference in ('BetaPowerExisting', 'HoggAgeSexSpatial'):
            ref = merged.loc[merged.method.eq(reference)].set_index('ubigeo').loc[candidate.ubigeo]
            value = paired_inference(candidate.binary_NLL.to_numpy(), ref.binary_NLL.to_numpy(), candidate.province.to_numpy(), rng)
            contrasts.append({'scope': scope, 'outcome': endpoint, 'candidate': 'BetaDesignUncertainty',
                              'reference': reference, 'districts': len(candidate), **value})
    return pd.concat(detail, ignore_index=True), pd.DataFrame(summary), pd.DataFrame(contrasts), pd.DataFrame(excluded), {
        'scope': scope, 'response_sha256': sha(response_path), 'prediction_lock_sha256': sha(lock_path),
        'prediction_inputs': lock['predictions']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'resultados/externa_confirmatory')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    # Both completed locks are required before any score is exposed to analyst.
    for scope in ('temporal', 'territorial'):
        predictions(scope)
    results = [score_scope(scope) for scope in ('temporal', 'territorial')]
    contrast = pd.concat([r[2] for r in results], ignore_index=True)
    if len(contrast) != 8:
        raise RuntimeError('Fixed complete primary family is not available')
    contrast['p_Holm8'] = holm(contrast.p_two_sided.to_numpy())
    contrast['candidate_significantly_better'] = (contrast.difference < 0) & (contrast.p_Holm8 < .05)
    contrast['candidate_significantly_worse'] = (contrast.difference > 0) & (contrast.p_Holm8 < .05)
    args.output.mkdir(parents=True)
    pd.concat([r[0] for r in results], ignore_index=True).to_csv(args.output / 'district_scores.csv', index=False)
    pd.concat([r[1] for r in results], ignore_index=True).to_csv(args.output / 'summary.csv', index=False)
    contrast.to_csv(args.output / 'contrasts_Holm8.csv', index=False)
    pd.concat([r[3] for r in results], ignore_index=True).to_csv(args.output / 'excluded_district_endpoints.csv', index=False)
    proof = {'status': 'COMPLETE_RESERVED_EXTERNAL_VALIDATION', 'completed_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
             'source_sha256': sha(Path(__file__)), 'protocol_sha256': sha(ROOT / 'protocol_freeze.json'),
             'family': 'eight endpoint-specific binary-NLL contrasts; Holm8 fixed', 'inputs': [r[4] for r in results],
             'bootstrap_repetitions': BOOTSTRAPS, 'bootstrap_scope': 'descriptive unadjusted95percentile pairedprovince',
             'limitations': ['Noisy direct is not latent prevalence truth', 'No ENDES latent interval coverage claim',
                             'Temporal forecast assumes indicator persistence and uses2019 outcomes only',
                             'Public national2024 snippets seen during official source discovery are disclosed',
                             'Province-cluster t inference is an approximation, not exact finite-survey randomization'],
             'output_sha256': {p.name: sha(p) for p in args.output.glob('*.csv')}}
    (args.output / 'completion.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps({'status': proof['status'], 'contrasts': len(contrast), 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
