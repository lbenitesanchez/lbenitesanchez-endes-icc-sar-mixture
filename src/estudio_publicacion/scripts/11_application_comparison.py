"""Common held-out district comparison; never treats noisy survey means as truth.

The score averages design-weighted binary cross-entropy within each district,
then gives each observed district equal weight. Two marginal scores are reported
separately: they do not define a child's joint anaemia/stunting distribution.
Previously reused territorial folds make this an exploratory external check.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'resultados'
COMP = RESULT / 'resultados_comparadores'
OUTCOMES = ('anemia', 'stunting')
GATES = ('FH', 'MFH', 'SFH', 'MSFH', 'GaoStrict', 'GaoCommon', 'Hogg', 'BetaPowerHMCGraph')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prediction(path, area_name):
    if not path.exists():
        raise FileNotFoundError(path)
    p = pd.read_csv(path, dtype={area_name: str}).rename(columns={area_name: 'ubigeo'})
    if p.ubigeo.duplicated().any():
        raise ValueError(f'Duplicate districts: {path}')
    return p


def binary_score(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def bootstrap_difference(candidate, reference, blocks, repetitions, rng):
    """Paired province bootstrap of district-average score and RMSE differences."""
    labels = np.unique(blocks)
    indices = [np.flatnonzero(blocks == label) for label in labels]
    values = np.empty((repetitions, 2))
    for k in range(repetitions):
        index = np.concatenate([indices[j] for j in rng.integers(len(labels), size=len(labels))])
        values[k, 0] = candidate[index, 0].mean() - reference[index, 0].mean()
        values[k, 1] = np.sqrt(candidate[index, 1].mean()) - np.sqrt(reference[index, 1].mean())
    point = np.array([candidate[:, 0].mean() - reference[:, 0].mean(),
                      np.sqrt(candidate[:, 1].mean()) - np.sqrt(reference[:, 1].mean())])
    limits = np.quantile(values, [.025, .975], axis=0)
    return point, limits


def compare_year(year, candidate_tag, include_age_sex, repetitions, extra_specs=None):
    base = RESULT / f'aplicacion_{year}_who2024'
    path = base / 'locked_test_area_scores.csv'
    old = pd.read_csv(path, dtype={'ubigeo': str, 'province': str})
    old = old[old.validation == 'province_locked_test'].copy()
    truth = old[old.method == 'FH'][['ubigeo', 'province', 'direct_anemia', 'direct_stunting']].copy()
    if truth.ubigeo.duplicated().any():
        raise ValueError('Duplicate reference districts')
    expected = set(truth.ubigeo)
    rows = []
    sources = [{'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}]
    for method in ('FH', 'MFH', 'SFH', 'MSFH'):
        part = old[old.method == method]
        if set(part.ubigeo) != expected:
            raise ValueError(f'Unequal holdout support for {method}')
        for outcome in OUTCOMES:
            if not np.allclose(part.set_index('ubigeo')[f'direct_{outcome}'].reindex(truth.ubigeo),
                               truth[f'direct_{outcome}'], atol=1e-14):
                raise ValueError('Inconsistent held-out measurements')
        rows.append(part[['ubigeo', 'pred_anemia', 'pred_stunting']].assign(method=method, information_arm='common_area_X'))
    specs = [
        ('GaoStrict', RESULT / f'competidores_gao/endes_{year}_author_strict_locked_train/predictions.csv', 'area', 'p_mean_', 'common_area_X'),
        ('GaoCommon', RESULT / f'competidores_gao/endes_{year}_common_continuity_locked_train/predictions.csv', 'area', 'p_mean_', 'common_area_X'),
        ('Hogg', RESULT / f'competidores_hogg/endes_{year}_locked_train/predictions.csv', 'ubigeo', 'p_mean_', 'common_area_X'),
        ('BetaPowerHMCGraph', COMP / candidate_tag / f'aplicacion_{year}_who2024/province_test_predictions.csv', 'ubigeo', 'pred_', 'common_area_X'),
    ]
    if include_age_sex and year == 2025:
        specs.append(('HoggAgeSex', RESULT / 'competidores_hogg/endes_2025_locked_train_age_sex/predictions.csv', 'ubigeo', 'p_mean_', 'survey_age_sex_extra'))
    for extra in extra_specs or []:
        if int(extra['year']) == year:
            specs.append((extra['method'], ROOT / extra['path'], extra.get('area_key', 'ubigeo'),
                          extra.get('mean_prefix', 'p_mean_'), extra['information_arm']))
    if len({spec[0] for spec in specs}) != len(specs):
        raise ValueError('Duplicate comparator label')
    for method, path, key, prefix, arm in specs:
        pred = load_prediction(path, key)
        part = truth[['ubigeo']].merge(pred, on='ubigeo', how='left', validate='one_to_one')
        for outcome in OUTCOMES:
            part[f'pred_{outcome}'] = part[f'{prefix}{outcome}']
            value = part[f'pred_{outcome}'].to_numpy()
            if not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
                raise ValueError(f'Invalid held-out mean: {method}/{outcome}')
        rows.append(part[['ubigeo', 'pred_anemia', 'pred_stunting']].assign(method=method, information_arm=arm))
        sources.append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path)})
    detail = pd.concat(rows, ignore_index=True).merge(truth, on='ubigeo', validate='many_to_one')
    detail['year'] = year
    summary = []
    for outcome in OUTCOMES:
        detail[f'binary_NLL_{outcome}'] = binary_score(detail[f'pred_{outcome}'], detail[f'direct_{outcome}'])
        detail[f'squared_error_{outcome}'] = (detail[f'pred_{outcome}'] - detail[f'direct_{outcome}']) ** 2
        for (method, arm), part in detail.groupby(['method', 'information_arm']):
            summary.append(dict(year=year, method=method, information_arm=arm, outcome=outcome,
                                districts=len(part), provinces=part.province.nunique(),
                                binary_NLL=float(part[f'binary_NLL_{outcome}'].mean()),
                                rmse=float(np.sqrt(part[f'squared_error_{outcome}'].mean())),
                                log_score_clipped_districts=int(((part[f'pred_{outcome}'] < 1e-12) |
                                                               (part[f'pred_{outcome}'] > 1-1e-12)).sum()),
                                scope='noisy_measurement_concordance_exploratory'))
    contrasts = []
    rng = np.random.default_rng(20261005 + year)
    for outcome in OUTCOMES:
        candidate = detail[detail.method == 'BetaPowerHMCGraph'].sort_values('ubigeo')
        matrix = candidate[[f'binary_NLL_{outcome}', f'squared_error_{outcome}']].to_numpy()
        for reference in sorted(set(detail.method) - {'BetaPowerHMCGraph'}):
            ref = detail[detail.method == reference].set_index('ubigeo').loc[candidate.ubigeo]
            point, limits = bootstrap_difference(matrix, ref[[f'binary_NLL_{outcome}', f'squared_error_{outcome}']].to_numpy(),
                                                  candidate.province.to_numpy(), repetitions, rng)
            contrasts.append(dict(year=year, candidate='BetaPowerHMCGraph', reference=reference, outcome=outcome,
                                  binary_NLL_difference=point[0], binary_NLL_ci_low=limits[0, 0], binary_NLL_ci_high=limits[1, 0],
                                  rmse_difference=point[1], rmse_ci_low=limits[0, 1], rmse_ci_high=limits[1, 1],
                                  bootstrap_repetitions=repetitions, districts=len(candidate), provinces=candidate.province.nunique(),
                                  multiplicity='exploratory_unadjusted', weighting='equal_district; paired_province_resampling'))
    return detail, pd.DataFrame(summary), pd.DataFrame(contrasts), sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-tag', default='application_final_candidate_v2')
    parser.add_argument('--candidate-tag-2025')
    parser.add_argument('--include-age-sex', action='store_true')
    parser.add_argument('--extra-predictions-json', type=Path)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--output-tag', default='application_comparison_final_v1')
    args = parser.parse_args()
    out = COMP / args.output_tag
    if out.exists():
        raise FileExistsError(out)
    # Compute before making the output directory; missing fits cannot look complete.
    extra_specs = json.loads(args.extra_predictions_json.read_text()) if args.extra_predictions_json else []
    result = [compare_year(year, args.candidate_tag_2025 if year == 2025 and args.candidate_tag_2025 else args.candidate_tag,
                           args.include_age_sex, args.bootstrap, extra_specs) for year in (2019, 2025)]
    out.mkdir()
    for i, name in enumerate(('district_scores', 'summary', 'paired_province_contrasts')):
        pd.concat([r[i] for r in result], ignore_index=True).to_csv(out / f'{name}.csv', index=False)
    manifest = dict(status='complete', years=[2019, 2025], bootstrap=args.bootstrap,
                    log_score_numerical_probability_floor=1e-12,
                    common_methods=list(GATES), age_sex_extra=args.include_age_sex, additional_comparators=extra_specs,
                    information='Historical area auxiliaries; age/sex arm explicitly separate',
                    score='Design-weighted marginal child binary cross-entropy, averaged equally over held-out districts',
                    limitations=['Reused province folds: exploratory', 'Noisy direct means are not latent truth',
                                 'No joint child disease probability', 'No interval coverage claim from ENDES'],
                    source_sha256=digest(Path(__file__)), input_files=[x for r in result for x in r[3]])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'status': 'complete', 'output': str(out.relative_to(ROOT))}))


if __name__ == '__main__':
    main()
