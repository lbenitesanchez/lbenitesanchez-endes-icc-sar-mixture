"""Descriptive 2019 application on the sealed, matched district frame.

Capital-coordinate displays do not transfer contemporary district polygons to
the 2017 census frame. This does not fit models or open individual records.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from public_verify import prediction_preflight, completed_result_gates

ROOT = Path(__file__).resolve().parents[1]
SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
METHODS = ('Selected', 'Sikov_FH', 'Sikov_SAR', 'Sikov_SAR_Published', 'ICC')
ENDPOINTS = ('anemia', 'stunting')


def read(path):
    return json.loads(Path(path).read_text())


def local(path):
    """Resolve archived absolute paths without editing the sealed JSON files."""
    p = Path(path)
    if not p.is_absolute():
        return ROOT / p
    anchor = '/estudio_publicacion/sikov_2026-10-10/'
    if anchor in str(p):
        return ROOT / str(p).split(anchor, 1)[1]
    study_anchor = '/estudio_publicacion/'
    if study_anchor in str(p):
        return ROOT.parent / str(p).split(study_anchor, 1)[1]
    raise ValueError('Unrecognized archived path: ' + str(p))


def gate():
    # Reuse the release preflight: validate the exact prediction seal and both
    # complete 256-world cohorts before hashing or opening scored aggregates.
    meta = prediction_preflight()
    completed_result_gates(meta)
    joint_path = ROOT / 'validacion/joint_prediction_lock.json'
    joint = meta['joint']
    audit_path = ROOT / 'validacion/sikov_final_independent_audit.json'
    return joint, joint_path, audit_path


def main():
    joint, joint_path, audit_path = gate()
    prepared = ROOT / 'datos/real_2019_full'
    manifest = read(prepared / 'preparation_manifest.json')
    if manifest['heldout_outcome_rows_read'] or manifest['year'] != 2019:
        raise RuntimeError('Wrong application training role')
    for name in ('fit_inputs.npz', 'area_frame.csv'):
        if SHA(prepared / name) != manifest['output_sha256'][name]:
            raise RuntimeError('Changed application frame')
    areas = pd.read_csv(prepared / 'area_frame.csv', dtype={'ubigeo': str})
    with np.load(prepared / 'fit_inputs.npz', allow_pickle=False) as z:
        sample_idx = z['sample_idx'].copy()
        y = z['y'].copy()
        D = z['D'].copy()
    if (len(areas) != 1874 or areas.ubigeo.duplicated().any()
            or not areas.ubigeo.str.fullmatch(r'\d{6}').all()):
        raise ValueError('Unequal geographic support')
    lon = areas.longitude.to_numpy(float)
    lat = areas.latitude.to_numpy(float)
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()
            and (np.abs(lon) <= 180).all() and (np.abs(lat) <= 90).all()):
        raise ValueError('Invalid capital coordinates')
    predictions = []
    source_paths = [prepared / 'preparation_manifest.json',
                    prepared / 'area_frame.csv', prepared / 'fit_inputs.npz']
    for folder in (ROOT / 'evaluacion/real_temporal',
                   ROOT / 'evaluacion/published_sar/real_temporal'):
        completion = read(folder / 'completion.json')
        path = folder / 'predictions.csv'
        if (completion['protocol_sha256'] != joint['protocol_sha256']
                or completion['heldout_outcomes_read_by_fits']
                or SHA(path) != completion['predictions_sha256']):
            raise RuntimeError('Application predictions differ from seal')
        predictions.append(pd.read_csv(path, dtype={'ubigeo': str}))
        source_paths += [path, folder / 'completion.json']
    predictions = pd.concat(predictions, ignore_index=True)
    if predictions[['ubigeo', 'method']].duplicated().any():
        raise ValueError('Duplicated application prediction')
    for method, frame in predictions.groupby('method'):
        if set(frame.ubigeo) != set(areas.ubigeo):
            raise ValueError('Different district frame for ' + method)
    required = ['p_' + e for e in ENDPOINTS] + ['lo95_' + e for e in ENDPOINTS] + ['hi95_' + e for e in ENDPOINTS]
    if not np.isfinite(predictions[required].to_numpy(float)).all():
        raise ValueError('Non-finite sealed application output')
    if not set(METHODS).issubset(set(predictions.method)):
        raise ValueError('Missing comparison method')
    observed = {}
    for j, endpoint in enumerate(ENDPOINTS):
        mask = np.isfinite(y[:, j]) & np.isfinite(D[:, j, j]) & (D[:, j, j] > 0)
        observed[endpoint] = np.zeros(len(areas), dtype=bool)
        observed[endpoint][sample_idx[mask]] = True
    rows = []
    ordered = {}
    for method in METHODS:
        frame = predictions[predictions.method.eq(method)].set_index('ubigeo').loc[areas.ubigeo]
        ordered[method] = frame
        for endpoint in ENDPOINTS:
            p = frame['p_' + endpoint].to_numpy(float)
            lo = frame['lo95_' + endpoint].to_numpy(float)
            hi = frame['hi95_' + endpoint].to_numpy(float)
            if np.any(lo > hi):
                raise ValueError('Reversed application interval')
            for group, mask in [('all', np.ones(len(areas), dtype=bool)),
                                ('observed', observed[endpoint]),
                                ('unobserved', ~observed[endpoint])]:
                q = np.quantile(p[mask], [.05, .5, .95])
                rows.append({'method': method, 'outcome': endpoint, 'group': group,
                             'districts': int(mask.sum()), 'mean_prevalence_pp': 100 * float(p[mask].mean()),
                             'q05_prevalence_pp': 100 * q[0], 'median_prevalence_pp': 100 * q[1],
                             'q95_prevalence_pp': 100 * q[2], 'minimum_prevalence_pp': 100 * float(p[mask].min()),
                             'maximum_prevalence_pp': 100 * float(p[mask].max()),
                             'mean_interval_width_pp': 100 * float((hi - lo)[mask].mean()),
                             'outside_probability_range_count': int(((p[mask] < 0) | (p[mask] > 1)).sum()),
                             'summary_target': 'unweighted distribution across districts, not national child prevalence',
                             'accuracy_or_latent_coverage_claim': False})
    out = ROOT / 'resultados/aplicacion2019'
    out.mkdir(parents=True, exist_ok=True)
    table = out / 'matched_application2019_districts.csv'
    predictions.merge(areas[['ubigeo', 'latitude', 'longitude', 'poverty_stratum']],
                      on='ubigeo', validate='many_to_one').to_csv(table, index=False)
    summary = out / 'matched_application2019_summary.csv'
    pd.DataFrame(rows).to_csv(summary, index=False)
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(9, 9.4), constrained_layout=True)
    for j, endpoint in enumerate(ENDPOINTS):
        candidate = 100 * ordered['Selected']['p_' + endpoint].to_numpy(float)
        reference = 100 * ordered['Sikov_SAR_Published']['p_' + endpoint].to_numpy(float)
        name = 'Anemia' if endpoint == 'anemia' else 'Stunting'
        for row, values, limits, cmap, title, label in [
                (0, candidate, (0, 100), 'viridis', name + ': selected mixture', 'Model estimate (%)'),
                (1, candidate - reference, (-20, 20), 'coolwarm', name + ': mixture minus sampled-graph SAR', 'Difference (percentage points)')]:
            ax = axes[row, j]
            sc = ax.scatter(lon, lat, c=values, vmin=limits[0], vmax=limits[1],
                            cmap=cmap, s=6, linewidths=0, rasterized=True)
            missing = ~observed[endpoint]
            ax.scatter(lon[missing], lat[missing], facecolors='none', edgecolors='0.35',
                       s=9, linewidths=.25, rasterized=True)
            ax.set(title=title, xlabel='Longitude', ylabel='Latitude',
                   xlim=(-82, -68), ylim=(-19, 1))
            ax.set_aspect('equal', adjustable='box')
            ax.spines[['top', 'right']].set_visible(False)
            cb = fig.colorbar(sc, ax=ax, shrink=.8, pad=.025, extend='both')
            cb.set_label(label)
    fig.suptitle('Matched ENDES 2019 application: 1,874 district capitals\n'
                 'Children aged 6–59 months; historical WHO 2011 anemia definition', fontsize=12)
    fig.supxlabel('Circles mark capitals without usable 2019 endpoint estimates. Colours display raw sealed predictions.\n'
                  'Capital locations represent district estimates; differences describe model disagreement, not accuracy.', fontsize=8)
    png = out / 'matched_application2019_pointmaps.png'
    pdf = out / 'matched_application2019_pointmaps.pdf'
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    result = {'status': 'PASS_SEALED_DESCRIPTIVE_APPLICATION2019_FIGURES',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'protocol_sha256': joint['protocol_sha256'], 'addendum_sha256': joint['addendum_sha256'],
              'joint_prediction_lock_sha256': SHA(joint_path), 'independent_audit_sha256': SHA(audit_path),
              'source_sha256': SHA(__file__), 'districts': len(areas),
              'observed_districts_by_endpoint': {e: int(observed[e].sum()) for e in ENDPOINTS},
              'individual_records_opened': False, 'future_heldout_outcomes_opened': False,
              'model_or_prediction_modified': False, 'display_geometry': '2019 capital coordinates; no polygon transfer',
              'probability_or_difference_values_clipped': False,
              'colour_limits_only': {'prevalence_percent': [0, 100], 'difference_pp': [-20, 20], 'extend': 'both'},
              'summaries_are_national_population_prevalences': False,
              'comparative_accuracy_or_latent_coverage_claim': False,
              'visual_review_complete': False,
              'inputs_sha256': {str(p): SHA(p) for p in source_paths},
              'outputs_sha256': {str(p): SHA(p) for p in [table, summary, png, pdf]}}
    dest = ROOT / 'validacion/comparison_application_figures.json'
    dest.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'districts': result['districts'],
                      'visual_review_complete': False, 'outputs': list(result['outputs_sha256'])}))


if __name__ == '__main__':
    main()
