"""Descriptive application changes; these are not errors against latent truth."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'resultados/resultados_comparadores'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    data = pd.read_csv(path, dtype={'ubigeo': str})
    if data.ubigeo.duplicated().any() or len(data) != 1874:
        raise ValueError('Invalid application district universe')
    return data.set_index('ubigeo')


def main():
    folder = OUT / 'application_sensitivity_v2'
    complete = json.loads((folder / 'completion.json').read_text())
    if not complete['all_requested_jobs_recorded'] or not complete['all_endpoints_converged']:
        raise RuntimeError('Sensitivity fits have not all passed diagnostics')
    statuses = json.loads((folder / 'queue_status.json').read_text())
    expected = {'historical_2019','historical_2025','rank20','rank80',
                'spatial_sd025','spatial_sd1','no_cluster'}
    if complete.get('jobs') != 7 or set(statuses) != expected:
        raise RuntimeError('Every declared application sensitivity must be present exactly once')
    if any(s.get('returncode') != 0 or not s.get('all_endpoints_converged') for s in statuses.values()):
        raise RuntimeError('A sensitivity execution or convergence gate failed')
    paths = []
    rows = []
    for name, status in statuses.items():
        current = ROOT / status['output']
        fit_status = json.loads((current/'fit_status.json').read_text())
        endpoints = fit_status.get('endpoints',[])
        if len(endpoints) != 2 or any(not s.get('success') or s.get('beta_shape_guard_cells') != 0 for s in endpoints):
            raise RuntimeError('The saved sensitivity fit does not pass both endpoint gates')
        year = 2019 if name == 'historical_2019' else 2025
        baseline_tag = 'application_final_candidate_v2'
        baseline = OUT / baseline_tag / f'aplicacion_{year}_who2024/district_predictions.csv'
        source = current / 'district_predictions.csv'
        base, variant = read(baseline), read(source)
        if set(base.index) != set(variant.index):
            raise ValueError('Unequal application geography')
        variant = variant.loc[base.index]
        sampled = base.sampled.to_numpy(bool)
        for outcome in ('anemia', 'stunting'):
            delta = variant[f'p_mean_{outcome}'].to_numpy() - base[f'p_mean_{outcome}'].to_numpy()
            width0 = base[f'hi95_{outcome}'].to_numpy() - base[f'lo95_{outcome}'].to_numpy()
            width1 = variant[f'hi95_{outcome}'].to_numpy() - variant[f'lo95_{outcome}'].to_numpy()
            if not np.isfinite(np.concatenate([delta,width0,width1])).all():
                raise ValueError('Nonfinite sensitivity prediction')
            for predictions in (base,variant):
                mean,low,high = [predictions[f'{prefix}_{outcome}'].to_numpy() for prefix in ('p_mean','lo95','hi95')]
                if (mean < 0).any() or (mean > 1).any() or (low < 0).any() or (high > 1).any() or (low > high).any():
                    raise ValueError('Invalid sensitivity probability or interval bounds')
            for group, mask in [('all', np.ones(1874,bool)),('sampled',sampled),('unsampled',~sampled)]:
                rows.append(dict(variant=name,year=year,outcome=outcome,group=group,districts=int(mask.sum()),
                                 mean_absolute_change_pp=float(100*np.mean(abs(delta[mask]))),
                                 max_absolute_change_pp=float(100*np.max(abs(delta[mask]))),
                                 root_mean_square_change_pp=float(100*np.sqrt(np.mean(delta[mask]**2))),
                                 baseline_mean_width_pp=float(100*np.mean(width0[mask])),
                                 variant_mean_width_pp=float(100*np.mean(width1[mask])),
                                 scope='Descriptive independently fitted maps; no latent truth or joint contrast posterior'))
        paths.extend([{'path':str(p.relative_to(ROOT)), 'sha256':digest(p)} for p in (baseline,source,current/'fit_status.json')])
    pd.DataFrame(rows).to_csv(folder / 'descriptive_summary.csv', index=False)
    (folder / 'summary_manifest.json').write_text(json.dumps(dict(status='complete', rows=len(rows),
                 new_independent_populations=0,source_sha256=digest(Path(__file__)),inputs=paths,
                 declared_jobs=7,completed_endpoint_gates=14,
                 interpretation='Changes of posterior means and widths, not estimation error or health change'),indent=2))


if __name__ == '__main__':
    main()
