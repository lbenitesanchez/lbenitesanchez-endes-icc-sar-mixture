"""Application-only computational recovery of the unchanged beta-power posterior.

The independent simulation cohort retains its frozen dense-metric configuration.
This runner preserves the completed full-data application and refits only the
territorial training sample with a diagonal metric. All four-chain gates remain.
Private sampler files are diagnostic material and must not be redistributed.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import beta_power_application as application
import beta_power_hmc as hmc
from models import spatial_basis

ROOT = application.ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DiagonalMetric:
    def __init__(self, model):
        self.model = model

    def sample(self, **kwargs):
        matrix = np.asarray(kwargs['inv_metric'])
        if matrix.ndim != 2 or np.any(np.diag(matrix) <= 0):
            raise ValueError('Invalid positive diagonal of initial covariance')
        kwargs['inv_metric'] = np.diag(matrix).copy()
        return self.model.sample(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--source-tag', default='application_final_candidate_v1')
    parser.add_argument('--output-tag', default='application_final_candidate_recovery_v1')
    args = parser.parse_args()
    source = ROOT / 'resultados/resultados_comparadores' / args.source_tag / f'aplicacion_{args.year}_who2024'
    out = ROOT / 'resultados/resultados_comparadores' / args.output_tag / f'aplicacion_{args.year}_who2024'
    if out.exists():
        raise FileExistsError(out)
    full_status = json.loads((source / 'fit_status.json').read_text())
    if not all(s['success'] for s in full_status['endpoints']):
        raise RuntimeError('The full-data application must already have passed all gates')
    allareas, X, y, D, ne, idx, df, unsupported, data_manifest = application.app.dataset(args.year, 'who2024')
    units = application.prepare_units(args.year, 'who2024', allareas)
    basis = spatial_basis(allareas[['latitude', 'longitude']].to_numpy(float), rank=40)
    folds = pd.read_csv(ROOT / f'resultados/aplicacion_{args.year}_who2024/validation_folds.csv', dtype={'ubigeo': str})
    fold = folds.set_index('ubigeo').loc[allareas.iloc[idx].ubigeo]
    locked = fold.locked_test.to_numpy(bool)
    train = np.flatnonzero(~locked)
    test = np.flatnonzero(locked)
    training_units = units.loc[units.area.isin(idx[train])]
    if set(training_units.area) & set(idx[test]):
        raise AssertionError('Held-out responses leaked into model fitting')
    checks = []
    for j, name in enumerate(('anemia', 'stunting')):
        u = training_units.loc[np.isfinite(training_units[f'y_{name}'])]
        w = u[f'weight_{name}']; total = w.groupby(u.area).sum()
        mean = (w * u[f'y_{name}']).groupby(u.area).sum() / total
        kish = total**2 / (w*w).groupby(u.area).sum()
        dy = float(np.nanmax(abs(mean.reindex(idx[train]).to_numpy() - y[train, j])))
        dn = float(np.nanmax(abs(kish.reindex(idx[train]).to_numpy() - ne[train, j])))
        if dy > 1e-10 or dn > 1e-8:
            raise AssertionError((name, dy, dn))
        checks.append(dict(outcome=name, max_direct_difference=dy, max_Kish_difference=dn))
    out.mkdir(parents=True)
    for name in ('district_predictions.csv', 'fit_status.json', 'data_checks.json'):
        shutil.copy2(source / name, out / name)
    hmc._MODEL = DiagonalMetric(hmc.get_model())
    sampler = dict(metric='diag_e', adapt_delta=.95, retry_adapt_delta=.99,
                   initial_metric='diagonal of unrepaired positive MAP covariance; otherwise identity')
    tick = time.perf_counter()
    result = hmc.fit(X, y[train], ne[train], idx[train], training_units, basis,
                    seed=20261005+args.year+10000, sampler_config=sampler)
    # Test responses are used only after the posterior computations finish.
    detail = allareas.iloc[idx[test]][['ubigeo', 'prov_code']].reset_index(drop=True)
    for j, name in enumerate(('anemia', 'stunting')):
        detail[f'direct_{name}'] = y[test, j]
        detail[f'pred_{name}'] = result['p_mean'][idx[test], j]
        detail[f'binary_NLL_{name}'] = application.score(result['p_mean'][idx[test], j], y[test, j])
        detail[f'squared_error_{name}'] = (result['p_mean'][idx[test], j] - y[test, j])**2
    detail.to_csv(out / 'province_test_predictions.csv', index=False)
    (out / 'test_fit_status.json').write_text(json.dumps(result['status'], indent=2))
    (out / 'training_design_checks.json').write_text(json.dumps(checks, indent=2))
    complete = all(s['success'] for s in result['status'])
    manifest = dict(year=args.year, rule='who2024', q=40, data=data_manifest,
                    full_endpoints_converged=True, heldout_endpoints_converged=complete,
                    recovery_scope='Application-only computational change; likelihood and all statistical priors unchanged',
                    sampler=sampler, full_data_source=str(source.relative_to(ROOT)),
                    full_data_sha256={n: digest(source/n) for n in ('district_predictions.csv','fit_status.json','data_checks.json')},
                    source_sha256=digest(Path(__file__)), heldout_fit_seconds=time.perf_counter()-tick,
                    simulation_protocol_changed=False, selection_uses_heldout_responses=False,
                    analysis_scope='Exploratory marginal binary scoring on previously reused province folds',
                    uncertainty='Conditional working ICC; no claim of latent ENDES coverage')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(completed=complete, year=args.year, seconds=time.perf_counter()-tick)), flush=True)
    if not complete:
        raise RuntimeError('Recovery did not satisfy the unchanged four-chain gates')


if __name__ == '__main__':
    main()
