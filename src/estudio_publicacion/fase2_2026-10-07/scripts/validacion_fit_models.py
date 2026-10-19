"""Phase2 frozen baseline and strong Hogg fits; reserved responses are gated.

Every output stays in phase2. Existing phase1 sources/compile stamps are read
only. Hogg retains measured age/sex in S1 and its published BYM2 S2 component;
its extra survey information and iid S1 variance limitation are disclosed.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT.parent
CORE = STUDY / 'scripts/competitors'
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(STUDY / 'scripts'))
import hogg_tsln as h
import hogg_diagnostics as hd
import hogg_spatial as sp
import beta_power_hmc as beta_hmc
from beta_power_application_recovery import DiagonalMetric
from models import spatial_basis


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_seed(year, method):
    """CmdStan requires uint32; bound all inherited endpoint/retry offsets."""
    seed = 20261007 + (50 if method == 'baseline' else 100) + int(year)
    max_offset = 1009 + 104729 if method == 'baseline' else 100 + 21
    if method not in ('baseline', 'hogg') or not 0 < seed < 2**32 - max_offset:
        raise ValueError('CmdStan seed plus inherited offsets must fit uint32')
    return seed


def gate(year):
    p = ROOT / 'protocol_freeze.json'
    proof = json.loads(p.read_text())
    if not proof.get('approved_for_development_fits', False):
        raise RuntimeError('Frozen development fitting gate unavailable')
    if year == 2024 and not proof.get('approved_for_reserved_outcome_access', False):
        # Access approval may be a separate append-only operational lock; it
        # must identify the unchanged protocol SHA explicitly.
        access = ROOT / 'validacion/externa_reserved_access_approval.json'
        a = json.loads(access.read_text()) if access.exists() else {}
        if not a.get('approved_for_reserved_outcome_access', False) or a.get('protocol_sha256') != sha(p):
            raise RuntimeError('Reserved responses remain locked')
    frozen = proof.get('source_sha256', {})
    for name, value in frozen.items():
        path = ROOT / name
        if not path.exists():
            path = STUDY / name
        if not path.exists() or sha(path) != value:
            raise RuntimeError('Frozen source changed: ' + name)
    return sha(p)


def load_binary(name):
    source = CORE / (name + '.stan')
    executable = CORE / name
    stamp = CORE / (name + '_compiled_source.json')
    if not stamp.exists():
        # The beta executable lacks a phase1 compile stamp. Compile a byte-exact
        # source/include snapshot in phase2 rather than writing any phase1 file.
        import shutil
        cache = ROOT / 'scripts/validacion_models'
        cache.mkdir(parents=True, exist_ok=True)
        local_source = cache / source.name
        included = CORE / 'beta_power_stable_functions.stan'
        local_include = cache / included.name
        for original, local in [(source, local_source), (included, local_include)]:
            if local.exists() and sha(local) != sha(original):
                raise RuntimeError('Phase2 compilation snapshot differs from frozen kernel')
            if not local.exists():
                shutil.copyfile(original, local)
        local_stamp = cache / (name + '_compiled_source.json')
        old = json.loads(local_stamp.read_text()) if local_stamp.exists() else {}
        model = h._backend().CmdStanModel(stan_file=str(local_source),
                 force_compile=old.get('stan_sha256') != sha(local_source) or old.get('include_sha256') != sha(local_include))
        proof = {'stan_sha256': sha(local_source), 'include_sha256': sha(local_include),
                 'executable_sha256': sha(model.exe_file), 'phase1_sources_modified': False}
        local_stamp.write_text(json.dumps(proof, indent=2))
        return model
    proof = json.loads(stamp.read_text())
    if sha(source) != proof['stan_sha256'] or sha(executable) != proof['executable_sha256']:
        raise RuntimeError('Compiled source/executable mismatch: ' + name)
    return h._backend().CmdStanModel(stan_file=str(source), exe_file=str(executable), compile=False)


def unit_dataset(path, areas):
    units = pd.read_csv(path, dtype={'ubigeo': str, 'psu_key': str})
    units['ubigeo'] = units.ubigeo.str.zfill(6)
    lookup = dict(zip(areas.ubigeo, np.arange(len(areas))))
    units['area'] = units.ubigeo.map(lookup)
    excluded = units.loc[units.area.isna()].groupby('ubigeo').size().to_dict()
    units = units.loc[units.area.notna()].copy()
    units['area'] = units.area.astype(int)
    units['psu'] = units.psu_key
    units['y_anemia'] = units.y_anemia_who2024
    for name in ('anemia', 'stunting'):
        units[f'weight_{name}'] = np.where(np.isfinite(units[f'y_{name}']), units.weight, np.nan)
    return units, excluded


def assert_training_only(units, year, path):
    if 'year' not in units or not units.year.eq(year).all():
        raise RuntimeError('Survey edition in unit records differs from declared training edition')
    if year == 2024 and 'test' in Path(path).name.lower():
        raise RuntimeError('Reserved territorial test response file forbidden in fitting process')
    if year == 2024:
        held = units.ubigeo.str[:4].map(lambda province: int(hashlib.sha256(('ENDES-phase2-20261007|' + province).encode()).hexdigest()[:8], 16) % 5 == 0)
        if held.any():
            raise RuntimeError('Reserved province response passed training boundary')


def assert_prepared_training(p, units, areas, year):
    idx = np.asarray(p['sample_idx'], int)
    if not set(idx).issubset(set(units.area)):
        raise RuntimeError('Prepared outcome contains a nontraining domain')
    if year == 2024:
        codes = areas.ubigeo.iloc[idx]
        held = codes.str[:4].map(lambda province: int(hashlib.sha256(('ENDES-phase2-20261007|' + province).encode()).hexdigest()[:8], 16) % 5 == 0)
        if held.any():
            raise RuntimeError('Prepared outcome includes reserved province')
    for j, endpoint in enumerate(('anemia', 'stunting')):
        u = units.loc[np.isfinite(units[f'y_{endpoint}']) & np.isfinite(units[f'weight_{endpoint}'])]
        weights = u[f'weight_{endpoint}']
        sums = weights.groupby(u.area).sum()
        direct = (weights * u[f'y_{endpoint}']).groupby(u.area).sum() / sums
        ne = sums ** 2 / (weights ** 2).groupby(u.area).sum()
        if not np.allclose(direct.reindex(idx).to_numpy(), p['y'][:, j], rtol=0, atol=1e-10, equal_nan=True):
            raise RuntimeError('Prepared direct response differs from training-only child records')
        if not np.allclose(ne.reindex(idx).to_numpy(), p['kish_n'][:, j], rtol=0, atol=1e-8, equal_nan=True):
            raise RuntimeError('Prepared Kish-n differs from training-only child records')


def graph(areas, dest):
    from scipy.spatial.distance import cdist
    latitude = areas.latitude.to_numpy(float)
    xy = np.column_stack([111.32 * areas.longitude.to_numpy(float) * np.cos(np.deg2rad(latitude.mean())), 111.32 * latitude])
    distance = cdist(xy, xy)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argpartition(distance, 7, axis=1)[:, :8]
    pd.DataFrame({'area1': np.repeat(areas.ubigeo.to_numpy(), 8),
                  'area2': areas.ubigeo.to_numpy()[nearest.ravel()]}).to_csv(dest, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['baseline', 'hogg'], required=True)
    parser.add_argument('--year', type=int, choices=[2019, 2024], required=True)
    parser.add_argument('--units', type=Path, required=True)
    parser.add_argument('--aux', type=Path, required=True)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parallel-chains', type=int, default=1)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    protocol_sha = gate(args.year)
    areas = pd.read_csv(args.aux, dtype={'ubigeo': str})
    areas['ubigeo'] = areas.ubigeo.str.zfill(6)
    if areas.ubigeo.duplicated().any():
        raise ValueError('Auxiliary district duplicates')
    units, excluded = unit_dataset(args.units, areas)
    assert_training_only(units, args.year, args.units)
    metadata = {'year': args.year, 'method': args.method, 'protocol_sha256': protocol_sha,
                'input_sha256': {'units': sha(args.units), 'aux': sha(args.aux)},
                'excluded_no_auxiliary_records_by_domain': excluded,
                'input_response_scope': 'development2019' if args.year == 2019 else 'reserved_2024_training_provinces_only',
                'source_sha256': sha(Path(__file__)),
                'historical_sources_modified': False, 'reserved_test_responses_read': False}
    if args.preflight:
        print(json.dumps({**metadata, 'fit_dispatched': False}), flush=True)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    if args.method == 'baseline':
        if args.prepared is None:
            raise ValueError('Baseline needs common prepared X/y/Kish-n/basis')
        p = np.load(args.prepared / 'fit_inputs.npz')
        code = pd.read_csv(args.prepared / 'area_ids.csv', dtype={'ubigeo': str})
        if not np.array_equal(code.ubigeo, areas.ubigeo):
            raise ValueError('Common auxiliary/prepared district order differs')
        assert_prepared_training(p, units, areas, args.year)
        beta_hmc._MODEL = DiagonalMetric(load_binary('beta_power_hybrid'))
        fit = beta_hmc.fit(p['X'], p['y'], p['kish_n'], p['sample_idx'], units, p['basis'],
                           seed=fit_seed(args.year, 'baseline'),
                           config={'spatial_sd': .5, 'cluster_adjustment': True},
                           sampler_config={'metric': 'diag_e', 'adapt_delta': .95, 'retry_adapt_delta': .99,
                                           'parallel_chains': args.parallel_chains},
                           output_dir=args.output / 'private_sampler')
        predictions = areas[['ubigeo']].copy()
        for j, name in enumerate(('anemia', 'stunting')):
            for key, target in [('p_mean', 'p_mean_'), ('lower95', 'lo95_'), ('upper95', 'hi95_')]:
                predictions[target + name] = fit[key][:, j]
        predictions.to_csv(args.output / 'predictions.csv', index=False)
        metadata.update(success=all(s['success'] and s['beta_shape_guard_cells'] == 0 for s in fit['status']), endpoints=fit['status'],
                        preparation_manifest_sha256=sha(args.prepared / 'preparation_manifest.json'),
                        information_budget='Six frozen district auxiliary features; training survey workingICC')
    else:
        units, features, feature_meta = hd.age_sex_features(units)
        cfg = h.MCMCConfig(parallel_chains=args.parallel_chains, seed=fit_seed(args.year, 'hogg'),
                           phase='phase2_temporal2019' if args.year == 2019 else 'phase2_2024_territorial')
        models = {name: load_binary(name) for name in ('hogg_stage1', 'hogg_stage2')}
        source = args.output / 'age_sex_source'
        pair = h.fit_hogg_pair(units, areas, area_col='ubigeo', area_features=h.DEFAULT_FEATURES,
                               unit_features=features, population_N=None, config=cfg, output_dir=source, models=models,
                               outcomes=('y_anemia_who2024', 'y_stunting'), weights=('weight', 'weight'),
                               retry_on_diagnostics=True, retain_sampler_csv=True)
        if not pair['status']['success']:
            h.write_json(args.output / 'status.json', {**metadata, 'success': False, 'source_status': pair['status']})
            raise RuntimeError('Hogg source age/sex fit failed final diagnostic gate')
        edges = args.output / 'adjacency_edges.csv'
        graph(areas, edges)
        summaries = {}
        recovery = {}
        for endpoint, response in [('anemia', 'y_anemia_who2024'), ('stunting', 'y_stunting')]:
            attempt = len(pair['status']['all_attempts'][endpoint]) - 1
            codes, data, proof = sp.recover_stage2_from_saved(source / endpoint / f'attempt_{attempt}', units, areas,
                                outcome=response, weight='weight', area_col='ubigeo', area_features=h.DEFAULT_FEATURES,
                                unit_features=features, population_N=None)
            dest = args.output / f'source_modular_summary_{endpoint}.npz'
            np.savez_compressed(dest, area=codes, **data)
            summaries[endpoint] = dest
            recovery[endpoint] = proof
        spatial = sp.fit_spatial_pair(summaries, edges, output_dir=args.output / 'spatial', config=cfg,
                                     model=load_binary('hogg_stage2_bym2'), retry_on_diagnostics=True, retain_sampler_csv=True)
        spatial['predictions'].rename(columns={'area': 'ubigeo'}).to_csv(args.output / 'predictions.csv', index=False)
        metadata.update(success=spatial['status']['success'], source_status=pair['status'], spatial_status=spatial['status'],
                        stage1_recovery=recovery, features=feature_meta,
                        information_budget='Six common district features plus training survey age/sex in S1; spatial S2',
                        variance_limit='Hogg iid S1 sampling variance, no PSU correction; Australian spatial prior transfer unchanged',
                        graph_limit='Coordinate symmetric eight-nearest-neighbour graph, not verified district border contiguity')
    metadata['seconds'] = time.monotonic() - started
    metadata['prediction_sha256'] = sha(args.output / 'predictions.csv')
    (args.output / 'status.json').write_text(json.dumps(metadata, indent=2))
    if not metadata['success']:
        raise RuntimeError('Final phase2 model numerical gate failed')
    print(json.dumps({k: metadata[k] for k in ['method', 'year', 'success', 'seconds', 'prediction_sha256']}), flush=True)


if __name__ == '__main__':
    main()
