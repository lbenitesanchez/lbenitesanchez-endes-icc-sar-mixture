"""Recover JSON metadata after a fully completed frozen Hogg CLI's type error.

This operational repair never fits a model or changes a scientific source.
The original wrapper exit remains an incident; recovery requires final gates,
actual retained chains, unchanged input/source identities and identical maps.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
from pathlib import Path
import argparse
import copy
import datetime as dt
import json
import re
import traceback
import numpy as np
import pandas as pd
from validacion_fit_models import (ROOT, CORE, gate, sha, fit_seed, unit_dataset,
                                   assert_training_only, h, hd, sp)


def normalize(value):
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, np.ndarray):
        return normalize(value.tolist())
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('Nonfinite metadata cannot be repaired into success')
    return value


def header(path):
    found = {}
    with path.open() as stream:
        for row in stream:
            if not row.startswith('#'):
                break
            hit = re.match(r'#\s*(\w+)\s*=\s*(.*?)\s*(?:\(Default\))?\s*$', row)
            if hit and hit[1] in ('seed', 'num_samples', 'num_warmup', 'delta', 'max_depth', 'id', 'save_warmup', 'model'):
                found[hit[1]] = hit[2].replace(' (Default)', '').strip()
    return found


def diagnostic_audit(directory, status, stage, seed, spatial=False, with_prevalence=False):
    config = status['config']
    cfg = h.MCMCConfig(**config)
    files = sorted((directory / stage).glob('*.csv'))
    if len(files) != 4:
        raise RuntimeError('Four actual retained chains required for ' + str(directory / stage))
    summaries = pd.read_csv(directory / (stage + '_parameter_summary.csv'), index_col=0)
    if spatial:
        names = summaries.index.to_series().astype(str)
        sampled = names.str.startswith(('intercept', 'beta[', 'sigma_area', 'rho', 'spatial_z[', 'omega[', 'sigma_gvf'))
        summaries = summaries.loc[sampled.values]
    methods, prevalence, records = [], [], []
    for index, path in enumerate(files):
        meta = header(path)
        expected = {'seed': str(seed), 'num_samples': str(cfg.samples), 'num_warmup': str(cfg.warmup),
                    'delta': str(cfg.adapt_delta), 'max_depth': str(cfg.max_treedepth), 'id': str(index+1),
                    'save_warmup': 'false'}
        if any(meta.get(key) != val for key, val in expected.items()):
            raise RuntimeError('Actual chain seed/sampler header differs: ' + str(path))
        columns = ['energy__', 'divergent__', 'treedepth__']
        data = pd.read_csv(path, comment='#', usecols=lambda name: name in columns or
                           (with_prevalence and name.startswith('prevalence.')))
        if len(data) != cfg.samples or not np.isfinite(data.to_numpy(float)).all():
            raise RuntimeError('Retained chain is incomplete or nonfinite')
        methods.append(data[columns].to_numpy(float))
        if with_prevalence:
            prevalence.append(data[[f'prevalence.{j+1}' for j in range(sum(name.startswith('prevalence.') for name in data))]].to_numpy(float))
        records.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'header': meta,
                        'retained_rows': len(data)})
    methods = np.stack(methods, axis=1)

    class RetainedMethods:
        chains = 4
        num_draws_sampling = cfg.samples
        def method_variables(self):
            return {name: methods[:, :, j] for j, name in enumerate(columns)}

    actual = normalize(h.diagnose_fit(RetainedMethods(), cfg, summary=summaries))
    expected = status['stage2' if spatial else stage]
    for key in actual:
        if key == 'thresholds':
            if actual[key] != expected[key]:
                raise RuntimeError('Recorded diagnostic thresholds differ')
        elif isinstance(actual[key], list):
            if not np.allclose(actual[key], expected[key], atol=1e-12, rtol=0):
                raise RuntimeError('Recorded actual-chain diagnostic differs: ' + key)
        elif isinstance(actual[key], (int, bool)):
            if actual[key] != expected[key]:
                raise RuntimeError('Recorded actual-chain diagnostic differs: ' + key)
        elif abs(actual[key] - expected[key]) > 1e-12:
            raise RuntimeError('Recorded actual-chain diagnostic differs: ' + key)
    if not actual['success']:
        raise RuntimeError('A final retained-chain diagnostic gate failed')
    return {'actual_diagnostics': actual, 'chains': records,
            'summary_sha256': sha(directory / (stage + '_parameter_summary.csv'))}, (
            np.concatenate(prevalence) if with_prevalence else None)


def recover(year, units_path, auxiliary, output, original_log=None, original_exec_session=None):
    protocol_sha = gate(year)
    if (output / 'status.json').exists() or (output / 'status_recovered.json').exists():
        raise FileExistsError('Preserve existing Hogg wrapper/recovered status')
    if year == 2024 and original_log is None:
        raise RuntimeError('2024 automatic recovery requires the original complete frozen-CLI log')
    if original_log is not None:
        trace = original_log.read_text()
        if ('TypeError: Object of type ndarray is not JSON serializable' not in trace
                or "json.dumps(metadata, indent=2)" not in trace
                or 'validacion_fit_models.py' not in trace):
            raise RuntimeError('Frozen CLI failed for a different reason; no automatic recovery')
    areas = pd.read_csv(auxiliary, dtype={'ubigeo': str})
    areas['ubigeo'] = areas.ubigeo.str.zfill(6)
    units, excluded = unit_dataset(units_path, areas)
    assert_training_only(units, year, units_path)
    units, features, feature_meta = hd.age_sex_features(units)
    source_dir = output / 'age_sex_source'
    spatial_dir = output / 'spatial'
    source_status = json.loads((source_dir / 'status.json').read_text())
    spatial_status = json.loads((spatial_dir / 'status.json').read_text())
    if not source_status.get('success') or not spatial_status.get('success'):
        raise RuntimeError('Hogg source/spatial final gates did not pass')
    audits, recovery = {}, {}
    predictions = pd.read_csv(output / 'predictions.csv', dtype={'ubigeo': str})
    if not np.array_equal(predictions.ubigeo, areas.ubigeo):
        raise RuntimeError('Hogg final prediction frame/order differs from the fixed auxiliaries')
    map_errors = {}
    for j, (endpoint, response) in enumerate((('anemia', 'y_anemia_who2024'), ('stunting', 'y_stunting'))):
        source_attempt = len(source_status['all_attempts'][endpoint])-1
        source_run = source_dir / endpoint / f'attempt_{source_attempt}'
        final_source = source_status['outcomes'][endpoint]
        expected_seed = fit_seed(year, 'hogg') + 100*j
        if final_source['config']['seed'] != expected_seed:
            raise RuntimeError('Frozen endpoint seed differs')
        audits[endpoint] = {}
        for stage, seed_offset in (('stage1', 0), ('stage2', 1)):
            audit, _ = diagnostic_audit(source_run, final_source, stage, expected_seed+seed_offset)
            audits[endpoint][stage] = audit
        codes, data, proof = sp.recover_stage2_from_saved(source_run, units, areas,
            outcome=response, weight='weight', area_col='ubigeo', area_features=h.DEFAULT_FEATURES,
            unit_features=features, population_N=None)
        saved = output / f'source_modular_summary_{endpoint}.npz'
        with np.load(saved) as old:
            if not np.array_equal(codes, old['area']):
                raise RuntimeError('Recovered S1 area codes differ')
            errors = {name: float(np.max(np.abs(np.asarray(data[name])-old[name]))) if np.asarray(data[name]).size else 0.
                      for name in data}
        if max(errors.values()) > 1e-12:
            raise RuntimeError('Recovered S1 modular data differ from the actual fitted summaries')
        recovery[endpoint] = proof
        audits[endpoint]['stage1_modular_summary'] = {'sha256': sha(saved), 'max_difference_by_field': errors}
        spatial_attempt = len(spatial_status['all_attempts'][endpoint])-1
        spatial_run = spatial_dir / endpoint / f'attempt_{spatial_attempt}'
        final_spatial = spatial_status['outcomes'][endpoint]
        if final_spatial['config']['seed'] != expected_seed:
            raise RuntimeError('Frozen spatial endpoint seed differs')
        audit, draws = diagnostic_audit(spatial_run, final_spatial, 'stage2', expected_seed,
                                        spatial=True, with_prevalence=True)
        audits[endpoint]['spatial_stage2'] = audit
        means, lo, hi = draws.mean(axis=0), *np.quantile(draws, [.025, .975], axis=0)
        differences = {key: float(np.max(np.abs(values-predictions[prefix+endpoint].to_numpy(float))))
                       for key, prefix, values in (('mean', 'p_mean_', means), ('lower95', 'lo95_', lo), ('upper95', 'hi95_', hi))}
        with np.load(spatial_run / 'posterior_area_draws.npz') as saved_draws:
            differences['saved_posterior_draws'] = float(np.max(np.abs(draws-saved_draws['prevalence'])))
        if max(differences.values()) > 1e-12:
            raise RuntimeError('Actual spatial-chain maps differ from the frozen wrapper output')
        map_errors[endpoint] = differences
    compiled = {}
    for name in ('hogg_stage1', 'hogg_stage2', 'hogg_stage2_bym2'):
        stamp = CORE / (name+'_compiled_source.json')
        proof = json.loads(stamp.read_text())
        if (sha(CORE / (name+'.stan')) != proof['stan_sha256']
                or sha(CORE/name) != proof['executable_sha256']):
            raise RuntimeError('Compiled source/executable identity changed')
        compiled[name] = {'stamp_sha256': sha(stamp), **proof}
    metadata = {'year': year, 'method': 'hogg', 'protocol_sha256': protocol_sha,
        'input_sha256': {'units': sha(units_path), 'aux': sha(auxiliary)},
        'excluded_no_auxiliary_records_by_domain': excluded,
        'input_response_scope': 'development2019' if year == 2019 else 'reserved_2024_training_provinces_only',
        'source_sha256': sha(ROOT / 'scripts/validacion_fit_models.py'),
        'historical_sources_modified': False, 'reserved_test_responses_read': False,
        'success': True, 'source_status': source_status, 'spatial_status': spatial_status,
        'stage1_recovery': recovery, 'features': feature_meta,
        'information_budget': 'Six common district features plus training survey age/sex in S1; spatial S2',
        'variance_limit': 'Hogg iid S1 sampling variance, no PSU correction; Australian spatial prior transfer unchanged',
        'graph_limit': 'Coordinate symmetric eight-nearest-neighbour graph, not verified district border contiguity',
        'seconds': None, 'prediction_sha256': sha(output / 'predictions.csv')}
    # Reproduce the documented plain-json failure with the actual NumPy BFMI
    # object type returned by the unchanged scientific diagnostic function.
    type_replay = copy.deepcopy(metadata)
    type_replay['source_status']['outcomes']['anemia']['stage1']['BFMI_per_chain'] = np.asarray(
        type_replay['source_status']['outcomes']['anemia']['stage1']['BFMI_per_chain'])
    try:
        json.dumps(type_replay, indent=2)
    except TypeError as error:
        if str(error) != 'Object of type ndarray is not JSON serializable':
            raise
        replay_trace = traceback.format_exc()
    else:
        raise RuntimeError('Documented NumPy serialization incident was not reproduced')
    if normalize(type_replay) != metadata:
        raise RuntimeError('Type normalization changed a scientific metadata value')
    incident = {'status': 'RECOVERED_JSON_SERIALIZATION_ONLY_AFTER_ACTUAL_GATES',
        'recovered_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'operational_source_sha256': sha(Path(__file__)), 'original_wrapper_exit_code': 1,
        'original_exception': 'TypeError: Object of type ndarray is not JSON serializable',
        'original_log_sha256': sha(original_log) if original_log is not None else None,
        'original_exec_session': original_exec_session,
        'missing_original_log_limitation': 'Original frozen CLI trace remains in the tool session; retained CmdStan stdout files remain untouched.' if original_log is None else None,
        'exception_type_reproduced': True, 'type_normalization_preserves_values': True,
        'elapsed_seconds_reason': 'elapsed_wrapper_seconds_not_recoverable_after_json_failure',
        'preserved_recorded_component_seconds': {'source': source_status['seconds_total'], 'spatial': spatial_status['seconds_total']},
        'no_refit': True, 'no_frozen_source_modified': True, 'no_statistical_choice_modified': True,
        'actual_retained_chains': audits, 'actual_chain_map_max_differences': map_errors,
        'tolerance': 1e-12, 'compiled_source_provenance': compiled,
        'source_status_sha256': sha(source_dir/'status.json'),
        'spatial_status_sha256': sha(spatial_dir/'status.json')}
    (output / 'serialization_incident_replay.txt').write_text(replay_trace)
    incident_path = output / 'serialization_recovery_manifest.json'
    incident_path.write_text(json.dumps(normalize(incident), indent=2, allow_nan=False))
    metadata['operational_serialization_recovery'] = {'incident_manifest': str(incident_path.relative_to(ROOT)),
        'incident_manifest_sha256': sha(incident_path), 'original_wrapper_exit_code': 1,
        'elapsed_seconds_reason': incident['elapsed_seconds_reason']}
    status_path = output / 'status_recovered.json'
    status_path.write_text(json.dumps(normalize(metadata), indent=2, allow_nan=False))
    return {'status': 'HOGG_STATUS_OPERATIONALLY_RECOVERED', 'success': True,
            'status_path': str(status_path.relative_to(ROOT)), 'status_sha256': sha(status_path),
            'prediction_sha256': sha(output / 'predictions.csv'), 'actual_chain_map_max_differences': map_errors}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--year', type=int, choices=[2019, 2024], required=True)
    p.add_argument('--units', type=Path, required=True)
    p.add_argument('--aux', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--original-log', type=Path)
    p.add_argument('--original-exec-session', type=int)
    args = p.parse_args()
    print(json.dumps(recover(args.year, args.units.resolve(), args.aux.resolve(), args.output.resolve(),
                            args.original_log, args.original_exec_session)), flush=True)


if __name__ == '__main__':
    main()
