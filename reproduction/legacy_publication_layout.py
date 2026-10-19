#!/usr/bin/env python3
"""Normalize the completed private publication layout without changing results.

The frozen scorer defaults to externa_confirmatory; frozen publication helpers
read externa_final. A checked symlink exposes the same fresh files under that
publication name. Partial figures from a failed finalization are moved intact
to a new backup directory. This helper never fits, scores, changes source, loads
microdata/draw arrays, or promotes numerical agreement to PASS.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import itertools
import json
import os
from pathlib import Path

PHASE = 'fase2_2026-10-07'
STAGES = ('previous800', 'cut40', 'application2025', 'external2024', 'cut16')
SCOPES = ('temporal', 'territorial')
ENDPOINTS = ('anemia', 'stunting')
METHODS = ('BetaPowerExisting', 'HoggAgeSexSpatial', 'BetaDesignUncertainty')
ALLOWLIST = ('completion.json', 'summary.csv', 'contrasts_Holm8.csv',
             'district_scores.csv', 'excluded_district_endpoints.csv')
SCHEMAS = {
    'summary.csv': ('scope', 'outcome', 'method', 'districts', 'provinces',
                    'binary_NLL', 'RMSE_noisy_direct_pp', 'Brier_binary_expectation',
                    'clipped_cells', 'latent_coverage_observed'),
    'contrasts_Holm8.csv': ('scope', 'outcome', 'candidate', 'reference', 'districts',
                          'difference', 'cluster_standard_error', 'province_clusters',
                          't_df', 'p_two_sided', 'unadjusted_t_ci_low',
                          'unadjusted_t_ci_high', 'descriptive_bootstrap_ci_low',
                          'descriptive_bootstrap_ci_high', 'bootstrap_repetitions',
                          'p_Holm8', 'candidate_significantly_better',
                          'candidate_significantly_worse'),
    'district_scores.csv': ('ubigeo', 'pred_anemia', 'pred_stunting', 'method',
                           'province', 'outcome', 'direct',
                           'weighted_observed_denominator',
                           'valid_positive_weight_records', 'binary_NLL',
                           'squared_error_against_noisy_direct',
                           'Brier_binary_expectation', 'numerical_clipping_applied',
                           'scope'),
    'excluded_district_endpoints.csv': ('ubigeo', 'province', 'outcome', 'direct',
                                       'weighted_observed_denominator',
                                       'valid_positive_weight_records', 'scope', 'reason'),
}
FIGURE_STEMS = ('design_diagnostics', 'external_contrasts',
                'application2025_ICC_capitals', 'application2025_capitals')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def csv_rows(path, expected):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == expected, 'CSV schema differs: ' + path.name)
        return list(reader)


def validate(study, finalization, reference=None):
    run = study.parent
    phase = study / PHASE
    source = phase / 'resultados/externa_confirmatory'
    preflight = read(finalization / 'preflight.json')
    require(preflight.get('status') == 'PASS_ALL_FRESH_COHORTS_AND_CORRECTED_LEGACY_LEDGERS_CLOSED'
            and preflight.get('approved_for_final_scoring') is True
            and not preflight.get('pending') and not preflight.get('failed'), 'Fresh cohort preflight is not closed')
    stage_sha = {}
    for stage in STAGES:
        path = run / ('legacy_' + stage + '_execution.json')
        require(read(path)['status'] == 'ALL_COMMANDS_EXITED_ZERO', 'Stage is not qualified: ' + stage)
        require(sha(path) == preflight['corrected_legacy_execution_receipt_sha256'][path.name],
                'Stage receipt changed after finalization preflight: ' + stage)
        stage_sha[stage] = sha(path)
    require(read(finalization / 'finalization.json').get('finished_utc'), 'Original finalization is not terminal')
    retirement_path = run / 'previous800_relay/original_controller_retirement.json'
    retirement = read(retirement_path)
    require(retirement.get('status') == 'ORIGINAL_CONTROLLER_INTERRUPTED_PRESERVED'
            and retirement.get('original_full_controller_exit_zero_claimed') is False,
            'Original controller retirement proof is absent or misclassified')
    require(set(path.name for path in source.iterdir()) == set(ALLOWLIST)
            and all((source / name).is_file() and not (source / name).is_symlink()
                    for name in ALLOWLIST), 'Fresh aggregate directory differs from the exact allowlist')
    completion = read(source / 'completion.json')
    require(completion['status'] == 'COMPLETE_RESERVED_EXTERNAL_VALIDATION', 'External scoring is not complete')
    preparation = read(run / 'legacy_preparation.json')
    script = phase / 'scripts/validacion_score_external.py'
    require(sha(script) == completion['source_sha256']
            == preparation['source_sha256'][PHASE + '/scripts/validacion_score_external.py'],
            'Frozen scoring source identity changed')
    require(sha(phase / 'protocol_freeze.json') == completion['protocol_sha256'], 'External protocol identity changed')
    # Only parse constant syntax. Do not import or execute scoring functions.
    endpoint_pair_found = any(isinstance(node, ast.For)
                             and isinstance(node.iter, (ast.List, ast.Tuple))
                             and ast.literal_eval(node.iter) == [('anemia', 'y_anemia_who2024'),
                                                                ('stunting', 'y_stunting')]
                             for node in ast.walk(ast.parse(script.read_text()))
                             if isinstance(node, ast.For) and isinstance(node.iter, (ast.List, ast.Tuple)))
    require(endpoint_pair_found, 'Frozen scorer does not declare the WHO2024 anaemia endpoint')
    require(read(phase / 'validacion/externa_first_reserved_open.json').get('WHO2024_endpoint_preserved') is True,
            'Fresh WHO2024 import declaration is missing')
    files = {name: sha(source / name) for name in ALLOWLIST}
    require(set(completion['output_sha256']) == set(ALLOWLIST) - {'completion.json'},
            'Scoring output hash inventory differs from allowlist')
    for name, value in completion['output_sha256'].items():
        require(files[name] == value, 'Fresh scoring output changed: ' + name)
    rows = {name: csv_rows(source / name, SCHEMAS[name]) for name in SCHEMAS}
    summary_keys = [(row['scope'], row['outcome'], row['method']) for row in rows['summary.csv']]
    expected_summary = set(itertools.product(SCOPES, ENDPOINTS, METHODS))
    require(len(summary_keys) == 12 and set(summary_keys) == expected_summary,
            'Expected 12 external summary groups are not complete and unique')
    contrasts = [(row['scope'], row['outcome'], row['candidate'], row['reference'])
                 for row in rows['contrasts_Holm8.csv']]
    expected_contrasts = {(scope, endpoint, 'BetaDesignUncertainty', reference_method)
                         for scope, endpoint, reference_method
                         in itertools.product(SCOPES, ENDPOINTS, METHODS[:2])}
    require(len(contrasts) == 8 and set(contrasts) == expected_contrasts,
            'Expected Holm8 contrasts are not complete and unique')
    groups = {}
    for row in rows['district_scores.csv']:
        key = (row['scope'], row['outcome'], row['method'])
        require(key in expected_summary, 'Unexpected district aggregate group')
        require(len(row['ubigeo']) == 6 and row['ubigeo'].isdigit()
                and row['province'] == row['ubigeo'][:4], 'District aggregate area keys differ')
        groups.setdefault(key, []).append(row['ubigeo'])
    for row in rows['summary.csv']:
        key = (row['scope'], row['outcome'], row['method'])
        codes = groups.get(key, [])
        require(len(codes) == len(set(codes)) == int(row['districts'])
                and len({code[:4] for code in codes}) == int(row['provinces']),
                'District aggregate support differs from its fresh summary')
    reference_headers = {}
    if reference:
        root = reference / PHASE / 'resultados/externa_final'
        for name, expected in SCHEMAS.items():
            with (root / name).open(newline='') as handle:
                header = tuple(next(csv.reader(handle)))
            require(header == expected, 'Historical publication CSV header differs: ' + name)
            reference_headers[name] = list(header)
    locks = []
    inputs = {row['scope']: row for row in completion['inputs']}
    require(len(completion['inputs']) == 2 and set(inputs) == set(SCOPES), 'External input scopes differ')
    for scope in SCOPES:
        lock_path = phase / 'validacion' / ('externa_prediction_lock_' + scope + '.json')
        lock = read(lock_path)
        require(sha(lock_path) == inputs[scope]['prediction_lock_sha256']
                and lock['predictions'] == inputs[scope]['prediction_inputs']
                and lock['approved_for_scoring'] and not lock['reserved_territorial_test_outcomes_used_in_training'],
                'Completed score does not bind the fresh prediction lock: ' + scope)
        require(lock['protocol_sha256'] == completion['protocol_sha256'], 'Prediction lock protocol differs')
        require({row['method'] for row in lock['predictions']} == set(METHODS)
                and len(lock['predictions']) == 3, 'Prediction lock methods differ')
        expected_year = 2019 if scope == 'temporal' else 2024
        for prediction in lock['predictions']:
            status_path = phase / prediction['status_path']
            require(sha(phase / prediction['path']) == prediction['sha256']
                    and sha(status_path) == prediction['status_sha256'], 'Locked prediction/status changed')
            status = read(status_path)
            require(status['success'] and status['year'] == expected_year, 'Fresh prediction year/success differs')
            if prediction['method'] != 'BetaDesignUncertainty':
                require(status['reserved_test_responses_read'] is False
                        and status['historical_sources_modified'] is False,
                        'Fresh fitting response/source scope differs')
            elif scope == 'territorial':
                require(status['reserved_test_outcomes_read'] is False, 'CUT8 used territorial test responses')
        response_name = 'children_2024_scoring_only.csv' if scope == 'temporal' else 'children_2024_test.csv'
        require(sha(phase / 'datos/private_external' / response_name) == inputs[scope]['response_sha256'],
                'Previously scored response file identity changed')
        locks.append({'scope': scope, 'prediction_year': expected_year, 'prediction_lock_sha256': sha(lock_path)})
    return {'source_files_sha256': files, 'aggregate_schema_columns': {k: list(v) for k, v in SCHEMAS.items()},
            'aggregate_row_counts': {k: len(v) for k, v in rows.items()},
            'reference_schema_headers': reference_headers, 'external_protocol_sha256': completion['protocol_sha256'],
            'scoring_source_sha256': completion['source_sha256'], 'five_stage_receipt_sha256': stage_sha,
            'preflight_sha256': sha(finalization / 'preflight.json'),
            'original_finalization_sha256': sha(finalization / 'finalization.json'),
            'retirement_certificate_sha256': sha(retirement_path), 'prediction_locks': locks,
            'WHO2024_scoring_endpoint_syntax_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--finalization-root', type=Path, required=True)
    parser.add_argument('--operation-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--partial-backup-name', default='figuras_partial_failed_v2_20261014')
    args = parser.parse_args()
    study = args.study_root.resolve()
    require(Path(args.partial_backup_name).name == args.partial_backup_name,
            'Figure backup must be a directory basename')
    proof = validate(study, args.finalization_root.resolve(),
                     args.reference_root.resolve() if args.reference_root else None)
    phase = study / PHASE
    source = phase / 'resultados/externa_confirmatory'
    alias = phase / 'resultados/externa_final'
    partial = phase / 'figuras'
    backup = phase / args.partial_backup_name
    require(not os.path.lexists(alias), 'Publication alias already exists; preserve it')
    require(not os.path.lexists(backup), 'Partial figure backup already exists; preserve it')
    partial_files = {}
    if partial.exists():
        require(partial.is_dir() and not partial.is_symlink(), 'Partial figures must be an ordinary directory')
        for path in partial.iterdir():
            require(path.is_file() and not path.is_symlink() and path.stem in FIGURE_STEMS
                    and path.suffix in ('.png', '.svg'), 'Unexpected partial figure asset; preserve untouched')
            partial_files[path.name] = sha(path)
    operation = args.operation_root.resolve()
    operation.mkdir(parents=True, exist_ok=False)
    intent = {'status': 'PRIVATE_LAYOUT_NORMALIZATION_INTENT', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'helper_sha256': sha(Path(__file__)), 'validation': proof,
              'partial_figures_sha256': partial_files}
    (operation / 'intent.json').write_text(json.dumps(intent, indent=2) + '\n')
    if partial.exists():
        partial.rename(backup)
        require({path.name: sha(path) for path in backup.iterdir()} == partial_files,
                'Moved partial figure bytes changed')
    alias.symlink_to(source.name, target_is_directory=True)
    require(alias.resolve() == source.resolve()
            and {name: sha(alias / name) for name in ALLOWLIST} == proof['source_files_sha256'],
            'Publication alias is not identical to the completed fresh aggregates')
    result = {**intent, 'status': 'PASS_PRIVATE_PUBLICATION_LAYOUT_NORMALIZATION',
              'completed_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'classification': 'DIRECTORY_NAME_ALIAS_ONLY_AND_PARTIAL_FIGURE_BYTES_PRESERVED',
              'source_directory': 'resultados/externa_confirmatory',
              'publication_alias': 'resultados/externa_final',
              'relative_symlink_target': source.name, 'partial_backup_directory': args.partial_backup_name,
              'same_aggregate_file_bytes': True, 'old_aggregate_outputs_copied': False,
              'model_fits_or_scoring_dispatched': False, 'individual_records_or_draw_arrays_loaded': False,
              'numerical_values_or_statistical_sources_changed': False,
              'original_failed_finalization_certificates_or_logs_changed': False,
              'article_reproduction_confirmed': False, 'numerical_reproduction_confirmed': False}
    output = operation / 'completion.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'allowlisted_files': len(ALLOWLIST),
                      'preserved_partial_figure_files': len(partial_files),
                      'completion_sha256': sha(output)}, indent=2))


if __name__ == '__main__':
    main()
