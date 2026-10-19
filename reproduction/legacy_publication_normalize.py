#!/usr/bin/env python3
"""Expose completed fresh external aggregates to the frozen publication readers.

Run after the five legacy stages, before the first finalization. No previous
finalization, failed run, relay, retired controller, or historical fits are
required. Only an exact relative private-directory alias is created. The helper
does not fit, score, alter figures, change numbers, or certify article agreement.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import itertools
import json
import os
from pathlib import Path

import legacy_runner
import legacy_v2_runner
from legacy_publication_layout import (ALLOWLIST, ENDPOINTS, METHODS, PHASE,
                                       SCHEMAS, SCOPES, STAGES, csv_rows, read,
                                       require, sha)

HERE = Path(__file__).resolve().parent
IDENTITY_STATUS = 'PASS_800_OBSERVED_INPUTS_EXACT_FIT_GEOMETRY_ROUNDOFF'


def validate_identity(run, catalogue=None):
    identity_path = run / 'legacy_v2_input_identity.json'
    identity = read(identity_path)
    require(identity.get('status') == IDENTITY_STATUS
            and identity.get('worlds') == 800 and identity.get('csv_hashes_checked') == 4000
            and identity.get('failures') == [], 'Complete exact 800-world observed-input identity is required')
    rule_sha = sha(HERE / 'legacy_orientation_rule.json')
    require(identity.get('orientation_rule_sha256') == rule_sha, 'Input identity orientation rule changed')
    contract = read(run / 'legacy_v2_generation_contract.json')
    require(contract.get('classification') == 'INPUT_GENERATION_CORRECTION'
            and contract.get('operational_helper_sha256') == legacy_v2_runner.hashes()
            and contract.get('archive_generating_geometry_truth_response_or_fit_outputs_loaded_by_generation') is False,
            'Fresh generation source/helper isolation contract differs')
    generation_path = run / 'study/resultados/resultados_comparadores/input_validation_final800.json'
    generation = read(generation_path)
    require(generation.get('passed') is True and generation.get('independent_populations') == 800
            and generation.get('orientation_rule_sha256') == rule_sha
            and generation.get('dispatcher_sha256') == sha(HERE / 'legacy_v2_worker.py')
            and generation.get('source_models_population_seeds_and_diagnostic_gates_unchanged') is True
            and generation.get('historical_responses_truth_geometry_arrays_or_fits_loaded_in_production') is False,
            'Fresh generation certificate is incomplete or changed')
    setup_path = run / 'legacy_initial_setup_completion.json'
    setup = read(setup_path)
    require(setup.get('complete') is True and setup.get('800_observed_input_identity_confirmed') is True
            and setup.get('input_identity_certificate_sha256') == sha(identity_path),
            'Initial fit authorization does not bind this exact input identity')
    stages = {}
    for stage in STAGES:
        path = run / ('legacy_' + stage + '_execution.json')
        record = read(path)
        require(record.get('status') == 'ALL_COMMANDS_EXITED_ZERO'
                and record.get('stage') == stage
                and record.get('input_identity_certificate_sha256') == sha(identity_path),
                'Legacy stage is incomplete or binds other inputs: ' + stage)
        stages[stage] = sha(path)
    result = {'input_identity_sha256': sha(identity_path), 'fresh_generation_sha256': sha(generation_path),
              'generation_contract_sha256': sha(run / 'legacy_v2_generation_contract.json'),
              'initial_fit_authorization_sha256': sha(setup_path), 'five_stage_receipt_sha256': stages}
    if catalogue:
        strict_path = run / 'legacy_public_strict_input_identity.json'
        strict = read(strict_path)
        require(strict.get('status') == IDENTITY_STATUS and strict.get('worlds') == 800
                and strict.get('csv_hashes_checked') == 4000 and strict.get('failures') == []
                and strict.get('expected_catalogue_sha256') == sha(catalogue)
                and strict.get('strict_public_catalogue_gate') is True
                and strict.get('all_direct_npz_keys_checked_against_constant_allowlist_before_array_loading') is True
                and strict.get('all_800_orientation_generation_receipts_checked') is True
                and strict.get('fresh_input_generation_certificate_sha256') == sha(generation_path),
                'Strict public catalogue identity gate is absent or changed')
        result.update(input_catalogue_sha256=sha(catalogue), strict_public_identity_sha256=sha(strict_path))
    return result


def validate_aggregates(source):
    require(set(path.name for path in source.iterdir()) == set(ALLOWLIST)
            and all((source / name).is_file() and not (source / name).is_symlink()
                    for name in ALLOWLIST), 'Fresh aggregate directory differs from the exact allowlist')
    completion = read(source / 'completion.json')
    require(completion.get('status') == 'COMPLETE_RESERVED_EXTERNAL_VALIDATION', 'External scoring is incomplete')
    files = {name: sha(source / name) for name in ALLOWLIST}
    require(set(completion['output_sha256']) == set(ALLOWLIST) - {'completion.json'},
            'Scoring output hash inventory differs')
    for name, value in completion['output_sha256'].items():
        require(files[name] == value, 'Fresh scoring output changed: ' + name)
    rows = {name: csv_rows(source / name, SCHEMAS[name]) for name in SCHEMAS}
    expected_summary = set(itertools.product(SCOPES, ENDPOINTS, METHODS))
    keys = [(row['scope'], row['outcome'], row['method']) for row in rows['summary.csv']]
    require(len(keys) == 12 and set(keys) == expected_summary, 'Expected 12 summary groups differ')
    expected_contrasts = {(scope, endpoint, 'BetaDesignUncertainty', reference)
                         for scope, endpoint, reference in itertools.product(SCOPES, ENDPOINTS, METHODS[:2])}
    keys = [(row['scope'], row['outcome'], row['candidate'], row['reference'])
            for row in rows['contrasts_Holm8.csv']]
    require(len(keys) == 8 and set(keys) == expected_contrasts, 'Expected unique Holm8 contrasts differ')
    groups = {}
    for row in rows['district_scores.csv']:
        key = (row['scope'], row['outcome'], row['method'])
        require(key in expected_summary and len(row['ubigeo']) == 6 and row['ubigeo'].isdigit()
                and row['province'] == row['ubigeo'][:4], 'District aggregate scope or area key differs')
        groups.setdefault(key, []).append(row['ubigeo'])
    for row in rows['summary.csv']:
        codes = groups.get((row['scope'], row['outcome'], row['method']), [])
        require(len(codes) == len(set(codes)) == int(row['districts'])
                and len({code[:4] for code in codes}) == int(row['provinces']),
                'District aggregate support differs from its fresh summary')
    return completion, {'source_files_sha256': files,
                        'aggregate_schema_columns': {k: list(v) for k, v in SCHEMAS.items()},
                        'aggregate_row_counts': {k: len(v) for k, v in rows.items()}}


def validate(study, catalogue=None):
    require(study.name == 'study', 'Use the private runner study directory')
    run, phase = study.parent, study / PHASE
    proof = validate_identity(run, catalogue)
    checked = legacy_runner.check_sources(study)
    preparation = read(run / 'legacy_preparation.json')
    for name, value in preparation['source_sha256'].items():
        require(sha(study / name) == value, 'Prepared frozen source changed: ' + name)
    source = phase / 'resultados/externa_confirmatory'
    completion, aggregate_proof = validate_aggregates(source)
    script = phase / 'scripts/validacion_score_external.py'
    require(sha(script) == completion['source_sha256']
            == preparation['source_sha256'][PHASE + '/scripts/validacion_score_external.py'],
            'Frozen scoring source identity changed')
    require(sha(phase / 'protocol_freeze.json') == completion['protocol_sha256'], 'External protocol changed')
    expected_pair = [('anemia', 'y_anemia_who2024'), ('stunting', 'y_stunting')]
    found = False
    for node in ast.walk(ast.parse(script.read_text())):
        if isinstance(node, ast.For) and isinstance(node.iter, (ast.List, ast.Tuple)):
            try:
                found = found or ast.literal_eval(node.iter) == expected_pair
            except (ValueError, TypeError):
                pass
    require(found, 'Frozen scorer does not declare WHO2024 anaemia')
    import_proof = read(phase / 'validacion/externa_first_reserved_open.json')
    require(import_proof.get('WHO2024_endpoint_preserved') is True, 'Fresh WHO2024 import declaration is missing')
    inputs = {row['scope']: row for row in completion['inputs']}
    require(len(completion['inputs']) == 2 and set(inputs) == set(SCOPES), 'External input scopes differ')
    locks = []
    for scope in SCOPES:
        path = phase / 'validacion' / ('externa_prediction_lock_' + scope + '.json')
        lock = read(path)
        require(sha(path) == inputs[scope]['prediction_lock_sha256']
                and lock['predictions'] == inputs[scope]['prediction_inputs']
                and lock['approved_for_scoring'] and not lock['reserved_territorial_test_outcomes_used_in_training']
                and lock['protocol_sha256'] == completion['protocol_sha256'], 'Completed scoring lock changed: ' + scope)
        require(len(lock['predictions']) == 3
                and {row['method'] for row in lock['predictions']} == set(METHODS), 'Locked methods differ')
        year = 2019 if scope == 'temporal' else 2024
        prefix = 'externa_temporal_2019' if scope == 'temporal' else 'externa_territorial_2024'
        for prediction in lock['predictions']:
            suffix = {'BetaPowerExisting': 'baseline', 'HoggAgeSexSpatial': 'hogg',
                      'BetaDesignUncertainty': 'designcut8'}[prediction['method']]
            expected_directory = 'resultados/' + prefix + '_' + suffix + '/'
            expected_prediction = ('district_predictions.csv'
                                   if scope == 'temporal' and suffix == 'designcut8' else 'predictions.csv')
            expected_status = 'completion.json' if suffix == 'designcut8' else 'status.json'
            require(prediction['path'] == expected_directory + expected_prediction
                    and prediction['status_path'] == expected_directory + expected_status,
                    'Locked files differ from the canonical fresh method/year paths')
            require(sha(phase / prediction['path']) == prediction['sha256']
                    and sha(phase / prediction['status_path']) == prediction['status_sha256'], 'Locked prediction/status changed')
            status = read(phase / prediction['status_path'])
            require(status.get('success') is True and status.get('year') == year, 'Prediction year/success differs')
            if prediction['method'] != 'BetaDesignUncertainty':
                require(status.get('reserved_test_responses_read') is False
                        and status.get('historical_sources_modified') is False, 'Fitting scope/source declaration differs')
            elif scope == 'territorial':
                require(status.get('reserved_test_outcomes_read') is False, 'CUT8 used territorial test responses')
        response = 'children_2024_scoring_only.csv' if scope == 'temporal' else 'children_2024_test.csv'
        require(sha(phase / 'datos/private_external' / response) == inputs[scope]['response_sha256'],
                'Previously scored response-file identity changed')
        locks.append({'scope': scope, 'prediction_year': year, 'prediction_lock_sha256': sha(path),
                      'locked_utc': lock['locked_utc']})
    require(locks[0]['locked_utc'] < import_proof['time_utc']
            and locks[1]['locked_utc'] < completion['completed_utc'], 'Prediction lock/import/scoring order differs')
    return {**proof, **aggregate_proof, 'frozen_statistical_sources_checked': len(checked),
            'WHO2024_scoring_endpoint_syntax_verified': True, 'prediction_locks': locks,
            'external_protocol_sha256': completion['protocol_sha256'], 'scoring_source_sha256': completion['source_sha256']}


def normalize(study, output=None, catalogue=None, check_only=False):
    require(not (check_only and output is not None), '--check-only is write-free; omit --output')
    study = study.resolve()
    repository = HERE.parent
    require(not study.is_relative_to(repository), 'Private study must remain outside the public checkout')
    if output:
        output = output.resolve()
        require(not output.is_relative_to(repository) and not os.path.lexists(output),
                'Private certificate must be new and outside the public checkout')
    require(check_only or output is not None, 'Mutation requires a new private --output certificate')
    proof = validate(study, catalogue.resolve() if catalogue else None)
    source = study / PHASE / 'resultados/externa_confirmatory'
    alias = source.with_name('externa_final')
    existed = os.path.lexists(alias)
    if existed:
        require(alias.is_symlink() and os.readlink(alias) == source.name
                and alias.resolve() == source.resolve(), 'Existing publication path is not the exact fresh relative alias')
    created = False
    if not check_only and not existed:
        alias.symlink_to(source.name, target_is_directory=True)
        created = True
    if created or existed:
        require({name: sha(alias / name) for name in ALLOWLIST} == proof['source_files_sha256'], 'Alias bytes differ')
    result = {'format': 'iaos-portable-publication-layout-normalization-v1',
              'status': 'PASS_CHECK_ONLY_NO_WRITES' if check_only else 'PASS_PRIVATE_PUBLICATION_ALIAS_VERIFIED',
              'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'validation': proof, 'helper_sha256': sha(Path(__file__)),
              'pure_validation_dependency_sha256': sha(HERE / 'legacy_publication_layout.py'),
              'publication_alias': 'resultados/externa_final', 'relative_symlink_target': source.name,
              'check_only': check_only, 'alias_created': created, 'alias_already_present': existed,
              'requires_previous_finalization_or_relay_or_retirement': False,
              'figures_or_existing_certificates_modified': False, 'model_fits_or_scoring_dispatched': False,
              'individual_records_or_draw_arrays_loaded': False,
              'numerical_values_or_statistical_sources_changed': False,
              'article_reproduction_confirmed': False, 'numerical_reproduction_confirmed': False}
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as handle:
            json.dump(result, handle, indent=2)
            handle.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--input-catalogue', type=Path)
    parser.add_argument('--output', type=Path, help='New private certificate; required for normalization, forbidden with --check-only')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    result = normalize(args.study_root, args.output, args.input_catalogue, args.check_only)
    print(json.dumps({'status': result['status'], 'alias_created': result['alias_created'],
                      'five_stages': len(result['validation']['five_stage_receipt_sha256']),
                      'aggregate_files': len(ALLOWLIST), 'article_reproduction_confirmed': False}, indent=2))


if __name__ == '__main__':
    main()
