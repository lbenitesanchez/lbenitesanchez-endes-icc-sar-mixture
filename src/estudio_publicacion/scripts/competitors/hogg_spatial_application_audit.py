"""Audit the four final spatial applications without reading evaluation truth.

This is a diagnostic/provenance reader. It never fits a model or computes a
predictive score. Individual records and sampler draws are not exported.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'resultados/competidores_hogg'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    while True:
        summary = read_json(RESULTS / 'spatial_age_sex_application_summary.json')
        ready = (isinstance(summary, list) and len(summary) == 2
                 and {row.get('mode') for row in summary} == {'full', 'locked_train'}
                 and all('seconds_spatial_fit' in row and not row.get('prepared_only', False) for row in summary))
        if ready:
            break
        if not args.wait:
            raise FileNotFoundError('Both final spatial applications are still pending')
        time.sleep(15)
    freeze_path = RESULTS / 'hogg_spatial_freeze.json'
    amendment_path = RESULTS / 'hogg_spatial_numerical_amendment.json'
    freeze, amendment = read_json(freeze_path), read_json(amendment_path)
    original_unchanged = all(sha(ROOT / 'scripts/competitors' / name) == expected
                             for name, expected in freeze['source_sha256'].items())
    adapter_unchanged = sha(ROOT / 'scripts/competitors/hogg_spatial_numeric.py') == amendment['adapter_sha256']
    assert original_unchanged and adapter_unchanged, 'Actual sampler sources differ from their approved records'
    endpoints = []
    applications = []
    for mode in ['full', 'locked_train']:
        folder = RESULTS / f'endes_2025_{mode}_age_sex_s2_bym2'
        source = RESULTS / f'endes_2025_{mode}_age_sex'
        manifest = read_json(folder / 'application_manifest.json')
        status = read_json(folder / 'status.json')
        assert status['completed'] and status['S1_refit'] is False and manifest['S1_refit'] is False
        assert manifest['source_age_sex_manifest_sha256'] == sha(source / 'application_manifest.json')
        assert manifest['graph_sha256'] == sha(ROOT / f'resultados/competidores_gao/inputs/endes_2025_{mode}/adjacency_edges.csv')
        assert all(max(check['recovery_summary_maxdiff'].values()) <= 1e-10
                   and check['no_S1_refit'] and check['same_subset_indices']
                   and check['source_inputs_hash_exactly_match']
                   for check in manifest['S1_recovery'].values())
        with (folder / 'predictions.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        with (source / 'predictions.csv').open(newline='') as stream:
            baseline = list(csv.DictReader(stream))
        codes = {row['ubigeo'] for row in rows}
        assert len(rows) == len(codes) == 1874 and codes == {row['ubigeo'] for row in baseline}
        boundary_counts = {}
        for endpoint in ['anemia', 'stunting']:
            columns = ['p_mean_' + endpoint, 'lo95_' + endpoint, 'hi95_' + endpoint,
                       f'conditional_probability_{endpoint}_gt_{0.3 if endpoint == "anemia" else 0.15}']
            values = [[float(row[column]) for column in columns] for row in rows]
            assert all(all(math.isfinite(value) and 0 <= value <= 1 for value in record)
                       and record[1] <= record[2] for record in values)
            boundary_counts[endpoint] = {
                'lo95_exact_zero': sum(record[1] == 0 for record in values),
                'hi95_exact_one': sum(record[2] == 1 for record in values)}
            final = status['outcomes'][endpoint]
            diagnostic = final['stage2']
            config = final['config']
            numeric_fields = [diagnostic['max_parameter_Rhat'], diagnostic['min_parameter_ESS_bulk'],
                diagnostic['min_parameter_ESS_tail'], *diagnostic['BFMI_per_chain']]
            passed = (diagnostic['all_parameter_diagnostics_finite'] is True
                and all(math.isfinite(value) for value in numeric_fields)
                and diagnostic['max_parameter_Rhat'] <= 1.01
                and diagnostic['min_parameter_ESS_bulk'] >= 400
                and diagnostic['min_parameter_ESS_tail'] >= 400
                and len(diagnostic['BFMI_per_chain']) == 4 and min(diagnostic['BFMI_per_chain']) >= .3
                and diagnostic['divergences'] == diagnostic['max_treedepth_hits'] == 0)
            assert final['success'] == diagnostic['success'] == passed
            assert config['chains'] == 4 and config['parallel_chains'] == 1
            assert diagnostic['sampled_parameter_count'] == 1884
            assert diagnostic['draws'] == 4 * config['samples']
            graph = final['graph']
            assert graph['M'] == 1874 and graph['rank'] == 1872 and graph['components'] == 2
            assert graph['all_positive_modes_retained'] is True and graph['low_rank_approximation'] is False
            attempts = status['all_attempts'][endpoint]
            initial_path = folder / endpoint / f'attempt_{len(attempts) - 1}/stage2/numerical_initialization_summary.json'
            initialization = read_json(initial_path)
            assert initialization['adapter_source_sha256'] == amendment['adapter_sha256']
            endpoints.append({'mode': mode, 'endpoint': endpoint, 'all_gates_passed': passed,
                'attempts_retained': len(attempts), 'final_diagnostics': diagnostic,
                'sampler_config': config, 'seconds_final_attempt': final['seconds_total'],
                'actual_numerical_initialization_sha256': sha(initial_path),
                'status_sha256': sha(folder / 'status.json')})
        applications.append({'mode': mode, 'districts': 1874, 'success': status['success'],
            'S1_refit': False, 'prediction_path': str((folder / 'predictions.csv').relative_to(ROOT)),
            'prediction_sha256': sha(folder / 'predictions.csv'),
            'manifest_sha256': sha(folder / 'application_manifest.json'),
            'seconds_spatial_pair': manifest['seconds_spatial_fit'],
            'saved_interval_numeric_boundary_counts': boundary_counts})
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'audit_source_sha256': sha(__file__),
        'completed_application_pairs': 2, 'completed_endpoints': 4,
        'all_four_endpoints_passed': all(record['all_gates_passed'] for record in endpoints),
        'original_nine_sources_unchanged': original_unchanged, 'approved_adapter_unchanged': adapter_unchanged,
        'original_spatial_freeze_sha256': sha(freeze_path), 'numerical_amendment_sha256': sha(amendment_path),
        'applications': applications, 'endpoints': endpoints,
        'scope': 'Separate higher-information S1 age/sex plus full-rank S2 BYM2 sensitivity; no full Australian application or benchmarking replication.',
        'uncertainty': 'Gaussian modular model posterior; S2 spatial borrowing retains native S1 iid/SRS Psi/GVF. No uniform design-coverage, joint disease-posterior, response-IPW correction or operational district-ranking claim.',
        'prior_transfer': 'rho Beta(3.05,1.65) was informed by the Australian application and transferred as a declared sensitivity, not calibrated to Peru.',
        'graph': 'Symmetric coordinate8NN graph; componentwise exact Qplus geometric scaling and sum-zero constraints, all1872 positive modes retained.',
        'data_access': 'Status/provenance and district prediction artifacts only; no child rows, truth, raw posterior arrays or predictive scores.'}
    path = RESULTS / 'spatial_application_final_diagnostics.json'
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({'path': str(path.relative_to(ROOT)), 'all_four_endpoints_passed': report['all_four_endpoints_passed'],
        'completed_endpoints': 4, 'sha256': sha(path)}), flush=True)


if __name__ == '__main__':
    main()
