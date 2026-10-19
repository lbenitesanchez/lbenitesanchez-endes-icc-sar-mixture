"""Additive four-method external evaluation after a joint real-prediction seal.

Imports unchanged survey and inference functions from the original frozen scorer.
Both scopes and both old/new prediction files are checked before any 2024 response
value is read. The original Holm8 family and added Holm12 family remain separate.
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

import align_external_score as old

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['Selected', 'Sikov_FH', 'Sikov_SAR', 'Sikov_SAR_Published']
ENDPOINTS = old.ENDPOINTS


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def locked_hashes(lock):
    result = set()
    for key in ('files', 'predictions', 'locked_files'):
        items = lock.get(key, [])
        if isinstance(items, dict):
            result.update(items.values())
        elif isinstance(items, list):
            result.update(r.get('sha256', r.get('predictions_sha256')) for r in items)
    for key in ('input_sha256', 'file_sha256', 'output_sha256'):
        if isinstance(lock.get(key), dict):
            result.update(lock[key].values())
    return result


def gate(args):
    protocol = json.loads(args.protocol.read_text())
    addendum = json.loads(args.addendum.read_text())
    joint = json.loads(args.joint_lock.read_text())
    psha, asha = old.sha(args.protocol), old.sha(args.addendum)
    if not joint.get('approved_for_scoring', False):
        raise RuntimeError('Joint prediction lock is not approved for scoring')
    if joint.get('protocol_sha256') != psha or joint.get('addendum_sha256') != asha:
        raise RuntimeError('Joint protocol/addendum seal mismatch')
    if addendum.get('protocol_sha256') != psha:
        raise RuntimeError('Addendum does not identify the original protocol')
    if joint.get('heldout_outcomes_used', False):
        raise RuntimeError('Joint lock reports heldout outcomes used')
    if not joint.get('all_expected_worlds', False):
        raise RuntimeError('Joint lock omits complete fresh cohort confirmation')
    if joint.get('fresh_truth_read_before_joint', True) or joint.get('heldout2024responses_used_before_joint', True):
        raise RuntimeError('Fresh or external responses were read before the joint seal')
    if joint.get('fresh_cohort_size', joint.get('fresh_worlds')) != 256:
        raise RuntimeError('Joint fresh cohort must contain all 256 frozen worlds')
    for document in (protocol, addendum):
        for field in ('statistical_sources_sha256', 'additional_sources_sha256', 'frozen_inputs_sha256',
                      'runtime_binary_sha256'):
            for path, digest in document.get(field, {}).items():
                if old.sha(path) != digest:
                    raise RuntimeError('Sealed source/input changed: ' + path)
    source_hashes = (set(addendum.get('statistical_sources_sha256', {}).values()) |
                     set(addendum.get('additional_sources_sha256', {}).values()))
    if old.sha(__file__) not in source_hashes:
        raise RuntimeError('Extended scorer is absent from the frozen addendum')
    bound = locked_hashes(joint)
    for record in joint.get('files', []):
        if old.sha(record['path']) != record['sha256']:
            raise RuntimeError('Changed jointly sealed file: ' + record['path'])
    for key, filename in (('fresh_original_prediction_lock_sha256', 'fresh_prediction_lock.json'),
                          ('fresh_published_prediction_lock_sha256', 'sikov_published_fresh_prediction_lock.json')):
        if not joint.get(key) or not any(Path(r['path']).name == filename and r['sha256'] == joint[key]
                                       for r in joint.get('files', [])):
            raise RuntimeError('Joint lock omits the separately sealed fresh cohort: ' + key)
    inputs = {}
    # The original and additional files for BOTH scopes must be locked first.
    for scope in ('temporal', 'territorial'):
        original = args.original_root / ('real_' + scope)
        published = args.published_root / ('real_' + scope)
        for directory in (original, published):
            for filename in ('predictions.csv', 'completion.json'):
                expected_path = (directory / filename).resolve()
                if not any(Path(r['path']).resolve() == expected_path and
                           r['sha256'] == old.sha(expected_path) for r in joint.get('files', [])):
                    raise RuntimeError('Joint lock omits ' + str(directory / filename))
        first = json.loads((original / 'completion.json').read_text())
        second = json.loads((published / 'completion.json').read_text())
        if first['protocol_sha256'] != psha or second['protocol_sha256'] != psha:
            raise RuntimeError('Real fit protocol mismatch')
        if second['addendum_sha256'] != asha:
            raise RuntimeError('Published SAR fit addendum mismatch')
        if first.get('heldout_outcomes_read_by_fits', True) or second.get('heldout_outcomes_read_by_fits', True):
            raise RuntimeError('Real fitting used heldout outcomes')
        for method in METHODS[:3]:
            if not first['convergence'].get(method, False):
                raise RuntimeError('Original primary method did not converge: ' + method)
        if not second.get('convergence', {}).get(METHODS[3], second.get('all_converged', False)):
            raise RuntimeError('Published SAR did not converge')
        if second.get('original_predictions_sha256') != old.sha(original / 'predictions.csv'):
            raise RuntimeError('Published SAR did not preserve the original prediction seal')
        if second.get('original_completion_sha256') != old.sha(original / 'completion.json'):
            raise RuntimeError('Published SAR original completion binding mismatch')
        first_frame = old.prediction_frame(original / 'predictions.csv', METHODS[:3])
        second_frame = old.prediction_frame(published / 'predictions.csv', [METHODS[3]])
        frame = pd.concat([first_frame, second_frame], ignore_index=True)
        if frame[['ubigeo', 'method']].duplicated().any():
            raise ValueError('Repeated joined prediction key')
        inputs[scope] = {'frame': frame, 'original': original, 'published': published,
                         'completion': first, 'published_completion': second}
    return inputs, psha, asha


def score_frame(pred, target):
    support = set(pred.ubigeo)
    detail, summary, excluded = [], [], []
    for endpoint in ENDPOINTS:
        td = target.loc[target.outcome.eq(endpoint)].copy()
        supported = td.ubigeo.isin(support)
        observed = td.direct.notna() & td.weighted_observed_denominator.gt(0)
        for row in td.loc[~supported | ~observed].to_dict('records'):
            excluded.append({**row, 'reason': 'unsupported_auxiliary_geography'
                             if row['ubigeo'] not in support else 'no_valid_positive_weight_outcome'})
        f = pred.merge(td.loc[supported & observed], on='ubigeo', validate='many_to_one')
        p = f['p_' + endpoint].to_numpy(float)
        y = f.direct.to_numpy(float)
        q = np.clip(p, old.EPS, 1 - old.EPS)
        f['MSE'] = (p - y) ** 2
        f['NLL'] = -(y * np.log(q) + (1 - y) * np.log1p(-q))
        f['Brier'] = y * (1 - p) ** 2 + (1 - y) * p ** 2
        f['MSE_bounded_sensitivity'] = (np.clip(p, 0, 1) - y) ** 2
        f['outside_probability_range'] = (p < 0) | (p > 1)
        f['NLL_clipped'] = (p < old.EPS) | (p > 1 - old.EPS)
        if 'lo95_' + endpoint in f:
            if not np.isfinite(f[['lo95_' + endpoint, 'hi95_' + endpoint]].to_numpy()).all():
                raise ValueError('Missing intervals in a compared method')
            f['noisy_direct_in_interval'] = ((y >= f['lo95_' + endpoint]) &
                                             (y <= f['hi95_' + endpoint]))
            f['interval_width_pp'] = 100 * (f['hi95_' + endpoint] - f['lo95_' + endpoint])
        detail.append(f)
        for method in METHODS:
            part = f.loc[f.method.eq(method)]
            row = {'method': method, 'outcome': endpoint, 'districts': len(part),
                   'provinces': part.province.nunique(),
                   'RMSE_noisy_direct_pp': 100 * np.sqrt(part.MSE.mean()),
                   'MSE': part.MSE.mean(), 'Brier': part.Brier.mean(), 'NLL': part.NLL.mean(),
                   'RMSE_bounded_sensitivity_pp': 100 * np.sqrt(part.MSE_bounded_sensitivity.mean()),
                   'outside_probability_range_count': int(part.outside_probability_range.sum()),
                   'NLL_clipped_count': int(part.NLL_clipped.sum()), 'NLL_clipping_epsilon': old.EPS}
            if 'noisy_direct_in_interval' in part:
                row.update(noisy_direct_interval_containment_descriptive=float(part.noisy_direct_in_interval.mean()),
                           mean_interval_width_pp=float(part.interval_width_pp.mean()),
                           latent_interval_coverage_identified=False)
            summary.append(row)
    return pd.concat(detail, ignore_index=True), summary, excluded


def contrasts(detail, scope, references):
    all_codes = sorted(detail.ubigeo.unique())
    province = np.array([code[:4] for code in all_codes])
    rng = np.random.default_rng(202610102024 + (10000 if scope == 'territorial' else 0))
    records = []
    for metric in ('MSE', 'NLL'):
        candidate = detail.loc[detail.method.eq(METHODS[0])].pivot(index='ubigeo', columns='outcome', values=metric)
        candidate = candidate.reindex(index=all_codes, columns=ENDPOINTS).to_numpy(float)
        for reference in references:
            ref = detail.loc[detail.method.eq(reference)].pivot(index='ubigeo', columns='outcome', values=metric)
            ref = ref.reindex(index=all_codes, columns=ENDPOINTS).to_numpy(float)
            value = old.balanced_pair(candidate, ref, province, rng)
            records.append({'scope': scope, 'candidate': METHODS[0], 'reference': reference,
                            'metric': metric, **value})
    return records


def write_scope(path, scope, inputs, detail, summary, excluded, references, psha, asha, args):
    path.mkdir(parents=True, exist_ok=False)
    methods = [METHODS[0]] + references
    subset = detail.loc[detail.method.isin(methods)].copy()
    summaries = [{**row, 'scope': scope} for row in summary if row['method'] in methods]
    paired = contrasts(subset, scope, references)
    subset.to_csv(path / 'district_scores.csv', index=False)
    pd.DataFrame(summaries).to_csv(path / 'endpoint_summary.csv', index=False)
    pd.DataFrame(paired).to_csv(path / 'paired_balanced_contrasts_unadjusted.csv', index=False)
    pd.DataFrame(excluded).to_csv(path / 'excluded_domains.csv', index=False)
    report = {'status': 'PASS_SECONDARY_REUSED2024_SCORING', 'scope': scope,
              'candidate': METHODS[0], 'references': references,
              'scoring_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_sha256': old.sha(__file__), 'frozen_original_scorer_sha256': old.sha(old.__file__),
              'protocol_sha256': psha, 'addendum_sha256': asha,
              'joint_prediction_lock_sha256': old.sha(args.joint_lock),
              'original_predictions_sha256': old.sha(inputs['original'] / 'predictions.csv'),
              'published_predictions_sha256': old.sha(inputs['published'] / 'predictions.csv'),
              'methods_supplied_same1874domain_frame': True, 'no_favourable_prediction_intersection': True,
              'anemia_definition': 'WHO2011 HC56<110, same as2019training',
              'universe': 'de facto children6to59months; outcome-specific available-case directweightedratios',
              'balanced_endpoint_scores': '.5mean_district(anemia)+.5mean_district(stunting), preserving endpoint covariance with provincecluster sandwich/bootstrap',
              'inference': 'two-sided Studentt provincecluster sandwich; descriptive3999 pairedprovince percentilebootstrap; complete Holm family collection required',
              'prior2024_evaluation_observed_in_previous_phase': True,
              'independent_new_validation_claim_permitted': False, 'latent_prevalence_or_coverage_revealed': False,
              'raw_unbounded_gaussian_predictions_score_primary_RMSE_and_Brier': True,
              'shared_NLL_clipping_epsilon': old.EPS, 'clipping_not_hidden': True,
              'bounded_RMSE_sensitivity_separate': True, 'contrasts': paired, 'endpoint_summary': summaries,
              'excluded_domain_endpoint_count': len(excluded), 'private_unit_records_exported': False,
              'Holm8_pending_other_scope': len(references) == 2,
              'both_scopes_jointly_sealed_before_response_read': True,
              'original_candidate_reference_files_not_modified': True}
    save(path / 'summary.json', report)
    return path / 'summary.json'


def collect12(paths, output):
    reports = [json.loads(path.read_text()) for path in paths]
    if {r['scope'] for r in reports} != {'temporal', 'territorial'}:
        raise ValueError('Both scopes required')
    frame = pd.DataFrame([row for r in reports for row in r['contrasts']])
    if len(frame) != 12 or frame[['scope', 'reference', 'metric']].duplicated().any():
        raise ValueError('Complete 12-contrast family required')
    p = frame.p_two_sided.to_numpy(float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid contrast p-values')
    order = np.argsort(p, kind='stable')
    adjusted = np.empty(12)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(12, 0, -1)))
    frame['p_Holm12'] = adjusted
    frame['classification'] = np.where(adjusted >= .05, 'difference_not_detected',
                                      np.where(frame.difference < 0, 'candidate_improvement', 'candidate_deterioration'))
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / 'paired_balanced_contrasts_Holm12.csv', index=False)
    endpoint = pd.DataFrame([row for r in reports for row in r['endpoint_summary']])
    endpoint.to_csv(output / 'endpoint_summary_both_scopes.csv', index=False)
    report = {'status': 'PASS_COMPLETE_SECONDARY_HOLM12_FAMILY',
              'family': 'two scopes times FH/fullSAR/publishedSAR references times balanced MSE/NLL',
              'candidate': METHODS[0], 'references': METHODS[1:],
              'protocol_sha256': reports[0]['protocol_sha256'], 'addendum_sha256': reports[0]['addendum_sha256'],
              'classification_counts': frame.classification.value_counts().to_dict(),
              'input_sha256': {str(path): old.sha(path) for path in paths},
              'source_sha256': old.sha(__file__), 'latent_coverage_claim_permitted': False,
              'primary_scientific_success_not_decided_by_this_reused_secondary_dataset': True,
              'original_Holm8_family_reported_separately': True}
    save(output / 'external_final_summary.json', report)


def run(args):
    inputs, psha, asha = gate(args)
    if args.output.exists() and any((args.output / p).exists() for p in
                                   ('external_temporal', 'external_territorial', 'external_extended_temporal',
                                    'external_extended_territorial', 'external_comparison', 'external_comparison_Holm12')):
        raise FileExistsError('External results already exist; no overwrite allowed')
    args.output.mkdir(parents=True, exist_ok=True)
    core_paths, extended_paths = [], []
    for scope in ('temporal', 'territorial'):
        response = old.OLD / ('datos/private_external/children_2024_scoring_only.csv'
                             if scope == 'temporal' else 'datos/private_external/children_2024_test.csv')
        # First and only real response read for this scope occurs after joint gate.
        target = old.direct_measurements(response, scope)
        detail, summary, excluded = score_frame(inputs[scope]['frame'], target)
        core_paths.append(write_scope(args.output / ('external_' + scope), scope, inputs[scope],
                                      detail, summary, excluded, METHODS[1:3], psha, asha, args))
        extended_paths.append(write_scope(args.output / ('external_extended_' + scope), scope, inputs[scope],
                                          detail, summary, excluded, METHODS[1:], psha, asha, args))
    old.collect(core_paths, args.output / 'external_comparison')
    collect12(extended_paths, args.output / 'external_comparison_Holm12')
    print(json.dumps({'status': 'PASS_LOCKED_HOLM8_AND_HOLM12_EXTERNAL_EVALUATION',
                      'both_scopes_jointly_sealed_before_response_read': True,
                      'original_candidate_predictions_modified': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--addendum', type=Path, required=True)
    parser.add_argument('--joint-lock', type=Path, required=True)
    parser.add_argument('--original-root', type=Path, default=ROOT / 'evaluacion')
    parser.add_argument('--published-root', type=Path, default=ROOT / 'evaluacion/published_sar')
    parser.add_argument('--output', type=Path, default=ROOT / 'resultados')
    run(parser.parse_args())
