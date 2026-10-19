"""Synthetic joint-lock, four-method, Holm8/Holm12 and independent-audit QA."""
from pathlib import Path
import argparse
import hashlib
import json
import tempfile

import numpy as np
import pandas as pd

import align_external_extended_score as scorer
import align_external_extended_audit as extended_auditor
import align_external_independent_audit as core_auditor

ROOT = Path(__file__).resolve().parents[1]


def main():
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    with tempfile.TemporaryDirectory(prefix='ENDES_extended_external_synthetic_QA_') as temporary:
        temp = Path(temporary)
        protocol = temp / 'protocol.json'
        protocol.write_text(json.dumps({'statistical_sources_sha256': {str(Path(scorer.old.__file__)): scorer.old.sha(scorer.old.__file__)}}))
        addendum = temp / 'addendum.json'
        addendum.write_text(json.dumps({'protocol_sha256': scorer.old.sha(protocol),
                                      'additional_sources_sha256': {str(Path(scorer.__file__)): scorer.old.sha(scorer.__file__),
                                                                   str(Path(extended_auditor.__file__)): scorer.old.sha(extended_auditor.__file__)}}))
        original, published = temp / 'original', temp / 'published'
        auxiliary = pd.read_csv(ROOT / 'datos/aux_sikov14_official.csv', dtype={'ubigeo': str})
        rows, predictions = [], []
        for code in auxiliary.ubigeo:
            for k in range(4):
                rows.append({'year': 2024, 'ubigeo': code, 'de_facto': 1, 'age_months': 18,
                             'weight': 1, 'y_anemia_old': int(k == 0), 'y_stunting': int(k < 2), 'psu_key': 'fake'})
            for method, p in zip(scorer.METHODS, [.28, .42, .47, .40]):
                predictions.append({'ubigeo': code, 'method': method, 'p_anemia': p, 'p_stunting': p,
                                    'lo95_anemia': p - .2, 'hi95_anemia': p + .2,
                                    'lo95_stunting': p - .2, 'hi95_stunting': p + .2})
        predictions = pd.DataFrame(predictions)
        predictions.loc[1, 'p_anemia'] = 1.25
        predictions.loc[3, 'p_stunting'] = -.1
        files = []
        for scope in ('temporal', 'territorial'):
            op, pp = original / ('real_' + scope), published / ('real_' + scope)
            op.mkdir(parents=True); pp.mkdir(parents=True)
            predictions.loc[predictions.method.isin(scorer.METHODS[:3])].to_csv(op / 'predictions.csv', index=False)
            predictions.loc[predictions.method.eq(scorer.METHODS[3])].to_csv(pp / 'predictions.csv', index=False)
            scorer.save(op / 'completion.json', {'protocol_sha256': scorer.old.sha(protocol),
                                                  'convergence': dict.fromkeys(scorer.METHODS[:3], True),
                                                  'heldout_outcomes_read_by_fits': False})
            scorer.save(pp / 'completion.json', {'protocol_sha256': scorer.old.sha(protocol),
                                                  'addendum_sha256': scorer.old.sha(addendum),
                                                  'convergence': {scorer.METHODS[3]: True},
                                                  'heldout_outcomes_read_by_fits': False,
                                                  'original_predictions_sha256': scorer.old.sha(op / 'predictions.csv'),
                                                  'original_completion_sha256': scorer.old.sha(op / 'completion.json')})
            for directory in (op, pp):
                for filename in ('predictions.csv', 'completion.json'):
                    path = directory / filename
                    files.append({'path': str(path), 'sha256': scorer.old.sha(path)})
        fresh_hashes = {}
        for key, filename in [('fresh_original_prediction_lock_sha256', 'fresh_prediction_lock.json'),
                              ('fresh_published_prediction_lock_sha256', 'sikov_published_fresh_prediction_lock.json')]:
            path = temp / filename
            scorer.save(path, {'synthetic': True, 'worlds': 256})
            fresh_hashes[key] = scorer.old.sha(path)
            files.append({'path': str(path), 'sha256': scorer.old.sha(path)})
        joint = temp / 'joint_prediction_lock.json'
        scorer.save(joint, {'approved_for_scoring': True, 'protocol_sha256': scorer.old.sha(protocol),
                           'addendum_sha256': scorer.old.sha(addendum), 'files': files,
                           'heldout_outcomes_used': False, 'all_expected_worlds': True,
                           'fresh_truth_read_before_joint': False,
                           'heldout2024responses_used_before_joint': False,
                           'fresh_cohort_size': 256, **fresh_hashes})
        args = argparse.Namespace(protocol=protocol, addendum=addendum, joint_lock=joint,
                                  original_root=original, published_root=published, output=temp / 'results')
        bad = temp / 'unapproved_joint.json'
        bad.write_text(joint.read_text().replace('"approved_for_scoring": true', '"approved_for_scoring": false'))
        args.joint_lock = bad
        reads = []
        actual_direct = scorer.old.direct_measurements

        def counted_direct(path, scope):
            reads.append(scope)
            return actual_direct(path, scope)

        scorer.old.direct_measurements = counted_direct
        try:
            scorer.run(args)
        except RuntimeError:
            check('unapproved joint seal blocked before response access', not reads)
        else:
            raise AssertionError('unapproved joint seal accepted')
        args.joint_lock = joint
        broken = dict(json.loads(joint.read_text()))
        broken['files'] = [r for r in broken['files'] if 'real_territorial/predictions.csv' not in r['path']]
        bad2 = temp / 'incomplete_joint.json'
        scorer.save(bad2, broken); args.joint_lock = bad2
        try:
            scorer.run(args)
        except RuntimeError:
            check('both scopes must be completely sealed before any response read', not reads)
        else:
            raise AssertionError('incomplete joint seal accepted')
        args.joint_lock = joint
        scorer.old.OLD = temp / 'fake_phase'
        private = scorer.old.OLD / 'datos/private_external'; private.mkdir(parents=True)
        fake_children = pd.DataFrame(rows)
        fake_children.to_csv(private / 'children_2024_scoring_only.csv', index=False)
        held = fake_children.ubigeo.str[:4].map(lambda p: int(hashlib.sha256(('ENDES-phase2-20261007|' + p).encode()).hexdigest()[:8], 16) % 5 == 0)
        fake_children.loc[held].to_csv(private / 'children_2024_test.csv', index=False)
        scorer.run(args)
        check('exactly one fake response read per scope after complete seals', reads == ['temporal', 'territorial'])
        core_auditor.audit([args.output / 'external_temporal', args.output / 'external_territorial'],
                           args.output / 'external_comparison', temp / 'core_audit.json')
        extended_auditor.audit([args.output / 'external_extended_temporal', args.output / 'external_extended_territorial'],
                               args.output / 'external_comparison_Holm12', temp / 'extended_audit.json')
        check('original Holm8 family independently passes', json.loads((temp / 'core_audit.json').read_text())['checks_passed'] >= 145)
        check('added Holm12 family independently passes', json.loads((temp / 'extended_audit.json').read_text())['checks_passed'] > 145)
        core = pd.read_csv(args.output / 'external_comparison/paired_balanced_contrasts_Holm8.csv')
        extended = pd.read_csv(args.output / 'external_comparison_Holm12/paired_balanced_contrasts_Holm12.csv')
        check('complete separate 8 and 12 contrast families', len(core) == 8 and len(extended) == 12)
        common = core.merge(extended, on=['scope', 'reference', 'metric'], suffixes=('_old', '_added'))
        check('original losses and province t inference unchanged by extra reference',
              np.allclose(common.difference_old, common.difference_added, atol=1e-14, rtol=0) and
              np.allclose(common.p_two_sided_old, common.p_two_sided_added, atol=1e-14, rtol=0))
        endpoint = pd.read_csv(args.output / 'external_comparison_Holm12/endpoint_summary_both_scopes.csv')
        check('every scope outcome retains identical support for all four methods', endpoint.groupby(['scope', 'outcome']).districts.nunique().eq(1).all())
        check('raw invalid probabilities and clipping are visible', endpoint.outside_probability_range_count.sum() >= 2 and endpoint.NLL_clipped_count.sum() >= 2)
        check('noisy direct interval diagnostics retained for every method', endpoint.noisy_direct_interval_containment_descriptive.notna().all())
        check('all original prediction bytes remain identical', all(scorer.old.sha(r['path']) == r['sha256'] for r in files))
    report = {'status': 'PASS_SYNTHETIC_ONLY_EXTENDED_EXTERNAL_QA', 'checks_passed': len(checks), 'checks': checks,
              'genuine_2024_response_files_opened': False, 'frozen_original_statistical_sources_modified': False,
              'scorer_sha256': scorer.old.sha(scorer.__file__), 'auditor_sha256': scorer.old.sha(extended_auditor.__file__),
              'test_source_sha256': scorer.old.sha(__file__)}
    scorer.save(ROOT / 'validacion/align_external_extended_synthetic_qa.json', report)
    print(json.dumps({'status': report['status'], 'checks_passed': len(checks), 'genuine_2024_response_files_opened': False}), flush=True)


if __name__ == '__main__':
    main()
