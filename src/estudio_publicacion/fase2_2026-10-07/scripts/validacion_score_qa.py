"""Independent synthetic fixtures for the frozen primary scoring arithmetic."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import t
import validacion_score_external as score


def main():
    checks = []
    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})
    provinces = np.array(['0101', '0101', '0201', '0301', '0301', '0301'])
    candidate = np.array([.12, .17, .28, .31, .32, .36])
    reference = np.array([.15, .21, .23, .35, .38, .40])
    delta = candidate - reference
    expected_mean = sum(delta) / 6
    values = np.array([sum(delta[:2]) - 2 * expected_mean, delta[2] - expected_mean,
                       sum(delta[3:]) - 3 * expected_mean])
    expected_se = np.sqrt(3 / 2 * sum(values ** 2)) / 6
    actual = score.paired_inference(candidate, reference, provinces, np.random.default_rng(741))
    check('equaldistrict_difference_with_unequal_province_sizes', abs(actual['difference'] - expected_mean) < 1e-15)
    check('manual_cluster_sandwich_SE', abs(actual['cluster_standard_error'] - expected_se) < 1e-15)
    check('finite_sample_t2_reference', abs(actual['p_two_sided'] - 2 * t.sf(abs(expected_mean / expected_se), 2)) < 1e-15)
    check('fixed3999_bootstrap_replicates', actual['bootstrap_repetitions'] == 3999)
    check('bootstrap_limits_ordered', actual['descriptive_bootstrap_ci_low'] <= actual['descriptive_bootstrap_ci_high'])
    tie = score.paired_inference(candidate, candidate, provinces, np.random.default_rng(741))
    check('exact_ties_no_spurious_advantage', tie['difference'] == 0 and tie['cluster_standard_error'] == 0 and tie['p_two_sided'] == 1)
    check('fixed_Holm8_matches_known_family', np.allclose(score.holm([.001,.01,.02,.2,.5,.8,.9,1]), [.008,.07,.12,1,1,1,1,1], atol=1e-15))
    for values in ([.001] * 7, [.1] * 9, [float('nan')] * 8, [-.1] * 8):
        rejected = False
        try:
            score.holm(values)
        except ValueError:
            rejected = True
        check('incomplete_or_invalid_Holm_family_rejected_' + str(len(checks)), rejected)
    rejected = False
    try:
        score.paired_inference(candidate, reference, np.array(['0101'] * 6), np.random.default_rng(741))
    except ValueError:
        rejected = True
    check('singleprovince_inference_rejected', rejected)
    report = {'status': 'PASS_SYNTHETIC_RESERVED_SCORER_ARITHMETIC' if all(c['passed'] for c in checks) else 'FAIL',
              'checks': checks, 'actual2024responses_read': False,
              'scorer_sha256': score.sha(Path(score.__file__))}
    dest = score.ROOT / 'validacion/externa_scorer_arithmetic_qa.json'
    dest.write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': report['status'], 'checks': len(checks)}))
    if report['status'] == 'FAIL':
        raise RuntimeError('Synthetic scoring arithmetic failed')


if __name__ == '__main__':
    main()
