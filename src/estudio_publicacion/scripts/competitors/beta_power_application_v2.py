"""Complete ENDES refits with the audited evaluator and diagonal NUTS metric.

Both full and territorial-training targets are sampled afresh. No pre-repair
posterior is copied. Statistical priors and fixed province folds are unchanged.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import hashlib
import json
import sys
from pathlib import Path
import beta_power_application as application
import beta_power_hmc as hmc
from beta_power_application_recovery import DiagonalMetric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, choices=[2019, 2025], required=True)
    parser.add_argument('--tag', default='application_final_candidate_v2')
    args = parser.parse_args()
    gate = application.ROOT / 'resultados/resultados_comparadores/numerical_amendment_v2.json'
    if not json.loads(gate.read_text()).get('approved_for_dispatch', False):
        raise RuntimeError('Numerical amendment has not been approved')
    protocol = json.loads((gate.parent / 'protocol_freeze.json').read_text())
    for name in ('beta_power_hmc.py','beta_power.py','beta_power_stable.py',
                 'beta_power_hybrid.stan','beta_power_stable_functions.stan'):
        actual = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        if actual != protocol['source_sha256'][name]:
            raise RuntimeError('Frozen numerical source changed: '+name)
    model = hmc.get_model()
    hmc._MODEL = DiagonalMetric(model)
    original_fit = hmc.fit
    sampler = dict(metric='diag_e', adapt_delta=.95, retry_adapt_delta=.99,
                   initial_metric='diagonal of unrepaired positive MAP covariance; otherwise identity')

    def sample_target(*positional, **keywords):
        keywords['sampler_config'] = sampler
        return original_fit(*positional, **keywords)

    application.fit = sample_target
    sys.argv = [str(Path(application.__file__)), '--year', str(args.year), '--tag', args.tag]
    application.main()
    out = application.ROOT / 'resultados/resultados_comparadores' / args.tag / f'aplicacion_{args.year}_who2024'
    full = json.loads((out / 'fit_status.json').read_text())['endpoints']
    train = json.loads((out / 'test_fit_status.json').read_text())
    success = all(s['success'] and s['beta_shape_guard_cells']==0 for s in full+train)
    manifest_path = out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(heldout_endpoints_converged=all(s['success'] for s in train),
                    all_final_endpoints_passed=success, sampler=sampler,
                    previous_posterior_reused=False,
                    numerical_amendment_sha256=hashlib.sha256(gate.read_bytes()).hexdigest(),
                    source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ('beta_power_application_v2.py','beta_power_application.py',
                         'beta_power_hmc.py','beta_power.py','beta_power_stable.py',
                         'beta_power_hybrid.stan','beta_power_stable_functions.stan',
                         'beta_power_application_recovery.py')})
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if not success:
        raise RuntimeError('Final ENDES refit did not satisfy all numerical gates')


if __name__ == '__main__':
    main()
