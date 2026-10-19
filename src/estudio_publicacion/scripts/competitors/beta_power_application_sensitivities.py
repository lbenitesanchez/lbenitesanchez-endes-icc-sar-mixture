"""Declared application sensitivities; not model selection or new populations.

The numerical diagonal metric samples the unchanged target for each declared
statistical prior/rank/information specification. Full four-chain gates apply.
The queue is idle until its runtime-only worker budget is one.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'resultados/resultados_comparadores/application_sensitivity_v2'
JOBS = [
    dict(name='historical_2019', args=['--year','2019','--rule','old','--tag','application_historical_candidate_v2','--skip-test']),
    dict(name='historical_2025', args=['--year','2025','--rule','old','--tag','application_historical_candidate_v2','--skip-test']),
    dict(name='rank20', args=['--year','2025','--tag','application_candidate_rank20_v2','--q','20','--skip-test']),
    dict(name='rank80', args=['--year','2025','--tag','application_candidate_rank80_v2','--q','80','--skip-test']),
    dict(name='spatial_sd025', args=['--year','2025','--tag','application_candidate_spatial025_v2','--spatial-sd','.25','--skip-test']),
    dict(name='spatial_sd1', args=['--year','2025','--tag','application_candidate_spatial1_v2','--spatial-sd','1','--skip-test']),
    dict(name='no_cluster', args=['--year','2025','--tag','application_candidate_no_cluster_v2','--no-cluster','--skip-test']),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_job(job):
    import beta_power_application as application
    import beta_power_hmc as hmc
    from beta_power_application_recovery import DiagonalMetric
    model = hmc.get_model()
    hmc._MODEL = DiagonalMetric(model)
    original_fit = hmc.fit
    sampler = dict(metric='diag_e', adapt_delta=.95, retry_adapt_delta=.99,
                   initial_metric='diagonal of unrepaired positive MAP covariance; otherwise identity')

    def sample_target(*args, **kwargs):
        kwargs['sampler_config'] = sampler
        return original_fit(*args, **kwargs)

    application.fit = sample_target
    sys.argv = [str(Path(application.__file__)), *job['args']]
    application.main()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', choices=[j['name'] for j in JOBS])
    parser.add_argument('--budget-file', default='application_sensitivity_budget.json')
    args = parser.parse_args()
    gate = ROOT / 'resultados/resultados_comparadores/numerical_amendment_v2.json'
    if not json.loads(gate.read_text()).get('approved_for_dispatch', False):
        raise RuntimeError('Numerical amendment has not been approved')
    if args.job:
        run_job(next(j for j in JOBS if j['name'] == args.job))
        return
    BASE.mkdir(parents=True, exist_ok=True)
    freeze = BASE / 'declared_plan.json'
    plan = dict(jobs=JOBS, source_sha256={p.name: digest(p) for p in [Path(__file__),
                    Path(__file__).with_name('beta_power_application.py'),
                    Path(__file__).with_name('beta_power_hmc.py'),
                    Path(__file__).with_name('beta_power_application_recovery.py'),
                    Path(__file__).with_name('beta_power.py'),
                    Path(__file__).with_name('beta_power_stable.py'),
                    Path(__file__).with_name('beta_power_hybrid.stan'),
                    Path(__file__).with_name('beta_power_stable_functions.stan')]},
                use='Application-only descriptive sensitivity, not primary model selection',
                additional_independent_populations=0, selection_uses_heldout_scores=False,
                original_statistical_plan='resultados/resultados_comparadores/application_sensitivity_v1/declared_plan.json',
                original_statistical_plan_sha256=digest(BASE.parent/'application_sensitivity_v1/declared_plan.json'),
                numerical_amendment_sha256=digest(gate))
    if freeze.exists() and json.loads(freeze.read_text()) != plan:
        raise RuntimeError('Application sensitivity plan or source changed')
    if not freeze.exists():
        freeze.write_text(json.dumps(plan, indent=2))
    budget = BASE / args.budget_file
    completed = BASE / 'queue_status.json'
    states = json.loads(completed.read_text()) if completed.exists() else {}
    for job in JOBS:
        if job['name'] in states:
            continue
        while not budget.exists() or int(json.loads(budget.read_text()).get('active_workers',0)) < 1:
            time.sleep(2)
        with (BASE / f"{job['name']}.log").open('w') as log:
            tick = time.perf_counter()
            result = subprocess.run([sys.executable, str(Path(__file__)), '--job', job['name']],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        tag = job['args'][job['args'].index('--tag')+1]
        year = job['args'][job['args'].index('--year')+1]
        rule = job['args'][job['args'].index('--rule')+1] if '--rule' in job['args'] else 'who2024'
        folder = BASE.parent / tag / f'aplicacion_{year}_{rule}'
        status = json.loads((folder/'fit_status.json').read_text()) if (folder/'fit_status.json').exists() else {}
        converged = bool(status.get('endpoints')) and all(s['success'] and s['beta_shape_guard_cells']==0 for s in status['endpoints'])
        states[job['name']] = dict(returncode=result.returncode, seconds=time.perf_counter()-tick,
                                  all_endpoints_converged=converged, output=str(folder.relative_to(ROOT)))
        temp = completed.with_suffix('.tmp'); temp.write_text(json.dumps(states, indent=2)); os.replace(temp, completed)
    (BASE/'completion.json').write_text(json.dumps(dict(completed=True, jobs=len(states),
             all_requested_jobs_recorded=len(states)==len(JOBS),
             all_endpoints_converged=all(v['all_endpoints_converged'] for v in states.values()),
             additional_independent_populations=0), indent=2))


if __name__ == '__main__':
    main()
