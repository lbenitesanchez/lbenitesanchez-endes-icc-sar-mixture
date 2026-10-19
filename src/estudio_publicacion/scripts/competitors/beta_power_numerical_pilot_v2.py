"""Final-source numerical gates; never reads simulation truth or scores."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import beta_power_hmc as hmc
from benchmark_inputs import SCENARIOS

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'resultados/resultados_comparadores'
OUT = BASE / 'validation/beta_power_hybrid_v2'
JOBS = [('binary_informative_mar', 10012), ('sparse_srs_2025', 10000),
        ('sparse_psu_2025', 10000)]
SOURCES = ['beta_power_hmc.py', 'beta_power.py', 'beta_power_stable.py',
           'beta_power_hybrid.stan', 'beta_power_stable_functions.stan']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = {name: digest(Path(__file__).with_name(name)) for name in SOURCES}
    plan = dict(jobs=JOBS, source_sha256=source, uses_evaluation_truth=False,
                purpose='Numerical convergence validation only; all 800 original IDs retained',
                sampler_config=hmc.HMC_CONFIG)
    declared = OUT / 'declared_plan.json'
    if declared.exists() and json.loads(declared.read_text()) != json.loads(json.dumps(plan)):
        raise RuntimeError('Final-source numerical pilot plan changed')
    declared.write_text(json.dumps(plan, indent=2))
    states = []
    for scenario, rep in JOBS:
        target = OUT / scenario / f'rep_{rep:05d}'
        target.mkdir(parents=True, exist_ok=True)
        done = target / 'status.json'
        if done.exists():
            previous = json.loads(done.read_text())
            if previous['source_sha256'] != source:
                raise RuntimeError('Pilot source mismatch')
            states.append(previous)
            continue
        src = BASE / 'inputs' / scenario / f'rep_{rep:05d}' / 'model_inputs'
        direct = np.load(src / 'direct.npz')
        area = pd.read_csv(src / 'area.csv')
        X = np.c_[np.ones(len(area)), area[['X1', 'X2']].to_numpy()]
        basis = np.load(src / 'fit_basis.npy')
        sample = pd.read_csv(src / 'sample.csv')
        seed = int(np.random.SeedSequence([20261005, 8117, SCENARIOS.index(scenario), rep]).generate_state(1)[0])
        print(json.dumps(dict(event='start', scenario=scenario, rep=rep, seed=seed)), flush=True)
        tick = time.perf_counter()
        fit = hmc.fit(X, direct['y'], direct['n_eff'], direct['sample_idx'],
                      units=sample, basis=basis, seed=seed, output_dir=target / 'sampler')
        state = dict(scenario=scenario, rep=rep, seed=seed, seconds=time.perf_counter()-tick,
                     endpoints=fit['status'], source_sha256=source,
                     model_input_sha256={p.name: digest(p) for p in src.iterdir() if p.is_file()},
                     all_endpoints_converged=all(s['success'] and s['beta_shape_guard_cells']==0 for s in fit['status']),
                     uses_evaluation_truth=False)
        done.write_text(json.dumps(state, indent=2))
        states.append(state)
        (OUT / 'progress.json').write_text(json.dumps(dict(completed=len(states), total=len(JOBS), states=states), indent=2))
        print(json.dumps({k: state[k] for k in ('scenario','rep','seconds','all_endpoints_converged')}), flush=True)
    (OUT / 'completion.json').write_text(json.dumps(dict(completed=True, source_sha256=source,
       all_endpoints_converged=all(s['all_endpoints_converged'] for s in states),
       jobs=len(states), uses_evaluation_truth=False), indent=2))


if __name__ == '__main__':
    main()
