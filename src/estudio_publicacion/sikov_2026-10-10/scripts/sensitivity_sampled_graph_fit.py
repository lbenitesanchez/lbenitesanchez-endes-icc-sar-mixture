"""Fit all eleven frozen methods on the separate 64-world sensitivity.

Only generated training arrays are read. Initially at most two worlds run;
four are permitted only after the 256 original worlds close. This operational
change does not alter four-chain NUTS, retries, or native B100 bootstrap.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse, concurrent.futures, datetime as dt, json, time, traceback
import numpy as np
import pandas as pd
from comparison_evaluate import fits
from sikov_published_fit import fit_arrays
from sensitivity_sampled_graph_generate import (ROOT, DEST, CONTEXT, IDS,
    METHODS, generate, gate, sha, save)


def work(rep):
    plan, plansha, protocol = gate()
    source = DEST / 'inputs' / f'rep_{rep:05d}'
    if not source.exists():
        generate(rep)
    manifest = json.loads((source / 'manifest.json').read_text())
    for filename, key in (('inputs.npz', 'fit_inputs_sha256'),
                          ('sample.csv', 'sample_sha256')):
        if sha(source / 'model_inputs' / filename) != manifest[key]:
            raise RuntimeError('Changed shared generated training input')
    if manifest['sensitivity_plan_sha256'] != plansha or \
            not manifest['truth_excluded_from_model_inputs']:
        raise RuntimeError('Wrong generated input role or prospective seal')
    d = dict(np.load(source / 'model_inputs/inputs.npz', allow_pickle=False))
    units = pd.read_csv(source / 'model_inputs/sample.csv')
    out = DEST / 'fits' / f'rep_{rep:05d}'
    if out.exists():
        raise FileExistsError('Preserve incomplete numerical fit ' + str(out))
    seed = int(np.random.SeedSequence([20261010, 973, rep]).generate_state(1)[0]) \
        % (2 ** 32 - 200000)
    pubseed = int(np.random.SeedSequence([20261010, 974, rep]).generate_state(1)[0])
    before = time.perf_counter()
    # The native wrapper fits FH, full-N SAR and graph variants, ICC and the
    # unchanged 75/25 full-N mixture, retaining the original numerical gates.
    status = fits(d, units, out / 'original', seed,
                 protocol['selected_recipe'], sensitivity=True)
    _, _, published = fit_arrays(d, out / 'published', pubseed)
    convergence = {key: value['converged'] for key, value in status.items()}
    convergence.update({name: published['converged'] for name in
                        ('Sikov_SAR_Published', 'Sikov_SAR_Published_BC')})
    if set(convergence) != set(METHODS):
        raise RuntimeError('Incomplete eleven-method universe')
    gate()
    record = {'context': CONTEXT, 'replication': rep,
        'protocol_sha256': plan['protocol_sha256'],
        'addendum_sha256': plan['addendum_sha256'], 'sensitivity_plan_sha256': plansha,
        'fit_inputs_sha256': manifest['fit_inputs_sha256'],
        'sample_sha256': manifest['sample_sha256'],
        'input_manifest_sha256': sha(source / 'manifest.json'),
        'truth_read_by_fits': False, 'selected_recipe': protocol['selected_recipe'],
        'original_fit_seed_stream': [20261010, 973, rep],
        'original_fit_seed': seed, 'published_bootstrap_seed_stream':
            [20261010, 974, rep], 'published_bootstrap_seed': pubseed,
        'seconds': time.perf_counter() - before, 'convergence': convergence,
        'output_sha256': {str(p.relative_to(out)): sha(p)
            for prefix in ('original', 'published')
            for p in (out / prefix).iterdir() if p.is_file()}}
    save(out / 'completion.json', record)
    return record


def original_cohort_closed():
    plan, _, protocol = gate()
    for context in protocol['contexts']:
        for rep in protocol['evaluation_ids']:
            path = ROOT / 'evaluacion/fits' / context / f'rep_{rep:05d}/completion.json'
            if not path.is_file():
                return False
            record = json.loads(path.read_text())
            if record['protocol_sha256'] != plan['protocol_sha256'] or \
                    record['truth_read_by_fits'] is not False or \
                    set(record['convergence']) != set(METHODS[:9]):
                raise RuntimeError('Original completion invalid; do not expand worker budget')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wait-score', action='store_true')
    parser.add_argument('--expand-after-originals', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('Sensitivity worker limit is one to four')
    if args.workers > 2 and not original_cohort_closed():
        raise RuntimeError('More than two active worlds requires all original 256 closures')
    gate()
    jobs = list(IDS)
    if args.resume:
        jobs = [rep for rep in jobs if not
                (DEST / 'fits' / f'rep_{rep:05d}/completion.json').exists()]
    DEST.mkdir(parents=True, exist_ok=True)
    events = DEST / 'dispatch_events.jsonl'
    errors, done = [], 0
    active_budget = args.workers
    max_budget = 4 if args.expand_after_originals else args.workers
    pending = iter(jobs)
    exhausted = False
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_budget) as pool:
        futures = {}
        while futures or not exhausted:
            if args.expand_after_originals and active_budget < 4 and original_cohort_closed():
                active_budget = 4
                print(json.dumps({'event': 'operational_worker_expansion',
                    'active_world_budget': 4, 'original_256_closed': True,
                    'accuracy_or_truth_read': False}), flush=True)
            while not exhausted and len(futures) < active_budget:
                rep = next(pending, None)
                if rep is None:
                    exhausted = True
                else:
                    futures[pool.submit(work, rep)] = rep
            if not futures:
                continue
            finished, _ = concurrent.futures.wait(futures, timeout=15,
                return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                rep = futures.pop(future)
                try:
                    result = future.result(); done += 1
                    event = {'time_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                        'context': CONTEXT, 'replication': rep, 'completed': done,
                        'requested': len(jobs), 'seconds': round(result['seconds'], 2),
                        'all_converged': all(result['convergence'].values())}
                except Exception as exc:
                    event = {'time_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                        'context': CONTEXT, 'replication': rep, 'error': repr(exc),
                        'traceback': traceback.format_exc()}
                    errors.append(event)
                with events.open('a') as stream:
                    stream.write(json.dumps(event) + '\n')
                print(json.dumps(event), flush=True)
    save(DEST / 'dispatch_status.json', {'requested': len(jobs), 'completed': done,
        'errors': errors, 'sensitivity_plan_sha256': sha(ROOT /
            'validacion/sensitivity_sampled_graph_freeze.json'), 'truth_read': False})
    if errors:
        raise RuntimeError('Incomplete sensitivity; failures retained, scoring blocked')
    from sensitivity_sampled_graph_audit import audit_predictions
    audit_predictions()
    if args.wait_score:
        from sensitivity_sampled_graph_score import main as score_main
        score_main(['--wait'])


if __name__ == '__main__':
    main()
