"""Two sequential complete application refits on one explicitly allocated lane."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'resultados/resultados_comparadores/application_final_candidate_v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    BASE.mkdir(parents=True,exist_ok=True)
    driver = Path(__file__).with_name('beta_power_application_v2.py')
    plan = dict(years=[2019,2025], full_and_locked_refitted=True, posterior_reused=False,
                source_sha256=sha(driver), uses_heldout_scores_to_choose_sampler=False,
                workers=1, numerical_amendment_sha256=sha(BASE.parent/'numerical_amendment_v2.json'))
    declared = BASE/'queue_plan.json'
    if declared.exists() and json.loads(declared.read_text()) != plan:
        raise RuntimeError('Application queue plan changed')
    declared.write_text(json.dumps(plan,indent=2))
    budget = BASE/'runtime_budget.json'
    recorded = BASE/'queue_status.json'
    states = json.loads(recorded.read_text()) if recorded.exists() else {}
    for year in plan['years']:
        if str(year) in states:
            continue
        while not budget.exists() or json.loads(budget.read_text()).get('active_workers',0)<1:
            time.sleep(2)
        if sha(driver) != plan['source_sha256']:
            raise RuntimeError('Application driver changed after queue declaration')
        tick = time.perf_counter()
        with (BASE/f'application_{year}.log').open('w') as log:
            process = subprocess.run([sys.executable,str(driver),'--year',str(year)],
                                     cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        manifest_path = BASE/f'aplicacion_{year}_who2024/manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        states[str(year)] = dict(returncode=process.returncode, seconds=time.perf_counter()-tick,
                                all_final_endpoints_passed=bool(manifest.get('all_final_endpoints_passed')),
                                manifest_sha256=sha(manifest_path) if manifest_path.exists() else None)
        recorded.write_text(json.dumps(states,indent=2))
        print(json.dumps(dict(year=year,**states[str(year)])),flush=True)
    (BASE/'queue_completion.json').write_text(json.dumps(dict(completed=True, years=len(states),
        all_final_endpoints_passed=all(s['all_final_endpoints_passed'] for s in states.values())),indent=2))


if __name__ == '__main__':
    main()
