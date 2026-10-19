#!/usr/bin/env python3
"""Runtime-only queue; approved methods and independent cohorts stay fixed."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from benchmark_inputs import OUT,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--budget-file',type=Path,required=True)
    p.add_argument('--numerical-version',type=int,choices=[1,2],default=1);args=p.parse_args()
    runner=Path(__file__).with_name('benchmark_run.py')
    if args.numerical_version==2:
        amendment=json.loads((OUT/'numerical_amendment_v2.json').read_text())
        if not (amendment.get('approved_for_dispatch') and
                amendment.get('statistical_configuration_changed') is False and
                amendment.get('all_population_ids_rerun_unchanged')):
            raise RuntimeError('Numerical v2 requires an explicitly approved same-target amendment')
    jobs=[(f'final_primary_v{args.numerical_version}',['binary_informative_mar','sparse_srs_2025','sparse_psu_2025'],range(10000,10200)),
          (f'final_structural_v{args.numerical_version}',['sparse_srs_iid','sparse_psu_bym2'],range(20000,20100))]
    queue=OUT/'candidate_queue_manifest.json'
    manifest=dict(queue_source_sha256=sha256(Path(__file__)),runner_sha256=sha256(runner),
        budget_file=str(args.budget_file.resolve()),jobs=[{'tag':t,'scenarios':s,'ids':list(r)}for t,s,r in jobs],
        state='waiting_for_approved_runtime_budget',new_populations=800,parameter_tuning=False,
        numerical_version=args.numerical_version)
    queue.write_text(json.dumps(manifest,indent=2))
    while not (OUT/'input_validation_final800.json').exists() or int(json.loads(args.budget_file.read_text())['active_workers'])<1:
        time.sleep(10)
    validation=json.loads((OUT/'input_validation_final800.json').read_text())
    assert validation['passed'] and validation['independent_populations']==800
    for tag,scenarios,reps in jobs:
        completion=OUT/tag/'completion.json'
        if completion.exists():
            done=json.loads(completion.read_text())
            if not done['completed']:raise RuntimeError(f'Incomplete existing experiment {tag}')
            continue
        if (OUT/tag).exists():raise RuntimeError(f'Existing partial tag {tag}; never silently refit or overwrite')
        command=[sys.executable,str(runner),'--tag',tag,'--workers','3','--worker-limit-file',str(args.budget_file),
            '--scenarios',*scenarios,'--replications',*map(str,reps)]
        manifest.update(state='running',active_tag=tag);queue.write_text(json.dumps(manifest,indent=2))
        print(json.dumps({'dispatch':tag,'runtime_workers':json.loads(args.budget_file.read_text())['active_workers']}),flush=True)
        with (OUT/f'{tag}.log').open('w')as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            manifest.update(state='execution_failed',failed_tag=tag,exit_code=result.returncode)
            queue.write_text(json.dumps(manifest,indent=2))
            raise RuntimeError(f'{tag} failed, see its log; no seed replacement')
    manifest.update(state='completed',completed_populations=800)
    queue.write_text(json.dumps(manifest,indent=2));print(json.dumps({'completed':800}),flush=True)


if __name__=='__main__':main()
