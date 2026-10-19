"""Immutable Hogg benchmark runner; every requested seed remains a denominator.

Task CSV columns: scenario,replication,arm. Arms are primary,finite_plusH,
design_PSU. The runner never reads evaluation_only or population outcomes.
Python/Stan/executables are snapshotted before the worker pool starts.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from datetime import datetime,timezone

for variable in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[variable]='1'
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
_WORKER={}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))


def snapshot_sources(destination):
    # Source compilation happens in one parent process before immutable copying.
    sys.path.insert(0,str(HERE))
    import hogg_tsln as h
    import hogg_design_extension as ex
    h.compile_models();ex.compile_extension()
    names=['hogg_tsln.py','hogg_diagnostics.py','hogg_design_extension.py',
           'hogg_stage1.stan','hogg_stage2.stan','hogg_stage2_design_extension.stan',
           'hogg_stage1','hogg_stage2','hogg_stage2_design_extension',
           'hogg_design.md','hogg_strong_design.md','hogg_benchmark_batch.py']
    destination.mkdir(parents=True,exist_ok=False)
    for name in names:shutil.copy2(HERE/name,destination/name)
    frozen={'source_sha256':{n:sha(destination/n) for n in names},
        'core_scientific_frozen_records':{p.name:sha(p) for p in [
            ROOT/'resultados/competidores_hogg/hogg_sampler_freeze.json',
            ROOT/'resultados/competidores_hogg/hogg_strong_arms_freeze.json']},
        'utc':datetime.now(timezone.utc).isoformat(),
        'implementation':'Independent implementation; no author repository or unlicensed source copied.',
        'backend':'CmdStan2.40.0 exactNUTS; unchanged modularGaussianS2 integrated analytically.'}
    write_json(destination/'snapshot_manifest.json',frozen)
    return frozen


def initialize_worker(source_dir):
    source=Path(source_dir)
    sys.path.insert(0,str(source))
    # Spawn imports this runner, not the parent process module cache. Loading
    # the snapshot first keeps any subsequent source edit out of active fits.
    import hogg_tsln as h
    import hogg_diagnostics as hd
    import hogg_design_extension as ex
    h.ROOT=ROOT  # Runtime location only; code, priors and likelihood are fixed.
    h.CMDSTAN=ROOT/'.competitor_env/cmdstan/cmdstan-2.40.0'
    backend=h._backend()
    models={name:backend.CmdStanModel(stan_file=str(source/(name+'.stan')),
                                    exe_file=str(source/name))
            for name in ['hogg_stage1','hogg_stage2']}
    extension=backend.CmdStanModel(stan_file=str(source/'hogg_stage2_design_extension.stan'),
                                  exe_file=str(source/'hogg_stage2_design_extension'))
    _WORKER.update(h=h,hd=hd,ex=ex,models=models,extension=extension)


def fit_task(task):
    h=_WORKER['h'];hd=_WORKER['hd'];ex=_WORKER['ex']
    output=Path(task['output']);output.mkdir(parents=True,exist_ok=True)
    inputs=ROOT/'resultados/resultados_comparadores/inputs'/task['scenario']/f'rep_{task["replication"]:05d}'/'model_inputs'
    source=Path(h.__file__).parent
    started=time.monotonic()
    run_metadata={**task,'input_directory':str(inputs),
        'source_snapshot_manifest_sha256':sha(source/'snapshot_manifest.json'),
        'source_python_sha256':sha(source/'hogg_tsln.py'),
        'evaluation_truth_access':'None; fit only reads model_inputs/sample,area,direct'}
    try:
        for name in ['sample.csv','area.csv','direct.npz']:
            if not (inputs/name).is_file():raise FileNotFoundError(str(inputs/name))
        run_metadata['input_sha256']={n:sha(inputs/n)for n in ['sample.csv','area.csv','direct.npz']}
        # Same endpoint seeds and same requested population across methods.
        cfg=h.MCMCConfig(parallel_chains=1,phase='final_independent_benchmark',
                         seed=20261005+int(task['replication']))
        if task['arm']=='design_PSU':
            result=ex.fit_design_pair(inputs/'sample.csv',inputs/'area.csv',inputs/'direct.npz',
                output_dir=output,config=cfg,primary_models=_WORKER['models'],
                extension_model=_WORKER['extension'],retry_on_diagnostics=True,
                retain_sampler_csv=False)
        else:
            features=['H'] if task['arm']=='finite_plusH' else []
            result=h.fit_hogg_pair(inputs/'sample.csv',inputs/'area.csv',
                area_features=['X1','X2'],unit_features=features,population_N='N_population',
                output_dir=output,config=cfg,models=_WORKER['models'],
                retry_on_diagnostics=True,retain_sampler_csv=False)
            hd.annotate_pair(result,inputs/'sample.csv',inputs/'area.csv',
                area_features=['X1','X2'],unit_features=features,population_N='N_population')
        # Preserve the exact area-level Gaussian S1 modular input so later
        # spatial sensitivities can reuse S1 without refitting or approximating
        # it. These are simulation summaries, not unit records or truth.
        for endpoint,fit in result['results'].items():
            np.savez_compressed(output/f'stage2_modular_summary_{endpoint}.npz',
                                area=fit.areas,**fit.stage2_data)
        run_metadata.update(completed=True,success=bool(result['status']['success']),
                            seconds=time.monotonic()-started,
                            diagnostic_retry_count=sum(len(a)-1 for a in result['status']['all_attempts'].values()))
    except Exception as error:
        # Never replace a difficult population with a new seed or silently
        # discard its place in the requested experiment.
        run_metadata.update(completed=False,success=False,seconds=time.monotonic()-started,
                            error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        write_json(output/'execution_failure.json',run_metadata)
    write_json(output/'task_execution.json',run_metadata)
    return run_metadata


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tasks',type=Path,required=True)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--source-snapshot',type=Path,help='Reuse an existing immutable Hogg source snapshot for disjoint shards')
    parser.add_argument('--protocol-file',type=Path,default=ROOT/'resultados/resultados_comparadores/protocol_freeze.json')
    parser.add_argument('--allow-development-smoke',action='store_true')
    args=parser.parse_args()
    tasks=pd.read_csv(args.tasks)
    required={'scenario','replication','arm'}
    if not required.issubset(tasks):raise ValueError('Task CSV requires scenario,replication,arm')
    if tasks[list(required)].duplicated().any():raise ValueError('Duplicate requested task')
    if not tasks.arm.isin(['primary','finite_plusH','design_PSU']).all():raise ValueError('Unknown arm')
    if not args.allow_development_smoke and (tasks.replication<10000).any():
        raise ValueError('Development populations cannot enter final batch')
    protocol=None
    if not args.allow_development_smoke:
        if not args.protocol_file.exists():raise FileNotFoundError('Approved independent protocol required before final fits')
        protocol=json.loads(args.protocol_file.read_text())
        if not protocol.get('approved_for_independent_evaluation'):
            raise ValueError('Independent protocol is not approved')
        for row in tasks.to_dict('records'):
            if int(row['replication']) not in protocol.get('independent_seed_ids',{}).get(row['scenario'],[]):
                raise ValueError('Requested population is outside the approved independent protocol')
            if row['arm']!='primary':
                label={'finite_plusH':'HoggTSLNFullH','design_PSU':'HoggDesignPsiPSU'}[row['arm']]
                allowed=protocol.get('paired_additional_arms',{}).get(label,{})
                if row['scenario']!=allowed.get('scenario') or int(row['replication'])not in allowed.get('seed_ids',[]):
                    raise ValueError('Sensitivity arm is outside the approved paired subset')
        for key,name in [('hogg','hogg_sampler_freeze.json'),('hogg_strong','hogg_strong_arms_freeze.json')]:
            expected=protocol.get('competitor_freeze_manifest_sha256',{}).get(key)
            if expected is None or sha(ROOT/'resultados/competidores_hogg'/name)!=expected:
                raise ValueError('Scientific Hogg freeze manifest differs from the approved protocol')
    # Verify every requested input exists before starting any final posterior.
    for row in tasks.to_dict('records'):
        input_dir=ROOT/'resultados/resultados_comparadores/inputs'/row['scenario']/f'rep_{int(row["replication"]):05d}'/'model_inputs'
        if not all((input_dir/n).is_file()for n in ['sample.csv','area.csv','direct.npz']):
            raise FileNotFoundError(f'Incomplete requested input {input_dir}')
    batch=ROOT/'resultados/competidores_hogg'/args.tag
    if batch.exists():raise FileExistsError('Use a new immutable batch tag')
    batch.mkdir(parents=True)
    shutil.copy2(args.tasks,batch/'requested_tasks.csv')
    if protocol is not None:shutil.copy2(args.protocol_file,batch/'independent_protocol_snapshot.json')
    if args.source_snapshot is None:
        source_dir=batch/'source_snapshot'
        frozen=snapshot_sources(source_dir)
    else:
        source_dir=args.source_snapshot.resolve()
        frozen=json.loads((source_dir/'snapshot_manifest.json').read_text())
        for name,expected in frozen['source_sha256'].items():
            if sha(source_dir/name)!=expected:raise ValueError('Immutable source snapshot hash changed')
        write_json(batch/'reused_source_snapshot.json',{'path':str(source_dir),
            'manifest_sha256':sha(source_dir/'snapshot_manifest.json'),'snapshot':frozen})
    records=[]
    for row in tasks.to_dict('records'):
        row['replication']=int(row['replication'])
        row['output']=str(batch/row['arm']/row['scenario']/f'rep_{row["replication"]:05d}')
        records.append(row)
    pd.DataFrame(records).to_csv(batch/'task_output_contract.csv',index=False)
    metadata={'tag':args.tag,'workers':args.workers,'requested_tasks':len(records),
        'tasks_sha256':sha(batch/'requested_tasks.csv'),'snapshot':frozen,'source_snapshot_directory':str(source_dir),
        'scientific_denominators':tasks.groupby(['arm','scenario']).size().to_dict(),
        'failure_policy':'Every requested seed retained; fixed diagnosticretryonly; no replacement.',
        'sampler_seed_rule':'20261005+replication, withendpointoffset100; fixedbeforefinalfit, independentoftruth/performance.',
        'started_utc':datetime.now(timezone.utc).isoformat()}
    if protocol is not None:metadata['independent_protocol_sha256']=sha(batch/'independent_protocol_snapshot.json')
    metadata['scientific_denominators']={f'{a}/{s}':int(n) for(a,s),n in metadata['scientific_denominators'].items()}
    write_json(batch/'batch_manifest.json',metadata)
    completed=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker,initargs=(str(source_dir),))as pool:
        futures={pool.submit(fit_task,t):t for t in records}
        for future in as_completed(futures):
            task=futures[future]
            try:record=future.result()
            except Exception as error:
                record={**task,'completed':False,'success':False,'error_type':type(error).__name__,
                    'error':str(error),'traceback':traceback.format_exc()}
                write_json(Path(task['output'])/'worker_failure.json',record)
            completed.append(record)
            pd.DataFrame([{k:v for k,v in r.items()if k not in ['traceback','input_sha256']}
                for r in completed]).to_csv(batch/'execution_status.csv',index=False)
            if len(completed)%10==0 or len(completed)==len(records):
                print(json.dumps({'done':len(completed),'requested':len(records),
                    'completed_fits':sum(bool(r['completed'])for r in completed),
                    'converged':sum(bool(r['success'])for r in completed),
                    'elapsed_seconds':time.monotonic()-started}),flush=True)
    metadata.update(finished_utc=datetime.now(timezone.utc).isoformat(),
        seconds_total=time.monotonic()-started,completed_tasks=len(completed),
        completed_fits=sum(bool(r['completed'])for r in completed),
        converged=sum(bool(r['success'])for r in completed),
        all_requested_tasks_accounted_for=len(completed)==len(records))
    write_json(batch/'batch_manifest.json',metadata)


if __name__=='__main__':main()
