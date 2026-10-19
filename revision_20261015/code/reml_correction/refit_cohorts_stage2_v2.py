#!/usr/bin/env python3
"""Second-stage truth-blind REML and bootstrap recovery of the same reconstruction.

Inventory is deterministic and binds every input, original prediction/status,
endpoint seed, graph and bootstrap size. Output mirrors the original modern and
new-cohort trees; inputs, ICC draws and outcomes are never copied or opened.
Each method is checkpointed independently. ``lock`` requires all planned jobs
before any score adapter may open an evaluation outcome. Original replay is a
separate numerical control, never a corrected fit.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import hashlib
import importlib
import json
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np

CONTEXTS = ('SAR_probability_srs', 'SAR_logit_srs', 'SAR_logit_psu_sparse', 'IID_logit_psu_sparse')
KEYS = ('mean', 'lower', 'upper', 'mse', 'mse_plugin', 'mse_delta',
        'mse_naive', 'mse_bc', 'naive_mcse', 'bc_mcse')
COHORTS = ('principal_256', 'development_32', 'sparse57', 'observed_graph_64',
           'real_temporal', 'real_territorial', 'B300_subset_8')
SOURCE_FILES = ('sikov_model.py', 'sikov_bootstrap.py', 'sikov_published_reference.py')


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    def clean(x):
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,np.generic):return clean(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        return x
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False, default=serial) + '\n')
    temporary.replace(path)


def make_inventory(repo, private):
    """Training paths only. No evaluation_only files or public metrics opened."""
    old = private / 'iaos_reproduction_2026-10-13/sikov'
    new = private / 'iaos_revision_2026-10-15/numerical_cohorts'
    jobs = []

    def add(cohort, context, rep, method, model_input, original_dir, outdir,
            seeds, B=100, graph='full', spatial=True, development=False,
            reverse_coords=False, filename=None, stream=None):
        pred = original_dir / ('predictions.npz' if development else (filename or method) + '.npz')
        status = original_dir / ('status.json' if development else (filename or method) + '_status.json')
        if not all(p.is_file() for p in (model_input, pred, status)):
            raise FileNotFoundError(str([str(p) for p in (model_input, pred, status) if not p.is_file()]))
        oldstatus = read(status)
        endpoints = oldstatus.get('status' if development else 'fits', [])
        if B:
            assert len(endpoints) == 2, (cohort, method, status)
            assert all(end['bootstrap_B'] == B and all(sf['bootstrap_B'] == B for sf in end['fit']) for end in endpoints)
        job = dict(job_id=f'{cohort}/{context}/rep_{rep:05d}/{method}', cohort=cohort,
                   context=context, replication=rep, method=method, spatial=spatial,
                   graph=graph, uncertainty='parametric_bootstrap_naive' if B else 'delta',
                   bootstrap_B=B, endpoint_seeds=list(map(int, seeds)) if seeds else [],
                   seed_stream=stream, reverse_coordinate_columns=reverse_coords,
                   input_path=str(model_input.relative_to(private)), input_sha256=sha(model_input),
                   original_prediction_path=str(pred.relative_to(private)), original_prediction_sha256=sha(pred),
                   original_status_path=str(status.relative_to(private)), original_status_sha256=sha(status),
                   output_relative_dir=str(outdir),
                   prediction_filename='predictions.npz' if development else (filename or method) + '.npz',
                   status_filename='status.json' if development else (filename or method) + '_status.json',
                   development_status_schema=development,
                   baseline_converged=(all(oldstatus['success']) if development else bool(oldstatus['converged'])))
        jobs.append(job)

    for c in CONTEXTS:
        for r in range(30000, 30064):
            source = old / f'evaluacion/inputs/{c}/rep_{r:05d}/model_inputs/inputs.npz'
            base = old / f'evaluacion/fits/{c}/rep_{r:05d}'
            dest = Path(f'sikov/evaluacion/fits/{c}/rep_{r:05d}')
            seed = int(np.random.SeedSequence([20261010, 811, CONTEXTS.index(c), r]).generate_state(1)[0]) % (2**32 - 200000)
            for method, spatial in (('Sikov_FH', False), ('Sikov_SAR', True)):
                add('principal_256', c, r, method, source, base, dest,
                    [seed + j*1009 + (777 if spatial else 333) for j in range(2)], spatial=spatial,
                    stream=[20261010, 811, CONTEXTS.index(c), r])
            for method, graph in (('Sikov_SAR_k4', 'k4'), ('Sikov_SAR_k8', 'k8'), ('Sikov_SAR_haversine', 'haversine')):
                add('principal_256', c, r, method, source, base, dest, [], B=0, graph=graph)
            ps = int(np.random.SeedSequence([20261010, 941, CONTEXTS.index(c), r]).generate_state(1)[0])
            add('principal_256', c, r, 'Sikov_SAR_Published', source,
                old / f'evaluacion/published_sar/{c}/rep_{r:05d}',
                Path(f'sikov/evaluacion/published_sar/{c}/rep_{r:05d}'), [ps, ps+1009], graph='observed',
                stream=[20261010, 941, CONTEXTS.index(c), r])
        for r in range(8):
            source = old / f'desarrollo/inputs/{c}/rep_{r:05d}/model_inputs/inputs.npz'
            for index, (method, spatial) in enumerate((('Sikov_FH', False), ('Sikov_SAR', True))):
                seeds = [int(np.random.SeedSequence([20261010, 712, CONTEXTS.index(c), r, index, j]).generate_state(1)[0]) for j in range(2)]
                add('development_32', c, r, method, source,
                    old / f'desarrollo/recreated_native/fit_results/{c}/rep_{r:05d}/{method}',
                    Path(f'sikov/desarrollo/recreated_native/fit_results/{c}/rep_{r:05d}/{method}'),
                    seeds, spatial=spatial, development=True,
                    stream=[20261010, 712, CONTEXTS.index(c), r, index, 'endpoint_index'])
        for r in range(50000, 50064):
            source = new / f'sparse57/evaluacion/inputs/{c}/rep_{r:05d}/model_inputs/inputs.npz'
            base = new / f'sparse57/fits/{c}/rep_{r:05d}'
            dest = Path(f'numerical_cohorts/sparse57/fits/{c}/rep_{r:05d}')
            seed = int(np.random.SeedSequence([20261015, 1811, CONTEXTS.index(c), r]).generate_state(1)[0]) % (2**32 - 200000)
            for method, spatial in (('Sikov_FH', False), ('Sikov_SAR', True)):
                add('sparse57', c, r, method, source, base, dest,
                    [seed+j*1009+(777 if spatial else 333) for j in range(2)], spatial=spatial,
                    stream=[20261015, 1811, CONTEXTS.index(c), r])
            ps = int(np.random.SeedSequence([20261015, 1941, CONTEXTS.index(c), r]).generate_state(1)[0])
            add('sparse57', c, r, 'Sikov_SAR_Published', source, base, dest,
                [ps, ps+1009], graph='observed', stream=[20261015, 1941, CONTEXTS.index(c), r])
        for r in (30000, 30001):
            source = old / f'evaluacion/inputs/{c}/rep_{r:05d}/model_inputs/inputs.npz'
            base = old / f'evaluacion/b300/{c}/rep_{r:05d}'
            dest = Path(f'sikov/evaluacion/b300/{c}/rep_{r:05d}')
            add('B300_subset_8', c, r, 'full_graph', source, base, dest,
                [202610100000+CONTEXTS.index(c)*1000000+r*2+j for j in range(2)], B=300,
                stream=['202610100000+context_index*1000000+replication*2+endpoint_index'])
            ps = int(np.random.SeedSequence([20261010, 975, CONTEXTS.index(c), r]).generate_state(1)[0])
            add('B300_subset_8', c, r, 'published', source, base, dest,
                [ps, ps+1009], B=300, graph='observed', stream=[20261010, 975, CONTEXTS.index(c), r])
    c = 'SAR_probability_sampled_graph_srs'
    for r in range(30000, 30064):
        source = old / f'sensibilidad_sampled_graph/inputs/rep_{r:05d}/model_inputs/inputs.npz'
        base = old / f'sensibilidad_sampled_graph/fits/rep_{r:05d}'
        dest = Path(f'sikov/sensibilidad_sampled_graph/fits/rep_{r:05d}')
        seed = int(np.random.SeedSequence([20261010, 973, r]).generate_state(1)[0]) % (2**32 - 200000)
        for method, spatial in (('Sikov_FH', False), ('Sikov_SAR', True)):
            add('observed_graph_64', c, r, method, source, base/'original', dest/'original',
                [seed+j*1009+(777 if spatial else 333) for j in range(2)], spatial=spatial,
                stream=[20261010, 973, r])
        for method, graph in (('Sikov_SAR_k4', 'k4'), ('Sikov_SAR_k8', 'k8'), ('Sikov_SAR_haversine', 'haversine')):
            add('observed_graph_64', c, r, method, source, base/'original', dest/'original', [], B=0, graph=graph)
        ps = int(np.random.SeedSequence([20261010, 974, r]).generate_state(1)[0])
        add('observed_graph_64', c, r, 'Sikov_SAR_Published', source, base/'published', dest/'published',
            [ps, ps+1009], graph='observed', stream=[20261010, 974, r])
    for scope, year in (('temporal', 2019), ('territorial', 2024)):
        source = old / ('datos/real_2019_full' if year == 2019 else 'datos/real_2024_training') / 'fit_inputs.npz'
        base = old / f'evaluacion/real_{scope}'
        dest = Path(f'sikov/evaluacion/real_{scope}')
        seed = 20261010+year
        for method, spatial in (('Sikov_FH', False), ('Sikov_SAR', True)):
            add('real_'+scope, scope, year, method, source, base, dest,
                [seed+j*1009+(777 if spatial else 333) for j in range(2)], spatial=spatial,
                reverse_coords=True, stream=['20261010+training_year'])
        for method, graph in (('Sikov_SAR_k4', 'k4'), ('Sikov_SAR_k8', 'k8'), ('Sikov_SAR_haversine', 'haversine')):
            add('real_'+scope, scope, year, method, source, base, dest, [], B=0, graph=graph, reverse_coords=True)
        ps = int(np.random.SeedSequence([20261010, 942, 0 if scope == 'temporal' else 1]).generate_state(1)[0])
        add('real_'+scope, scope, year, 'Sikov_SAR_Published', source,
            old / f'evaluacion/published_sar/real_{scope}', Path(f'sikov/evaluacion/published_sar/real_{scope}'),
            [ps, ps+1009], graph='observed', reverse_coords=True,
            stream=[20261010, 942, 0 if scope == 'temporal' else 1])
    jobs.sort(key=lambda x: x['job_id'])
    assert len(jobs) == 2780 and len({x['job_id'] for x in jobs}) == 2780
    assert sum(bool(x['bootstrap_B']) for x in jobs) == 1814
    src = repo / 'src/estudio_publicacion/sikov_2026-10-10/scripts'
    return dict(format='reml-correction-jobs-v1', generated_utc=utc(), truth_read=False,
                original_files_modified=False, count_unit='method x independent input population/scope; bootstrap contains two endpoints',
                source_models_sha256={str((src/name).relative_to(repo)):sha(src/name) for name in SOURCE_FILES},
                seed_generator_numpy_version=np.__version__,
                graph_variants_have_delta_uncertainty_not_bootstrap=True,
                development_nuts_reuses_corrected_native_development_fits=True,
                legacy_scope=dict(earlier_800='Separate Gaussian logit multivariate area model in scripts/models.py, different 2-6 covariance parameters and predictive-error bootstrap; not Sikov A/rho likelihood or its bootstrap. Preserved as historical supporting evidence, not newly KKT-certified.',
                                  CUT40='Separate ICC/CUT variance-module posterior sampler; no Sikov REML/bootstrap call.'),
                jobs=jobs)


def inventory(args):
    path = args.output / 'job_inventory.json'
    if path.exists():
        raise FileExistsError('Preserve existing inventory: '+str(path))
    result = make_inventory(args.repo, args.private_run_root)
    save(path, result)
    public = args.repo/'revision_20261015/results/reml_correction'
    save(public/'job_inventory.json', result)
    layout = dict(modern_run_root=str(args.output/'sikov'), numerical_run_root=str(args.output/'numerical_cohorts'),
                  old_modern_root=str(args.private_run_root/'iaos_reproduction_2026-10-13/sikov'),
                  old_numerical_root=str(args.private_run_root/'iaos_revision_2026-10-15/numerical_cohorts'),
                  job_inventory_sha256=sha(path), private_arrays_remain_private=True,
                  unchanged_inputs_and_ICC_overlay='Scoring adapter may symlink unchanged inputs/ICC files individually; do not write through symlinks.',
                  corrected_job_completion='Each corrected method has <method>_reml_correction_completion.json; no original world completion is overwritten.',
                  global_lock=str(args.output/'prediction_lock_before_score.json'))
    save(args.output/'scoring_layout.json', layout)
    counts=collections.Counter((j['cohort'],j['uncertainty']) for j in result['jobs'])
    print(json.dumps(dict(jobs=len(result['jobs']), by_cohort_uncertainty={str(k):v for k,v in sorted(counts.items())},
                         inventory_sha256=sha(path)),indent=2))


def completion_path(output, job):
    return output/job['output_relative_dir']/(job['method']+'_reml_correction_completion.json')


def verify_checkpoint(output, job):
    path = completion_path(output,job)
    if not path.is_file():return None
    obj=read(path)
    assert obj['job_id']==job['job_id'] and obj['input_sha256']==job['input_sha256']
    for rel,h in obj['outputs_sha256'].items():assert sha(output/rel)==h,rel
    return obj


def modules(repo, corrected):
    src=repo/'src/estudio_publicacion/sikov_2026-10-10/scripts'
    sys.path.insert(0,str(src))
    sys.path.insert(0,str(Path(__file__).parent))
    sm=importlib.import_module('sikov_model'); sb=importlib.import_module('sikov_bootstrap')
    solver=None
    if corrected:
        solver=importlib.import_module('stable_reml_solver_stage2_v2');solver.install();solver.reset_counters()
    # Published wrapper captures the corrected bootstrap function on first import.
    pr=importlib.import_module('sikov_published_reference')
    return sm,sb,pr,solver


def fit_job(payload):
    args,job=payload
    repo=Path(args['repo']);private=Path(args['private_run_root']);output=Path(args['output'])
    corrected=not args.get('original_replay',False)
    saved=verify_checkpoint(output,job)
    if saved:return dict(job_id=job['job_id'],status='REUSED_VERIFIED_CHECKPOINT',seconds=0)
    plan=read(output/'job_inventory.json') if (output/'job_inventory.json').is_file() else read(Path(args['inventory']))
    for rel,h in plan['source_models_sha256'].items():assert sha(repo/rel)==h,rel
    protocol=Path(args['protocol']) if args.get('protocol') else None
    if corrected:
        if protocol is None or not protocol.is_file():raise RuntimeError('A registered numerical-correction protocol is required before corrected fitting')
    for rel,h in args.get('execution_sources',{}).items():assert sha(repo/rel)==h,rel
    inp=private/job['input_path'];assert sha(inp)==job['input_sha256']
    for key,h in (('original_prediction_path','original_prediction_sha256'),('original_status_path','original_status_sha256')):
        assert sha(private/job[key])==job[h]
    sm,sb,pr,solver=modules(repo,corrected)
    if solver:solver.reset_counters()
    tick=time.perf_counter()
    d=dict(np.load(inp,allow_pickle=False));names=d['covariate_names'].astype(str).tolist()
    if job['reverse_coordinate_columns']:d['coords']=d['coords'][:,::-1]
    graph=job['graph']
    W=d['W'] if graph in ('full','observed') else sm.knn_weights(d['coords'],k=4 if graph=='k4' else 8 if graph=='k8' else 6,strata=d['strata'],haversine=graph=='haversine')
    values=[]
    try:
        for j,endpoint in enumerate(('anemia','stunting')):
            mask=np.isfinite(d['y'][:,j])&np.isfinite(d['D'][:,j,j])&(d['D'][:,j,j]>0)
            common=dict(spatial=job['spatial'],outcome=endpoint,covariate_names=names)
            x,y,D,s=d['X_raw'],d['y'][mask,j],d['D'][mask,j,j],d['sample_idx'][mask]
            if job['bootstrap_B']:
                boot=dict(B=job['bootstrap_B'],seed=job['endpoint_seeds'][j],interval_estimator='naive')
                if graph=='observed':
                    value=pr.fit_sikov_published_bootstrap(x,y,D,s,d['coords'],d['strata'],outcome=endpoint,covariate_names=names,**boot)
                else:value=sb.fit_sikov_bootstrap(x,y,D,s,W,d['strata'],**common,**boot)
            else:value=sm.fit_sikov(x,y,D,s,W,d['strata'],uncertainty='delta',**common)
            values.append(value)
        keys=KEYS if job['bootstrap_B'] else KEYS[:6]
        pred={k:np.column_stack([v[k] for v in values]) for k in keys}
        assert all(np.isfinite(pred[k]).all() for k in ('mean','lower','upper','mse'))
        converged=all(v['converged'] for v in values)
        if corrected:
            assert converged,job['job_id']
            for v in values:
                for sf in v['fit']:
                    cert=sf.get('stationarity_certificate',sf.get('numerical_correction',{}).get('certificate'))
                    if cert is not None:assert cert['passed'],(job['job_id'],sf)
        folder=output/job['output_relative_dir'];folder.mkdir(parents=True,exist_ok=True)
        path=folder/job['prediction_filename'];statuspath=folder/job['status_filename']
        # Never permit a scoring overlay to redirect a fit mutation into baseline files.
        for p in (path,statuspath):
            if p.exists() or p.is_symlink():raise FileExistsError('Preserve incomplete method output: '+str(p))
        np.savez_compressed(path,**pred)
        status=(dict(status=values,success=[bool(v['converged']) for v in values],fit_seconds=time.perf_counter()-tick,phase='development')
                if job['development_status_schema'] else dict(converged=converged,fits=values))
        save(statuspath,status)
        outputs={str(p.relative_to(output)):sha(p) for p in (path,statuspath)}
        if job['bootstrap_B'] and job['method'] in ('Sikov_FH','Sikov_SAR','Sikov_SAR_Published') and not job['development_status_schema']:
            bc={k:v.copy() for k,v in pred.items()};bc['mse']=bc['mse_bc'].copy()
            se=np.sqrt(np.maximum(bc['mse'],0));bc['lower']=bc['mean']-1.959963984540054*se;bc['upper']=bc['mean']+1.959963984540054*se
            bcp=folder/(job['method']+'_BC.npz');bcs=folder/(job['method']+'_BC_status.json')
            np.savez_compressed(bcp,**bc);save(bcs,dict(converged=converged,negative_bc_MSE_cells=int((bc['mse_bc']<0).sum()),negative_MSE_floor_only_for_interval_sqrt=0.,same_bootstrap_draws_as_primary=True))
            outputs.update({str(p.relative_to(output)):sha(p) for p in (bcp,bcs)})
        baseline=dict(np.load(private/job['original_prediction_path'],allow_pickle=False))
        delta={k:float(np.max(abs(pred[k]-baseline[k]))) for k in pred if k in baseline}
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss=int(rss if platform.system()=='Darwin' else rss*1024)
        telemetry=solver.get_counters() if solver else dict(mode='original_replay')
        for rel,h in args.get('execution_sources',{}).items():assert sha(repo/rel)==h,rel
        record=dict(job_id=job['job_id'],cohort=job['cohort'],context=job['context'],replication=job['replication'],method=job['method'],
                    created_utc=utc(),seconds=time.perf_counter()-tick,process_peak_RSS_bytes=rss,
                    input_sha256=job['input_sha256'],job_inventory_sha256=sha(Path(args['inventory'])),
                    source_sha256=sha(__file__),solver_sha256=sha(Path(__file__).with_name('stable_reml_solver_stage2_v2.py')) if solver else None,
                    protocol_sha256=sha(protocol) if protocol else None,
                    fitting_sources_sha256=args.get('execution_sources',{}),
                    corrected=corrected,converged=converged,truth_read=False,
                    base_solver_sha256=sha(Path(__file__).with_name('stable_reml_solver.py')) if solver else None,
                    job_selection_sha256=sha(Path(args['job_selection'])) if args.get('job_selection') else None,
                    same_graph_model_seeds_B=True,endpoint_seeds=job['endpoint_seeds'],bootstrap_B=job['bootstrap_B'],
                    max_absolute_delta_vs_original=delta,
                    telemetry=telemetry,outputs_sha256=outputs)
        save(completion_path(output,job),record)
        if not corrected and max(delta.values(),default=0)>1e-10:
            raise RuntimeError('Original replay differs; evidence retained '+job['job_id'])
        return {k:record[k] for k in ('job_id','seconds','process_peak_RSS_bytes','converged','max_absolute_delta_vs_original','telemetry')}
    except BaseException as exc:
        folder=output/'failures';folder.mkdir(parents=True,exist_ok=True)
        error=dict(job_id=job['job_id'],failed_utc=utc(),exception=repr(exc),traceback=traceback.format_exc(),truth_read=False)
        save(folder/(hashlib.sha256(job['job_id'].encode()).hexdigest()[:16]+'_'+dt.datetime.now().strftime('%Y%m%dT%H%M%S')+'.json'),error)
        raise


def selected_jobs(args, plan):
    jobs=plan['jobs']
    if args.job_selection:
        selection=read(args.job_selection)
        assert selection['registered_inventory_sha256']==sha(args.inventory)
        assert selection['truth_read'] is False
        ids=set(selection['selected_job_ids'])
        assert len(ids)==len(selection['selected_job_ids'])
        assert ids.issubset({j['job_id'] for j in jobs})
        jobs=[j for j in jobs if j['job_id'] in ids]
    for key in ('cohort','context','replication','method'):
        value=getattr(args,key)
        if value is not None:jobs=[j for j in jobs if j[key]==value]
    if args.limit is not None:jobs=jobs[:args.limit]
    return jobs


def run(args):
    plan=read(args.inventory);jobs=selected_jobs(args,plan)
    if not jobs:raise ValueError('No matching jobs')
    payload={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    sources=[Path(__file__).resolve()]
    if not args.original_replay:sources.append(Path(__file__).with_name('stable_reml_solver_stage2_v2.py').resolve())
    if not args.original_replay:sources.append(Path(__file__).with_name('stable_reml_solver.py').resolve())
    payload['execution_sources']={str(p.relative_to(args.repo)):sha(p) for p in sources}
    pending=[j for j in jobs if not verify_checkpoint(args.output,j)]
    failures=[]
    log=args.output/'dispatch_events.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(fit_job,(payload,j)):j for j in pending}
        for future in cf.as_completed(futures):
            job=futures[future]
            try:event=dict(status='METHOD_CLOSED',utc=utc(),**future.result())
            except BaseException as exc:
                event=dict(status='FAILED_RETAINED',job_id=job['job_id'],utc=utc(),exception=repr(exc));failures.append(event)
            with log.open('a') as f:f.write(json.dumps(event,default=serial)+'\n')
            print(json.dumps(event,default=serial),flush=True)
    save(args.output/'last_dispatch.json',dict(requested=len(jobs),pending_at_start=len(pending),workers=args.workers,failures=failures,finished_utc=utc(),truth_read=False))
    if failures:raise RuntimeError(str(len(failures))+' corrected methods failed; retain receipts and block global lock')


def status(args, lock=False):
    plan=read(args.inventory);records=[];remaining=[]
    selected=selected_jobs(args,plan)
    for job in selected:
        record=verify_checkpoint(args.output,job)
        if record is None:remaining.append(job['job_id'])
        else:records.append(record)
    counts=collections.Counter(j['cohort'] for j in selected)
    closed=collections.Counter(j['cohort'] for j in records)
    result=dict(updated_utc=utc(),truth_read=False,planned=len(selected),completed=len(records),remaining=len(remaining),
                cohorts={c:dict(planned=counts[c],completed=closed[c]) for c in sorted(counts)},
                all_closed=not remaining,all_closed_converged=all(x['converged'] for x in records),
                observed_closed_seconds=sum(x['seconds'] for x in records),
                inventory_sha256=sha(args.inventory),remaining_job_ids=remaining)
    save(args.output/'status.json',result)
    if lock:
        if remaining or not result['all_closed_converged']:raise RuntimeError('Complete correction cohort required before prediction lock')
        if not all(x['corrected'] for x in records):raise RuntimeError('Original replay is not a corrected prediction lock')
        if len({x['solver_sha256'] for x in records}) != 1 or len({x['source_sha256'] for x in records}) != 1:
            raise RuntimeError('Mixed fitting source versions cannot form a global correction lock')
        target=args.output/'prediction_lock_before_score.json'
        if target.exists():raise FileExistsError('Prediction lock is immutable: '+str(target))
        result.update(locked_utc=utc(),status='ALL_CORRECTED_PREDICTIONS_LOCKED_BEFORE_RESCORING',
                      complete=True,truth_read=False,outputs_sha256={rel:h for x in records for rel,h in x['outputs_sha256'].items()},
                      truth_opened_by_refits=False,items=[dict(job_id=x['job_id'],completion_relative_path=str(completion_path(args.output,j).relative_to(args.output)),completion_sha256=sha(completion_path(args.output,j)),outputs_sha256=x['outputs_sha256'])
                          for j in selected for x in [verify_checkpoint(args.output,j)]])
        save(target,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('items','remaining_job_ids')},indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=('inventory','run','status','lock'))
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--private-run-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--inventory',type=Path)
    ap.add_argument('--protocol',type=Path)
    ap.add_argument('--job-selection',type=Path,required=True)
    ap.add_argument('--cohort',choices=COHORTS)
    ap.add_argument('--context')
    ap.add_argument('--replication',type=int)
    ap.add_argument('--method')
    ap.add_argument('--limit',type=int)
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--original-replay',action='store_true')
    args=ap.parse_args()
    for name in ('repo','private_run_root','output','inventory','protocol','job_selection'):
        if getattr(args,name) is not None:setattr(args,name,getattr(args,name).expanduser().resolve())
    if args.output.is_relative_to(args.repo):raise ValueError('Private arrays must remain outside public repository')
    if not 1<=args.workers<=10:raise ValueError('Single-thread worker budget must be1..10')
    args.inventory=args.inventory or args.output/'job_inventory.json'
    if args.action=='inventory':inventory(args)
    elif args.action=='run':run(args)
    else:status(args,args.action=='lock')


if __name__=='__main__':main()
