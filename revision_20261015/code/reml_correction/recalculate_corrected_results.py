#!/usr/bin/env python3
"""Recalculate the IAOS evidence after truth-blind REML correction.

The prepare action reads no evaluation truth. It builds a separate private
scoring tree, reuses ICC predictions/draws verbatim, reconstructs the original
mixture CDFs and locks every scoring input and prediction. The score action
checks that complete lock before invoking the unchanged score arithmetic.
Nothing in this adapter writes to the original private or public freezes.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import ast
import copy
import datetime as dt
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd

CONTEXTS = ('SAR_probability_srs', 'SAR_logit_srs', 'SAR_logit_psu_sparse', 'IID_logit_psu_sparse')
ORIGINAL = ('Sikov_FH','Sikov_SAR','Sikov_FH_BC','Sikov_SAR_BC',
            'Sikov_SAR_k4','Sikov_SAR_k8','Sikov_SAR_haversine','ICC','Selected')
PUBLISHED = ('Sikov_SAR_Published','Sikov_SAR_Published_BC')
DEVELOPMENT = ('Sikov_FH','Sikov_SAR','ICC','ICC_poverty','ICC_SAR_w25','ICC_SAR_w50','ICC_SAR_w75')
DISPLAY={'ICC':'ICC beta','ICC_poverty':'Poverty-stratum ICC',
         'ICC_SAR_w25':'ICC/SAR 25/75','ICC_SAR_w50':'ICC/SAR 50/50',
         'ICC_SAR_w75':'Proposed mixture (75/25)',
         'SAR_probability_srs':'Probability SAR / SRS','SAR_logit_srs':'Logit SAR / SRS',
         'SAR_logit_psu_sparse':'Logit SAR / sparse PSUs','IID_logit_psu_sparse':'IID logit / sparse PSUs'}
ARCHIVED_RUNTIME_ROUTES=None


def counts_text(counts):
    return ', '.join(f'{DISPLAY.get(method,method)}: {count}' for method,count in sorted(counts.items()))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Refusing to write through a private reuse link: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=lambda x:
        x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n')


def arrays(path):
    with np.load(path, allow_pickle=False) as value:
        return {key: value[key] for key in value.files}


def evaluation_truth(folder):
    """Called only by the score phase, after both complete prediction locks."""
    path=folder/'evaluation_only/truth.npz'
    if sha(path)!=read(folder/'manifest.json')['truth_sha256']:
        raise ValueError('Evaluation population truth differs from its locked generation manifest')
    return arrays(path)


def save_arrays(path, data):
    path = Path(path)
    if path.is_symlink():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k:v for k,v in data.items() if isinstance(v,np.ndarray)})


def link(source, target):
    source, target = Path(source).resolve(), Path(target)
    if not source.exists():
        raise FileNotFoundError(source)
    if target.exists() or target.is_symlink():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=source.is_dir())


def imports(repo):
    for part in ('reproduction','revision_20261015/code','src/estudio_publicacion/scripts/competitors',
                 'src/estudio_publicacion/scripts','src/estudio_publicacion/sikov_2026-10-10/scripts'):
        sys.path.insert(0,str(repo/part))
    return importlib.import_module('improve_candidates')


def reused_folder(old, new, exclude=()):
    """Reuse only direct files plus sampler directories, never old completions."""
    new.mkdir(parents=True,exist_ok=True)
    for path in old.iterdir():
        if path.name in set(exclude) | {'completion.json','predictions.csv'}:
            continue
        if path.is_file() or path.name in ('hmc_chains','icc_sampler'):
            link(path,new/path.name)


def converged(folder, method, development=False):
    path = folder/method/'status.json' if development else folder/(method+'_status.json')
    status = read(path)
    if 'converged' in status:
        return bool(status['converged'])
    if 'success' in status:
        return bool(all(status['success']))
    raise ValueError('No explicit method convergence: '+str(path))


def checked_seconds(value,source):
    value=float(value)
    if not np.isfinite(value) or value<0:
        raise ValueError('Invalid observed runtime in '+str(source))
    return value


def configure_archived_runtime(args):
    global ARCHIVED_RUNTIME_ROUTES
    ARCHIVED_RUNTIME_ROUTES=(args.corrected_root,args.old_root,args.old_numerical_root)


def canonical_ICC_runtime(folder,method,development):
    corrected,modern,numerical=ARCHIVED_RUNTIME_ROUTES
    relative=folder.relative_to(corrected)
    if relative.parts[0]=='sikov':base=modern;label='ORIGINAL_MODERN'
    elif relative.parts[0]=='numerical_cohorts':base=numerical;label='ORIGINAL_NUMERICAL_COHORTS'
    else:raise ValueError('Unknown archived ICC runtime routing')
    previous=base.joinpath(*relative.parts[1:])
    location=previous/method if development else previous
    status=location/'status.json' if development else previous/(method+'_status.json')
    prediction=location/'predictions.npz' if development else previous/(method+'.npz')
    draws=location/'conditional_beta_draws.npz' if development else previous/(method+'_conditional_draws.npz')
    receipt=previous/(method+'_completion.json')
    if receipt.is_file() and 'seconds' in read(receipt):source=receipt;seconds=read(receipt)['seconds']
    else:
        source=status;value=read(status)
        key=next((key for key in ('seconds','fit_seconds','elapsed_seconds') if key in value),None)
        if key is None:raise ValueError('Archived ICC has no observed runtime: '+str(source))
        seconds=value[key]
    files={p.relative_to(base).as_posix():sha(p) for p in (status,prediction,draws)}
    files[source.relative_to(base).as_posix()]=sha(source)
    seconds=checked_seconds(seconds,source)
    return dict(seconds=seconds,current_reml_seconds=0.,historical_reused_seconds=seconds,
        mixture_recompute_seconds=0.,scope='historical_ICC_fit_reused_without_refitting',
        source_archive=label,canonical_archived_runtime_source=True,source_sha256=files)


def method_runtime(folder,method,development=False):
    """Separate retained native execution, archived ICC time and new CDF work.

    Retained-source seconds are not total stage1+stage2 search cost. Rejected
    numerical stages remain in their own native receipts and are never charged
    as additional statistical populations or bootstrap draws.
    """
    base=method[:-3] if method.endswith('_BC') else method
    location=folder/base if development else folder
    receipt=location/(base+'_reml_correction_completion.json')
    status_path=location/'status.json' if development else folder/(method+'_status.json')
    status=read(status_path)
    if receipt.is_file():
        seconds=checked_seconds(read(receipt)['seconds'],receipt)
        return dict(seconds=seconds,current_reml_seconds=seconds,historical_reused_seconds=0.,
                    mixture_recompute_seconds=0.,scope='corrected_retained_native_method_execution',
                    source_sha256={receipt.name:sha(receipt)})
    if method in ('ICC','ICC_poverty') and ARCHIVED_RUNTIME_ROUTES is not None:
        return canonical_ICC_runtime(folder,method,development)
    if 'runtime_composition' in status:return status['runtime_composition']
    if method not in ('ICC','ICC_poverty'):
        raise ValueError('Missing corrected method runtime receipt: '+str(receipt))
    old_receipt=folder/(method+'_completion.json')
    if old_receipt.is_file() and 'seconds' in read(old_receipt):
        source=old_receipt;seconds=read(source)['seconds']
    else:
        source=status_path
        key=next((key for key in ('seconds','fit_seconds','elapsed_seconds') if key in status),None)
        if key is None:raise ValueError('Reused ICC fit has no archived observed runtime: '+str(source))
        seconds=status[key]
    seconds=checked_seconds(seconds,source)
    return dict(seconds=seconds,current_reml_seconds=0.,historical_reused_seconds=seconds,
                mixture_recompute_seconds=0.,scope='historical_ICC_fit_reused_without_refitting',
                source_sha256={source.name:sha(source)})


def world_runtime(folder,methods,development=False):
    timings={method:method_runtime(folder,method,development) for method in methods}
    # BC aliases reuse the same fit; mixtures reuse their two components.
    bases=[row for method,row in timings.items() if not method.endswith('_BC') and
           method not in ('Selected','Selected_B300','ICC_SAR_w25','ICC_SAR_w50','ICC_SAR_w75')]
    parts={name:sum(row[name] for row in bases) for name in
           ('current_reml_seconds','historical_reused_seconds')}
    if 'Selected_B300' in timings and 'ICC' not in timings:
        parts['historical_reused_seconds']+=timings['Selected_B300']['historical_reused_seconds']
    parts['mixture_recompute_seconds']=sum(row['mixture_recompute_seconds'] for row in timings.values())
    return dict(seconds=sum(parts.values()),method_runtime=timings,runtime_composition=parts,
        seconds_scope='sum_of_retained_native_method_times_archived_reused_ICC_and_new_CDF_work_not_single_run_wall_time',
        discarded_numerical_stage_cost_included=False)


def timed_rows(rows,timing):
    for row in rows:
        row.update(fit_seconds_current_reml=timing['current_reml_seconds'],
            fit_seconds_historical_reused=timing['historical_reused_seconds'],
            fit_seconds_mixture_recompute=timing['mixture_recompute_seconds'],
            fit_seconds_scope=timing['scope'])
    return rows


def timing_summary(frame,summarize):
    summary=summarize(frame)
    keys=['scenario','method','protocol','group','outcome']
    names=['fit_seconds_current_reml','fit_seconds_historical_reused','fit_seconds_mixture_recompute']
    if not all(name in frame for name in names):return summary
    extra=frame.groupby(keys,dropna=False)[names].mean().reset_index().rename(
        columns={name:'mean_'+name for name in names})
    scopes=frame.groupby(keys,dropna=False).fit_seconds_scope.agg(
        lambda x:';'.join(sorted(set(x)))).reset_index(name='fit_seconds_scope')
    return summary.merge(extra,on=keys,validate='one_to_one').merge(scopes,on=keys,validate='one_to_one')


def runtime_accounting(args):
    """Public-safe cost aggregates; repeats are computation, never more worlds."""
    lock=read(args.native_lock);rows=[]
    for item in lock['items']:
        path=args.corrected_root/item['completion_relative_path'];record=read(path)
        rows.append(dict(cohort=record['cohort'],method=record['method'],
            retained_stage=item.get('stage','single_native_stage'),
            observed_seconds=checked_seconds(record['seconds'],path)))
    frame=pd.DataFrame(rows)
    summary=frame.groupby(['cohort','method','retained_stage']).observed_seconds.agg(
        ['count','sum','mean','min','max']).reset_index().rename(columns={
        'count':'retained_method_jobs','sum':'retained_native_seconds',
        'mean':'mean_retained_native_seconds','min':'min_retained_native_seconds',
        'max':'max_retained_native_seconds'})
    summary.to_csv(args.corrected_root/'runtime_corrected_native_summary.csv',index=False)
    inventory=read(args.corrected_root/'job_inventory.json')['jobs']
    stages=[]
    stage_roots=[Path(x) for x in lock.get('allowed_stage_roots',[])] or [args.corrected_root]
    for index,root in enumerate(stage_roots,1):
        seconds=[];sources={}
        for job in inventory:
            relative=job['output_relative_dir']+'/'+job['method']+'_reml_correction_completion.json'
            path=root/relative
            if path.is_file():
                seconds.append(checked_seconds(read(path)['seconds'],path));sources[relative]=sha(path)
        stages.append(dict(stage='stage'+str(index),closed_method_executions=len(seconds),
            observed_closed_method_execution_seconds=sum(seconds),completion_receipts_sha256=sources))
    value=dict(status='OBSERVED_NUMERICAL_REVISION_RUNTIME_ACCOUNTING',
        retained_native_method_jobs=len(rows),retained_native_method_seconds=float(frame.observed_seconds.sum()),
        stages=stages,scopes=dict(
            corrected_native='Observed seconds of the retained native completion for each registered method job; B100/B300 bootstrap and graph variants are included.',
            current_all_closed_stages='Sum of closed method receipts in all numerical stages, including numerical reruns; failed incomplete attempts and orchestration wall time are excluded.',
            reused_ICC='Archived saved-fit seconds are recorded separately in per-method and world scoring completions; no ICC computation was repeated.',
            mixture='New CDF computation seconds are measured during preparation; the method total also reports archived ICC plus retained corrected SAR components.',
            limitations='Separate-stage observed costs are machine-dependent component measurements, not one common hardware benchmark or elapsed parallel wall time. Development NUTS reuses the same corrected Gaussian references and does not add native jobs.'),
        discarded_numerical_stages_are_additional_populations=False)
    save(args.corrected_root/'runtime_accounting.json',value)


def mixture(folder, method, weight, module, development=False):
    if development:
        icc = arrays(folder/'ICC/predictions.npz')
        draws = arrays(folder/'ICC/conditional_beta_draws.npz')
        sar = arrays(folder/'Sikov_SAR/predictions.npz')
        destination = folder/method/'predictions.npz'
        status_path = folder/method/'status.json'
    else:
        icc = arrays(folder/'ICC.npz')
        draws = arrays(folder/'ICC_conditional_draws.npz')
        sar = arrays(folder/'Sikov_SAR.npz')
        destination = folder/(method+'.npz')
        status_path = folder/(method+'_status.json')
    base = {'p_mean':icc['mean'],'p_variance':icc.get('p_variance',icc.get('mse'))}
    if 'anemia_marginal' in draws:
        base['marginal_draws'] = [draws['anemia_marginal'],draws['stunting_marginal']]
    else:
        base['p_draws'] = draws['p_draws']
    tick=time.perf_counter()
    prediction = module.mixture_prediction(base,sar,weight)
    elapsed=time.perf_counter()-tick
    icctime=method_runtime(folder,'ICC',development)
    sartime=method_runtime(folder,'Sikov_SAR',development)
    timing=dict(seconds=icctime['seconds']+sartime['seconds']+elapsed,
        current_reml_seconds=sartime['current_reml_seconds'],
        historical_reused_seconds=icctime['historical_reused_seconds'],
        mixture_recompute_seconds=elapsed,
        scope=('archived_reused_ICC_plus_corrected_retained_SAR_plus_new_mixture_CDF_execution'+
               ('_SAR_reference_reused_from_Laplace_development_no_additional_fit' if sartime.get('reused_in_cohort')=='development_nuts' else '')),
        source_sha256={'ICC':icctime['source_sha256'],'Sikov_SAR':sartime['source_sha256']})
    save_arrays(destination,prediction)
    if status_path.is_symlink():status_path.unlink()
    success = converged(folder,'ICC',development) and converged(folder,'Sikov_SAR',development)
    if development:
        save(status_path,{'status':prediction['status'],'success':[success,success],
                          'fit_seconds':timing['seconds'],'runtime_composition':timing,
                          'phase':'development','numerical_revision':'strict_REML_20261018'})
    else:
        save(status_path,{'converged':success,'recipe':prediction['status'],'runtime_composition':timing,
                          'numerical_revision':'strict_REML_20261018'})


def boundary_alias(folder, method):
    data = arrays(folder/(method+'.npz'))
    if 'mse_bc' not in data:
        # The runner must preserve raw bootstrap outputs needed by BC sensitivity.
        raise ValueError('Corrected bootstrap output lacks raw BC uncertainty: '+method)
    bc = {k:v.copy() for k,v in data.items()}
    bc['mse']=bc['mse_bc']
    sd=np.sqrt(np.maximum(bc['mse'],0.))
    bc['lower']=bc['mean']-1.959963984540054*sd
    bc['upper']=bc['mean']+1.959963984540054*sd
    prediction=folder/(method+'_BC.npz')
    # Preserve bytes of aliases already covered by the native correction lock.
    if prediction.exists():
        saved=arrays(prediction)
        if not all(k in saved and np.array_equal(saved[k],v,equal_nan=True) for k,v in bc.items()):
            raise ValueError('Native BC alias differs from unchanged arithmetic: '+str(prediction))
    else:save_arrays(prediction,bc)
    status=folder/(method+'_BC_status.json')
    if status.exists():
        if bool(read(status)['converged'])!=converged(folder,method):
            raise ValueError('Native BC convergence alias differs')
        return
    if status.is_symlink():status.unlink()
    save(status,{'converged':converged(folder,method),'negative_bc_MSE_cells':int((bc['mse_bc']<0).sum()),
                 'negative_MSE_floor_only_for_interval_sqrt':0.,'same_bootstrap_draws_as_primary':True})


def seal_b300_mixture(folder,original,module):
    """Build the B300 sensitivity distribution before any evaluation truth."""
    icc=arrays(original/'ICC.npz');draws=arrays(original/'ICC_conditional_draws.npz')
    sar=arrays(folder/'full_graph.npz')
    base={'p_mean':icc['mean'],'p_variance':icc['mse'],
          'marginal_draws':[draws['anemia_marginal'],draws['stunting_marginal']]}
    tick=time.perf_counter();prediction=module.mixture_prediction(base,sar,.75);elapsed=time.perf_counter()-tick
    icctime=method_runtime(original,'ICC');sartime=method_runtime(folder,'full_graph')
    timing=dict(seconds=icctime['seconds']+sartime['seconds']+elapsed,
        current_reml_seconds=sartime['current_reml_seconds'],
        historical_reused_seconds=icctime['historical_reused_seconds'],
        mixture_recompute_seconds=elapsed,
        scope='archived_reused_ICC_plus_corrected_retained_B300_SAR_plus_new_mixture_CDF_execution',
        source_sha256={'ICC':icctime['source_sha256'],'SAR_B300':sartime['source_sha256']})
    save_arrays(folder/'Selected_B300.npz',prediction)
    save(folder/'Selected_B300_status.json',dict(
        converged=converged(original,'ICC') and converged(folder,'full_graph'),
        recipe=prediction['status'],runtime_composition=timing,
        truth_read=False,fixed_ICC_weight=.75,component_prediction_sha256={
            'ICC':sha(original/'ICC.npz'),'ICC_draws':sha(original/'ICC_conditional_draws.npz'),
            'SAR_B300':sha(folder/'full_graph.npz')}))


def sealed_b300_scorer(module):
    """Adapt only prediction loading; retain the frozen B300 metric arithmetic."""
    if getattr(module._full_b300,'_iaos_sealed_B300_loading',False):return module._full_b300
    source=inspect.getsource(module._full_b300);tree=ast.parse(source);changed=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mixed' for t in node.targets):
            if not isinstance(node.value,ast.Call) or ast.unparse(node.value.func)!='mixture_prediction':
                raise ValueError('Unexpected frozen B300 mixture assignment')
            node.value=ast.parse("_arrays(run / 'evaluacion/b300' / context / f'rep_{rep:05d}' / 'Selected_B300.npz')",mode='eval').body
            changed.append(node)
    if len(changed)!=1:raise ValueError('Expected exactly one B300 distribution loading adaptation')
    namespace=dict(vars(module))
    exec(compile(ast.fix_missing_locations(tree),module.__file__,'exec'),namespace)
    result=namespace['_full_b300'];result._iaos_sealed_B300_loading=True
    result._iaos_loading_provenance=dict(
        historical_function_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        adapted_function_AST_sha256=hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        changed_assignments=1,adaptation='Load pre-truth locked Selected_B300.npz instead of deriving its mixture after truth access',
        metric_arithmetic_changed=False)
    return result


def sealed_published_b300_scorer(module):
    """Use the pre-truth BC alias with the original descriptive score code."""
    if getattr(module._published_b300,'_iaos_sealed_B300_loading',False):return module._published_b300
    source=inspect.getsource(module._published_b300);tree=ast.parse(source)
    class LoadSealedBC(ast.NodeTransformer):
        changed=0
        def visit_Call(self,node):
            if ast.unparse(node.func)=='m.bc_arrays':
                if len(node.args)!=1 or ast.unparse(node.args[0])!='native':
                    raise ValueError('Unexpected original B300 BC alias call')
                self.changed+=1
                return ast.parse("_arrays(run / 'evaluacion/b300' / context / f'rep_{rep:05d}' / 'published_BC.npz')",mode='eval').body
            return self.generic_visit(node)
    change=LoadSealedBC();tree=change.visit(tree)
    if change.changed!=1:raise ValueError('Expected exactly one B300 BC prediction loading adaptation')
    namespace=dict(vars(module));exec(compile(ast.fix_missing_locations(tree),module.__file__,'exec'),namespace)
    result=namespace['_published_b300'];result._iaos_sealed_B300_loading=True
    result._iaos_loading_provenance=dict(
        historical_function_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        adapted_function_AST_sha256=hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        changed_calls=1,adaptation='Load pre-truth locked published_BC.npz instead of deriving its intervals after truth access',
        metric_arithmetic_changed=False)
    return result


def completion(oldfolder,newfolder,methods,changes=None):
    old=read(oldfolder/'completion.json')
    obj={**old,'historical_completion_sha256':sha(oldfolder/'completion.json'),
         'numerical_revision':'strict_REML_20261018',
         'historical_runtime_seconds':old.get('seconds'),
         'convergence':{m:converged(newfolder,m) for m in methods},
         'output_sha256':{p.name:sha(p) for p in newfolder.iterdir()
                          if p.is_file() and p.name!='completion.json' and p.suffix in ('.npz','.json','.csv')}}
    obj.update(world_runtime(newfolder,methods))
    if changes:obj.update(changes)
    if 'all_converged' in obj:obj['all_converged']=all(obj['convergence'].values())
    save(newfolder/'completion.json',obj)
    return obj


def native_gate(args):
    lock=read(args.native_lock)
    if lock.get('complete') is not True or lock.get('truth_read') is not False:
        raise ValueError('Complete truth-blind native correction lock required')
    if lock.get('planned')!=2780 or len(lock.get('items',[]))!=2780:
        raise ValueError('The complete registered 2780-method correction inventory is required')
    inventory=args.corrected_root/'job_inventory.json'
    if not inventory.is_file() or sha(inventory)!=lock.get('inventory_sha256'):
        raise ValueError('The native correction lock does not bind its complete job inventory')
    registered=read(inventory)['jobs']
    if len(registered)!=2780 or {job['job_id'] for job in registered}!={item['job_id'] for item in lock['items']}:
        raise ValueError('Native correction job IDs differ from the sealed registered inventory')
    if 'resolution_manifest_sha256' in lock:
        resolution=args.corrected_root/'resolution_manifest.json'
        if sha(resolution)!=lock['resolution_manifest_sha256']:
            raise ValueError('Native stage-resolution provenance changed')
        if read(resolution)['allowed_stage_roots']!=lock.get('allowed_stage_roots'):
            raise ValueError('Native stage roots differ from the sealed resolution')
    outputs=lock.get('outputs_sha256')
    if not outputs:
        raise ValueError('Native correction lock must bind outputs_sha256 by corrected-root-relative path')
    if len({item['job_id'] for item in lock['items']})!=2780:
        raise ValueError('Native correction lock duplicates or omits registered methods')
    bindings=dict(outputs)
    for item in lock['items']:
        name=item['completion_relative_path'];digest=item['completion_sha256']
        if name in bindings and bindings[name]!=digest:
            raise ValueError('Inconsistent native method-completion hash')
        bindings[name]=digest
    allowed=[args.corrected_root]+[Path(path).resolve() for path in lock.get('allowed_stage_roots',[])]
    for name,digest in bindings.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe native output reference')
        path=args.corrected_root/name
        if not path.parent.resolve().is_relative_to(args.corrected_root):
            raise ValueError('Native overlay must link individual files, never mutable fit directories')
        if not any(path.resolve().is_relative_to(root) for root in allowed):
            raise ValueError('Native link does not resolve to a declared immutable correction stage')
        if sha(path)!=digest:raise ValueError('Native corrected output changed: '+name)
    return lock


def prepare(args):
    native_gate(args)
    configure_archived_runtime(args)
    destination=args.corrected_root
    if (destination/'scoring_prediction_lock.json').exists():
        return verify_scoring_lock(args)
    module=imports(args.repo)
    run,old=destination/'sikov',args.old_root
    for relative in ('evaluacion/inputs','desarrollo/inputs','sensibilidad_sampled_graph/inputs','datos',
                     'real_preparation_receipt.json','protocol_freeze.json'):
        link(old/relative,run/relative)
    # Principal and observed-graph worlds reuse the exact ICC draws.
    worlds=[]
    for context in CONTEXTS:
        for rep in range(30000,30064):
            worlds.append((f'evaluacion/fits/{context}/rep_{rep:05d}',
                           f'evaluacion/published_sar/{context}/rep_{rep:05d}'))
    for rep in range(30000,30064):
        worlds.append((f'sensibilidad_sampled_graph/fits/rep_{rep:05d}/original',
                       f'sensibilidad_sampled_graph/fits/rep_{rep:05d}/published'))
    for first,second in worlds:
        for relative,methods in ((first,ORIGINAL),(second,PUBLISHED)):
            folder=run/relative;previous=old/relative
            reused_folder(previous,folder)
            if methods==ORIGINAL:
                for method in ('Sikov_FH','Sikov_SAR'):boundary_alias(folder,method)
                mixture(folder,'Selected',.75,module)
                obj=completion(previous,folder,methods)
            else:
                boundary_alias(folder,'Sikov_SAR_Published')
                obj=completion(previous,folder,methods,{'original_completion_sha256':sha(run/first/'completion.json')})
        if first.startswith('sensibilidad'):
            parent=(run/first).parent;prior=(old/first).parent
            combined={**read(parent/'original/completion.json')['convergence'],
                      **read(parent/'published/completion.json')['convergence']}
            base=read(prior/'completion.json')
            childtimes=[read(parent/part/'completion.json') for part in ('original','published')]
            save(parent/'completion.json',{**base,'convergence':combined,'numerical_revision':'strict_REML_20261018',
                'historical_runtime_seconds':base.get('seconds'),
                'seconds':sum(row['seconds'] for row in childtimes),
                'seconds_scope':childtimes[0]['seconds_scope'],
                'runtime_composition':{key:sum(row['runtime_composition'][key] for row in childtimes)
                    for key in childtimes[0]['runtime_composition']},
                'historical_completion_sha256':sha(prior/'completion.json'),
                'output_sha256':{f'{part}/completion.json':sha(parent/part/'completion.json') for part in ('original','published')}})
    print(json.dumps({'stage':'prepared_principal_and_graph_predictions','populations':320,'truth_read':False}),flush=True)
    # Real predictions are reconstructed from corrected arrays, using fixed IDs.
    for scope in ('temporal','territorial'):
        prepared=run/'datos'/('real_2019_full' if scope=='temporal' else 'real_2024_training')
        ids=pd.read_csv(prepared/'area_frame.csv',dtype={'ubigeo':str}).ubigeo.tolist()
        original=run/'evaluacion'/('real_'+scope)
        for relative,methods in ((f'evaluacion/real_{scope}',ORIGINAL),
                                 (f'evaluacion/published_sar/real_{scope}',PUBLISHED)):
            folder=run/relative;previous=old/relative;reused_folder(previous,folder)
            for method in (('Sikov_FH','Sikov_SAR') if methods==ORIGINAL else ('Sikov_SAR_Published',)):
                boundary_alias(folder,method)
            if methods==ORIGINAL:mixture(folder,'Selected',.75,module)
            rows=[]
            for method in methods:
                pred=arrays(folder/(method+'.npz'))
                assert pred['mean'].shape==(len(ids),2)
                for i,code in enumerate(ids):
                    rows.append({'ubigeo':code,'method':method,**{
                        prefix+endpoint:pred[key][i,j] for j,endpoint in enumerate(('anemia','stunting'))
                        for prefix,key in (('p_','mean'),('lo95_','lower'),('hi95_','upper'))}})
            pd.DataFrame(rows).to_csv(folder/'predictions.csv',index=False)
            changes={'predictions_sha256':sha(folder/'predictions.csv')}
            if methods==PUBLISHED:
                changes.update(original_completion_sha256=sha(original/'completion.json'),
                               original_predictions_sha256=sha(original/'predictions.csv'))
            completion(previous,folder,methods,changes)
    print(json.dumps({'stage':'prepared_real_predictions','training_scopes':2,'truth_read':False}),flush=True)
    # Recreate all original development mixtures without reselecting evaluation weights.
    dev=run/'desarrollo/recreated_native';olddev=old/'desarrollo/recreated_native'
    for filename in ('development_plan.json',):link(olddev/filename,dev/filename)
    for context in CONTEXTS:
        for rep in range(8):
            relative=f'fit_results/{context}/rep_{rep:05d}'
            folder=dev/relative;previous=olddev/relative
            for method in DEVELOPMENT:
                if (previous/method).is_dir():reused_folder(previous/method,folder/method)
            for method,weight in (('ICC_SAR_w25',.25),('ICC_SAR_w50',.5),('ICC_SAR_w75',.75)):
                mixture(folder,method,weight,module,True)
            success={method:[converged(folder,method,True)]*2 for method in DEVELOPMENT}
            save(folder/'completion.json',{'complete':True,'phase':'development','all_method_success':success,
                 'reference_B':100,'ICC_reuse_verbatim':True,'numerical_revision':'strict_REML_20261018',
                 **world_runtime(folder,DEVELOPMENT,True)})
    print(json.dumps({'stage':'prepared_Laplace_development_predictions','populations':32,'truth_read':False}),flush=True)
    self_numerical=args.corrected_root/'numerical_cohorts'
    for cohort,reps in (('sparse57',range(50000,50064)),('development_nuts',range(8))):
        methods=ORIGINAL[:2]+('Sikov_SAR_Published','ICC','Selected') if cohort=='sparse57' else DEVELOPMENT
        if cohort=='sparse57':link(args.old_numerical_root/'sparse57/evaluacion/inputs',self_numerical/'sparse57/evaluacion/inputs')
        for context in CONTEXTS:
            for rep in reps:
                relative=f'{cohort}/fits/{context}/rep_{rep:05d}'
                folder=self_numerical/relative;previous=args.old_numerical_root/relative
                reused_folder(previous,folder)
                if cohort=='development_nuts':
                    native=dev/f'fit_results/{context}/rep_{rep:05d}'
                    for method in ('Sikov_FH','Sikov_SAR'):
                        prediction=native/method/'predictions.npz';status=read(native/method/'status.json')
                        target=folder/(method+'.npz')
                        if target.is_symlink():target.unlink()
                        shutil.copy2(prediction,target)
                        targetstatus=folder/(method+'_status.json')
                        if targetstatus.is_symlink():targetstatus.unlink()
                        timing=method_runtime(native,method,True)
                        timing['reused_in_cohort']='development_nuts'
                        timing['scope']='corrected_development_reference_reused_for_NUTS_integration_no_additional_Gaussian_fit'
                        save(targetstatus,{'converged':converged(native,method,True),
                            'fits':status.get('status',status.get('fits')),'reused_exactly':True,
                            'corrected_reference_source_sha256':sha(prediction),
                            'runtime_composition':timing})
                    for method,weight in (('ICC_SAR_w25',.25),('ICC_SAR_w50',.5),('ICC_SAR_w75',.75)):
                        mixture(folder,method,weight,module)
                else:
                    mixture(folder,'Selected',.75,module)
                oldq=read(previous/'completion.json')
                for method in methods:
                    method_receipt=folder/(method+'_completion.json')
                    original_receipt=previous/(method+'_completion.json')
                    old_method=read(original_receipt)
                    timing=method_runtime(folder,method)
                    if method_receipt.is_symlink():method_receipt.unlink()
                    save(method_receipt,{**old_method,
                        'historical_completion_sha256':sha(original_receipt),
                        'historical_runtime_seconds':old_method.get('seconds'),
                        'seconds':timing['seconds'],'runtime_composition':timing,
                        'numerical_revision':'strict_REML_20261018',
                        'converged':converged(folder,method),
                        'outputs':{name:sha(folder/name) for name in (method+'.npz',method+'_status.json')}})
                obj={**oldq,'historical_completion_sha256':sha(previous/'completion.json'),
                     'historical_runtime_seconds':oldq.get('seconds'),
                     'numerical_revision':'strict_REML_20261018',
                     'convergence':{method:converged(folder,method) for method in methods},
                     'outputs':{p.name:sha(p) for p in folder.iterdir() if p.is_file()
                                and p.name!='completion.json' and p.suffix in ('.npz','.json')}}
                obj.update(world_runtime(folder,methods))
                save(folder/'completion.json',obj)
    print(json.dumps({'stage':'prepared_sparse57_and_NUTS_development_predictions','populations':288,'truth_read':False}),flush=True)
    # B300 metadata is used only to route sealed new arrays; retain original RNG receipts.
    for context in CONTEXTS:
        for rep in (30000,30001):
            relative=f'evaluacion/b300/{context}/rep_{rep:05d}'
            folder=run/relative;previous=old/relative;reused_folder(previous,folder)
            for method in ('full_graph','published'):boundary_alias(folder,method)
            seal_b300_mixture(folder,run/f'evaluacion/fits/{context}/rep_{rep:05d}',module)
            oldq=read(previous/'completion.json')
            obj={**oldq,'historical_completion_sha256':sha(previous/'completion.json'),
                 'historical_runtime_seconds':oldq.get('seconds'),
                 'convergence':{method:converged(folder,method) for method in ('full_graph','published')},
                 'numerical_revision':'strict_REML_20261018','output_sha256':{
                     p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='completion.json'}}
            current_original=run/f'evaluacion/fits/{context}/rep_{rep:05d}/completion.json'
            current_published=run/f'evaluacion/published_sar/{context}/rep_{rep:05d}/completion.json'
            obj['original_completion_sha256']=sha(current_original)
            obj['published_completion_sha256']=sha(current_published)
            obj['main_prediction_completion_bindings']=[dict(path=str(run/f'evaluacion/{family}/{c}/rep_{r:05d}/completion.json'),
                sha256=sha(run/f'evaluacion/{family}/{c}/rep_{r:05d}/completion.json'))
                for c in CONTEXTS for r in range(30000,30064) for family in ('fits','published_sar')]
            obj.update(world_runtime(folder,('full_graph','published','Selected_B300')))
            save(folder/'completion.json',obj)
    print(json.dumps({'stage':'prepared_B300_mixture_and_BC_predictions','subset_populations':8,'truth_read':False}),flush=True)
    runtime_accounting(args)
    bindings={}
    # Include array inputs, masks, fitted predictions, status and reused draws;
    # evaluation_only is deliberately absent from this truth-blind lock.
    for subtree in (run/'evaluacion',run/'desarrollo',run/'sensibilidad_sampled_graph',self_numerical):
        for path in subtree.rglob('*'):
            if not path.is_file() or 'evaluation_only' in path.parts:continue
            if path.suffix in ('.npz','.json','.csv') and ('fits' in path.parts or 'fit_results' in path.parts or
                'model_inputs' in path.parts or 'b300' in path.parts or any(p.startswith('real_') for p in path.parts)):
                bindings[str(path.relative_to(destination))]=sha(path)
    # Directory symlinks are not traversed by pathlib.rglob: explicitly bind inputs.
    sources=[run/'evaluacion/inputs'/c/f'rep_{r:05d}/model_inputs/inputs.npz' for c in CONTEXTS for r in range(30000,30064)]
    sources += [run/'desarrollo/inputs'/c/f'rep_{r:05d}/model_inputs/inputs.npz' for c in CONTEXTS for r in range(8)]
    sources += [run/'sensibilidad_sampled_graph/inputs'/f'rep_{r:05d}/model_inputs/inputs.npz' for r in range(30000,30064)]
    sources += [self_numerical/'sparse57/evaluacion/inputs'/c/f'rep_{r:05d}/model_inputs/inputs.npz' for c in CONTEXTS for r in range(50000,50064)]
    for path in sources:
        for artifact in (path,path.parent/'sample.csv',path.parent.parent/'manifest.json'):
            bindings[str(artifact.relative_to(destination))]=sha(artifact)
    bindings['sikov/desarrollo/recreated_native/development_plan.json']=sha(dev/'development_plan.json')
    for filename in ('protocol_freeze.json','real_preparation_receipt.json'):
        bindings['sikov/'+filename]=sha(run/filename)
    for filename in ('runtime_corrected_native_summary.csv','runtime_accounting.json'):
        bindings[filename]=sha(destination/filename)
    for relative in ('datos/real_2019_full/fit_inputs.npz','datos/real_2024_training/fit_inputs.npz'):
        bindings['sikov/'+relative]=sha(run/relative)
        for name in ('area_frame.csv','preparation_manifest.json'):
            path=(run/relative).parent/name
            bindings[str(path.relative_to(destination))]=sha(path)
    failures=[]
    for path in (destination/'sikov',destination/'numerical_cohorts'):
        for receipt in path.rglob('completion.json'):
            if receipt.is_symlink():continue
            record=read(receipt)
            for method,accepted in record.get('convergence',{}).items():
                if not accepted:failures.append({'completion':str(receipt.relative_to(destination)),'method':method})
    result={'status':'LOCKED_ALL_CORRECTED_AND_REUSED_SCORING_PREDICTIONS_BEFORE_TRUTH',
        'complete':True,'truth_read':False,'locked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'native_lock_sha256':sha(args.native_lock),'source_sha256':sha(__file__),
        'source_snapshot_relative_path':'scoring_preparation_adapter.py',
        'ICC_reused_without_refitting':True,'fixed_primary_ICC_weight':.75,
        'B300_selected_distributions_derived_and_sealed_before_truth':8,
        'B300_published_BC_distributions_derived_and_sealed_before_truth':8,
        'generation_manifests_sealed':len(sources),'development_selection_plan_sealed':True,
        'convergence_failures_retained':failures,
        'cohorts':{'principal':256,'development_Laplace':32,'development_NUTS':32,'sparse57':256,
                   'sampled_graph':64,'B300_subset':8,'real_scopes':2},'outputs_sha256':bindings}
    snapshot=destination/'scoring_preparation_adapter.py'
    if snapshot.exists() and sha(snapshot)!=sha(__file__):
        archive=destination/'preparation_adapter_history'/('adapter_'+sha(snapshot)+'.py')
        archive.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(snapshot,archive)
    shutil.copy2(__file__,snapshot)
    save(destination/'scoring_prediction_lock.json',result)
    print(json.dumps({'status':result['status'],'files_bound':len(bindings)}),flush=True)
    return result


def verify_scoring_lock(args):
    native_gate(args)
    lock=read(args.corrected_root/'scoring_prediction_lock.json')
    if lock.get('complete') is not True or lock.get('truth_read') is not False:
        raise ValueError('All corrected and reused predictions must be locked before truth scoring')
    if lock['native_lock_sha256']!=sha(args.native_lock):raise ValueError('Native correction lock changed')
    if sha(args.corrected_root/lock['source_snapshot_relative_path'])!=lock['source_sha256']:
        raise ValueError('Prediction-preparation source snapshot changed')
    if lock.get('generation_manifests_sealed')!=608 or lock.get('development_selection_plan_sealed') is not True or \
            'sikov/desarrollo/recreated_native/development_plan.json' not in lock['outputs_sha256']:
        raise ValueError('All generation contracts and the unchanged development selection plan must be sealed')
    b300={f'sikov/evaluacion/b300/{c}/rep_{r:05d}/Selected_B300{suffix}'
          for c in CONTEXTS for r in (30000,30001) for suffix in ('.npz','_status.json')}
    if lock.get('B300_selected_distributions_derived_and_sealed_before_truth')!=8 or not b300<=set(lock['outputs_sha256']):
        raise ValueError('All eight B300 mixture distributions must be derived and sealed before truth access')
    bc={f'sikov/evaluacion/b300/{c}/rep_{r:05d}/published_BC{suffix}'
        for c in CONTEXTS for r in (30000,30001) for suffix in ('.npz','_status.json')}
    if lock.get('B300_published_BC_distributions_derived_and_sealed_before_truth')!=8 or not bc<=set(lock['outputs_sha256']):
        raise ValueError('All eight B300 published BC distributions must be derived and sealed before truth access')
    for name,digest in lock['outputs_sha256'].items():
        if sha(args.corrected_root/name)!=digest:raise ValueError('A locked scoring input changed: '+name)
    return lock


def context(args):
    imports(args.repo)
    worker=importlib.import_module('sikov_worker')
    return worker.initialize(args.repo/'src/estudio_publicacion',args.reference_root,
                             args.corrected_root/'sikov',args.exe,args.cmdstan)


def compatibility_gate(args,ctx):
    """New complete numerical-revision gate; historical gate is not relabelled."""
    lock=verify_scoring_lock(args)
    result={'status':'PASS_CORRECTED_NUMERICAL_REVISION_PREDICTIONS_BEFORE_SCORING',
        'approved_for_scoring':True,'protocol_sha256':ctx['protocol_sha256'],
        'addendum_sha256':ctx['addendum_sha256'],'selected_recipe':{'family':'ICC_SAR_mixture','icc_weight':.75},
        'principal_worlds':256,'sampled_graph_worlds':64,'B300_worlds':8,
        'actual_scopes':['temporal','territorial'],'convergence_failures_retained':lock['convergence_failures_retained'],
        'numerical_revision_lock_sha256':sha(args.corrected_root/'scoring_prediction_lock.json'),
        'historical_gate_or_selection_reproduction_claimed':False,
        'fresh_truth_read_before_lock':False,'heldout_outcomes_used':False}
    path=ctx['run']/'validacion/sikov_fresh_prediction_lock.json'
    if path.exists() and read(path)!=result:raise ValueError('Different compatibility lock already exists')
    save(path,result)
    return result


def score_development(args,ctx):
    run=ctx['run'];module=ctx['modules']['improve_development'];selection=ctx['modules']['improve_select']
    folder=run/'desarrollo/recreated_native';records=[]
    for c in CONTEXTS:
        for rep in range(8):
            inp=run/'desarrollo/inputs'/c/f'rep_{rep:05d}'
            fit=folder/'fit_results'/c/f'rep_{rep:05d}'
            d=arrays(inp/'model_inputs/inputs.npz');truth=evaluation_truth(inp)
            for method in DEVELOPMENT:
                timing=method_runtime(fit,method,True)
                rows=timed_rows(module.scores(truth,d,arrays(fit/method/'predictions.npz'),method,c,rep,
                                   [converged(fit,method,True)]*2,timing['seconds']),timing)
                for row in rows:row['protocol']='same14_native_naive_MSE_B100_Laplace_ICC_development'
                records.extend(rows)
    frame=pd.DataFrame(records);frame.to_csv(folder/'replications.csv',index=False)
    plan=read(folder/'development_plan.json')
    chosen,ranking,held=selection.select(frame.loc[frame.method.isin(plan['candidate_methods'])].copy(),plan)
    ranking.to_csv(folder/'candidate_ranking.csv',index=False)
    held.to_csv(folder/'LOPO_selection_rule_metrics.csv',index=False)
    # This is a sensitivity report, not a new choice for the evaluation mixture.
    chosen['frozen_primary_recipe_changed']=False
    save(folder/'corrected_development_selection.json',chosen)
    save(folder/'selected_recipe.json',{'selected_recipe':{'family':'ICC_SAR_mixture','icc_weight':.75},
         'original_recipe_retained':True,'corrected_grid_winner':chosen['selected_method']})
    save(folder/'completion.json',{'complete':True,'completed_worlds':32,'expected_worlds':32,
         'numerical_revision':'strict_REML_20261018','weights_reselected_for_primary':False})


def score_new_cohorts(args,ctx):
    scorer=ctx['modules']['improve_development'];benchmark=importlib.import_module('benchmark_metrics')
    root=args.corrected_root/'numerical_cohorts'
    for cohort,reps,methods in (('sparse57',range(50000,50064),('Sikov_FH','Sikov_SAR','Sikov_SAR_Published','ICC','Selected')),
                               ('development_nuts',range(8),DEVELOPMENT)):
        rows=[];diagnostics=[]
        for c in CONTEXTS:
            for rep in reps:
                fit=root/cohort/'fits'/c/f'rep_{rep:05d}';q=read(fit/'completion.json')
                inp=(root/'sparse57/evaluacion/inputs' if cohort=='sparse57' else ctx['run']/'desarrollo/inputs')/c/f'rep_{rep:05d}'
                d=arrays(inp/'model_inputs/inputs.npz');truth=evaluation_truth(inp)
                for method in methods:
                    timing=q['method_runtime'][method]
                    result=timed_rows(scorer.scores(truth,d,arrays(fit/(method+'.npz')),method,c,rep,
                                         [q['convergence'][method]]*2,timing['seconds']),timing)
                    for row in result:
                        row.update(phase=cohort,protocol='new_exploratory_revision_20261015_v1')
                        row['rmse_population']=float(np.sqrt(row['mse']))
                        row['coverage_distance_population']=abs(row['coverage']-.95)
                    rows.extend(result)
                    status=read(fit/(method+'_status.json'))
                    for j,end in enumerate(status.get('endpoints',[])):
                        for trial in end['sampler_trials']:
                            diagnostics.append(dict(cohort=cohort,context=c,replication=rep,method=method,
                                endpoint=('anemia','stunting')[j],**trial,min_E_BFMI=min(trial['E_BFMI']),
                                beta_shape_guard_cells=end['beta_shape_guard_cells'],NUTS_final_success=end['success']))
        out=args.output/'numerical_cohorts'/cohort;out.mkdir(parents=True,exist_ok=True)
        frame=pd.DataFrame(rows);frame.to_csv(out/'replications.csv',index=False)
        timing_summary(frame,benchmark.summarize).to_csv(out/'summary.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(out/'NUTS_diagnostics_all_attempts.csv',index=False)
        save(out/'scoring_receipt.json',{'cohort':cohort,'complete':True,'worlds':len(CONTEXTS)*len(reps),
            'all_models_converged':bool(frame.converged.all()),
            'prediction_lock_sha256':sha(args.corrected_root/'scoring_prediction_lock.json'),
            'unit':'independent population','outputs':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='scoring_receipt.json'}})


def score(args):
    verify_scoring_lock(args)
    configure_archived_runtime(args)
    if (args.output/'scoring_completion.json').exists():raise FileExistsError('Preserve completed corrected scoring')
    args.output.mkdir(parents=True,exist_ok=True)
    ctx=context(args)
    main=importlib.import_module('sikov_score')
    figures=importlib.import_module('sikov_figures')
    new_gate=lambda *_a,**_k:compatibility_gate(args,ctx)
    main.pre_truth_gate=new_gate;figures.pre_truth_gate=new_gate
    main._full_b300=sealed_b300_scorer(main)
    main._published_b300=sealed_published_b300_scorer(main)
    new_gate()
    score_development(args,ctx)
    # The real scorer imports pre_truth_gate from the already patched module.
    real=importlib.import_module('real_score')
    real_args=argparse.Namespace(source_root=args.repo/'src/estudio_publicacion',
        reference_root=args.reference_root,run_root=ctx['run'],exe=args.exe,cmdstan=args.cmdstan)
    real.score(real_args)
    # The original scoring arithmetic receives method-specific observed times;
    # the historical scorer otherwise assigns one old whole-world time to every
    # method. Separate current REML, archived reused ICC and new CDF components.
    metrics=importlib.import_module('benchmark_metrics')
    scorer=ctx['modules']['improve_development'];original_scores=scorer.scores
    original_summary=metrics.summarize
    def principal_timed_scores(truth,d,pred,method,c,rep,success,_world_seconds):
        family='published_sar' if method in PUBLISHED else 'fits'
        folder=ctx['run']/f'evaluacion/{family}'/c/f'rep_{rep:05d}'
        timing=method_runtime(folder,method)
        return timed_rows(original_scores(truth,d,pred,method,c,rep,success,timing['seconds']),timing)
    scorer.scores=principal_timed_scores
    metrics.summarize=lambda frame:timing_summary(frame,original_summary)
    try:main.score_all(ctx)
    finally:
        scorer.scores=original_scores;metrics.summarize=original_summary
    figures.generate(ctx)
    score_new_cohorts(args,ctx)
    save(args.output/'scoring_completion.json',{'status':'COMPLETE_CORRECTED_NUMERICAL_SCORING',
        'source_sha256':sha(__file__),'prediction_lock_sha256':sha(args.corrected_root/'scoring_prediction_lock.json'),
        'primary_weight_unchanged':.75,'original_Holm_families_unchanged':['Holm3','Holm12'],
        'B300_mixture_prediction_loading':main._full_b300._iaos_loading_provenance,
        'B300_BC_prediction_loading':main._published_b300._iaos_loading_provenance,
        'completed_utc':dt.datetime.now(dt.timezone.utc).isoformat()})


def export_aggregates(args):
    """Use the public export allowlist, with the new correction lock as gate."""
    exporter=importlib.import_module('export_public_fresh_results')
    run=args.corrected_root/'sikov';target=args.layer/'data/sikov_2026-10-10'
    roots=(('CORRECTED_RUN',args.corrected_root),('PUBLIC_SOURCE',args.repo/'src/estudio_publicacion'),
           ('ORIGINAL_RUN',args.old_root),('ORIGINAL_NUMERICAL',args.old_numerical_root),
           ('REFERENCE',args.reference_root))
    files=[]
    for label,(certfile,key,status,allowlist) in exporter.CERTIFICATES.items():
        certificate=read(run/certfile)
        if certificate['status']!=status:raise ValueError('Unclosed aggregate producer: '+label)
        if set(certificate[key])!=set(allowlist):raise ValueError('Aggregate output universe differs: '+label)
        for relative,kind in allowlist.items():
            source=run/relative
            if source.is_symlink() or sha(source)!=certificate[key][relative]:
                raise ValueError('Corrected aggregate changed before export: '+relative)
            payload=source.read_bytes()
            if kind=='csv':exporter.validate_csv(payload,relative)
            elif kind=='json':
                value=exporter.sanitize(read(source),roots);exporter.no_private_payload(value)
                payload=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
            destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(payload)
            files.append({'path':str(destination.relative_to(args.layer)),
                          'private_source_sha256':sha(source),'public_sha256':sha(destination)})
        public=exporter.sanitize(certificate,roots);exporter.no_private_payload(public)
        save(args.layer/'results/reml_numerical_certificates'/(label+'.json'),public)
    save(args.layer/'results/reml_numerical_certificates/export_manifest.json',{
        'status':'CORRECTED_AGGREGATES_EXPORTED_ALLOWLIST_ONLY','files':files,
        'prediction_lock_sha256':sha(args.corrected_root/'scoring_prediction_lock.json'),
        'raw_records_draws_chains_or_native_inputs_exported':False})


def corrected_figures(args):
    """Regenerate visible labels from corrected numeric data, never old pixels."""
    spec=importlib.util.spec_from_file_location('iaos_corrected_figure_helpers',
        args.repo/'revision_20261015/code/terminology_20261018/render_proposal_figures.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    base=args.layer/'data/sikov_2026-10-10';scripts=args.repo/'src/estudio_publicacion/sikov_2026-10-10/scripts'
    inc=args.layer/'results/incremental_icc'
    figures=[]
    def render(kind,source,namespace,replacements,destination):
        if kind=='interval':
            parsed=ast.parse(source.read_text())
            main=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='main')
            first=next(i for i,n in enumerate(main.body) if helper.assignment(n,'raw'))
            last=next(i for i,n in enumerate(main.body) if helper.assignment(n,'report'))
            body=[copy.deepcopy(n) for n in main.body[first:last] if not (
                isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and
                ast.unparse(n.value.func) in ('fig.savefig','plt.close'))]
            before=ast.Module(body=body,type_ignores=[])
        else:before=helper.plot_nodes(source,kind)
        after=copy.deepcopy(before)
        transformer=helper.ReplaceText(replacements);after=transformer.visit(after)
        fingerprints=[]
        for updated,nodes in ((False,before),(True,after)):
            plt.rcdefaults()
            labels=('Sikov FH','Full-graph SAR','Published-scope SAR','ICC beta',
                    'Proposed mixture' if updated else 'Frozen 75/25 mixture')
            env=dict(np=np,pd=pd,plt=plt,Path=Path,ROOT=args.repo,out=destination.parent,
                     CONTEXTS=CONTEXTS,NAMES=helper.NAMES,METHODS=helper.METHODS,
                     COLORS=helper.COLORS,Z=helper.Z,LABELS=labels)
            env.update(namespace)
            exec(compile(ast.fix_missing_locations(nodes),str(source),'exec'),env)
            fig=env['fig'];fingerprints.append(helper.artist_fingerprint(fig))
            if updated:
                destination.parent.mkdir(parents=True,exist_ok=True)
                fig.savefig(destination,dpi=250 if kind=='incremental' else 300,
                            **({'bbox_inches':'tight'} if kind=='incremental' else {}))
                fig.savefig(destination.with_suffix('.svg'))
                if kind!='incremental':fig.savefig(destination.with_suffix('.pdf'))
            plt.close(fig)
        if fingerprints[0]!=fingerprints[1]:raise ValueError('Visible terminology changed plotted numeric artists')
        figures.append({'path':str(destination.relative_to(args.layer)),'sha256':sha(destination),
            'source':str(source.relative_to(args.repo)),'source_sha256':sha(source),
            'corrected_numeric_artists_sha256':fingerprints[1]['sha256'],
            'models_numerically_corrected':True,'historical_pixels_reused':False,
            'label_change_numeric_artist_identity_verified':True})
    render('principal',scripts/'comparison_figures.py',{'source':base/'resultados/published_sar_comparison'},
           {},base/'resultados/figuras/matched_rmse_coverage.png')
    render('incremental',args.repo/'revision_20261015/code/incremental_icc.py',
           {'summary':pd.read_csv(inc/'incremental_summary.csv'),'out':inc},
           {'Selected − ICC':'Proposed mixture − ICC'},inc/'Selected_vs_ICC_incremental.png')
    render('interval',scripts/'comparison_figures.py',{
        'source':base/'resultados/published_sar_comparison',
        'REFERENCES':('Sikov_FH','Sikov_SAR','Sikov_SAR_Published')},
        {'Frozen improvement on 256 fresh finite populations':'Proposed mixture on 256 finite populations',
         'Difference in balanced interval score (lower favours improvement)':
         'Difference in balanced interval score (lower favours proposed mixture)'},
        base/'resultados/figuras/matched_interval_score.png')
    area=args.repo/'src/estudio_publicacion/sikov_2026-10-10/datos/real_2019_full/area_frame.csv'
    areas=pd.read_csv(area,dtype={'ubigeo':str})
    district=base/'resultados/aplicacion2019/matched_application2019_districts.csv'
    data=pd.read_csv(district,dtype={'ubigeo':str})
    mask=pd.read_csv(args.repo/'revision_20261015/results/terminology_20261018/district_availability_for_map.csv',
                     dtype={'ubigeo':str}).set_index('ubigeo').loc[areas.ubigeo]
    ordered={method:frame.set_index('ubigeo').loc[areas.ubigeo] for method,frame in data.groupby('method')}
    render('map',scripts/'comparison_application_figures.py',{
        'ordered':ordered,'ENDPOINTS':('anemia','stunting'),'lon':areas.longitude.to_numpy(),
        'lat':areas.latitude.to_numpy(),'observed':{e:mask[e+'_observed'].to_numpy(bool) for e in ('anemia','stunting')}},
        {': selected mixture':': Proposed mixture',': mixture minus sampled-graph SAR':': Proposed mixture minus sampled-graph SAR'},
        base/'resultados/aplicacion2019/matched_application2019_pointmaps.png')
    save(args.layer/'results/reml_numerical_certificates/corrected_figures.json',{
        'status':'CORRECTED_FIGURES_REGENERATED_FROM_CORRECTED_AGGREGATES','figures':figures,
        'visual_review_complete':False,'numeric_revision_not_presentation_only':True})


def incremental_layer(args):
    inc=importlib.import_module('incremental_icc');ext=importlib.import_module('cohort_extensions')
    base=args.layer/'results';out=base/'incremental_icc';out.mkdir(parents=True,exist_ok=True)
    devout=base/'devgrid';devout.mkdir(parents=True,exist_ok=True)
    auditout=base/'audit';auditout.mkdir(parents=True,exist_ok=True)
    # A private path view lets unchanged analysis arithmetic address the new
    # public matrices while source/historical files keep their real identities.
    proxy=args.corrected_root/'aggregate_analysis_view'
    link(args.repo/'src',proxy/'src')
    link(args.layer/'data/sikov_2026-10-10',proxy/'results_fresh/modern')
    link(args.layer/'results',proxy/'revision_20261015/results')
    link(args.repo/'revision_20261015/protocols',proxy/'revision_20261015/protocols')
    frame,provenance=inc.world_scores(args.corrected_root/'sikov',
        args.repo/'revision_20261015/protocols/incremental_icc_exploratory_protocol.json')
    inc.score_csv_check(frame,proxy)
    frame.to_csv(out/'per_population_Selected_ICC_metrics.csv',index=False)
    summary=inc.summarize_increment(frame);summary.to_csv(out/'incremental_summary.csv',index=False)
    audit=inc.audit_original_families(proxy,auditout)
    dev=inc.development_audit(args.corrected_root/'sikov',args.repo,devout)
    dev.update(numerical_revision='strict_REML_20261018',
               status='PASS_CORRECTED_LAPLACE_GRID_AUDIT',original_primary_weight_changed=False,
               NUTS_replay_required=False,NUTS_integration_sensitivity_available=True,
               ICC_NUTS_fits_reused_without_refitting=True)
    save(devout/'Laplace_development_grid_audit.json',dev)
    (devout/'NUTS_development_replay_contract.md').write_text(
        '# Completed NUTS integration sensitivity after numerical correction\n\n'
        'All 32 original populations and five eligible candidates are retained. '
        'The completed ICC and poverty-stratum ICC NUTS predictions and conditional draws are reused verbatim. '
        'The reconstructed SAR reference and its B100 uncertainty use the numerical correction with the same models and seeds. '
        'Mixtures are rebuilt with the unchanged CDF algorithm and weights 0.25, 0.50 and 0.75. '
        'Corrected candidate ranks and LOPO margins are supplied separately. '
        'No evaluation outcome reselects the fixed primary 75/25 recipe.\n')
    inc.builder_exports(summary,dev,audit,out)
    # Replace old directional prose with values calculated from this revision.
    narrative=read(out/'narrative.json');cl=narrative['raw_unrounded_claims']
    v=cl['global_interval_score'];r=cl['global_RMSE_sqrt_mean_MSE'];w=cl['global_width'];cv=cl['global_coverage']
    narrative['English_main_ready']=(f"After the same-model numerical correction, the exploratory paired comparison on the same 256 populations gives IS95 change {v['relative_difference_percent']:+.2f}% and RMSE change {r['relative_difference_percent']:+.2f}% relative to ICC. "
        f"The IS95 difference is {v['difference_selected_minus_icc']:+.6f} (MCSE {v['difference_mcse']:.6f}); its descriptive 95% Monte Carlo interval is [{v['descriptive95_low']:+.6f}, {v['descriptive95_high']:+.6f}]. "
        f"Mean width changes by {w['relative_difference_percent']:+.2f}%; coverage is {100*cv['selected']:.2f}% for the proposed mixture and {100*cv['icc']:.2f}% for ICC. These results remain exploratory and do not change the original 75/25 weights or the Holm family definitions.")
    narrative['English_context_ready']='Context-specific endpoint-balanced IS95 differences (proposed mixture minus ICC) are '+', '.join(
        f"{DISPLAY.get(row['scope'],row['scope'])}: {row['difference_selected_minus_icc']:+.6f} [{row['descriptive95_low']:+.6f}, {row['descriptive95_high']:+.6f}]"
        for row in cl['context_interval_score'])+'. These Monte Carlo intervals are descriptive and unadjusted.'
    narrative['English_development_ready']=(f"The corrected Laplace grid ranks {DISPLAY.get(dev['selected_method'],dev['selected_method'])} first and {DISPLAY.get(dev['second_method'],dev['second_method'])} second, with IS95 gap {dev['best_second_absolute_margin']:.6f}. "
        f"LOPO winner counts are {counts_text(dev['LOPO_counts'])}; the minimum training margin is {dev['LOPO_min_margin']:.8f}. The original 75/25 evaluation strategy remains fixed regardless of this numerical sensitivity.")
    narrative['numerical_revision']='strict_REML_20261018';save(out/'narrative.json',narrative)
    editorial=read(out/'editorial_contract.json')
    editorial['main_paragraphs']=[narrative['English_main_ready'],narrative['English_context_ready']]
    editorial['supplement_paragraphs'][2]=narrative['English_context_ready']
    editorial['supplement_paragraphs'][3]=(f"Independent recalculation verifies the unchanged three-member Holm3 and twelve-member Holm12 family structures from corrected primitive matrices. Maximum arithmetic errors are {audit['Holm3_max_absolute_error']:.3g} and {audit['Holm12_max_absolute_error']:.3g}. Their corrected numerical estimates and adjusted p values replace prior values; historical outputs remain preserved.")
    editorial['numerical_revision']='strict_REML_20261018';save(out/'editorial_contract.json',editorial)
    provenance.update(status='PASS_CORRECTED_EXPLORATORY_INCREMENTAL_ICC_ANALYSIS',
        numerical_revision='strict_REML_20261018',prediction_lock_sha256=sha(args.corrected_root/'scoring_prediction_lock.json'),
        analysis_code_sha256=sha(args.repo/'revision_20261015/code/incremental_icc.py'),adapter_sha256=sha(__file__),
        no_new_NUTS_or_ICC_fits=True,primary_weight_changed=False,self_tests=inc.self_test())
    save(out/'incremental_provenance.json',provenance)
    sparse=base/'sparse57_analysis';sparse.mkdir(parents=True,exist_ok=True)
    ext.sparse57(args.output/'numerical_cohorts/sparse57/replications.csv',sparse)
    nuts=base/'devgrid_nuts';nuts.mkdir(parents=True,exist_ok=True)
    ext.nuts_development(args.output/'numerical_cohorts/development_nuts/replications.csv',nuts,devout)
    nuts_audit=read(nuts/'NUTS_development_grid_audit.json')
    nuts_audit.update(integration='Archived four-chain ICC NUTS fits and draws reused without resampling; same-model corrected B100 SAR development references reused after new prediction and unchanged input hash checks.',
        numerical_revision='strict_REML_20261018',ICC_NUTS_fits_reused_without_refitting=True,
        corrected_prediction_lock_sha256=sha(args.corrected_root/'scoring_prediction_lock.json'))
    save(nuts/'NUTS_development_grid_audit.json',nuts_audit)
    for folder in (sparse,nuts):
        for path in folder.glob('*.md'):
            text=path.read_text().replace('No original Holm3 or Holm12 result is changed.',
                'The original Holm3 and Holm12 family definitions are unchanged; their numerical results are recalculated in this revision.')
            text=text.replace('fresh native SAR B100 reused after input hash matches',
                'corrected native SAR B100 reused after matching input and corrected-prediction hashes')
            path.write_text(text)
    return proxy


def diagnostic_layer(args,proxy):
    diagnostics=importlib.import_module('diagnostics_inventory')
    inv=diagnostics.Inventory(args.corrected_root);inv.modern()
    rows=inv.summary();modern_keys={(x['cohort'],x['method'],x['integration']) for x in rows}
    old=read(args.repo/'revision_20261015/results/diagnostics/diagnostics_inventory.json')
    rows += [row for row in old['summary'] if (row['cohort'],row['method'],row['integration']) not in modern_keys]
    out=args.layer/'results/diagnostics';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'diagnostics_summary.csv',index=False)
    compact=[row for row in rows if row['cohort'] in {'principal_256','real_temporal','real_territorial','observed_graph_64','development_32_Laplace'}
             and row['method'] in {'ICC','Sikov_FH','Sikov_SAR','Sikov_SAR_Published'}]
    pd.DataFrame(compact).to_csv(out/'diagnostics_modern_compact.csv',index=False)
    catalogue=read(args.repo/'revision_20261015/results/diagnostics/source_receipts_catalogue.json')
    catalogue['source_json_sha256'].update(inv.sources);save(out/'source_receipts_catalogue.json',catalogue)
    payload={**old,'summary':rows,'source_receipts_catalogue_sha256':sha(out/'source_receipts_catalogue.json'),
             'source_receipts':len(catalogue['source_json_sha256']),
             'aggregate_ledger_rows':None,'corrected_modern_aggregate_ledger_rows':len(inv.items),
             'historical_remaining_rows_preserved_without_new_fit_reads':True,
             'numerical_revision':'strict_REML_20261018',
             'scope':'Corrected reconstructed modern univariate FH/SAR only; legacy multivariate REML remains historical evidence without a universal KKT certification claim.'}
    save(out/'diagnostics_inventory.json',payload)
    for name in ('reproduction_specification.json','materials_inventory.json','original_table_figure_manifest.json',
                 'original_table_figure_manifest.csv','key_bibliography_check.json'):
        source=args.repo/'revision_20261015/results/diagnostics'/name
        if source.is_file():shutil.copy2(source,out/name)
    # Reuse the actual saved NUTS summaries; export new central/bootstrap diagnostics.
    run=importlib.import_module('numerical_cohorts_portable')
    run.REPO=args.repo;run.SOURCE=args.repo/'src/estudio_publicacion'
    run.ARTICLE=run.SOURCE/'sikov_2026-10-10'
    run.PROTOCOL=args.repo/'revision_20261015/protocols/new_computation_protocol_v1.json'
    run.PRIVATE=args.corrected_root/'numerical_cohorts';run.PUBLIC=args.layer/'results/numerical_cohorts'
    run.OLD=args.corrected_root;run.EXE=args.exe;run.CMDSTAN=args.cmdstan
    run.status()
    numdiag=importlib.import_module('numerical_diagnostics');numdiag.execute()
    oldpub=args.repo/'revision_20261015/results/numerical_cohorts'
    for name in ('computation_environment.json','sparse57_generation_audit.json','retained_sampler_capture_audit.json'):
        source=oldpub/name
        if source.is_file():shutil.copy2(source,run.PUBLIC/name)
    for cohort in ('sparse57','development_nuts'):
        shutil.copytree(args.output/'numerical_cohorts'/cohort,run.PUBLIC/cohort,dirs_exist_ok=True)
    run.REPO=proxy;run.PROTOCOL=proxy/'revision_20261015/protocols/new_computation_protocol_v1.json'
    run.PUBLIC=proxy/'revision_20261015/results/numerical_cohorts'
    contracts=importlib.import_module('new_cohort_editorial_contracts');contracts.write()
    # Explicitly replace historical directional/failure prose with new evidence.
    contract=run.PUBLIC.parent/'new_cohorts/editorial_contract.json'
    value=read(contract);sparse=read(args.layer/'results/sparse57_analysis/sparse57_summary.json')
    globalrows=sparse['all_masks_global']
    get=lambda mask,metric:next(x for x in globalrows if x['reference_method']=='ICC' and x['outcome']=='endpoint_balanced' and x['mask']==mask and x['metric']==metric)
    iv=get('all','interval_score');spans=iv['descriptive95_low']<=0<=iv['descriptive95_high']
    for key in ('main_paragraphs','supplement_paragraphs'):
        value[key]=[text.replace('includes zero.','includes zero.' if spans else 'excludes zero.') for text in value[key]]
    outside=get('unsampled_common','RMSE_sqrt_mean_MSE');width=get('unsampled_common','width');coverage=get('unsampled_common','coverage')
    outside_text=(f"Outside the common sample, RMSE changes by {outside['relative_difference_percent']:+.2f}% and width by {width['relative_difference_percent']:+.2f}% relative to ICC. Coverage is {100*coverage['selected']:.2f}% versus {100*coverage['reference']:.2f}%. All endpoint/context/support comparisons remain available; these are exploratory results after the same-model numerical correction.")
    value['main_paragraphs'][2]=outside_text;value['supplement_paragraphs'][3]=outside_text
    rm=get('all','RMSE_sqrt_mean_MSE');cv=get('all','coverage');wd=get('all','width');dc=get('all','aggregate_coverage_distance95')
    tradeoff=(f"Against ICC, balanced RMSE is {100*rm['selected']:.3f} versus {100*rm['reference']:.3f} percentage points ({rm['relative_difference_percent']:+.2f}%). "
        f"Mean width is {100*wd['selected']:.2f} versus {100*wd['reference']:.2f} percentage points ({wd['relative_difference_percent']:+.2f}%), and coverage is {100*cv['selected']:.2f}% versus {100*cv['reference']:.2f}%. "
        f"Absolute aggregate coverage deviation is {100*dc['selected']:.2f} versus {100*dc['reference']:.2f} percentage points. "
        + ('The proposed mixture has greater absolute deviation from nominal coverage.' if dc['difference_selected_minus_reference']>0 else
           'The proposed mixture has smaller absolute deviation from nominal coverage.' if dc['difference_selected_minus_reference']<0 else
           'Both strategies have the same aggregate absolute deviation from nominal coverage.'))
    value['main_paragraphs'][1]=tradeoff;value['supplement_paragraphs'][2]=tradeoff
    value['supplement_paragraphs'][4]=(f"The historical two observed-domain SAR central failures and the additional native-success/nonstationary solutions were audited under the same model and bounds. The corrected cohort retains all 256 populations. Its recorded method/endpoint gate failures number {sparse['numerical_gates']['method_endpoint_world_gate_cells_failed']}; no finite population was deleted. Corrected SAR uncertainty and mixture predictions use the original seeds and B100, with explicit projected-score certification.")
    prior_audit='revision_20261015/results/qa/sparse57_final_independent_audit.json'
    value['historical_initial_cohort_audit']={'path':prior_audit,'sha256':sha(args.repo/prior_audit),
        'scope':'Preserved pre-correction audit; it does not certify the corrected predictions or scores.'}
    value['inputs']=[item for item in value['inputs'] if item!=prior_audit]+[
        str((args.layer/'results/diagnostics/corrected_verification.json').relative_to(args.repo)),
        'revision_20261015/protocols/reml_correction_20261018.json',
        'revision_20261015/protocols/reml_correction_stage2_20261018.json']
    value['numerical_revision']='strict_REML_20261018';save(contract,value)
    devcontract=contract.with_name('development_editorial_contract.json');devc=read(devcontract)
    lap=read(args.layer/'results/devgrid/Laplace_development_grid_audit.json')
    nuts=read(args.layer/'results/devgrid_nuts/NUTS_development_grid_audit.json')
    text=(f"The corrected original 32-population Laplace grid ranks {DISPLAY.get(lap['selected_method'],lap['selected_method'])} first and {DISPLAY.get(lap['second_method'],lap['second_method'])} second; the IS95 gap is {lap['best_second_absolute_margin']:.6f}. LOPO winner counts are {counts_text(lap['LOPO_counts'])}. "
          f"Reusing the saved NUTS ICC fits with corrected SAR, the NUTS grid ranks {DISPLAY.get(nuts['best_method'],nuts['best_method'])} first and {DISPLAY.get(nuts['second_method'],nuts['second_method'])} second, with gap {nuts['best_second_absolute_margin']:.6f}; its LOPO winner counts are {counts_text(nuts['LOPO_counts'])}. The original 75/25 primary recipe is retained and no evaluation outcome reselects its weight.")
    devc['main_paragraphs']=[text];devc['supplement_paragraphs'][0]=text
    devc['supplement_paragraphs'][1]=devc['supplement_paragraphs'][1].replace(
        'Their exact fresh B100 development predictions are reused only after the unchanged input and prediction hashes match.',
        'The same-model corrected FH/SAR B100 development predictions are reused only after their unchanged input hashes and new sealed prediction hashes match; NUTS ICC draws remain archived and unchanged.')
    devc['inputs']+=[str((args.layer/'results/diagnostics/corrected_verification.json').relative_to(args.repo)),
        'revision_20261015/protocols/reml_correction_20261018.json',
        'revision_20261015/protocols/reml_correction_stage2_20261018.json']
    devc['numerical_revision']='strict_REML_20261018';save(devcontract,devc)


def numerical_impact(repo,layer):
    """Compare public matrices only, after all corrected scoring has closed."""
    previous=repo/'expected/current_freeze/data/sikov_2026-10-10'
    current=layer/'data/sikov_2026-10-10';sources={}
    def csv(path):sources[str(path.relative_to(repo))]=sha(path);return pd.read_csv(path)
    def js(path):sources[str(path.relative_to(repo))]=sha(path);return read(path)
    def finite(value):return None if not np.isfinite(value) else float(value)
    summary=[csv(base/'resultados/published_sar_comparison/summary.csv') for base in (previous,current)]
    global_rows=[]
    for method in ORIGINAL+PUBLISHED:
        pair=[]
        for frame in summary:
            d=frame.loc[frame.method.eq(method)&frame.group.eq('all')]
            if len(d)!=8 or set(d.scenario)!=set(CONTEXTS):raise ValueError('Incomplete balanced primary comparison')
            pair.append({'RMSE_sqrt_mean_MSE':float(np.sqrt(np.mean(d.rmse**2))),
                         **{name:float(d[name].mean()) for name in ('interval_score','coverage','width','bias')}})
        for metric in pair[0]:global_rows.append(dict(method=method,metric=metric,
            historical=pair[0][metric],corrected=pair[1][metric],change=pair[1][metric]-pair[0][metric]))
    families={}
    for family,relative,keys in (
        ('Holm3','resultados/published_sar_comparison/paired_balanced_contrasts_Holm3.csv',['candidate','reference','metric']),
        ('Holm12','resultados/external_comparison_Holm12/paired_balanced_contrasts_Holm12.csv',['scope','candidate','reference','metric'])):
        old,new=(csv(base/relative) for base in (previous,current))
        if family=='Holm3':old=old.loc[old.metric.eq('interval_score')];new=new.loc[new.metric.eq('interval_score')]
        if len(old)!=(3 if family=='Holm3' else 12) or len(old)!=len(new):raise ValueError('Multiplicity family changed')
        merged=old.merge(new,on=keys,suffixes=('_historical','_corrected'),validate='one_to_one')
        if len(merged)!=len(old):raise ValueError('Multiplicity comparison membership changed')
        rows=[]
        for row in merged.to_dict('records'):
            value={key:row[key] for key in keys}
            for name in ('difference','p_'+family):
                value[name+'_historical']=finite(row[name+'_historical']);value[name+'_corrected']=finite(row[name+'_corrected'])
                value[name+'_change']=finite(row[name+'_corrected']-row[name+'_historical'])
            value['reject_05_historical']=bool(row['p_'+family+'_historical']<.05)
            value['reject_05_corrected']=bool(row['p_'+family+'_corrected']<.05)
            if family=='Holm12':
                value['classification_historical']=row['classification_historical']
                value['classification_corrected']=row['classification_corrected']
            rows.append(value)
        families[family]=dict(members=len(rows),membership_changed=False,
            reject_05_historical=sum(row['reject_05_historical'] for row in rows),
            reject_05_corrected=sum(row['reject_05_corrected'] for row in rows),
            decisions_changed=sum(row['reject_05_historical']!=row['reject_05_corrected'] for row in rows),rows=rows)
    sparse_old=js(repo/'revision_20261015/results/sparse57_analysis/sparse57_summary.json')
    sparse_new=js(layer/'results/sparse57_analysis/sparse57_summary.json')
    def sparse_rows(value):return {(row['reference_method'],row['metric']):row for row in value['all_masks_global']
        if row['outcome']=='endpoint_balanced' and row['mask']=='all'}
    old,new=sparse_rows(sparse_old),sparse_rows(sparse_new)
    if old.keys()!=new.keys():raise ValueError('Sparse comparison membership changed')
    sparse=[]
    for key in sorted(old):
        before,after=old[key],new[key]
        sparse.append(dict(reference=key[0],metric=key[1],
            historical=before['difference_selected_minus_reference'],corrected=after['difference_selected_minus_reference'],
            change=after['difference_selected_minus_reference']-before['difference_selected_minus_reference'],
            historical_descriptive95=[before['descriptive95_low'],before['descriptive95_high']],
            corrected_descriptive95=[after['descriptive95_low'],after['descriptive95_high']],
            historical_interval_includes_zero=bool(before['descriptive95_low']<=0<=before['descriptive95_high']),
            corrected_interval_includes_zero=bool(after['descriptive95_low']<=0<=after['descriptive95_high'])))
    development={}
    for label,folder,name,winner in (
        ('Laplace','devgrid','Laplace_development_grid_audit.json','selected_method'),
        ('NUTS','devgrid_nuts','NUTS_development_grid_audit.json','best_method')):
        before=js(repo/'revision_20261015/results'/folder/name);after=js(layer/'results'/folder/name)
        development[label]=dict(historical_winner=before[winner],corrected_winner=after[winner],
            historical_LOPO_counts=before['LOPO_counts'],corrected_LOPO_counts=after['LOPO_counts'],
            historical_best_IS95=before['best_IS95'],corrected_best_IS95=after['best_IS95'],
            historical_best_second_margin=before['best_second_absolute_margin'],
            corrected_best_second_margin=after['best_second_absolute_margin'],
            fixed_primary_ICC_weight=.75,primary_weight_reselected=False)
    oldtables=js(repo/'expected/current_freeze/current_expected_tables_and_text.json')['tables']
    newtables=js(layer/'current_expected_tables_and_text.json')['tables']
    indexed=lambda values:{(row['family'],row['id']):row['rows'] for row in values}
    old,new=indexed(oldtables),indexed(newtables)
    if old.keys()!=new.keys() or len(old)!=22:raise ValueError('The 22 original table identities must remain mapped')
    table_changes=[]
    for key in sorted(old):
        a,b=old[key],new[key]
        if len(a)!=len(b) or list(map(len,a))!=list(map(len,b)):raise ValueError('Original formatted table structure changed')
        changes=sum(x!=y for first,second in zip(a,b) for x,y in zip(first,second))
        table_changes.append(dict(family=key[0],id=key[1],printed_cells=sum(map(len,a)),printed_cells_changed=changes))
    return dict(status='COMPLETE_OLD_TO_CORRECTED_PUBLIC_MATRIX_COMPARISON',
        scope='Descriptive changes after same-model numerical correction; historical files are preserved. No equivalence margin, weight retuning, new confirmatory family, or independent populations are introduced.',
        balanced_primary_global=global_rows,original_families=families,sparse57_global=sparse,
        sparse57_gate_cells_historical=sparse_old['numerical_gates']['method_endpoint_world_gate_cells_failed'],
        sparse57_gate_cells_corrected=sparse_new['numerical_gates']['method_endpoint_world_gate_cells_failed'],
        development=development,original_formatted_tables=table_changes,
        original_formatted_printed_cells_changed=sum(row['printed_cells_changed'] for row in table_changes),
        inputs_sha256=sources,derived_from_Word=False,analysis_source_sha256=sha(__file__))


def export_layer(args):
    verify_scoring_lock(args)
    if read(args.output/'scoring_completion.json')['status']!='COMPLETE_CORRECTED_NUMERICAL_SCORING':
        raise ValueError('All corrected scores must close before public layer export')
    if args.layer is None:raise ValueError('--layer is required for export')
    args.layer=args.layer.resolve();args.layer.relative_to(args.repo/'revision_20261015/results')
    if (args.layer/'layer_manifest.json').exists():raise FileExistsError('Preserve a complete correction layer')
    args.layer.mkdir(parents=True,exist_ok=True)
    shutil.copytree(args.repo/'expected/current_freeze/data',args.layer/'data',dirs_exist_ok=True)
    export_aggregates(args)
    for filename in ('runtime_corrected_native_summary.csv','runtime_accounting.json'):
        shutil.copy2(args.corrected_root/filename,args.layer/'results/reml_numerical_certificates'/filename)
    proxy=incremental_layer(args)
    corrected_figures(args)
    diagnostic_layer(args,proxy)
    # This refuses partial native verification and closes the corrected central
    # and bootstrap diagnostic tables before their hashes enter the layer seal.
    importlib.import_module('reml_correction.corrected_diagnostic_editorial').build(args.repo,args.layer)
    contract=importlib.import_module('current_revision_contract')
    derived=contract.derive(args.repo,args.layer/'data')
    save(args.layer/'current_expected_tables_and_text.json',derived)
    save(args.layer/'results/reml_numerical_certificates/numerical_impact_comparison.json',
         numerical_impact(args.repo,args.layer))
    files={str(path.relative_to(args.layer)):sha(path) for path in args.layer.rglob('*') if path.is_file() and path.name!='layer_manifest.json'}
    result={'status':'COMPLETE_CORRECTED_EDITORIAL_LAYER',
        'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'prediction_lock_sha256':sha(args.corrected_root/'scoring_prediction_lock.json'),
        'native_prediction_lock_sha256':sha(args.native_lock),
        'scoring_receipt_sha256':sha(args.output/'scoring_completion.json'),
        'adapter_sha256':sha(__file__),'derived_from_Word':False,
        'original_freeze_modified':False,'ICC_models_or_draws_refitted':False,
        'primary_ICC_weight':.75,'original_multiplicity_families':['Holm3','Holm12'],
        'scope':'Same-model numerical correction of reconstructed univariate FH/SAR and their bootstrap; legacy 800/CUT8 multivariate REML is preserved historical evidence and not universally KKT-certified.',
        'document_visual_QA_state_at_numerical_seal':'not_yet_built_or_reviewed',
        'document_visual_QA_current_receipt':'revision_20261015/results/qa/final_editorial_delivery_verification.json',
        'document_visual_QA_scope':'The state above records the numerical-seal instant only. The current document/page review state is defined by the separate SHA-bound final editorial delivery receipt.',
        'files_sha256':files,'outputs_sha256':files}
    save(args.layer/'layer_manifest.json',result)
    print(json.dumps({'status':result['status'],'files':len(files),'layer':str(args.layer)}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=('prepare','verify-lock','score','export'))
    for name in ('repo','old-root','old-numerical-root','corrected-root','native-lock','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    for name in ('reference-root','exe','cmdstan'):
        ap.add_argument('--'+name,type=Path)
    ap.add_argument('--layer',type=Path)
    args=ap.parse_args()
    for name in ('repo','old_root','old_numerical_root','corrected_root','native_lock','output'):
        setattr(args,name,getattr(args,name).resolve())
    if args.corrected_root.is_relative_to(args.repo):raise ValueError('All arrays stay in an external private run')
    if args.corrected_root in (args.old_root,args.old_numerical_root):raise ValueError('Never modify an old run')
    if args.action in ('score','export') and any(getattr(args,name) is None for name in ('reference_root','exe','cmdstan')):
        ap.error(args.action+' requires --reference-root, --exe and --cmdstan')
    if args.action=='prepare':prepare(args)
    elif args.action=='verify-lock':print(json.dumps({'verified':verify_scoring_lock(args)['status']}))
    elif args.action=='export':
        imports(args.repo);export_layer(args)
    else:
        score(args)


if __name__=='__main__':
    main()
