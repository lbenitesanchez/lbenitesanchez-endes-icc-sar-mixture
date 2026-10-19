#!/usr/bin/env python3
"""New exploratory IAOS sensitivity cohorts, registered before score access.

Original source files are read-only. Private NUTS state remains outside repo.
Use register, prepare, run (checkpointed), status, then score. No partial score.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','STAN_NUM_THREADS'):
    os.environ[key]='1'
import argparse, concurrent.futures as cf, datetime as dt, hashlib, importlib, json, logging
from pathlib import Path
import shutil, sys, time, traceback
import numpy as np
import pandas as pd

# Operational path routing only. All scientific functions below are unchanged.
REPO=Path(__file__).resolve().parents[2]
_PATH_PARSER=argparse.ArgumentParser(add_help=False)
for _flag in ('config','private-run-root','previous-run-root','runtime-executable','cmdstan-root','protocol'):
    _PATH_PARSER.add_argument('--'+_flag,type=Path)
_PATH_ARGS,_PATH_REMAINDER=_PATH_PARSER.parse_known_args()
_CONFIG_PATH=_PATH_ARGS.config or (Path(os.environ['IAOS_NUMERICAL_CONFIG']) if os.environ.get('IAOS_NUMERICAL_CONFIG') else None)
_CONFIG=json.loads(_CONFIG_PATH.expanduser().resolve().read_text()) if _CONFIG_PATH else {}
def _route(cli_name,environment,key,default):
    value=getattr(_PATH_ARGS,cli_name,None) or os.environ.get(environment) or _CONFIG.get(key) or default
    path=Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():path=(_CONFIG_PATH.expanduser().resolve().parent if _CONFIG_PATH else REPO)/path
    path=path.resolve();os.environ[environment]=str(path);return path
SOURCE=REPO/'src/estudio_publicacion'
ARTICLE=SOURCE/'sikov_2026-10-10'
PRIVATE=_route('private_run_root','IAOS_PRIVATE_RUN_ROOT','private_run_root',Path.home()/'.local/share/iaos/revision_2026-10-15/numerical_cohorts')
OLD=_route('previous_run_root','IAOS_PREVIOUS_RUN_ROOT','previous_run_root',Path.home()/'.local/share/iaos/reproduction_2026-10-13')
PUBLIC=REPO/'revision_20261015/results/numerical_cohorts'
PROTOCOL=_route('protocol','IAOS_EXPLORATORY_PROTOCOL','protocol',REPO/'revision_20261015/protocols/new_computation_protocol_v1.json')
EXE=_route('runtime_executable','IAOS_RUNTIME_EXECUTABLE','runtime_executable',OLD/'runtime/beta_power_hybrid')
CMDSTAN=_route('cmdstan_root','IAOS_CMDSTAN_ROOT','cmdstan_root',Path.home()/'.cmdstan/cmdstan-2.40.0')
if PRIVATE==REPO or PRIVATE.is_relative_to(REPO):
    raise ValueError('Private sampler state must remain outside the public repository')
CONTEXTS=('SAR_probability_srs','SAR_logit_srs','SAR_logit_psu_sparse','IID_logit_psu_sparse')
METHODS=('Sikov_FH','Sikov_SAR','Sikov_SAR_Published','ICC','Selected')
DEVMETHODS=('Sikov_FH','Sikov_SAR','ICC','ICC_poverty','ICC_SAR_w25','ICC_SAR_w50','ICC_SAR_w75')
_MODULES=None

def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n');tmp.replace(p)
def manifest(folder):return {str(p.relative_to(folder)):sha(p) for p in sorted(Path(folder).rglob('*')) if p.is_file() and p.name!='completion.json'}
def modules():
    global _MODULES
    if _MODULES is not None:return _MODULES
    for p in (SOURCE/'scripts/competitors', SOURCE/'scripts', ARTICLE/'scripts', REPO/'reproduction'):sys.path.insert(0,str(p))
    names=('comparison_simulation','comparison_evaluate','sikov_bootstrap','sikov_published_fit','improve_candidates','improve_development','benchmark_metrics','improve_select','capture_private_stan_state')
    _MODULES={n:importlib.import_module(n) for n in names}
    import cmdstanpy
    cmdstanpy.set_cmdstan_path(str(CMDSTAN));logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    backend=cmdstanpy.CmdStanModel(stan_file=str(SOURCE/'scripts/competitors/beta_power_hybrid.stan'),exe_file=str(EXE),compile=False)
    _MODULES['comparison_evaluate'].hmc._MODEL=_MODULES['comparison_evaluate'].DiagonalMetric(backend)
    return _MODULES

def generator_text():
    original=(ARTICLE/'scripts/comparison_simulation.py').read_text()
    replacements={"EVALUATION_IDS=list(range(30000,30064))":"EVALUATION_IDS=list(range(50000,50064))",'SeedSequence([20261010,701,':'SeedSequence([20261015,1701,','SeedSequence([20261010,702,':'SeedSequence([20261015,1702,','g),32,replace=False)':'g),19,replace=False)',"'population_stream':[20261010,701,":"'population_stream':[20261015,1701,","'sampling_stream':[20261010,702,":"'sampling_stream':[20261015,1702,"}
    for old,new in replacements.items():
        if original.count(old)!=1:raise ValueError('Expected exactly one registered generator substitution: '+old)
        original=original.replace(old,new)
    return original,replacements

def gate():
    sys.path.insert(0,str(REPO/'reproduction'))
    p=read(PROTOCOL)
    for name,digest in p['original_sources_sha256'].items():
        from code_only_sources import historical_source_matches
        if not historical_source_matches(REPO/name,digest):raise RuntimeError('Unrecorded scientific source change: '+name)
    if sha(EXE)!=p['runtime_executable_sha256']:raise RuntimeError('Runtime executable changed')
    if hashlib.sha256(generator_text()[0].encode()).hexdigest()!=p['adapted_generator_text_sha256']:raise RuntimeError('Registered adapted generator changed')
    return p

def register():
    if PROTOCOL.exists():raise FileExistsError('Registration is immutable: '+str(PROTOCOL))
    if (PRIVATE/'sparse57/evaluacion/inputs').exists() or (PRIVATE/'development_nuts/fits').exists():raise RuntimeError('New fitting/generation already exists before registration')
    adapted,replacements=generator_text()
    paths=[p for base in (ARTICLE/'scripts',SOURCE/'scripts/competitors',SOURCE/'scripts') for p in base.glob('*.py')]
    paths+=list((SOURCE/'scripts/competitors').glob('*.stan'))
    p=dict(protocol='new-exploratory-computation-v1',registered_utc=utc(),role='New exploratory sensitivity; not original prespecification',
      manuscript_original_primary_families_unchanged=['Holm3','Holm12'],original_mixture=dict(ICC=.75,SAR_full_graph=.25,weights_reselected=False),
      sparse57=dict(worlds=256,replications_per_context=64,contexts=CONTEXTS,replication_ids=list(range(50000,50064)),areas=120,stratum_sizes=[40,40,40],observed_per_stratum=19,observed_total=57,unsampled_total=63,observed_fraction=.475,ENDES_fraction=888/1874,population_seed_sequence=[20261015,1701,'context_index','replication'],sampling_seed_sequence=[20261015,1702,'context_index','replication'],fit_seed_sequence=[20261015,1811,'context_index','replication'],published_sar_bootstrap_seed_sequence=[20261015,1941,'context_index','replication'],methods=METHODS,all_other_generation_rules_unchanged=True),
      development_nuts=dict(worlds=32,replications_per_context=8,ids=list(range(8)),same_original_inputs=True,methods=DEVMETHODS,reuse='Exact fresh FH/full-SAR development predictions and B100 MSE, bound to same inputs',candidate_weights=[.25,.5,.75],fit_seed_sequence=[20261015,1711,'context_index','replication'],poverty_seed_offset=90001,weights_reselected_for_primary=False),
      NUTS=dict(chains=4,parallel_chains=1,warmup=1000,sampling=1000,adapt_delta=.95,retry_warmup=2000,retry_sampling=2000,retry_adapt_delta=.99,max_treedepth=12,metric='diag_e',gates=dict(max_Rhat=1.01,min_bulk_ESS=400,min_tail_ESS=400,min_E_BFMI=.3,divergences=0,treedepth_hits=0,beta_shape_guard_cells=0),initialization='unchanged original MAP and positive covariance diagonal or identity'),
      REML=dict(bootstrap_B=100,interval_MSE='native naive',original_optimizer_and_stationarity_gates=True),
      scoring=dict(no_truth_read_by_fit=True,complete_prediction_lock_before_any_new_truth_score=True,metrics=['IS95','RMSE','bias','coverage','width','absolute_distance_coverage_from_95pct'],groups=['all','observed_endpoint','unsampled_common'],endpoints=['anemia','stunting'],district_weights='equal within population and group',endpoint_weights='equal',context_weights='equal',independent_replication_unit='simulated population',MCSE='sqrt(sum_c(s_c^2/R_c)/C^2) after aggregation within population',RMSE='sqrt(weighted mean population MSE); paired delta method, not mean population RMSE',interval='descriptive normal 95% Monte Carlo interval; exploratory, no changes to original Holm families'),
      failure_policy=dict(retain_all_planned_cases=True,automatic_NUTS_attempts=2,existing_sampler_retry_rule=True,no_changed_seeds_or_gates_on_failure=True,no_deletion_or_favorable_replacement=True,failed_cases_kept_in_denominator=True,no_complete_claim_on_partial=True,checkpoint='per method; preserve partials and exception receipts; rerun only absent components; manual retry must use new attempt directory and declared reason'),
      generator_substitutions=replacements,original_sources_sha256={str(q.relative_to(REPO)):sha(q) for q in sorted(set(paths))},adapted_generator_text_sha256=hashlib.sha256(adapted.encode()).hexdigest(),runner_initial_sha256=sha(__file__),runtime_executable_sha256=sha(EXE),versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),public_output_privacy='Only synthetic population aggregate metrics and sanitized status; full chains and MAP/native JSON private')
    save(PROTOCOL,p);PRIVATE.mkdir(parents=True,exist_ok=True);os.chmod(PRIVATE,0o700)
    snapshot=PRIVATE/'registered_sources';snapshot.mkdir();(snapshot/'comparison_simulation_sparse57.py').write_text(adapted);shutil.copy2(__file__,snapshot/'numerical_cohorts_registered.py');shutil.copy2(PROTOCOL,snapshot/PROTOCOL.name)
    print(json.dumps({'registered':str(PROTOCOL),'sha256':sha(PROTOCOL)}),flush=True)

def prepare():
    gate();m=modules();root=PRIVATE/'sparse57';template=root/'datos/simulation_template';template.parent.mkdir(parents=True,exist_ok=True)
    if not template.exists():shutil.copytree(ARTICLE/'datos/simulation_template',template)
    text,_=generator_text();ns={'__name__':'registered_sparse57_generator','__file__':str(ARTICLE/'scripts/comparison_simulation.py')};exec(compile(text,'registered_sparse57_generator','exec'),ns);ns['ROOT']=root
    for c in CONTEXTS:
        for r in range(50000,50064):
            target=root/'evaluacion/inputs'/c/f'rep_{r:05d}'
            if not (target/'manifest.json').exists():ns['generate'](c,r,'evaluation')
            d=np.load(target/'model_inputs/inputs.npz');assert len(d['sample_idx'])==57
    save(PRIVATE/'sparse57/preparation.json',dict(created_utc=utc(),worlds=256,template_sha256=sha(template/'auxiliary.npz'),protocol_sha256=sha(PROTOCOL),truth_scored=False))
    print(json.dumps({'prepared_worlds':256,'truth_scored':False}),flush=True)

def method_done(folder,method):
    receipt=folder/(method+'_completion.json')
    if not receipt.exists():return False
    obj=read(receipt)
    for name,digest in obj['outputs'].items():
        if sha(folder/name)!=digest:raise RuntimeError('Checkpoint modified '+name)
    return True

def store_method(folder,method,pred,status,seconds):
    path=folder/(method+'.npz');np.savez_compressed(path,**{k:np.asarray(v) for k,v in pred.items() if isinstance(v,np.ndarray)})
    save(folder/(method+'_status.json'),status)
    save(folder/(method+'_completion.json'),dict(method=method,finished_utc=utc(),seconds=seconds,converged=bool(status['converged']),outputs={p.name:sha(p) for p in (path,folder/(method+'_status.json'))}))

def fit_world(job):
    cohort,c,r=job;gate();m=modules();tick=time.perf_counter()
    inp=(PRIVATE/'sparse57/evaluacion/inputs' if cohort=='sparse57' else OLD/'sikov/desarrollo/inputs')/c/f'rep_{r:05d}'
    folder=PRIVATE/cohort/'fits'/c/f'rep_{r:05d}';folder.mkdir(parents=True,exist_ok=True)
    completion=folder/'completion.json'
    if completion.exists():return read(completion)
    d=dict(np.load(inp/'model_inputs/inputs.npz',allow_pickle=False));units=pd.read_csv(inp/'model_inputs/sample.csv');names=d['covariate_names'].astype(str).tolist()
    seed=int(np.random.SeedSequence([20261015,1811 if cohort=='sparse57' else 1711,CONTEXTS.index(c),r]).generate_state(1)[0])%(2**32-200000)
    methods=METHODS if cohort=='sparse57' else DEVMETHODS
    recorder=None
    try:
        for method,spatial in (('Sikov_FH',False),('Sikov_SAR',True)):
            if method_done(folder,method):continue
            start=time.perf_counter()
            if cohort=='development_nuts':
                old=OLD/'sikov/desarrollo/recreated_native/fit_results'/c/f'rep_{r:05d}'
                seal=read(old/'predictions_seal_before_development_scoring.json')
                if seal['common_inputs_sha256']!=sha(inp/'model_inputs/inputs.npz') or seal['prediction_sha256'][method]!=sha(old/method/'predictions.npz'):raise RuntimeError('Development FH/SAR reuse hash mismatch')
                pred=dict(np.load(old/method/'predictions.npz',allow_pickle=False));st=read(old/method/'status.json')
                status=dict(converged=all(st['success']),fits=st['status'],reused_exactly=True,source_prediction_sha256=sha(old/method/'predictions.npz'),same_input_sha256=seal['common_inputs_sha256'])
            else:
                values=[]
                for j,name in enumerate(('anemia','stunting')):
                    mask=np.isfinite(d['y'][:,j])&np.isfinite(d['D'][:,j,j])&(d['D'][:,j,j]>0)
                    values.append(m['sikov_bootstrap'].fit_sikov_bootstrap(d['X_raw'],d['y'][mask,j],d['D'][mask,j,j],d['sample_idx'][mask],d['W'],d['strata'],spatial=spatial,outcome=name,covariate_names=names,B=100,seed=seed+j*1009+(777 if spatial else 333),interval_estimator='naive'))
                pred={key:np.column_stack([v[key] for v in values]) for key in ('mean','lower','upper','mse','mse_plugin','mse_delta','mse_naive','mse_bc','naive_mcse','bc_mcse')};status=dict(converged=all(v['converged'] for v in values),fits=values)
            store_method(folder,method,pred,status,time.perf_counter()-start)
        if cohort=='sparse57' and not method_done(folder,'Sikov_SAR_Published'):
            start=time.perf_counter();sub=folder/'observed_sar_work'
            if sub.exists():raise FileExistsError('Preserve partial observed SAR; explicit retry required')
            ps=int(np.random.SeedSequence([20261015,1941,CONTEXTS.index(c),r]).generate_state(1)[0]);pred,_,status=m['sikov_published_fit'].fit_arrays(d,sub,ps);store_method(folder,'Sikov_SAR_Published',pred,status,time.perf_counter()-start)
        import cmdstanpy
        from beta_power import EndpointPosterior
        for method in (('ICC',) if cohort=='sparse57' else ('ICC','ICC_poverty')):
            if method_done(folder,method):continue
            start=time.perf_counter();state=folder/(method+'_private_stan_state')
            if state.exists():raise FileExistsError('Preserve partial NUTS; explicit retry required '+str(state))
            recorder=m['capture_private_stan_state'].Recorder(state,cmdstanpy.CmdStanModel,EndpointPosterior,cmdstanpy.__version__).install()
            common=dict(seed=seed+(90001 if method=='ICC_poverty' else 0),sampler_config=m['comparison_evaluate'].SAMPLER,output_dir=folder/(method+'_chains'))
            try:
                if method=='ICC':base=m['comparison_evaluate'].hmc.fit(d['X'],d['y'],d['n_eff'],d['sample_idx'],units,d['basis'],**common)
                else:base=m['improve_candidates'].fit_icc_poverty(d['X'],d['y'],d['n_eff'],d['sample_idx'],units,d['basis'],strata=d['strata'],integration='hmc',**common)
            finally:
                recorder.uninstall();recorder.finish('NEW_EXPLORATORY_FIT_FINISHED_OR_EXCEPTION');recorder=None
            pred={'mean':base['p_mean'],'lower':base['lower95'],'upper':base['upper95'],'mse':base['p_variance'],'p_variance':base['p_variance']}
            np.savez_compressed(folder/(method+'_conditional_draws.npz'),p_draws=base['p_draws'],anemia_marginal=base['marginal_draws'][0],stunting_marginal=base['marginal_draws'][1])
            status=dict(converged=all(x['success'] and x['beta_shape_guard_cells']==0 for x in base['status']),endpoints=base['status']);store_method(folder,method,pred,status,time.perf_counter()-start)
        icc=dict(np.load(folder/'ICC.npz'));draws=dict(np.load(folder/'ICC_conditional_draws.npz'));base=dict(p_mean=icc['mean'],p_variance=icc['mse'],marginal_draws=[draws['anemia_marginal'],draws['stunting_marginal']]);sar=dict(np.load(folder/'Sikov_SAR.npz'))
        for method,w in ([('Selected',.75)] if cohort=='sparse57' else [('ICC_SAR_w25',.25),('ICC_SAR_w50',.5),('ICC_SAR_w75',.75)]):
            if method_done(folder,method):continue
            start=time.perf_counter();mixed=m['improve_candidates'].mixture_prediction(base,sar,w)
            converged=read(folder/'ICC_completion.json')['converged'] and read(folder/'Sikov_SAR_completion.json')['converged'];store_method(folder,method,mixed,dict(converged=converged,recipe=mixed['status']),time.perf_counter()-start)
        gate();result=dict(cohort=cohort,context=c,replication=r,completed_utc=utc(),complete=True,seconds=time.perf_counter()-tick,protocol_sha256=sha(PROTOCOL),input_sha256=sha(inp/'model_inputs/inputs.npz'),sample_sha256=sha(inp/'model_inputs/sample.csv'),truth_read_by_fit=False,convergence={k:read(folder/(k+'_completion.json'))['converged'] for k in methods},outputs=manifest(folder))
        save(completion,result);return {k:v for k,v in result.items() if k!='outputs'}
    except BaseException as exc:
        save(folder/('failure_'+dt.datetime.now().strftime('%Y%m%dT%H%M%S')+'.json'),dict(world=job,error=repr(exc),traceback=traceback.format_exc(),created_utc=utc(),case_retained=True));raise

def jobs_for(cohort):return [(cohort,c,r) for r in (range(50000,50064) if cohort=='sparse57' else range(8)) for c in CONTEXTS]
def status():
    rows=[]
    for cohort in ('development_nuts','sparse57'):
        jobs=jobs_for(cohort);closed=[];failures=[]
        for _,c,r in jobs:
            f=PRIVATE/cohort/'fits'/c/f'rep_{r:05d}'
            if (f/'completion.json').exists():closed.append(read(f/'completion.json'))
            failures.extend([str(p.relative_to(PRIVATE)) for p in f.glob('failure_*.json')])
        row=dict(cohort=cohort,planned=len(jobs),completed=len(closed),remaining=len(jobs)-len(closed),all_planned_completed=len(closed)==len(jobs),closed_with_gate_failure=sum(not all(x['convergence'].values()) for x in closed),recorded_exception_attempts=len(failures),exception_receipts=failures,observed_closed_seconds=[x['seconds'] for x in closed]);rows.append(row)
    result=dict(updated_utc=utc(),protocol_sha256=sha(PROTOCOL),cohorts=rows,complete=all(x['all_planned_completed'] for x in rows),private_results_exported=False)
    save(PUBLIC/'computation_status.json',result);save(PRIVATE/'status.json',result);return result

def run(cohort,workers,limit):
    gate();jobs=[j for j in jobs_for(cohort) if not (PRIVATE/cohort/'fits'/j[1]/f'rep_{j[2]:05d}/completion.json').exists()]
    if limit is not None:jobs=jobs[:limit]
    log=PRIVATE/(cohort+'_dispatch_events.jsonl');failures=0
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fit_world,j):j for j in jobs}
        for future in cf.as_completed(futures):
            job=futures[future]
            try:result=future.result();event=dict(world=job,status='WORLD_CLOSED',seconds=result['seconds'],convergence=result['convergence'],utc=utc())
            except BaseException as exc:event=dict(world=job,status='FAILED_CASE_RETAINED',error=repr(exc),utc=utc());failures+=1
            with log.open('a') as stream:stream.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True);status()
    save(PRIVATE/(cohort+'_dispatch_completion.json'),dict(completed_utc=utc(),requested=len(jobs),exception_attempts=failures,workers=workers));print(json.dumps(status()),flush=True)

def score(cohort):
    gate();m=modules();planned=jobs_for(cohort);methods=METHODS if cohort=='sparse57' else DEVMETHODS
    items=[]
    for _,c,r in planned:
        f=PRIVATE/cohort/'fits'/c/f'rep_{r:05d}'
        if not (f/'completion.json').exists():raise RuntimeError('Partial cohort: no new truth scoring '+str((cohort,c,r)))
        q=read(f/'completion.json')
        for name,digest in q['outputs'].items():
            if sha(f/name)!=digest:raise RuntimeError('Completed output changed '+name)
        items.append(dict(context=c,replication=r,completion_sha256=sha(f/'completion.json'),convergence=q['convergence']))
    lock=PRIVATE/cohort/'prediction_lock_before_score.json'
    if not lock.exists():save(lock,dict(locked_utc=utc(),worlds=len(items),protocol_sha256=sha(PROTOCOL),items=items,new_truth_not_scored_before_lock=True))
    records=[];diagnostics=[]
    for _,c,r in planned:
        f=PRIVATE/cohort/'fits'/c/f'rep_{r:05d}';inp=(PRIVATE/'sparse57/evaluacion/inputs' if cohort=='sparse57' else OLD/'sikov/desarrollo/inputs')/c/f'rep_{r:05d}';q=read(f/'completion.json');d=dict(np.load(inp/'model_inputs/inputs.npz'));truth=dict(np.load(inp/'evaluation_only/truth.npz'))
        for method in methods:
            pred=dict(np.load(f/(method+'.npz')));rows=m['improve_development'].scores(truth,d,pred,method,c,r,[q['convergence'][method]]*2,q['seconds'])
            for row in rows:row.update(phase=cohort,protocol='new_exploratory_revision_20261015_v1');row['rmse_population']=float(np.sqrt(row['mse']));row['coverage_distance_population']=abs(row['coverage']-.95)
            records.extend(rows)
            st=read(f/(method+'_status.json'))
            for j,end in enumerate(st.get('endpoints',[])):
                for trial in end['sampler_trials']:
                    diagnostics.append(dict(cohort=cohort,context=c,replication=r,method=method,endpoint=('anemia','stunting')[j],**trial,min_E_BFMI=min(trial['E_BFMI']),beta_shape_guard_cells=end['beta_shape_guard_cells'],NUTS_final_success=end['success']))
    out=PUBLIC/cohort;out.mkdir(parents=True,exist_ok=True);frame=pd.DataFrame(records);frame.to_csv(out/'replications.csv',index=False);pd.DataFrame(diagnostics).to_csv(out/'NUTS_diagnostics_all_attempts.csv',index=False)
    m['benchmark_metrics'].summarize(frame).to_csv(out/'summary.csv',index=False)
    save(out/'scoring_receipt.json',dict(cohort=cohort,complete=True,worlds=len(planned),all_models_converged=all(all(x['convergence'].values()) for x in items),protocol_sha256=sha(PROTOCOL),prediction_lock_sha256=sha(lock),scored_utc=utc(),unit='independent population',outputs=manifest(out)))
    print(json.dumps({'scored':cohort,'worlds':len(planned),'output':str(out)}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__,parents=[_PATH_PARSER]);p.add_argument('action',choices=('register','prepare','run','status','score'));p.add_argument('--cohort',choices=('sparse57','development_nuts'),default='development_nuts');p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int);a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('Workers must be1..4; all chains sequential per endpoint')
    if a.action=='register':register()
    elif a.action=='prepare':prepare()
    elif a.action=='run':run(a.cohort,a.workers,a.limit)
    elif a.action=='score':score(a.cohort)
    else:print(json.dumps(status(),indent=2))
if __name__=='__main__':main()
