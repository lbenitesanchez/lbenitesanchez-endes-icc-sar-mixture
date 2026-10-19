"""Frozen matched fits on fresh populations, without reading their truth."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse,concurrent.futures,datetime as dt,json,sys,time,traceback
import numpy as np
import pandas as pd
from comparison_simulation import ROOT,STUDY,CONTEXTS,EVALUATION_IDS,generate,save,sha
from sikov_model import fit_sikov,knn_weights,select_columns
from sikov_bootstrap import fit_sikov_bootstrap,_certificate
from improve_candidates import fit_icc_poverty,mixture_prediction
CORE=STUDY/'scripts/competitors';sys.path.insert(0,str(CORE))
import beta_power_hmc as hmc
class DiagonalMetric:
    """Operational NUTS adapter; posterior and diagnostic gates are unchanged."""
    def __init__(self,model):self.model=model
    def sample(self,**kwargs):
        matrix=np.asarray(kwargs['inv_metric'])
        if matrix.ndim!=2 or np.any(np.diag(matrix)<=0):raise ValueError('Invalid initial covariance diagonal')
        kwargs['inv_metric']=np.diag(matrix).copy()
        return self.model.sample(**kwargs)

SAMPLER={'metric':'diag_e','adapt_delta':.95,'retry_adapt_delta':.99,'parallel_chains':1,
 'initial_metric':'positive unrepaired MAP covariance diagonal or identity'}
METHODS=('Sikov_FH','Sikov_SAR','Sikov_FH_BC','Sikov_SAR_BC','Sikov_SAR_k4','Sikov_SAR_k8','Sikov_SAR_haversine','ICC','Selected')

def gate():
    p=ROOT/'protocol_freeze.json';d=json.loads(p.read_text())
    if not d.get('approved_for_fresh_fits'):raise RuntimeError('No approved freeze')
    for path,digest in d['statistical_sources_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Frozen source changed: '+path)
    for path,digest in d['frozen_inputs_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Frozen input changed: '+path)
    for path,digest in d.get('runtime_binary_sha256',{}).items():
        if sha(path)!=digest:raise RuntimeError('Frozen runtime executable changed: '+path)
    return d,sha(p)

def get_backend():
    import cmdstanpy
    local_cmdstan=STUDY/'.competitor_env/cmdstan/cmdstan-2.40.0'
    if local_cmdstan.is_dir():cmdstanpy.set_cmdstan_path(str(local_cmdstan))
    sdk=Path('/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk')
    if sdk.exists():os.environ.setdefault('SDKROOT',str(sdk))
    # Explicit exe_file avoids writing to or recompiling the frozen old kernels.
    executable=CORE/'beta_power_hybrid'
    if executable.is_file():
        model=cmdstanpy.CmdStanModel(stan_file=str(CORE/'beta_power_hybrid.stan'),exe_file=str(executable),compile=False)
    else:
        model=cmdstanpy.CmdStanModel(stan_file=str(CORE/'beta_power_hybrid.stan'))
    hmc._MODEL=DiagonalMetric(model)
    return model

def fits(d,units,out,seed,recipe,sensitivity=True):
    names=d['covariate_names'].astype(str).tolist();out.mkdir(parents=True,exist_ok=True)
    preds={};statuses={}
    refs=[('Sikov_FH',False,d['W']),('Sikov_SAR',True,d['W'])]
    if sensitivity:
        refs += [(f'Sikov_SAR_k{k}',True,knn_weights(d['coords'],k=k,strata=d['strata'])) for k in (4,8)]
        refs += [('Sikov_SAR_haversine',True,knn_weights(d['coords'],k=6,strata=d['strata'],haversine=True))]
    endpoint_masks=[np.isfinite(d['y'][:,j]) & np.isfinite(d['D'][:,j,j]) & (d['D'][:,j,j]>0) for j in range(2)]
    endpoint_y=[d['y'][m,j] for j,m in enumerate(endpoint_masks)]
    endpoint_D=[d['D'][m,j,j] for j,m in enumerate(endpoint_masks)]
    endpoint_idx=[d['sample_idx'][m] for m in endpoint_masks]
    for name,spatial,W in refs:
        if name in ('Sikov_FH','Sikov_SAR'):
            values=[fit_sikov_bootstrap(d['X_raw'],endpoint_y[j],endpoint_D[j],endpoint_idx[j],W,d['strata'],spatial=spatial,outcome=endpoint,covariate_names=names,B=100,seed=seed+j*1009+(777 if spatial else 333),interval_estimator='naive') for j,endpoint in enumerate(('anemia','stunting'))]
        else:
            values=[fit_sikov(d['X_raw'],endpoint_y[j],endpoint_D[j],endpoint_idx[j],W,d['strata'],spatial=spatial,outcome=endpoint,covariate_names=names,uncertainty='delta') for j,endpoint in enumerate(('anemia','stunting'))]
            # Numerical certification uses only the training REML objective.
            # Preserve optimizer flags and do not alter returned point estimates.
            for j,v in enumerate(values):
                for sf in v['fit']:
                    if sf['converged']:continue
                    ids=np.flatnonzero(d['strata']==sf['stratum']);mask=np.isin(endpoint_idx[j],ids)
                    ss=np.searchsorted(ids,endpoint_idx[j][mask]);cols=select_columns(('anemia','stunting')[j],sf['stratum'],names,spatial)
                    WW=W[np.ix_(ids,ids)];WW=WW/WW.sum(axis=1)[:,None]
                    certificate=_certificate(d['X_raw'][ids][:,cols],endpoint_y[j][mask],endpoint_D[j][mask],ss,WW,spatial,sf['variance'],sf['rho'])
                    sf['raw_optimizer_converged']=False;sf['stationarity_certificate']=certificate;sf['converged']=certificate['passed']
                v['converged']=all(sf['converged'] for sf in v['fit'])
        preds[name]={key:np.column_stack([v[key] for v in values]) for key in ('mean','lower','upper','mse','mse_plugin','mse_delta')}
        statuses[name]={'converged':all(v['converged'] for v in values),'fits':values}
        if name in ('Sikov_FH','Sikov_SAR'):
            pred_bc={key:np.column_stack([v[key] for v in values]) for key in ('mean','mse_plugin','mse_delta','mse_naive','mse_bc','naive_mcse','bc_mcse')}
            pred_bc['mse']=pred_bc['mse_bc'];se=np.sqrt(np.maximum(pred_bc['mse_bc'],0))
            pred_bc['lower']=pred_bc['mean']-1.959963984540054*se;pred_bc['upper']=pred_bc['mean']+1.959963984540054*se
            preds[name+'_BC']=pred_bc
            statuses[name+'_BC']={'converged':statuses[name]['converged'],'negative_bc_MSE_cells':int((pred_bc['mse_bc']<0).sum()),'negative_MSE_floor_only_for_interval_sqrt':0.,'same_bootstrap_draws_as_primary':True}
    get_backend()
    before=time.perf_counter()
    base=hmc.fit(d['X'],d['y'],d['n_eff'],d['sample_idx'],units,d['basis'],seed=seed,sampler_config=SAMPLER)
    np.savez_compressed(out/'ICC_conditional_draws.npz',p_draws=base['p_draws'],anemia_marginal=base['marginal_draws'][0],stunting_marginal=base['marginal_draws'][1])
    preds['ICC']={'mean':base['p_mean'],'lower':base['lower95'],'upper':base['upper95'],'mse':base['p_variance']}
    statuses['ICC']={'converged':all(v['success'] and v['beta_shape_guard_cells']==0 for v in base['status']),'endpoints':base['status'],'seconds':time.perf_counter()-before}
    if recipe['family']=='ICC_SAR_mixture':
        preds['Selected']=mixture_prediction(base,preds['Sikov_SAR'],recipe['icc_weight'])
        statuses['Selected']={'converged':statuses['ICC']['converged'] and statuses['Sikov_SAR']['converged'],'recipe':recipe}
    elif recipe.get('stratum_deviations'):
        poverty=fit_icc_poverty(d['X'],d['y'],d['n_eff'],d['sample_idx'],units,d['basis'],strata=d['strata'],seed=seed+90001,sampler_config=SAMPLER,integration='hmc')
        preds['Selected']={'mean':poverty['p_mean'],'lower':poverty['lower95'],'upper':poverty['upper95'],'mse':poverty['p_variance']}
        statuses['Selected']={'converged':all(v['success'] and v['beta_shape_guard_cells']==0 for v in poverty['status']),'endpoints':poverty['status'],'recipe':recipe}
    else:preds['Selected']=preds['ICC'];statuses['Selected']={**statuses['ICC'],'recipe':recipe}
    for name,value in preds.items():
        arrays={k:np.asarray(v) for k,v in value.items() if isinstance(v,np.ndarray)}
        if not np.isfinite(np.stack([arrays[k] for k in ('mean','lower','upper')])).all():raise ValueError('Nonfinite predictions '+name)
        np.savez_compressed(out/(name+'.npz'),**arrays)
        save(out/(name+'_status.json'),statuses[name])
    return statuses

def work(job):
    context,rep=job;proof,freeze=gate();source=ROOT/'evaluacion/inputs'/context/f'rep_{rep:05d}'
    if not source.exists():generate(context,rep,'evaluation')
    manifest=json.loads((source/'manifest.json').read_text())
    d=dict(np.load(source/'model_inputs/inputs.npz',allow_pickle=False));units=pd.read_csv(source/'model_inputs/sample.csv')
    for name,key in [('inputs.npz','fit_inputs_sha256'),('sample.csv','sample_sha256')]:
        if sha(source/'model_inputs'/name)!=manifest[key]:raise RuntimeError('Changed fit input')
    out=ROOT/'evaluacion/fits'/context/f'rep_{rep:05d}'
    if out.exists():raise FileExistsError(out)
    seed=int(np.random.SeedSequence([20261010,811,CONTEXTS.index(context),rep]).generate_state(1)[0]) % (2**32-200000)
    before=time.perf_counter();status=fits(d,units,out,seed,proof['selected_recipe'])
    gate()  # A concurrent source change invalidates the fit before sealing it.
    completion={'context':context,'replication':rep,'protocol_sha256':freeze,'fit_inputs_sha256':manifest['fit_inputs_sha256'],'sample_sha256':manifest['sample_sha256'],'input_manifest_sha256':sha(source/'manifest.json'),'truth_read_by_fits':False,'seconds':time.perf_counter()-before,'convergence':{k:v['converged'] for k,v in status.items()},'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    save(out/'completion.json',completion)
    return completion

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--pilot',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not 1<=a.workers<=8:raise ValueError('workers1..8')
    proof,freeze=gate();jobs=[(c,r) for c in CONTEXTS for r in EVALUATION_IDS]
    if a.pilot:jobs=jobs[:1]
    if a.resume:jobs=[(c,r) for c,r in jobs if not (ROOT/'evaluacion/fits'/c/f'rep_{r:05d}/completion.json').exists()]
    failures=[];completed=0
    log=ROOT/'evaluacion/dispatch_events.jsonl'
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(work,job):job for job in jobs}
        for f in concurrent.futures.as_completed(futures):
            try:
                result=f.result();completed+=1
                event={'time_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'completed_this_dispatch':completed,'requested_this_dispatch':len(jobs),'world':[result['context'],result['replication']],'seconds':round(result['seconds'],2),'all_converged':all(result['convergence'].values())}
            except Exception as exc:
                event={'time_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'world':futures[f],'error':repr(exc),'traceback':traceback.format_exc()};failures.append(event)
            with log.open('a') as stream:stream.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
    save(ROOT/('evaluacion/pilot_dispatch.json' if a.pilot else 'evaluacion/final_dispatch.json'),{'protocol_sha256':freeze,'requested':len(jobs),'completed':completed,'failures':failures,'truth_read':False})
    if failures:raise RuntimeError('Failures preserved, no automatic score on partial cohort')
if __name__=='__main__':main()
