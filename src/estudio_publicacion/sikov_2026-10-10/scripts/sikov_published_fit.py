"""Additional reference fits under a separate pre-outcome addendum seal.

Preserves all base fits, predictions and selection. Only training arrays are
read. Fresh jobs start after all 256 original worlds close; real jobs start
after their corresponding original real fit closes.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse,concurrent.futures,datetime as dt,hashlib,json,time,traceback
import numpy as np
import pandas as pd
from sikov_published_reference import fit_sikov_published_bootstrap,METHOD,BC_METHOD

ROOT=Path(__file__).resolve().parents[1]
ADDENDUM=ROOT/'validacion/sikov_published_addendum_freeze.json'
CONTEXTS=('SAR_probability_srs','SAR_logit_srs','SAR_logit_psu_sparse','IID_logit_psu_sparse')
IDS=tuple(range(30000,30064));OUTCOMES=('anemia','stunting')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):Path(path).write_text(json.dumps(data,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n')

def gate():
    base=ROOT/'protocol_freeze.json';add=json.loads(ADDENDUM.read_text())
    if not add['approved_for_additional_reference_fits'] or add['accuracy_or_heldout_outcomes_read_before_addendum']:
        raise RuntimeError('Approved pre-outcome addendum required')
    if add['base_protocol_sha256']!=sha(base):raise RuntimeError('Changed base protocol')
    frozen=json.loads(base.read_text())
    for block in ('statistical_sources_sha256','frozen_inputs_sha256','runtime_binary_sha256'):
        for path,digest in frozen.get(block,{}).items():
            if sha(path)!=digest:raise RuntimeError('Changed base frozen file '+path)
    for path,digest in add['additional_sources_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Changed additional frozen file '+path)
    return add,sha(ADDENDUM)

def fit_arrays(d,out,seed):
    names=d['covariate_names'].astype(str).tolist();values=[];out.mkdir(parents=True,exist_ok=False)
    masks=[np.isfinite(d['y'][:,j])&np.isfinite(d['D'][:,j,j])&(d['D'][:,j,j]>0) for j in range(2)]
    for j,endpoint in enumerate(OUTCOMES):
        mask=masks[j]
        values.append(fit_sikov_published_bootstrap(d['X_raw'],d['y'][mask,j],d['D'][mask,j,j],d['sample_idx'][mask],d['coords'],d['strata'],outcome=endpoint,covariate_names=names,B=100,seed=seed+j*1009,interval_estimator='naive'))
    primary={key:np.column_stack([v[key] for v in values]) for key in ('mean','lower','upper','mse','mse_plugin','mse_delta','mse_naive','mse_bc','naive_mcse','bc_mcse')}
    bc={key:value.copy() for key,value in primary.items()};bc['mse']=bc['mse_bc'];se=np.sqrt(np.maximum(bc['mse'],0));bc['lower']=bc['mean']-1.959963984540054*se;bc['upper']=bc['mean']+1.959963984540054*se
    for name,prediction in ((METHOD,primary),(BC_METHOD,bc)):
        if not np.isfinite(np.stack([prediction[k] for k in ('mean','lower','upper')])).all():raise ValueError('Non-finite additional prediction '+name)
        np.savez_compressed(out/(name+'.npz'),**prediction)
    status={'converged':all(v['converged'] for v in values),'fits':values,'endpoint_training_rows':[int(m.sum()) for m in masks]}
    save(out/(METHOD+'_status.json'),status)
    save(out/(BC_METHOD+'_status.json'),{'converged':status['converged'],'negative_bc_MSE_cells':int((bc['mse_bc']<0).sum()),'negative_MSE_floor_only_for_interval_sqrt':0.,'same_bootstrap_draws_as_primary':True})
    return primary,bc,status

def work(job):
    context,rep=job;add,addsha=gate();source=ROOT/'evaluacion/inputs'/context/f'rep_{rep:05d}';old=ROOT/'evaluacion/fits'/context/f'rep_{rep:05d}/completion.json'
    if not old.is_file():raise RuntimeError('Original world must close first')
    original=json.loads(old.read_text());manifest=json.loads((source/'manifest.json').read_text())
    model=source/'model_inputs/inputs.npz'
    if original['protocol_sha256']!=add['base_protocol_sha256'] or original['truth_read_by_fits'] or sha(model)!=manifest['fit_inputs_sha256'] or sha(source/'manifest.json')!=original['input_manifest_sha256']:
        raise RuntimeError('Wrong training array or original seal')
    d=dict(np.load(model,allow_pickle=False));out=ROOT/'evaluacion/published_sar'/context/f'rep_{rep:05d}'
    seed=int(np.random.SeedSequence([20261010,941,CONTEXTS.index(context),rep]).generate_state(1)[0])
    before=time.perf_counter();_,_,status=fit_arrays(d,out,seed);gate()
    result={'context':context,'replication':rep,'protocol_sha256':add['base_protocol_sha256'],'addendum_sha256':addsha,'original_completion_sha256':sha(old),'fit_inputs_sha256':sha(model),'input_manifest_sha256':sha(source/'manifest.json'),'truth_read_by_fits':False,'bootstrap_seed':seed,'seconds':time.perf_counter()-before,'convergence':{METHOD:status['converged'],BC_METHOD:status['converged']},'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    save(out/'completion.json',result);return result

def real(scope):
    add,addsha=gate();oldfolder=ROOT/'evaluacion'/('real_'+scope);old=oldfolder/'completion.json'
    if not old.is_file():raise RuntimeError('Original real fit must close first')
    original=json.loads(old.read_text());prepared=ROOT/'datos'/('real_2019_full' if scope=='temporal' else 'real_2024_training');manifest=json.loads((prepared/'preparation_manifest.json').read_text())
    if manifest['heldout_outcome_rows_read'] or original['heldout_outcomes_read_by_fits'] or original['protocol_sha256']!=add['base_protocol_sha256'] or sha(prepared/'preparation_manifest.json')!=original['training_manifest_sha256']:
        raise RuntimeError('Wrong real training or original seal')
    for name,digest in manifest['output_sha256'].items():
        if sha(prepared/name)!=digest:raise RuntimeError('Changed real training input')
    model=prepared/'fit_inputs.npz';d=dict(np.load(model,allow_pickle=False))
    if manifest['coordinates_npz_order']!=['latitude','longitude']:raise RuntimeError('Wrong real coordinate order')
    d['coords']=d['coords'][:,::-1]
    out=ROOT/'evaluacion/published_sar'/('real_'+scope);seed=int(np.random.SeedSequence([20261010,942,0 if scope=='temporal' else 1]).generate_state(1)[0]);before=time.perf_counter()
    primary,bc,status=fit_arrays(d,out,seed);ids=pd.read_csv(prepared/'area_frame.csv',dtype={'ubigeo':str});rows=[]
    for name,prediction in ((METHOD,primary),(BC_METHOD,bc)):
        for i,code in enumerate(ids.ubigeo):
            rows.append({'ubigeo':code,'method':name,**{f'{prefix}{endpoint}':prediction[key][i,j] for j,endpoint in enumerate(OUTCOMES) for prefix,key in (('p_','mean'),('lo95_','lower'),('hi95_','upper'))}})
    pred=out/'predictions.csv';pd.DataFrame(rows).to_csv(pred,index=False);gate()
    completion={'scope':scope,'protocol_sha256':add['base_protocol_sha256'],'addendum_sha256':addsha,'original_completion_sha256':sha(old),'original_predictions_sha256':sha(oldfolder/'predictions.csv'),'training_manifest_sha256':sha(prepared/'preparation_manifest.json'),'fit_inputs_sha256':sha(model),'heldout_outcomes_read_by_fits':False,'bootstrap_seed':seed,'predictions_sha256':sha(pred),'seconds':time.perf_counter()-before,'convergence':{METHOD:status['converged'],BC_METHOD:status['converged']},'all_converged':status['converged'],'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    save(out/'completion.json',completion)
    save(ROOT/f'validacion/sikov_published_prediction_lock_{scope}.json',{'scope':scope,'protocol_sha256':add['base_protocol_sha256'],'addendum_sha256':addsha,'approved_for_scoring':False,'requires_complete_joint_prediction_lock':True,'heldout_outcomes_used':False,'completion_sha256':sha(out/'completion.json'),'predictions_sha256':sha(pred),'predictions_path':str(pred),'locked_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    print(json.dumps({k:v for k,v in completion.items() if k!='output_sha256'},indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=('temporal','territorial'));parser.add_argument('--workers',type=int,default=4);parser.add_argument('--resume',action='store_true');args=parser.parse_args();gate()
    if args.scope:real(args.scope);return
    if not 1<=args.workers<=4:raise ValueError('Additional fresh workers must be1..4')
    jobs=[(c,r) for c in CONTEXTS for r in IDS]
    if not all((ROOT/'evaluacion/fits'/c/f'rep_{r:05d}/completion.json').is_file() for c,r in jobs):raise RuntimeError('All original fresh jobs must close before additional dispatch')
    if args.resume:jobs=[(c,r) for c,r in jobs if not (ROOT/'evaluacion/published_sar'/c/f'rep_{r:05d}/completion.json').is_file()]
    events=ROOT/'evaluacion/published_sar_dispatch_events.jsonl';errors=[];done=0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(work,job):job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                result=future.result();done+=1;event={'world':futures[future],'completed':done,'requested':len(jobs),'seconds':result['seconds'],'convergence':result['convergence']}
            except Exception as exc:event={'world':futures[future],'error':repr(exc),'traceback':traceback.format_exc()};errors.append(event)
            with events.open('a') as stream:stream.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
    save(ROOT/'evaluacion/published_sar_dispatch.json',{'requested':len(jobs),'completed':done,'errors':errors,'addendum_sha256':sha(ADDENDUM),'truth_read':False})
    if errors:raise RuntimeError('Additional fits incomplete; preserve failures and block scoring')

if __name__=='__main__':main()
