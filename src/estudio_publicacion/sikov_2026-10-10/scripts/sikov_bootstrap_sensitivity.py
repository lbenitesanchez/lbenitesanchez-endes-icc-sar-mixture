"""Predefined SAR REML bootstrap interval sensitivity; never reads truth.

Development smoke: context SAR_probability_srs, rep 0, B100, both endpoints.
Final sensitivity: reps 30000,30001 of four fixed contexts after source lock.
No choice of uncertainty estimator is made by this script or its outputs.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from sikov_bootstrap import fit_sikov_bootstrap

ROOT=Path(__file__).resolve().parents[1]
CONTEXTS=('SAR_probability_srs','SAR_logit_srs','SAR_logit_psu_sparse','IID_logit_psu_sparse')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def serializer(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return str(x)

def prediction_gate(lock,context,rep):
    """Require the full prediction seal before descriptive final sensitivity."""
    lock=Path(lock);sealed=json.loads(lock.read_text())
    protocol=ROOT/'protocol_freeze.json';frozen=json.loads(protocol.read_text())
    if not sealed.get('approved_for_scoring') or sealed.get('fresh_truth_read_before_lock') is not False:
        raise ValueError('a complete fresh prediction lock is required')
    if sealed.get('protocol_sha256')!=sha(protocol):
        raise ValueError('prediction lock and frozen protocol differ')
    if sealed.get('worlds')!=frozen.get('fresh_worlds'):
        raise ValueError('prediction lock is incomplete')
    for section in ('statistical_sources_sha256','frozen_inputs_sha256','runtime_binary_sha256'):
        for path,digest in frozen.get(section,{}).items():
            if sha(path)!=digest:raise RuntimeError('changed frozen file '+path)
    matching=[item for item in sealed['prediction_items'] if item['context']==context and item['replication']==rep]
    if len(matching)!=1:raise ValueError('sensitivity world absent from final lock')
    item=matching[0];folder=ROOT/'evaluacion/fits'/context/f'rep_{rep:05d}'
    if sha(folder/'completion.json')!=item['completion_sha256']:
        raise ValueError('changed completion seal')
    for filename,digest in item['outputs'].items():
        if sha(folder/filename)!=digest:raise ValueError('changed closed prediction '+filename)
    return sha(lock),folder

def run(phase,context,rep,B,lock=None):
    if phase=='development':
        if rep!=0:raise ValueError('smoke rep fixed to 0')
    elif rep not in (30000,30001):
        raise ValueError('final subset fixed before evaluation')
    if B<100:raise ValueError('B must be at least 100')
    if phase=='evaluation':
        if B<300:raise ValueError('prespecified final sensitivity requires B>=300')
        if lock is None or not Path(lock).is_file():
            raise ValueError('final bootstrap requires existing source/selection lock')
        lock_sha,closed_predictions=prediction_gate(lock,context,rep)
    else:lock_sha,closed_predictions=None,None
    folder=('desarrollo' if phase=='development' else 'evaluacion')
    path=ROOT/folder/'inputs'/context/f'rep_{rep:05d}'/'model_inputs'/'inputs.npz'
    input_sha=sha(path)
    d=dict(np.load(path,allow_pickle=False));start=time.monotonic()
    stem=f'sikov_bootstrap_{phase}_{context}_rep{rep:05d}_B{B}'
    out=ROOT/'validacion'/f'{stem}.npz'
    if out.exists():raise FileExistsError('preserve completed sensitivity '+str(out))
    results={};metadata=[]
    source_sha=sha(ROOT/'scripts'/'sikov_model.py')
    bootstrap_sha=sha(ROOT/'scripts'/'sikov_bootstrap.py')
    for t,name in enumerate(('anemia','stunting')):
        mask=np.isfinite(d['y'][:,t])&np.isfinite(d['D'][:,t,t])&(d['D'][:,t,t]>0)
        f=fit_sikov_bootstrap(d['X_raw'],d['y'][mask,t],d['D'][mask,t,t],d['sample_idx'][mask],d['W'],d['strata'],spatial=True,outcome=name,covariate_names=d['covariate_names'].astype(str).tolist(),B=B,seed=202610100000+CONTEXTS.index(context)*1000000+rep*2+t,interval_estimator='naive')
        for key in ('mean','mse','mse_naive','mse_bc','naive_mcse','bc_mcse','mse_plugin','mse_delta','lower','upper'):
            results[f'{name}_{key}']=f[key]
        metadata.append({'outcome':name,'converged':f['converged'],'fit':f['fit'],'bootstrap_vs_delta_median_ratio':float(np.median(f['mse']/np.maximum(f['mse_delta'],1e-12)))})
    point_difference=None
    if closed_predictions is not None:
        closed=dict(np.load(closed_predictions/'Sikov_SAR.npz',allow_pickle=False))
        sensitivity_mean=np.column_stack([results[f'{endpoint}_mean'] for endpoint in ('anemia','stunting')])
        point_difference=float(np.max(np.abs(sensitivity_mean-closed['mean'])))
        if point_difference>1e-10:raise RuntimeError('B sensitivity altered the closed point estimator')
    if source_sha!=sha(ROOT/'scripts'/'sikov_model.py'):
        raise RuntimeError('source changed during bootstrap; result invalidated')
    if bootstrap_sha!=sha(ROOT/'scripts'/'sikov_bootstrap.py'):
        raise RuntimeError('bootstrap source changed; result invalidated')
    if not path.is_file() or input_sha!=sha(path):
        raise RuntimeError('inputs changed during bootstrap; no output saved')
    if lock is not None and sha(lock)!=lock_sha:raise RuntimeError('prediction lock changed during sensitivity')
    np.savez_compressed(out,**results)
    summary={'phase':phase,'context':context,'rep':rep,'B':B,'elapsed_seconds':time.monotonic()-start,'input_sha256':input_sha,'model_source_sha256':source_sha,'bootstrap_source_sha256':bootstrap_sha,'script_sha256':sha(__file__),'lock_sha256':lock_sha,'protocol_sha256':sha(ROOT/'protocol_freeze.json') if phase=='evaluation' else None,'closed_point_max_absolute_difference':point_difference,'outputs_sha256':sha(out),'truth_was_read':False,'uncertainty_selection_performed':False,'bootstrap_randomization':'independent prespecified B300 streams, same draws shared by naive and BC','primary_uncertainty':'prespecified naive parametric bootstrap B100; B300 and BC descriptive sensitivity','metadata':metadata}
    (ROOT/'validacion'/f'{stem}.json').write_text(json.dumps(summary,indent=2,default=serializer)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='metadata'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','evaluation'],default='development');p.add_argument('--context',choices=CONTEXTS,default=CONTEXTS[0]);p.add_argument('--rep',type=int,default=0);p.add_argument('--B',type=int,default=100);p.add_argument('--lock',type=Path)
    a=p.parse_args();run(a.phase,a.context,a.rep,a.B,a.lock)
