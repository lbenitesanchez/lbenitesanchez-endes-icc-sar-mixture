#!/usr/bin/env python3
"""Regenerate registered, rejected bootstrap draws without evaluation truth.

This diagnostic does not replace any official fit. It tests stage-two recovery
on the fitting-only unresolved cases in a saved independent verification cut.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,datetime,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--private-run-root',type=Path,required=True);ap.add_argument('--corrected-run',type=Path,required=True)
    ap.add_argument('--verification-cut',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--job-id',action='append')
    ap.add_argument('--solver-module',default='stable_reml_solver_stage2')
    args=ap.parse_args();repo=args.repo.resolve()
    sys.path.insert(0,str(repo/'src/estudio_publicacion/sikov_2026-10-10/scripts'))
    import sikov_model as sm
    import importlib
    stage2=importlib.import_module(args.solver_module)
    from sikov_published_reference import triangular_weights
    plan=json.loads((args.corrected_run/'job_inventory.json').read_text())
    byid={j['job_id']:j for j in plan['jobs']}
    cut=json.loads(args.verification_cut.read_text())
    selected=[j['job_id']for j in cut['job_summaries']if j['bootstrap_retained']<j['bootstrap_attempts']]
    if args.job_id:
        selected=args.job_id
    results=[]
    for jobid in selected:
        job=byid[jobid];folder=args.corrected_run/job['output_relative_dir']
        statuspath=folder/job['status_filename'];status=json.loads(statuspath.read_text())
        values=status['status'if job['development_status_schema']else'fits']
        d=dict(np.load(args.private_run_root/job['input_path'],allow_pickle=False))
        names=d['covariate_names'].astype(str).tolist()
        if job['reverse_coordinate_columns']:d['coords']=d['coords'][:,::-1]
        cases=[]
        for endpoint,value in enumerate(values):
            mask=np.isfinite(d['y'][:,endpoint])&np.isfinite(d['D'][:,endpoint,endpoint])&(d['D'][:,endpoint,endpoint]>0)
            sample=d['sample_idx'][mask];D=d['D'][mask,endpoint,endpoint]
            W=(triangular_weights(d['coords'],sample,d['strata'])if job['graph']=='observed'else d['W'])
            rng=np.random.default_rng(job['endpoint_seeds'][endpoint]);B=job['bootstrap_B']
            for fit in value['fit']:
                level=fit['stratum'];ids=np.flatnonzero(d['strata']==level)
                obs=np.isin(sample,ids);s=np.searchsorted(ids,sample[obs]);DD=D[obs]
                cols=sm.select_columns(('anemia','stunting')[endpoint],level,names,job['spatial'])
                X=d['X_raw'][ids][:,cols];WW=W[np.ix_(ids,ids)].copy()
                sums=WW.sum(axis=1);WW=np.divide(WW,sums[:,None],out=np.zeros_like(WW),where=sums[:,None]>0)
                U=rng.normal(size=(B,len(ids)));E=rng.normal(size=(B,len(s)))
                factor=splu(sparse.eye(len(ids),format='csc')-fit['rho']*sparse.csc_matrix(WW))if job['spatial']else None
                for previous in fit['bootstrap_draw_diagnostics']:
                    if previous['retained']:continue
                    b=previous['draw_index_zero_based']
                    innovation=np.sqrt(fit['variance'])*U[b]
                    latent=factor.solve(innovation)if factor is not None else innovation
                    latent_target=X@np.asarray(fit['beta'])+latent
                    bootstrap_y=latent_target[s]+np.sqrt(DD)*E[b]
                    tick=time.perf_counter()
                    answer=stage2.corrected_fit_stratum(X,bootstrap_y,DD,s,WW,job['spatial'])
                    elapsed=time.perf_counter()-tick
                    meta=answer['numerical_correction'].get('stage2_numerical_correction')
                    require_attempt=not(previous['final_certificate']['passed']and previous['final_stable_certificate']['passed'])
                    assert meta is not None if require_attempt else True
                    if meta is not None:
                        original_delta=abs(meta['stage1_final_original_certificate']['projected_gradient_max']-previous['final_certificate']['projected_gradient_max'])
                        stable_delta=abs(meta['stage1_final_stable_certificate']['projected_gradient_max']-previous['final_stable_certificate']['projected_gradient_max'])
                        assert max(original_delta,stable_delta)<1e-7,(jobid,b,original_delta,stable_delta)
                    else:original_delta=stable_delta=None
                    cases.append({'endpoint':('anemia','stunting')[endpoint],'stratum':level,'draw_index_zero_based':b,
                                  'stage1_certificate_reproduction_max_abs_error':original_delta,
                                  'stage1_stable_certificate_reproduction_max_abs_error':stable_delta,
                                  'stage1_original_certificate':previous['final_certificate'],
                                  'stage1_stable_certificate':previous['final_stable_certificate'],
                                  'recovered':bool(answer['converged']),
                                  'final_original_certificate':answer['stationarity_certificate'],
                                  'final_stable_certificate':answer['stable_stationarity_certificate'],
                                  'stage2_numerical_correction':meta,'elapsed_seconds':elapsed})
        results.append({'job_id':jobid,'input_sha256':sha(args.private_run_root/job['input_path']),
                        'stage1_status_sha256':sha(statuspath),'cases':cases})
        print(json.dumps({'job_id':jobid,'draws':len(cases),'recovered':sum(c['recovered']for c in cases)}),flush=True)
    cases=[c for r in results for c in r['cases']]
    result={'status':'PASS_ALL_SELECTED_UNRESOLVED_DRAWS_RECOVERED'if all(c['recovered']for c in cases)else'UNRESOLVED_STAGE2_DRAWS_REMAIN',
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'truth_read':False,'official_predictions_modified':False,'new_central_parameters_selected':False,
            'stage2_source_sha256':sha(Path(stage2.__file__)),
            'base_source_sha256':sha(Path(stage2.base.__file__)),
            'verification_cut_sha256':sha(args.verification_cut),
            'selected_jobs':len(results),'tested_draws':len(cases),'recovered_draws':sum(c['recovered']for c in cases),
            'results':results,'counters':stage2.get_counters()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['status'],result['tested_draws'],result['recovered_draws'])


if __name__=='__main__':main()
