#!/usr/bin/env python3
"""Truth-blind deterministic tests of same-model REML rescue and algebra."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,datetime,json,sys,hashlib,time
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--private-run-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();root=args.repo.resolve()
    sys.path.insert(0,str(root/'src/estudio_publicacion/sikov_2026-10-10/scripts'))
    import sikov_model as sm
    from sikov_published_reference import triangular_weights
    import stable_reml_solver as ss
    tests=[]
    for context,rep,stratum,method in [('IID_logit_psu_sparse',50026,1,'observed'),('SAR_probability_srs',50057,1,'observed'),('SAR_logit_psu_sparse',50017,3,'full'),('SAR_probability_srs',50041,3,'observed')]:
        p=args.private_run_root/'sparse57/evaluacion/inputs'/context/f'rep_{rep:05d}/model_inputs/inputs.npz'
        d=dict(np.load(p,allow_pickle=False));j=1;names=d['covariate_names'].astype(str).tolist()
        mask=np.isfinite(d['y'][:,j])&np.isfinite(d['D'][:,j,j])&(d['D'][:,j,j]>0)
        s0=d['sample_idx'][mask];y=d['y'][mask,j];D=d['D'][mask,j,j]
        W=d['W']if method=='full'else triangular_weights(d['coords'],s0,d['strata'])
        ids=np.flatnonzero(d['strata']==stratum);keep=np.isin(s0,ids);s=np.searchsorted(ids,s0[keep])
        XX=d['X_raw'][ids][:,sm.select_columns('stunting',stratum,names,True)];yy=y[keep];DD=D[keep]
        WW=W[np.ix_(ids,ids)].copy();WW/=WW.sum(axis=1)[:,None]
        old=ss._ORIGINAL_FIT(XX,yy,DD,s,WW,True)
        control=ss._predict_at(XX,yy,DD,s,WW,True,old['variance'],old['rho'])
        errors={k:float(np.max(np.abs(control[k]-old[k])))for k in ('mean','mse_plugin','mse_delta','beta','beta_cov')}
        assert max(errors.values())<1e-9,errors
        started=time.perf_counter();new=ss.corrected_fit_stratum(XX,yy,DD,s,WW,True);elapsed=time.perf_counter()-started
        assert new['converged']and new['stationarity_certificate']['passed'],new['numerical_correction']
        assert new['numerical_correction']['repaired']
        assert new['numerical_correction']['stable_nll_change']<=1e-8
        tests.append({'context':context,'replication':rep,'stratum':stratum,'method':method,
                      'original_prediction_algebra_max_abs_errors':errors,'corrected_variance':new['variance'],
                      'corrected_rho':new['rho'],'numerical_correction':new['numerical_correction'],
                      'final_certificate':new['stationarity_certificate'],'elapsed_seconds':elapsed})
        print(context,rep,method,elapsed,new['variance'],new['rho'],new['stationarity_certificate'],flush=True)
    # FH variance-zero boundary: a response lying exactly in the fixed-effect space.
    rng=np.random.default_rng(123);n=12;X=np.c_[np.ones(n),np.linspace(-1,1,n)]
    y=X@np.array([.3,.02]);D=np.repeat(.01,n);s=np.arange(n);W=np.roll(np.eye(n),1,axis=1)
    for spatial in (False,True):
        new=ss.corrected_fit_stratum(X,y,D,s,W,spatial)
        assert new['variance']==0 and new['stationarity_certificate']['passed']
        tests.append({'test':'exact_fixed_effect_variance_zero','spatial':spatial,'variance':new['variance'],'rho':new['rho'],'final_certificate':new['stationarity_certificate']})
    # A certified historical result must preserve all numerical arrays exactly.
    new1=ss._ORIGINAL_FIT(X,y+rng.normal(0,.1,n),D,s,W,False)
    # Same deterministic response in both calls, separate random draw already frozen.
    yy=X@np.array([.3,.02])+np.linspace(-.1,.1,n)**2
    old=ss._ORIGINAL_FIT(X,yy,D,s,W,False);new=ss.corrected_fit_stratum(X,yy,D,s,W,False)
    assert not new['numerical_correction']['repaired']
    assert all(np.array_equal(old[k],new[k])for k in ('mean','mse_plugin','mse_delta','beta','beta_cov'))
    result={'status':'PASS_SAME_MODEL_FOUR_RESCUES_BOUNDARY_AND_ALGEBRA','truth_read':False,'originals_modified':False,
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'solver_sha256':hashlib.sha256(Path(ss.__file__).read_bytes()).hexdigest(),'tests':tests,'counters':ss.get_counters()}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'])


if __name__=='__main__':main()
