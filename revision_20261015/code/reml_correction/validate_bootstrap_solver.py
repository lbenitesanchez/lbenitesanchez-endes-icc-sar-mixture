#!/usr/bin/env python3
"""Same-seed bootstrap identity/control and a truth-blind B100 pilot."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,datetime,hashlib,json,sys,time
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--private-run-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();root=args.repo.resolve()
    sys.path.insert(0,str(root/'src/estudio_publicacion/sikov_2026-10-10/scripts'))
    import sikov_model as sm
    import sikov_bootstrap as sb
    from sikov_published_reference import triangular_weights
    import stable_reml_solver as ss
    n=12;X=np.c_[np.ones(n),np.linspace(-1,1,n)];y=X@np.array([.3,.02]);D=np.repeat(.01,n)
    s=np.arange(n);W=np.roll(np.eye(n),1,axis=1);strata=np.ones(n,int)
    seed=20261018;normal_checksums=[];actual_rng=np.random.default_rng
    class RecordingRNG:
        def __init__(self,seed):self.rng=actual_rng(seed)
        def normal(self,*a,**k):
            value=self.rng.normal(*a,**k)
            normal_checksums.append({'shape':list(value.shape),'sha256':hashlib.sha256(value.tobytes()).hexdigest()})
            return value
    np.random.default_rng=RecordingRNG
    try:
        old=sb.fit_sikov_bootstrap(X,y,D,s,W,strata,spatial=False,B=100,seed=seed)
        old_draws=normal_checksums.copy();normal_checksums.clear()
        ss.reset_counters();ss.install()
        try:new=sb.fit_sikov_bootstrap(X,y,D,s,W,strata,spatial=False,B=100,seed=seed)
        finally:ss.uninstall()
        new_draws=normal_checksums.copy()
    finally:np.random.default_rng=actual_rng
    assert old_draws==new_draws
    assert new['numerical_correction']['counters'].get('rescue_attempts',0)==0,new['numerical_correction']
    keys=('mean','mse','mse_plugin','mse_delta','mse_naive','mse_bc','naive_mcse','bc_mcse','lower','upper')
    identity={k:bool(np.array_equal(old[k],new[k]))for k in keys};assert all(identity.values()),identity
    assert len(new['fit'][0]['bootstrap_draw_diagnostics'])==100
    assert all(r['final_certificate']['passed']and r['final_stable_certificate']['passed']for r in new['fit'][0]['bootstrap_draw_diagnostics'])
    checks=[{'name':'FH_A0_same_seed_B100_exact_array_identity','arrays_byte_identical':identity,'normal_arrays_identical':True,
             'standard_normal_array_checksums':old_draws,'counters':new['numerical_correction']['counters']}]
    # Explicit zero innovations also exercise the same A0 spatial bootstrap path.
    innovations={1:{'u':np.zeros((100,n)),'e':np.zeros((100,n))}}
    old=sb.fit_sikov_bootstrap(X,y,D,s,W,strata,spatial=True,B=100,seed=seed,innovations=innovations)
    ss.reset_counters();ss.install()
    try:new=sb.fit_sikov_bootstrap(X,y,D,s,W,strata,spatial=True,B=100,seed=seed,innovations=innovations)
    finally:ss.uninstall()
    identity={k:bool(np.array_equal(old[k],new[k]))for k in keys};assert all(identity.values()),identity
    checks.append({'name':'SAR_A0_explicit_zero_innovations_exact_array_identity','arrays_byte_identical':identity,
                   'counters':new['numerical_correction']['counters']})
    # Controlled original fitter exception must be rescued with unavailable raw metadata.
    original_fit=ss._ORIGINAL_FIT
    def numerical_exception(*args,**kwargs):raise np.linalg.LinAlgError('controlled numerical exception')
    ss._ORIGINAL_FIT=numerical_exception
    try:exception_fit=ss.corrected_fit_stratum(X,y,D,s,W,False)
    finally:ss._ORIGINAL_FIT=original_fit
    assert exception_fit['converged'] and exception_fit['raw_optimizer_converged'] is None
    assert exception_fit['numerical_correction']['original_variance']is None
    checks.append({'name':'numerical_exception_recovery_without_invented_raw_status',
                   'metadata':exception_fit['numerical_correction'],'final_certificate':exception_fit['stationarity_certificate']})
    # Actual frozen difficult case, same original seed, no evaluation truth loaded.
    context='IID_logit_psu_sparse';rep=50026;j=1
    path=args.private_run_root/'sparse57/evaluacion/inputs'/context/f'rep_{rep:05d}/model_inputs/inputs.npz'
    d=dict(np.load(path,allow_pickle=False));names=d['covariate_names'].astype(str).tolist()
    mask=np.isfinite(d['y'][:,j])&np.isfinite(d['D'][:,j,j])&(d['D'][:,j,j]>0)
    sample=d['sample_idx'][mask];yy=d['y'][mask,j];DD=d['D'][mask,j,j]
    W=triangular_weights(d['coords'],sample,d['strata'])
    ss.reset_counters();ss.install();started=time.perf_counter()
    try:pilot=sb.fit_sikov_bootstrap(d['X_raw'],yy,DD,sample,W,d['strata'],spatial=True,
                                    outcome='stunting',covariate_names=names,B=100,seed=3647776684)
    finally:ss.uninstall()
    elapsed=time.perf_counter()-started
    assert pilot['converged']
    for fit in pilot['fit']:
        assert fit['stationarity_certificate']['passed'] and fit['stable_stationarity_certificate']['passed']
        assert fit['bootstrap_success']>=95
        for draw in fit['bootstrap_draw_diagnostics']:
            if draw['retained']:assert draw['final_certificate']['passed']and draw['final_stable_certificate']['passed']
    checks.append({'name':'difficult_sparse_IID50026_observed_SAR_stunting_B100','seed':3647776684,
                   'elapsed_seconds':elapsed,'counters':pilot['numerical_correction']['counters'],
                   'strata':[{'stratum':f['stratum'],'success':f['bootstrap_success'],'failed':f['bootstrap_failed'],
                              'rescued':f['bootstrap_numerical_rescued'],
                              'original_cert_failed':f['bootstrap_original_certificate_failed']}for f in pilot['fit']]})
    report={'status':'PASS_BOOTSTRAP_SAME_SEED_IDENTITY_EXCEPTION_AND_B100_PILOT','truth_read':False,
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'solver_sha256':hashlib.sha256(Path(ss.__file__).read_bytes()).hexdigest(),'checks':checks}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
