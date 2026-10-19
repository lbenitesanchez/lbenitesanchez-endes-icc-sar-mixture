#!/usr/bin/env python3
"""Truth-blind stage-two boundary, reuse and actual unresolved-draw checks."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,datetime,hashlib,json,sys,time
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--private-run-root',type=Path,required=True)
    ap.add_argument('--corrected-run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--solver-module',default='stable_reml_solver_stage2')
    args=ap.parse_args();root=args.repo.resolve()
    sys.path.insert(0,str(root/'src/estudio_publicacion/sikov_2026-10-10/scripts'))
    import sikov_model as sm
    import sikov_bootstrap as sb
    import stable_reml_solver as base
    import importlib
    ss=importlib.import_module(args.solver_module)
    from sikov_published_reference import triangular_weights
    hashes_before={str(p):sha(p)for p in [Path(base.__file__),root/'src/estudio_publicacion/sikov_2026-10-10/scripts/sikov_model.py',root/'src/estudio_publicacion/sikov_2026-10-10/scripts/sikov_bootstrap.py']}
    original_func=base.corrected_fit_stratum
    n=12;X=np.c_[np.ones(n),np.linspace(-1,1,n)];y=X@np.array([.3,.02]);D=np.repeat(.01,n)
    s=np.arange(n);W=np.roll(np.eye(n),1,axis=1);strata=np.ones(n,int);seed=20261018
    old=sb.fit_sikov_bootstrap(X,y,D,s,W,strata,spatial=False,B=100,seed=seed)
    ss.reset_counters();ss.install()
    try:new=sb.fit_sikov_bootstrap(X,y,D,s,W,strata,spatial=False,B=100,seed=seed)
    finally:ss.uninstall()
    keys=('mean','mse','mse_plugin','mse_delta','mse_naive','mse_bc','naive_mcse','bc_mcse','lower','upper')
    identity={k:bool(np.array_equal(old[k],new[k]))for k in keys};assert all(identity.values()),identity
    assert ss.get_counters().get('stage2_rescue_attempts',0)==0
    assert base.corrected_fit_stratum is original_func
    checks=[{'name':'FH_A0_same_seed_B100_exact_identity','arrays_identical':identity,'counters':ss.get_counters()}]
    plan=json.loads((args.corrected_run/'job_inventory.json').read_text())
    job=next(x for x in plan['jobs']if x['job_id']=='B300_subset_8/IID_logit_psu_sparse/rep_30000/published')
    status=json.loads((args.corrected_run/job['output_relative_dir']/job['status_filename']).read_text())
    d=dict(np.load(args.private_run_root/job['input_path'],allow_pickle=False));j=1
    names=d['covariate_names'].astype(str).tolist()
    mask=np.isfinite(d['y'][:,j])&np.isfinite(d['D'][:,j,j])&(d['D'][:,j,j]>0)
    sample=d['sample_idx'][mask];yy=d['y'][mask,j];DD=d['D'][mask,j,j]
    W=triangular_weights(d['coords'],sample,d['strata'])
    ss.reset_counters();ss.install();started=time.perf_counter()
    try:
        fit=sb.fit_sikov_bootstrap(d['X_raw'],yy,DD,sample,W,d['strata'],spatial=True,
                                  outcome='stunting',covariate_names=names,B=300,
                                  seed=job['endpoint_seeds'][j])
    finally:ss.uninstall()
    elapsed=time.perf_counter()-started
    baseline=status['fits'][j]
    assert np.array_equal(fit['mean'],np.asarray(baseline['mean']))
    changed_strata=[];previously_retained_parameter_identity=True;newly_retained=[]
    for newfit,oldfit in zip(fit['fit'],baseline['fit']):
        assert newfit['stratum']==oldfit['stratum']
        assert newfit['bootstrap_success']==300
        for a,b in zip(newfit['bootstrap_draw_diagnostics'],oldfit['bootstrap_draw_diagnostics']):
            assert a['draw_index_zero_based']==b['draw_index_zero_based']
            if b['retained']:
                assert a['retained']
                assert a['variance']==b['variance']and a['rho']==b['rho']
            elif a['retained']:
                newly_retained.append({'stratum':newfit['stratum'],'draw_index_zero_based':a['draw_index_zero_based'],
                                      'variance':a['variance'],'rho':a['rho'],
                                      'final_certificate':a['final_certificate'],
                                      'final_stable_certificate':a['final_stable_certificate'],
                                      'stage2_numerical_correction':a['stage2_numerical_correction']})
            assert a['final_certificate']['passed']and a['final_stable_certificate']['passed']
        ids=d['strata']==newfit['stratum']
        if not np.array_equal(fit['mse'][ids],np.asarray(baseline['mse'])[ids]):
            changed_strata.append(newfit['stratum'])
    assert len(newly_retained)==1 and newly_retained[0]['stratum']==2 and newly_retained[0]['draw_index_zero_based']==128
    assert changed_strata==[2]
    checks.append({'name':'actual_B300_IID30000_observed_SAR_stunting_same_seed',
                   'seed':job['endpoint_seeds'][j],'input_sha256':sha(args.private_run_root/job['input_path']),
                   'stage1_status_sha256':sha(args.corrected_run/job['output_relative_dir']/job['status_filename']),
                   'central_means_unchanged':True,'all_previously_retained_parameters_unchanged':True,
                   'strata_with_changed_MSE':changed_strata,'newly_retained_draws':newly_retained,
                   'stratum_success_counts':[f['bootstrap_success']for f in fit['fit']],
                   'elapsed_seconds':elapsed,'counters':ss.get_counters()})
    assert base.corrected_fit_stratum is original_func
    assert all(sha(Path(p))==h for p,h in hashes_before.items())
    report={'status':'PASS_STAGE2_SAME_MODEL_AND_SEED_RESCUE_WITH_STAGE1_PRESERVED',
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'truth_read':False,'stage1_sources_and_outputs_modified':False,
            'source_sha256':sha(Path(ss.__file__)),'base_solver_sha256':sha(Path(base.__file__)),
            'amendment_sha256':sha(root/'revision_20261015/protocols/reml_correction_stage2_20261018.json'),
            'checks':checks}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
