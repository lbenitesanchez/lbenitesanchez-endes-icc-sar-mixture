"""Independent algebra and native bootstrap QA for the additive reference."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import hashlib,json,time
import numpy as np
from sikov_published_reference import triangular_weights,fit_sikov_published_bootstrap
from sikov_model import fit_sikov,knn_weights,select_columns,_SAR

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    before=time.monotonic();rng=np.random.default_rng(20261010943)
    N=60;strata=np.repeat([1,2,3],20);s=np.concatenate([np.arange(i*20,i*20+15) for i in range(3)])
    coords=rng.uniform(size=(N,2));names=['intercept','Altitud','PisoTierra','Cemento','Agua','Elect','Desague','Internet','SIS','EsSalud','NoSeguro','Refrig','Castellano','Analfabet','Rural']
    X=np.column_stack([np.ones(N),rng.uniform(size=(N,14))]);y=rng.uniform(.12,.42,len(s));D=rng.uniform(.003,.008,len(s))
    W=triangular_weights(coords,s,strata);WD=knn_weights(coords[s],k=6,strata=strata[s])
    full=fit_sikov(X,y,D,s,W,strata,spatial=True,outcome='anemia',covariate_names=names,uncertainty='plugin')
    observed=fit_sikov(X[s],y,D,np.arange(len(s)),WD,strata[s],spatial=True,outcome='anemia',covariate_names=names,uncertainty='plugin')
    errors={'sampled_points':float(abs(full['mean'][s]-observed['mean']).max()),'sampled_plugin_MSE':float(abs(full['mse_plugin'][s]-observed['mse_plugin']).max()),'sampled_covariance':0.,'sampled_covariance_relative':0.,'fixed_interior_covariance':0.,'fixed_A_rho_sampled_predictor':0.,'fixed_A_rho_nonsampled_formula':0.,'nonsampled_formula':0.,'REML_nll':0.,'dense_all_area_point':0.}
    conditions=[]
    for a,b in zip(full['fit'],observed['fit']):
        ids=np.flatnonzero(strata==a['stratum']);obs=np.isin(s,ids);ss=np.searchsorted(ids,s[obs]);uu=np.setdiff1d(np.arange(len(ids)),ss);cols=select_columns('anemia',a['stratum'],names,True);XX=X[ids][:,cols]
        WW=W[np.ix_(ids,ids)];wobs=WD[np.ix_(np.flatnonzero(strata[s]==a['stratum']),np.flatnonzero(strata[s]==a['stratum']))]
        rho=a['rho'];A=a['variance'];inv=np.linalg.inv(np.eye(len(ids))-rho*WW);C=inv@inv.T;invD=np.linalg.inv(np.eye(len(ss))-rho*wobs);CD=invD@invD.T
        difference=float(abs(C[np.ix_(ss,ss)]-CD).max())
        errors['sampled_covariance']=max(errors['sampled_covariance'],difference)
        errors['sampled_covariance_relative']=max(errors['sampled_covariance_relative'],difference/max(float(abs(CD).max()),1.))
        probe=np.linalg.inv(np.eye(len(ids))-.37*WW);probeD=np.linalg.inv(np.eye(len(ss))-.37*wobs)
        errors['fixed_interior_covariance']=max(errors['fixed_interior_covariance'],float(abs((probe@probe.T)[np.ix_(ss,ss)]-probeD@probeD.T).max()))
        Cs,_,Call,_,_=_SAR(WW,ss,True).covariance(.37,full=True)
        Vprobe=.01*Cs+np.diag(D[obs]);Viprobe=np.linalg.inv(Vprobe)
        betaprobe=np.linalg.solve(XX[ss].T@Viprobe@XX[ss],XX[ss].T@Viprobe@y[obs])
        predictionprobe=XX@betaprobe+.01*Call@Viprobe@(y[obs]-XX[ss]@betaprobe)
        CDprobe=probeD@probeD.T;Vid=np.linalg.inv(.01*CDprobe+np.diag(D[obs]));betad=np.linalg.solve(XX[ss].T@Vid@XX[ss],XX[ss].T@Vid@y[obs]);uhatd=.01*CDprobe@Vid@(y[obs]-XX[ss]@betad)
        errors['fixed_A_rho_sampled_predictor']=max(errors['fixed_A_rho_sampled_predictor'],float(abs(predictionprobe[ss]-XX[ss]@betad-uhatd).max()))
        Fprobe=np.linalg.solve(np.eye(len(uu))-.37*WW[np.ix_(uu,uu)],.37*WW[np.ix_(uu,ss)])
        errors['fixed_A_rho_nonsampled_formula']=max(errors['fixed_A_rho_nonsampled_formula'],float(abs(predictionprobe[uu]-XX[uu]@betad-Fprobe@uhatd).max()))
        conditions.append({'stratum':a['stratum'],'rho':rho,'condition_I_minus_rho_W':float(np.linalg.cond(np.eye(len(ids))-rho*WW))})
        errors['REML_nll']=max(errors['REML_nll'],abs(a['reml_nll']-b['reml_nll']))
        Vi=np.linalg.inv(A*CD+np.diag(D[obs]));beta=np.linalg.solve(XX[ss].T@Vi@XX[ss],XX[ss].T@Vi@y[obs]);uhat=A*CD@Vi@(y[obs]-XX[ss]@beta)
        F=np.linalg.solve(np.eye(len(uu))-rho*WW[np.ix_(uu,uu)],rho*WW[np.ix_(uu,ss)])
        predicted0=XX[uu]@beta+F@uhat;errors['nonsampled_formula']=max(errors['nonsampled_formula'],float(abs(predicted0-full['mean'][ids[uu]]).max()))
        dense=XX@beta+A*C[:,ss]@Vi@(y[obs]-XX[ss]@beta);errors['dense_all_area_point']=max(errors['dense_all_area_point'],float(abs(dense-full['mean'][ids]).max()))
    boot=fit_sikov_published_bootstrap(X,y,D,s,coords,strata,outcome='anemia',covariate_names=names,B=12,seed=20261010944)
    errors['bootstrap_keeps_point']=float(abs(boot['mean']-full['mean']).max())
    checks={
        'sampled_to_unsampled_block_zero':bool(np.all(W[np.ix_(s,np.setdiff1d(np.arange(N),s))]==0)),
        'sampled_graph_exactly_kNN6_on_sampled_only':bool(np.array_equal(W[np.ix_(s,s)],WD)),
        'non_sampled_rows_kNN6_on_all_areas':bool(np.array_equal(W[np.setdiff1d(np.arange(N),s)],knn_weights(coords,k=6,strata=strata)[np.setdiff1d(np.arange(N),s)])),
        'all_rows_unit_sum_and_zero_diagonal':bool(np.allclose(W.sum(axis=1),1,atol=1e-14,rtol=0) and np.all(np.diag(W)==0)),
        'sampled_covariance_matches_D_only':errors['sampled_covariance_relative']<1e-12,
        'interior_rho_covariance_absolute_error_below_1e_minus12':errors['fixed_interior_covariance']<1e-12,
        'fixed_A_rho_sampled_predictor_identical_below_1e_minus10':errors['fixed_A_rho_sampled_predictor']<1e-10,
        'fixed_A_rho_nonsampled_formula_identical_below_1e_minus10':errors['fixed_A_rho_nonsampled_formula']<1e-10,
        'sampled_point_and_plugin_match_D_only':max(errors['sampled_points'],errors['sampled_plugin_MSE'])<1e-6,
        'sampled_REML_matches_D_only':errors['REML_nll']<1e-8,
        'non_sampled_prediction_matches_2024_formula':errors['nonsampled_formula']<1e-8,
        'full_prediction_matches_independent_dense_algebra':errors['dense_all_area_point']<1e-8,
        'bootstrap_point_is_unchanged':errors['bootstrap_keeps_point']<1e-12,
        'bootstrap_naive_and_BC_finite_same_draws':bool(np.isfinite(boot['mse_naive']).all() and np.isfinite(boot['mse_bc']).all()),
        'all_bootstrap_strata_have_at_least95percent_valid':bool(all(sf['bootstrap_success']>=.95*12 for sf in boot['fit'])),
    }
    result={'status':'PASS_PUBLISHED_SAMPLED_GRAPH_AND_DIRECTED_EXTENSION_QA' if all(checks.values()) else 'FAIL','checks':checks,'max_absolute_errors':errors,'condition_diagnostics':conditions,'near_rho_boundary_numerical_policy':'Relative covariance tolerance1e-12 and absolute interior-rho check1e-12; independent sparse fits may differ by1e-6 in probabilities because of objective rounding. This is not a claim of bitwise identical optimization.','synthetic_domains':N,'synthetic_sampled_domains':len(s),'bootstrap_B_for_numerical_QA_only':12,'bootstrap_success':[sf['bootstrap_success'] for sf in boot['fit']],'accuracy_or_heldout_truth_read':False,'model_selection_performed':False,'elapsed_seconds':time.monotonic()-before,'sources_sha256':{str(ROOT/'scripts'/p):sha(ROOT/'scripts'/p) for p in ('sikov_published_reference.py','sikov_model.py','sikov_bootstrap.py','sikov_published_validate.py')}}
    (ROOT/'validacion/sikov_published_reference_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if result['status']=='FAIL':raise RuntimeError('Additional reference QA failed')

if __name__=='__main__':main()
