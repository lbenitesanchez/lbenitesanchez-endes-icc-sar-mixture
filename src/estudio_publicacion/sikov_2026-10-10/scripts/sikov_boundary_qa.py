"""Training-only, independent REML profile check of the widened SAR domain."""
import hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize_scalar
from sikov_model import _fit_stratum,select_columns,RHO_BOUND
from sikov_bootstrap import _certificate

ROOT=Path(__file__).resolve().parents[1]
start=time.monotonic();path=ROOT/'desarrollo/inputs/SAR_logit_psu_sparse/rep_00002/model_inputs/inputs.npz'
d=dict(np.load(path));ids=np.flatnonzero(d['strata']==3);mask=np.isin(d['sample_idx'],ids)
s=np.searchsorted(ids,d['sample_idx'][mask]);cols=select_columns('anemia',3,d['covariate_names'].tolist(),True)
X=d['X_raw'][ids][:,cols];Xs=X[s];y=d['y'][mask,0];D=d['D'][mask,0,0]
W=d['W'][np.ix_(ids,ids)].copy();W/=W.sum(axis=1)[:,None]
new=_fit_stratum(X,y,D,s,W,True)
spec=importlib.util.spec_from_file_location('sikov_old_bounds',ROOT/'fuentes/sikov_model_bounds098_snapshot.py')
oldmodule=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldmodule)
old=oldmodule._fit_stratum(X,y,D,s,W,True)
scale=max(float(np.var(y)),float(np.mean(D)),1e-5)

def dense_nll(a,rho):
    B=np.eye(len(W))-rho*W;inv=np.linalg.inv(B);C=inv@inv.T
    V=a*C[np.ix_(s,s)]+np.diag(D);cf=cho_factor(V,lower=True)
    ViX=cho_solve(cf,Xs);Viy=cho_solve(cf,y);bf=cho_factor(Xs.T@ViX,lower=True)
    beta=cho_solve(bf,Xs.T@Viy);res=y-Xs@beta
    return .5*(2*np.log(np.diag(cf[0])).sum()+2*np.log(np.diag(bf[0])).sum()+res@cho_solve(cf,res)+(len(y)-Xs.shape[1])*np.log(2*np.pi))

grid=[]
for rho in [-RHO_BOUND,-.98,-.8,-.6,0.,.3,.6,.9,RHO_BOUND]:
    op=minimize_scalar(lambda z:dense_nll(z*scale,rho),bounds=(0,100),method='bounded',options={'xatol':1e-10})
    grid.append({'rho':rho,'A':float(op.x*scale),'nll':float(op.fun)})
cert=_certificate(X,y,D,s,W,True,new['variance'],new['rho'])
checks=[{'name':'shared_interior_bound_matches_CRAN_safeguard','passed':RHO_BOUND==.999},
        {'name':'widened_bound_does_not_worsen_training_REML','passed':new['reml_nll']<=old['reml_nll']+1e-8},
        {'name':'negative_boundary_reaches_new_domain','passed':abs(new['rho']+RHO_BOUND)<1e-8},
        {'name':'dense_independent_REML_equals_native','passed':abs(dense_nll(new['variance'],new['rho'])-new['reml_nll'])<1e-9},
        {'name':'stationary_KKT_certificate','passed':cert['passed']},
        {'name':'no_better_coarse_profile_point','passed':new['reml_nll']<=min(g['nll'] for g in grid)+1e-8},
        {'name':'both_bound_edges_are_nonsingular','passed':all(np.linalg.svd(np.eye(len(W))-rho*W,compute_uv=False).min()>1e-8 for rho in (-RHO_BOUND,RHO_BOUND))}]
result={'passed':all(c['passed'] for c in checks),'checks':checks,'native_new':{k:new[k]for k in ['variance','rho','reml_nll','converged','message']},'old_bounds098':{k:old[k]for k in ['variance','rho','reml_nll']},'independent_profile':grid,'certificate':cert,'truth_read':False,'development_only_training_inputs':str(path),'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'elapsed_seconds':time.monotonic()-start}
serialized=json.dumps(result,indent=2,default=lambda value:value.item() if isinstance(value,np.generic) else value)
(ROOT/'validacion/sikov_boundary_validation.json').write_text(serialized+'\n')
print(serialized)
if not result['passed']:raise RuntimeError('boundary verification failed')
