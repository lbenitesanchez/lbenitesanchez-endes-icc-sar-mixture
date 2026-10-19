"""Independent dense universal-kriging check of the scalable area estimator.

Tests numerical moments, including partial measurements and unsampled areas.
Does not assert statistical calibration: that is evaluated by Monte Carlo.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json
from pathlib import Path
import numpy as np
from models import fit_model, METHODS

root=Path(__file__).resolve().parents[1]
rng=np.random.default_rng(64832)
m=18
X=np.column_stack([np.ones(m),rng.normal(size=m)])
B=rng.normal(size=(m,4));B-=B.mean(axis=0)
B/=np.sqrt(np.mean(np.sum(B*B,axis=1)))
idx=np.array([0,1,3,4,6,7,8,10,12,13,14,16,17])
true=X@np.array([[-.9,-1.5],[.2,.4]])+rng.normal(size=(m,2))*.5
n=np.repeat(rng.integers(8,45,size=len(idx))[:,None],2,axis=1)
D=np.array([[[.3,.04],[.04,.45]] for _ in idx])
y=true[idx]+np.array([rng.multivariate_normal([0,0],d) for d in D])
y[2,1]=np.nan;y[8,0]=np.nan
results=[]
for method in METHODS:
    f=fit_model(X,y,D,n,idx,B,method,transform='identity',D_mode='smooth',
                df=np.repeat(5,len(idx)),maxiter=180,multistart=True)
    got=f.predict()
    K=np.kron(np.eye(m),f.A)
    if f.S is not None:K+=np.kron(B@B.T,f.S)
    H=np.zeros((2*m,4));H[0::2,:2]=X;H[1::2,2:]=X
    all_idx=(idx[:,None]*2+np.arange(2)).ravel()
    take=np.isfinite(f.z.ravel());obs_idx=all_idx[take]
    block=np.zeros((2*len(idx),2*len(idx)))
    for i,d in enumerate(f.D):block[2*i:2*i+2,2*i:2*i+2]=d
    C=K[np.ix_(obs_idx,obs_idx)]+block[np.ix_(take,take)]
    Hobs=H[obs_idx];z=f.z.ravel()[take]
    Ci=np.linalg.inv(C)
    beta_cov=np.linalg.inv(Hobs.T@Ci@Hobs)
    beta=beta_cov@Hobs.T@Ci@z
    cross=K[:,obs_idx]
    expected=H@beta+cross@Ci@(z-Hobs@beta)
    correction=H-cross@Ci@Hobs
    covariance=K-cross@Ci@cross.T+correction@beta_cov@correction.T
    diag=np.array([covariance[2*i:2*i+2,2*i:2*i+2] for i in range(m)])
    em=float(np.max(np.abs(expected.reshape(m,2)-got['eta_mean'])))
    ec=float(np.max(np.abs(diag-got['eta_cov'])))
    eigen=float(np.min(np.linalg.eigvalsh(got['eta_cov'])))
    results.append(dict(method=method,mean_error=em,covariance_error=ec,
                        min_marginal_eigenvalue=eigen,optimizer=f.status,
                        partial_measurements=2,unsampled_areas=m-len(idx)))
    assert em<1e-8 and ec<1e-8,(method,em,ec)
    assert eigen>0,(method,eigen)
dest=root/'validacion';dest.mkdir(exist_ok=True)
(dest/'dense_equivalence.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
