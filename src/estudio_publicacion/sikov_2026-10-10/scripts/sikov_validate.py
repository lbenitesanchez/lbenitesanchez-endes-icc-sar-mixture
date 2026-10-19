"""Independent numerical checks versus CRAN sae and heldout invariance."""
import csv,json,os,sys,subprocess,time
from pathlib import Path
import numpy as np
from sikov_model import fit_sikov,knn_weights,_SAR

ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/'validacion'/'sikov_sae_reference'
DIR.mkdir(parents=True,exist_ok=True)
INPUT=DIR/'inputs';OUTPUT=DIR/'outputs'
INPUT.mkdir(exist_ok=True);OUTPUT.mkdir(exist_ok=True)

def read_csv(path):
    with open(path) as f:return list(csv.DictReader(f))

def run():
    start=time.monotonic();rng=np.random.default_rng(12677);n=100
    coords=rng.normal(size=(n,2));x=rng.uniform(0,1,n);X=np.c_[np.ones(n),x]
    D=rng.uniform(.0005,.0015,n);checks=[];native={}
    for k in (4,6,8):
        W=knn_weights(coords,k)
        u=np.linalg.solve(np.eye(n)-.65*W,rng.normal(0,np.sqrt(.004),n))
        y=.42+.08*x+u+rng.normal(0,np.sqrt(D))
        np.savetxt(INPUT/f'knn{k}_data.csv',np.c_[y,x,D],delimiter=',',header='y,x,D',comments='')
        np.savetxt(INPUT/f'knn{k}_W.csv',W,delimiter=',')
        native[k]=fit_sikov(X,y,D,np.arange(n),W,uncertainty='delta')
        native[(k,'FH')]=fit_sikov(X,y,D,np.arange(n),W,spatial=False)
    env=os.environ.copy();env['SIKOV_SAE_LIBRARY']=str(ROOT/'fuentes'/'sae_library')
    subprocess.run(['Rscript',str(ROOT/'scripts'/'sikov_sae_reference.R'),str(INPUT),str(OUTPUT)],env=env,check=True)
    for k in (4,6,8):
        pred=read_csv(OUTPUT/f'knn{k}_pred.csv');par=read_csv(OUTPUT/f'knn{k}_param.csv')[0]
        f=native[k];ff=native[(k,'FH')]
        ref=np.array([float(r['mean']) for r in pred]); fhref=np.array([float(r['fhmean']) for r in pred])
        err=float(abs(ref-f['mean']).max());aerr=abs(float(par['A'])-f['fit'][0]['variance']);rerr=abs(float(par['rho'])-f['fit'][0]['rho']);fherr=float(abs(fhref-ff['mean']).max())
        checks.append(dict(check='CRAN_sae_predictor_and_REML',k=k,max_abs_predictor=err,A_abs_error=aerr,rho_abs_error=rerr,FH_max_abs_predictor=fherr,pass_check=err<2e-5 and aerr<2e-6 and rerr<2e-4 and fherr<2e-5))
    # Every training calculation ignores unobserved y and D; keep full latent W.
    s=np.arange(0,n,2);W=knn_weights(coords,6);y=np.full(n,.4);D=np.full(n,.001)
    y[s]=.4+.1*x[s]+rng.normal(0,.05,len(s))
    f1=fit_sikov(X,y,D,s,W)
    y[np.setdiff1d(np.arange(n),s)]=np.nan;D[np.setdiff1d(np.arange(n),s)]=np.nan
    f2=fit_sikov(X,y,D,s,W)
    checks.append(dict(check='no_heldout_response_or_variance_access',max_error=float(abs(f1['mean']-f2['mean']).max()),pass_check=np.array_equal(f1['mean'],f2['mean'])))
    # SAR covariance must equal the defining equation, including directed W.
    sar=_SAR(W,s,True);C,dC,Call,diagC,dCall=sar.covariance(.53,True)
    B=np.eye(n)-.53*W;B_inv=np.linalg.inv(B);ref=B_inv@B_inv.T
    checks.append(dict(check='full_latent_SAR_covariance',max_error=float(abs(Call-ref[:,s]).max()),pass_check=np.allclose(Call,ref[:,s],atol=1e-11) and np.allclose(diagC,np.diag(ref),atol=1e-11)))
    # Finite-difference covariance derivative prevents asymmetric-W order error.
    eps=1e-6;Cp=sar.covariance(.53+eps)[0];Cm=sar.covariance(.53-eps)[0]
    checks.append(dict(check='directed_covariance_derivative',max_error=float(abs((Cp-Cm)/(2*eps)-dC).max()),pass_check=np.allclose((Cp-Cm)/(2*eps),dC,atol=1e-8)))
    # Conditional uncertainty agrees with direct universal-kriging equations.
    A=f1['fit'][0]['variance'];rho=f1['fit'][0]['rho'];B=np.eye(n)-rho*W;C=np.linalg.solve(B,np.eye(n));G=A*C@C.T
    V=G[np.ix_(s,s)]+np.diag(D[s]);Vi=np.linalg.inv(V);Q=np.linalg.inv(X[s].T@Vi@X[s]);K=G[:,s]@Vi;H=X-K@X[s]
    expected=np.diag(G-G[:,s]@Vi@G[s,:])+np.sum((H@Q)*H,axis=1)
    checks.append(dict(check='universal_kriging_plugin_mse',max_error=float(abs(expected-f1['mse_plugin']).max()),pass_check=np.allclose(expected,f1['mse_plugin'],atol=1e-11)))
    result=dict(pass_check=all(c['pass_check'] for c in checks),checks=checks,seconds=time.monotonic()-start,reference='CRAN sae 1.3, eblupSFH/eblupFH REML precision 1e-9',note='Native delta intervals extend uncertainty; not article-defined or identical to sae analytic bias correction.')
    (ROOT/'validacion'/'sikov_model_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return 0 if result['pass_check'] else 1
if __name__=='__main__':sys.exit(run())
