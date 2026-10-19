"""Parametric bootstrap MSE of the fixed Sikov FH/SAR reconstruction.

Naive and Molina et al. (2009) bias-corrected estimators use SAME replicates.
BC = 2*plugin(original) - mean(plugin(refit))
     + mean((empirical predictor - fixed-parameter BLUP)**2).
This is CRAN sae::pbmseSFH's formula, extended to latent unsampled domains.
No truth or validation outcome argument is accepted. Native point model frozen.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor,cho_solve
from scipy.sparse.linalg import splu
from sikov_model import fit_sikov,_fit_stratum,_SAR,select_columns,RHO_BOUND


def _certificate(X,y,D,sample_idx,W,spatial,A,rho,tolerance=1e-5):
    """Check restricted-likelihood KKT at returned point, without refitting.

    Addresses occasional L-BFGS line-search flag at a valid boundary optimum.
    The raw optimizer flag remains separately auditable. No response truth used.
    """
    s=np.asarray(sample_idx,int);Xs=X[s];model=_SAR(W,s,spatial)
    C,dC=model.covariance(rho);V=A*C+np.diag(D)
    fac=cho_factor(V,lower=True,check_finite=False)
    Vi=cho_solve(fac,np.eye(len(y)),check_finite=False)
    ViX=Vi@Xs;Q=np.linalg.inv(Xs.T@ViX)
    beta=Q@Xs.T@Vi@y;vir=Vi@(y-Xs@beta)
    P=Vi-ViX@Q@ViX.T
    scale=max(float(np.var(y)),float(np.mean(D)),1e-5)
    grad=np.array([.5*(np.sum(P*(scale*C).T)-vir@(scale*C)@vir)])
    if spatial:grad=np.r_[grad,.5*(np.sum(P*(A*dC).T)-vir@(A*dC)@vir)]
    projected=grad.copy()
    if A/scale<=1e-8 and grad[0]>0:projected[0]=0
    if A/scale>=100-1e-8 and grad[0]<0:projected[0]=0
    if spatial:
        if rho<=-RHO_BOUND+1e-8 and grad[1]>0:projected[1]=0
        if rho>=RHO_BOUND-1e-8 and grad[1]<0:projected[1]=0
    norm=float(abs(projected).max())
    return {'projected_gradient_max':norm,'passed':norm<tolerance,'tolerance':tolerance,'rho_bound':RHO_BOUND,'at_rho_boundary':bool(spatial and abs(rho)>=RHO_BOUND-1e-8),'gradient_scaled':grad.tolist()}


def fit_sikov_bootstrap(X,y,D,sample_idx,W,strata=None,*,spatial=True,
                       outcome=None,covariate_names=None,B=100,seed=20261010,
                       interval_estimator='naive',innovations=None,
                       bootstrap_min_success=.95):
    """Same signature/data convention as fit_sikov; B>=100 primary by default.

    Return mean/mse/lower/upper plus mse_naive, mse_bc (raw), mse_plugin,
    naive_mcse, bc_mcse and fit/converged. Negative BC estimates stay visible;
    only sqrt for intervals applies disclosed zero floor when BC requested.
    innovations: optional {level: {'u':B x Nlevel,'e':B x nsampled}} standard
    normal arrays. Enables independent cross-language identical-draw tests.
    """
    if B<2:raise ValueError('bootstrap needs >=2 draws; final plan requires B>=100')
    if interval_estimator not in ('naive','bc'):raise ValueError('unknown estimator')
    X=np.asarray(X,float); y=np.asarray(y,float); D=np.asarray(D,float)
    sample_idx=np.asarray(sample_idx,int);W=np.asarray(W,float);n=len(X)
    strata=np.ones(n,int) if strata is None else np.asarray(strata)
    if len(y)==n:y=y[sample_idx]
    if len(D)==n:D=D[sample_idx]
    initial=fit_sikov(X,y,D,sample_idx,W,strata,spatial=spatial,outcome=outcome,covariate_names=covariate_names,uncertainty='delta')
    naive=np.full(n,np.nan);bc=np.full(n,np.nan);nse=np.full(n,np.nan);bcse=np.full(n,np.nan)
    rng=np.random.default_rng(seed);records=[]
    for sf in initial['fit']:
        level=sf['stratum'];ids=np.flatnonzero(strata==level);obs=np.isin(sample_idx,ids)
        s=np.searchsorted(ids,sample_idx[obs]);yy=y[obs];DD=D[obs]
        cols=np.arange(X.shape[1]) if outcome is None else select_columns(outcome,level,covariate_names,spatial)
        XX=X[ids][:,cols];WW=W[np.ix_(ids,ids)].copy()
        sums=WW.sum(axis=1);WW=np.divide(WW,sums[:,None],out=np.zeros_like(WW),where=sums[:,None]>0)
        A=sf['variance'];rho=sf['rho'];beta=np.asarray(sf['beta'])
        model=_SAR(WW,s,spatial);C,dC,Call,diagC,dCall=model.covariance(rho,full=True)
        V=A*C+np.diag(DD);fac=cho_factor(V,lower=True,check_finite=False)
        ViX=cho_solve(fac,XX[s],check_finite=False);Q=np.linalg.inv(XX[s].T@ViX)
        beta_map=Q@ViX.T
        K=A*Call@cho_solve(fac,np.eye(len(s)),check_finite=False)
        oracle_map=(XX-K@XX[s])@beta_map+K
        factor=splu(sparse.eye(len(ids),format='csc')-rho*sparse.csc_matrix(WW)) if spatial else None
        cert=_certificate(XX,yy,DD,s,WW,spatial,A,rho)
        # Successful raw optimizer or independently verified stationary point.
        sf['raw_optimizer_converged']=sf['converged']
        sf['stationarity_certificate']=cert
        sf['converged']=bool(sf['converged'] or cert['passed'])
        if innovations is None:
            U=rng.normal(size=(B,len(ids)));E=rng.normal(size=(B,len(s)))
        else:
            ii=innovations[level];U=np.asarray(ii['u']);E=np.asarray(ii['e'])
            if U.shape!=(B,len(ids)) or E.shape!=(B,len(s)):raise ValueError('bad innovation arrays')
        sq=[];bc_terms=[];failed=0;recovered=0
        for b in range(B):
            innovation=np.sqrt(A)*U[b]
            latent=factor.solve(innovation) if spatial else innovation
            truth=XX@beta+latent
            boot_y=truth[s]+np.sqrt(DD)*E[b]
            try:
                boot=_fit_stratum(XX,boot_y,DD,s,WW,spatial)
                valid=boot['converged']
                if not valid:
                    audit=_certificate(XX,boot_y,DD,s,WW,spatial,boot['variance'],boot['rho'])
                    valid=audit['passed'];recovered+=int(valid)
                if valid and np.isfinite(boot['mean']).all():
                    sq.append((boot['mean']-truth)**2)
                    oracle=oracle_map@boot_y
                    bc_terms.append(-boot['mse_plugin']+(boot['mean']-oracle)**2)
                else:failed+=1
            except (ValueError,np.linalg.LinAlgError,RuntimeError):failed+=1
        successes=len(sq)
        if successes<bootstrap_min_success*B:
            raise RuntimeError(f'Only {successes}/{B} bootstrap fits in stratum {level}')
        sq=np.asarray(sq);bc_terms=np.asarray(bc_terms)
        naive[ids]=sq.mean(axis=0);bc[ids]=2*initial['mse_plugin'][ids]+bc_terms.mean(axis=0)
        nse[ids]=sq.std(axis=0,ddof=1)/np.sqrt(successes);bcse[ids]=bc_terms.std(axis=0,ddof=1)/np.sqrt(successes)
        sf.update(bootstrap_B=B,bootstrap_success=successes,bootstrap_failed=failed,bootstrap_stationarity_recovered=recovered)
        records.append(sf)
    selected=naive if interval_estimator=='naive' else bc
    se=np.sqrt(np.maximum(selected,0));z=1.959963984540054
    return {**initial,'mse':selected,'mse_naive':naive,'mse_bc':bc,'naive_mcse':nse,'bc_mcse':bcse,'lower':initial['mean']-z*se,'upper':initial['mean']+z*se,'fit':records,'converged':all(r['converged'] for r in records),'uncertainty':'parametric_bootstrap_'+interval_estimator,'bootstrap_B':B,'negative_bc_estimates':int((bc<0).sum()),'BC_formula_reference':'CRAN sae 1.3 pbmseSFH.R; Molina Salvati Pratesi (2009)','bootstrap_partial_fit_policy':'retain valid fits if >=95%; counts and independent KKT recovery explicit'}
