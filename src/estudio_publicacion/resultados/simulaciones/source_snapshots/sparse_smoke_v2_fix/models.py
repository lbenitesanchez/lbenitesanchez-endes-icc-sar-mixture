"""Reproducible Gaussian empirical-Bayes area models.

REML estimates an outcome-specific intercept/regression and an iid district
covariance. Spatial models add a scaled, fixed low-rank geographic field.
The spatial field is an approximation, not a full CAR or a BYM2 model.
All matrices use area-major, outcome-minor ordering. Partial observations are
accepted. Returned intervals integrate GLS coefficient uncertainty but, unless
bootstrap_intervals is called, condition on fitted variance parameters.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.special import digamma, polygamma, gammaln
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

METHODS = ("FH", "MFH", "SFH", "MSFH")


def _as_neff(n_eff, n):
    a = np.asarray(n_eff, float)
    if a.ndim == 1:
        a = np.repeat(a[:, None], 2, axis=1)
    if a.shape != (n, 2):
        raise ValueError(f"n_eff shape {a.shape}, expected {(n, 2)}")
    return np.where(np.isfinite(a) & (a > 0), np.maximum(a, 1.), 1.)


def spatial_basis(coords, rank=40, neighbors=8):
    """Positive-eigenvalue graph spectral basis from symmetric KNN graph.

    Coordinates are latitude/longitude, converted to approximate kilometres.
    Columns are eigenvectors weighted by positive eigenvalues. Scaling makes
    the mean diagonal of BB' exactly one. A fixed basis is allowed to use the
    complete census geography: no survey responses enter this function.
    """
    coords = np.asarray(coords, float)
    if not np.all(np.isfinite(coords)):
        raise ValueError("Spatial basis needs finite coordinates")
    m = len(coords)
    if m < 5:
        raise ValueError("At least five areas are required")
    xy = np.column_stack((111.32 * coords[:, 1] * np.cos(np.deg2rad(np.mean(coords[:, 0]))), 111.32 * coords[:, 0]))
    d = cdist(xy, xy)
    np.fill_diagonal(d, np.inf)
    k = min(neighbors, m - 1)
    ii = np.arange(m).repeat(k)
    jj = np.argpartition(d, k - 1, axis=1)[:, :k].ravel()
    w = coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(m, m)).tocsr()
    w = w.maximum(w.T)
    deg = np.asarray(w.sum(axis=1)).ravel()
    ws = w.multiply((1 / np.sqrt(deg))[:, None]).multiply((1 / np.sqrt(deg))[None, :]).tocsr()
    # The first eigenvector of normalized adjacency is omitted. The remaining
    # vectors capture broad connected-neighborhood structure, not iid noise.
    q = min(rank + 1, m - 2)
    val, vec = eigsh(ws, k=q, which="LA", v0=np.linspace(.1, 1., m), tol=1e-8)
    order = np.argsort(val)[::-1]
    val, vec = val[order][1:], vec[:, order][:, 1:]
    use = val > 1e-8
    b = vec[:, use] * np.sqrt(val[use])
    b -= b.mean(axis=0)
    b /= np.sqrt(np.mean(np.sum(b*b, axis=1)))
    return b


def prepare_observations(y, D, n_eff, Xall=None, sample_idx=None,
                         transform="logit", D_mode="smooth", df=None, pooling=None, prior_df=20.):
    """Transform and smooth sampling uncertainty using training observations.

    Jeffreys/Kish adjustment (ny+.5)/(n+1) handles observed 0/1 proportions.
    A pooled log design-effect GVF provides a reference variance. Per-area log
    variances shrink toward it with weight df/(df+20). Correlations shrink on
    the Fisher-z scale toward the pooled correlation; this preserves PSD.
    Raw mode retains positive marginal variances, floors zero variances at 5%
    of the GVF and clips correlations to .95. It is named regularized-raw in
    reports and is not claimed to recover unavailable design information.
    df should be survey residual degrees of freedom; otherwise n_eff-1 is a
    sensitivity proxy, clearly identified in metadata.
    """
    y = np.asarray(y, float)
    D = np.asarray(D, float)
    n = len(y)
    ne = _as_neff(n_eff, n)
    if y.shape != (n, 2) or D.shape != (n, 2, 2):
        raise ValueError("Expected y(n,2), D(n,2,2)")
    if transform not in ("logit", "identity") or D_mode not in ("smooth", "smooth_biascorrected", "raw"):
        raise ValueError("Unsupported transform or sampling variance option")
    obs = np.isfinite(y)
    if transform == "logit":
        if np.any((y[obs] < 0) | (y[obs] > 1)):
            raise ValueError("Proportions must lie in [0,1]")
        padj = (y * ne + .5)/(ne + 1.)
        z = logit(padj)
        base = padj*(1-padj)/ne
        # Reciprocal derivative of logit(p_adj) with respect to the original
        # direct proportion, including d p_adj/d y = n_eff/(n_eff+1).
        derivative = padj*(1-padj)*(ne+1.)/ne
    else:
        padj = y.copy()
        z = y.copy()
        base = 1/ne
        derivative = np.ones_like(y)
    diag = np.stack((D[:,0,0], D[:,1,1]), axis=1)
    zero_tolerance=np.maximum(1e-15,np.nan_to_num(base,nan=1.)*1e-12)
    if df is None:
        dfa = np.maximum(ne - 1, 0)
        df_source = "Kish n_eff minus one proxy"
    else:
        dfa = np.asarray(df, float)
        if dfa.ndim == 1:
            dfa = np.repeat(dfa[:, None], 2, axis=1)
        dfa = np.maximum(np.nan_to_num(dfa, nan=0.), 0.)
        df_source = "supplied survey residual degrees of freedom"
    pooled = []
    logdeff = []
    for j in range(2):
        good = obs[:, j] & np.isfinite(diag[:,j]) & (diag[:,j]>zero_tolerance[:,j]) & np.isfinite(base[:,j])
        ratio = diag[good,j]/base[good,j]
        logs = np.log(np.maximum(ratio, 1e-12))
        # Robust pooling limits domination by highly unstable small-domain D.
        if D_mode == "smooth_biascorrected" and len(logs):
            shape = np.maximum(dfa[good,j],1.)/2
            bias = digamma(shape)-np.log(shape)
            weight = 1/polygamma(1,shape)
            pooledlog = np.sum(weight*(logs-bias))/np.sum(weight)
        else:
            pooledlog = np.median(logs) if len(logs) else 0.
        pooled.append(float(np.exp(pooledlog)))
        logdeff.append(logs)
    if pooling is not None:
        pooled = list(pooling["pooled_design_effect"])
    reference = base * np.array(pooled)
    for j in range(2):
        reference[~np.isfinite(reference[:,j]), j] = np.nanmedian(reference[:,j]) if np.any(np.isfinite(reference[:,j])) else 1.
    positive = np.isfinite(diag) & (diag > zero_tolerance)
    lam = dfa / (dfa + float(prior_df))
    lam = np.where(positive, lam, 0.)
    if D_mode in ("smooth", "smooth_biascorrected"):
        logvariance = lam*np.log(np.where(positive,diag,reference))+(1-lam)*np.log(np.maximum(reference,1e-12))
        if D_mode == "smooth_biascorrected":
            shape=np.maximum(dfa,1.)/2
            # Exact fractional chi-square moment, not a lognormal delta rule:
            # E[(Dhat/Dtrue)^lambda] = shape^-lambda Gamma(shape+lambda)/Gamma(shape).
            # When the GVF reference equals Dtrue this corrects the geometric
            # shrinkage mean, including tiny df and lambda=0 or lambda=1.
            logmoment=-lam*np.log(shape)+gammaln(shape+lam)-gammaln(shape)
            logvariance -= logmoment
        variance = np.exp(logvariance)
    else:
        variance = np.where(positive,diag,.05*reference)
    denominator = np.sqrt(np.maximum(diag[:,0]*diag[:,1],1e-30))
    rho_raw = D[:,0,1]/denominator
    validrho = np.all(positive & obs,axis=1) & np.isfinite(rho_raw) & (np.abs(rho_raw) <= 1.000001)
    rho_pool = float(np.tanh(np.median(np.arctanh(np.clip(rho_raw[validrho],-.95,.95))))) if np.any(validrho) else 0.
    if pooling is not None:
        rho_pool = float(pooling["pooled_sampling_correlation"])
    rweight = np.minimum(lam[:,0],lam[:,1])
    safe_rho = np.where(validrho,np.clip(rho_raw,-.95,.95),rho_pool)
    if D_mode in ("smooth", "smooth_biascorrected"):
        rho = np.tanh(rweight*np.arctanh(safe_rho)+(1-rweight)*np.arctanh(rho_pool))
    else:
        rho = safe_rho
    result = np.zeros_like(D)
    result[:,0,0], result[:,1,1] = variance[:,0],variance[:,1]
    result[:,0,1] = result[:,1,0] = rho*np.sqrt(variance[:,0]*variance[:,1])
    result /= derivative[:,:,None]*derivative[:,None,:]
    # Missing endpoints never enter likelihood. Their placeholders are finite
    # solely to permit fixed-shape block computations.
    for j in range(2):
        result[~obs[:,j],j,j] = 1.
    result[~np.all(obs,axis=1),0,1] = 0.
    result[~np.all(obs,axis=1),1,0] = 0.
    return {"z":z,"D":result,"observed":obs,"p_adjusted":padj,
            "metadata":{"transform":transform,"D_mode":D_mode,
                "df_source":df_source,"pooled_design_effect":pooled,
                "pooled_sampling_correlation":rho_pool,
                "zero_or_missing_variances":int(np.sum(obs & ~positive)),
                "observations":int(obs.sum()),"variance_prior_df":float(prior_df)}}


def _outcome_design(x):
    n,p = x.shape
    xx = np.zeros((n,2,2*p))
    xx[:,0,:p],xx[:,1,p:] = x,x
    return xx


def _covariance(params, full):
    sd = np.exp(params[:2])
    rho = np.tanh(params[2]) if full else 0.
    a = np.diag(sd**2)
    a[0,1] = a[1,0] = rho*sd[0]*sd[1]
    return a


@dataclass
class AreaFit:
    Xall: np.ndarray
    z: np.ndarray
    D: np.ndarray
    sample_idx: np.ndarray
    basis: np.ndarray | None
    method: str
    transform: str
    params: np.ndarray
    A: np.ndarray
    S: np.ndarray | None
    beta: np.ndarray
    beta_cov: np.ndarray
    Tinv: np.ndarray
    coef_mean: np.ndarray | None
    coef_cov: np.ndarray | None
    coef_x: np.ndarray | None
    status: dict
    preparation: dict

    def predict(self, level=.95):
        from scipy.stats import norm
        m,p = self.Xall.shape
        xa = _outcome_design(self.Xall)
        mean = np.einsum("mip,p->mi",xa,self.beta)
        proj = xa.copy()
        vc = np.zeros((m,2,2))
        if self.S is not None:
            ua = np.kron(self.basis,np.eye(2)).reshape(m,2,-1)
            mean += np.einsum("miq,q->mi",ua,self.coef_mean)
            proj -= np.einsum("miq,qp->mip",ua,self.coef_x)
            vc += np.einsum("miq,qr,mjr->mij",ua,self.coef_cov,ua,optimize=True)
        var = vc + self.A
        idx = self.sample_idx
        gain = np.einsum("ij,njk->nik",self.A,self.Tinv)
        r = np.nan_to_num(self.z - mean[idx],nan=0.)
        mean[idx] += np.einsum("nij,nj->ni",gain,r)
        h = np.eye(2)[None,:,:] - gain
        var[idx] = self.A - np.einsum("nij,jk->nik",gain,self.A) + np.einsum("nij,njk,nlk->nil",h,vc[idx],h,optimize=True)
        proj[idx] = np.einsum("nij,njp->nip",h,proj[idx])
        var += np.einsum("mip,pq,mjq->mij",proj,self.beta_cov,proj,optimize=True)
        var = (var+var.transpose(0,2,1))/2
        sd = np.sqrt(np.maximum(np.stack((var[:,0,0],var[:,1,1]),axis=1),0.))
        c = norm.ppf((1+level)/2)
        lo,hi = mean-c*sd,mean+c*sd
        if self.transform == "logit":
            # 20-point Gauss-Hermite integration gives the conditional posterior
            # mean of p; inverse logit of mean is the conditional median.
            nodes,weights = np.polynomial.hermite.hermgauss(20)
            pm = np.sum(expit(mean[:,:,None]+np.sqrt(2)*sd[:,:,None]*nodes)*weights,axis=2)/np.sqrt(np.pi)
            pl,pu = expit(lo),expit(hi)
        else:
            pm,pl,pu = mean,lo,hi
        return {"eta_mean":mean,"eta_cov":var,"p_mean":pm,"p_lo":pl,"p_hi":pu,
                "eta_lo":lo,"eta_hi":hi,"level":level,
                "interval_type":"universal-kriging Gaussian, plug-in covariance parameters"}

    def sample_latent(self, rng):
        m = len(self.Xall)
        eta = _outcome_design(self.Xall) @ self.beta
        eta += rng.multivariate_normal(np.zeros(2),self.A,size=m)
        if self.S is not None:
            c = rng.multivariate_normal(np.zeros(2),self.S,size=self.basis.shape[1])
            eta += self.basis @ c
        return eta


def fit_model(Xall,y,D,n_eff,sample_idx=None,basis=None,method="FH",
              transform="logit",D_mode="smooth",df=None,maxiter=120,
              start=None,multistart=False,prior_df=20.):
    Xall = np.asarray(Xall,float)
    y,D = np.asarray(y,float),np.asarray(D,float)
    idx = np.arange(len(y)) if sample_idx is None else np.asarray(sample_idx,int)
    if len(idx)!=len(y) or len(np.unique(idx)) != len(idx):
        raise ValueError("sample_idx must contain one unique index per y row")
    if method not in METHODS or not np.all(np.isfinite(Xall)):
        raise ValueError("Unknown method or nonfinite design")
    if method in ("SFH","MSFH"):
        if basis is None or len(basis)!=len(Xall):
            raise ValueError("Spatial models require a known basis for all areas")
        basis = np.asarray(basis,float)
    else:
        basis = None
    prep = prepare_observations(y,D,n_eff,transform=transform,D_mode=D_mode,df=df,prior_df=prior_df)
    z,dd,obs = prep["z"],prep["D"],prep["observed"]
    if method in ("FH","SFH"):
        dd = dd.copy()
        dd[:,0,1] = dd[:,1,0] = 0.
    n,p = len(y),Xall.shape[1]
    if obs.sum()<=2*p+4:
        raise ValueError("Too few observations for REML")
    xx = _outcome_design(Xall[idx])
    xx *= obs[:,:,None]
    xmat = xx.reshape(2*n,2*p)
    zv = np.nan_to_num(z,nan=0.).ravel()
    u = np.kron(basis[idx],np.eye(2)) if basis is not None else None
    if u is not None:
        u *= obs.ravel()[:,None]
    full = method in ("MFH","MSFH")
    na = 3 if full else 2
    q = basis.shape[1] if basis is not None else 0

    def evaluate(t,retain=False):
        a = _covariance(t[:na],full)
        s = _covariance(t[na:],full) if q else None
        tv = dd + a
        inv = np.zeros_like(tv)
        both = np.all(obs,axis=1)
        det = tv[:,0,0]*tv[:,1,1]-tv[:,0,1]**2
        if np.any(det[both]<=0):
            return 1e30
        inv[both,0,0] = tv[both,1,1]/det[both]
        inv[both,1,1] = tv[both,0,0]/det[both]
        inv[both,0,1] = inv[both,1,0] = -tv[both,0,1]/det[both]
        only0,only1 = obs[:,0]&~obs[:,1],obs[:,1]&~obs[:,0]
        inv[only0,0,0] = 1/tv[only0,0,0]
        inv[only1,1,1] = 1/tv[only1,1,1]
        ld = np.log(det[both]).sum()+np.log(tv[only0,0,0]).sum()+np.log(tv[only1,1,1]).sum()
        def tinv(v):
            return np.einsum("nij,njk->nik",inv,v.reshape(n,2,-1)).reshape(2*n,-1)
        tu = tinv(u) if q else None
        cf,cc = None,None
        if q:
            sinv = np.linalg.inv(s)
            precision = np.kron(np.eye(q),sinv)+u.T@tu
            cf = cho_factor(precision,lower=True,check_finite=False)
            ld += q*np.linalg.slogdet(s)[1]+2*np.log(np.diag(cf[0])).sum()
        def vinv(v):
            tvv = tinv(v)
            return tvv-tu@cho_solve(cf,u.T@tvv,check_finite=False) if q else tvv
        vx = vinv(xmat)
        vz = vinv(zv[:,None])[:,0]
        h = xmat.T@vx
        hc = cho_factor(h,lower=True,check_finite=False)
        bc = cho_solve(hc,np.eye(2*p),check_finite=False)
        beta = bc@(xmat.T@vz)
        residual = zv-xmat@beta
        quad = residual@vinv(residual[:,None])[:,0]
        objective = .5*(ld+2*np.log(np.diag(hc[0])).sum()+quad+(obs.sum()-2*p)*np.log(2*np.pi))
        if not retain:
            return float(objective)
        if q:
            cc = cho_solve(cf,np.eye(2*q),check_finite=False)
            cm = cc@(u.T@tinv(residual[:,None])[:,0])
            cx = cc@(u.T@tinv(xmat))
        else:
            cm,cx = None,None
        return a,s,beta,bc,inv,cm,cc,cx,float(objective)

    # Broad scale bounds prevent overflow. Hitting a bound is recorded, never
    # silently interpreted as evidence of zero random-effect variation.
    marginal = np.nanvar(z,axis=0)
    init = np.log(np.sqrt(np.maximum(marginal*.3,.03)))
    t0 = np.r_[init,0.] if full else init
    if q:
        t0 = np.r_[t0,(np.r_[init-.4,0.] if full else init-.4)]
    if start is not None:
        t0 = np.asarray(start,float)
    bounds = [(-6.,2.),(-6.,2.)]+([(-2.,2.)] if full else [])
    if q:
        bounds += [(-6.,2.),(-6.,2.)]+([(-2.,2.)] if full else [])
    candidates = [t0]
    if multistart:
        alt = t0.copy()
        if q:
            alt[na:na+2] -= 1.
        candidates.append(alt)
    runs=[]
    for initial in candidates:
        res = minimize(evaluate,initial,method="L-BFGS-B",bounds=bounds,
                       options={"maxiter":maxiter,"ftol":1e-8,"gtol":2e-4,"maxls":30})
        runs.append(res)
    good=[r for r in runs if np.isfinite(r.fun)]
    if not good:
        raise RuntimeError("REML produced no finite result")
    opt = min(good,key=lambda r:r.fun)
    a,s,beta,bc,inv,cm,cc,cx,nll = evaluate(opt.x,True)
    boundary=[int(i) for i,(v,(low,high)) in enumerate(zip(opt.x,bounds)) if min(v-low,high-v)<.01]
    status={"success":bool(opt.success),"message":str(opt.message),"iterations":int(opt.nit),
            "nfev":int(opt.nfev),"reml_nll":nll,"boundary_parameters":boundary,
            "method":method,"spatial_rank":q,"n_observed":int(obs.sum()),
            "iid_covariance":a.tolist(),"spatial_covariance":s.tolist() if s is not None else None,
            "optimizer":"L-BFGS-B","likelihood":"Gaussian REML","variance_parameter_count":len(opt.x)}
    status["sampling_covariance_used"] = method in ("MFH","MSFH")
    return AreaFit(Xall,z,dd,idx,basis,method,transform,opt.x,a,s,beta,bc,inv,cm,cc,cx,status,prep["metadata"])


def bootstrap_intervals(fit, n_eff, B=100, seed=20261005, df=None,
                        resample_D=False, maxiter=100, level=.95):
    """Parametric predictive-error bootstrap with variance-parameter refits.

    Conditional D is the fitted smoothed matrix. Optional Wishart D draws are
    a sensitivity model, not survey replication. An interval is centered on
    the fitted EB logit prediction and uses quantiles of bootstrap prediction
    errors; its finite-B tail Monte Carlo uncertainty must be reported.
    """
    rng=np.random.default_rng(seed)
    idx=fit.sample_idx
    ne=_as_neff(n_eff,len(idx))
    errors=[]
    statuses=[]
    for b in range(B):
        true=fit.sample_latent(rng)
        noise=np.array([rng.multivariate_normal(np.zeros(2),d) for d in fit.D])
        zz=true[idx]+noise
        zz[~np.isfinite(fit.z)] = np.nan
        ddraw=fit.D.copy()
        if resample_D:
            dfs=np.maximum(np.min(ne,axis=1)-1,1.) if df is None else np.maximum(np.asarray(df,float),1.)
            if dfs.ndim>1:
                dfs=dfs.min(axis=1)
            ddraw=np.array([((lambda e: e.T@e/int(v))(rng.multivariate_normal(np.zeros(2),d,size=int(v)))) for d,v in zip(fit.D,dfs)])
        # Identity keeps the already transformed sampling matrices and makes
        # their generated scale explicit. Smoothing can be requested only in
        # the separate simulation experiment; do not smooth twice here.
        f=fit_model(fit.Xall,zz,ddraw,ne,idx,fit.basis,fit.method,transform="identity",
                    D_mode="raw",df=df,maxiter=maxiter,start=fit.params)
        statuses.append(f.status)
        if f.status["success"]:
            errors.append(f.predict()["eta_mean"]-true)
    if len(errors)<max(20,.8*B):
        raise RuntimeError(f"Only {len(errors)}/{B} converged bootstrap fits")
    err=np.stack(errors)
    alpha=(1-level)/2
    pred=fit.predict(level)
    lo=pred["eta_mean"]-np.quantile(err,1-alpha,axis=0)
    hi=pred["eta_mean"]-np.quantile(err,alpha,axis=0)
    return {"eta_lo":lo,"eta_hi":hi,"p_lo":expit(lo) if fit.transform=="logit" else lo,
            "p_hi":expit(hi) if fit.transform=="logit" else hi,
            "mse_eta":np.mean(err**2,axis=0),"bias_eta":np.mean(err,axis=0),
            "B_requested":B,"B_success":len(errors),"statuses":statuses,
            "D_resampling":"Wishart sensitivity" if resample_D else "conditional fitted D",
            "interval_type":"parametric predictive-error bootstrap with REML refits",
            "tail_probability_MCSE":float(np.sqrt(alpha*(1-alpha)/len(errors))),"level":level}
