"""Faithful reconstruction of Sikov et al. (2023), not a numerical replication.

Gaussian FH/SAR, raw proportions, REML, poverty strata, published covariates.
The article does not release W, exact k, direct estimates or interval algorithm.
Unobserved areas stay in the latent SAR graph. No held-out y enters fitting.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse.linalg import splu
from scipy.spatial import distance_matrix

# CRAN sae::eblupSFH safeguards Fisher updates at +/-0.999.  It subsequently
# rounds those boundary values to +/-1, which may be singular for row-normalized
# W. We retain the nonsingular interior safeguard rather than rounding to a pole.
RHO_BOUND = 0.999


TABLE_COVARIATES = {
    "anemia": {
        False: {1: ["intercept", "Altitud", "Internet", "Castellano"],
                2: ["intercept", "Altitud", "Cemento", "Agua", "Internet", "SIS", "Castellano", "Rural"],
                3: ["intercept", "Altitud", "Cemento", "Desague", "SIS", "Castellano"]},
        True: {1: ["intercept", "Altitud", "Internet", "Castellano"],
               2: ["intercept", "Altitud", "Cemento", "Agua", "Internet", "SIS", "Castellano", "Rural"],
               3: ["intercept", "Cemento", "SIS", "Castellano"]}},
    "stunting": {
        False: {1: ["intercept", "Altitud", "Refrig", "Internet", "Analfabet"],
                2: ["intercept", "Agua", "Elect", "Refrig", "NoSeguro", "Analfabet"],
                3: ["intercept", "Altitud", "Elect", "Refrig", "SIS", "Analfabet"]},
        True: {1: ["intercept", "Altitud", "Refrig", "Internet", "Analfabet"],
               2: ["intercept", "Agua", "Elect", "Refrig", "NoSeguro", "Analfabet"],
               3: ["intercept", "Altitud", "Elect", "Refrig", "SIS", "Analfabet"]}}}

# Tables 3.2--3.7, SFH column; used only to define a competitor-friendly
# synthetic DGP, never imposed as fitted coefficients on ENDES data.
PUBLISHED_SFH_BETA = {
    "anemia": {1:[.472,.017,-.221,-.176],
               2:[.752,.019,-.148,-.089,-.369,-.189,-.232,-.071],
               3:[.772,-.221,-.287,-.221]},
    "stunting": {1:[.176,.012,-.388,.269,1.498],
                 2:[.511,.118,-.146,-.202,-.260,.771],
                 3:[.647,.018,-.322,-.354,-.345,1.492]}}


def select_columns(outcome, stratum, covariate_names, spatial=True):
    names = TABLE_COVARIATES[outcome][bool(spatial)][int(stratum)]
    return np.array([list(covariate_names).index(name) for name in names],int)


def poverty_strata(poverty_proportion):
    """Documented boundary convention: [0,.30), [.30,.55], (.55,1]."""
    p = np.asarray(poverty_proportion, float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("poverty must be finite proportions")
    return np.where(p < .30, 1, np.where(p <= .55, 2, 3))


def knn_weights(coords, k=6, strata=None, haversine=False):
    """Directed kNN, deterministic stable ties, row normalized, within stratum.

    coords longitude/latitude degrees; Euclidean lon/lat is the local-script
    primary reconstruction. Haversine is a geometry-only sensitivity.
    """
    coords = np.asarray(coords, float)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("coords must be N x 2 finite longitude/latitude")
    if k < 1:
        raise ValueError("k must be positive")
    n = len(coords)
    strata = np.ones(n, int) if strata is None else np.asarray(strata)
    W = np.zeros((n, n))
    for level in np.unique(strata):
        ids = np.flatnonzero(strata == level)
        c = coords[ids]
        if len(ids) < 2:
            continue
        if haversine:
            lon, lat = np.deg2rad(c).T
            a = np.sin((lat[:, None] - lat[None, :]) / 2)**2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2)**2
            dist = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        else:
            dist = distance_matrix(c, c)
        np.fill_diagonal(dist, np.inf)
        kk = min(int(k), len(ids)-1)
        neighbors = np.argsort(dist, axis=1, kind="stable")[:, :kk]
        W[ids[:, None], ids[neighbors]] = 1 / kk
    return W


class _SAR:
    def __init__(self, W, sample_idx, spatial):
        self.W = sparse.csc_matrix(W)
        self.n = W.shape[0]
        self.s = np.asarray(sample_idx, int)
        self.spatial = spatial
        self.E = np.eye(self.n)[:, self.s]
        self.cache = None

    def covariance(self, rho, full=False):
        if not self.spatial:
            C = np.eye(len(self.s))
            if full:
                return C, np.zeros_like(C), self.E, np.ones(self.n), np.zeros_like(self.E)
            return C, np.zeros_like(C)
        if self.cache is not None and rho == self.cache[0]:
            fac, L, dL, C, dC = self.cache[1:]
        else:
            B = sparse.eye(self.n, format="csc") - rho*self.W
            fac = splu(B)
            L = fac.solve(self.E, trans="T")
            dL = fac.solve(self.W.T @ L, trans="T")
            C = L.T @ L
            dC = dL.T @ L + L.T @ dL
            self.cache = rho, fac, L, dL, C, dC
        if not full:
            return C, dC
        Call = fac.solve(L)
        dCall = fac.solve(self.W @ Call + dL)
        inverseB = fac.solve(np.eye(self.n))
        diagC = np.sum(inverseB*inverseB, axis=1)
        return C, dC, Call, diagC, dCall


def _fit_stratum(Xall, y, D, sample_idx, W, spatial=True, tolerance=1e-9):
    s = np.asarray(sample_idx, int)
    X = Xall[s]
    if len(y) <= X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("stratum lacks sampled degrees of freedom or full-rank X")
    if np.any(D <= 0) or not np.isfinite(D).all():
        raise ValueError("sampling variances must be strictly positive; no silent GVF")
    model = _SAR(W, s, spatial)
    scale = max(float(np.var(y)), float(np.mean(D)), 1e-5)

    def evaluate(A, rho, gradients=False):
        C, dC = model.covariance(rho)
        V = A*C + np.diag(D)
        fac = cho_factor(V, lower=True, check_finite=False)
        ViX = cho_solve(fac, X, check_finite=False)
        Viy = cho_solve(fac, y, check_finite=False)
        beta_precision = X.T @ ViX
        beta_factor = cho_factor(beta_precision, lower=True, check_finite=False)
        beta = cho_solve(beta_factor, X.T @ Viy, check_finite=False)
        resid = y-X@beta
        vir = cho_solve(fac, resid, check_finite=False)
        loss = .5*(2*np.log(np.diag(fac[0])).sum() + 2*np.log(np.diag(beta_factor[0])).sum() + resid@vir + (len(y)-X.shape[1])*np.log(2*np.pi))
        if not gradients:
            return loss, fac, beta, cho_solve(beta_factor, np.eye(X.shape[1]), check_finite=False), ViX
        Vi = cho_solve(fac, np.eye(len(y)), check_finite=False)
        Q = cho_solve(beta_factor, np.eye(X.shape[1]), check_finite=False)
        P = Vi - ViX@Q@ViX.T
        derivs = [scale*C, A*dC]
        grad = [.5*(np.sum(P*d.T)-vir@d@vir) for d in derivs]
        return loss, np.array(grad)

    if spatial:
        def obj(z):
            try:
                return evaluate(z[0]*scale, z[1], True)
            except (np.linalg.LinAlgError, RuntimeError, ValueError):
                return 1e30, np.zeros(2)
        starts = ((.5, .0), (.5, .5), (.5, -.5), (.05, .85))
        candidates = [minimize(obj, z, jac=True, method="L-BFGS-B", bounds=((0., 100.),(-RHO_BOUND,RHO_BOUND)), options={"ftol": tolerance, "gtol":1e-7,"maxiter":200}) for z in starts]
        best = min(candidates, key=lambda r: r.fun)
        A, rho = float(best.x[0]*scale), float(best.x[1])
        converged = bool(best.success)
        message = str(best.message)
        # A=0 makes rho unidentified; preserve the exact FH boundary likelihood.
        boundary = evaluate(0., 0.)[0]
        if boundary <= best.fun + 1e-8:
            A, rho, converged, message = 0., 0., True, "REML variance boundary; rho unidentified"
    else:
        opt = minimize_scalar(lambda a: evaluate(a*scale, 0.)[0], bounds=(0.,100.),method="bounded",options={"xatol":1e-10})
        A, rho, converged, message = float(opt.x*scale), 0., bool(opt.success),str(opt.message)
        if evaluate(0., 0.)[0] <= opt.fun:
            A, message = 0., "REML variance boundary"
    loss, fac, beta, Q, ViX = evaluate(A,rho)
    C, dC, Call, diagC, dCall = model.covariance(rho, full=True)
    Gs = A*Call
    ViGs = cho_solve(fac, Gs.T,check_finite=False)
    K = ViGs.T
    residual = y-X@beta
    mean = Xall@beta + K@residual
    g1 = A*diagC-np.sum(Gs*ViGs.T,axis=1)
    H = Xall-K@X
    g2 = np.sum((H@Q)*H,axis=1)
    plugin = np.maximum(g1+g2,0)
    # Parameter delta term: expected squared derivative of kriging adjustment.
    # The 2*g3 extension is reported separately; article does not define CIs.
    Vi = cho_solve(fac,np.eye(len(y)),check_finite=False)
    P = Vi-ViX@Q@ViX.T
    dVs = [C] + ([A*dC] if spatial and A>1e-12 else [])
    dGs = [Call] + ([A*dCall] if spatial and A>1e-12 else [])
    info = np.array([[.5*np.trace(P@a@P@b) for b in dVs] for a in dVs])
    covpar = np.linalg.pinv(info, rcond=1e-10)
    dK = [(dg-K@dv)@Vi for dg,dv in zip(dGs,dVs)]
    V = A*C+np.diag(D)
    g3 = sum(covpar[i,j]*np.sum((a@V)*b,axis=1) for i,a in enumerate(dK) for j,b in enumerate(dK))
    mse_delta = np.maximum(plugin+2*np.maximum(g3,0),0)
    return {"mean":mean,"mse_plugin":plugin,"mse_delta":mse_delta,"variance":A,"rho":rho,"beta":beta,"beta_cov":Q,"converged":converged,"message":message,"reml_nll":float(loss),"param_info_condition":float(np.linalg.cond(info)),"_model":model,"_sample_idx":s}


def fit_sikov(X, y, D, sample_idx, W, strata=None, *, spatial=True,
              outcome=None, covariate_names=None, uncertainty="delta", bootstrap_B=0,
              seed=20261010, bootstrap_min_success=.95):
    """Fit published FH/SAR per poverty stratum, no response leakage.

    X: N x p raw proportion auxiliaries + intercept; Altitud in km.
    y,D: vectors of len(sample_idx), OR full N (only sample_idx entries read).
    W: N x N row-standardized directed graph; block separated by poverty.
    strata: N labels (1,2,3). None means one generic family-control stratum.
    outcome: 'anemia'/'stunting' selects published tables; None uses all X.
    uncertainty: 'plugin' universal-kriging, 'delta' adds estimated parameter
      term (documented reconstruction), 'bootstrap' parametric REML B>=100.
    Return unbounded Gaussian mean/lower/upper. Do NOT clip favorable results.
    """
    X=np.asarray(X,float); y=np.asarray(y,float); D=np.asarray(D,float)
    sample_idx=np.asarray(sample_idx,int); W=np.asarray(W,float)
    n=len(X); strata=np.ones(n,int) if strata is None else np.asarray(strata)
    if X.ndim!=2 or W.shape!=(n,n) or len(strata)!=n:
        raise ValueError("incompatible X/W/strata")
    if len(np.unique(sample_idx))!=len(sample_idx) or np.any((sample_idx<0)|(sample_idx>=n)):
        raise ValueError("sample_idx must be unique valid indices")
    if len(y)==n: y=y[sample_idx]
    if len(D)==n: D=D[sample_idx]
    if len(y)!=len(sample_idx) or len(D)!=len(sample_idx):
        raise ValueError("y/D lengths must be N or sampled N")
    if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isfinite(W).all() or np.any(W<0) or np.any(np.diag(W)!=0):
        raise ValueError("finite X/y/W and zero-diagonal nonnegative W required")
    if np.any((y<0)|(y>1)):
        raise ValueError("Sikov direct estimands must be proportions in [0,1]")
    if uncertainty not in ("plugin","delta","bootstrap"):
        raise ValueError("unknown uncertainty")
    if uncertainty=="bootstrap" and bootstrap_B<100:
        raise ValueError("final bootstrap needs at least 100 replicates")
    if outcome is not None and covariate_names is None:
        raise ValueError("published-table mode requires covariate_names")
    mean=np.full(n,np.nan); mse=np.full(n,np.nan); plugin=np.full(n,np.nan); delta=np.full(n,np.nan)
    fits=[]; rng=np.random.default_rng(seed)
    for level in np.unique(strata):
        ids=np.flatnonzero(strata==level)
        sampled_mask=np.isin(sample_idx,ids)
        global_s=sample_idx[sampled_mask]
        local_s=np.searchsorted(ids,global_s)
        if not len(global_s):
            raise ValueError(f"poverty stratum {level} has no sampled areas")
        cols=np.arange(X.shape[1]); names=None
        if outcome is not None:
            names=TABLE_COVARIATES[outcome][bool(spatial)][int(level)]
            cols=select_columns(outcome,level,covariate_names,spatial)
        XX=X[ids][:,cols]
        WW=W[np.ix_(ids,ids)].copy()
        rows=WW.sum(axis=1)
        if spatial and len(ids)>1 and np.any(rows<=0):
            raise ValueError("W has isolated within-stratum areas: rebuild graph before fitting")
        WW=np.divide(WW,rows[:,None],out=np.zeros_like(WW),where=rows[:,None]>0)
        yy=y[sampled_mask]; DD=D[sampled_mask]
        fit=_fit_stratum(XX,yy,DD,local_s,WW,spatial)
        selected=fit["mse_plugin"] if uncertainty=="plugin" else fit["mse_delta"]
        bootstrap_success=0
        if uncertainty=="bootstrap":
            sq=[]
            A=fit["variance"]; beta=fit["beta"];rho=fit["rho"]
            factor=splu(sparse.eye(len(ids),format="csc")-rho*sparse.csc_matrix(WW)) if spatial else None
            for b in range(bootstrap_B):
                innovation=rng.normal(0,np.sqrt(A),len(ids))
                u=factor.solve(innovation) if spatial else innovation
                truth=XX@beta+u
                boot_y=truth[local_s]+rng.normal(0,np.sqrt(DD))
                try:
                    # Bootstrap Gaussian responses legitimately leave [0,1].
                    boot=_fit_stratum(XX,boot_y,DD,local_s,WW,spatial)
                    if boot["converged"] and np.isfinite(boot["mean"]).all():
                        sq.append((boot["mean"]-truth)**2)
                except (ValueError,np.linalg.LinAlgError,RuntimeError):
                    pass
            bootstrap_success=len(sq)
            if bootstrap_success<bootstrap_min_success*bootstrap_B:
                raise RuntimeError(f"parametric bootstrap only {bootstrap_success}/{bootstrap_B} valid fits")
            selected=np.mean(sq,axis=0)
        mean[ids]=fit["mean"];mse[ids]=selected;plugin[ids]=fit["mse_plugin"];delta[ids]=fit["mse_delta"]
        metadata={k:v for k,v in fit.items() if not k.startswith("_") and k not in ("mean","mse_plugin","mse_delta")}
        metadata.update(stratum=int(level),covariates=names,n_areas=len(ids),n_sampled=len(global_s),bootstrap_success=bootstrap_success,bootstrap_B=bootstrap_B if uncertainty=="bootstrap" else 0)
        fits.append(metadata)
    se=np.sqrt(mse)
    return {"mean":mean,"mse":mse,"mse_plugin":plugin,"mse_delta":delta,"lower":mean-1.959963984540054*se,"upper":mean+1.959963984540054*se,"fit":fits,"converged":all(f["converged"] for f in fits),"uncertainty":uncertainty,"scale":"proportion","numerical_replication":False}
