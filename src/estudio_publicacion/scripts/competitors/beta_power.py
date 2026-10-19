"""Spatial beta shrinkage with a design-weighted power likelihood.

The fractional k and n are power-likelihood exponents, not integer binomial
counts. Hyperparameter uncertainty is integrated by a Laplace approximation.
This is a proposed approximation, not an exact survey posterior or a new
distributional family. No held-out outcome, population outcome or truth enters
the fit. The working intracluster adjustment is estimated from training units.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, digamma, expit, ndtri
from scipy.stats import beta as beta_distribution
from scipy.stats import qmc
from beta_power_stable import log_likelihood_and_gradient

CONFIG = dict(beta_sd=2.5, intercept_sd=5.0, spatial_sd=0.5,
              log_kappa_mean=float(np.log(20)), log_kappa_sd=1.5,
              posterior_draws=2048, cluster_adjustment=True,
              implementation="collapsed beta-power hyperposterior Laplace v1")


def working_icc(units, outcome):
    """Unweighted nested ANOVA MOM with area fixed effects.

    Only areas with >=2 represented PSUs contribute. This is a working binary
    correlation estimate, not a design-consistent weighted variance component.
    Its uncertainty is not included in the base intervals and is checked in a
    sensitivity analysis. PSU totals for each area's effective PSU count use
    the endpoint weights. Single-child PSUs cannot identify within-PSU ICC.
    """
    if units is None:
        return 0.0, {"available": False, "reason": "no training unit records"}
    yc, wc = f"y_{outcome}", f"weight_{outcome}"
    u = units.loc[np.isfinite(units[yc]) & np.isfinite(units[wc]) & (units[wc]>0)].copy()
    g = u.groupby(["area", "psu"])[yc].agg(["size", "mean", "var"])
    counts = g.groupby(level=0).size()
    g = g.loc[g.index.get_level_values(0).isin(counts[counts>=2].index)]
    if len(g)==0:
        return 0.0, {"available": False, "reason": "no areas with multiple PSUs"}
    N=float(g["size"].sum()); J=len(g); A=g.index.get_level_values(0).nunique()
    dfw=N-J; dfb=J-A
    if dfw<=0 or dfb<=0:
        return 0.0, {"available": False, "reason": "no residual ICC degrees of freedom", "N":N,"J":J,"A":A}
    totals=g.groupby(level=0)["size"].sum()
    means=(g["size"]*g["mean"]).groupby(level=0).sum()/totals
    area_means=g.index.get_level_values(0).map(means).to_numpy(float)
    ssw=float(((g["size"]-1)*g["var"].fillna(0)).sum())
    ssb=float((g["size"]*(g["mean"].to_numpy()-area_means)**2).sum())
    C=(N-float(((g["size"]**2).groupby(level=0).sum()/totals).sum()))/dfb
    msw=ssw/dfw; msb=ssb/dfb
    tau=max((msb-msw)/C,0.0)
    icc=tau/(tau+msw) if tau+msw>0 else 0.0
    return float(icc), dict(available=True,N=N,J=J,A=A,df_within=dfw,df_between=dfb,
                           MS_within=msw,MS_between=msb,C=C,ICC=float(icc),
                           uncertainty_propagated=False, weighting="unweighted working nested ANOVA")


def cluster_effective_n(units, outcome, sample_idx, n_eff, icc):
    if units is None or icc==0:
        return np.asarray(n_eff,float).copy(),np.full(len(n_eff),np.nan)
    yc,wc=f"y_{outcome}",f"weight_{outcome}"
    u=units.loc[np.isfinite(units[yc]) & np.isfinite(units[wc]) & (units[wc]>0)]
    totals=u.groupby(["area","psu"])[wc].sum()
    je=(totals.groupby(level=0).sum()**2)/(totals**2).groupby(level=0).sum()
    je=je.reindex(sample_idx).to_numpy(float)
    ne=np.asarray(n_eff,float)
    deff=1+np.maximum(ne/np.maximum(je,1)-1,0)*icc
    return ne/deff,je


class EndpointPosterior:
    def __init__(self, X, y, n, sample_idx, basis=None, config=None):
        self.config={**CONFIG,**(config or {})}
        self.X=np.asarray(X,float)
        self.B=np.zeros((len(X),0)) if basis is None else np.asarray(basis,float)
        self.Z=np.c_[self.X,self.B]
        self.idx=np.asarray(sample_idx,int)
        self.y=np.asarray(y,float);self.n=np.asarray(n,float)
        keep=np.isfinite(self.y)&np.isfinite(self.n)&(self.n>0)
        self.idx=self.idx[keep];self.y=self.y[keep];self.n=self.n[keep]
        if np.any((self.y<0)|(self.y>1)): raise ValueError("Prevalence outside [0,1]")
        self.k=self.n*self.y
        self.Zs=self.Z[self.idx]
        sd=np.full(self.X.shape[1],self.config["beta_sd"]);sd[0]=self.config["intercept_sd"]
        self.sd=np.r_[sd,np.full(self.B.shape[1],self.config["spatial_sd"]),self.config["log_kappa_sd"]]
        self.prior_mean=np.zeros(len(self.sd));self.prior_mean[-1]=self.config["log_kappa_mean"]

    def objective(self, h):
        eta=self.Zs@h[:-1]
        terms,deta,dlk=log_likelihood_and_gradient(eta,h[-1],self.k,self.n-self.k)
        ll=np.sum(terms)
        grad=np.r_[self.Zs.T@deta,np.sum(dlk)]
        dev=(h-self.prior_mean)/self.sd
        logprior=-0.5*np.dot(dev,dev)
        gradprior=-(h-self.prior_mean)/self.sd**2
        return float(-ll-logprior),-grad-gradprior

    def optimize(self):
        start=self.prior_mean.copy()
        prevalence=np.average(self.y,weights=self.n) if len(self.y) else .5
        start[0]=np.log(np.clip(prevalence,1e-4,1-1e-4)/(1-np.clip(prevalence,1e-4,1-1e-4)))
        bounds=[(-16,16)]*(len(start)-1)+[(-6,12)]
        fits=[minimize(self.objective,start,jac=True,method="L-BFGS-B",bounds=bounds,
                       options=dict(maxiter=500,ftol=1e-11,gtol=1e-6))]
        if not fits[0].success or np.max(abs(fits[0].jac))>.005:
            alt=start.copy();alt[-1]=np.log(5)
            fits.append(minimize(self.objective,alt,jac=True,method="L-BFGS-B",bounds=bounds,
                                 options=dict(maxiter=700,ftol=1e-12,gtol=1e-6)))
            # A projected initial step can produce an extreme logistic surface
            # before line search. BFGS provides an independent conservative
            # start; the final stationary point remains checked explicitly.
            other=minimize(self.objective,start,jac=True,method="BFGS",
                           options=dict(maxiter=700,gtol=1e-6))
            fits.append(other)
            fits.append(minimize(self.objective,other.x,jac=True,method="L-BFGS-B",bounds=bounds,
                                 options=dict(maxiter=700,ftol=1e-12,gtol=1e-6)))
        # Failed line searches can report NaN ``fun`` while returning the last
        # finite x. Re-evaluate each candidate rather than sorting NaNs.
        for candidate in fits:
            candidate.fun,candidate.jac=self.objective(candidate.x)
        valid=[candidate for candidate in fits if np.isfinite(candidate.fun) and np.all(np.isfinite(candidate.jac))]
        if not valid: raise RuntimeError("No finite posterior optimization result")
        optimum=min(valid,key=lambda f:f.fun)
        h=optimum.x;p=len(h);H=np.empty((p,p))
        for j in range(p):
            delta=1e-4*max(abs(h[j]),1);up=h.copy();down=h.copy()
            up[j]+=delta;down[j]-=delta
            H[:,j]=(self.objective(up)[1]-self.objective(down)[1])/(2*delta)
        H=(H+H.T)/2
        eigen,vec=np.linalg.eigh(H)
        floor=max(np.max(eigen)*1e-9,1e-8)
        repaired=np.maximum(eigen,floor)
        self.cov=(vec/repaired)@vec.T;self.mode=h
        self.status=dict(success=bool(np.isfinite(optimum.fun) and np.max(abs(optimum.jac))<.01 and np.min(eigen)>0),
            optimizer_reported_success=bool(optimum.success),
            message=str(optimum.message), max_gradient=float(np.max(abs(optimum.jac))),
            hessian_min_eigenvalue=float(np.min(eigen)), hessian_repairs=int(np.sum(eigen<floor)),
            bounds_touched=bool(any(abs(h[j]-bounds[j][k])<1e-3 for j in range(p) for k in (0,1))),
            log_kappa_mode=float(h[-1]),kappa_mode=float(np.exp(h[-1])),
            posterior_approximation="joint hyperparameter Laplace; exact conditional beta draws",
            config=self.config)
        return self

    def draws(self, seed, draws=None):
        draws=draws or self.config["posterior_draws"]
        if draws & (draws-1): raise ValueError("Sobol draws must be a power of two")
        U=qmc.Sobol(d=len(self.mode)+len(self.X),scramble=True,seed=seed).random_base2(int(np.log2(draws)))
        U=np.clip(U,1e-12,1-1e-12)
        L=np.linalg.cholesky(self.cov)
        h=self.mode+ndtri(U[:,:len(self.mode)])@L.T
        eta=h[:,:-1]@self.Z.T;mu=expit(eta);mubar=expit(-eta)
        # Clipping here is a numerical guard; affected draws are reported.
        kap=np.exp(np.clip(h[:,-1],-15,20))[:,None]
        a=np.maximum(kap*mu,1e-10);b=np.maximum(kap*mubar,1e-10)
        a[:,self.idx]+=self.k;b[:,self.idx]+=self.n-self.k
        means=a/(a+b)
        posterior=beta_distribution.ppf(U[:,len(self.mode):],a,b)
        self.status["hyperparameter_guard_draws"]=int(np.sum((h[:,-1]<-15)|(h[:,-1]>20)))
        return dict(p_mean=means.mean(axis=0),p_variance=np.mean(a*b/((a+b)**2*(a+b+1)),axis=0)+np.var(means,axis=0),
                    lower95=np.quantile(posterior,.025,axis=0),upper95=np.quantile(posterior,.975,axis=0),
                    p_draws=posterior,hyper_draws=h,status=self.status)


def fit(X,y,n_eff,sample_idx,units=None,basis=None,seed=20261005,config=None):
    config={**CONFIG,**(config or {})}
    y=np.asarray(y,float);ne=np.asarray(n_eff,float)
    if y.ndim==1:y=y[:,None];ne=ne[:,None]
    outputs=[];status=[];effective=[]
    for j,name in enumerate(("anemia","stunting")[:y.shape[1]]):
        icc,info=working_icc(units,name) if config["cluster_adjustment"] else (0.,{"available":False,"reason":"disabled"})
        n,je=cluster_effective_n(units,name,sample_idx,ne[:,j],icc)
        post=EndpointPosterior(X,y[:,j],n,sample_idx,basis,config).optimize()
        result=post.draws(seed+j*1009)
        result["status"]["ICC_working_estimate"]=info
        result["status"]["effective_n_min"]=float(np.nanmin(n));result["status"]["effective_n_median"]=float(np.nanmedian(n))
        outputs.append(result);status.append(result["status"]);effective.append(n)
    return dict(p_mean=np.column_stack([o["p_mean"] for o in outputs]),
                lower95=np.column_stack([o["lower95"] for o in outputs]),
                upper95=np.column_stack([o["upper95"] for o in outputs]),
                p_variance=np.column_stack([o["p_variance"] for o in outputs]),
                p_draws=np.stack([o["p_draws"] for o in outputs],axis=-1),
                effective_n=np.column_stack(effective),status=status)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("input");parser.add_argument("--output",required=True)
    parser.add_argument("--nonspatial",action="store_true")
    args=parser.parse_args();folder=Path(args.input);d=np.load(folder/"direct.npz");areas=pd.read_csv(folder/"area.csv")
    X=np.c_[np.ones(len(areas)),areas[["X1","X2"]]]
    result=fit(X,d["y"],d["n_eff"],d["sample_idx"],pd.read_csv(folder/"sample.csv"),
               None if args.nonspatial else np.load(folder/"spatial_basis.npy"))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"predictions.npz",**{k:result[k] for k in ["p_mean","lower95","upper95","p_variance","effective_n"]})
    (out/"status.json").write_text(json.dumps(result["status"],indent=2))
