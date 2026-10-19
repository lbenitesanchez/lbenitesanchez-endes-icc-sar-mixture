#!/usr/bin/env python3
"""Independent numerical checks of simulation identities and tail integration."""
from pathlib import Path
import importlib.util
import json
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.special import gammaln, logit, ndtr
from scipy.stats import multivariate_normal

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("simulation", Path(__file__).with_name("03_simulation.py"))
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)
rng = np.random.default_rng(673212)
checks = {}

# E[(chi2_df/df)^lambda] is known independently of models.py. For fixed correct
# reference D, the proposed moment adjustment must recover E[D_smoothed]=D.
moment_checks = []
for df in (1, 2, 4, 8):
    lam = df/(df+20)
    chi = rng.chisquare(df, 200000)/df
    log_moment = -lam*np.log(df/2)+gammaln(df/2+lam)-gammaln(df/2)
    corrected = chi**lam / np.exp(log_moment)
    se = corrected.std(ddof=1)/np.sqrt(len(corrected))
    z = (corrected.mean()-1)/se
    moment_checks.append(dict(df=df, lambda_=lam, mean=float(corrected.mean()),
                              mcse=float(se), z=float(z), passed=bool(abs(z)<5)))
checks["chi_square_moment_identity"] = moment_checks

# Compare the fixed quadrature against a separate multivariate Normal CDF.
mu = rng.normal(-.5, .8, (30,2))
sd = rng.uniform(.05, 1, (30,2))
rho = rng.uniform(-.95,.95,30)
R = np.stack([np.stack([np.ones(30),rho],-1),
              np.stack([rho,np.ones(30)],-1)],-2)
cov = sd[:,:,None]*R*sd[:,None,:]
a = (logit(sim.THRESHOLDS)-mu)/sd
reference = np.array([1-ndtr(z[0])-ndtr(z[1])+multivariate_normal.cdf(
    z, mean=[0,0], cov=[[1,r],[r,1]], rng=np.random.default_rng(42),
    maxpts=500000, abseps=1e-9, releps=1e-9) for z,r in zip(a,rho)])
quadrature = sim.probability_both(mu,cov)
error = float(np.max(np.abs(quadrature-reference)))
checks["bivariate_normal_tail"] = dict(max_abs_error=error, tolerance=.0002,
                                        passed=bool(error<.0002))

# A fixed finite population with informative inclusion and response validates
# the Poisson/IPW covariance formula at a sample size where linearization is
# expected to work. It does not assert exact unbiasedness of a ratio estimator.
N,R = 1000,5000
H = rng.binomial(1,.45,N)
rates = np.column_stack([.20+.25*H,.09+.12*H])
Y = (rng.uniform(size=(N,2))<rates).astype(float)
truth = Y.mean(axis=0)
size = np.where(H==1,1.8,.55)
pi = 70*size/size.sum()
response = np.column_stack([np.where(H==1,.82,.95),np.where(H==1,.88,.97)])
q = pi[:,None]*response
means, variances = [],[]
for rep in range(R):
    take = rng.uniform(size=N)<pi
    seen = take[:,None] & (rng.uniform(size=(N,2))<response)
    w = np.where(seen,1/q,0.)
    den = w.sum(axis=0)
    p = np.sum(w*Y,axis=0)/den
    residual = Y-p
    D = np.zeros((2,2))
    D[np.diag_indices(2)] = np.sum(w*w*(1-q)*residual**2,axis=0)/den**2
    both = seen.all(axis=1)
    D[0,1] = D[1,0] = np.sum(both*(1-pi)*w[:,0]*w[:,1]*residual[:,0]*residual[:,1])/np.prod(den)
    means.append(p);variances.append(D)
means = np.asarray(means); variances=np.asarray(variances)
empirical=np.cov(means.T,ddof=1)
estimated=variances.mean(axis=0)
ratio=np.diag(estimated)/np.diag(empirical)
bias=means.mean(axis=0)-truth
checks["finite_population_poisson_linearization"] = dict(
    replications=R, true_proportions=truth.tolist(), mean_bias=bias.tolist(),
    bias_mcse=(means.std(axis=0,ddof=1)/np.sqrt(R)).tolist(),
    empirical_covariance=empirical.tolist(), mean_estimated_covariance=estimated.tolist(),
    marginal_variance_ratios=ratio.tolist(),
    passed=bool(np.all(np.abs(bias)<.005) and np.all((ratio>.80)&(ratio<1.2))),
    limitation="Hájek ratios and their covariance are first-order approximations; sample mean bias is not forced to zero")

checks["all_passed"] = bool(all(x["passed"] for x in moment_checks) and
                             checks["bivariate_normal_tail"]["passed"] and
                             checks["finite_population_poisson_linearization"]["passed"])
out=root/"resultados"/"simulaciones"/"validation_generator.json"
out.write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
if not checks["all_passed"]:
    raise SystemExit(1)
