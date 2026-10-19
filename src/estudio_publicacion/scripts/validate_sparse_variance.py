#!/usr/bin/env python3
"""Independent finite-population check of the saved complete-response oracle D."""
from pathlib import Path
import json
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import numpy as np
from scipy.special import expit

rng=np.random.default_rng(2019453)
G,L,nchild,R=40,30,5,5000
cluster_eta=rng.normal(size=(G,2))@np.linalg.cholesky(np.array([[1.,.6],[.6,1.]])).T
p=expit(np.array([-1.,-1.8])+cluster_eta*np.array([.9,.8]))
Y=(rng.uniform(size=(G,L,2))<p[:,None,:]).astype(float)
truth=Y.mean(axis=(0,1))
cluster_means=Y.mean(axis=1)
within=np.mean([np.cov(y.T,ddof=1) for y in Y],axis=0)
checks=[]
for J in (1,3,10):
    first=(1-J/G)*np.cov(cluster_means.T,ddof=1)/J
    second=within*(1-nchild/L)/(J*nchild)
    oracle=first+second
    estimates=[];ultimate=[]
    for rep in range(R):
        ids=rng.choice(G,J,replace=False)
        observed=np.array([Y[i,rng.choice(L,nchild,replace=False)].mean(axis=0) for i in ids])
        estimates.append(observed.mean(axis=0))
        ultimate.append(np.zeros((2,2)) if J==1 else (1-J/G)*np.cov(observed.T,ddof=1)/J)
    empirical=np.cov(np.asarray(estimates).T,ddof=1)
    ratio=np.diag(empirical)/np.diag(oracle)
    checks.append(dict(psus=J,replications=R,truth=truth.tolist(),
        empirical_variance=empirical.tolist(),oracle_complete_design_variance=oracle.tolist(),
        empirical_over_oracle=ratio.tolist(),ultimate_cluster_mean_variance=np.mean(ultimate,axis=0).tolist(),
        passed=bool(np.all((ratio>.93)&(ratio<1.07)))))
result=dict(all_passed=all(c["passed"] for c in checks),checks=checks,
    scope="Known realized finite population; SRS of equal-size PSUs and SRS of five children per PSU; complete response",
    limitation="No claim that this exact complete-response oracle covariance equals the covariance under MAR ratio weighting")
out=Path(__file__).resolve().parents[1]/"resultados"/"simulaciones"/"validation_sparse_oracle.json"
out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if not result["all_passed"]:raise SystemExit(1)
