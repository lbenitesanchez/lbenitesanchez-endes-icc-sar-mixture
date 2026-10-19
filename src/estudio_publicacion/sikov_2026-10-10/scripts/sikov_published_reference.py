"""Sampled-only Sikov SAR with declared directed non-sampled extension.

The 2023 article identifies a D-dimensional sampled graph but leaves its exact
neighbor rule and non-sampled algorithm undisclosed. We fix sampled k6 and use
the authors' 2024 preprint section 2, equations (7)--(10), option 3 extension:
W11=W_D, W10=0; non-sampled rows use k6 among all within-stratum domains.
No outcomes or evaluation truth determine this graph or its tuning constants.
"""
from __future__ import annotations
import numpy as np
from sikov_model import knn_weights
from sikov_bootstrap import fit_sikov_bootstrap

METHOD='Sikov_SAR_Published'
BC_METHOD=METHOD+'_BC'
K=6

def triangular_weights(coords,sample_idx,strata,k=K):
    coords=np.asarray(coords,float);s=np.asarray(sample_idx,int)
    strata=np.asarray(strata);n=len(coords)
    if len(np.unique(s))!=len(s) or np.any((s<0)|(s>=n)):
        raise ValueError('sample_idx must be distinct valid area indices')
    if len(strata)!=n:raise ValueError('strata and coordinates differ')
    W=knn_weights(coords,k=k,strata=strata)
    WD=knn_weights(coords[s],k=k,strata=strata[s])
    W[s,:]=0.;W[np.ix_(s,s)]=WD
    if np.any(W[np.ix_(s,np.setdiff1d(np.arange(n),s))]!=0):
        raise RuntimeError('Sampled-to-unsampled links must be zero')
    if not np.allclose(W.sum(axis=1),1.,atol=1e-14,rtol=0):
        raise ValueError('Each stratum needs at least two sampled and total domains')
    return W

def fit_sikov_published_bootstrap(X,y,D,sample_idx,coords,strata,*,
                                outcome,covariate_names,B=100,seed=20261010,
                                interval_estimator='naive',innovations=None):
    """Reuse frozen model/bootstrap, changing only the predeclared graph.

    Inputs y,D contain finite training endpoints with D>0. Caller performs the
    common endpoint mask. Return has frozen wrapper keys plus graph metadata.
    """
    W=triangular_weights(coords,sample_idx,strata)
    result=fit_sikov_bootstrap(X,y,D,sample_idx,W,strata,spatial=True,
            outcome=outcome,covariate_names=covariate_names,B=B,seed=seed,
            interval_estimator=interval_estimator,innovations=innovations)
    result['graph_extension']={
        'sampled_graph':'within-stratum k6 on sampled coordinates only',
        'W11':'sampled-only row-normalized kNN6','W10':'zero',
        'non_sampled_rows':'within-stratum k6 on all coordinates, excluding self',
        'non_sampled_predictor':'X0*beta+(I-rho*W00)^(-1)*rho*W01*uhat_sampled',
        'source':'Sikov and Cerda-Hernandez 2024 author preprint, section2 equations7-10 option3',
        'exact_2023_neighbor_rule_or_non_sampled_implementation_known':False,
    }
    return result
