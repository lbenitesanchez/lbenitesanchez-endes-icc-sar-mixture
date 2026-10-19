"""Predefined structural sensitivity truths; no estimator outcomes enter here.

Only latent area effects change. The archived PSU/MAR sampling algorithm stays
unchanged. New scenario keys have distinct SeedSequence context indices and do
not claim to reconstruct any phase-one finite population.
"""
import numpy as np
from scipy.special import logit

STRUCTURAL_SCENARIOS=("sparse_psu_iid","sparse_psu_bym2","sparse_srs_iid")


def rook_icar_factor(coords):
    """Sum-zero ICAR generalized inverse, geometric marginal-variance scaling."""
    coords=np.asarray(coords,float)
    distance=np.sqrt(np.sum((coords[:,None]-coords[None,:])**2,axis=2))
    positive=distance[distance>1e-12]
    spacing=float(positive.min())
    W=np.isclose(distance,spacing,rtol=0,atol=1e-12).astype(float)
    Q=np.diag(W.sum(axis=1))-W
    values,vectors=np.linalg.eigh(Q)
    use=values>1e-10
    if use.sum()!=len(coords)-1:raise ValueError("ICAR structural graph must be connected")
    factor=vectors[:,use]/np.sqrt(values[use])[None,:]
    variance=np.sum(factor**2,axis=1)
    scaling=float(np.exp(np.mean(np.log(variance))))
    factor/=np.sqrt(scaling)
    np.testing.assert_allclose(factor.sum(axis=0),0,atol=1e-11)
    np.testing.assert_allclose(np.exp(np.mean(np.log(np.sum(factor**2,axis=1)))),1.,atol=1e-12)
    return factor,dict(graph="connected rook-grid",constraint="sumzero",
        precision_rank=int(use.sum()),unscaled_generalized_variance=scaling,
        scaling="Qplus divided by geometric mean of its marginal variances")


def latent_structural(rng,X,coords,scenario):
    beta=np.array([[logit(.28),logit(.14)],[.30,.22],[.18,.12]])
    sigma=np.array([.45,.40])
    # Published competitors are endpoint-independent; these contexts additionally
    # remove the original cross-outcome latent area correlation (rho=.6).
    independent=rng.normal(size=(len(X),2))*sigma
    if scenario in ("sparse_psu_iid","sparse_srs_iid"):return X@beta+independent
    if scenario!="sparse_psu_bym2":raise ValueError("Unknown structural scenario")
    factor,_=rook_icar_factor(coords)
    field=(factor@rng.normal(size=(factor.shape[1],2)))*sigma
    return X@beta+np.sqrt(.5)*independent+np.sqrt(.5)*field


def structural_spec(scenario,coords):
    common=dict(latent_area_cross_outcome_rho=0.,latent_area_sd=[.45,.40],
        unit_binary_copula_rho=.4,within_area_PSU_effect_rho=.6,
        sampling="archived two-stage PSU algorithm and endpoint MAR, unchanged",
        original_population_exact_matching_claimed=False)
    if scenario=="sparse_srs_iid":
        return {**common,"latent_structure":"iid Gaussian area effects","spatial_fraction":0.,
            "sampling":"archived simple-random-without-replacement and endpoint MAR algorithm, unchanged",
            "within_area_PSU_effect_rho":None,"within_area_PSU_effect_present":False}
    if scenario=="sparse_psu_iid":return {**common,"latent_structure":"iid Gaussian area effects","spatial_fraction":0.}
    _,spec=rook_icar_factor(coords)
    return {**common,"latent_structure":"BYM2-like scaled ICAR+iid Gaussian","spatial_fraction":.5,**spec}
