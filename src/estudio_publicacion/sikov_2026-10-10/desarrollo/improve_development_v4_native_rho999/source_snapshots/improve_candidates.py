"""Finite, common-information improvements to ICC; no truth is accepted here.

The stratum model adds sum-zero poverty-stratum deviations to every existing
area covariate, with a fixed shrinkage prior. The ensemble is a distribution
mixture of ICC conditional beta draws and the reference's normal plug-in
prediction distribution, not an average of confidence limits. Its weight is
selected only on the separately identified development populations.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
from scipy.linalg import helmert
from scipy.special import ndtr, ndtri

COMPETITORS = Path(__file__).resolve().parents[2] / "scripts" / "competitors"
if str(COMPETITORS) not in sys.path:
    sys.path.insert(0, str(COMPETITORS))

STRATUM_DEVIATION_SD = 0.75
ENSEMBLE_ICC_WEIGHTS = (0.25, 0.50, 0.75)
CANDIDATES = {
    "ICC": {"family": "ICC", "stratum_deviations": False},
    "ICC_poverty": {"family": "ICC", "stratum_deviations": True,
                    "stratum_deviation_sd": STRATUM_DEVIATION_SD},
    **{f"ICC_SAR_w{int(100*w):02d}": {"family": "ICC_SAR_mixture",
                                        "icc_weight": w}
       for w in ENSEMBLE_ICC_WEIGHTS},
}


def _poverty_labels(strata, n):
    labels = np.asarray(strata)
    if labels.shape != (n,) or not np.isin(labels, [1, 2, 3]).all():
        raise ValueError("Provide one prespecified poverty label 1/2/3 per area")
    return labels.astype(int)


def stratified_design(X, strata, beta_sd=2.5, deviation_sd=STRATUM_DEVIATION_SD):
    """Sum-zero, partially pooled deviations; no added auxiliary information.

    X must already use the common covariate encoding/scaling for both methods.
    Reparameterization preserves the original Stan model's scalar coefficient
    prior: deviation coefficients have SD 0.75 while original slopes retain
    SD 2.5. Orthogonal Helmert contrasts give symmetric treatment to the three
    poverty groups. The reference-group coding is deliberately avoided.
    """
    X = np.asarray(X, float)
    if X.ndim != 2 or not np.isfinite(X).all() or not np.allclose(X[:, 0], 1):
        raise ValueError("Finite common X with intercept in column zero required")
    labels = _poverty_labels(strata, len(X))
    if beta_sd <= 0 or deviation_sd <= 0:
        raise ValueError("Prior standard deviations must be positive")
    contrasts = helmert(3, full=False).T[labels-1]
    deviations = (X[:, :, None] * contrasts[:, None, :]).reshape(len(X), -1)
    return np.column_stack((X, deviations * deviation_sd / beta_sd))


def fit_icc_poverty(X, y, n_eff, sample_idx, units=None, basis=None,
                    strata=None, seed=20261010, config=None,
                    sampler_config=None, output_dir=None, integration="hmc"):
    """Use original ICC machinery with a fixed, expanded common-X design.

    Laplace is an explicitly labelled development diagnostic. HMC and its
    unchanged convergence gates are required for a final ICC comparison.
    Poverty must come from the same declared information as the reference;
    converting an unrelated simulation covariate into poverty is forbidden.
    """
    from beta_power import CONFIG
    cfg = {**CONFIG, **(config or {})}
    expanded = stratified_design(X, strata, beta_sd=cfg["beta_sd"])
    if integration == "hmc":
        from beta_power_hmc import fit
        result = fit(expanded, y, n_eff, sample_idx, units, basis, seed=seed,
                     config=config, sampler_config=sampler_config,
                     output_dir=output_dir)
    elif integration == "laplace_development":
        from beta_power import fit
        result = fit(expanded, y, n_eff, sample_idx, units, basis,
                     seed=seed, config=config)
    else:
        raise ValueError("integration must be hmc or laplace_development")
    for endpoint in result["status"]:
        endpoint["poverty_improvement"] = {
            "coding": "orthonormal Helmert sum-zero deviations",
            "original_X_columns": int(np.asarray(X).shape[1]),
            "expanded_X_columns": int(expanded.shape[1]),
            "stratum_deviation_sd": STRATUM_DEVIATION_SD,
            "outcome_or_truth_used_for_strata": False,
            "same_information_as_Sikov": True,
            "integration_role": integration,
        }
    return result


def _draw_marginals(icc):
    """Keep all endpoint draws if retry lengths differ; no joint claim."""
    if "marginal_draws" in icc:
        draws = [np.asarray(x, float) for x in icc["marginal_draws"]]
    elif "p_draws" in icc:
        joint = np.asarray(icc["p_draws"], float)
        if joint.ndim != 3:
            raise ValueError("ICC p_draws must have shape draws x areas x endpoints")
        draws = [joint[:, :, j] for j in range(joint.shape[2])]
    else:
        raise ValueError("ICC posterior draws are required; three saved quantiles are insufficient")
    if any(x.ndim != 2 or len(x) < 128 or not np.isfinite(x).all()
           or np.any((x < 0) | (x > 1)) for x in draws):
        raise ValueError("Finite ICC marginal beta draws in [0,1] required")
    return draws


def mixture_prediction(icc, sar, icc_weight, alpha=0.05):
    """Normal-SAR / empirical-beta-ICC mixture CDF, retaining disagreement.

    SAR must supply mean and mse on the SAME all-area/endpoints universe.
    Bounds are intentionally untruncated: the reference Gaussian component
    can leave [0,1]. Neither truth nor a test-fold response is accepted here.
    The final assessment measures coverage instead of asserting it.
    """
    w = float(icc_weight)
    if not 0 < w < 1 or not 0 < alpha < 1:
        raise ValueError("Mixture weights must be interior and alpha in (0,1)")
    draws = _draw_marginals(icc)
    p = np.asarray(icc["p_mean"], float)
    variance_icc = np.asarray(icc["p_variance"], float)
    mu = np.asarray(sar["mean"], float)
    variance_sar = np.asarray(sar["mse"], float)
    if p.ndim == 1:
        p, variance_icc = p[:, None], variance_icc[:, None]
    if mu.ndim == 1:
        mu, variance_sar = mu[:, None], variance_sar[:, None]
    if not (p.shape == mu.shape == variance_icc.shape == variance_sar.shape):
        raise ValueError("Both estimators must predict exactly the same universe")
    if len(draws) != p.shape[1] or any(x.shape[1] != p.shape[0] for x in draws):
        raise ValueError("ICC draw universe does not match point predictions")
    if (not np.isfinite(mu).all() or not np.isfinite(variance_sar).all()
            or np.any(variance_sar < 0) or not np.isfinite(p).all()
            or not np.isfinite(variance_icc).all() or np.any(variance_icc < 0)):
        raise ValueError("Nonnegative finite component uncertainties required")
    lower, upper = np.empty_like(p), np.empty_like(p)
    max_cdf_step = 0.0
    for j, raw in enumerate(draws):
        ordered = np.sort(raw, axis=0)
        sd = np.sqrt(variance_sar[:, j])
        # Normal zero-variance limits are point masses, not numerical floors.
        def cdf(q):
            beta_cdf = np.mean(ordered <= q[None, :], axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                normal_cdf = ndtr((q-mu[:, j])/sd)
            normal_cdf = np.where(sd == 0, (q >= mu[:, j]).astype(float), normal_cdf)
            return w * beta_cdf + (1-w) * normal_cdf
        for target, destination in ((alpha/2, lower), (1-alpha/2, upper)):
            left = np.minimum(0, mu[:, j]-12*sd)
            right = np.maximum(1, mu[:, j]+12*sd)
            for _ in range(56):
                middle = (left+right)/2
                high = cdf(middle) >= target
                right = np.where(high, middle, right)
                left = np.where(high, left, middle)
            destination[:, j] = right
        max_cdf_step = max(max_cdf_step, w/len(raw))
    mean = w*p+(1-w)*mu
    variance = (w*variance_icc+(1-w)*variance_sar
                + w*(1-w)*(p-mu)**2)
    return {"mean": mean, "lower": lower, "upper": upper,
            "p_variance": variance,
            "status": {"icc_weight": w, "sar_weight": 1-w,
                       "uncertainty": "mixture of ICC posterior marginal and SAR normal plug-in MSE law",
                       "quantiles": "generalized inverse of mixture CDF, 56 bisections",
                       "max_empirical_CDF_jump": max_cdf_step,
                       "between_component_disagreement_in_variance": True,
                       "untruncated_SAR_component": True,
                       "no_joint_endpoint_distribution_claimed": True}}
