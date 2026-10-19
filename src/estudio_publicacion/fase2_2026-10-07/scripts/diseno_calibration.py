"""Full-frame first-stage Taylor contributions and replicate variance calibration.

This component never fits prevalence or evaluates held-out outcomes. Replicates
are a first-stage, with-replacement approximation: lower-stage sampling,
measurement nonresponse and lonely-stratum uncertainty are not identified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG = {
    "replicates": 1024,
    "mixture_states": 8,
    "replicate_seed": 20261007031,
    "state_seed": 20261007041,
    "shrinkage_df": 4.0,
    "ridge_slopes": 1.0,
    "variance_numerical_floor": 1e-15,
    "singleton_policy": "fixed observed adjusted Taylor contribution; uncertainty not propagated",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_from_households(households):
    """Inventory PSUs before domain, age or response filtering."""
    frame = households[["psu_key", "stratum"]].drop_duplicates().copy()
    if frame.psu_key.duplicated().any():
        raise ValueError("PSU keys must identify exactly one stratum")
    return frame.sort_values(["stratum", "psu_key"]).reset_index(drop=True)


def taylor_contributions(frame, units, outcome, area_ids):
    """Exact nonnegative decomposition of first-stage Taylor variance.

    Scores of PSUs with no children in a domain are zero. For m>1 the
    contribution is m/(m-1)*(score - full-stratum score mean)^2. For m=1
    it is score^2, reproducing the adjusted lonely-PSU rule for a ratio
    whose full-frame grand score is zero.
    """
    area_ids = np.asarray(area_ids)
    psu_lookup = dict(zip(frame.psu_key, np.arange(len(frame))))
    area_lookup = dict(zip(area_ids, np.arange(len(area_ids))))
    yc = f"y_{outcome}"
    wc = f"weight_{outcome}"
    u = units.loc[units.area.isin(area_ids) & np.isfinite(units[yc])
                  & np.isfinite(units[wc]) & (units[wc] > 0)].copy()
    if not u.psu_key.isin(frame.psu_key).all():
        raise ValueError("Observed child PSU absent from complete household frame")
    narea = len(area_ids)
    denominator = np.zeros(narea)
    numerator = np.zeros(narea)
    squared_weight = np.zeros(narea)
    records = np.zeros(narea, dtype=int)
    ai = u.area.map(area_lookup).to_numpy(int)
    pi = u.psu_key.map(psu_lookup).to_numpy(int)
    w = u[wc].to_numpy(float)
    y = u[yc].to_numpy(float)
    if np.any((y < 0) | (y > 1)):
        raise ValueError("Binary outcome outside [0,1]")
    np.add.at(denominator, ai, w)
    np.add.at(numerator, ai, w*y)
    np.add.at(squared_weight, ai, w*w)
    np.add.at(records, ai, 1)
    direct = np.divide(numerator, denominator, out=np.full(narea, np.nan), where=denominator > 0)
    kish = np.divide(denominator**2, squared_weight, out=np.full(narea, np.nan), where=squared_weight > 0)
    scores = np.zeros((len(frame), narea))
    psu_den = np.zeros_like(scores)
    np.add.at(scores, (pi, ai), w*(y-direct[ai])/denominator[ai])
    np.add.at(psu_den, (pi, ai), w)
    psu_count = np.sum(psu_den > 0, axis=0)
    psu_eff = np.divide(denominator**2, np.sum(psu_den**2, axis=0),
                        out=np.full(narea, np.nan), where=denominator > 0)
    contributions = np.zeros_like(scores)
    represented_strata = np.zeros(narea, dtype=int)
    full_singletons = np.zeros(len(frame), dtype=bool)
    for _, row_idx in frame.groupby("stratum", sort=True).indices.items():
        ix = np.asarray(row_idx)
        m = len(ix)
        represented_strata += np.any(psu_den[ix] > 0, axis=0)
        if m > 1:
            centered = scores[ix] - scores[ix].mean(axis=0)
            contributions[ix] = m/(m-1)*centered**2
        else:
            contributions[ix] = scores[ix]**2
            full_singletons[ix] = True
    variance = contributions.sum(axis=0)
    variance[denominator == 0] = np.nan
    return dict(area_ids=area_ids, y=direct, n_kish=kish, n_records=records,
                n_psu=psu_count, n_psu_eff=psu_eff, n_strata=represented_strata,
                df_proxy=np.maximum(psu_count-represented_strata, 0),
                variance=variance, contributions=contributions,
                singleton_contribution=contributions[full_singletons].sum(axis=0),
                full_singleton_mask=full_singletons,
                full_score_sums=scores.sum(axis=0), scores=scores, observed_unit_count=len(u))


def stratified_psu_counts(frame, config=None):
    """Ordinary m-draw full-frame PSU bootstrap, within fixed full strata.

    These are counts for bootstrapping the centered linearized PSU moment
    estimator of variance. They are not Rao-Wu weights for replicated ratios.
    """
    cfg = {**CONFIG, **(config or {})}
    rng = np.random.default_rng(cfg["replicate_seed"])
    b = cfg["replicates"]
    counts_all = np.ones((b, len(frame)), dtype=int)
    checks = []
    for stratum, row_idx in frame.groupby("stratum", sort=True).indices.items():
        ix = np.asarray(row_idx)
        m = len(ix)
        if m > 1:
            counts = rng.multinomial(m, np.repeat(1/m, m), size=b)
            counts_all[:, ix] = counts
            checks.append(dict(stratum=str(stratum), m=m,
                               max_total_weight_error=float(np.max(abs(counts_all[:, ix].sum(axis=1)-m)))))
        else:
            checks.append(dict(stratum=str(stratum), m=m, max_total_weight_error=0.0,
                               unidentified_uncertainty_fixed=True))
    if not all(c["max_total_weight_error"] < 1e-10 for c in checks):
        raise AssertionError("Replicate stratum mass not conserved")
    return counts_all, checks


def replicated_centered_variance(frame, scores, counts):
    """First-order conditional bootstrap of linearized PSU variance.

    For m>=2, V_bh=m^2/(m-1)^2 * [sum_i C_bhi*u_hi^2 -
    (sum_i C_bhi*u_hi)^2/m]. Its conditional expectation is the observed
    first-stage Taylor component m/(m-1)*sum_i(u_hi-u_bar_h)^2. Recentring
    inside the resample is essential; m2 strata generate 0 or2*observedV.
    Full singleton components are added without resampling or uncertainty.
    """
    scores=np.asarray(scores,float)
    counts=np.asarray(counts,int)
    output=np.zeros((len(counts), scores.shape[1]))
    roundoff_negative=0
    for _, row_idx in frame.groupby("stratum",sort=True).indices.items():
        ix=np.asarray(row_idx)
        m=len(ix)
        if m>1:
            first=counts[:,ix]@scores[ix]
            second=counts[:,ix]@(scores[ix]**2)
            centered=second-first**2/m
            # Algebra guarantees positivity. Only cancellation at zero can
            # yield tiny negative IEEE values; large negative values fail.
            tolerance=1e-13*np.maximum(second,1e-30)
            if np.any(centered < -tolerance):
                raise FloatingPointError("Substantively negative centered replicate PSU moment")
            roundoff_negative+=int(np.sum(centered<0))
            output+=m*m/(m-1)**2*np.maximum(centered,0)
        else:
            if not np.all(counts[:,ix]==1):
                raise ValueError("Full singleton uncertainty must remain fixed under the default protocol")
            output+=scores[ix][0]**2
    return output,roundoff_negative


def gvf_stabilize(y, kish, psu_eff, psu_count, df_proxy, variance, config=None):
    """Prespecified log-DEFF ridge GVF plus direct-variance shrinkage.

    d=PSUs minus represented strata is a stability proxy, not chi-square
    degrees of freedom. Original sample boundaries remain zero/one in the
    likelihood. A Jeffreys surrogate only makes their variance scale positive.
    """
    cfg = {**CONFIG, **(config or {})}
    y, kish, psu_eff, variance = map(lambda x: np.asarray(x, float), (y, kish, psu_eff, variance))
    df_proxy = np.asarray(df_proxy, float)
    observed = np.isfinite(y) & np.isfinite(kish) & (kish > 0)
    interior = observed & (y > 0) & (y < 1)
    pscale = np.where(interior, y, (kish*y+.5)/(kish+1))
    base_variance = pscale*(1-pscale)/kish
    pred_raw = np.column_stack((np.log(kish), np.log(psu_eff)))
    train = interior & (np.asarray(psu_count) >= 2) & np.isfinite(variance) & (variance > cfg["variance_numerical_floor"])
    if np.sum(train) < 5:
        raise ValueError("Fewer than five stable interior training domains for GVF")
    weights = np.maximum(df_proxy[train], 1.0)
    center = np.average(pred_raw[train], axis=0, weights=weights)
    scale = np.sqrt(np.average((pred_raw[train]-center)**2, axis=0, weights=weights))
    scale = np.where(scale > 1e-10, scale, 1.0)
    x = np.column_stack((np.ones(len(y)), (pred_raw-center)/scale))
    z = np.log(variance[train]/base_variance[train])
    xt = x[train]
    penalty = np.diag([0.0, cfg["ridge_slopes"], cfg["ridge_slopes"]])
    coef = np.linalg.solve(xt.T@(weights[:, None]*xt)+penalty, xt.T@(weights*z))
    fitted_deff = np.exp(x@coef)
    predicted_variance = base_variance*fitted_deff
    alpha = df_proxy/(df_proxy+cfg["shrinkage_df"])
    alpha[~interior] = 0.0
    stable_variance = alpha*variance+(1-alpha)*predicted_variance
    stable_variance = np.maximum(stable_variance, cfg["variance_numerical_floor"])
    n = pscale*(1-pscale)/stable_variance
    n[~observed] = np.nan
    return dict(effective_n=n, stabilized_variance=stable_variance,
                predicted_variance=predicted_variance, fitted_deff=fitted_deff,
                alpha=alpha, pscale=pscale, coefficients=coef, center=center, scale=scale,
                training_domains=int(train.sum()), training_mask=train,
                numerical_variance_floor_cells=int(np.sum(observed & (stable_variance <= cfg["variance_numerical_floor"]))))


def calibrate(frame, units, area_ids, outcome, config=None, counts=None):
    cfg = {**CONFIG, **(config or {})}
    direct = taylor_contributions(frame, units, outcome, area_ids)
    if counts is None:
        counts, replicate_checks = stratified_psu_counts(frame, cfg)
    else:
        replicate_checks = []
    central = gvf_stabilize(direct["y"], direct["n_kish"], direct["n_psu_eff"],
                            direct["n_psu"], direct["df_proxy"], direct["variance"], cfg)
    replicated_variance,roundoff_cells = replicated_centered_variance(frame,direct["scores"],counts)
    ne = np.full_like(replicated_variance, np.nan)
    coefficients = np.zeros((len(counts), 3))
    for b, vr in enumerate(replicated_variance):
        result = gvf_stabilize(direct["y"], direct["n_kish"], direct["n_psu_eff"],
                              direct["n_psu"], direct["df_proxy"], vr, cfg)
        ne[b] = result["effective_n"]
        coefficients[b] = result["coefficients"]
    selection_rng = np.random.default_rng(cfg["state_seed"])
    state_idx = selection_rng.permutation(len(counts))[:cfg["mixture_states"]]
    status = dict(config=cfg, full_psus=len(frame), full_strata=frame.stratum.nunique(),
                  full_singleton_strata=int(direct["full_singleton_mask"].sum()),
                  full_singleton_uncertainty_propagated=False,
                  training_domains=central["training_domains"],
                  mean_replication_variance_ratio=float(np.nanmean(np.mean(replicated_variance, axis=0)/np.where(direct["variance"] > 0, direct["variance"], np.nan))),
                  resampling_unit="full-frame first-stage PSU within survey stratum",
                  resampled_target="within-bootstrap recentered first-order linearized PSU moments",
                  bootstrap_type="conditional m-draw PSU empirical variance bootstrap with finite-moment expectation correction; not Rao-Wu ratio bootstrap",
                  bootstrap_roundoff_negative_cells=roundoff_cells,
                  unchanged_direct_point_estimates=True, lower_stage_variance_propagated=False,
                  nonresponse_uncertainty_propagated=False, area_df_is_stability_proxy=True,
                  max_absolute_full_score_sum=float(np.nanmax(abs(direct["full_score_sums"]))),
                  mixture_state_indices=state_idx.tolist(), replicate_checks=replicate_checks)
    return dict(direct=direct, central=central, replicated_variance=replicated_variance,
                replicate_effective_n=ne, mixture_effective_n=ne[state_idx],
                replicate_coefficients=coefficients, status=status)


def write_summary(result, output, outcome):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    d, c = result["direct"], result["central"]
    table = pd.DataFrame({"area": d["area_ids"], "direct": d["y"],
                          "n_kish": d["n_kish"], "n_psu": d["n_psu"],
                          "n_psu_eff": d["n_psu_eff"], "n_strata": d["n_strata"],
                          "df_proxy": d["df_proxy"], "Taylor_variance": d["variance"],
                          "singleton_Taylor_component": d["singleton_contribution"],
                          "stabilized_variance": c["stabilized_variance"],
                          "GVF_variance": c["predicted_variance"],
                          "central_effective_n": c["effective_n"],
                          "mixture_n_median": np.nanmedian(result["mixture_effective_n"], axis=0),
                          "mixture_n_min": np.nanmin(result["mixture_effective_n"], axis=0),
                          "mixture_n_max": np.nanmax(result["mixture_effective_n"], axis=0)})
    table.to_csv(output/f"{outcome}_calibration.csv", index=False)
    np.savez_compressed(output/f"{outcome}_calibration.npz", y=d["y"], area_ids=d["area_ids"],
                        central_effective_n=c["effective_n"],
                        mixture_effective_n=result["mixture_effective_n"],
                        replicate_effective_n=result["replicate_effective_n"],
                        replicate_variance=result["replicated_variance"],
                        coefficients=result["replicate_coefficients"])
    (output/f"{outcome}_status.json").write_text(json.dumps(result["status"], indent=2))
