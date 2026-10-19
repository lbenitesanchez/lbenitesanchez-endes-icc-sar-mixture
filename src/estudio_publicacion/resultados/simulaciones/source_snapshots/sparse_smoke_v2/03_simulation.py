#!/usr/bin/env python3
"""Monte Carlo validation of latent district proportions; no ENDES records used.

Primary experiments use a declared Gaussian model on the logit scale. The
secondary experiment uses finite binary populations and informative Poisson
sampling with outcome-specific response. All covariance estimates are noisy,
and model variance components are estimated anew in each replication.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtr, ndtri

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "resultados" / "simulaciones"
SEED = 20261005
METHODS = ("FH", "MFH", "SFH", "MSFH")
THRESHOLDS = np.array([0.30, 0.15])
OUTCOME_NAMES = ("anemia", "stunting")
SCENARIOS = {
    "g_r0_s0": dict(kind="gaussian", rho=0.0, spatial=0.0, reps=500),
    "g_r6_s0": dict(kind="gaussian", rho=0.6, spatial=0.0, reps=500),
    "g_r0_s5": dict(kind="gaussian", rho=0.0, spatial=0.5, reps=500),
    "g_r6_s5": dict(kind="gaussian", rho=0.6, spatial=0.5, reps=500),
    "g_misspecified": dict(kind="gaussian", rho=0.6, spatial=0.5,
                            misspecified=True, reps=200),
    "binary_informative_mar": dict(kind="binary", rho=0.6, spatial=0.5,
                                    reps=200),
}


def simulation_models():
    """Load the frozen model source belonging to this experiment."""
    name = "endes_simulation_models"
    if name in sys.modules:
        return sys.modules[name]
    source = os.environ.get("ENDES_SIM_MODEL_PATH", str(ROOT / "scripts" / "models.py"))
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def geometry(m=80, q=12):
    rng = np.random.default_rng(SEED)
    side = math.ceil(math.sqrt(m))
    coords = np.stack(np.meshgrid(np.linspace(0, 1, side),
                                 np.linspace(0, 1, side)), -1).reshape(-1, 2)[:m]
    # External auxiliary information is fixed before every random sample.
    x1 = rng.normal(size=m)
    x2 = (coords[:, 1] - coords[:, 1].mean()) / coords[:, 1].std()
    X = np.column_stack([np.ones(m), x1, x2])
    distances = np.sum((coords[:, None] - coords[None, :]) ** 2, axis=2)
    K = np.exp(-distances / (2 * 0.24 ** 2))
    P = np.eye(m) - np.ones((m, m)) / m
    eigval, eigvec = np.linalg.eigh(P @ K @ P)
    order = np.argsort(eigval)[::-1][:q]
    basis = eigvec[:, order] * np.sqrt(np.maximum(eigval[order], 0))[None, :]
    basis /= np.sqrt(np.mean(np.sum(basis ** 2, axis=1)))
    return X, coords, basis


def latent_truth(rng, X, basis, cfg):
    beta = np.array([[logit(.28), logit(.14)], [.30, .22], [.18, .12]])
    sigma = np.array([.45, .40])
    R = np.array([[1., cfg["rho"]], [cfg["rho"], 1.]])
    chol = np.linalg.cholesky(R)
    independent = rng.normal(size=(len(X), 2)) @ chol.T
    if cfg.get("misspecified"):
        independent *= np.sqrt(2 / rng.chisquare(4, size=(len(X), 1)))
    independent *= sigma
    coeff = (rng.normal(size=(basis.shape[1], 2)) @ chol.T) * sigma
    spatial = basis @ coeff
    eta = X @ beta + np.sqrt(1 - cfg["spatial"]) * independent \
          + np.sqrt(cfg["spatial"]) * spatial
    if cfg.get("misspecified"):
        eta += np.column_stack([.18 * (X[:, 1] ** 2 - 1),
                                -.15 * (X[:, 1] ** 2 - 1)])
    return eta


def generate_gaussian(rep, scenario, X, coords, basis):
    k = list(SCENARIOS).index(scenario)
    seed = np.random.SeedSequence([SEED, k, rep])
    rng = np.random.default_rng(seed)
    cfg = SCENARIOS[scenario]
    eta = latent_truth(rng, X, basis, cfg)
    m = len(X)
    n_eff = rng.integers(5, 61, size=(m, 2)).astype(float)
    sampling_sd = np.column_stack([
        np.sqrt(8 * np.exp(.2 * X[:, 1]) / n_eff[:, 0]),
        np.sqrt(12 * np.exp(-.2 * X[:, 1]) / n_eff[:, 1])])
    sampling_rho = .25
    R = np.array([[1., sampling_rho], [sampling_rho, 1.]])
    true_D = sampling_sd[:, :, None] * R * sampling_sd[:, None, :]
    obs = eta + np.einsum("ijk,ik->ij", np.linalg.cholesky(true_D),
                          rng.normal(size=(m, 2)))
    idx = np.sort(rng.choice(m, int(.8 * m), replace=False))
    # An independent Wishart covariance estimate models finite design degrees of
    # freedom. df is a simulation parameter, not a literal ENDES PSU count.
    dfs = rng.integers(1, 9, size=len(idx))
    D = np.empty((len(idx), 2, 2))
    for j, (i, df) in enumerate(zip(idx, dfs)):
        E = rng.normal(size=(df, 2)) @ np.linalg.cholesky(true_D[i]).T
        D[j] = E.T @ E / df
    return dict(X=X, coords=coords, basis=basis, eta=eta, theta=expit(eta),
                obs=obs[idx], y=obs[idx], D=D, true_D=true_D[idx],
                n_eff=n_eff[idx], sample_idx=idx, df=dfs,
                transform="identity", seed=list(seed.entropy),
                metadata=dict(expected_sample_n=float(n_eff.mean()),
                              boundary_rate=0.0, outcome_missing_rate=0.0,
                              psd_projection_count=0, unmeasured_area_count=0,
                              covariance_df_min=int(dfs.min()), covariance_df_max=int(dfs.max())))


def generate_binary(rep, scenario, X, coords, basis):
    k = list(SCENARIOS).index(scenario)
    seed = np.random.SeedSequence([SEED, k, rep])
    rng = np.random.default_rng(seed)
    cfg = SCENARIOS[scenario]
    eta = latent_truth(rng, X, basis, cfg)
    m, population_n = len(X), 500
    H = rng.binomial(1, .45, (m, population_n))
    person_p = expit(eta[:, None, :] + (H[:, :, None] - .45) *
                    np.array([.50, .35])[None, None, :])
    latent = rng.normal(size=(m, population_n, 2)) @ \
             np.linalg.cholesky(np.array([[1., .40], [.40, 1.]])).T
    outcomes = (ndtr(latent) < person_p).astype(float)
    theta = outcomes.mean(axis=1)
    # Population truth is the realized finite-population proportion, not p above.
    target_n = rng.integers(5, 61, m)
    size = np.where(H == 1, 1.8, .55)
    inclusion = np.minimum(target_n[:, None] * size / size.sum(axis=1)[:, None], .8)
    resp = np.stack([np.where(H == 1, .82, .95),
                     np.where(H == 1, .88, .97)], -1)
    sampled_domains = np.sort(rng.choice(m, int(.8 * m), replace=False))
    yrows, drows, nrows, idx = [], [], [], []
    total_sampled = total_measured = 0
    projection_count = unmeasured_count = 0
    for i in sampled_domains:
        take = rng.uniform(size=population_n) < inclusion[i]
        observe = take[:, None] & (rng.uniform(size=(population_n, 2)) < resp[i])
        if np.any(observe.sum(axis=0) == 0):
            unmeasured_count += 1
            continue
        q = inclusion[i, :, None] * resp[i]
        w = np.where(observe, 1 / q, 0.)
        denominator = w.sum(axis=0)
        p = (w * outcomes[i]).sum(axis=0) / denominator
        centered = outcomes[i] - p
        d = np.zeros((2, 2))
        for t in range(2):
            d[t, t] = np.sum(w[:, t] ** 2 * (1 - q[:, t]) *
                             centered[:, t] ** 2) / denominator[t] ** 2
        both = observe.all(axis=1)
        d[0, 1] = d[1, 0] = np.sum(
            both * (1 - inclusion[i]) * w[:, 0] * w[:, 1] *
            centered[:, 0] * centered[:, 1]) / np.prod(denominator)
        # Different marginal response patterns can make a finite-sample
        # linearized estimate slightly indefinite; nearest-PSD projection is
        # logged and used identically for every fitted model.
        ev, U = np.linalg.eigh(d)
        projection_count += int(ev.min() < -1e-12)
        d = (U * np.maximum(ev, 0)) @ U.T
        yrows.append(p); drows.append(d)
        nrows.append(denominator ** 2 / (w ** 2).sum(axis=0))
        idx.append(i)
        total_sampled += int(take.sum()) * 2
        total_measured += int(observe.sum())
    y, D, n_eff, idx = map(np.asarray, (yrows, drows, nrows, idx))
    eta_true = logit(np.clip(theta, 1e-8, 1-1e-8))
    return dict(X=X, coords=coords, basis=basis, eta=eta_true, theta=theta,
                y=y, obs=y, D=D, true_D=None, n_eff=n_eff, sample_idx=idx,
                transform="logit", seed=list(seed.entropy),
                metadata=dict(expected_sample_n=float(target_n.mean()),
                              boundary_rate=float(np.mean((y == 0) | (y == 1))),
                              outcome_missing_rate=1-total_measured/max(total_sampled, 1),
                              psd_projection_count=projection_count,
                              unmeasured_area_count=unmeasured_count))


def probability_both(mean, cov, thresholds=THRESHOLDS):
    """Deterministic one-dimensional quadrature for a bivariate Normal tail."""
    sd = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 1e-14))
    a = (logit(thresholds)[None, :] - mean) / sd
    rho = np.clip(cov[:, 0, 1] / np.prod(sd, axis=1), -.999999, .999999)
    nodes, weights = np.polynomial.legendre.leggauss(32)
    lower = ndtr(a[:, 0])
    u = lower[:, None] + (nodes[None, :] + 1) * (1-lower[:, None]) / 2
    x = ndtri(np.clip(u, 1e-15, 1-1e-15))
    conditional_tail = ndtr((rho[:, None] * x - a[:, 1, None]) /
                             np.sqrt(1-rho[:, None] ** 2))
    return (1-lower) / 2 * (conditional_tail @ weights)


def prediction_values(pred):
    mean = np.asarray(pred["eta_mean"])
    cov = np.asarray(pred["eta_cov"])
    sd = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0))
    nodes, weights = np.polynomial.hermite.hermgauss(12)
    p = (expit(mean[:, :, None] + np.sqrt(2) * sd[:, :, None] * nodes) @
         weights) / np.sqrt(np.pi)
    lo, hi = expit(mean-1.959963984540054*sd), expit(mean+1.959963984540054*sd)
    return p, lo, hi, probability_both(mean, cov)


def replication_metrics(data, pred, method, D_mode, scenario, rep, fit_seconds,
                        converged, interval="conditional", custom_limits=None):
    p, lo, hi, prob = prediction_values(pred)
    if custom_limits is not None:
        lo, hi = custom_limits
    theta = data["theta"]
    sampled = np.zeros(len(theta), dtype=bool)
    sampled[data["sample_idx"]] = True
    truth_priority = (theta > THRESHOLDS).all(axis=1)
    rows = []
    for group, mask in (("all", np.ones(len(theta), bool)),
                        ("sampled", sampled), ("unsampled", ~sampled)):
        if not mask.any():
            continue
        for t, outcome in enumerate(OUTCOME_NAMES):
            error = p[mask, t] - theta[mask, t]
            rows.append(dict(scenario=scenario, replication=rep, method=method,
                             D_mode=D_mode, interval=interval,
                             group=group, outcome=outcome, bias=error.mean(),
                             mse=np.mean(error ** 2), mae=np.mean(np.abs(error)),
                             coverage=np.mean((lo[mask, t] <= theta[mask, t]) &
                                              (theta[mask, t] <= hi[mask, t])),
                             width=np.mean(hi[mask, t]-lo[mask, t]),
                             priority_brier=np.mean((prob[mask]-truth_priority[mask])**2),
                             priority_accuracy=np.mean((prob[mask]>=.5)==truth_priority[mask]),
                             fit_seconds=fit_seconds, converged=bool(converged),
                             areas=int(mask.sum())))
    return rows


def bootstrap_prediction_errors(fit, data, B, seed, resample_D, maxiter):
    """Fit-based parametric prediction-error bootstrap, retaining nonconvergence.

    The generated D estimate is Wishart around the fitted covariance, and the
    same smoothing estimator is refitted when D is resampled. df=1 rank-one
    estimates are generated directly, rather than replaced with df=2.
    This is a model sensitivity experiment, not replication of an ENDES design.
    """
    models = simulation_models()
    rng = np.random.default_rng(seed)
    idx = fit.sample_idx
    dfs = np.asarray(data["df"], int)
    chol = np.linalg.cholesky(fit.D)
    errors, statuses = [], []
    for b in range(B):
        true = fit.sample_latent(rng)
        noise = np.einsum("ijk,ik->ij", chol, rng.normal(size=(len(idx), 2)))
        zstar = true[idx] + noise
        Dstar = fit.D.copy()
        if resample_D:
            for i, df in enumerate(dfs):
                E = rng.normal(size=(df, 2)) @ chol[i].T
                Dstar[i] = E.T @ E / df
        try:
            fstar = models.fit_model(fit.Xall, zstar, Dstar, data["n_eff"], idx,
                                     fit.basis, fit.method, transform="identity",
                                     D_mode=fit.preparation["D_mode"] if resample_D else "raw",
                                     df=dfs, maxiter=maxiter, start=fit.params)
            epred = fstar.predict()["eta_mean"]
            if not np.isfinite(epred).all():
                raise ValueError("Nonfinite bootstrap prediction")
            errors.append(epred-true)
            statuses.append(dict(b=b, finite=True, **fstar.status))
        except Exception as exc:
            statuses.append(dict(b=b, finite=False, success=False, error=repr(exc)))
    if len(errors) < max(20, .90*B):
        raise RuntimeError(f"Only {len(errors)}/{B} finite bootstrap predictions")
    errors = np.stack(errors)
    pred = fit.predict()
    lo = pred["eta_mean"] - np.quantile(errors, .975, axis=0)
    hi = pred["eta_mean"] - np.quantile(errors, .025, axis=0)
    return expit(lo), expit(hi), dict(
        B_requested=B, B_finite=len(errors),
        B_converged=sum(s["success"] for s in statuses), statuses=statuses,
        tail_probability_mcse=float(np.sqrt(.025*.975/len(errors))),
        D_resampling="Wishart around fitted D, original df, same GVF refitted" if resample_D
                     else "Fitted D fixed; REML components refitted",
        interval_type="Basic parametric prediction-error interval; no data-dependent calibration")


def run_replication(job):
    scenario, rep, modes, maxiter, save_inputs, bootstrap_B = job
    X, coords, basis = geometry()
    maker = generate_binary if SCENARIOS[scenario]["kind"] == "binary" else generate_gaussian
    data = maker(rep, scenario, X, coords, basis)
    if save_inputs:
        folder = OUT / "inputs" / scenario
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f"rep_{rep:04d}.npz",
                            theta=data["theta"], y=data["y"], D=data["D"],
                            sample_idx=data["sample_idx"], n_eff=data["n_eff"],
                            seed=np.asarray(data["seed"]))
    rows, failures = [], []
    models = simulation_models()
    for mode in modes:
        for method in METHODS:
            tick = time.perf_counter()
            try:
                fit = models.fit_model(data["X"], data["y"], data["D"], data["n_eff"],
                                       data["sample_idx"], basis=data["basis"],
                                       method=method, D_mode=mode,
                                       transform=data["transform"], df=data.get("df"), maxiter=maxiter)
                pred = fit.predict()
                if not all(np.isfinite(pred[key]).all() for key in ("eta_mean", "eta_cov")):
                    raise ValueError("Nonfinite posterior moments")
                converged = fit.status["success"]
                rows.extend(replication_metrics(data, pred, method, mode, scenario, rep,
                                                time.perf_counter()-tick, converged))
                if not converged:
                    failures.append(dict(scenario=scenario, replication=rep, method=method,
                                         D_mode=mode, status="optimization_not_converged",
                                         message=str(fit.status["message"])))
                if bootstrap_B and method == "MSFH" and SCENARIOS[scenario]["kind"] == "gaussian":
                    for resample_D in (False, True):
                        btick = time.perf_counter()
                        try:
                            lo, hi, binfo = bootstrap_prediction_errors(
                                fit, data, bootstrap_B,
                                np.random.SeedSequence([SEED, 7000, list(SCENARIOS).index(scenario), rep,
                                                        int(resample_D)]), resample_D, maxiter)
                            blabel = "bootstrap_D" if resample_D else "bootstrap_fixed_D"
                            rows.extend(replication_metrics(data, pred, method, mode, scenario, rep,
                                                            time.perf_counter()-btick, converged,
                                                            interval=blabel, custom_limits=(lo, hi)))
                            statusdir = OUT / "bootstrap_status" / os.environ.get("ENDES_SIM_EXPERIMENT_TAG", "untagged") / scenario
                            statusdir.mkdir(parents=True, exist_ok=True)
                            (statusdir / f"rep_{rep:04d}_{mode}_{blabel}.json").write_text(
                                json.dumps(binfo, indent=2))
                        except Exception as exc:
                            failures.append(dict(scenario=scenario, replication=rep, method=method,
                                                 D_mode=mode, status="bootstrap_exception",
                                                 resample_D=resample_D, message=repr(exc)))
            except Exception as exc:
                failures.append(dict(scenario=scenario, replication=rep, method=method,
                                     D_mode=mode, status="exception", message=repr(exc),
                                     traceback=traceback.format_exc()))
    metadata = dict(scenario=scenario, replication=rep, seed=data["seed"],
                    sampled_areas=len(data["sample_idx"]), **data["metadata"])
    return rows, failures, metadata


def summarize(df):
    keys = ["scenario", "method", "D_mode", "interval", "group", "outcome"]
    summaries = []
    for key, entire_block in df.groupby(keys, sort=True):
        for sensitivity in ("all_valid_predictions", "converged_only"):
            block = entire_block if sensitivity=="all_valid_predictions" else entire_block[entire_block.converged]
            R = len(block)
            if R == 0:
                continue
            result = dict(zip(keys, key), replications=R, sensitivity=sensitivity)
            for metric in ("bias", "mae", "coverage", "width", "priority_brier", "priority_accuracy"):
                result[metric] = float(block[metric].mean())
                result[metric+"_mcse"] = float(block[metric].std(ddof=1)/np.sqrt(R)) if R>1 else None
            rmse = float(np.sqrt(block.mse.mean()))
            result["rmse"] = rmse
            result["rmse_mcse"] = float(block.mse.std(ddof=1)/(2*rmse*np.sqrt(R))) if R>1 else None
            result["optimizer_convergence_rate"] = float(entire_block.converged.mean())
            result["mean_fit_seconds"] = float(block.fit_seconds.mean())
            summaries.append(result)
    return pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--reps", type=int, help="Override default replication counts")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--modes", nargs="+", default=["raw", "smooth"])
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--save-inputs", action="store_true")
    parser.add_argument("--bootstrap-B", type=int, default=0,
                        help="Additional MSFH fixed-D and estimated-D bootstrap intervals")
    parser.add_argument("--tag", default="full")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot_dir = OUT / "source_snapshots" / args.tag
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    model_snapshot = snapshot_dir / "models.py"
    if model_snapshot.exists():
        raise FileExistsError(f"Experiment source snapshot already exists: {model_snapshot}")
    shutil.copy2(ROOT / "scripts" / "models.py", model_snapshot)
    shutil.copy2(Path(__file__), snapshot_dir / Path(__file__).name)
    os.environ["ENDES_SIM_MODEL_PATH"] = str(model_snapshot)
    os.environ["ENDES_SIM_EXPERIMENT_TAG"] = args.tag
    config = dict(seed=SEED, m=80, q=12, scenario_definitions=SCENARIOS,
                  methods=METHODS, analytic_priority_thresholds=THRESHOLDS.tolist(),
                  requested_args=vars(args), interval_level=.95,
                  models_source_sha256=hashlib.sha256(model_snapshot.read_bytes()).hexdigest(),
                  simulation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  covariance_estimation="Independent Wishart(df=1..8) in Gaussian experiments; Poisson linearization in binary experiment",
                  truth="Known latent proportions; realized population proportions for binary regime",
                  conditional_interval="Approximate empirical Bayes universal-kriging interval on logit scale with plug-in REML variance components; not claimed calibrated")
    (OUT / f"config_{args.tag}.json").write_text(json.dumps(config, indent=2))
    jobs = [(scenario, rep, args.modes, args.maxiter, args.save_inputs, args.bootstrap_B)
            for scenario in args.scenarios
            for rep in range(args.start, args.start+(args.reps or SCENARIOS[scenario]["reps"]))]
    rowpath = OUT / f"replications_{args.tag}.jsonl"
    failpath = OUT / f"failures_{args.tag}.jsonl"
    metapath = OUT / f"metadata_{args.tag}.jsonl"
    # Refuse to silently append overlapping experiments.
    if rowpath.exists():
        raise FileExistsError(f"Choose a new tag or remove an explicitly obsolete experiment: {rowpath}")
    start = time.perf_counter()
    with rowpath.open("w") as rf, failpath.open("w") as ff, metapath.open("w") as mf:
        if args.workers == 1:
            iterator = map(run_replication, jobs)
            executor = None
        else:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
            iterator = executor.map(run_replication, jobs, chunksize=1)
        try:
            for done, (rows, failures, metadata) in enumerate(iterator, start=1):
                for row in rows:
                    rf.write(json.dumps(row)+"\n")
                for failure in failures:
                    ff.write(json.dumps(failure)+"\n")
                mf.write(json.dumps(metadata)+"\n")
                rf.flush(); ff.flush(); mf.flush()
                if done % 10 == 0 or done == len(jobs):
                    print(json.dumps(dict(completed=done, total=len(jobs),
                                          seconds=round(time.perf_counter()-start, 1),
                                          recent_failures=len(failures))), flush=True)
        finally:
            if executor:
                executor.shutdown(wait=True)
    df = pd.read_json(rowpath, lines=True)
    df.to_csv(OUT / f"replications_{args.tag}.csv", index=False)
    summary = summarize(df)
    summary.to_csv(OUT / f"summary_{args.tag}.csv", index=False)
    (OUT / f"completion_{args.tag}.json").write_text(json.dumps(
        dict(replications=len(jobs), metric_rows=len(df), elapsed_seconds=time.perf_counter()-start,
             failures=sum(1 for _ in failpath.open()), completed=True), indent=2))
    print(summary[(summary.group=="all")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
