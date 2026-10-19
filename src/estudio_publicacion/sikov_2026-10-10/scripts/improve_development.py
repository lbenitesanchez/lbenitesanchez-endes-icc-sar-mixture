"""Fit and select finite candidates on the 32 NEW development worlds only.

This runner cannot generate or read evaluation streams. Laplace integrates the
same ICC posterior approximately during development; frozen fresh evaluation
must use the established NUTS integration and its diagnostic gates.
"""
from __future__ import annotations
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
import traceback
import numpy as np
import pandas as pd

from comparison_simulation import ROOT, CONTEXTS, DEVELOPMENT_IDS, generate, save, sha
from sikov_model import fit_sikov
from improve_candidates import COMPETITORS, CANDIDATES, fit_icc_poverty, mixture_prediction
from improve_select import select
if str(COMPETITORS) not in sys.path:
    sys.path.insert(0, str(COMPETITORS))
from beta_power import fit as fit_icc
from benchmark_metrics import interval_score, summarize

METHODS = ("Sikov_FH", "Sikov_SAR", *CANDIDATES)


def scores(truth, direct, predictions, method, context, rep, success, seconds):
    p, lo, hi = (np.asarray(predictions[k], float) for k in ("mean", "lower", "upper"))
    theta = truth["theta"]
    if not (p.shape == lo.shape == hi.shape == theta.shape):
        raise ValueError("Every method must predict the identical complete finite universe")
    if not np.isfinite(np.stack((p, lo, hi))).all() or np.any(lo > hi):
        raise ValueError("Invalid complete predictions; no silent deletion or fallback")
    observed = np.zeros(theta.shape, bool)
    observed[direct["sample_idx"]] = np.isfinite(direct["y"])
    rows = []
    for j, endpoint in enumerate(("anemia", "stunting")):
        for group, mask in {"all": np.ones(len(theta), bool),
                            "observed_endpoint": observed[:, j],
                            "unsampled_common": ~np.isin(np.arange(len(theta)), direct["sample_idx"])}.items():
            err = p[mask, j]-theta[mask, j]
            rows.append({"phase": "development", "scenario": context, "replication": rep,
                         "method": method, "protocol": "same14_covariates_raw_Sikov_Laplace_ICC_development",
                         "group": group, "outcome": endpoint,
                         "mse": float(np.mean(err**2)), "bias": float(np.mean(err)),
                         "mae": float(np.mean(abs(err))),
                         "interval_score": float(interval_score(theta[mask, j], lo[mask, j], hi[mask, j]).mean()),
                         "coverage": float(np.mean((lo[mask, j] <= theta[mask, j]) & (theta[mask, j] <= hi[mask, j]))),
                         "width": float(np.mean(hi[mask, j]-lo[mask, j])),
                         "point_outside_0_1": float(np.mean((p[mask, j] < 0) | (p[mask, j] > 1))),
                         "interval_outside_0_1": float(np.mean((lo[mask, j] < 0) | (hi[mask, j] > 1))),
                         "areas": int(mask.sum()), "areas_requested": int(mask.sum()), "availability_rate": 1.,
                         "converged": bool(success[j]), "fit_seconds": seconds})
    return rows


def work(job):
    context, rep, destination = job
    source = ROOT / "desarrollo" / "inputs" / context / f"rep_{rep:05d}"
    if not source.exists():
        generate(context, rep, "development")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["phase"] != "development" or manifest["replication"] not in DEVELOPMENT_IDS:
        raise ValueError("Development role or replication stream mismatch")
    d = dict(np.load(source / "model_inputs" / "inputs.npz", allow_pickle=False))
    units = pd.read_csv(source / "model_inputs" / "sample.csv")
    names = d["covariate_names"].astype(str).tolist()
    tick = time.perf_counter()
    predictions, statuses, fit_times, success = {}, {}, {}, {}
    for method, spatial in (("Sikov_FH", False), ("Sikov_SAR", True)):
        before = time.perf_counter()
        fitted = [fit_sikov(d["X_raw"], d["y"][:, j], d["D"][:, j, j],
                           d["sample_idx"], d["W"], d["strata"], spatial=spatial,
                           outcome=name, covariate_names=names, uncertainty="delta")
                  for j, name in enumerate(("anemia", "stunting"))]
        predictions[method] = {key: np.column_stack([f[key] for f in fitted])
                               for key in ("mean", "lower", "upper", "mse")}
        statuses[method] = fitted
        success[method] = [f["converged"] for f in fitted]
        fit_times[method] = time.perf_counter()-before
    seed = int(np.random.SeedSequence([20261010, 711, CONTEXTS.index(context), rep]).generate_state(1)[0])
    before = time.perf_counter()
    base = fit_icc(d["X"], d["y"], d["n_eff"], d["sample_idx"], units,
                   d["basis"], seed=seed)
    fit_times["ICC"] = time.perf_counter()-before
    before = time.perf_counter()
    poverty = fit_icc_poverty(d["X"], d["y"], d["n_eff"], d["sample_idx"], units,
                              d["basis"], strata=d["strata"], seed=seed,
                              integration="laplace_development")
    fit_times["ICC_poverty"] = time.perf_counter()-before
    for method, fit in (("ICC", base), ("ICC_poverty", poverty)):
        predictions[method] = {key: fit[val] for key, val in
                               (("mean", "p_mean"), ("lower", "lower95"),
                                ("upper", "upper95"), ("p_variance", "p_variance"))}
        statuses[method] = fit["status"]
        success[method] = [s["success"] for s in fit["status"]]
    for method, recipe in CANDIDATES.items():
        if recipe["family"] != "ICC_SAR_mixture":
            continue
        before = time.perf_counter()
        mixed = mixture_prediction(base, predictions["Sikov_SAR"], recipe["icc_weight"])
        predictions[method] = {k: v for k, v in mixed.items() if k != "status"}
        statuses[method] = mixed["status"]
        success[method] = [a and b for a, b in zip(success["ICC"], success["Sikov_SAR"])]
        fit_times[method] = time.perf_counter()-before+fit_times["ICC"]+fit_times["Sikov_SAR"]
    out = Path(destination) / "fit_results" / context / f"rep_{rep:05d}"
    out.mkdir(parents=True)
    sealed = {}
    for method in METHODS:
        m = out / method
        m.mkdir()
        np.savez_compressed(m / "predictions.npz", **predictions[method])
        save(m / "status.json", {"status": statuses[method], "success": success[method],
                                "fit_seconds": fit_times[method], "phase": "development"})
        sealed[method] = sha(m / "predictions.npz")
    # Save actual draws for distribution-mixture reproducibility, not in metrics.
    np.savez_compressed(out / "ICC" / "conditional_beta_draws.npz", p_draws=base["p_draws"])
    np.savez_compressed(out / "ICC_poverty" / "conditional_beta_draws.npz", p_draws=poverty["p_draws"])
    save(out / "predictions_seal_before_development_scoring.json",
         {"phase": "development", "prediction_sha256": sealed,
          "common_inputs_sha256": sha(source / "model_inputs" / "inputs.npz"),
          "fit_never_reads_truth": True})
    truth = dict(np.load(source / "evaluation_only" / "truth.npz", allow_pickle=False))
    rows = []
    for method in METHODS:
        rows.extend(scores(truth, d, predictions[method], method, context, rep,
                           success[method], fit_times[method]))
    save(out / "completion.json", {"phase": "development", "complete": True,
                                   "elapsed_seconds": time.perf_counter()-tick,
                                   "all_method_success": success,
                                   "all_finite_nonconvergence_retained": True})
    return rows, {"context": context, "replication": rep,
                  "seconds": time.perf_counter()-tick, "success": success}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--tag", default="improve_development_v1")
    args = p.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Workers must be 1..4 under the agreed runtime budget")
    out = ROOT / "desarrollo" / args.tag
    if out.exists():
        raise FileExistsError("A development run cannot be overwritten")
    out.mkdir(parents=True)
    sources = [Path(__file__), Path(__file__).with_name("improve_candidates.py"),
               Path(__file__).with_name("improve_candidates_protocol.json"),
               Path(__file__).with_name("improve_select.py"),
               Path(__file__).with_name("comparison_simulation.py"),
               Path(__file__).with_name("sikov_model.py"),
               COMPETITORS / "beta_power.py", COMPETITORS / "beta_power_stable.py"]
    plan = {"phase": "development", "independent_final_outcomes_excluded": True,
            "candidate_methods": list(CANDIDATES), "scenarios": list(CONTEXTS),
            "development_replication_ids": DEVELOPMENT_IDS, "candidate_recipes": CANDIDATES,
            "reference_methods": ["Sikov_FH", "Sikov_SAR"], "integration": "Laplace diagnostic of ICC same posterior; fresh NUTS evaluation required",
            "same_information": "Common raw14 covariates, fixed poverty30/55 labels, observed direct endpoints, design, weights/PSUs and public geography",
            "sources_sha256": {str(f): sha(f) for f in sources}}
    save(out / "development_plan.json", plan)
    jobs = [(c, r, str(out)) for c in CONTEXTS for r in DEVELOPMENT_IDS]
    rows, events, failures = [], [], []
    tick = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                records, event = future.result()
                rows.extend(records); events.append(event)
                with (out / "replications.jsonl").open("a") as stream:
                    stream.writelines(json.dumps(row)+"\n" for row in records)
                with (out / "world_completion.jsonl").open("a") as stream:
                    stream.write(json.dumps(event)+"\n")
                print(json.dumps({"completed": len(events), "total": len(jobs),
                                  "world": job[:2], "seconds": round(event["seconds"], 2),
                                  "all_models_converged": all(all(v) for v in event["success"].values())}), flush=True)
            except Exception as exc:
                failure = {"job": job, "exception": repr(exc), "traceback": traceback.format_exc()}
                failures.append(failure)
                with (out / "failures.jsonl").open("a") as stream:
                    stream.write(json.dumps(failure)+"\n")
                print(json.dumps({"failed": job[:2], "exception": repr(exc)}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "replications.csv", index=False)
    if len(frame):
        summarize(frame).to_csv(out / "summary.csv", index=False)
    completion = {"phase": "development", "complete": len(events) == len(jobs) and not failures,
                  "completed_worlds": len(events), "expected_worlds": len(jobs),
                  "failures": failures, "elapsed_seconds": time.perf_counter()-tick}
    save(out / "completion.json", completion)
    if not completion["complete"]:
        raise RuntimeError("All 32 development worlds are required; selection was not performed")
    for f, digest in plan["sources_sha256"].items():
        if sha(f) != digest:
            raise RuntimeError(f"A source changed during development; no silent selection: {f}")
    result, ranking, held = select(frame[frame.method.isin(CANDIDATES)], plan)
    ranking.to_csv(out / "candidate_ranking.csv", index=False)
    held.to_csv(out / "LOPO_selection_rule_metrics.csv", index=False)
    result["development_records_sha256"] = sha(out / "replications.csv")
    result["development_plan_sha256"] = sha(out / "development_plan.json")
    result["Laplace_development_not_NUTS_performance"] = True
    save(out / "selected_recipe.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
