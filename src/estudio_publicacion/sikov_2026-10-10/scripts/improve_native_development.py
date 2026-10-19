"""Native parametric-bootstrap reference uncertainty, new development selection.

ICC Laplace fits and posterior draws are reused verbatim from development v2.
Only the reference's uncertainty convention changes, following the original
article's explicit parametric-bootstrap statement. Fresh evaluation has not
been generated or scored. Native naive MSE is primary; raw BC MSE is retained
as a separately labelled numerical sensitivity without favourable truncation.
"""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import argparse
import concurrent.futures
import json
import shutil
import time
import traceback
import numpy as np
import pandas as pd
from comparison_simulation import ROOT, CONTEXTS, DEVELOPMENT_IDS, save, sha
from improve_candidates import COMPETITORS, CANDIDATES, mixture_prediction
from improve_select import select
from improve_development import scores, METHODS
from benchmark_metrics import summarize
from sikov_bootstrap import fit_sikov_bootstrap


def work(job):
    context, rep, destination, original, B = job
    source = ROOT / "desarrollo" / "inputs" / context / f"rep_{rep:05d}"
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["phase"] != "development" or rep not in DEVELOPMENT_IDS:
        raise ValueError("Only declared development inputs may enter native selection")
    d = dict(np.load(source / "model_inputs" / "inputs.npz", allow_pickle=False))
    names = d["covariate_names"].astype(str).tolist()
    old = Path(original) / "fit_results" / context / f"rep_{rep:05d}"
    oldseal = json.loads((old / "predictions_seal_before_development_scoring.json").read_text())
    if sha(source / "model_inputs" / "inputs.npz") != oldseal["common_inputs_sha256"]:
        raise ValueError("Reused ICC must correspond to exactly unchanged common inputs")
    before = time.perf_counter()
    predictions, statuses, fit_times, success, reused = {}, {}, {}, {}, {}
    out = Path(destination) / "fit_results" / context / f"rep_{rep:05d}"
    out.mkdir(parents=True)
    for method in ("ICC", "ICC_poverty"):
        if sha(old / method / "predictions.npz") != oldseal["prediction_sha256"][method]:
            raise ValueError("An original ICC development fit differs from its seal")
        predictions[method] = dict(np.load(old / method / "predictions.npz", allow_pickle=False))
        status = json.loads((old / method / "status.json").read_text())
        statuses[method], fit_times[method], success[method] = status["status"], status["fit_seconds"], status["success"]
        for filename in ("predictions.npz", "status.json", "conditional_beta_draws.npz"):
            dst = out / method / filename
            dst.parent.mkdir(exist_ok=True)
            shutil.copy2(old / method / filename, dst)
            reused[f"{method}/{filename}"] = {"source": str(old / method / filename), "sha256": sha(dst)}
    draws = dict(np.load(out / "ICC" / "conditional_beta_draws.npz", allow_pickle=False))
    base = {"p_mean": predictions["ICC"]["mean"], "p_variance": predictions["ICC"]["p_variance"],
            "p_draws": draws["p_draws"]}
    for index, (method, spatial) in enumerate((("Sikov_FH", False), ("Sikov_SAR", True))):
        tick = time.perf_counter()
        fitted = []
        for j, endpoint in enumerate(("anemia", "stunting")):
            seed = int(np.random.SeedSequence([20261010, 712, CONTEXTS.index(context), rep, index, j]).generate_state(1)[0])
            fitted.append(fit_sikov_bootstrap(d["X_raw"], d["y"][:, j], d["D"][:, j, j],
                                              d["sample_idx"], d["W"], d["strata"], spatial=spatial,
                                              outcome=endpoint, covariate_names=names,
                                              B=B, seed=seed, interval_estimator="naive"))
        predictions[method] = {key: np.column_stack([f[key] for f in fitted])
                               for key in ("mean", "lower", "upper", "mse", "mse_naive", "mse_bc",
                                           "naive_mcse", "bc_mcse", "mse_plugin", "mse_delta")}
        statuses[method] = fitted
        success[method] = [bool(f["converged"]) for f in fitted]
        fit_times[method] = time.perf_counter()-tick
    for method, recipe in CANDIDATES.items():
        if recipe["family"] != "ICC_SAR_mixture":
            continue
        tick = time.perf_counter()
        mixed = mixture_prediction(base, predictions["Sikov_SAR"], recipe["icc_weight"])
        predictions[method] = {key: value for key, value in mixed.items() if key != "status"}
        statuses[method] = mixed["status"]
        success[method] = [a and b for a, b in zip(success["ICC"], success["Sikov_SAR"])]
        fit_times[method] = time.perf_counter()-tick+fit_times["ICC"]+fit_times["Sikov_SAR"]
    sealed = {}
    for method in METHODS:
        m = out / method
        m.mkdir(exist_ok=True)
        if method not in ("ICC", "ICC_poverty"):
            np.savez_compressed(m / "predictions.npz", **predictions[method])
            save(m / "status.json", {"status": statuses[method], "success": success[method],
                                     "fit_seconds": fit_times[method], "phase": "development"})
        sealed[method] = sha(m / "predictions.npz")
    save(out / "predictions_seal_before_development_scoring.json",
         {"phase": "development", "prediction_sha256": sealed,
          "common_inputs_sha256": sha(source / "model_inputs" / "inputs.npz"),
          "fit_never_reads_truth": True, "ICC_reuse_verbatim": reused,
          "reference_uncertainty": "native parametric bootstrap naive MSE", "B": B})
    truth = dict(np.load(source / "evaluation_only" / "truth.npz", allow_pickle=False))
    records = []
    for method in METHODS:
        metric = scores(truth, d, predictions[method], method, context, rep, success[method], fit_times[method])
        for row in metric:
            row["protocol"] = "same14_native_naive_MSE_B100_Laplace_ICC_development"
        records.extend(metric)
    save(out / "completion.json", {"complete": True, "phase": "development",
                                   "all_method_success": success, "reference_B": B,
                                   "ICC_reuse_verbatim": True,
                                   "elapsed_seconds": time.perf_counter()-before})
    return records, {"context": context, "replication": rep,
                     "seconds": time.perf_counter()-before, "success": success}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tag", default="improve_development_v3_native")
    p.add_argument("--reuse-tag", default="improve_development_v2")
    p.add_argument("--B", type=int, default=100)
    args = p.parse_args()
    if not 1 <= args.workers <= 6 or args.B != 100:
        raise ValueError("Predeclared native development requires B100 and workers1..6")
    out = ROOT / "desarrollo" / args.tag
    original = ROOT / "desarrollo" / args.reuse_tag
    if out.exists():
        raise FileExistsError("Native-development runs cannot be overwritten")
    oldplan = json.loads((original / "development_plan.json").read_text())
    oldcompletion = json.loads((original / "completion.json").read_text())
    if not oldcompletion["complete"] or oldcompletion["completed_worlds"] != 32:
        raise ValueError("ICC reuse requires complete previous development cohort")
    for name in ("beta_power.py", "beta_power_stable.py", "improve_candidates.py"):
        paths = [Path(f) for f in oldplan["sources_sha256"] if Path(f).name == name]
        if len(paths) != 1 or sha(paths[0]) != oldplan["sources_sha256"][str(paths[0])]:
            raise ValueError("Reused ICC source/configuration must be unchanged: "+name)
    out.mkdir(parents=True)
    sources = [Path(__file__), Path(__file__).with_name("improve_candidates.py"),
               Path(__file__).with_name("improve_candidates_protocol.json"),
               Path(__file__).with_name("improve_select.py"),
               Path(__file__).with_name("comparison_simulation.py"),
               Path(__file__).with_name("sikov_model.py"), Path(__file__).with_name("sikov_bootstrap.py"),
               COMPETITORS / "beta_power.py", COMPETITORS / "beta_power_stable.py"]
    plan = {"phase": "development", "independent_final_outcomes_excluded": True,
            "candidate_methods": list(CANDIDATES), "scenarios": list(CONTEXTS),
            "development_replication_ids": DEVELOPMENT_IDS, "candidate_recipes": CANDIDATES,
            "reference_methods": ["Sikov_FH", "Sikov_SAR"],
            "reference_uncertainty_primary": "naive parametric bootstrap MSE", "reference_B": args.B,
            "reference_uncertainty_sensitivity": "raw BC MSE with MCSE retained; no clipping or tuning primary uncertainty",
            "ICC_reused_from": str(original), "ICC_prior_records_sha256": sha(original / "replications.csv"),
            "integration": "Laplace diagnostic; fresh same-posterior NUTS evaluation required",
            "source_amendment": "Original article explicitly states parametric-bootstrap MSE; deltaMSE development becomes sensitivity only",
            "amendment_selected_using_final_outcomes": False,
            "sources_sha256": {str(f): sha(f) for f in sources}}
    save(out / "development_plan.json", plan)
    jobs = [(c, r, str(out), str(original), args.B) for c in CONTEXTS for r in DEVELOPMENT_IDS]
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
                print(json.dumps({"completed": len(events), "total": len(jobs), "world": job[:2],
                                  "seconds": round(event["seconds"], 2),
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
    completion = {"phase": "development", "complete": len(events) == 32 and not failures,
                  "completed_worlds": len(events), "expected_worlds": 32,
                  "failures": failures, "elapsed_seconds": time.perf_counter()-tick}
    save(out / "completion.json", completion)
    if not completion["complete"]:
        raise RuntimeError("All32 native development worlds required; no partial selection")
    for path, expected in plan["sources_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError("Source changed during native development: "+path)
    result, ranking, held = select(frame[frame.method.isin(CANDIDATES)], plan)
    result.update(development_records_sha256=sha(out / "replications.csv"),
                  development_plan_sha256=sha(out / "development_plan.json"),
                  Laplace_development_not_NUTS_performance=True,
                  native_reference_MSE_B=100, native_reference_MSE="naive",
                  delta_development_selection_superseded=True)
    ranking.to_csv(out / "candidate_ranking.csv", index=False)
    held.to_csv(out / "LOPO_selection_rule_metrics.csv", index=False)
    save(out / "selected_recipe.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
