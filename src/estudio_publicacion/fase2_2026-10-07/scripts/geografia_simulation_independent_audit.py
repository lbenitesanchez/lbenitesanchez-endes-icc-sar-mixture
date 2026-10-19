"""Independent arithmetic audit after all 40 frozen simulation worlds complete.

No fit/generation/calibration imports; the audit uses the final prediction CSVs,
finite-population evaluation arrays, and Python scalar arithmetic/statistics.
Failed but finite predictions remain in every planned population's metrics.
"""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

from pathlib import Path
from collections import defaultdict
import csv
import datetime as dt
import hashlib
import json
import math
import statistics as stats
import numpy as np

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE / "resultados/design_simulation"
EXPECTED = {(context, rep) for context in ("dense6", "sparse2") for rep in range(20)}
TASKS = ["baseline", "point"] + [f"cut_state_{k:02d}" for k in range(8)]
METRICS = ["mse", "bias", "coverage95", "mean_width", "interval_score95"]
TOLERANCE = 1e-12


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def mcse(values):
    return stats.stdev(values) / math.sqrt(len(values))


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    # Before this gate, no predictions or population truth are opened.
    completions = {}
    for context, rep in sorted(EXPECTED):
        path = ROOT / context / f"rep_{rep:03d}" / "completion.json"
        if not path.is_file():
            raise RuntimeError(f"All40 completion gate not met: missing {context}/{rep}")
        value = read(path)
        check(value.get("complete") is True, f"Incomplete population {context}/{rep}")
        check((value["context"], value["rep"]) == (context, rep), "World identity changed")
        completions[(context, rep)] = value
    final_path = ROOT / "final/completion.json"
    if not final_path.is_file():
        raise RuntimeError("All40 worlds are complete; waiting for root's final collector before truth access/comparison")
    final = read(final_path)
    check(final.get("all40_completed") is True and final["unique_populations"] == 40, "Final collector incomplete")
    final_worlds = {(x["context"], x["rep"]) for x in final["worlds"]}
    check(final_worlds == EXPECTED and len(final["worlds"]) == 40, "Collector cohort is not the frozen40")
    protocol_path = PHASE / "protocol_freeze.json"
    protocol = read(protocol_path)
    protocol_sha = sha(protocol_path)
    source_sha = sha(PHASE / "scripts/sim_design_diagnostics.py")
    check(source_sha == protocol["source_sha256"]["scripts/sim_design_diagnostics.py"], "Frozen simulation source changed")
    check(final["source_sha256"] == source_sha, "Collector source identity changed")
    comparison_failures = []
    population_metrics, worlds, failed_tasks = [], [], []
    prediction_rows = 0
    for context, rep in sorted(EXPECTED):
        out = ROOT / context / f"rep_{rep:03d}"
        completion = completions[(context, rep)]
        check(completion["protocol_sha256"] == protocol_sha, "World protocol mismatch")
        check(completion["source_sha256"] == source_sha, "World source mismatch")
        check(completion["known_truth_opened"] is False, "Fitting declared truth exposure")
        prediction_path = out / "predictions.csv"
        check(sha(prediction_path) == completion["predictions_sha256"], "Saved predictions changed")
        gen_path = PHASE / f"datos/synthetic_design/{context}/rep_{rep:03d}/generation.json"
        generation = read(gen_path)
        check(generation["finite_population_truth_read_by_fitting"] is False, "Generated truth not isolated")
        check(generation["population_stream"] == [20261007, 702, 6 if context == "dense6" else 2, rep], "Population seed stream changed")
        check(generation["sampling_stream"] == [20261007, 703, 6 if context == "dense6" else 2, rep], "Sampling seed stream changed")
        truth_path = PHASE / f"datos/synthetic_design_evaluation/{context}/rep_{rep:03d}/realized_population_truth.npz"
        check(sha(truth_path) == generation["realized_truth_sha256"], "Finite population truth changed")
        with np.load(truth_path, allow_pickle=False) as data:
            theta = data["theta"].copy()
            population_size = data["population_size"].copy()
        check(theta.shape == (80, 2), "Truth shape mismatch")
        check(np.isfinite(theta).all() and (theta >= 0).all() and (theta <= 1).all(), "Truth outside probability bounds")
        check(population_size.shape == (80,) and (population_size > 0).all(), "Invalid finite population sizes")
        conditional = completion["conditional_tasks"]
        check(len(conditional) == 10, "Conditional task count changed")
        successes = []
        for task, status in zip(TASKS, conditional):
            task_path = out / task
            task_completion = read(task_path / "completion.json")
            check(sha(task_path / "fit_status.json") == task_completion["sampler_status_sha256"], "Conditional diagnostic proof changed")
            check(read(task_path / "fit_status.json") == status, "World diagnostics do not equal retained conditional diagnostics")
            check(task_completion["success"] == status["success"], "Conditional completion status inconsistent")
            check(status["label"] == f"{context}_{rep:03d}_{task}", "Conditional task label reordered")
            successes.append(bool(status["success"]))
            if not status["success"]:
                failed_tasks.append({"context": context, "rep": rep, "task": task,
                                     "fit_status_path": str((task_path / "fit_status.json").relative_to(PHASE)),
                                     "fit_status_sha256": sha(task_path / "fit_status.json")})
        check(completion["all_conditional_states_passed"] == all(successes), "World convergence flag inconsistent")
        grouped = defaultdict(dict)
        predictions = csv_rows(prediction_path)
        prediction_rows += len(predictions)
        check(len(predictions) == 480, "World prediction grid incomplete")
        for row in predictions:
            area = int(row["area"])
            key = (row["method"], row["outcome"])
            check(area not in grouped[key], "Duplicated area prediction")
            grouped[key][area] = row
        check(set(grouped) == {(m, y) for m in ("baseline", "point", "cut8") for y in ("anemia", "stunting")}, "Method/outcome grid changed")
        for (method, outcome), records in sorted(grouped.items()):
            check(set(records) == set(range(80)), "Unfavourable/missing area removed")
            endpoint = 0 if outcome == "anemia" else 1
            squared_errors, errors, coverages, widths, interval_scores = [], [], [], [], []
            for area in range(80):
                row = records[area]
                mean, low, high = [float(row[k]) for k in ("mean", "lower95", "upper95")]
                truth = float(theta[area, endpoint])
                check(all(math.isfinite(x) for x in (mean, low, high)), "Non-finite prediction retained silently")
                check(0 <= mean <= 1 and 0 <= low <= high <= 1, "Invalid probability/interval")
                error = mean - truth
                width = high - low
                # Central95% interval score: width +2/0.05 times one-sided miss.
                penalty = 40 * (low - truth) if truth < low else 40 * (truth - high) if truth > high else 0
                squared_errors.append(error * error)
                errors.append(error)
                coverages.append(float(low <= truth <= high))
                widths.append(width)
                interval_scores.append(width + penalty)
            method_passed = successes[0] if method == "baseline" else successes[1] if method == "point" else all(successes[2:])
            population_metrics.append({"context": context, "rep": rep, "method": method, "outcome": outcome,
                "mse": stats.fmean(squared_errors), "bias": stats.fmean(errors),
                "coverage95": stats.fmean(coverages), "mean_width": stats.fmean(widths),
                "interval_score95": stats.fmean(interval_scores), "areas": 80,
                "all_conditional_states_passed": all(successes), "method_conditional_states_passed": method_passed})
        worlds.append({"context": context, "rep": rep, "completion_sha256": sha(out / "completion.json"),
                       "predictions_sha256": sha(prediction_path), "truth_sha256": sha(truth_path),
                       "all_conditional_states_passed": all(successes), "failed_conditional_tasks": 10 - sum(successes)})
    check(len(population_metrics) == 240 and prediction_rows == 19200, "Full cohort arithmetic count mismatch")
    summaries, paired = [], []
    grouped_metrics = defaultdict(list)
    for row in population_metrics:
        grouped_metrics[(row["context"], row["method"], row["outcome"])].append(row)
    for (context, method, outcome), rows in sorted(grouped_metrics.items()):
        check(len(rows) == 20 and {r["rep"] for r in rows} == set(range(20)), "Metric cohort not all20")
        record = {"context": context, "method": method, "outcome": outcome, "populations": 20}
        for metric in METRICS:
            values = [r[metric] for r in rows]
            record[metric] = stats.fmean(values)
            record[metric + "_MCSE"] = mcse(values)
        record["rmse"] = math.sqrt(record["mse"])
        record["rmse_MCSE_delta_method"] = record["mse_MCSE"] / (2 * record["rmse"])
        record["method_nonconverged_worlds_retained"] = sum(not r["method_conditional_states_passed"] for r in rows)
        summaries.append(record)
    values_by_key = {(r["context"], r["rep"], r["method"], r["outcome"]): r for r in population_metrics}
    for context in ("dense6", "sparse2"):
        for outcome in ("anemia", "stunting"):
            for reference in ("baseline", "point"):
                for metric in METRICS:
                    deltas = [values_by_key[(context, r, "cut8", outcome)][metric] - values_by_key[(context, r, reference, outcome)][metric] for r in range(20)]
                    paired.append({"context": context, "outcome": outcome, "reference": reference,
                                   "metric": metric, "populations": 20,
                                   "cut8_minus_reference": stats.fmean(deltas), "paired_MCSE": mcse(deltas)})
    maximum_absolute_differences = {}
    def compare(filename, independent, keys, columns):
        published = csv_rows(ROOT / "final" / filename)
        keyfunc = lambda row: tuple(str(row[k]) for k in keys)
        expected = {keyfunc(row): row for row in independent}
        got = {keyfunc(row): row for row in published}
        check(len(got) == len(published) and set(got) == set(expected), "Collector output cohort mismatch: " + filename)
        differences = []
        for key, row in got.items():
            for col in columns:
                delta = abs(float(row[col]) - expected[key][col])
                differences.append(delta)
                if delta > TOLERANCE:
                    comparison_failures.append({"file": filename, "key": key, "column": col, "absolute_difference": delta})
        maximum_absolute_differences[filename] = max(differences, default=0)
    compare("population_metrics.csv", population_metrics, ["context", "rep", "method", "outcome"], METRICS + ["areas"])
    compare("summary.csv", summaries, ["context", "method", "outcome"], ["populations", "rmse"] + METRICS + [x + "_MCSE" for x in METRICS])
    compare("paired_diagnostics.csv", paired, ["context", "outcome", "reference", "metric"], ["populations", "cut8_minus_reference", "paired_MCSE"])
    check(len(summaries) == 12 and len(paired) == 40, "Summary dimensions changed")
    by_summary = {(x["context"], x["method"], x["outcome"]): x for x in summaries}
    performance_findings = []
    for context in ("dense6", "sparse2"):
        for outcome in ("anemia", "stunting"):
            candidate = by_summary[(context, "cut8", outcome)]
            baseline = by_summary[(context, "baseline", outcome)]
            point = by_summary[(context, "point", outcome)]
            performance_findings.append({"context": context, "outcome": outcome,
                "cut8_minus_ICC_rmse": candidate["rmse"] - baseline["rmse"],
                "cut8_to_ICC_rmse_ratio": candidate["rmse"] / baseline["rmse"],
                "cut8_minus_ICC_interval_score95": candidate["interval_score95"] - baseline["interval_score95"],
                "cut8_coverage95": candidate["coverage95"], "ICC_coverage95": baseline["coverage95"],
                "central_point_coverage95": point["coverage95"],
                "cut8_absolute_distance_to_nominal_coverage": abs(candidate["coverage95"] - .95),
                "central_point_absolute_distance_to_nominal_coverage": abs(point["coverage95"] - .95)})
    worse_all4 = all(x["cut8_minus_ICC_rmse"] > 0 and x["cut8_minus_ICC_interval_score95"] > 0 for x in performance_findings)
    final_digest = {name: sha(ROOT / "final" / name) for name in ("completion.json", "population_metrics.csv", "summary.csv", "paired_diagnostics.csv")}
    proof = {"status": "PASS_INDEPENDENT_FULL40_ARITHMETIC_FAILURES_RETAINED" if not comparison_failures else "FAIL",
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "audit_source_sha256": sha(__file__), "frozen_simulation_source_sha256": source_sha,
        "frozen_protocol_sha256": protocol_sha, "scope": "Independent scalar metrics and population-level MCSE; diagnostic supplement, not external Holm8 and not a convergence guarantee.",
        "all40_completion_gate_checked_before_prediction_and_truth_access": True,
        "predictions_rows": prediction_rows, "population_metric_rows": len(population_metrics),
        "summaries_rows": len(summaries), "paired_diagnostics_rows": len(paired),
        "worlds": worlds, "worlds_with_any_failed_conditional_task": sum(not x["all_conditional_states_passed"] for x in worlds),
        "failed_conditional_task_count": len(failed_tasks), "failed_conditional_tasks_preserved": failed_tasks,
        "independent_summaries": summaries, "independent_paired_diagnostics": paired,
        "performance_findings_ALL4_context_endpoint_cells": performance_findings,
        "adverse_finding": {"CUT8_worse_than_ICC_on_RMSE_and_interval_score_in_all4_cells": worse_all4,
            "interpretation": "The new diagnostic cohort does not establish CUT8 superiority over the ICC baseline. The coverage benefit versus the central point variant does not cancel higher RMSE and interval score versus ICC. Results from all40 worlds remain included; no tuning or replacement is justified by this audit."},
        "collector_file_sha256": final_digest, "maximum_absolute_differences": maximum_absolute_differences,
        "comparison_tolerance_absolute": TOLERANCE, "comparison_failures": comparison_failures,
        "metric_definition": {"aggregation": "Equal district weights within each80-district finite population; equal weight20 independent realized populations per context.",
            "interval_score95": "width +(2/0.05)*lower-tail miss +(2/0.05)*upper-tail miss",
            "MCSE": "Sample standard deviation across20 independent population metrics divided by sqrt20; not pooled19200-area sampling uncertainty.",
            "paired_MCSE": "Sample standard deviation of20 same-population CUT8-minus-reference metric differences divided by sqrt20.",
            "rmse_MCSE_delta_method": "mse_MCSE/(2*sqrt(mean_population_mse)); supplemental delta-method quantity, not present in original collector."},
        "selection_and_integrity": {"all_planned_populations_retained": True, "failed_finite_predictions_excluded": False,
            "replacement_seeds_or_new_fits": False, "phase1_population_reuse": False,
            "model_or_protocol_edited": False, "heldout_ENDES2024_outcomes_or_scores_opened": False}}
    destination = PHASE / "validacion/phase2_simulation_independent_audit.json"
    with destination.open("x") as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: proof[k] for k in ("status", "predictions_rows", "population_metric_rows", "worlds_with_any_failed_conditional_task", "failed_conditional_task_count", "maximum_absolute_differences")}))


if __name__ == "__main__":
    main()
