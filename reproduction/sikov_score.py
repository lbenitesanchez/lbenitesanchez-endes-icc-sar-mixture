"""Closed-cohort scoring and manuscript-value comparison for fresh Sikov fits."""
from pathlib import Path
import importlib
import json
import numpy as np
import pandas as pd

import sikov_worker as worker


def _arrays(path):
    return dict(np.load(path, allow_pickle=False))


def pre_truth_gate(ctx, require_sensitivity=True, require_b300=True):
    """Validate every new prediction seal before loading any evaluation value."""
    worker.source_gate(); run = ctx["run"]; items = []; failures = []
    development = run / "desarrollo/recreated_native"
    if not worker.read_json(development / "completion.json").get("complete") or \
            worker.read_json(development / "selected_recipe.json")["selected_recipe"] != worker.RECIPE:
        raise RuntimeError("All 32 fresh development fits must reproduce the frozen selection")
    audit = importlib.import_module("sikov_final_audit")
    for context in worker.CONTEXTS:
        for rep in range(30000, 30064):
            source = run / "evaluacion/inputs" / context / f"rep_{rep:05d}"
            manifest = worker.read_json(source / "manifest.json")
            for family, methods in (("fits", worker.METHODS[:9]), ("published_sar", worker.METHODS[9:])):
                folder = run / "evaluacion" / family / context / f"rep_{rep:05d}"
                record = worker.validate_completion(folder, methods)
                if record["truth_read_by_fits"] or record["fit_inputs_sha256"] != worker.sha(source / "model_inputs/inputs.npz") or \
                        record["input_manifest_sha256"] != worker.sha(source / "manifest.json"):
                    raise RuntimeError("Wrong fresh prediction/input role")
                if not all(record["convergence"].values()):
                    failures.append([context, rep, family, record["convergence"]])
                if family == "fits":
                    audit.check_hmc_status(worker.read_json(folder / "ICC_status.json"))
                items.append({"path": str(folder / "completion.json"), "sha256": worker.sha(folder / "completion.json")})
            if manifest["truth_excluded_from_model_inputs"] is not True:
                raise RuntimeError("Truth present in fitting data")
    for scope in ("temporal", "territorial"):
        for family, methods in (("", worker.METHODS[:9]), ("published_sar", worker.METHODS[9:])):
            folder = run / "evaluacion" / family / ("real_" + scope)
            record = worker.validate_completion(folder, methods)
            if record.get("heldout_outcomes_read_by_fits") is not False:
                raise RuntimeError("Held-out response leak or absent actual-fit role")
            required = ("Selected", "Sikov_FH", "Sikov_SAR") if not family else ("Sikov_SAR_Published",)
            if not all(record["convergence"].get(k, False) for k in required):
                raise RuntimeError("A primary actual-data model failed its unchanged gate")
            if not family:
                audit.check_hmc_status(worker.read_json(folder / "ICC_status.json"))
            items.append({"path": str(folder / "completion.json"), "sha256": worker.sha(folder / "completion.json")})
    if require_sensitivity:
        for rep in range(30000, 30064):
            out = run / "sensibilidad_sampled_graph/fits" / f"rep_{rep:05d}"
            record = worker.validate_completion(out, worker.METHODS)
            if record["truth_read_by_fits"] is not False:
                raise RuntimeError("Sampled-graph truth accessed by fitting")
            for family, methods in (("original", worker.METHODS[:9]), ("published", worker.METHODS[9:])):
                worker.validate_completion(out / family, methods)
            audit.check_hmc_status(worker.read_json(out / "original/ICC_status.json"))
            if not all(record["convergence"].values()):
                failures.append(["sampled_graph", rep, record["convergence"]])
            items.append({"path": str(out / "completion.json"), "sha256": worker.sha(out / "completion.json")})
    if require_b300:
        for context in worker.CONTEXTS:
            for rep in (30000, 30001):
                folder = run / "evaluacion/b300" / context / f"rep_{rep:05d}"
                record = worker.validate_completion(folder, ("full_graph", "published"))
                if record["B"] != 300 or record["truth_read_by_fits"] is not False:
                    raise RuntimeError("Wrong B300 sensitivity role")
                if not all(record["convergence"].values()):
                    failures.append(["B300", context, rep, record["convergence"]])
                items.append({"path": str(folder / "completion.json"), "sha256": worker.sha(folder / "completion.json")})
    result = {"status": "PASS_COMPLETE_FRESH_PREDICTION_COHORT_BEFORE_SCORING", "approved_for_scoring": True,
              "protocol_sha256": ctx["protocol_sha256"], "addendum_sha256": ctx["addendum_sha256"],
              "principal_worlds": 256, "sampled_graph_worlds": 64 if require_sensitivity else 0,
              "B300_worlds": 8 if require_b300 else 0, "actual_scopes": ["temporal", "territorial"],
              "files": items, "fresh_truth_read_before_lock": False, "heldout_outcomes_used": False,
              "selected_recipe": worker.RECIPE, "convergence_failures_retained": failures,
              "source_hashes_sha256": worker.source_gate(), "adapter_sha256": worker.sha(__file__)}
    destination = run / "validacion/sikov_fresh_prediction_lock.json"
    if destination.exists() and worker.read_json(destination) != result:
        raise RuntimeError("The existing fresh prediction lock differs; preserve and investigate")
    worker.write_json(destination, result)
    return result


def _prediction_set(run, context, rep, sampled=False):
    if sampled:
        source = run / "sensibilidad_sampled_graph/inputs" / f"rep_{rep:05d}"
        folder = run / "sensibilidad_sampled_graph/fits" / f"rep_{rep:05d}"
        original, published = folder / "original", folder / "published"
    else:
        source = run / "evaluacion/inputs" / context / f"rep_{rep:05d}"
        original = run / "evaluacion/fits" / context / f"rep_{rep:05d}"
        published = run / "evaluacion/published_sar" / context / f"rep_{rep:05d}"
    predictions = {m: _arrays((original if i < 9 else published) / (m + ".npz")) for i, m in enumerate(worker.METHODS)}
    old = worker.read_json(original / "completion.json"); pub = worker.read_json(published / "completion.json")
    convergence = {**old["convergence"], **pub["convergence"]}
    return source, original, published, predictions, convergence, old["seconds"] + pub["seconds"]


def _truth(source):
    manifest = worker.read_json(source / "manifest.json")
    if worker.sha(source / "evaluation_only/truth.npz") != manifest["truth_sha256"]:
        raise RuntimeError("Generated population truth changed after closure")
    return _arrays(source / "evaluation_only/truth.npz")


def _save_csv(frame, path, produced):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False); produced.append(path)


def compare_frame(actual, expected, keys, atol=1e-10):
    a, b = actual.copy(), expected.copy()
    if len(a) != len(b) or a.duplicated(keys).any() or b.duplicated(keys).any():
        return {"passed": False, "reason": "cohort size or duplicate comparison key", "fresh_rows": len(a), "reference_rows": len(b)}
    a = a.set_index(keys).sort_index(); b = b.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        return {"passed": False, "reason": "comparison key universe differs"}
    columns = [k for k in a.columns if k in b.columns and pd.api.types.is_numeric_dtype(a[k]) and
               pd.api.types.is_numeric_dtype(b[k]) and k not in ("fit_seconds", "mean_fit_seconds", "elapsed_seconds", "seconds")]
    results = {}; maximum = 0.
    for key in columns:
        x, y = a[key].to_numpy(float), b[key].to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        maximum_difference = float(np.max(np.abs(x[finite]-y[finite]))) if finite.any() else 0.
        passed = bool(np.allclose(x, y, atol=atol, rtol=0., equal_nan=True))
        results[key] = {"passed": passed, "max_absolute_difference": maximum_difference}
        maximum = max(maximum, maximum_difference)
    return {"passed": all(x["passed"] for x in results.values()), "absolute_tolerance": atol,
            "rows": len(a), "numeric_columns": len(columns), "maximum_absolute_difference": maximum, "columns": results}


def _full_b300(ctx, produced):
    from improve_candidates import mixture_prediction
    run = ctx["run"]; rows = []
    for context in worker.CONTEXTS:
        for rep in (30000, 30001):
            source, original, _, predictions, _, _ = _prediction_set(run, context, rep)
            theta = _truth(source)["theta"]
            arrays = _arrays(run / "evaluacion/b300" / context / f"rep_{rep:05d}/full_graph.npz")
            icc = predictions["ICC"]; draws = _arrays(original / "ICC_conditional_draws.npz")
            mixed = mixture_prediction({"p_mean": icc["mean"], "p_variance": icc["mse"],
                "marginal_draws": [draws["anemia_marginal"], draws["stunting_marginal"]]}, arrays, .75)
            models = {"SAR_B100": predictions["Sikov_SAR"], "SAR_B300": arrays,
                      "Selected_B100": predictions["Selected"], "Selected_B300": mixed}
            for method, prediction in models.items():
                for j, outcome in enumerate(("anemia", "stunting")):
                    y, mean, lo, hi = theta[:, j], prediction["mean"][:, j], prediction["lower"][:, j], prediction["upper"][:, j]
                    rows.append(dict(context=context, replication=rep, outcome=outcome, method=method, areas=120,
                        interval_score=float(np.mean(hi-lo+40*np.maximum(lo-y, 0)+40*np.maximum(y-hi, 0))),
                        coverage=float(np.mean((lo <= y) & (y <= hi))), width=float(np.mean(hi-lo)), mse=float(np.mean((mean-y)**2))))
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["context", "method"])[["interval_score", "coverage", "width", "mse"]].mean().reset_index()
    overall = summary.groupby("method")[["interval_score", "coverage", "width", "mse"]].mean().reset_index()
    overall.insert(0, "context", "equal_four_contexts"); summary = pd.concat([summary, overall], ignore_index=True)
    contrasts = []
    for context in (*worker.CONTEXTS, "equal_four_contexts"):
        local = summary[summary.context == context].set_index("method")
        for candidate, reference in (("Selected_B100", "SAR_B100"), ("Selected_B300", "SAR_B300"), ("Selected_B300", "Selected_B100"), ("SAR_B300", "SAR_B100")):
            contrasts.append({"context": context, "candidate": candidate, "reference": reference,
                "worlds": 8 if context == "equal_four_contexts" else 2,
                **{key: float(local.loc[candidate, key]-local.loc[reference, key]) for key in ("interval_score", "coverage", "width", "mse")}})
    for frame, name in ((detail, "scores"), (summary, "balanced"), (pd.DataFrame(contrasts), "differences")):
        _save_csv(frame, run / ("validacion/sikov_B300_descriptive_" + name + ".csv"), produced)


def _published_b300(ctx, produced):
    run = ctx["run"]; m = ctx["modules"]["sikov_published_b300_sensitivity"]; poverty = ctx["modules"]["sikov_poverty_stratum_descriptive"]
    rows = []
    for context in worker.CONTEXTS:
        for rep in (30000, 30001):
            source, _, _, closed, convergence, _ = _prediction_set(run, context, rep)
            d = _arrays(source / "model_inputs/inputs.npz"); theta = _truth(source)["theta"]
            native = _arrays(run / "evaluacion/b300" / context / f"rep_{rep:05d}/published.npz")
            predictions = {"Selected_B100": closed["Selected"], "Published_B100": closed["Sikov_SAR_Published"],
                          "Published_B100_BC": closed["Sikov_SAR_Published_BC"], "Published_B300": native,
                          "Published_B300_BC": m.bc_arrays(native)}
            success = {k: convergence["Selected"] if k == "Selected_B100" else convergence["Sikov_SAR_Published"] for k in predictions}
            rows.extend(m.score_cells(theta, d, predictions, success, context, rep))
    frame = pd.DataFrame(rows); balanced = poverty.with_endpoint_balance(frame)
    summary = poverty.summarize_populations(balanced, worker.CONTEXTS, (30000, 30001))
    differences = m.paired_scores(balanced, worker.CONTEXTS, (30000, 30001))
    for value, name in ((frame, "per_population_scores.csv"), (summary, "summary.csv"), (differences, "selected_reference_differences.csv")):
        _save_csv(value.drop(columns=["poverty_stratum", "poverty_stratum_label"]), run / "resultados/published_b300_descriptive" / name, produced)


def _article_tables(ctx, produced):
    """Use the manuscript's actual formatting functions to compare printed cells."""
    report = importlib.import_module("report_documents"); run = ctx["run"]; reference = ctx["reference"]
    def data(root):
        folder = root / "resultados/published_sar_comparison"
        paired = pd.read_csv(folder / "paired_balanced_contrasts_Holm3.csv")
        physical = pd.read_csv(folder / "physical_projection_contrasts.csv")
        out = {"summary": pd.read_csv(folder / "summary.csv"),
            "primary": paired[paired.metric.eq("interval_score")],
            "physical": physical[physical.metric.eq("interval_score")],
            "B300_summary": pd.read_csv(root / "validacion/sikov_B300_descriptive_balanced.csv"),
            "published_B300_summary": pd.read_csv(root / "resultados/published_b300_descriptive/summary.csv"),
            "poverty_differences": pd.read_csv(root / "resultados/poverty_stratum_descriptive/selected_reference_descriptive_differences.csv"),
            "sampled_graph_summary": pd.read_csv(root / "sensibilidad_sampled_graph/resultados/descriptive_summary.csv"),
            "sampled_graph_contrasts": pd.read_csv(root / "sensibilidad_sampled_graph/resultados/descriptive_paired_contrasts.csv")}
        return out
    a, b = data(run), data(reference)
    tables = {}
    for name, function in (("Table1_principal", report.aggregate_rows), ("Table2_Holm3", report.primary_rows),
                           ("Table5_BC", report.BC_rows), ("Table6_B300", report.B300_rows),
                           ("Table7_poverty", report.poverty_rows), ("Table8_sampled_graph", report.sampled_graph_rows),
                           ("Supplement_context", report.context_rows)):
        x, y = function(a), function(b)
        mismatches = [{"row": i+1, "column": j+1, "fresh": xv, "article": yv}
                      for i, (rowx, rowy) in enumerate(zip(x, y)) for j, (xv, yv) in enumerate(zip(rowx, rowy)) if xv != yv]
        tables[name] = {"passed": x == y, "rows": len(x), "cells": sum(len(row) for row in x), "mismatches": mismatches,
                        "fresh_rows": x, "reference_rows": y}
    def compare_printed(name, x, y):
        tables[name] = {"passed": x == y, "rows": len(x), "cells": sum(len(row) for row in x),
            "mismatches": [{"row": i+1, "column": j+1, "fresh": xv, "article": yv}
                for i, (rx, ry) in enumerate(zip(x, y)) for j, (xv, yv) in enumerate(zip(rx, ry)) if xv != yv],
            "fresh_rows": x, "reference_rows": y}
    def physical_rows(data):
        return [[report.LABELS[p.reference], report.f(p.difference, 4),
                 "["+report.f(p.MC_CI95_low, 4)+", "+report.f(p.MC_CI95_high, 4)+"]"] for p in data["physical"].itertuples()]
    compare_printed("Table4_physical_projection", physical_rows(a), physical_rows(b))
    external_relative = "resultados/external_comparison_Holm12/paired_balanced_contrasts_Holm12.csv"
    external_pending = not (run / external_relative).is_file()
    if not external_pending:
        def external_rows(root):
            frame = pd.read_csv(root / external_relative)
            return [[p.scope, report.LABELS.get(p.reference, p.reference), p.metric, report.f(p.difference, 5),
                     "["+report.f(p.unadjusted_t_ci_low, 5)+", "+report.f(p.unadjusted_t_ci_high, 5)+"]", report.sci(p.p_Holm12)]
                    for p in frame.itertuples()]
        compare_printed("Table3_external_real", external_rows(run), external_rows(reference))
    destination = run / "validacion/sikov_article_table_cells.json"
    worker.write_json(destination, {"all_checked_printed_cells_match": all(x["passed"] for x in tables.values()),
        "table_function_source_sha256": worker.sha(ctx["source"] / "scripts/report_documents.py"), "tables": tables,
        "external_Table3_pending_separate_real_worker_check": external_pending,
        "legacy_appendix_tables_A1_A2_A3_require_separate_800_40_reproduction": True})
    produced.append(destination)
    return tables


def score_all(ctx):
    run = ctx["run"]; m = ctx["modules"]
    lock = pre_truth_gate(ctx)
    marker = run / "validacion/sikov_score_completion.json"
    if marker.exists():
        raise FileExistsError("Preserve the complete fresh score comparison")
    produced = []; rows = []; physical_rows = []; strata_rows = []
    for context in worker.CONTEXTS:
        for rep in range(30000, 30064):
            source, _, _, predictions, convergence, seconds = _prediction_set(run, context, rep)
            d = _arrays(source / "model_inputs/inputs.npz"); truth = _truth(source)
            for method, prediction in predictions.items():
                for physical, destination in ((False, rows), (True, physical_rows)):
                    pred = prediction if not physical else {**prediction, **{k: np.clip(prediction[k], 0, 1) for k in ("mean", "lower", "upper")}}
                    records = m["improve_development"].scores(truth, d, pred, method, context, rep, [convergence[method]]*2, seconds)
                    published = method in worker.METHODS[9:]
                    for record in records:
                        record["phase"] = "fresh_evaluation_addendum" if published else "fresh_evaluation"
                        if physical:
                            record["phase"] += "_shared_physical_projection"
                            record["protocol"] = "all_methods_shared_projection_0_1_secondary_not_selected"
                        else:
                            record["protocol"] = "sampled_SAR2023_family_directed2024_extension_parametricbootstrap100" if published else "same14_Sikov_parametricbootstrap100_ICC_NUTS_frozen"
                    destination.extend(records)
            strata_rows.extend(m["sikov_poverty_stratum_descriptive"].cells(truth["theta"], d, predictions, convergence, context, rep))
    from benchmark_metrics import summarize
    frame, physical = pd.DataFrame(rows), pd.DataFrame(physical_rows)
    if len(frame) != 256*11*2*3:
        raise RuntimeError("All 16,896 principal score cells are required")
    folder = run / "resultados/published_sar_comparison"
    for value, name in ((frame, "per_replication_scores.csv"), (summarize(frame), "summary.csv"),
                        (m["sikov_published_collect"].expanded_contrasts(frame), "paired_balanced_contrasts_Holm3.csv"),
                        (physical, "physical_projection_scores.csv"), (summarize(physical), "physical_projection_summary.csv"),
                        (pd.DataFrame([r for ref in ("Sikov_FH", "Sikov_SAR", "Sikov_SAR_Published") for r in m["comparison_collect"].balanced(physical, "Selected", ref)]), "physical_projection_contrasts.csv")):
        _save_csv(value, folder / name, produced)
    # Preserve the historical two-contrast family separately from Holm3.
    original = frame[frame.method.isin(worker.METHODS[:9])]
    contrasts = pd.DataFrame([r for ref in worker.METHODS[:8] for r in m["comparison_collect"].balanced(original, "Selected", ref)])
    primary = contrasts.metric.eq("interval_score") & contrasts.reference.isin(("Sikov_FH", "Sikov_SAR"))
    ps = contrasts.loc[primary, "p_two_sided"].to_numpy(); order = np.argsort(ps); adjusted = np.empty(2)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(ps[order]*np.array([2, 1])))
    contrasts.loc[primary, "p_Holm2"] = adjusted
    _save_csv(contrasts, run / "resultados/fresh_balanced_paired_contrasts.csv", produced)
    poverty = m["sikov_poverty_stratum_descriptive"]; strata = pd.DataFrame(strata_rows)
    balanced = poverty.with_endpoint_balance(strata)
    for value, name in ((strata, "per_population_stratum_scores.csv"),
                        (poverty.summarize_populations(balanced, worker.CONTEXTS, tuple(range(30000, 30064))), "stratum_descriptive_summary.csv"),
                        (poverty.paired_descriptive(balanced, worker.CONTEXTS, tuple(range(30000, 30064))), "selected_reference_descriptive_differences.csv")):
        _save_csv(value, run / "resultados/poverty_stratum_descriptive" / name, produced)
    sampled = m["sensitivity_sampled_graph_score"]; sampled_rows = []; sampled_physical = []
    for rep in range(30000, 30064):
        source, _, _, predictions, convergence, _ = _prediction_set(run, worker.CONTEXTS[0], rep, sampled=True)
        theta = _truth(source)["theta"]; d = _arrays(source / "model_inputs/inputs.npz")
        for method, prediction in predictions.items():
            sampled_rows.extend(sampled.score_arrays(theta, d, prediction, method, rep, convergence[method]))
            sampled_physical.extend(sampled.score_arrays(theta, d, prediction, method, rep, convergence[method], physical=True))
    sf, sp = pd.DataFrame(sampled_rows), pd.DataFrame(sampled_physical)
    for value, name in ((sf, "descriptive_replications.csv"), (sampled.descriptive_summary(sf), "descriptive_summary.csv"),
                        (sampled.descriptive_contrasts(sf), "descriptive_paired_contrasts.csv"),
                        (sp, "physical_projection_replications.csv"), (sampled.descriptive_summary(sp), "physical_projection_summary.csv"),
                        (sampled.descriptive_contrasts(sp), "physical_projection_paired_contrasts.csv")):
        _save_csv(value, run / "sensibilidad_sampled_graph/resultados" / name, produced)
    _full_b300(ctx, produced); _published_b300(ctx, produced)
    comparisons = {}
    specs = [("resultados/published_sar_comparison/summary.csv", ["scenario", "method", "protocol", "group", "outcome"]),
             ("resultados/published_sar_comparison/paired_balanced_contrasts_Holm3.csv", ["candidate", "reference", "metric"]),
             ("resultados/published_sar_comparison/physical_projection_contrasts.csv", ["candidate", "reference", "metric"]),
             ("resultados/poverty_stratum_descriptive/stratum_descriptive_summary.csv", ["method", "outcome", "poverty_stratum", "group", "scenario", "metric"]),
             ("resultados/poverty_stratum_descriptive/selected_reference_descriptive_differences.csv", ["method", "outcome", "poverty_stratum", "group", "scenario", "metric", "reference"]),
             ("sensibilidad_sampled_graph/resultados/descriptive_summary.csv", ["method", "outcome", "group"]),
             ("sensibilidad_sampled_graph/resultados/descriptive_paired_contrasts.csv", ["candidate", "reference", "group", "metric"]),
             ("validacion/sikov_B300_descriptive_balanced.csv", ["context", "method"]),
             ("resultados/published_b300_descriptive/summary.csv", ["scenario", "method", "outcome", "group", "metric"]),
             ("resultados/published_b300_descriptive/selected_reference_differences.csv", ["scenario", "method", "outcome", "group", "metric", "reference"])]
    for relative, keys in specs:
        comparisons[relative] = compare_frame(pd.read_csv(run / relative), pd.read_csv(ctx["reference"] / relative), keys)
    tables = _article_tables(ctx, produced)
    result = {"status": "COMPLETED_FRESH_STATISTICAL_REFIT_COMPARISON", "protocol_sha256": ctx["protocol_sha256"],
              "joint_fresh_prediction_lock_sha256": worker.sha(run / "validacion/sikov_fresh_prediction_lock.json"),
              "all_aggregate_numeric_cells_match_at_1e10": all(x["passed"] for x in comparisons.values()),
              "all_checked_article_printed_cells_match": all(x["passed"] for x in tables.values()),
              "all_models_converged": not lock["convergence_failures_retained"], "comparisons": comparisons,
              "outputs_sha256": {str(path.relative_to(run)): worker.sha(path) for path in produced},
              "external_real_results_check_required_separately": True,
              "truth_scoring_only_after_complete_new_prediction_lock": True, "kernels_gates_or_weights_changed": False}
    worker.write_json(marker, result)
    print(json.dumps({k: result[k] for k in ("status", "all_aggregate_numeric_cells_match_at_1e10", "all_checked_article_printed_cells_match", "all_models_converged")}), flush=True)
    return result
