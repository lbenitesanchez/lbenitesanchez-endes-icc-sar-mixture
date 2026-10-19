"""Regenerate proposal tables from certified derived CSVs, without refitting.

The source numerical functions are retained under soporte/originales. This
adapter changes file routing and the current article's presentation labels.
It does not read Word files or copy the final tables. Fitting, sampler gates
and prediction certification belong to the complete original pipeline.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


REPORT = "src/estudio_publicacion/sikov_2026-10-10/scripts/report_documents.py"
INCREMENTAL = "revision_20261015/code/incremental_icc.py"
DIAGNOSTIC = "revision_20261015/code/reml_correction/corrected_diagnostic_editorial.py"
SOURCE_PATHS = [REPORT, INCREMENTAL, DIAGNOSTIC]
MAIN = [f"Articulo_Tabla_{i}" for i in range(1, 6)]
SUPPLEMENT = [f"Suplemento_Tabla_S{i}" for i in range(1, 10)] + ["Suplemento_Tabla_S23"]
OBJECTS = MAIN + SUPPLEMENT
LABELS = {"ICC": "ICC beta", "Sikov_FH": "Fay–Herriot",
          "Sikov_SAR": "Full-graph SAR", "Sikov_SAR_Published": "Observed-domain SAR",
          "ICC_poverty": "Poverty-stratum ICC", "ICC_SAR_w25": "ICC/SAR 25/75",
          "ICC_SAR_w50": "ICC/SAR 50/50", "ICC_SAR_w75": "Proposed mixture (75/25)"}
SOURCE_TRANSFORMS = {"SAR full-N": "SAR full graph", "Sikov FH": "FH",
                     "SAR published-scope": "SAR observed", "anemia": "anaemia"}
_MODULES = {}
_INCREMENTAL = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(root, relative, name):
    path = (Path(root) / "soporte/originales" / relative).resolve()
    key = (str(path), name)
    if key not in _MODULES:
        spec = importlib.util.spec_from_file_location("iaos_simple_" + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULES[key] = module
    return _MODULES[key]


def _clean(value, *, source=False, full_proposal=False):
    """Current displayed terminology; no numerical parsing or substitution."""
    value = str(value)
    value = {"Mix B100": "Proposed mixture B100",
             "Mix B300": "Proposed mixture B300"}.get(value, value)
    if source:
        for old, new in SOURCE_TRANSFORMS.items():
            value = value.replace(old, new)
    if full_proposal and value == "Selected mixture":
        value = "Proposed mixture (75% ICC beta + 25% SAR)"
    value = value.replace("Selected mixture", "Proposed mixture")
    value = re.sub(r"\bSelected\b", "Proposed mixture", value)
    value = re.sub(r"\bS / ICC\b", "P / ICC", value)
    value = value.replace("\u2011", "-").replace("\\u2003", " ").replace("anemia", "anaemia")
    value = re.sub(r"\b(same|has|than|underlying|descriptive|below|above|all|with|against|from|development|and|at)(?=\d)",
                   r"\1 ", value, flags=re.I)
    return value


def _read(root, relative, evidence):
    path = Path(root) / relative
    evidence[relative] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return pd.read_csv(path)


def _pvalues(rows):
    for row in rows[1:]:
        if float(row[-1]) < .001:
            row[-1] = "<0.001"


def _incremental(root, evidence):
    """Recompute the 450 summaries from paired population primitives."""
    inc = _load(root, INCREMENTAL, "incremental")
    relative = "datos/incremental/per_population_Selected_ICC_metrics.csv"
    oldrelative = "datos/incremental/incremental_summary.csv"
    for name in (relative, oldrelative):
        path = Path(root) / name
        evidence[name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    cachekey = (str(Path(root).resolve()), evidence[relative]["sha256"], evidence[oldrelative]["sha256"])
    if cachekey not in _INCREMENTAL:
        frame = pd.read_csv(Path(root) / relative)
        expected = pd.read_csv(Path(root) / oldrelative)
        recomputed = inc.summarize_increment(frame)
        keys = ["scope", "outcome", "mask", "metric"]
        a = recomputed.sort_values(keys).reset_index(drop=True)
        b = expected.sort_values(keys).reset_index(drop=True)
        if len(a) != 450 or not a[keys].equals(b[keys]):
            raise ValueError("The complete 450-row paired-summary universe differs")
        numeric = [k for k in a if pd.api.types.is_numeric_dtype(a[k]) and not pd.api.types.is_bool_dtype(a[k])]
        maximum = 0.
        for key in numeric:
            x, y = a[key].to_numpy(float), b[key].to_numpy(float)
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                raise ValueError("Paired-summary missingness differs: " + key)
            finite = np.isfinite(x) & np.isfinite(y)
            if not np.allclose(x[finite], y[finite], rtol=1e-11, atol=1e-11):
                raise ValueError("Paired-summary arithmetic differs: " + key)
            if finite.any():
                maximum = max(maximum, float(np.max(np.abs(x[finite] - y[finite]))))
        _INCREMENTAL[cachekey] = (recomputed, {"rows": 450, "numeric_columns_compared": numeric,
                                             "max_absolute_difference": maximum,
                                             "rtol": 1e-11, "atol": 1e-11})
    return _INCREMENTAL[cachekey]


def _incremental_rows(summary, supplement=False):
    """Same table arithmetic as incremental_icc.builder_exports."""
    def get(metric, scope="equal_four_contexts", mask="all", outcome="endpoint_balanced"):
        selected = summary.loc[summary.metric.eq(metric) & summary.scope.eq(scope) &
                               summary["mask"].eq(mask) & summary.outcome.eq(outcome)]
        if len(selected) != 1:
            raise ValueError("Nonunique incremental table cell")
        return selected.iloc[0]
    if not supplement:
        metrics = ["interval_score", "RMSE_sqrt_mean_MSE", "mean_population_RMSE", "bias",
                   "coverage", "width", "aggregate_coverage_distance95"]
        names = {"interval_score": "IS95", "RMSE_sqrt_mean_MSE": "RMSE: sqrt(mean MSE), pp",
                 "mean_population_RMSE": "Mean population RMSE, pp", "bias": "Signed bias, pp",
                 "coverage": "Coverage, %", "width": "Mean interval width, pp",
                 "aggregate_coverage_distance95": "Absolute aggregate coverage deviation, pp"}
        rows = [["Metric", "Selected", "ICC", "Difference", "MCSE", "Descriptive95% interval", "Relative change,%"]]
        for metric in metrics:
            v = get(metric)
            scale = 1 if metric == "interval_score" else 100
            f = lambda x: f"{scale*x:.4f}"
            relative = "—" if pd.isna(v.relative_difference_percent) else f"{v.relative_difference_percent:+.2f}"
            rows.append([names[metric], f(v.selected), f(v.icc), f(v.difference_selected_minus_icc),
                         f(v.difference_mcse), f"[{f(v.descriptive95_low)}, {f(v.descriptive95_high)}]", relative])
        return rows
    contexts = {"SAR_probability_srs": "Probability SAR / SRS", "SAR_logit_srs": "Logit SAR / SRS",
                "SAR_logit_psu_sparse": "Logit SAR / sparse PSU", "IID_logit_psu_sparse": "Logit IID / sparse PSU"}
    masks = {"all": "All", "observed_endpoint": "Observed endpoint", "unsampled_common": "Outside common sample"}
    rows = [["Context", "Outcome", "Domains", "IS95 S / ICC", "IS95 difference (MCSE)",
             "RMSE S / ICC,pp", "Coverage S / ICC,%", "Width S / ICC,pp"]]
    for context in contexts:
        for outcome in ("anemia", "stunting", "endpoint_balanced"):
            for mask in masks:
                iv, rv, cv, wv = [get(k, context, mask, outcome) for k in
                                  ("interval_score", "RMSE_sqrt_mean_MSE", "coverage", "width")]
                rows.append([contexts[context], outcome.replace("_", " "), masks[mask],
                    f"{iv.selected:.4f} / {iv.icc:.4f}",
                    f"{iv.difference_selected_minus_icc:+.4f} ({iv.difference_mcse:.4f})",
                    f"{100*rv.selected:.3f} / {100*rv.icc:.3f}",
                    f"{100*cv.selected:.2f} / {100*cv.icc:.2f}",
                    f"{100*wv.selected:.2f} / {100*wv.icc:.2f}"])
    return rows


def _sparse_rows(frame, supplement=False):
    """Presentation loops retained from new_cohort_editorial_contracts."""
    d = frame.loc[frame.scope.eq("equal_four_contexts") & frame.outcome.eq("endpoint_balanced")]
    if len(d) != 120 or d.duplicated(["mask", "reference_method", "metric"]).any():
        raise ValueError("Incomplete sparse57 balanced summaries")
    def get(mask, reference, metric):
        v = d.loc[d["mask"].eq(mask) & d.reference_method.eq(reference) & d.metric.eq(metric)]
        if len(v) != 1:
            raise ValueError("Sparse57 table cell is not unique")
        return v.iloc[0]
    def ci(v):
        return f"{v.difference_selected_minus_reference:+.6f} [{v.descriptive95_low:+.6f}, {v.descriptive95_high:+.6f}]"
    refs = ["ICC", "Sikov_FH", "Sikov_SAR", "Sikov_SAR_Published"]
    if not supplement:
        rows = [["Reference", "IS95 difference [MC 95% interval]", "Relative change (%)", "RMSE difference (pp)"]]
        for ref in refs:
            iv, rv = get("all", ref, "interval_score"), get("all", ref, "RMSE_sqrt_mean_MSE")
            rows.append([LABELS[ref], ci(iv), f"{iv.relative_difference_percent:.2f}",
                         f"{100*rv.difference_selected_minus_reference:.4f}"])
        return rows
    rows = [["Support", "Reference", "IS95 difference [MC 95% interval]", "RMSE difference (pp)",
             "Coverage difference (pp)", "Width difference (pp)", "Distance95 difference (pp)"]]
    for mask, title in (("all", "All"), ("observed_endpoint", "Observed endpoint"), ("unsampled_common", "Outside sample")):
        for ref in refs:
            iv, rv, cv, wv, dv = [get(mask, ref, k) for k in
                                  ("interval_score", "RMSE_sqrt_mean_MSE", "coverage", "width", "aggregate_coverage_distance95")]
            rows.append([title, LABELS[ref], ci(iv), f"{100*rv.difference_selected_minus_reference:.4f}",
                         *[f"{100*v.difference_selected_minus_reference:.3f}" for v in (cv, wv, dv)]])
    return rows


def _diagnostic_rows(root, evidence):
    """Run only the original aggregate table-construction prefix.

    No fit-certification call is represented as running here. Complete fitting
    gates remain in the original pipeline; these inputs are its certified
    diagnostic summaries, not raw fit attempts.
    """
    for relative in ("datos/diagnosticos/diagnostics_summary.csv",
                     "datos/diagnosticos/observed_numerical_diagnostic_summary.csv"):
        path = Path(root) / relative
        evidence[relative] = {"sha256": sha(path), "bytes": path.stat().st_size}
    source = Path(root) / "soporte/originales" / DIAGNOSTIC
    tree = ast.parse(source.read_text())
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "build")
    cut = next(i for i, x in enumerate(function.body) if isinstance(x, ast.Assign) and
               any(isinstance(t, ast.Name) and t.id == "audit_root" for t in x.targets))
    function.body = function.body[:cut] + [ast.Return(value=ast.Name(id="table", ctx=ast.Load()))]
    class Routes(ast.NodeTransformer):
        def visit_Constant(self, node):
            replace = {"results/diagnostics": "diagnosticos",
                       "results/numerical_cohorts/observed_numerical_diagnostic_summary.csv":
                       "diagnosticos/observed_numerical_diagnostic_summary.csv"}
            if isinstance(node.value, str) and node.value in replace:
                return ast.copy_location(ast.Constant(replace[node.value]), node)
            return node
    module = Routes().visit(ast.Module(body=[function], type_ignores=[]))
    ns = {"Path": Path, "math": math, "rows": lambda p: list(csv.DictReader(p.open()))}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), ns)
    rows = ns["build"](Path(root), Path(root) / "datos")
    # Exact applicability pattern: never replace an observed zero.
    for row in rows[1:]:
        for col in range(2, len(row)):
            if row[col] in ("NA", "NA / NA"):
                row[col] = "—"
    expected = {(1, c) for c in range(2, 8)}
    expected |= {(r, c) for r in (2, 3, 4, 5, 16, 17, 18) for c in (6, 7)}
    expected |= {(r, c) for r in (*range(6, 16), 19, 20, 21) for c in (2, 3, 4, 5)}
    observed = {(r, c) for r, row in enumerate(rows) for c, value in enumerate(row) if value == "—"}
    if len(observed) != 72 or observed != expected:
        raise ValueError("S4 applicability must have exactly its 72 method-specific cells")
    return rows


def _correction_rows(frame):
    keys = ("central_fits", "central_rescued", "bootstrap_requested", "bootstrap_rescued", "bootstrap_failed")
    aggregate = defaultdict(lambda: defaultdict(int))
    if frame.duplicated(["cohort", "method"]).any():
        raise ValueError("Duplicate REML method summary")
    for row in frame.to_dict("records"):
        for key in keys:
            aggregate[row["cohort"]][key] += int(row[key])
    if sum(x["central_fits"] for x in aggregate.values()) != 16680 or sum(x["bootstrap_requested"] for x in aggregate.values()) != 1107600:
        raise ValueError("Correction fit/bootstrap universe differs")
    if sum(x["bootstrap_failed"] for x in aggregate.values()) != 0:
        raise ValueError("The supplied final bootstrap summaries contain rejected fits")
    labels = {"B300_subset_8": "B300 subset of 8", "development_32": "Development32", "observed_graph_64": "Graph64",
              "principal_256": "Principal256", "real_temporal": "Real temporal", "real_territorial": "Real territorial", "sparse57": "NewSparse256"}
    rows = [["Cohort", "Central fits", "Central rescued", "Bootstrap draws", "Bootstrap rescued", "Bootstrap rejected"]]
    for cohort in ("development_32", "principal_256", "sparse57", "observed_graph_64", "B300_subset_8", "real_temporal", "real_territorial"):
        rows.append([labels[cohort]] + [str(aggregate[cohort][key]) for key in keys])
    return rows


def generate(root, object_id, output_dir):
    """Create a CSV for one current table; return input/output provenance."""
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    if object_id not in OBJECTS:
        raise ValueError("Unknown table: " + object_id)
    inputs, source = {}, False
    details = {"operation": "regenerate_from_certified_derived_inputs", "model_fits_run": False,
               "sampler_or_REML_certification_rerun": False}
    used_code = []
    report = None
    if object_id in (MAIN[0], MAIN[1], MAIN[4], *[f"Suplemento_Tabla_S{i}" for i in range(5, 10)]):
        report = _load(root, REPORT, "report")
        used_code.append(REPORT)
        source = True
    sikov = "datos/sikov/"
    if object_id == "Articulo_Tabla_1":
        d = _read(root, sikov + "resultados/published_sar_comparison/summary.csv", inputs)
        rows = [["Estimator", "Outcome", "RMSE", "Coverage", "Width", "IS95"], *report.aggregate_rows({"summary": d})]
    elif object_id == "Articulo_Tabla_2":
        d = _read(root, sikov + "resultados/published_sar_comparison/paired_balanced_contrasts_Holm3.csv", inputs)
        rows = [["Reference", "Difference", "95% MC interval", "Holm p"], *report.primary_rows({"primary": d.loc[d.metric.eq("interval_score")]})]
        _pvalues(rows)
    elif object_id in ("Articulo_Tabla_3", "Suplemento_Tabla_S2"):
        summary, check = _incremental(root, inputs)
        details["paired_recalculation_check"] = check
        used_code.append(INCREMENTAL)
        rows = _incremental_rows(summary, supplement=object_id.startswith("Suplemento"))
    elif object_id in ("Articulo_Tabla_4", "Suplemento_Tabla_S3"):
        d = _read(root, "datos/sparse57/sparse57_exploratory_summary.csv", inputs)
        rows = _sparse_rows(d, supplement=object_id.startswith("Suplemento"))
        details["operation"] = "format_certified_sparse57_summaries"
    elif object_id == "Articulo_Tabla_5":
        d = _read(root, sikov + "resultados/external_comparison_Holm12/paired_balanced_contrasts_Holm12.csv", inputs)
        rows = [["Scope", "Reference", "Metric", "Difference", "95% cluster interval", "Holm p"]]
        rows += [[r.scope, report.LABELS.get(r.reference, r.reference), r.metric, report.f(r.difference, 5),
                  "[" + report.f(r.unadjusted_t_ci_low, 5) + ", " + report.f(r.unadjusted_t_ci_high, 5) + "]", report.sci(r.p_Holm12)]
                 for r in d.itertuples()]
        _pvalues(rows)
    elif object_id == "Suplemento_Tabla_S1":
        laplace = _read(root, "datos/desarrollo/Laplace_candidate_ranking.csv", inputs)
        nuts = _read(root, "datos/desarrollo/NUTS_candidate_ranking.csv", inputs)
        lopo = _read(root, "datos/desarrollo/NUTS_LOPO_selection_and_margins.csv", inputs)
        if len(laplace) != 5 or len(nuts) != 5 or len(lopo) != 32 or laplace.method.duplicated().any() or nuts.method.duplicated().any():
            raise ValueError("Incomplete five-candidate / 32-omission development grid")
        by = nuts.set_index("method")
        counts = Counter(lopo.selected_method)
        rows = [["Candidate", "Laplace IS95 (rank)", "NUTS IS95 (rank)", "NUTS LOPO wins"]]
        for row in laplace.itertuples():
            nx = by.loc[row.method]
            rows.append([LABELS[row.method], f"{row.interval_score:.6f} ({int(row.rank)})",
                         f"{nx.interval_score:.6f} ({int(nx['rank'])})", str(counts[row.method])])
    elif object_id == "Suplemento_Tabla_S4":
        rows = _diagnostic_rows(root, inputs)
        used_code.append(DIAGNOSTIC)
        details["not_applicable_cells"] = 72
    elif object_id == "Suplemento_Tabla_S5":
        d = _read(root, sikov + "resultados/published_sar_comparison/physical_projection_contrasts.csv", inputs)
        rows = [["Reference", "Difference", "95% MC interval"]] + [
            [report.LABELS[r.reference], report.f(r.difference, 4),
             "[" + report.f(r.MC_CI95_low, 4) + ", " + report.f(r.MC_CI95_high, 4) + "]"]
            for r in d.loc[d.metric.eq("interval_score")].itertuples()]
    elif object_id == "Suplemento_Tabla_S6":
        d = _read(root, sikov + "resultados/published_sar_comparison/summary.csv", inputs)
        rows = [["Estimator", "Outcome", "IS95", "Coverage %", "Width pp"], *report.BC_rows({"summary": d})]
    elif object_id == "Suplemento_Tabla_S7":
        a = _read(root, sikov + "validacion/sikov_B300_descriptive_balanced.csv", inputs)
        b = _read(root, sikov + "resultados/published_b300_descriptive/summary.csv", inputs)
        rows = [["Estimator", "IS95", "Coverage %", "Width pp"], *report.B300_rows({"B300_summary": a, "published_B300_summary": b})]
    elif object_id == "Suplemento_Tabla_S8":
        d = _read(root, sikov + "resultados/poverty_stratum_descriptive/selected_reference_descriptive_differences.csv", inputs)
        rows = [["Stratum", "Reference", "All districts", "Observed endpoint", "Outside sample"], *report.poverty_rows({"poverty_differences": d})]
    elif object_id == "Suplemento_Tabla_S9":
        d = _read(root, sikov + "sensibilidad_sampled_graph/resultados/descriptive_summary.csv", inputs)
        rows = [["Strategy", "RMSE pp", "Coverage %", "Width pp", "IS95"], *report.sampled_graph_rows({"sampled_graph_summary": d})]
    elif object_id == "Suplemento_Tabla_S23":
        d = _read(root, "datos/diagnosticos/corrected_method_diagnostic_summary.csv", inputs)
        rows = _correction_rows(d)
    else:
        raise AssertionError(object_id)
    rows = [[_clean(v, source=source, full_proposal=object_id == "Articulo_Tabla_1" and c == 0)
             for c, v in enumerate(row)] for row in rows]
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Ragged output table")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / (object_id.split("_", 1)[1] + ".csv")
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return {"object_id": object_id, **details, "output": str(destination), "sha256": sha(destination),
            "rows": len(rows), "columns": len(rows[0]), "cells": sum(map(len, rows)),
            "inputs": inputs, "source_code": {p: sha(root / "soporte/originales" / p) for p in used_code},
            "adapter_sha256": sha(__file__)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--object", choices=OBJECTS, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.root, args.object, args.output), indent=2, ensure_ascii=False))
