"""Regenerate Figures 1–3 from the newly closed fit/score cohort.

Uses the inspected original plotting statements verbatim, with fresh-output
paths and the new reproduction gate. Historical approval/result certificates
are never copied or fabricated to satisfy the older plotting entry points.
Image pixels can differ with fonts/renderers; plotted numeric data are compared.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import importlib
from pathlib import Path

import pandas as pd
import sikov_worker as worker
from sikov_score import pre_truth_gate, compare_frame


def plotting_block(path, begin_name, stop_name):
    text = Path(path).read_text()
    parsed = ast.parse(text, filename=str(path))
    main = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == name for x in node.targets)
    first = next(i for i, node in enumerate(main.body) if assigns(node, begin_name))
    last = next(i for i, node in enumerate(main.body) if i > first and assigns(node, stop_name))
    nodes = copy.deepcopy(main.body[first:last])
    digest = hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False).encode()).hexdigest()
    block = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    return compile(block, str(path), "exec"), {"first_source_line": main.body[first].lineno,
        "last_source_line": main.body[last-1].end_lineno, "ast_statement_sha256": digest,
        "source_sha256": worker.sha(path), "source": str(path)}


def generate(ctx):
    run = ctx["run"]; ref = ctx["reference"]
    pre_truth_gate(ctx)
    score = worker.read_json(run / "validacion/sikov_score_completion.json")
    for name, digest in score["outputs_sha256"].items():
        if worker.sha(run / name) != digest:
            raise RuntimeError("A completed score output changed before figure generation")
    destination = run / "validacion/sikov_figures_reproduction.json"
    if destination.exists():
        raise FileExistsError("Preserve the completed fresh figure comparison")
    plots = {}
    figure_source = ctx["source"] / "scripts/comparison_figures.py"
    module = importlib.import_module("comparison_figures")
    code, proof = plotting_block(figure_source, "d", "report")
    namespace = dict(vars(module)); namespace.update(ROOT=run, source=run / "resultados/published_sar_comparison")
    exec(code, namespace)
    plots["principal"] = proof
    application_source = ctx["source"] / "scripts/comparison_application_figures.py"
    module = importlib.import_module("comparison_application_figures")
    code, proof = plotting_block(application_source, "prepared", "result")
    namespace = dict(vars(module)); namespace.update(ROOT=run,
        joint=worker.read_json(run / "validacion/sikov_fresh_prediction_lock.json"))
    exec(code, namespace)
    plots["application2019"] = proof
    def read(root, path):
        return pd.read_csv(root / path, dtype={"ubigeo": str})
    summary_path = "resultados/published_sar_comparison/summary.csv"
    new = read(run, summary_path); original = read(ref, summary_path)
    filter_new = new.group.eq("all") & new.method.isin(("Sikov_FH", "Sikov_SAR", "Sikov_SAR_Published", "ICC", "Selected"))
    filter_old = original.group.eq("all") & original.method.isin(("Sikov_FH", "Sikov_SAR", "Sikov_SAR_Published", "ICC", "Selected"))
    comparisons = {"Figure1_plotted_context_endpoint_method_data": compare_frame(new[filter_new], original[filter_old], ["scenario", "method", "outcome"])}
    for name in ("paired_balanced_contrasts_Holm3.csv", "physical_projection_contrasts.csv"):
        path = "resultados/published_sar_comparison/" + name
        new, original = read(run, path), read(ref, path)
        comparisons["Figure2_"+name] = compare_frame(new[new.metric.eq("interval_score")], original[original.metric.eq("interval_score")], ["candidate", "reference", "metric"])
    for name, keys in (("matched_application2019_districts.csv", ["ubigeo", "method"]),
                       ("matched_application2019_summary.csv", ["method", "outcome", "group"])):
        path = "resultados/aplicacion2019/" + name
        comparisons["Figure3_"+name] = compare_frame(read(run, path), read(ref, path), keys)
    outputs = {}
    for folder in (run / "resultados/figuras", run / "resultados/aplicacion2019"):
        for path in folder.iterdir():
            if path.is_file() and path.suffix in (".png", ".pdf", ".csv"):
                outputs[str(path.relative_to(run))] = worker.sha(path)
    result = {"status": "COMPLETED_FRESH_FIGURE_RENDER_AND_NUMERIC_COMPARISON", "figure_numbers": [1, 2, 3],
              "same_plotting_statements": plots, "plotted_data_comparisons": comparisons,
              "all_plot_data_match_at_1e10": all(x["passed"] for x in comparisons.values()),
              "fresh_score_completion_sha256": worker.sha(run / "validacion/sikov_score_completion.json"),
              "new_prediction_lock_sha256": worker.sha(run / "validacion/sikov_fresh_prediction_lock.json"),
              "source_hashes_sha256": worker.source_gate(), "adapter_sha256": worker.sha(__file__),
              "outputs_sha256": outputs, "individual_records_or_heldout_responses_opened": False,
              "historical_certificates_copied_as_fresh_evidence": False, "image_byte_identity_required": False,
              "visual_review_complete": False, "legacy_Figures_A1_A2_A3_and_S1_require_separate_legacy_stage": True}
    worker.write_json(destination, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "reference-root", "run-root", "exe", "cmdstan"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    result = generate(worker.initialize(a.source_root, a.reference_root, a.run_root, a.exe, a.cmdstan))
    print({k: result[k] for k in ("status", "all_plot_data_match_at_1e10", "visual_review_complete")})


if __name__ == "__main__":
    main()
