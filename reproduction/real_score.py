"""Score fresh real fits against independently reparsed ENDES 2024 responses."""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys
import pandas as pd
import sikov_worker as shared
from real_worker import sha, write_json, locations


def score(args):
    if not args.exe or not args.cmdstan:
        raise ValueError("score requires --exe and --cmdstan for shared runtime initialization")
    ctx = shared.initialize(args.source_root, args.reference_root, args.run_root, args.exe, args.cmdstan)
    from sikov_score import pre_truth_gate, compare_frame
    # This validates the 32 development, 256 main, 64 sampled-graph, eight B300,
    # and both actual-data scopes before accessing reserved scoring responses.
    joint = pre_truth_gate(ctx, require_sensitivity=True, require_b300=True)
    run, source, reference = ctx["run"], ctx["source"], ctx["reference"]
    preparation = json.loads((run / "real_preparation_receipt.json").read_text())
    private = Path(preparation["private_study"])
    sys.path.insert(0, str(source / "scripts"))
    old = importlib.import_module("align_external_score")
    extended = importlib.import_module("align_external_extended_score")
    audit = importlib.import_module("align_external_extended_audit")
    old.ROOT, old.STUDY, old.OLD = run, private, private / "fase2_2026-10-07"
    extended.ROOT = run
    frames, sealed_files = {}, []
    for scope in ("temporal", "territorial"):
        original = run / "evaluacion" / ("real_" + scope)
        published = run / "evaluacion/published_sar" / ("real_" + scope)
        first = shared.validate_completion(original, shared.METHODS[:9])
        second = shared.validate_completion(published, shared.METHODS[9:])
        if first.get("heldout_outcomes_read_by_fits") is not False or second.get("heldout_outcomes_read_by_fits") is not False:
            raise RuntimeError("An actual-data fit used heldout responses")
        if second["original_completion_sha256"] != sha(original / "completion.json") or second["original_predictions_sha256"] != sha(original / "predictions.csv"):
            raise RuntimeError("Published reference is not bound to the fresh original real fit")
        folder = "real_2019_full" if scope == "temporal" else "real_2024_training"
        if first["training_manifest_sha256"] != sha(run / "datos" / folder / "preparation_manifest.json"):
            raise RuntimeError("Actual fit is not bound to raw SAV reconstruction")
        for root in (original, published):
            for name in ("completion.json", "predictions.csv"):
                sealed_files.append({"path": str(root / name), "sha256": sha(root / name)})
        frame = pd.concat([old.prediction_frame(original / "predictions.csv", extended.METHODS[:3]),
                           old.prediction_frame(published / "predictions.csv", [extended.METHODS[3]])], ignore_index=True)
        if frame[["ubigeo", "method"]].duplicated().any():
            raise RuntimeError("Repeated combined district/method prediction")
        frames[scope] = {"frame": frame, "original": original, "published": published,
                         "completion": first, "published_completion": second}
    root = run / "resultados"
    names = ("external_temporal", "external_territorial", "external_extended_temporal", "external_extended_territorial",
             "external_comparison", "external_comparison_Holm12")
    if any((root / name).exists() for name in names):
        raise FileExistsError("External scoring refuses an existing or partially written result")
    real_lock = {"status": "PASS_ALL_NEW_REAL_PREDICTIONS_LOCKED_BEFORE_SCORING", "approved_for_scoring": True,
        "protocol_sha256": ctx["protocol_sha256"], "addendum_sha256": ctx["addendum_sha256"],
        "complete_fresh_cohort_lock_sha256": sha(run / "validacion/sikov_fresh_prediction_lock.json"),
        "real_preparation_receipt_sha256": sha(run / "real_preparation_receipt.json"), "files": sealed_files,
        "new_real_fits_scopes": ["temporal", "territorial"], "scoring_heldout_read_before_this_gate": False,
        "2024_reexecution_disclosed_not_new_independent_validation": True,
        "fresh_basis_difference_disclosed": preparation.get("basis_difference_disclosed", False)}
    lockpath = run / "validacion/real_scoring_prediction_lock.json"
    write_json(lockpath, real_lock)
    root.mkdir(parents=True, exist_ok=True)
    core_paths, extended_paths, target_sources = [], [], {}
    # Supplying the exact protocol/addendum/joint fields expected by the frozen
    # writer changes only provenance locations. Scoring arithmetic is unchanged.
    writer_args = argparse.Namespace(joint_lock=lockpath)
    for scope in ("temporal", "territorial"):
        response = old.OLD / ("datos/private_external/children_2024_scoring_only.csv" if scope == "temporal"
                              else "datos/private_external/children_2024_test.csv")
        target_sources[scope] = {"path": str(response), "sha256": sha(response)}
        target = old.direct_measurements(response, scope)
        detail, summary, excluded = extended.score_frame(frames[scope]["frame"], target)
        core_paths.append(extended.write_scope(root / ("external_" + scope), scope, frames[scope], detail, summary,
            excluded, extended.METHODS[1:3], ctx["protocol_sha256"], ctx["addendum_sha256"], writer_args))
        extended_paths.append(extended.write_scope(root / ("external_extended_" + scope), scope, frames[scope], detail, summary,
            excluded, extended.METHODS[1:], ctx["protocol_sha256"], ctx["addendum_sha256"], writer_args))
    old.collect(core_paths, root / "external_comparison")
    extended.collect12(extended_paths, root / "external_comparison_Holm12")
    auditpath = run / "validacion/real_external_independent_numeric_audit.json"
    audit.audit([p.parent for p in extended_paths], root / "external_comparison_Holm12", auditpath)
    independent = json.loads(auditpath.read_text())
    if independent["checks_passed"] != 193:
        raise RuntimeError("Expected complete 193-check external arithmetic/exclusion audit")
    comparisons = {}
    for family, name, keys in (
        ("external_comparison_Holm12", "paired_balanced_contrasts_Holm12.csv", ["scope", "reference", "metric"]),
        ("external_comparison_Holm12", "endpoint_summary_both_scopes.csv", ["scope", "method", "outcome"]),
        ("external_comparison", "paired_balanced_contrasts_Holm8.csv", ["scope", "reference", "metric"])):
        fresh, archived = root / family / name, reference / "resultados" / family / name
        comparisons[family + "/" + name] = compare_frame(pd.read_csv(fresh), pd.read_csv(archived), keys)
    report = importlib.import_module("report_documents")
    def printed(path):
        frame = pd.read_csv(path)
        return [[p.scope, report.LABELS.get(p.reference, p.reference), p.metric, report.f(p.difference, 5),
                 "[" + report.f(p.unadjusted_t_ci_low, 5) + ", " + report.f(p.unadjusted_t_ci_high, 5) + "]", report.sci(p.p_Holm12)]
                for p in frame.itertuples()]
    rel = "external_comparison_Holm12/paired_balanced_contrasts_Holm12.csv"
    fresh_rows, article_rows = printed(root / rel), printed(reference / "resultados" / rel)
    mismatches = [{"row": i + 1, "column": j + 1, "fresh": fresh, "article": article}
                  for i, (freshrow, oldrow) in enumerate(zip(fresh_rows, article_rows))
                  for j, (fresh, article) in enumerate(zip(freshrow, oldrow)) if fresh != article]
    result = {"status": "FRESH_REAL_SCORING_COMPLETE", "independent_arithmetic_and_exclusion_checks_passed": independent["checks_passed"],
        "complete_fresh_prediction_lock_sha256": sha(run / "validacion/sikov_fresh_prediction_lock.json"),
        "real_scoring_lock_sha256": sha(lockpath), "raw2024_reparsed_target_sha256": target_sources,
        "frozen_numeric_scorers_sha256": {Path(m.__file__).name: sha(m.__file__) for m in (old, extended, audit)},
        "operational_score_adapter_sha256": sha(__file__), "reference_comparisons": comparisons,
        "Table3_printed_cells": {"passed": fresh_rows == article_rows, "rows": len(fresh_rows),
            "cells": sum(map(len, fresh_rows)), "mismatches": mismatches, "fresh_rows": fresh_rows, "article_rows": article_rows},
        "basis_difference_disclosed": preparation.get("basis_difference_disclosed", False),
        "raw_records_exported": False, "new_independent_2024_validation_claim": False,
        "all_raw_private_chain_and_draw_outputs_remain_local": True,
        "results_sha256": {str(p.relative_to(run)): sha(p) for name in names for p in (root / name).rglob("*") if p.is_file()}}
    write_json(run / "real_scoring_receipt.json", result)
    print(json.dumps({"status": result["status"], "numeric_audit_checks": 193,
                      "Table3_printed_cells_match": fresh_rows == article_rows,
                      "Table3_mismatches": len(mismatches)}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "reference-root", "run-root", "exe", "cmdstan"):
        parser.add_argument("--" + name, type=Path, required=True)
    score(parser.parse_args())


if __name__ == "__main__":
    main()
