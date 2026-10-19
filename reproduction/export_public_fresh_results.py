#!/usr/bin/env python3
"""Export explicitly sealed modern aggregates and figures to a local checkout.

Nothing is copied implicitly from a private run. Raw/unit/chain/draw files are
forbidden. JSON provenance is sanitized; numerical differences remain visible.
This is an export, never an article reproduction certificate or GitHub push.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re

MAIN_OUTPUTS = {
    **{f"resultados/published_sar_comparison/{name}": "csv" for name in (
        "per_replication_scores.csv", "summary.csv", "paired_balanced_contrasts_Holm3.csv",
        "physical_projection_scores.csv", "physical_projection_summary.csv", "physical_projection_contrasts.csv")},
    "resultados/fresh_balanced_paired_contrasts.csv": "csv",
    **{f"resultados/poverty_stratum_descriptive/{name}": "csv" for name in (
        "per_population_stratum_scores.csv", "stratum_descriptive_summary.csv", "selected_reference_descriptive_differences.csv")},
    **{f"sensibilidad_sampled_graph/resultados/{name}": "csv" for name in (
        "descriptive_replications.csv", "descriptive_summary.csv", "descriptive_paired_contrasts.csv",
        "physical_projection_replications.csv", "physical_projection_summary.csv", "physical_projection_paired_contrasts.csv")},
    **{f"validacion/{name}": "csv" for name in (
        "sikov_B300_descriptive_scores.csv", "sikov_B300_descriptive_balanced.csv", "sikov_B300_descriptive_differences.csv")},
    **{f"resultados/published_b300_descriptive/{name}": "csv" for name in (
        "per_population_scores.csv", "summary.csv", "selected_reference_differences.csv")},
    "validacion/sikov_article_table_cells.json": "json"}
REAL_OUTPUTS = {
    **{f"resultados/{scope}/{name}": "json" if name.endswith(".json") else "csv" for scope in (
        "external_temporal", "external_territorial", "external_extended_temporal", "external_extended_territorial")
       for name in ("summary.json", "endpoint_summary.csv", "paired_balanced_contrasts_unadjusted.csv",
                    "excluded_domains.csv", "district_scores.csv")},
    **{f"resultados/{family}/{name}": "json" if name.endswith(".json") else "csv" for family, names in (
        ("external_comparison", ("paired_balanced_contrasts_Holm8.csv", "external_final_summary.json", "endpoint_summary_both_scopes.csv")),
        ("external_comparison_Holm12", ("external_final_summary.json", "endpoint_summary_both_scopes.csv", "paired_balanced_contrasts_Holm12.csv")))
       for name in names}}
FIGURE_OUTPUTS = {
    **{f"resultados/figuras/{stem}.{suffix}": suffix for stem in (
        "matched_interval_score", "matched_rmse_coverage") for suffix in ("png", "pdf")},
    **{f"resultados/aplicacion2019/matched_application2019_pointmaps.{suffix}": suffix for suffix in ("png", "pdf")},
    "resultados/aplicacion2019/matched_application2019_summary.csv": "csv",
    "resultados/aplicacion2019/matched_application2019_districts.csv": "csv"}
CERTIFICATES = {
    "main": ("validacion/sikov_score_completion.json", "outputs_sha256", "COMPLETED_FRESH_STATISTICAL_REFIT_COMPARISON", MAIN_OUTPUTS),
    "real": ("real_scoring_receipt.json", "results_sha256", "FRESH_REAL_SCORING_COMPLETE", REAL_OUTPUTS),
    "figures": ("validacion/sikov_figures_reproduction.json", "outputs_sha256", "COMPLETED_FRESH_FIGURE_RENDER_AND_NUMERIC_COMPARISON", FIGURE_OUTPUTS)}
UNIT_FIELDS = re.compile(r"^(caseid|child_id|household_id|person_id|id_child|hhid|hvidx|bidx|v00[123]|hv00[123]|hc[0-9]+|age_months|sex|psu|psu_key|stratum|weight_anemia|weight_stunting)$", re.I)
DROP_KEYS = {"command", "commands", "argv", "log", "logs", "private_study", "runtime_R_normalizations"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe_file(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() not in (".csv", ".json", ".png", ".pdf"):
        raise ValueError("Only allowlisted aggregate/figure paths may be exported")
    target = root / path
    if not target.is_file() or target.is_symlink() or not target.resolve().is_relative_to(root):
        raise ValueError("Export source is missing, linked or outside the sealed run")
    return target


def public_path(value, roots):
    path = Path(value)
    for prefix, root in roots:
        try:
            return prefix + "/" + path.resolve().relative_to(root).as_posix()
        except ValueError:
            pass
    return "LOCAL_PATH_REDACTED_SHA256_" + hashlib.sha256(value.encode()).hexdigest()


def sanitize(value, roots):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in DROP_KEYS:
                continue
            output_key = public_path(key, roots) if key.startswith("/") else key
            if output_key in result:
                raise ValueError("Sanitized provenance keys collided")
            result[output_key] = sanitize(item, roots)
        return result
    if isinstance(value, list):
        return [sanitize(item, roots) for item in value]
    if isinstance(value, str):
        if value.startswith("/"):
            return public_path(value, roots)
        if "/Users/" in value or "/.codex/" in value or "/private/" in value:
            return "LOCAL_PATH_REDACTED_SHA256_" + hashlib.sha256(value.encode()).hexdigest()
    return value


def no_private_payload(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if UNIT_FIELDS.fullmatch(key) or key in DROP_KEYS or key.startswith("/"):
                raise ValueError("Record identifiers or private operational metadata in JSON")
            no_private_payload(item)
    elif isinstance(value, list):
        for item in value:
            no_private_payload(item)
    elif isinstance(value, str):
        if "/Users/" in value or "/.codex/" in value or "/private/" in value:
            raise ValueError("Personal machine path in public payload")


def validate_csv(data, relative):
    text = data.decode("utf-8-sig")
    if not text.strip():
        if relative.endswith("/excluded_domains.csv"):
            return {"rows": 0, "columns": [], "granularity": "empty excluded-domain aggregate"}
        raise ValueError("Unexpected empty aggregate CSV")
    frame = csv.DictReader(io.StringIO(text));columns = frame.fieldnames or []
    if len(columns) != len(set(columns)) or any(UNIT_FIELDS.fullmatch(column) for column in columns):
        raise ValueError("Unit identifiers/design records or duplicated CSV headers cannot be published")
    rows = list(frame)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed aggregate CSV rows")
    if any("/Users/" in value or "/.codex/" in value for row in rows for value in row.values()):
        raise ValueError("Personal path in an aggregate cell")
    keys = [key for key in ("ubigeo", "method", "outcome") if key in columns]
    granularity = "simulation/world or grouped model aggregate"
    if "ubigeo" in columns:
        granularity = "district-model/outcome aggregate; no child or household rows"
        if any(not re.fullmatch(r"\d{6}", row["ubigeo"]) for row in rows):
            raise ValueError("District codes must be six-digit public geographic identifiers")
        if len({tuple(row[key] for key in keys) for row in rows}) != len(rows):
            raise ValueError("District/model/outcome keys must uniquely identify every aggregate row")
    return {"rows": len(rows), "columns": columns, "granularity": granularity}


def gate(run, source):
    lock_path = run / "validacion/sikov_fresh_prediction_lock.json";lock = read(lock_path)
    if (lock.get("status") != "PASS_COMPLETE_FRESH_PREDICTION_COHORT_BEFORE_SCORING"
            or lock.get("approved_for_scoring") is not True or lock.get("principal_worlds") != 256
            or lock.get("sampled_graph_worlds") != 64 or lock.get("B300_worlds") != 8
            or set(lock.get("actual_scopes", [])) != {"temporal", "territorial"}
            or lock.get("fresh_truth_read_before_lock") is not False or lock.get("heldout_outcomes_used") is not False):
        raise ValueError("Modern export requires the complete sealed fresh prediction cohort")
    expected_seals = {"evaluacion/fits": 256, "evaluacion/published_sar": 256,
        "sensibilidad_sampled_graph/fits": 64, "evaluacion/b300": 8,
        "evaluacion/real_temporal": 1, "evaluacion/real_territorial": 1,
        "evaluacion/published_sar/real_temporal": 1,
        "evaluacion/published_sar/real_territorial": 1}
    seal_counts = dict.fromkeys(expected_seals, 0);seen = set()
    for item in lock["files"]:
        seal = Path(item["path"]).resolve()
        if not seal.is_relative_to(run) or seal.name != "completion.json" or sha(seal) != item["sha256"]:
            raise ValueError("Fresh completion seal changed after the joint lock")
        relative = seal.relative_to(run).as_posix()
        if relative in seen:
            raise ValueError("The fresh joint lock repeats a completion seal")
        seen.add(relative)
        prefix = next((key for key in sorted(expected_seals, key=len, reverse=True)
                       if relative.startswith(key + "/")), None)
        if prefix is None:
            raise ValueError("Unexpected completion scope in the fresh joint lock")
        seal_counts[prefix] += 1
    if seal_counts != expected_seals:
        raise ValueError("The complete modern completion universe is required")
    for source_name, checksum in lock["source_hashes_sha256"].items():
        file = Path(source_name).resolve()
        if not file.is_relative_to(source) or sha(file) != checksum:
            raise ValueError("Frozen scientific source differs from the scoring gate")
    return lock_path, lock


def reviewed_diagnostic(args, repo, run, lock_sha, roots, payloads):
    """Validate completed optional aggregate diagnostics before publishing."""
    audit = {}
    for label, report_path, receipt_path in (
        ("modern_printed_cells", args.modern_comparison, args.comparison_receipt),
        ("training_mcse_interpretation", args.interpretation_summary, args.interpretation_receipt)):
        if (report_path is None) != (receipt_path is None):
            raise ValueError("An optional diagnostic requires its execution receipt: " + label)
        if report_path is None:
            continue
        if report_path.suffix != ".json" or receipt_path.suffix != ".json":
            raise ValueError("Only explicitly named JSON aggregate diagnostics are allowed")
        report = read(report_path);receipt = read(receipt_path)
        if receipt.get("exit_code") != 0 or receipt.get("result_sha256") != sha(report_path):
            raise ValueError("Diagnostic execution did not close successfully with the current report")
        if report.get("numerical_reproduction_confirmed") is not False or report.get("article_reproduction_confirmed") is not False:
            raise ValueError("Partial diagnostic cannot certify full article reproduction")
        if label == "modern_printed_cells":
            if (report.get("status") != "COMPLETE_MODERN_PRINTED_CELL_COMPARISON_DIFFERENCES_RETAINED"
                    or report.get("fresh_modern_scoring_completed") is not True
                    or report.get("prediction_lock_sha256") != lock_sha
                    or report.get("modern_tables_evaluated") != 8 or report.get("total_modern_document_cells") != 323
                    or report.get("all_mismatches_included") is not True
                    or report.get("legacy_results_or_archived_accuracy_CSVs_or_truth_or_private_records_read") is not False):
                raise ValueError("Modern printed comparison is incomplete or not bound to the closed cohort")
            for field in ("fresh_score_certificate_sha256", "completed_modern_output_sha256"):
                for relative, checksum in report[field].items():
                    if sha(safe_file(run, relative)) != checksum:
                        raise ValueError("Printed comparison input changed: " + relative)
            audit[label] = {"source_sha256": sha(report_path), "execution_receipt_sha256": sha(receipt_path),
                "cells": report["total_modern_document_cells"], "differences": report["total_modern_different_cells"],
                "full_article_comparison_complete": report["full_manuscript_and_supplement_comparison_complete"]}
        else:
            counts = {"primary_original": 256, "primary_published": 256, "sampled_graph_original": 64,
                "sampled_graph_published": 64, "real_temporal_original": 1, "real_temporal_published": 1,
                "real_territorial_original": 1, "real_territorial_published": 1}
            if (report.get("status") != "FULL_COHORT_TRAINING_ONLY_DIAGNOSTIC_COMPLETE_NO_EQUIVALENCE_CERTIFICATE"
                    or report.get("full_cohort_complete") is not True or report.get("cohort_counts") != counts
                    or report.get("analyzed_world_count") != 322 or report.get("failed_world_count") != 0
                    or report.get("truth_or_accuracy_scores_read") is not False
                    or report.get("models_fitted_by_this_helper") is not False
                    or report.get("archived_MCSE_available") is not False
                    or report.get("statistical_equivalence_certified") is not False
                    or report.get("public_scalar_summary_only") is not True
                    or receipt.get("output_universe_complete") is not True or receipt.get("validation_errors") != []
                    or report.get("helper_sha256") != sha(repo / "reproduction/prediction_interpretation.py")):
                raise ValueError("The full training-only diagnostic has not completed under the reviewed contract")
            audit[label] = {"source_sha256": sha(report_path), "execution_receipt_sha256": sha(receipt_path),
                "worlds": 322, "failed_worlds": 0, "statistical_equivalence_certified": False,
                "archived_MCSE_available": False}
        value = sanitize(report, roots);no_private_payload(value)
        public_receipt = sanitize(receipt, roots);no_private_payload(public_receipt)
        for relative, value in (("comparisons/" + label + ".json", value),
                ("certificates/" + label + "_execution.json", public_receipt)):
            payloads[relative] = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        audit[label]["public_sha256"] = hashlib.sha256(payloads["comparisons/" + label + ".json"]).hexdigest()
    return audit


def export(args):
    cfg = read(args.config);repo = Path(cfg["repo"]).resolve();run = Path(cfg["sikov_run_root"]).resolve()
    source = Path(cfg["source_root"]).resolve();reference = Path(cfg["reference_root"]).resolve()
    target = repo / "results_fresh/modern"
    if run.is_relative_to(repo) or not (repo / "src/estudio_publicacion").is_dir():
        raise ValueError("Use an external sealed private run and the curated public repository")
    if target.exists() and not args.check_only:
        raise FileExistsError("Preserve an earlier export; do not silently overwrite it")
    lock_path, lock = gate(run, source);lock_sha = sha(lock_path)
    roots = (("PUBLIC_SOURCE", source), ("RUN_RELATIVE", run), ("REFERENCE_RELATIVE", reference))
    files, seals, payloads = [], {}, {}
    for label, (name, output_key, status, allowlist) in CERTIFICATES.items():
        certificate_path = run / name;certificate = read(certificate_path)
        if certificate.get("status") != status or set(certificate.get(output_key, {})) != set(allowlist):
            raise ValueError("Certificate status/output universe differs from the reviewed allowlist: " + label)
        for field in ("joint_fresh_prediction_lock_sha256", "complete_fresh_prediction_lock_sha256", "new_prediction_lock_sha256"):
            if field in certificate and certificate[field] != lock_sha:
                raise ValueError("Scoring or plotting certificate is not bound to the fresh cohort")
        if label == "figures" and certificate.get("fresh_score_completion_sha256") != sha(run / CERTIFICATES["main"][0]):
            raise ValueError("Figure inputs are not bound to the fresh score certificate")
        if label == "real" and certificate.get("independent_arithmetic_and_exclusion_checks_passed") != 193:
            raise ValueError("The full real-data arithmetic/exclusion audit is required")
        seals[label] = {"private_certificate_sha256": sha(certificate_path), "original_status": status}
        public_certificate = sanitize(certificate, roots);no_private_payload(public_certificate)
        public_certificate.update(public_provenance_sanitized=True, private_original_certificate_sha256=sha(certificate_path),
                                  article_reproduction_confirmed=False)
        payloads["certificates/" + label + "_comparison.json"] = (json.dumps(public_certificate, indent=2, allow_nan=False) + "\n").encode()
        for relative, kind in allowlist.items():
            path = safe_file(run, relative);original = path.read_bytes()
            if sha(path) != certificate[output_key][relative]:
                raise ValueError("Sealed aggregate/figure changed before export: " + relative)
            schema = {}
            if kind == "csv":
                schema = validate_csv(original, relative);published = original
            elif kind == "json":
                value = sanitize(json.loads(original), roots);no_private_payload(value)
                published = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
            else:
                if (kind == "png" and not original.startswith(b"\x89PNG\r\n\x1a\n")) or (kind == "pdf" and not original.startswith(b"%PDF-")):
                    raise ValueError("Figure file magic differs from its declared type")
                published = original
            payloads[relative] = published
            files.append({"path": relative, "certificate": label, "private_source_sha256": sha(path),
                          "public_sha256": hashlib.sha256(published).hexdigest(), "bytes": len(published),
                          "sanitized_JSON": kind == "json", **schema})
    diagnostics = reviewed_diagnostic(args, repo, run, lock_sha, roots, payloads)
    source_seal = {"status": lock["status"], "joint_lock_sha256": lock_sha,
        "principal_worlds": 256, "sampled_graph_worlds": 64, "B300_worlds": 8,
        "actual_scopes": lock["actual_scopes"], "selected_recipe": lock["selected_recipe"],
        "completion_seals_verified": len(lock["files"]), "statistical_sources_verified": len(lock["source_hashes_sha256"]),
        "truth_read_by_fits": False, "raw_or_unit_or_chain_or_draw_payloads_exported": False}
    real = read(run / CERTIFICATES["real"][0]);main = read(run / CERTIFICATES["main"][0]);figures = read(run / CERTIFICATES["figures"][0])
    table = real["Table3_printed_cells"]
    summary = {"status": "PUBLIC_MODERN_AGGREGATES_EXPORTED_WITH_NUMERICAL_DIFFERENCES" if not args.check_only else "CHECK_ONLY_PUBLIC_MODERN_EXPORT_ALLOWLIST_VALIDATED",
        "cohort": source_seal, "certificates": seals, "exported_source_files": len(files),
        "figures": [1, 2, 3], "visual_review_complete": figures["visual_review_complete"],
        "main_all_aggregate_numeric_cells_match": main["all_aggregate_numeric_cells_match_at_1e10"],
        "main_all_checked_printed_cells_match": main["all_checked_article_printed_cells_match"],
        "real_Table3_cells": table["cells"], "real_Table3_mismatches": len(table["mismatches"]),
        "real_Table3_matches": table["passed"], "all_modern_models_converged": main["all_models_converged"],
        "three_figure_data_matches": figures["all_plot_data_match_at_1e10"],
        "independent_real_arithmetic_checks": 193,
        "legacy_and_complete_article_comparison_pending": True,
        "numerical_reproduction_confirmed": False, "article_reproduction_confirmed": False,
        "private_records_or_chain_draw_raw_files_published": False,
        "private_machine_paths_removed": True, "incomplete_MCSE_watcher_outputs_published": False,
        "completed_optional_diagnostics": diagnostics,
        "public_artifacts_sha256": {relative: hashlib.sha256(data).hexdigest()
            for relative, data in sorted(payloads.items())},
        "helper_sha256": sha(Path(__file__)), "files": sorted(files, key=lambda item: item["path"])}
    if not args.check_only:
        target.mkdir(parents=True)
        for relative, data in sorted(payloads.items()):
            destination = target / relative;destination.parent.mkdir(parents=True, exist_ok=True);destination.write_bytes(data)
        (target / "export_manifest.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        (target / "README.md").write_text("# Fresh modern results\n\nThese are newly recomputed district/model aggregates and Figures 1–3. The original manuscript reference results remain separate. Numerical differences are retained, including 29 of 72 Table 3 cells. The legacy stages and complete article comparison remain pending. This folder is a local export, not a reproduction confirmation or journal submission.\n\nRaw records, person/household rows, chains, conditional draws and private machine paths are excluded. JSON provenance has been sanitized; original and public SHA-256 identities are recorded in export_manifest.json.\n")
        for item in files:
            if sha(target / item["path"]) != item["public_sha256"]:
                raise RuntimeError("Published aggregate hash differs after copying")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True);args.report.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "exported_source_files", "figures", "real_Table3_cells", "real_Table3_mismatches", "numerical_reproduction_confirmed", "article_reproduction_confirmed")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--modern-comparison", type=Path, help="Closed modern printed-cell aggregate JSON")
    parser.add_argument("--comparison-receipt", type=Path, help="Successful receipt bound to --modern-comparison")
    parser.add_argument("--interpretation-summary", type=Path, help="Closed public training-only MCSE summary")
    parser.add_argument("--interpretation-receipt", type=Path, help="Successful receipt bound to --interpretation-summary")
    export(parser.parse_args())


if __name__ == "__main__":
    main()
