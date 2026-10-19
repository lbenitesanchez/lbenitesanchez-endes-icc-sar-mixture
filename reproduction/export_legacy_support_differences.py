#!/usr/bin/env python3
"""Export four fresh simulation aggregates while retaining FAIL_SUPPORT.

These are new artifact-identity observations, not comparator-issued SHA seals
or numerical matches. All completed cohort, input, source and printed-table
gates must pass before any CSV payload is opened. No fits or scoring run here.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile

import export_public_fresh_legacy_v2 as legacy

PREFIX = "resultados/resultados_comparadores/final_evaluation/"
FILES = tuple(PREFIX + stem + ".csv" for stem in (
    "combined_replications", "combined_summary", "additional_arms_summary", "converged_only_summary"))
KEYS = ("scenario", "method", "protocol", "group", "outcome")
SCENARIOS = {"binary_informative_mar", "sparse_psu_2025", "sparse_psu_bym2", "sparse_srs_2025", "sparse_srs_iid"}
METHODS = {"BetaPowerHMCGraph", "Direct", "DirectWilsonCluster", "DirectWilsonKish", "FH",
    "GaoAuthorStrict", "GaoAuthorStrictFullH", "GaoCommonContinuityX", "HoggDesignPsiPSU",
    "HoggTSLN", "HoggTSLNFullH", "MFH", "MSFH", "SFH"}
PROTOCOLS = {"common_info_areaX_primary_df_graph", "experimental_design_PSU_v1",
    "extra_population_auxiliary_H_v1", "frozen_independent_v1"}


def complete_verification(value):
    if (value.get("format") != "iaos-final-reproduction-verification-v1"
            or value.get("status") != "COMPLETED_COMPARISON_WITH_DISCLOSED_DIFFERENCES"
            or value.get("comparison_complete") is not True
            or value.get("full_final_closure_ready_for_local_package") is not True
            or value.get("all_fresh_figures_visual_QA_pending") is not False
            or value.get("article_reproduction_confirmed") is not False
            or value.get("numerical_reproduction_confirmed") is not False
            or value.get("printed_article", {}).get("table_count") != 22
            or value.get("printed_article", {}).get("total_document_cells") != 1488
            or not value.get("finished_utc")):
        raise legacy.Pending("Completed disclosed-differences integration is required")


def schema(data, relative):
    if relative not in FILES:
        raise ValueError("CSV is outside the four-file aggregate allowlist")
    result = legacy.validate_csv(data, relative)
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    keys = ("scenario", "replication", "method", "protocol", "group", "outcome") if relative.endswith("combined_replications.csv") else KEYS
    if not rows or not set(keys).issubset(result["columns"]):
        raise ValueError("Required aggregate support keys are missing")
    if len({tuple(row[k] for k in keys) for row in rows}) != len(rows):
        raise ValueError("Aggregate support keys are not unique")
    if any(not all(row[k] for k in keys) or row["scenario"] not in SCENARIOS
            or row["method"] not in METHODS or row["protocol"] not in PROTOCOLS for row in rows):
        raise ValueError("Unknown or blank simulation support label")
    if "replication" in keys and any(not row["replication"].isdigit() for row in rows):
        raise ValueError("Simulation seed identifier is malformed")
    result.update({"unique_aggregate_keys": list(keys), "aggregate_keys_verified_unique": True,
        "scenario_count": len({row["scenario"] for row in rows}),
        "method_count": len({row["method"] for row in rows}),
        "protocol_count": len({row["protocol"] for row in rows})})
    return result


def support_claim(check, observed_sha):
    if check.get("status") != "FAIL_SUPPORT":
        raise ValueError("Supplementary export requires the retained FAIL_SUPPORT comparison")
    if check.get("fresh_sha256"):
        raise ValueError("Comparator-sealed aggregates belong in the primary export")
    return {"original_comparison_status": "FAIL_SUPPORT", "public_status": "FAIL_SUPPORT_RETAINED",
        "reference_support_matches": False, "numeric_comparison_passed": False,
        "comparator_issued_fresh_SHA_seal": False,
        "observed_fresh_artifact_sha256": observed_sha,
        "new_SHA_certifies_current_artifact_identity_only": True,
        "fresh_rows_in_original_comparison": check.get("fresh_rows"),
        "reference_rows_in_original_comparison": check.get("reference_rows")}


def export(args):
    cfg = legacy.read(args.config);repo = Path(cfg["repo"]).resolve()
    final = args.finalization_root.resolve();run = Path(cfg["legacy_run_root"]).resolve();study = run / "study"
    verification_path = legacy.require_path(args.verification.resolve());verification = legacy.read(verification_path)
    complete_verification(verification)
    refs = {(v["scope"], v["relative_reference"]): v["sha256"] for v in verification["provenance_references"]}
    for scope, name, path in (
        ("private_finalization", "finalization.json", final / "finalization.json"),
        ("private_postreview", "article_printed_cells_complete.json", args.printed_cells),
        ("private_postreview", "receipts/all22_printed_cells.json", args.printed_cells_receipt)):
        if refs.get((scope, name)) != legacy.sha(legacy.require_path(path)):
            raise ValueError("Final integration refers to another completed run")
    # This immutable checked exporter verifies all five execution ledgers, 800
    # input identities, scientific source bytes, scoring receipts and all22.
    checked = legacy.export(argparse.Namespace(finalization_root=final,
        printed_cells=args.printed_cells, printed_cells_receipt=args.printed_cells_receipt,
        check_only=True), cfg)
    if set(item["path"] for item in checked["csv_omissions"]) != set(FILES):
        raise ValueError("The primary export's omissions differ from this reviewed supplement")
    state = legacy.read(final / "finalization.json")
    comparison = legacy.read(final / "legacy_comparison.json")
    execution = legacy.read(run / "legacy_previous800_execution.json")
    # Both canonical and mixed relay closures retain actual collector receipts.
    receipts_root = run / "receipts" / ("previous800_relay" if execution.get("closure_basis") else "previous800")
    collector_receipts = {}
    for index in (6, 8, 9, 10, 17, 18, 19):
        path = legacy.require_path(receipts_root / f"{index:03d}.json");receipt = legacy.read(path)
        if receipt.get("exit_code") != 0 or not receipt.get("finished_utc"):
            raise legacy.Pending("A current collection/final-audit receipt is not complete")
        collector_receipts[f"{index:03d}"] = legacy.sha(path)
    payloads = {};files = []
    for relative in FILES:
        path = legacy.contained_file(study, relative, suffixes=(".csv",));data = path.read_bytes()
        observed = legacy.sha(path)
        if hashlib.sha256(data).hexdigest() != observed:
            raise ValueError("Aggregate changed during identity observation")
        row = {"path": relative, "bytes": len(data), **schema(data, relative),
            **support_claim(comparison["checks"][relative], observed)}
        if legacy.sha(path) != observed:
            raise ValueError("Aggregate changed after schema review")
        payloads[relative] = data;files.append(row)
    result = {"format": "iaos-fresh-support-difference-export-v1",
        "status": "CHECK_ONLY_SUPPORT_DIFFERENCES_VALIDATED" if args.check_only else "FRESH_AGGREGATES_EXPORTED_FAIL_SUPPORT_RETAINED",
        "files": files, "files_exported": len(files), "final_verification_sha256": legacy.sha(verification_path),
        "original_finalization_sha256": checked["complete_global_finalization_sha256"],
        "original_comparator_sha256": legacy.sha(final / "legacy_comparison.json"),
        "full22_printed_cells_sha256": checked["complete_printed_cells_sha256"],
        "fresh_previous800_stage_sha256": legacy.sha(run / "legacy_previous800_execution.json"),
        "current_collection_and_final_audit_receipts_sha256": collector_receipts,
        "strict_input_identity_sha256": legacy.sha(run / "legacy_public_strict_input_identity.json"),
        "all_five_stage_execution_and_source_gates_rechecked": True,
        "numerical_reproduction_confirmed": False, "article_reproduction_confirmed": False,
        "comparison_failures_reference_and_primary_export_modified": False,
        "new_hashes_certify_current_artifact_identity_only": True,
        "raw_records_chains_draws_private_paths_or_Stan_inputs_exported": False,
        "fits_scores_Git_operations_or_ZIP_dispatched": False,
        "helper_sha256": legacy.sha(__file__), "gate_helper_sha256": legacy.sha(legacy.__file__)}
    legacy.no_private_payload(result)
    if not args.check_only:
        target = repo / "results_fresh/legacy_support_differences"
        if target.exists():
            raise FileExistsError("Preserve any previous support-difference export")
        temporary = Path(tempfile.mkdtemp(prefix=".support-export-", dir=target.parent))
        try:
            for relative, data in payloads.items():
                p = temporary / relative;p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(data)
                if legacy.sha(p) != next(row["observed_fresh_artifact_sha256"] for row in files if row["path"] == relative):
                    raise ValueError("Copied aggregate differs from its newly observed identity")
            (temporary / "export_manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            (temporary / "README.md").write_text("# Fresh aggregates with protocol-key differences\n\nThese four newly generated simulation aggregates retain FAIL_SUPPORT from the completed comparison. Independent key diagnosis found the same model/world/group/outcome support and row counts, with different protocol labels for two methods. The original comparator includes protocol in its key and did not issue fresh CSV SHA seals. See reports/legacy_support_key_diagnosis.json from the repository root; that diagnosis does not compare numeric values. This separate export verifies aggregate schemas, unique simulation keys and completed fresh-run provenance, then records new current-file identity hashes. Those hashes do not certify numerical equality or replace the original failed comparison. The primary export and all original certificates remain intact. No person records, chains, draws or raw data are included.\n")
            temporary.rename(target)
        except BaseException:
            shutil.rmtree(temporary);raise
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--finalization-root", type=Path, required=True)
    p.add_argument("--printed-cells", type=Path, required=True)
    p.add_argument("--printed-cells-receipt", type=Path, required=True)
    p.add_argument("--verification", type=Path, required=True)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--report", type=Path)
    args = p.parse_args();value = export(args)
    if args.report:
        args.report.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:v for k,v in value.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
