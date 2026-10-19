#!/usr/bin/env python3
"""Publish local, sanitized legacy aggregates after full fresh finalization.

No fitting, scoring, Git operation or private-tree discovery is performed.
Only 25 named aggregate CSVs, four sealed PNGs and completed aggregate
certificates are eligible. Numerical/schema differences remain failures;
a CSV lacking its comparator's SHA is explicitly omitted, never relabelled.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from export_public_fresh_results import read, sha, sanitize, no_private_payload, validate_csv, gate as modern_gate, CERTIFICATES

HERE = Path(__file__).resolve().parent
STAGES = ("previous800", "cut40", "application2025", "external2024", "cut16")
PHASE = "fase2_2026-10-07"
CSV_PATHS = (
    *["resultados/resultados_comparadores/final_evaluation/" + name for name in (
        "combined_replications.csv", "combined_summary.csv", "primary_publication_table.csv",
        "primary_balanced_interval_score.csv", "structural_summary.csv", "additional_arms_summary.csv",
        "converged_only_summary.csv", "paired_contrasts.csv")],
    *["resultados/resultados_comparadores/spatial_evaluation/" + name for name in
      ("summary.csv", "replications.csv", "paired_contrasts.csv")],
    *[PHASE + "/resultados/design_simulation/final/" + name for name in
      ("population_metrics.csv", "summary.csv", "paired_diagnostics.csv")],
    *[PHASE + "/resultados/externa_final/" + name for name in ("summary.csv", "contrasts_Holm8.csv")],
    PHASE + "/resultados/application2025_summary.csv",
    *[PHASE + "/resultados/" + name for name in (
        "application2025_baseline/predictions.csv", "application2025_designcut8/district_predictions.csv",
        "externa_temporal_2019_baseline/predictions.csv", "externa_temporal_2019_designcut8/district_predictions.csv",
        "externa_temporal_2019_hogg/predictions.csv", "externa_territorial_2024_baseline/predictions.csv",
        "externa_territorial_2024_designcut8/district_predictions.csv", "externa_territorial_2024_hogg/predictions.csv")])
FIGURES = {"Figure_A1": "design_diagnostics", "Figure_A2": "external_contrasts",
    "Figure_A3": "application2025_ICC_capitals", "Figure_S1": "application2025_capitals"}
FINAL_LABELS = {"real_score", "sikov_score", "sikov_figures", "legacy_compare", "legacy_figures",
    "article_coverage", "strict_csv_schema"}
FINAL_STATES = {"NUMERICAL_REPRODUCTION_CONFIRMED_VISUAL_QA_PENDING",
    "COMPLETED_COMPARISON_DIFFERENCES_OR_INCOMPLETE_CHECKS_RETAINED"}
EXPECTED_COUNTS = {"primary_original": 256, "primary_published": 256, "sampled": 64,
    "sampled_original": 64, "sampled_published": 64, "B300": 8, "real_original": 2, "real_published": 2}
IDENTITY_STATUS = "PASS_800_OBSERVED_INPUTS_EXACT_FIT_GEOMETRY_ROUNDOFF"
RELAY_SNAPSHOT_NAMES = {"hogg_tsln.py", "hogg_diagnostics.py", "hogg_design_extension.py",
    "hogg_stage1.stan", "hogg_stage2.stan", "hogg_stage2_design_extension.stan",
    "hogg_stage1", "hogg_stage2", "hogg_stage2_design_extension",
    "hogg_design.md", "hogg_strong_design.md", "hogg_benchmark_batch.py"}


class Pending(RuntimeError):
    """Closure is missing; no aggregate payload should be opened."""


def require_path(path):
    if not path.is_file() or path.is_symlink():
        raise Pending("Missing or linked metadata: " + path.name)
    return path


def contained_file(root, relative, suffixes=(".json", ".csv", ".png", ".pdf")):
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or rel.suffix not in suffixes or "\\" in relative:
        raise ValueError("Unsafe path or forbidden publication type")
    path = root / rel
    require_path(path)
    if not path.resolve().is_relative_to(root):
        raise ValueError("Publication source escapes its certified root")
    if path.stat().st_size > 100_000_000:
        raise ValueError("Publication file exceeds the 100 MB limit")
    return path


def sealed_file(root, relative, digest):
    path = contained_file(root, relative)
    if not re.fullmatch(r"[0-9a-f]{64}", str(digest)) or sha(path) != digest:
        raise ValueError("Changed or unsealed fresh artifact: " + relative)
    return path


def validate_final_header(state):
    gate = state.get("gate", {})
    if (state.get("status") not in FINAL_STATES or not state.get("finished_utc")
            or gate.get("status") != "PASS_ALL_FRESH_COHORTS_AND_CORRECTED_LEGACY_LEDGERS_CLOSED"
            or gate.get("approved_for_final_scoring") is not True or gate.get("pending") != []
            or gate.get("failed") != [] or gate.get("counts") != EXPECTED_COUNTS
            or state.get("certificate_parse_errors") != []):
        raise Pending("Complete global finalization and all fresh cohorts are required")
    if state.get("article_reproduction_confirmed") is not False:
        raise ValueError("Visual/editorial review is not certified by this export")
    if not isinstance(state.get("numerical_reproduction_confirmed"), bool):
        raise ValueError("Final numerical comparison state is missing")
    if state.get("numerical_reproduction_confirmed") is True:
        if not state.get("numerical_checks") or not all(state["numerical_checks"].values()):
            raise ValueError("Numerical success contradicts the recorded failed checks")


def validate_identity(private, strict, catalogue_sha, rule_sha):
    for record in (private, strict):
        if (record.get("status") != IDENTITY_STATUS or record.get("worlds") != 800
                or record.get("csv_hashes_checked") != 4000 or record.get("failures") != []
                or record.get("orientation_rule_sha256") != rule_sha):
            raise Pending("The exact 800-world input identity has not passed")
    if (strict.get("strict_public_catalogue_gate") is not True
            or strict.get("expected_catalogue_sha256") != catalogue_sha
            or strict.get("all_direct_npz_keys_checked_against_constant_allowlist_before_array_loading") is not True
            or strict.get("all_800_orientation_generation_receipts_checked") is not True
            or len(strict.get("world_records", [])) != 800
            or any(record.get("passed") is not True for record in strict["world_records"])):
        raise ValueError("Strict public input gate is missing, incomplete or bound to another catalogue")


def validate_stage(stage, record, expected_commands, identity_sha):
    if (record.get("status") != "ALL_COMMANDS_EXITED_ZERO" or record.get("stage") != stage
            or record.get("commands") != expected_commands
            or record.get("input_identity_certificate_sha256") != identity_sha):
        raise Pending("Legacy stage has not closed successfully: " + stage)


def relay_receipts(legacy, stage_record, original_commands, identity_sha, strict_sha):
    """Accept the documented operational relay without inventing old exits."""
    op = legacy / "previous800_relay";plan_path = require_path(op / "relay_plan.json")
    plan = read(plan_path);gao_path = require_path(op / "adopted_gao_terminal.json");gao = read(gao_path)
    if (plan.get("classification") != "OPERATIONAL_CONTROLLER_HANDOFF_UNCHANGED_MODELS_SEEDS_AND_GATES"
            or plan.get("original_plan_sha256") != sha(legacy / "legacy_commands.json")
            or plan.get("operational_relay_sha256") != sha(HERE / "legacy_previous800_relay.py")
            or plan.get("input_identity") != {"private_identity_sha256": identity_sha, "public_strict_identity_sha256": strict_sha}
            or plan.get("tail_original_indices") != list(range(6, 20))
            or plan.get("Hogg_worker_count_actually_requested") != 4
            or plan.get("new_controller_does_not_claim_original_controller_exit_zero") is not True
            or stage_record.get("relay_plan_sha256") != sha(plan_path)
            or stage_record.get("adopted_gao_terminal_sha256") != sha(gao_path)
            or stage_record.get("original_full_controller_exit_zero_claimed") is not False
            or stage_record.get("actually_executed_new_tail_commands") != 14
            or stage_record.get("adopted_fresh_prefix_commands") != 5
            or stage_record.get("adopted_fresh_Gao_terminal_worker_certificate") is not True
            or stage_record.get("closure_basis") != "NEW_TAIL_ACTUAL_ZERO_EXITS_PLUS_EXPLICIT_FRESH_PREFIX_AND_GAO_R_WORKER_CERTIFICATE_ADOPTION"
            or stage_record.get("source_models_seeds_and_diagnostic_gates_changed") is not False):
        raise ValueError("The fresh operational relay contract changed or claims an unobserved old exit")
    expected_tail = [list(command) for command in original_commands[6:]]
    at = expected_tail[1].index("--workers");expected_tail[1][at + 1] = "4"
    expected_tail[1] += ["--source-snapshot", str(op / "source_snapshot")]
    if plan.get("new_tail_commands") != expected_tail:
        raise ValueError("Relay commands changed beyond worker count and byte-identical source snapshot")
    seals = []
    for index in range(5):
        path = require_path(legacy / "receipts/previous800" / f"{index:03d}.json");record = read(path)
        adopted = plan["adopted_prefix"][index]
        if (record.get("command") != original_commands[index] or record.get("exit_code") != 0
                or adopted.get("original_index") != index or adopted.get("receipt_sha256") != sha(path)
                or adopted.get("exit_code") != 0):
            raise ValueError("Adopted completed fresh prefix differs")
        seals.append(sha(path))
    terminal_path = require_path(legacy / "study/resultados/resultados_comparadores/dispatch_logs/fresh_reproduction_gao/execution.json")
    terminal = read(terminal_path)
    if (gao.get("classification") != "ADOPT_FRESH_GAO_TERMINAL_R_WORKER_EXIT_CERTIFICATE"
            or gao.get("terminal_execution_certificate_sha256") != sha(terminal_path)
            or gao.get("R_worker_exit_codes") != [0, 0] or gao.get("requested_fits") != 1800
            or gao.get("complete_valid_prediction_pairs") != 1800
            or gao.get("original_python_dispatcher_has_terminated") is not True
            or gao.get("original_controller_process_exit_and_command_005_receipt_observed") is not False
            or terminal.get("completed_execution") is not True or terminal.get("worker_exit_codes") != [0, 0]
            or terminal.get("requested_fits") != 1800 or terminal.get("completed_prediction_artifacts") != 1800
            or terminal.get("explicit_failure_artifacts") != 0 or terminal.get("missing_artifacts") != 0):
        raise Pending("The adopted Gao R-worker terminal certificate is incomplete")
    snapshot_path = require_path(op / "source_snapshot/snapshot_manifest.json");snapshot = read(snapshot_path)
    if (plan.get("source_snapshot_sha256") != sha(snapshot_path)
            or set(snapshot.get("source_sha256", {})) != RELAY_SNAPSHOT_NAMES
            or plan.get("frozen_snapshot_source_sha256") != snapshot.get("source_sha256")
            or snapshot.get("new_snapshot_copied_fitted_outputs_or_draws") is not False
            or snapshot.get("fresh_compilation_transfer_certificate_sha256") != sha(legacy / "legacy_v2_compilation_transfer.json")):
        raise ValueError("Relay source/fresh-compiler snapshot provenance changed")
    for name, checksum in snapshot["source_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Unsafe relay snapshot member")
        for path in (op / "source_snapshot" / name, legacy / "study/scripts/competitors" / name):
            if sha(require_path(path)) != checksum:
                raise ValueError("Relay source or freshly compiled bytes changed")
    for index, command in zip(range(6, 20), expected_tail):
        path = require_path(legacy / "receipts/previous800_relay" / f"{index:03d}.json");record = read(path)
        if (record.get("command") != command or record.get("exit_code") != 0 or not record.get("finished_utc")
                or record.get("relay_plan_sha256") != sha(plan_path)
                or record.get("relay_helper_sha256") != plan["operational_relay_sha256"]):
            raise Pending("The new relay tail has not actually exited zero")
        seals.append(sha(path))
    retirement_path = require_path(op / "original_controller_retirement.json");retirement = read(retirement_path)
    if (retirement.get("status") != "ORIGINAL_CONTROLLER_INTERRUPTED_PRESERVED"
            or retirement.get("original_full_controller_completed") is not False
            or retirement.get("original_full_controller_exit_zero_claimed") is not False):
        raise Pending("Original stopped controller retirement must remain explicit")
    return seals, {"execution_basis": "VERSIONED_FRESH_PREFIX_AND_GAO_TERMINAL_ADOPTION_PLUS_NEW_TAIL",
        "adopted_fresh_prefix_commands": 5, "Gao_terminal_R_worker_exits": [0, 0],
        "Gao_valid_prediction_pairs": 1800, "new_tail_actual_zero_exits": 14, "Hogg_workers_requested": 4,
        "original_full_controller_exit_zero_claimed": False, "source_models_seeds_or_diagnostic_gates_changed": False,
        "relay_plan_sha256": sha(plan_path), "adopted_Gao_terminal_sha256": sha(gao_path),
        "original_controller_retirement_sha256": sha(retirement_path)}


def metadata_gate(repo, main, legacy, final):
    state = read(require_path(final / "finalization.json"));validate_final_header(state)
    private_path = require_path(legacy / "legacy_v2_input_identity.json")
    strict_path = require_path(legacy / "legacy_public_strict_input_identity.json")
    private, strict = read(private_path), read(strict_path)
    validate_identity(private, strict, sha(repo / "expected/legacy_input_catalogue.json"), sha(HERE / "legacy_orientation_rule.json"))
    identity_sha = sha(private_path)
    stages = read(require_path(legacy / "legacy_commands.json"));execution = {}
    if set(stages) != set(STAGES):
        raise ValueError("The five legacy execution branches must all be present")
    for stage in STAGES:
        path = require_path(legacy / ("legacy_" + stage + "_execution.json"))
        record = read(path);validate_stage(stage, record, len(stages[stage]), identity_sha)
        name = path.name
        if state["gate"]["corrected_legacy_execution_receipt_sha256"].get(name) != sha(path):
            raise ValueError("Legacy execution receipt changed after global finalization")
        receipt_seals = [];extra = {}
        if stage == "previous800" and record.get("relay_plan_sha256"):
            receipt_seals, extra = relay_receipts(legacy, record, stages[stage], identity_sha, sha(strict_path))
        else:
            for index, command in enumerate(stages[stage]):
                receipt_path = require_path(legacy / "receipts" / stage / f"{index:03d}.json")
                receipt = read(receipt_path)
                if receipt.get("exit_code") != 0 or receipt.get("command") != command or not receipt.get("finished_utc"):
                    raise Pending("An underlying legacy command is not closed: " + stage)
                receipt_seals.append(sha(receipt_path))
        execution[stage] = {"status": record["status"], "commands": record["commands"],
            "stage_receipt_sha256": sha(path), "command_receipt_sha256": receipt_seals, **extra}
    study = legacy / "study"
    generation_path = require_path(study / "resultados/resultados_comparadores/input_validation_final800.json")
    contract_path = require_path(legacy / "legacy_v2_generation_contract.json")
    generation, contract = read(generation_path), read(contract_path)
    if (generation.get("passed") is not True or generation.get("independent_populations") != 800
            or generation.get("orientation_rule_sha256") != private["orientation_rule_sha256"]
            or generation.get("dispatcher_sha256") != sha(HERE / "legacy_v2_worker.py")
            or contract.get("classification") != "INPUT_GENERATION_CORRECTION"
            or strict.get("fresh_input_generation_certificate_sha256") != sha(generation_path)):
        raise ValueError("The corrected fresh generation provenance changed")
    evidence = state["gate"]["corrected_legacy_input_identity_evidence_sha256"]
    for path in (private_path, generation_path, contract_path):
        if evidence.get(path.name) != sha(path):
            raise ValueError("Input identity evidence changed after global finalization")
    setup = read(require_path(legacy / "legacy_initial_setup_completion.json"))
    if (setup.get("complete") is not True or setup.get("800_observed_input_identity_confirmed") is not True
            or setup.get("input_identity_certificate_sha256") != identity_sha):
        raise ValueError("Initial preparation is not bound to the corrected input identity")
    return state, execution, {"private_identity_sha256": identity_sha,
        "strict_public_identity_sha256": sha(strict_path), "worlds": 800, "csv_hashes": 4000,
        "direct_array_fingerprints": 4600, "orientation_rule_sha256": private["orientation_rule_sha256"],
        "public_catalogue_sha256": strict["expected_catalogue_sha256"],
        "maximum_fit_geometry_Gram_difference": strict["maximum_fit_geometry_Gram_difference"],
        "classification": "INPUT_GENERATION_CORRECTION"}


def final_certificates(state, final, main, repo):
    locations = {"real_score": main / "real_scoring_receipt.json",
        "sikov_score": main / "validacion/sikov_score_completion.json",
        "sikov_figures": main / "validacion/sikov_figures_reproduction.json",
        "legacy_compare": final / "legacy_comparison.json", "legacy_figures": final / "legacy_figures.json",
        "article_coverage": final / "article_coverage.json", "strict_csv_schema": final / "strict_csv_schema.json"}
    records = state.get("commands", []);labels = [record.get("label") for record in records]
    if len(labels) != len(set(labels)) or set(labels) not in (FINAL_LABELS, FINAL_LABELS | {"input_claims"}):
        raise Pending("Every planned finalization result and execution receipt is required")
    artifacts, receipts = {}, {}
    if "input_claims" in labels:
        locations["input_claims"] = final / "input_claims.json"
    for record in records:
        label = record["label"];path = require_path(locations[label]);receipt_path = require_path(final / "receipts" / (label + ".json"))
        receipt = read(receipt_path)
        if receipt != record or record.get("result_present") is not True or record.get("result_sha256") != sha(path):
            raise ValueError("Finalizer output/receipt binding changed: " + label)
        if not record.get("finished_utc") or record.get("exit_code") not in ((0, 2) if label in ("legacy_compare", "strict_csv_schema") else (0,)):
            raise Pending("Finalization command did not complete under its reviewed contract: " + label)
        helper_name = Path(record["command"][1]).name
        if sha(repo / "reproduction" / helper_name) != record.get("helper_sha256"):
            raise ValueError("A finalization helper changed after scoring: " + label)
        artifacts[label] = read(path);receipts[label] = receipt
    legacy = artifacts["legacy_compare"]
    if (legacy.get("status") not in ("PASS_ALL_LEGACY_NUMERICAL_TARGETS", "FAIL_DIFFERENCES_RETAINED")
            or legacy.get("pending") != [] or legacy.get("fresh_execution_evidence_present") is not True
            or not set(CSV_PATHS).issubset(legacy.get("checks", {}))):
        raise Pending("The complete legacy comparison is missing")
    coverage = artifacts["article_coverage"]
    if (coverage.get("status") != "FRESH_ARTIFACT_COVERAGE_COMPARISON_COMPLETE_DIFFERENCES_RETAINED"
            or coverage.get("fresh_execution_evidence_present") is not True or coverage.get("pending_execution_evidence") != []
            or len(coverage.get("manuscript_table_checks", {})) != 11 or len(coverage.get("supplement_table_checks", {})) != 11
            or coverage.get("legacy_empirical_claims_missing") != [] or "pending" in coverage.get("input_empirical_prose_checks", {})):
        raise Pending("The full 22-table/document coverage comparison is incomplete")
    strict = artifacts["strict_csv_schema"]
    if (strict.get("status") not in ("PASS_ALL_MATCHING_CSV_SCHEMAS_COHORTS_AND_VALUES", "FAIL_SCHEMA_COHORT_OR_VALUES_RETAINED")
            or strict.get("all_required_score_CSVs_present_on_both_sides") is not True
            or not strict.get("checks")
            or strict.get("fresh_score_gate", {}).get("approved") is not True
            or set(strict.get("matching_relative_paths", [])) != set(strict.get("checks", {}))):
        raise Pending("Required strict-schema comparisons are incomplete")
    coverage_command = receipts["article_coverage"]["command"]
    flag = "--input-claims-audit"
    if flag not in coverage_command:
        raise Pending("The completed input-claim audit is not bound to document coverage")
    claims_path = require_path(Path(coverage_command[coverage_command.index(flag) + 1]))
    claims = read(claims_path)
    if (coverage.get("input_empirical_prose_checks", {}).get("input_claim_audit_sha256") != sha(claims_path)
            or claims.get("calibration", {}).get("fresh_preparation_receipt_sha256") != sha(main / "real_preparation_receipt.json")):
        raise ValueError("Input-claim aggregate audit refers to another preparation")
    if "input_claims" not in artifacts:
        artifacts["input_claims_external"] = claims
    return artifacts, receipts


def validate_printed(report, receipt, coverage_path, repo, source):
    if (receipt.get("exit_code") != 0 or not receipt.get("finished_utc")
            or report.get("status") != "COMPLETE_PRINTED_CELL_COMPARISON_DIFFERENCES_RETAINED"
            or report.get("fresh_execution_evidence_present") is not True
            or report.get("all_mismatches_included") is not True or report.get("table_count") != 22
            or len(report.get("tables", [])) != 22 or report.get("total_document_cells") != 1488
            or report.get("private_records_or_truth_read_or_exported") is not False
            or report.get("article_reproduction_confirmed") is not False
            or report.get("models_fitted_by_this_helper") is not False
            or report.get("coverage_certificate_sha256") != sha(coverage_path)
            or report.get("article_sha256") != sha(repo / "docs/Articulo_IAOS.docx")
            or report.get("supplement_sha256") != sha(repo / "docs/Suplemento_IAOS.docx")
            or report.get("formatting_source_sha256") != sha(source / "sikov_2026-10-10/scripts/report_documents.py")
            or report.get("helper_sha256") != sha(HERE / "article_printed_cells.py")
            or receipt.get("helper_sha256") != report.get("helper_sha256")):
        raise Pending("A sealed fresh complete 22-table printed-cell payload is required")
    if sum(row.get("mismatch_count", -1) for row in report["tables"]) != report.get("total_different_document_cells"):
        raise ValueError("Printed-cell discrepancy totals are inconsistent")
    for row in report["tables"]:
        if row.get("mismatch_count") != len(row.get("all_mismatches", [])):
            raise ValueError("Not all printed-cell discrepancies were retained")


def csv_export_decision(check):
    if check.get("status") not in ("PASS", "FAIL_NUMERICAL_TARGET"):
        return "OMITTED_COMPARATOR_SCHEMA_SUPPORT_OR_PARSE_FAILURE"
    if not re.fullmatch(r"[0-9a-f]{64}", str(check.get("fresh_sha256", ""))):
        return "OMITTED_NO_COMPARATOR_SHA256_SEAL"
    return None


def plan(args, cfg):
    legacy = Path(cfg["legacy_run_root"]).resolve();final = args.finalization_root.resolve()
    pending = []
    for path in (final / "finalization.json", *[legacy / f"legacy_{stage}_execution.json" for stage in STAGES]):
        if not path.is_file():pending.append(path.name)
        elif path.name == "finalization.json":
            try:validate_final_header(read(path))
            except (Pending, ValueError):pending.append("global_finalization_not_complete")
        elif read(path).get("status") != "ALL_COMMANDS_EXITED_ZERO":pending.append(path.name + ":unfinished")
    for label, path in (("printed_cells", args.printed_cells), ("printed_cells_receipt", args.printed_cells_receipt)):
        if path is None or not path.is_file():pending.append(label)
    return {"status": "PENDING_FRESH_LEGACY_PUBLICATION" if pending else "CLOSURE_METADATA_PRESENT_REQUIRES_FULL_ALLOWLIST_CHECK",
        "pending": pending, "planned_aggregate_CSVs": 25, "planned_sealed_legacy_PNGs": 4,
        "planned_complete_printed_tables": 22, "aggregate_or_figure_or_raw_payloads_read": False,
        "files_copied": 0, "GitHub_commit_push_or_staging_performed": False,
        "article_reproduction_confirmed": False}


def export(args, cfg):
    repo = Path(cfg["repo"]).resolve();source = Path(cfg["source_root"]).resolve()
    main = Path(cfg["sikov_run_root"]).resolve();legacy = Path(cfg["legacy_run_root"]).resolve()
    final = args.finalization_root.resolve();reference = Path(cfg["reference_root"]).resolve();study = legacy / "study"
    if any(path.is_relative_to(repo) for path in (main, legacy, final)):
        raise ValueError("Fresh private computation must remain outside the curated checkout")
    state, execution, identity = metadata_gate(repo, main, legacy, final)
    lock_path, lock = modern_gate(main, source)
    if len(state["gate"]["fresh_completion_sha256"]) != sum(EXPECTED_COUNTS.values()):
        raise ValueError("The finalizer completion-seal universe is incomplete")
    for relative, checksum in state["gate"]["fresh_completion_sha256"].items():
        sealed_file(main, relative, checksum)
    preparation = read(require_path(legacy / "legacy_preparation.json"))
    for relative, checksum in preparation["source_sha256"].items():
        # Source files can include R/Stan and frozen protocols. Hashing these
        # allowlisted prepared/public files does not open any record payload.
        path = study / relative
        if not path.resolve().is_relative_to(study) or path.is_symlink() or sha(require_path(path)) != checksum:
            raise ValueError("Prepared immutable legacy source changed")
        public_root = source if (source / relative).is_file() else repo / "expected/reference_study"
        public = public_root / relative
        if not public.resolve().is_relative_to(public_root) or public.is_symlink() or sha(require_path(public)) != checksum:
            raise ValueError("Prepared legacy source/reference metadata differs from its curated public copy")
    artifacts, receipts = final_certificates(state, final, main, repo)
    lock_sha = sha(lock_path)
    for label, (_, key, status, allowlist) in CERTIFICATES.items():
        artifact = artifacts[{"main": "sikov_score", "real": "real_score", "figures": "sikov_figures"}[label]]
        field = {"main": "joint_fresh_prediction_lock_sha256", "real": "complete_fresh_prediction_lock_sha256",
            "figures": "new_prediction_lock_sha256"}[label]
        if artifact.get("status") != status or artifact.get(field) != lock_sha or set(artifact.get(key, {})) != set(allowlist):
            raise ValueError("A modern result is not bound to the sealed fresh cohort")
        for relative, checksum in artifact[key].items():
            sealed_file(main, relative, checksum)
    if args.printed_cells is None or args.printed_cells_receipt is None:
        raise Pending("Specify completed printed-cell report and its successful receipt")
    report_path = require_path(args.printed_cells);receipt_path = require_path(args.printed_cells_receipt)
    printed, printed_receipt = read(report_path), read(receipt_path)
    if printed_receipt.get("result_sha256") != sha(report_path):
        raise ValueError("Printed-cell report differs from its execution receipt")
    validate_printed(printed, printed_receipt, final / "article_coverage.json", repo, source)
    roots = (("PUBLIC_SOURCE", source), ("LEGACY_RELATIVE", legacy), ("MODERN_RELATIVE", main),
        ("FINALIZATION_RELATIVE", final), ("REFERENCE_RELATIVE", reference))
    payloads = {};files = [];omissions = []
    def public_json(relative, value):
        value = sanitize(value, roots);no_private_payload(value)
        if isinstance(value, dict) and value.get("article_reproduction_confirmed") is True:
            # The legacy comparator's field denotes only its legacy targets.
            # Preserve that source value while preventing a partial-scope
            # certificate from claiming full manuscript reproduction.
            value["original_partial_scope_article_reproduction_flag"] = True
            value["article_reproduction_confirmed"] = False
            value["public_scope_disclosure"] = "Partial numerical target comparison; full article and visual review are governed by global finalization."
        payloads[relative] = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    for relative in CSV_PATHS:
        check = artifacts["legacy_compare"]["checks"][relative];reason = csv_export_decision(check)
        if reason:
            omissions.append({"path": relative, "reason": reason, "comparison_status": check.get("status"),
                "file_published": False});continue
        path = sealed_file(study, relative, check["fresh_sha256"]);data = path.read_bytes()
        schema = validate_csv(data, relative)
        if any(column.lower() in {"individual_id", "unit_id", "participant_id", "record_id"} for column in schema["columns"]):
            raise ValueError("Record-level identifier in an aggregate CSV")
        payloads[relative] = data
        files.append({"path": relative, "comparator_source_sha256": check["fresh_sha256"],
            "public_sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), **schema})
    figures = artifacts["legacy_figures"]
    if (figures.get("status") != "FOUR_FRESH_LEGACY_FIGURES_CREATED_VISUAL_QA_PENDING"
            or figures.get("model_fits_dispatched") is not False or figures.get("old_predictions_substituted") is not False
            or len(figures.get("figures", [])) != 4 or set(item.get("figure") for item in figures["figures"]) != set(FIGURES)
            or figures.get("figure_source_sha256") != preparation["source_sha256"].get(PHASE + "/scripts/phase2_figures.py")):
        raise Pending("All four fresh, sealed legacy PNGs are required")
    for figure in figures["figures"]:
        label = figure["figure"];relative = PHASE + "/figuras/" + FIGURES[label] + ".png"
        actual = sealed_file(study, relative, figure["sha256"])
        if Path(figure["fresh_path"]).resolve() != actual.resolve():
            raise ValueError("Figure seal refers to another fresh source")
        exported = sealed_file(final, "regenerated_legacy_figures/" + label + ".png", figure["sha256"])
        data = exported.read_bytes()
        if not data.startswith(b"\x89PNG\r\n\x1a\n") or any(marker in data for marker in (b"/Users/", b"/.codex/", b"/private/")):
            raise ValueError("Invalid PNG or embedded private path")
        destination = "figures/" + label + ".png";payloads[destination] = data
        files.append({"path": destination, "figure_certificate_sha256": figure["sha256"],
            "public_sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    for label, artifact in artifacts.items():
        public_json("certificates/" + label + ".json", artifact)
        if label in receipts:
            public_json("certificates/" + label + "_execution.json", receipts[label])
    public_json("certificates/finalization.json", state)
    public_json("certificates/input_identity_summary.json", identity)
    public_json("certificates/legacy_execution_summary.json", execution)
    public_json("comparisons/article_printed_cells_complete.json", printed)
    public_json("certificates/article_printed_cells_execution.json", printed_receipt)
    numeric = bool(state["numerical_reproduction_confirmed"])
    summary = {"format": "iaos-public-fresh-legacy-export-v1",
        "status": "CHECK_ONLY_LEGACY_PUBLICATION_VALIDATED" if args.check_only else
            "PUBLIC_LEGACY_COMPARISON_EXPORTED_DIFFERENCES_RETAINED" if not numeric or omissions else
            "PUBLIC_LEGACY_NUMERIC_COMPARISON_EXPORTED_VISUAL_QA_PENDING",
        "complete_global_finalization_sha256": sha(final / "finalization.json"),
        "complete_printed_cells_sha256": sha(report_path), "modern_prediction_lock_sha256": sha(lock_path),
        "legacy_csvs_allowlisted": 25, "legacy_csvs_exported": 25 - len(omissions), "csv_omissions": omissions,
        "all_25_comparator_sealed_csvs_exported": not omissions, "legacy_figures_exported": 4,
        "legacy_SVGs_exported": False, "document_tables_compared": 22,
        "document_cells": printed["total_document_cells"], "document_cell_differences": printed["total_different_document_cells"],
        "finalizer_numerical_reproduction_confirmed": numeric,
        "numerical_reproduction_confirmed": numeric and not omissions,
        "failed_global_comparison_checks_retained": state["failed_checks"], "visual_review_complete": False,
        "article_reproduction_confirmed": False, "private_records_raw_chains_draws_Stan_inputs_exported": False,
        "private_paths_and_commands_removed": True, "GitHub_commit_push_staging_or_ZIP_performed": False,
        "helper_sha256": sha(__file__), "privacy_helper_sha256": sha(HERE / "export_public_fresh_results.py"),
        "original_finalization_result_sha256": {label: receipt["result_sha256"] for label, receipt in receipts.items()},
        "public_artifacts_sha256": {path: hashlib.sha256(data).hexdigest() for path, data in sorted(payloads.items())},
        "files": files}
    target = repo / "results_fresh/legacy"
    if not args.check_only:
        if target.exists():raise FileExistsError("Preserve the previous legacy export instead of overwriting it")
        target.parent.mkdir(exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".legacy-export-", dir=target.parent))
        try:
            for relative, data in sorted(payloads.items()):
                path = temporary / relative;path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(data)
                if sha(path) != summary["public_artifacts_sha256"][relative]:
                    raise RuntimeError("Public payload changed during export")
            (temporary / "export_manifest.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            (temporary / "README.md").write_text("# Fresh legacy comparison\n\nThis local export retains every completed comparison status and all 22 document-table discrepancies. CSVs without comparator SHA seals or with schema/support/parse failures are explicitly omitted in export_manifest.json. The four PNGs are tied to the final figure certificate; SVGs are excluded. No person/household records, chains, draws, raw files or private machine paths are published.\n\nNumerical differences and visual-review status remain explicit. Export completion does not certify article reproduction, author approval or journal submission.\n")
            temporary.rename(target)
        except BaseException:
            shutil.rmtree(temporary);raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--finalization-root", type=Path, required=True)
    parser.add_argument("--printed-cells", type=Path)
    parser.add_argument("--printed-cells-receipt", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args();cfg = read(args.config)
    try:
        result = plan(args, cfg) if args.plan_only else export(args, cfg)
    except Pending as error:
        result = {"status": "BLOCKED_INCOMPLETE_LEGACY_FINALIZATION", "reason": str(error),
            "files_copied": 0, "article_reproduction_confirmed": False}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2));raise SystemExit(2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("files", "public_artifacts_sha256")}, indent=2))


if __name__ == "__main__":
    main()
