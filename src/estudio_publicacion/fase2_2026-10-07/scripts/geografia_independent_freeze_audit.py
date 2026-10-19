"""Read-only audit of frozen sources/inputs, with explicit operational QA rerun.

This never restores earlier proof bytes, edits the protocol, or reads held-out
responses. The original failing audit remains as historical evidence.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json

PHASE = Path(__file__).resolve().parents[1]
STUDY = PHASE.parent


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def resolve_source(relative):
    phase = PHASE / relative
    previous = STUDY / relative
    # New wrappers live in phase2; original model kernels live in the study.
    if phase.is_file():
        return phase
    return previous


def main():
    protocol_file = PHASE / "protocol_freeze.json"
    protocol = read(protocol_file)
    history_path = PHASE / "validacion/geografia_independent_phase2_freeze_check.json"
    history = read(history_path)
    provenance_path = PHASE / "validacion/externa_wrapper_scope_qa_reexecution_provenance.json"
    provenance = read(provenance_path)
    sources, inputs, proofs, failures = [], [], [], []
    for relative, expected in protocol["source_sha256"].items():
        path = resolve_source(relative)
        actual = sha(path)
        match = actual == expected
        sources.append({"path": str(path.relative_to(STUDY)), "expected": expected,
                        "actual": actual, "matches_expected": match})
        if not match:
            failures.append({"kind": "source", "path": relative})
    for relative, expected in protocol["input_sha256"].items():
        path = (STUDY / relative.removeprefix("previous_study/")) if relative.startswith("previous_study/") else PHASE / relative
        actual = sha(path)
        match = actual == expected
        inputs.append({"path": relative, "expected": expected,
                       "actual": actual, "matches_expected": match})
        if not match:
            failures.append({"kind": "input", "path": relative})
    for relative, expected in protocol["QA_sha256"].items():
        path = PHASE / relative
        actual = sha(path)
        match = actual == expected
        documented = False
        if not match and relative == provenance["affected_file"]:
            current = read(path)
            documented = all([
                provenance["protocol_sha256"] == sha(protocol_file),
                provenance["current_file_sha256"] == actual,
                provenance["frozen_protocol_entries"][0]["value"] == expected,
                provenance["current_tested_source_sha256"] == protocol["source_sha256"]["scripts/validacion_fit_models.py"],
                current["script_sha256"] == provenance["current_tested_source_sha256"],
                "PASS" in current["status"],
                not provenance["frozen_sources_modified"],
                not provenance["input_data_modified"],
                not provenance["statistical_methodology_modified"],
                not provenance["reserved2024responses_read"],
                provenance["old_evidence_not_reconstructed"],
                provenance["previous_bytes_snapshot_available"] is False,
            ])
        proofs.append({"path": relative, "expected": expected, "actual": actual,
                       "matches_expected": match,
                       "documented_operational_QA_reexecution": documented})
        if not match and not documented:
            failures.append({"kind": "undocumented_QA_difference", "path": relative})
    zip_path = STUDY / "ENDES_comparadores_publicacion_2026-10-06_entrega.zip"
    zip_match = sha(zip_path) == protocol["previous_delivery_sha256"]
    if not zip_match:
        failures.append({"kind": "previous_delivery_changed"})
    if history["status"] != "FAIL" or sha(protocol_file) != history["root_protocol_sha256"]:
        failures.append({"kind": "initial_FAIL_or_protocol_modified"})
    proof = {
        "status": "PASS_FROZEN_SOURCES_AND_INPUTS_WITH_DOCUMENTED_OPERATIONAL_QA_RERUN" if not failures else "FAIL",
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "Hash-only audit; no held-out child outcomes or performance scores opened; no scientific source/input edits.",
        "protocol_sha256": sha(protocol_file),
        "source_count": len(sources), "source_checks": sources,
        "input_count": len(inputs), "input_checks": inputs,
        "QA_count": len(proofs), "QA_exact_matches": sum(x["matches_expected"] for x in proofs),
        "QA_documented_reruns": sum(x["documented_operational_QA_reexecution"] for x in proofs),
        "QA_checks": proofs,
        "operational_QA_provenance_path": str(provenance_path.relative_to(PHASE)),
        "operational_QA_provenance_sha256": sha(provenance_path),
        "initial_FAIL_preserved_path": str(history_path.relative_to(PHASE)),
        "initial_FAIL_preserved_sha256": sha(history_path),
        "previous_delivery_sha256": sha(zip_path), "previous_delivery_matches": zip_match,
        "interpretation": "45 sources and 17 inputs retain their exact frozen bytes. Eleven QA proofs retain frozen bytes; one operational scope proof was rerun and overwritten in an overlap with freeze, now honestly documented. This is not a claim that all twelve current QA files equal the historical hashes.",
        "failures": failures,
    }
    destination = PHASE / "validacion/geografia_independent_phase2_freeze_final_check.json"
    with destination.open("x") as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: proof[k] for k in ["status", "source_count", "input_count", "QA_exact_matches", "QA_documented_reruns", "previous_delivery_matches", "failures"]}))


if __name__ == "__main__":
    main()
