#!/usr/bin/env python3
"""Strict public catalogue gate for a completely regenerated legacy v2 cohort.

This independent wrapper does not change a running generator or its receipts.
It validates the public allowlist and catalogue, checks every fresh NPZ key
before any array is opened, then executes the frozen checksum comparator.
Expected geometry is used only in comparison, never in fresh model inputs.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import numpy as np

from legacy_input_audit import DIRECT_KEYS, permitted, sha
from legacy_input_catalogue import validate_catalogue


def verify(study_root, input_catalogue, output):
    study_root, input_catalogue, output = study_root.resolve(), input_catalogue.resolve(), output.resolve()
    repository = Path(__file__).resolve().parent.parent
    if study_root.is_relative_to(repository) or output.is_relative_to(repository):
        raise ValueError("Fresh model inputs and gate receipts must remain outside the public checkout")
    if output.exists():
        raise FileExistsError("Preserve the previous verification receipt; choose a new output path")
    catalogue_sha = sha(input_catalogue)
    catalogue = validate_catalogue(input_catalogue)
    if set(catalogue["direct_keys_allowlist"]) != DIRECT_KEYS:
        raise ValueError("The public direct-array allowlist differs from the constant scientific contract")
    base = study_root / "resultados/resultados_comparadores"
    generation_path = base / "input_validation_final800.json"
    generation = json.loads(generation_path.read_text())
    orientation_sha = sha(Path(__file__).with_name("legacy_orientation_rule.json"))
    if (generation.get("passed") is not True or generation.get("independent_populations") != 800
            or generation.get("orientation_rule_sha256") != orientation_sha
            or generation.get("input_generation_correction") != "Unspecified DGP eigenvector signs fixed by public immutable geometric anchors"):
        raise ValueError("A complete corrected fresh 800-world generation certificate is required")
    direct_hashes, orientation_hashes = {}, {}
    for world in catalogue["worlds"]:
        identifier = world["scenario"] + "/rep_" + f"{world['replication']:05d}"
        directory = base / "inputs" / identifier / "model_inputs"
        direct = permitted(directory, "direct.npz")
        with np.load(direct, allow_pickle=False) as arrays:
            # Reading .files inspects archive names without opening payload arrays.
            keys = set(arrays.files)
            if not keys.issubset(DIRECT_KEYS) or keys != set(world["direct"]):
                raise ValueError("Unknown, missing or truth-like direct array key: " + identifier)
        orientation_receipt = directory.parent / "orientation_generation_receipt.json"
        orientation = json.loads(orientation_receipt.read_text())
        if (orientation.get("status") != "PUBLIC_FIXED_ANCHOR_DGP_ORIENTATION_APPLIED"
                or orientation.get("historical_geometry_array_read_in_production") is not False
                or orientation.get("historical_truth_responses_or_fits_loaded") is not False):
            raise ValueError("Missing corrected fresh geometry isolation proof: " + identifier)
        direct_hashes[identifier] = sha(direct)
        orientation_hashes[identifier] = sha(orientation_receipt)
    from legacy_v2_worker import verify_catalogue_identity
    raw_comparison = output.with_name(output.stem + "_raw_comparison.json")
    if raw_comparison.exists():
        raise FileExistsError("Preserve the previous raw comparison; choose a new output path")
    verify_catalogue_identity(study_root, input_catalogue, raw_comparison)
    result = json.loads(raw_comparison.read_text())
    if (result.get("status") != "PASS_800_OBSERVED_INPUTS_EXACT_FIT_GEOMETRY_ROUNDOFF"
            or result.get("worlds") != 800 or result.get("csv_hashes_checked") != 4000
            or result.get("failures") != [] or result.get("expected_catalogue_sha256") != catalogue_sha
            or sha(input_catalogue) != catalogue_sha):
        raise RuntimeError("The public full-cohort identity certificate or catalogue binding failed")
    result.update({"strict_public_catalogue_gate": True,
        "all_direct_npz_keys_checked_against_constant_allowlist_before_array_loading": True,
        "all_800_orientation_generation_receipts_checked": True,
        "fresh_input_generation_certificate_sha256": sha(generation_path),
        "fresh_direct_npz_container_sha256_by_world": direct_hashes,
        "fresh_orientation_receipt_sha256_by_world": orientation_hashes,
        "strict_gate_adapter_sha256": sha(Path(__file__)),
        "retained_raw_comparison_sha256": sha(raw_comparison),
        "catalogue_builder_and_validator_sha256": sha(Path(__file__).with_name("legacy_input_catalogue.py")),
        "strict_gate_finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "article_result_reproduction_verified": False})
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "strict_public_catalogue_gate": True,
                     "worlds": 800, "CSV_checksums": 4000,
                     "expected_catalogue_sha256": catalogue_sha,
                     "article_result_reproduction_verified": False}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--input-catalogue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.study_root, args.input_catalogue, args.output)


if __name__ == "__main__":
    main()
