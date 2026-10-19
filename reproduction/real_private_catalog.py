"""Verify private CSVs against public checksums when archived records are absent.

The catalogue contains hashes, byte counts, row counts and column names only.
An identical CSV hash also verifies every record and its ordering. No private
CSV is copied to the catalogue or printed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
import real_worker as real
from code_only_sources import historical_source_matches

PRIVATE_PATHS = (
    "datos/private_competitors/households_2019.csv", "datos/private_competitors/children_2019.csv",
    "fase2_2026-10-07/datos/private_external/households_2024.csv",
    "fase2_2026-10-07/datos/private_external/children_2024_training.csv",
    "fase2_2026-10-07/datos/private_external/children_2024_test.csv",
    "fase2_2026-10-07/datos/private_external/children_2024_scoring_only.csv",
    "sikov_2026-10-10/datos/real_2019_full/private/units.csv",
    "sikov_2026-10-10/datos/real_2024_training/private/units.csv")


def build(reference_root, output):
    study, article = real.locations(reference_root)
    entries = {}
    for relative in PRIVATE_PATHS:
        path = study / relative
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        entries[relative] = {"sha256": real.sha(path), "bytes": path.stat().st_size,
            "rows": len(frame), "columns": list(frame.columns), "individual_values_exported": False}
    auxiliary = study / "datos/processed/aux_all_districts.csv"
    result = {"format": "private-csv-checksums-v1", "entries": entries,
        "original_protocol_sha256": real.sha(article / "protocol_freeze.json"),
        "original_addendum_sha256": real.sha(article / "validacion/sikov_published_addendum_freeze.json"),
        "public_aggregate_auxiliary_csv": {"relative_path": "datos/processed/aux_all_districts.csv",
            "sha256": real.sha(auxiliary)},
        "raw_sav_checksums": {relative: {"sha256": real.sha(study / relative), "bytes": (study / relative).stat().st_size}
            for relative in ("datos/raw/ENDES2019/Modulo64/RECH0.SAV", "datos/raw/ENDES2019/Modulo64/RECH1.sav",
                "datos/raw/ENDES2019/Modulo74/RECH6.SAV", "fase2_2026-10-07/datos/private_external_raw/RECH0_2024.SAV",
                "fase2_2026-10-07/datos/private_external_raw/RECH1_2024.SAV", "fase2_2026-10-07/datos/private_external_raw/RECH6_2024.SAV")},
        "builder_sha256": real.sha(__file__), "raw_or_individual_records_exported": False,
        "interpretation": "An absent private reference is verified by exact reconstructed CSV bytes against its archived SHA-256, row count and column order; an absent file is never treated as an unchecked pass."}
    real.write_json(output, result)
    return result


def install(real_module, catalogue_path, reference_root):
    study, article = real.locations(reference_root)
    catalogue = json.loads(Path(catalogue_path).read_text())
    if catalogue["format"] != "private-csv-checksums-v1":
        raise ValueError("Unknown checksum catalogue format")
    if catalogue["original_protocol_sha256"] != real.sha(article / "protocol_freeze.json"):
        raise RuntimeError("Catalogue refers to another original protocol")
    old_compare = real_module.compare_csv
    verified = []

    def compare_csv(new, old):
        old = Path(old)
        if old.is_file():
            return old_compare(new, old)
        relative = old.resolve().relative_to(study).as_posix()
        if relative not in catalogue["entries"]:
            raise FileNotFoundError("No archived private file or authenticated checksum: " + relative)
        item = catalogue["entries"][relative]
        new = Path(new)
        frame = pd.read_csv(new, dtype=str, keep_default_na=False)
        if (real.sha(new) != item["sha256"] or new.stat().st_size != item["bytes"] or
                len(frame) != item["rows"] or list(frame.columns) != item["columns"]):
            raise RuntimeError("Reconstructed private CSV fails public checksum/schema: " + relative)
        result = {"new": str(new), "reference": relative, "rows": len(frame), "columns": len(frame.columns),
            "same_row_order": True, "max_abs_numeric_difference": 0.0, "new_sha256": real.sha(new),
            "reference_sha256": item["sha256"], "exact_bytes": True,
            "verification": "public checksum + byte count + row count + ordered column schema",
            "catalogue_sha256": real.sha(catalogue_path), "private_reference_loaded": False}
        verified.append(result)
        return result

    def source_gate(source_study, source, reference_study):
        # Historical freezes remain in the externally supplied reference tree.
        reference_article = reference_study / source.name
        proof = json.loads((reference_article / "protocol_freeze.json").read_text())
        add = json.loads((reference_article / "validacion/sikov_published_addendum_freeze.json").read_text())
        checked = {}
        for document in (proof, add):
            for block in ("statistical_sources_sha256", "additional_sources_sha256", "base_kernel_sources_sha256"):
                for old, expected in document.get(block, {}).items():
                    target = real.remap(old, proof, source_study)
                    if not historical_source_matches(target, expected):
                        raise RuntimeError("Changed frozen source: " + str(target))
                    checked[str(target.relative_to(source_study))] = expected
            for old, expected in document.get("frozen_inputs_sha256", {}).items():
                target = real.remap(old, proof, reference_study)
                if target.is_file():
                    if real.sha(target) != expected:
                        raise RuntimeError("Changed frozen reference input")
                else:
                    relative = target.relative_to(reference_study).as_posix()
                    item = catalogue["entries"].get(relative)
                    if not item or item["sha256"] != expected:
                        raise RuntimeError("Frozen input missing without matching public private-input catalogue: " + relative)
        return proof, add, checked

    real_module.compare_csv = compare_csv
    real_module.source_gate = source_gate
    return {"catalogue": catalogue, "verified": verified, "sha256": real.sha(catalogue_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.reference_root, args.output)
    print(json.dumps({"private_reference_files": len(result["entries"]), "raw_records_exported": False,
                      "catalogue_sha256": real.sha(args.output)}), flush=True)


if __name__ == "__main__":
    main()
