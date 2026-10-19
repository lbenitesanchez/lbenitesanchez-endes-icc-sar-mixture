"""Build an external reference tree from public aggregates plus local SAV files.

Never creates placeholder private files. Saved private reference CSVs remain
absent; real_private_catalog verifies their freshly generated counterparts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import real_worker as real


def materialize(args):
    source_study, source = real.locations(args.public_study_root)
    expected = args.expected_study_root.resolve()
    target = args.reference_root.resolve()
    if target.exists() or target.is_relative_to(source_study):
        raise ValueError("Use a new reference directory outside the public source checkout")
    raw = args.raw_study_root.resolve()
    catalogue = json.loads(args.catalogue.read_text())
    if catalogue["original_protocol_sha256"] != real.sha(source / "protocol_freeze.json"):
        raise RuntimeError("Public checksum catalogue refers to a different protocol")
    copied, linked = {}, {}
    for tree in (source_study, expected):
        for path in sorted(tree.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(tree)
            if any(part.lower().startswith("private") for part in relative.parts) or path.suffix.lower() in (".sav", ".rds", ".pyc"):
                raise RuntimeError("A private/raw file is present in the claimed public input tree: " + str(relative))
            output = target / relative
            if output.exists() and real.sha(output) != real.sha(path):
                raise RuntimeError("Public source/expected reference conflict: " + str(relative))
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, output)
            copied[str(relative)] = real.sha(output)
    for relative, entry in catalogue["raw_sav_checksums"].items():
        original = raw / relative
        if real.sha(original) != entry["sha256"] or original.stat().st_size != entry["bytes"]:
            raise RuntimeError("Raw SAV does not match its documented source: " + relative)
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.symlink_to(original)
        linked[relative] = {"external_source": str(original), "sha256": real.sha(original)}
    for relative in catalogue["entries"]:
        if (target / relative).exists():
            raise RuntimeError("A private reference file was unexpectedly materialized")
    real.write_json(target / "reference_materialization_receipt.json", {
        "status": "PUBLIC_REFERENCE_WITH_EXTERNAL_RAW_INPUTS", "public_file_sha256": copied,
        "raw_external_symlink_sha256": linked, "private_reference_files_materialized": False,
        "public_private_input_catalogue_sha256": real.sha(args.catalogue),
        "operational_adapter_sha256": real.sha(__file__)})
    print(json.dumps({"public_files": len(copied), "raw_files_linked_externally": len(linked),
                      "private_reference_files_materialized": False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public-study-root", "expected-study-root", "raw-study-root", "reference-root", "catalogue"):
        parser.add_argument("--" + name, type=Path, required=True)
    materialize(parser.parse_args())


if __name__ == "__main__":
    main()
