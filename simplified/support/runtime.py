"""English table/figure interface with verified external inputs and outputs.

The original numerical adapters run unchanged in a temporary external view.
Verified sources and derived inputs are copied without symbolic links. No model
is refitted, and no generated artifact is written into the repository.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.resolve()
MARKER = ".iaos_simplified_output.json"
MARKER_KIND = "endes-icc-sar-mixture:simplified:english-aggregate-regeneration-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relative(value):
    if not isinstance(value, str) or "\\" in value or any(x in ("", ".", "..") for x in value.split("/")):
        raise ValueError("Invalid relative path: " + str(value))
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError("A relative path is required: " + value)
    return path


def _file(root, relative):
    root = Path(root).resolve()
    path = root / _relative(relative)
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Missing, linked, or out-of-tree file: " + relative)
    return path


def _external(path, label):
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(REPO) or REPO.is_relative_to(path):
        raise ValueError(label + " must be outside the repository")
    return path


def _contract():
    manifest = _read(ROOT / "input_hashes.json")
    if set(manifest) != {"sources_sha256", "external_inputs_sha256"}:
        raise ValueError("Unknown source/input contract")
    if len(manifest["sources_sha256"]) != 18 or len(manifest["external_inputs_sha256"]) != 35:
        raise ValueError("The contract must contain 18 sources and 35 external inputs")
    for collection in manifest.values():
        for relative, digest in collection.items():
            _relative(relative)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid SHA-256 digest: " + relative)
    if any(not p.startswith(("datos/", "referencias/")) for p in manifest["external_inputs_sha256"]):
        raise ValueError("Archived external inputs must be under datos/ or referencias/")
    return manifest


def verify_sources(required=None):
    """Check repository sources without reading inputs or creating outputs."""
    contract = _contract()["sources_sha256"]
    names = sorted(contract) if required is None else list(required)
    for relative in names:
        if relative not in contract:
            raise ValueError("Unregistered source: " + relative)
        if sha(_file(REPO, relative)) != contract[relative]:
            raise RuntimeError("Original source has changed: " + relative)
    return {"status": "PASS", "verified_sources": len(names)}


def verify_inputs(input_root):
    """Verify 18 sources and 35 external CSVs without writing files."""
    input_root = _external(input_root, "--data-root")
    if not input_root.is_dir():
        raise ValueError("--data-root must identify an existing directory")
    sources = verify_sources()
    inputs = _contract()["external_inputs_sha256"]
    for relative, digest in sorted(inputs.items()):
        if sha(_file(input_root, relative)) != digest:
            raise RuntimeError("External input has changed: " + relative)
    return {"status": "PASS", "verified_sources": sources["verified_sources"],
            "verified_external_inputs": len(inputs), "model_fits_run": False}


def _output_root(data_root, output_root):
    data_root = _external(data_root, "--data-root")
    output_root = _external(output_root, "--output-root")
    if output_root.is_relative_to(data_root) or data_root.is_relative_to(output_root):
        raise ValueError("Input and output directories must be separate and must not overlap")
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("--output-root must identify a directory")
    if output_root.exists() and any(path.is_symlink() for path in output_root.rglob("*")):
        raise ValueError("The output directory contains symbolic links; use a directory without links")
    expected = {"kind": MARKER_KIND, "inputs_manifest_sha256": sha(ROOT / "input_hashes.json")}
    if output_root.exists() and any(output_root.iterdir()):
        marker = output_root / MARKER
        if not marker.is_file() or marker.is_symlink() or _read(marker) != expected:
            raise ValueError("The output directory contains unrelated files; use a new directory")
    return output_root, expected


@contextmanager
def _view(data_root, output_root):
    """Copy the 53 verified files into a temporary view without links."""
    contract = _contract()
    with tempfile.TemporaryDirectory(prefix=".iaos_view_", dir=output_root) as temporary:
        view = Path(temporary)
        count = 0
        for base, entries, prefix in (
            (REPO, contract["sources_sha256"], "soporte/originales"),
            (data_root, contract["external_inputs_sha256"], ""),
        ):
            for relative, digest in sorted(entries.items()):
                source = _file(base, relative)
                if sha(source) != digest:
                    raise RuntimeError("A file changed while preparing the view: " + relative)
                destination = view / prefix / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if destination.is_symlink() or sha(destination) != digest:
                    raise RuntimeError("Temporary copy failed its identity check: " + relative)
                count += 1
        if count != 53:
            raise RuntimeError("Incomplete temporary input view")
        yield view


def _csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.reader(stream))


def artifacts():
    """Load the English interface map; legacy IDs remain internal to adapters."""
    records = _read(ROOT / "artifacts.json")
    if len(records) != 20 or len({x["id"] for x in records}) != 20:
        raise ValueError("The current proposal requires 20 artifact entries")
    if len({x["legacy_id"] for x in records}) != 20:
        raise ValueError("Legacy adapter identifiers must be unique")
    for record in records:
        _relative(record["folder"])
        if record["kind"] not in ("Table", "Figure"):
            raise ValueError("Unknown artifact type")
        if record["kind"] == "Table":
            _relative(record["reference_csv"])
    return records


def _english_filename(name):
    return name.replace("Tabla_", "Table_").replace("Figura_", "Figure_").replace("_comprobante", "_provenance")


def _english_evidence(evidence, record, destination):
    """Translate interface metadata while retaining actual archived input paths."""
    value = dict(evidence)
    if "object_id" in value:
        value["artifact_id"] = record["id"]
        del value["object_id"]
    if "output" in value:
        value["output"] = str(destination)
    if "output_files" in value:
        value["output_files"] = {_english_filename(k): v for k, v in value["output_files"].items()}
    if "provenance_scope" in value:
        value["provenance_scope"] = "Plot regeneration from certified aggregates; model fits and bootstrap are not rerun."
    return value


def _generate(view, record, output_root):
    out = output_root / record["folder"] / "output"
    if not out.resolve().is_relative_to(output_root):
        raise ValueError("Artifact output path escapes the designated directory")
    out.mkdir(parents=True, exist_ok=True)
    if record["kind"] == "Table":
        from soporte import table_adapter
        evidence = table_adapter.generate(view, record["legacy_id"], out)
        old = out / ("Tabla_" + record["number"] + ".csv")
        destination = out / ("Table_" + record["number"] + ".csv")
        actual, expected = _csv(old), _csv(view / record["reference_csv"])
        if actual != expected:
            raise RuntimeError("Table differs from its external reference: " + record["id"])
        old.replace(destination)
        comparison = {"identical_cells": sum(map(len, expected)), "rows": len(expected),
                      "reference_used_only_for_verification": True}
    else:
        from soporte import figure_adapter
        evidence = figure_adapter.generate(view, record["legacy_id"], out)
        if evidence["numeric_artist_fingerprint"]["sha256"] != record["expected_numeric_artist_sha256"]:
            raise RuntimeError("Plotted values differ from the registered figure: " + record["id"])
        for extension in ("png", "svg"):
            old = out / ("Figura_" + record["number"] + "." + extension)
            old.replace(out / ("Figure_" + record["number"] + "." + extension))
        destination = out / ("Figure_" + record["number"] + ".png")
        comparison = {"plotted_values_identical_to_current_edition": True,
                      "png_byte_identical": sha(destination) == record["original_sha256"],
                      "original_image_sha256": record["original_sha256"]}
    evidence = _english_evidence(evidence, record, destination)
    if record["kind"] == "Figure":
        old_provenance = out / ("Figura_" + record["number"] + "_comprobante.json")
        old_provenance.unlink()
        (out / ("Figure_" + record["number"] + "_provenance.json")).write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    result = {"artifact_id": record["id"], "status": "PASS", "scope": "regeneration_from_derived_inputs",
              "model_refits_run": False, "original_sources_modified": False,
              "inputs_distributed_in_git": False, "temporary_view_without_links": True,
              "output": str(destination.relative_to(output_root)), "comparison": comparison,
              "details": evidence}
    (out / "verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(record["id"] + ": PASS -> " + str(destination.relative_to(output_root)))
    return result


def regenerate(artifact_ids, data_root, output_root):
    """Regenerate selected artifacts with external input and output directories."""
    records = artifacts()
    aliases = {x["legacy_id"]: x["id"] for x in records}
    requested = [aliases.get(identifier, identifier) for identifier in artifact_ids]
    selected = [x for x in records if x["id"] in set(requested)]
    if not selected or len(selected) != len(set(requested)):
        raise ValueError("Unknown or empty artifact selection")
    data_root = _external(data_root, "--data-root")
    output_root, marker = _output_root(data_root, output_root)
    checks = verify_inputs(data_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / MARKER).write_text(json.dumps(marker, indent=2) + "\n")
    with _view(data_root, output_root) as view:
        results = [_generate(view, record, output_root) for record in selected]
    report = {"status": "PASS", "input_verification": checks, "artifact_count": len(results),
              "model_refits_run": False, "original_sources_modified": False,
              "temporary_view_removed": True, "artifacts": results}
    (output_root / "EXECUTION_VERIFICATION.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def _error_message(error):
    replacements = {
        "El cambio de texto alteró los valores dibujados": "Text changes altered the plotted values",
        "Figura desconocida": "Unknown figure",
    }
    message = str(error)
    for old, new in replacements.items():
        message = message.replace(old, new)
    return message


def main(artifact_id=None):
    records = artifacts()
    aliases = {x["legacy_id"]: x["id"] for x in records}
    aliases.update({"articulo": "article", "suplemento": "supplement"})
    normalize = lambda value: aliases.get(value, value)
    parser = argparse.ArgumentParser(description="Regenerate tables and figures from verified external inputs; no model refitting.")
    parser.add_argument("--data-root", type=Path, default=os.environ.get("IAOS_SIMPLE_DATA_ROOT"),
                        help="External directory containing datos/ and referencias/, or IAOS_SIMPLE_DATA_ROOT")
    parser.add_argument("--output-root", type=Path, default=os.environ.get("IAOS_SIMPLE_OUTPUT_ROOT"),
                        help="New or managed external output directory, or IAOS_SIMPLE_OUTPUT_ROOT")
    if artifact_id is None:
        choices = ["article", "supplement"] + [x["id"] for x in records]
        parser.add_argument("--only", type=normalize, choices=choices)
        parser.add_argument("--solo", type=normalize, choices=choices, dest="only", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.data_root is None or args.output_root is None:
        parser.error("Provide --data-root and --output-root, or their environment variables")
    if artifact_id is not None:
        ids = [normalize(artifact_id)]
    else:
        ids = [x["id"] for x in records if not args.only or x["id"] == args.only
               or x["folder"].startswith(args.only + "/")]
    try:
        report = regenerate(ids, args.data_root, args.output_root)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, "Error: " + _error_message(error) + "\n")
    print(f"Completed: {report['artifact_count']} verified artifacts; no model refitting.")


if __name__ == "__main__":
    main()
