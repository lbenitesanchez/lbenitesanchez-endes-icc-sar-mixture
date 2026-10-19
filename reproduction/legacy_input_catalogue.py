#!/usr/bin/env python3
"""Build public fingerprints of legacy fit-facing inputs, without row values.

Reads only the five allowlisted model_inputs CSV files, direct.npz and the
coordinate-derived fit_basis.npy. The public geometry copies are comparison
references only; they are never inputs to population generation or fitting.
No evaluation_only, generating basis, response truth or fitted object is read.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np

from legacy_input_audit import CSV_FILES, DIRECT_KEYS, PRIMARY, STRUCTURAL, permitted, sha

FORMAT = "legacy-observed-input-checksums-v1"
ARRAY_SHA_ALGORITHM = "sha256(str(dtype).encode() + json.dumps(shape).encode() + C_contiguous_array_bytes)"


def array_sha(array):
    """Same fingerprint contract as legacy_orientation_adapter.array_sha."""
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def expected_worlds():
    worlds = [(scenario, replication) for scenario in PRIMARY for replication in range(10000, 10200)]
    worlds += [(scenario, replication) for scenario in STRUCTURAL for replication in range(20000, 20100)]
    if len(worlds) != 800 or len(set(worlds)) != 800:
        raise RuntimeError("The fixed legacy cohort must contain all 800 unique populations")
    return worlds


def build(reference_study, output):
    reference_study, output = reference_study.resolve(), output.resolve()
    base = reference_study / "resultados/resultados_comparadores/inputs"
    geometry_dir = output.parent / "legacy_geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    records, geometry_refs = [], {}
    for index, (scenario, replication) in enumerate(expected_worlds(), 1):
        directory = base / scenario / f"rep_{replication:05d}" / "model_inputs"
        record = {"scenario": scenario, "replication": replication,
                  "csv": {name: sha(permitted(directory, name)) for name in CSV_FILES}, "direct": {}}
        direct_path = permitted(directory, "direct.npz")
        with np.load(direct_path, allow_pickle=False) as arrays:
            if not set(arrays.files).issubset(DIRECT_KEYS):
                raise ValueError("Unknown or forbidden direct array; no truth arrays may be read")
            for key in sorted(arrays.files):
                array = arrays[key]
                if array.dtype.hasobject or array.dtype.kind not in "biufc":
                    raise ValueError("Direct fingerprints require numeric arrays without objects")
                record["direct"][key] = {"shape": list(array.shape), "dtype": str(array.dtype),
                                         "sha256": array_sha(array)}
        basis_path = permitted(directory, "fit_basis.npy")
        basis = np.load(basis_path, allow_pickle=False)
        if (basis.ndim != 2 or basis.shape[0] != 80 or not 1 <= basis.shape[1] <= 80
                or basis.dtype.kind != "f" or not np.isfinite(basis).all()):
            raise ValueError("Expected a finite coordinate-derived 80-area fit basis")
        basis_hash = sha(basis_path)
        relative_geometry = "legacy_geometry/" + basis_hash + ".npy"
        target = output.parent / relative_geometry
        if target.exists():
            if sha(target) != basis_hash:
                raise RuntimeError("An existing public geometry reference changed")
        else:
            shutil.copyfile(basis_path, target)
        geometry_refs[relative_geometry] = {"file_sha256": basis_hash,
            "array_sha256": array_sha(basis), "shape": list(basis.shape), "dtype": str(basis.dtype),
            "role": "Coordinate-derived model-input basis; comparison only, never DGP generation or fitting input"}
        record.update(fit_basis_reference=relative_geometry, fit_basis_sha256=basis_hash)
        records.append(record)
        if index % 100 == 0:
            print(json.dumps({"catalogue_worlds": index, "unique_geometry_references": len(geometry_refs)}), flush=True)
    result = {"format": FORMAT, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "world_count": 800, "primary_world_count": 600, "structural_world_count": 200,
        "csv_files": list(CSV_FILES), "csv_hash_count": 4000,
        "direct_array_hash_count": sum(len(record["direct"]) for record in records),
        "direct_keys_allowlist": sorted(DIRECT_KEYS), "direct_array_sha256_algorithm": ARRAY_SHA_ALGORITHM,
        "fit_geometry_comparison": {"atol": 1e-12, "rtol": 1e-12, "compare": "B @ B.T",
            "reference_basis_substituted_into_fresh_inputs": False},
        "worlds": records, "geometry_references": geometry_refs,
        "builder_sha256": sha(Path(__file__)), "published_individual_csv_rows": False,
        "published_direct_response_array_values": False,
        "archive_truth_population_outcomes_generating_basis_or_fitted_objects_read": False,
        "reference_paths_read": "Only allowlisted files under resultados/resultados_comparadores/inputs/*/rep_*/model_inputs",
        "generation_adapter_may_read_this_catalogue": False,
        "purpose": "Check freshly regenerated observed input identity using public fingerprints; geometry files are comparison references only"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"format": FORMAT, "world_count": 800, "csv_hash_count": 4000,
        "direct_array_hash_count": sum(len(record["direct"]) for record in records),
        "geometry_reference_count": len(geometry_refs), "catalogue_sha256": sha(output),
        "individual_record_values_published": False, "old_truth_read": False}, indent=2), flush=True)
    return result


def validate_catalogue(path):
    """Validate the complete public contract before a verifier opens inputs."""
    path = Path(path).resolve()
    result = json.loads(path.read_text())
    geometry_contract = {"atol": 1e-12, "rtol": 1e-12, "compare": "B @ B.T",
                         "reference_basis_substituted_into_fresh_inputs": False}
    if (result.get("format") != FORMAT or result.get("world_count") != 800
            or result.get("csv_hash_count") != 4000 or result.get("csv_files") != list(CSV_FILES)
            or result.get("direct_keys_allowlist") != sorted(DIRECT_KEYS)
            or result.get("direct_array_sha256_algorithm") != ARRAY_SHA_ALGORITHM
            or result.get("fit_geometry_comparison") != geometry_contract):
        raise ValueError("Unknown, partial or changed legacy input catalogue contract")
    keys = [(row["scenario"], row["replication"]) for row in result["worlds"]]
    if len(keys) != 800 or len(set(keys)) != 800 or set(keys) != set(expected_worlds()):
        raise ValueError("The catalogue must contain exactly the 800 frozen populations")
    used_geometry = set()
    for row in result["worlds"]:
        expected_direct = DIRECT_KEYS - {"df_original_proxy"} if row["scenario"] == "binary_informative_mar" else DIRECT_KEYS
        if set(row["csv"]) != set(CSV_FILES) or set(row["direct"]) != expected_direct:
            raise ValueError("Catalogue input names differ from the allowlist")
        if any(not re.fullmatch("[0-9a-f]{64}", value) for value in row["csv"].values()):
            raise ValueError("Invalid CSV SHA-256 fingerprint")
        for entry in row["direct"].values():
            if (set(entry) != {"shape", "dtype", "sha256"}
                    or not re.fullmatch("[0-9a-f]{64}", entry["sha256"])
                    or not isinstance(entry["shape"], list)
                    or any(not isinstance(size, int) or size < 1 for size in entry["shape"])
                    or np.dtype(entry["dtype"]).kind not in "biufc"):
                raise ValueError("Invalid direct-array fingerprint or schema")
        relative = Path(row["fit_basis_reference"])
        if (relative.is_absolute() or ".." in relative.parts or relative.parent != Path("legacy_geometry")
                or relative.suffix != ".npy"):
            raise ValueError("Unsafe geometry comparison reference")
        if relative.as_posix() not in result["geometry_references"]:
            raise ValueError("Geometry reference missing from the contract")
        used_geometry.add(relative.as_posix())
        descriptor = result["geometry_references"][relative.as_posix()]
        reference = path.parent / relative
        if (descriptor["file_sha256"] != row["fit_basis_sha256"]
                or sha(reference) != descriptor["file_sha256"]):
            raise ValueError("Public fit geometry file differs from its fingerprint")
    if used_geometry != set(result["geometry_references"]):
        raise ValueError("Geometry catalogue has unused or missing references")
    if result.get("direct_array_hash_count") != sum(len(row["direct"]) for row in result["worlds"]):
        raise ValueError("Direct-array count differs from the complete cohort")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.reference_study_root, args.output)
    validate_catalogue(args.output)


if __name__ == "__main__":
    main()
