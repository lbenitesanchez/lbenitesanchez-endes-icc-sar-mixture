"""Prepare real ENDES data without author workbooks or archived private CSVs.

The author auxiliary workbook block in the original R parser is replaced by a
verified public aggregate CSV. Every SAV parsing, endpoint, sampling design and
variance operation remains byte-for-byte unchanged in that R runtime copy.
Private reference CSVs are checked against the public SHA/schema catalogue.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import real_worker as real
import real_private_catalog as catalogue_module


def prepare(args):
    catalogue_state = catalogue_module.install(real, args.catalogue, args.reference_root)
    catalogue = catalogue_state["catalogue"]
    study, reference = real.locations(args.reference_root)
    public_aux = args.public_aux_csv or study / catalogue["public_aggregate_auxiliary_csv"]["relative_path"]
    if real.sha(public_aux) != catalogue["public_aggregate_auxiliary_csv"]["sha256"]:
        raise RuntimeError("Public auxiliary CSV differs from the archived input")
    for relative, expected in catalogue["raw_sav_checksums"].items():
        path = study / relative
        if real.sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise RuntimeError("Official raw SAV differs from the documented version: " + relative)
    original_adapted_r = real.adapted_r
    python_source = inspect.getsource(real.prepare)
    start = python_source.index("    auxroot = args.author_aux_root\n")
    end = python_source.index("    first_source =", start)
    # This is an operational dependency normalization only. The kernel's SAV
    # parser and all downstream statistical source files remain immutable.
    normalized_python = python_source[:start] + "    auxroot = args.public_aux_csv.parent\n    workbook_names = ()\n" + python_source[end:]
    args.public_aux_csv = Path(public_aux).resolve()
    args.author_aux_root = None
    normalized_path = args.run_root.resolve().parent / "real_private_study/scripts/portable_prepare_normalized.py"
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_path.write_text(normalized_python)

    def portable_r(source, output, replacements, provenance):
        original_adapted_r(source, output, replacements, provenance)
        if Path(source).name != "01_prepare_data.R":
            return
        text = Path(output).read_text()
        beginning = text.index("# The auxiliary workbook lacks explicit extraction provenance.")
        ending = text.index("summaries <- list()", beginning)
        original_block = text[beginning:ending]
        replacement = ("# Verified public aggregate auxiliary input; no author workbook is read.\n"
            "aux <- read.csv(" + real.r_string(public_aux) + ",colClasses=c(ubigeo='character',dept_code='character',prov_code='character'),check.names=FALSE)\n"
            "stopifnot(!anyDuplicated(aux$ubigeo));writeout(aux,'aux_all_districts.csv')\n\n")
        normalized = text[:beginning] + replacement + text[ending:]
        if normalized[normalized.index("summaries <- list()"): ] != text[ending:]:
            raise RuntimeError("SAV parsing/endpoint/variance block changed")
        Path(output).write_text(normalized)
        provenance[-1]["normalizations"].append({"from": original_block, "to": replacement,
            "role": "replace only auxiliary workbook assembly by identical verified public aggregate CSV"})
        provenance[-1]["runtime_sha256"] = real.sha(output)
        provenance[-1]["SAV_endpoint_and_variance_suffix_unchanged"] = True

    namespace = dict(real.__dict__)
    namespace["adapted_r"] = portable_r
    code = compile(normalized_python, str(normalized_path), "exec")
    exec(code, namespace)
    namespace["prepare"](args)
    # Compare_npz has already disclosed any basis difference. Also diagnose
    # fresh-vs-archived aggregate spectral matrices independently of responses.
    from real_basis_check import diagnose
    diagnostic = diagnose(args.source_root, args.reference_root, args.run_root)
    real.write_json(args.run_root / "real_basis_diagnostic.json", diagnostic)
    path = args.run_root / "real_preparation_receipt.json"
    receipt = json.loads(path.read_text())
    receipt.update({"author_workbooks_read": False, "public_aggregate_auxiliary_sha256": real.sha(public_aux),
        "public_private_input_catalogue_sha256": real.sha(args.catalogue),
        "private_reference_checksum_checks": catalogue_state["verified"],
        "portable_prepare_adapter_sha256": real.sha(__file__),
        "private_catalogue_adapter_sha256": real.sha(catalogue_module.__file__),
        "borrowed_operational_prepare_adapter_sha256": real.sha(real.__file__),
        "runtime_python_dependency_normalization_sha256": real.sha(normalized_path),
        "runtime_python_normalization": "Remove only author-workbook presence/hash dependency; use verified public auxiliary CSV. Statistical kernels unchanged.",
        "basis_diagnostic_sha256": real.sha(args.run_root / "real_basis_diagnostic.json"),
        "fresh_basis_matches_unchanged_kernel_exactly": diagnostic["prepared_basis_matches_unchanged_source_exactly"],
        "fitting_permitted_using_fresh_basis": diagnostic["prepared_basis_matches_unchanged_source_exactly"],
        "article_result_reproduction_verified": False})
    real.write_json(path, receipt)
    print(json.dumps({"status": receipt["status"], "author_workbooks_read": False,
                      "private_reference_checksum_checks": len(catalogue_state["verified"]),
                      "private_records_exported": False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "reference-root", "run-root", "catalogue"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--public-aux-csv", type=Path)
    parser.add_argument("--rscript", type=Path)
    parser.add_argument("--r-library", type=Path)
    parser.add_argument("--author-aux-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    prepare(args)


if __name__ == "__main__":
    main()
