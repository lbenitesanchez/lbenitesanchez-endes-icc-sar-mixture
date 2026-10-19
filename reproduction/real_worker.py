"""Rebuild ENDES training inputs from SAV, then rerun frozen real-data fits.

All person-level data, fitted draws, and runtime scripts belong in --run-root,
outside this repository. Scientific kernels are imported unchanged; this file
adapts filesystem locations, the original R year loop, and output destinations.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from code_only_sources import historical_source_matches

sys.dont_write_bytecode = True
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")


def author_aux_root(args):
    """Resolve auxiliary inputs supplied explicitly for the private run."""
    value = args.author_aux_root or os.environ.get("ENDES_AUX_ROOT")
    if not value:
        raise ValueError("Supply --author-aux-root or ENDES_AUX_ROOT for the two original auxiliary workbooks")
    return Path(value).expanduser()


def locations(root):
    root = Path(root).resolve()
    if (root / "sikov_2026-10-10").is_dir():
        return root, root / "sikov_2026-10-10"
    if (root / "protocol_freeze.json").is_file() and (root / "scripts/comparison_real.py").is_file():
        return root.parent, root
    if (root / "estudio_publicacion/sikov_2026-10-10").is_dir():
        study = root / "estudio_publicacion"
        return study, study / "sikov_2026-10-10"
    raise ValueError("source/reference root must contain the frozen study or the Sikov directory")


def remap(old, proof, study):
    old = Path(old)
    anchor = Path(proof["path_anchor"])
    if old.is_absolute():
        return study / old.relative_to(anchor)
    return study / old


def source_gate(source_study, source, reference_study):
    # Historical freezes remain in the externally supplied reference tree.
    reference_article = reference_study / source.name
    proof = json.loads((reference_article / "protocol_freeze.json").read_text())
    add = json.loads((reference_article / "validacion/sikov_published_addendum_freeze.json").read_text())
    checked = {}
    for document in (proof, add):
        for block in ("statistical_sources_sha256", "additional_sources_sha256", "base_kernel_sources_sha256"):
            for old, expected in document.get(block, {}).items():
                target = remap(old, proof, source_study)
                if not historical_source_matches(target, expected):
                    raise RuntimeError("Changed frozen source: " + str(target))
                checked[str(target.relative_to(source_study))] = expected
        for old, expected in document.get("frozen_inputs_sha256", {}).items():
            target = remap(old, proof, reference_study)
            if not target.is_file() or sha(target) != expected:
                raise RuntimeError("Changed original frozen input/certificate: " + str(target))
    return proof, add, checked


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("Unexpected frozen R source contract: " + old)
    return text.replace(old, new, 1)


def r_string(path):
    return json.dumps(str(Path(path).resolve()))


def adapted_r(source, output, replacements, provenance):
    original = Path(source).read_text()
    text = original
    for old, new in replacements:
        text = replace_once(text, old, new)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(text)
    provenance.append({"source": str(source), "source_sha256": sha(source),
                       "runtime_copy": str(output), "runtime_sha256": sha(output),
                       "normalizations": [{"from": old, "to": new} for old, new in replacements],
                       "statistical_operations_changed": False})


def run_r(rscript, script, arguments, log, cwd):
    with Path(log).open("w") as stream:
        process = subprocess.run([str(rscript), str(script), *map(str, arguments)], cwd=cwd,
                                 stdout=stream, stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        raise RuntimeError("R preparation failed; inspect private log " + str(log))


def copy_aggregate(source, destination, receipts):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    receipts.append({"source": str(source), "source_sha256": sha(source),
                     "destination": str(destination), "role": "aggregate/certificate input",
                     "new_fitted_or_prepared_output_copied": False})


def compare_csv(new, old):
    import numpy as np
    import pandas as pd
    a = pd.read_csv(new, dtype={"ubigeo": str, "domain": str, "dept_code": str, "prov_code": str,
                                "child_id": str, "household_id": str, "psu_key": str})
    b = pd.read_csv(old, dtype={"ubigeo": str, "domain": str, "dept_code": str, "prov_code": str,
                                "child_id": str, "household_id": str, "psu_key": str})
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        raise RuntimeError("CSV frame/schema differs: " + str(new))
    maximum = 0.0
    for name in a:
        if pd.api.types.is_numeric_dtype(a[name]) and pd.api.types.is_numeric_dtype(b[name]):
            aa, bb = a[name].to_numpy(float), b[name].to_numpy(float)
            if not np.array_equal(np.isnan(aa), np.isnan(bb)) or not np.allclose(aa, bb, atol=1e-12, rtol=0, equal_nan=True):
                raise RuntimeError("CSV value/order differs: " + str(new) + ":" + name)
            finite = np.isfinite(aa) & np.isfinite(bb)
            if finite.any():
                maximum = max(maximum, float(np.max(np.abs(aa[finite] - bb[finite]))))
        elif not a[name].fillna("<NA>").astype(str).equals(b[name].fillna("<NA>").astype(str)):
            raise RuntimeError("CSV label/order differs: " + str(new) + ":" + name)
    return {"new": str(new), "reference": str(old), "rows": len(a), "columns": len(a.columns),
            "same_row_order": True, "max_abs_numeric_difference": maximum,
            "new_sha256": sha(new), "reference_sha256": sha(old), "exact_bytes": sha(new) == sha(old)}


def compare_npz(new, old):
    import numpy as np
    maximum = 0.0
    basis_difference = None
    with np.load(new, allow_pickle=False) as a, np.load(old, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise RuntimeError("NPZ array names differ: " + str(new))
        for name in a.files:
            aa, bb = a[name], b[name]
            if aa.shape != bb.shape:
                raise RuntimeError("NPZ shape differs: " + name)
            if np.issubdtype(aa.dtype, np.number) and np.issubdtype(bb.dtype, np.number):
                equal = np.array_equal(np.isnan(aa), np.isnan(bb)) and np.allclose(aa, bb, atol=1e-12, rtol=0, equal_nan=True)
                if name == "basis" and not equal:
                    basis_difference = {"exact": False, "max_abs_entry_difference": float(np.max(np.abs(aa - bb))),
                        "max_abs_prior_covariance_difference": float(np.max(np.abs(aa @ aa.T - bb @ bb.T))),
                        "relative_frobenius_prior_covariance_difference": float(np.linalg.norm(aa @ aa.T - bb @ bb.T) / np.linalg.norm(bb @ bb.T)),
                        "fresh_basis_retained_for_fitting": True, "archived_basis_copied_or_substituted": False,
                        "accepted_as_exact_input_reproduction": False}
                elif not equal:
                    raise RuntimeError("NPZ numerical values differ: " + str(new) + ":" + name)
                finite = np.isfinite(aa) & np.isfinite(bb)
                if finite.any():
                    maximum = max(maximum, float(np.max(np.abs(aa[finite] - bb[finite]))))
            elif not np.array_equal(aa, bb):
                raise RuntimeError("NPZ label/order differs: " + str(new) + ":" + name)
        count = len(a.files)
    return {"new": str(new), "reference": str(old), "arrays": count,
            "max_abs_numeric_difference": maximum, "new_sha256": sha(new),
            "reference_sha256": sha(old), "exact_bytes": sha(new) == sha(old),
            "basis_difference": basis_difference, "all_other_arrays_exact_within_1e12": True}


def prepare(args):
    source_study, source = locations(args.source_root)
    reference_study, reference = locations(args.reference_root)
    proof, add, frozen_sources = source_gate(source_study, source, reference_study)
    run = args.run_root.resolve()
    private = run.parent / "real_private_study"
    old = private / "fase2_2026-10-07"
    if (run / "datos/real_2019_full").exists() or (run / "datos/real_2024_training").exists():
        raise FileExistsError("Preparation refuses an existing prepared output tree")
    for p in (run / "datos", run / "validacion", private / "scripts", private / "datos/raw",
              old / "datos/private_external_raw", old / "validacion"):
        p.mkdir(parents=True, exist_ok=True)
    provenance, aggregate_inputs, comparisons = [], [], []
    logs = run / "preparation_logs"
    logs.mkdir(exist_ok=True)
    rscript = args.rscript or shutil.which("Rscript")
    if not rscript:
        raise RuntimeError("Rscript required for raw SAV preparation")
    rlib = args.r_library or reference_study / ".Rlib"
    if not rlib.is_dir():
        historical = Path(proof["path_anchor"]) / ".Rlib"
        if historical.is_dir():
            rlib = historical
    if rlib.is_dir() and not (private / ".Rlib").exists():
        (private / ".Rlib").symlink_to(rlib, target_is_directory=True)
    raw2019 = reference_study / "datos/raw/ENDES2019"
    (private / "datos/raw/ENDES2019").symlink_to(raw2019, target_is_directory=True)
    auxroot = author_aux_root(args)
    workbook_names = ("consolidado_nuevo_completo_con_directos.xlsx", "consolidado_censo_ubigeo_actualizado.xlsx")
    for name in workbook_names:
        if not (auxroot / name).is_file():
            raise FileNotFoundError(auxroot / name)
    first_source = source_study / "scripts/01_prepare_data.R"
    first = private / "scripts/01_prepare_data.R"
    adapted_r(first_source, first, [
        ("study <- if(dir.exists('datos/raw')) normalizePath('.') else normalizePath('estudio_publicacion')",
         "study <- normalizePath(" + r_string(private) + ")"),
        ("src <- Sys.getenv('ENDES_AUX_ROOT', unset='PRIVATE_INPUTS')", "src <- " + r_string(auxroot)),
        ("for(year in c(2019,2025)) {", "for(year in c(2019)) {")], provenance)
    run_r(rscript, first, [], logs / "raw2019_parse.log", private)
    export_source = source_study / "scripts/11_export_unit_inputs.R"
    export = private / "scripts/11_export_unit_inputs.R"
    adapted_r(export_source, export, [("for(year in c(2019,2025)){", "for(year in c(2019)){")], provenance)
    run_r(rscript, export, [], logs / "raw2019_unit_export.log", private)

    oldref = reference_study / "fase2_2026-10-07"
    for p in ("protocol_freeze.json", "validacion/externa_reserved_access_approval.json"):
        copy_aggregate(oldref / p, old / p, aggregate_inputs)
    parser_source = source_study / "fase2_2026-10-07/scripts/validacion_parse_reserved.R"
    parser = private / "scripts/validacion_parse_reserved_2024.R"
    adapted_r(parser_source, parser, [
        ("paths <- file.path(root,'datos/private_external_raw',c('RECH0_2024.SAV','RECH1_2024.SAV','RECH6_2024.SAV'))",
         "paths <- file.path(" + r_string(oldref / "datos/private_external_raw") + ",c('RECH0_2024.SAV','RECH1_2024.SAV','RECH6_2024.SAV'))")], provenance)
    run_r(rscript, parser, [old], logs / "raw2024_parse.log", private)
    write_json(old / "validacion/reexecution_reserved_parse.json", {
        "role": "reexecution of already observed 2024 source, no new independent validation",
        "first_reserved_access_event_replaced": False,
        "original_records_or_locks_modified": False,
        "source_parser_sha256": sha(parser_source), "runtime_parser_sha256": sha(parser)})

    # Rebuild fourteen official auxiliary predictors from saved census counts;
    # these source tables contain aggregates, not previously fitted estimates.
    copy_aggregate(oldref / "datos/geografia/aux_verified_2017_2018.csv",
                   old / "datos/geografia/aux_verified_2017_2018.csv", aggregate_inputs)
    query_names_old = ("census_floor", "census_water", "census_electricity", "census_sewer",
                       "census_refrigerator", "census_spanish", "census_illiteracy")
    query_names_new = ("census_walls", "census_internet", "census_sis", "census_essalud",
                       "census_noinsurance", "census_rural")
    for names, original_root, new_root, folder in ((query_names_old, oldref, old, "fuentes/geografia"),
                                                 (query_names_new, reference, run, "datos/census_sources")):
        for name in names:
            query = original_root / folder / (name + "_query.json")
            copy_aggregate(query, new_root / folder / query.name, aggregate_inputs)
            for table in json.loads(query.read_text())["tables"]:
                copy_aggregate(original_root / table["path"], new_root / table["path"], aggregate_inputs)
    sys.path[:0] = [str(source / "scripts"), str(source_study / "scripts"),
                   str(source_study / "fase2_2026-10-07/scripts")]
    aux = importlib.import_module("align_aux_sikov")
    aux.ROOT, aux.STUDY, aux.OLD, aux.OUT = run, run.parent, old, run / "datos"
    with (logs / "aux14.log").open("w") as stream, contextlib.redirect_stdout(stream):
        aux.main()
    survey_source = source / "scripts/align_survey_2019.R"
    survey = private / "scripts/align_survey_2019.R"
    adapted_r(survey_source, survey, [
        ("root <- normalizePath(args[1]); study <- normalizePath(file.path(root,'..'))",
         "root <- normalizePath(args[1]); study <- normalizePath(" + r_string(private) + ")")], provenance)
    run_r(rscript, survey, [run], logs / "survey2019_independent.log", private)
    for name in ("externa_prediction_lock_temporal.json", "externa_prediction_lock_territorial.json", "externa_first_reserved_open.json"):
        copy_aggregate(oldref / "validacion" / name, old / "validacion" / name, aggregate_inputs)
    build = importlib.import_module("align_build_inputs")
    build.ROOT, build.STUDY, build.OLD, build.OUT, build.VAL = run, private, old, run / "datos", run / "validacion"
    with (logs / "aligned_inputs.log").open("w") as stream, contextlib.redirect_stdout(stream):
        build.main()
    # Copy the protocol bytes unchanged; all changed runtime paths live solely
    # in this adapter and in the new manifests.
    copy_aggregate(source / "protocol_freeze.json", run / "protocol_freeze.json", aggregate_inputs)
    train = importlib.import_module("align_real_training")
    train.ROOT, train.STUDY, train.OLD = run, private, old
    with (logs / "real_training.log").open("w") as stream, contextlib.redirect_stdout(stream):
        train.prepare(2019, run / "datos/real_2019_full")
        train.prepare(2024, run / "datos/real_2024_training", run / "protocol_freeze.json")
    source_gate(source_study, source, reference_study)

    for year, directory, original in ((2019, private / "datos/private_competitors", reference_study / "datos/private_competitors"),
                                       (2024, old / "datos/private_external", oldref / "datos/private_external")):
        names = [f"households_{year}.csv", f"children_{year}.csv"] if year == 2019 else [
            "households_2024.csv", "children_2024_training.csv", "children_2024_test.csv", "children_2024_scoring_only.csv"]
        for name in names:
            comparisons.append(compare_csv(directory / name, original / name))
    for name in ("aux_sikov14_official.csv", "survey2019_harmonized_and_definition_sensitivities.csv"):
        comparisons.append(compare_csv(run / "datos" / name, reference / "datos" / name))
    for folder in ("real_2019_full", "real_2024_training"):
        comparisons.append(compare_npz(run / "datos" / folder / "fit_inputs.npz", reference / "datos" / folder / "fit_inputs.npz"))
        comparisons.append(compare_csv(run / "datos" / folder / "private/units.csv", reference / "datos" / folder / "private/units.csv"))
        comparisons.append(compare_csv(run / "datos" / folder / "area_calibration.csv", reference / "datos" / folder / "area_calibration.csv"))
    inputs = [raw2019 / "Modulo64/RECH0.SAV", raw2019 / "Modulo64/RECH1.sav", raw2019 / "Modulo74/RECH6.SAV",
              *[oldref / "datos/private_external_raw" / name for name in ("RECH0_2024.SAV", "RECH1_2024.SAV", "RECH6_2024.SAV")],
              *[auxroot / name for name in workbook_names]]
    has_basis_difference = any(c.get("basis_difference") for c in comparisons)
    receipt = {"status": "PASS_RAW_AUX_DIRECT_PREPARATION_WITH_DISCLOSED_BASIS_DIFFERENCE" if has_basis_difference else "PASS_FRESH_SAV_REAL_PREPARATION", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "private_study": str(private), "run_root": str(run), "source_root": str(source),
               "source_sha256": frozen_sources, "raw_and_author_input_sha256": {str(p): sha(p) for p in inputs},
               "protocol_sha256": sha(source / "protocol_freeze.json"),
               "addendum_sha256": sha(source / "validacion/sikov_published_addendum_freeze.json"),
               "runtime_R_normalizations": provenance, "aggregate_and_certificate_inputs": aggregate_inputs,
               "comparisons": comparisons, "fresh_microdata_preparation_from_SAV": True,
               "prepared_inputs_all_exact": not has_basis_difference,
               "basis_difference_disclosed": has_basis_difference,
               "previous_prepared_estimates_used_as_new_calculation": False,
               "fit_heldout_outcomes_used": False, "statistical_sources_modified": False,
               "2024_dataset_previously_observed_disclosed": True,
               "privacy": "raw, unit records, RDS, real chains and draws must be excluded from GitHub"}
    write_json(run / "real_preparation_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "comparisons": len(comparisons),
                      "private_study": str(private), "max_abs_difference": max(c["max_abs_numeric_difference"] for c in comparisons)}), flush=True)


def fit(args):
    if not args.scope or not args.exe or not args.cmdstan:
        raise ValueError("fit requires --scope, --exe and --cmdstan")
    import sikov_worker as shared
    state = shared.initialize(args.source_root, args.reference_root, args.run_root, args.exe, args.cmdstan)
    run = args.run_root.resolve()
    source_study, source = locations(args.source_root)
    reference_study, reference = locations(args.reference_root)
    preparation = json.loads((run / "real_preparation_receipt.json").read_text())
    private = Path(preparation["private_study"])
    if not preparation["fresh_microdata_preparation_from_SAV"]:
        raise RuntimeError("No verified fresh SAV preparation")
    sys.path.insert(0, str(source / "scripts"))
    base = importlib.import_module("comparison_real")
    base.ROOT = run
    proof = json.loads((source / "protocol_freeze.json").read_text())
    add = json.loads((source / "validacion/sikov_published_addendum_freeze.json").read_text())
    def base_gate():
        source_gate(source_study, source, reference_study)
        return proof, sha(source / "protocol_freeze.json")
    base.gate = base_gate
    output = run / "evaluacion" / ("real_" + args.scope)
    shared.set_chain_output(output / "icc_sampler")
    base.run(args.scope)
    # Seal retained real chains before the additional-reference fit binds the
    # original completion. No fitting or response read occurs while sealing.
    base_completion = json.loads((output / "completion.json").read_text())
    base_completion["chain_files_sha256"] = {
        str(p.relative_to(output)): sha(p)
        for p in sorted((output / "icc_sampler").rglob("*")) if p.is_file()}
    base_completion["raw_SAV_reconstruction_receipt_sha256"] = sha(run / "real_preparation_receipt.json")
    base_completion["real_operational_adapter_sha256"] = sha(__file__)
    shared.receipt(output, base_completion)
    lockpath = run / "validacion" / ("real_prediction_lock_" + args.scope + ".json")
    lock = json.loads(lockpath.read_text())
    lock["completion_sha256"] = sha(output / "completion.json")
    lock["retained_real_chains_sealed"] = True
    write_json(lockpath, lock)
    published = importlib.import_module("sikov_published_fit")
    published.ROOT = run
    published.ADDENDUM = source / "validacion/sikov_published_addendum_freeze.json"
    def published_gate():
        source_gate(source_study, source, reference_study)
        return add, sha(published.ADDENDUM)
    published.gate = published_gate
    published.real(args.scope)
    source_gate(source_study, source, reference_study)
    completion = json.loads((output / "completion.json").read_text())
    extra = json.loads((run / "evaluacion/published_sar" / ("real_" + args.scope) / "completion.json").read_text())
    convergence = {**completion["convergence"], **extra["convergence"]}
    receipt = {"status": "PASS_FRESH_REAL_ELEVEN_METHOD_FITS" if all(convergence.values()) else "FAIL_REAL_DIAGNOSTICS",
               "scope": args.scope, "preparation_receipt_sha256": sha(run / "real_preparation_receipt.json"),
               "convergence": convergence, "source_root": str(source), "run_root": str(run),
               "seconds": completion["seconds"] + extra["seconds"],
               "real_chain_outputs_retained": True, "heldout_outcomes_read_by_fits": False,
               "raw_SAV_and_private_records_excluded_from_public_repository": True,
               "all_methods": 11}
    write_json(run / ("real_fit_receipt_" + args.scope + ".json"), receipt)
    print(json.dumps(receipt), flush=True)
    if not all(convergence.values()):
        raise RuntimeError("Real convergence gates failed; all results retained")


def verify_preparation(args):
    """Close a completed SAV preparation without regenerating any records."""
    source_study, source = locations(args.source_root)
    reference_study, reference = locations(args.reference_root)
    proof, add, bindings = source_gate(source_study, source, reference_study)
    run = args.run_root.resolve()
    private = run.parent / "real_private_study"
    old, oldref = private / "fase2_2026-10-07", reference_study / "fase2_2026-10-07"
    comparisons = []
    for newdir, refdir, names in (
        (private / "datos/private_competitors", reference_study / "datos/private_competitors", ["households_2019.csv", "children_2019.csv"]),
        (old / "datos/private_external", oldref / "datos/private_external", ["households_2024.csv", "children_2024_training.csv", "children_2024_test.csv", "children_2024_scoring_only.csv"])):
        for name in names:
            comparisons.append(compare_csv(newdir / name, refdir / name))
    for name in ("aux_sikov14_official.csv", "survey2019_harmonized_and_definition_sensitivities.csv"):
        comparisons.append(compare_csv(run / "datos" / name, reference / "datos" / name))
    for folder in ("real_2019_full", "real_2024_training"):
        fresh = run / "datos" / folder
        manifest = json.loads((fresh / "preparation_manifest.json").read_text())
        for name, expected in manifest["output_sha256"].items():
            if sha(fresh / name) != expected:
                raise RuntimeError("Prepared file changed after its source calculation")
        if sha(fresh / "private/units.csv") != manifest["private_units_sha256"] or manifest["heldout_outcome_rows_read"]:
            raise RuntimeError("Private units changed or preparation leaked heldout outcomes")
        comparisons.append(compare_npz(fresh / "fit_inputs.npz", reference / "datos" / folder / "fit_inputs.npz"))
        comparisons.append(compare_csv(fresh / "private/units.csv", reference / "datos" / folder / "private/units.csv"))
        comparisons.append(compare_csv(fresh / "area_calibration.csv", reference / "datos" / folder / "area_calibration.csv"))
    auxroot = author_aux_root(args)
    adaptations = []
    copies = [(source_study / "scripts/01_prepare_data.R", private / "scripts/01_prepare_data.R", [
        ("study <- if(dir.exists('datos/raw')) normalizePath('.') else normalizePath('estudio_publicacion')", "study <- normalizePath(" + r_string(private) + ")"),
        ("src <- Sys.getenv('ENDES_AUX_ROOT', unset='PRIVATE_INPUTS')", "src <- " + r_string(auxroot)),
        ("for(year in c(2019,2025)) {", "for(year in c(2019)) {")]),
        (source_study / "scripts/11_export_unit_inputs.R", private / "scripts/11_export_unit_inputs.R", [("for(year in c(2019,2025)){", "for(year in c(2019)){")]),
        (source_study / "fase2_2026-10-07/scripts/validacion_parse_reserved.R", private / "scripts/validacion_parse_reserved_2024.R", [
            ("paths <- file.path(root,'datos/private_external_raw',c('RECH0_2024.SAV','RECH1_2024.SAV','RECH6_2024.SAV'))", "paths <- file.path(" + r_string(oldref / "datos/private_external_raw") + ",c('RECH0_2024.SAV','RECH1_2024.SAV','RECH6_2024.SAV'))")]),
        (source / "scripts/align_survey_2019.R", private / "scripts/align_survey_2019.R", [
            ("root <- normalizePath(args[1]); study <- normalizePath(file.path(root,'..'))", "root <- normalizePath(args[1]); study <- normalizePath(" + r_string(private) + ")")])]
    for original, runtime, replacements in copies:
        expected = original.read_text()
        for first, second in replacements:
            expected = replace_once(expected, first, second)
        if runtime.read_text() != expected:
            raise RuntimeError("Runtime R copy has undeclared modifications: " + str(runtime))
        adaptations.append({"source": str(original), "source_sha256": sha(original), "runtime_copy": str(runtime),
                            "runtime_sha256": sha(runtime), "normalizations": [{"from": x, "to": y} for x, y in replacements],
                            "statistical_operations_changed": False})
    raw = [reference_study / "datos/raw/ENDES2019" / n for n in ("Modulo64/RECH0.SAV", "Modulo64/RECH1.sav", "Modulo74/RECH6.SAV")]
    raw += [oldref / "datos/private_external_raw" / n for n in ("RECH0_2024.SAV", "RECH1_2024.SAV", "RECH6_2024.SAV")]
    raw += [auxroot / n for n in ("consolidado_nuevo_completo_con_directos.xlsx", "consolidado_censo_ubigeo_actualizado.xlsx")]
    basis = [c["basis_difference"] for c in comparisons if c.get("basis_difference")]
    from real_basis_check import diagnose
    diagnostic = diagnose(args.source_root, args.reference_root, args.run_root)
    write_json(run / "real_basis_diagnostic.json", diagnostic)
    receipt = {"status": "PASS_RAW_AUX_DIRECT_PREPARATION_WITH_DISCLOSED_BASIS_DIFFERENCE" if basis else "PASS_FRESH_SAV_REAL_PREPARATION",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "run_root": str(run), "private_study": str(private),
        "source_root": str(source), "source_sha256": bindings, "raw_and_author_input_sha256": {str(p): sha(p) for p in raw},
        "protocol_sha256": sha(source / "protocol_freeze.json"), "addendum_sha256": sha(source / "validacion/sikov_published_addendum_freeze.json"),
        "runtime_R_normalizations": adaptations, "comparisons": comparisons,
        "fresh_microdata_preparation_from_SAV": True, "previous_prepared_estimates_used_as_new_calculation": False,
        "statistical_sources_modified": False, "fit_heldout_outcomes_used": False,
        "2024_dataset_previously_observed_disclosed": True, "prepared_inputs_all_exact": not basis,
        "raw_auxiliary_and_direct_arrays_exact_within_1e12": True, "basis_difference_disclosed": bool(basis),
        "basis_diagnostic_sha256": sha(run / "real_basis_diagnostic.json"),
        "fresh_basis_matches_unchanged_kernel_exactly": diagnostic["prepared_basis_matches_unchanged_source_exactly"],
        "fitting_permitted_using_fresh_basis": diagnostic["prepared_basis_matches_unchanged_source_exactly"],
        "article_result_reproduction_verified": False,
        "privacy": "raw, unit records, RDS, real chains and draws must be excluded from GitHub"}
    write_json(run / "real_preparation_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "comparisons": len(comparisons),
                      "basis_difference_disclosed": receipt["basis_difference_disclosed"], "private_study": str(private)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "verify-preparation", "fit", "score"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--scope", choices=("temporal", "territorial"))
    parser.add_argument("--exe", type=Path)
    parser.add_argument("--cmdstan", type=Path)
    parser.add_argument("--rscript", type=Path)
    parser.add_argument("--r-library", type=Path)
    parser.add_argument("--author-aux-root", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
        verify_preparation(args)
    elif args.action == "verify-preparation":
        verify_preparation(args)
    elif args.action == "score":
        from real_score import score
        score(args)
    else:
        fit(args)


if __name__ == "__main__":
    main()
