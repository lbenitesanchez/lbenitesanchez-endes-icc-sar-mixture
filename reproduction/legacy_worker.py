#!/usr/bin/env python3
"""Operational helpers for legacy_runner; all files written here are private.

The frozen model modules are imported unchanged. The serialization adapter
converts NumPy scalars/arrays only while writing metadata; it does not touch
model arrays, initial values, priors, seeds, draws or diagnostic decisions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

PHASE = "fase2_2026-10-07"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def setup(study):
    os.environ["ENDES_STUDY_ROOT"] = str(study)
    os.environ["ENDES_EXPORT_PRIVATE_INPUTS"] = "1"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    sdk = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk")
    if sys.platform == "darwin" and sdk.is_dir():
        os.environ.setdefault("SDKROOT", str(sdk))
        os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "26.0")
        # Match the existing CmdStan PCH's deployment target. Compiler options
        # are operational; frozen Stan/Python source bytes remain unchanged.
        import cmdstanpy
        base_model = cmdstanpy.CmdStanModel
        class PortableModel(base_model):
            def __init__(self, *args, **kwargs):
                options = {"CXXFLAGS": "-mmacosx-version-min=26.0", **(kwargs.get("cpp_options") or {})}
                kwargs["cpp_options"] = options
                super().__init__(*args, **kwargs)
        cmdstanpy.CmdStanModel = PortableModel
    if (study / ".Rlib").exists():
        os.environ["ENDES_R_LIB"] = str((study / ".Rlib").resolve())
    sys.path[:0] = [str(study / PHASE / "scripts"), str(study / "scripts/competitors"), str(study / "scripts")]
    os.chdir(study)


def compile_models(study):
    import beta_power_hmc as beta
    import hogg_tsln as h
    import hogg_design_extension as ext
    import hogg_spatial as spatial
    models = h.compile_models()
    ext.compile_extension()
    spatial.compile_spatial()
    beta.get_model()
    save(study / "validacion/legacy_runtime_compilation.json", {
        "status": "COMPILED_NO_MODEL_FITS", "models": list(models),
        "CmdStan": str(h.CMDSTAN.resolve()), "SDKROOT": os.environ.get("SDKROOT"),
        "sources_modified": False, "model_fits_dispatched": False,
        "operational_worker_sha256": sha(Path(__file__))})


class MetadataJSON:
    """Narrow module proxy; retain stdlib behavior except the known NumPy bug."""
    def __getattr__(self, name):
        return getattr(json, name)

    @staticmethod
    def dumps(value, **kwargs):
        import numpy as np
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError("Cannot serialize " + type(obj).__name__)
        return json.dumps(value, default=convert, **kwargs)


def script(study, argv):
    relative, *rest = argv
    path = study / relative
    if not path.is_file() or not path.resolve().is_relative_to(study):
        raise ValueError("Only a prepared frozen script can run")
    sys.path.insert(0, str(path.parent))
    sys.argv = [str(path), *rest]
    if path.name == "validacion_fit_models.py":
        # Retain the original scientific source, repair metadata serialization
        # in memory instead of rerunning a successful Hogg fit after JSON fails.
        module = importlib.import_module("validacion_fit_models")
        module.json = MetadataJSON()
        module.main()
    else:
        runpy.run_path(str(path), run_name="__main__")


def cut40_generate(study, workers):
    """Generation only; separate population/sample RNG streams are unchanged."""
    from concurrent.futures import ProcessPoolExecutor
    # Spawn-safe workers run the unchanged generate CLI, never fit.
    commands = [[sys.executable, str(study / PHASE / "scripts/sim_design_diagnostics.py"),
                 "generate", "--context", context, "--rep", str(rep)]
                for context in ("dense6", "sparse2") for rep in range(20)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_generation, commands))
    if any(code != 0 for code in results):
        raise RuntimeError("A planned generator failed; no replacement seeds")
    save(study / PHASE / "validacion/legacy_cut40_generation.json", {
        "status": "GENERATED_40_NO_MODEL_FITS", "worlds": 40, "model_fits_dispatched": False,
        "operational_worker_sha256": sha(Path(__file__))})


def run_generation(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout[-1000:] + result.stderr[-2000:], flush=True)
    return result.returncode


def cut40_workers(study, workers):
    # Generate separately when not already generated by the setup action.
    if not (study / PHASE / "validacion/legacy_cut40_generation.json").exists():
        cut40_generate(study, workers)
    from concurrent.futures import ProcessPoolExecutor
    commands = [[sys.executable, str(Path(__file__).resolve()), "--study-root", str(study),
                 "cut40_fit_shard", str(worker), str(workers)] for worker in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_generation, commands))
    if any(code != 0 for code in results):
        raise RuntimeError("A fixed CUT40 worker failed; preserve failed populations")


def cut40_fit_shard(study, worker, workers):
    import sim_design_diagnostics as sim
    sim.gate()
    jobs = [(context, rep) for context in sim.CONTEXTS for rep in range(sim.POPULATIONS_PER_CONTEXT)]
    for index, (context, rep) in enumerate(jobs):
        if index % workers == worker:
            sim.fit_world(context, rep)
            print(json.dumps({"context": context, "rep": rep, "worker": worker, "fit_completed": True}), flush=True)


def prepare_actual(study, reference):
    marker = study / "validacion/legacy_actual_preparation.json"
    if marker.exists():
        d = json.loads(marker.read_text())
        for rel, digest in d["fresh_private_input_sha256"].items():
            if sha(study / rel) != digest:
                raise RuntimeError("Fresh actual input changed: " + rel)
        return
    required = ("ENDES2019/Modulo64/RECH0.SAV", "ENDES2019/Modulo64/RECH1.sav",
                "ENDES2019/Modulo74/RECH6.SAV", "ENDES2025/Base_Datos/RECH0_2025.SAV",
                "ENDES2025/Base_Datos/RECH1_2025.SAV", "ENDES2025/Base_Datos/RECH6_2025.SAV")
    records = []
    for rel in required:
        original = reference / "datos/raw" / rel
        if not original.is_file() or not original.stat().st_size:
            raise FileNotFoundError("Original official SAV unavailable: " + str(original))
        dest = study / "datos/raw" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, dest)
        records.append({"relative_path": rel, "sha256": sha(dest)})
    # Frozen public adapter, explicit output route. Every direct ratio is new.
    reconstructed = study / "datos/reconstruidos"
    subprocess.run(["Rscript", str(study / "codigo/01_prepare_data.R"),
                    "--raw-dir", str(study / "datos/raw"),
                    "--aux-file", str(study / "datos/processed/aux_all_districts.csv"),
                    "--output-dir", str(reconstructed)], check=True)
    import pandas as pd
    import numpy as np
    aggregate_checks = []
    for year in (2019, 2025):
        name = f"areas_{year}_6_59.csv"
        fresh = pd.read_csv(reconstructed / name, dtype={"ubigeo": str})
        old = pd.read_csv(reference / "datos/processed" / name, dtype={"ubigeo": str})
        a = fresh.sort_values("ubigeo").reset_index(drop=True)
        b = old.sort_values("ubigeo").reset_index(drop=True)
        if not a.ubigeo.equals(b.ubigeo):
            raise RuntimeError("Reconstructed district support differs")
        numerical = [c for c in a.columns if c in b and pd.api.types.is_numeric_dtype(a[c])
                     and pd.api.types.is_numeric_dtype(b[c])]
        errors = {}
        for column in numerical:
            av, bv = a[column].to_numpy(float), b[column].to_numpy(float)
            if not np.array_equal(np.isnan(av), np.isnan(bv)):
                raise RuntimeError("Reconstructed missingness differs: " + column)
            error = float(np.nanmax(np.abs(av-bv))) if np.isfinite(av-bv).any() else 0.0
            errors[column] = error
            if not np.allclose(av, bv, atol=1e-8, rtol=1e-10, equal_nan=True):
                raise RuntimeError("Reconstructed direct inputs differ: " + column)
        aggregate_checks.append({"year": year, "numeric_columns": len(numerical), "maximum_absolute_errors": errors})
    # Use reconstructed aggregates for all subsequent application inputs.
    for original in reconstructed.glob("*"):
        if original.is_file():
            dest = study / "datos/processed" / original.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, dest)
    subprocess.run(["Rscript", str(study / "scripts/11_export_unit_inputs.R")], check=True)
    private = study / "datos/private_competitors"
    save(marker, {"status": "RECONSTRUCTED_FROM_SAV_NO_MODEL_FITS", "raw_modules": records,
        "aggregate_comparison": aggregate_checks, "model_inputs_use_reconstructed_aggregates": True,
        "fresh_private_input_sha256": {str(p.relative_to(study)): sha(p) for p in private.glob("*.csv")},
        "individual_records_publicly_distributed": False, "model_fits_dispatched": False})


def import2024(study, reference, shared):
    phase = study / PHASE
    if not (phase / "validacion/externa_prediction_lock_temporal.json").exists():
        raise RuntimeError("Finish and lock every temporal prediction before importing scoring responses")
    dest = phase / "datos/private_external"
    if dest.exists():
        raise FileExistsError(dest)
    names = ("households_2024.csv", "children_2024_training.csv", "children_2024_test.csv", "children_2024_scoring_only.csv")
    source = shared / PHASE / "datos/private_external"
    # Reuse only newly parsed individual records, never saved predictions/fits.
    if not all((source / name).is_file() for name in names):
        raise FileNotFoundError("Run the real-data preparation parser first: " + str(source))
    dest.mkdir(parents=True)
    for name in names:
        shutil.copyfile(source / name, dest / name)
    protocol = phase / "protocol_freeze.json"
    save(phase / "validacion/externa_reserved_access_approval.json", {
        "approved_for_reserved_outcome_access": True, "protocol_sha256": sha(protocol),
        "scope": "Authorized user-requested reexecution of previously assessed 2024 responses",
        "untouched_test_or_prospective_validation_claim": False})
    save(phase / "validacion/externa_first_reserved_open.json", {
        "event": "FRESH_REEXECUTION_INPUT_IMPORT_AFTER_TEMPORAL_PREDICTION_LOCK",
        "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "previously_assessed_responses": True,
        "shared_preparation_may_precede_legacy_lock": True, "new_untouched_validation_claim": False,
        "source_files_sha256": {name: sha(dest / name) for name in names}, "WHO2024_endpoint_preserved": True})


def lock(study, scope):
    phase = study / PHASE
    prefix = "externa_temporal_2019" if scope == "temporal" else "externa_territorial_2024"
    specs = [{"method": "BetaPowerExisting", "path": f"resultados/{prefix}_baseline/predictions.csv",
              "status_path": f"resultados/{prefix}_baseline/status.json"},
             {"method": "HoggAgeSexSpatial", "path": f"resultados/{prefix}_hogg/predictions.csv",
              "status_path": f"resultados/{prefix}_hogg/status.json"},
             {"method": "BetaDesignUncertainty", "path": f"resultados/{prefix}_designcut8/" +
                  ("district_predictions.csv" if scope == "temporal" else "predictions.csv"),
              "status_path": f"resultados/{prefix}_designcut8/completion.json"}]
    specification = phase / f"validacion/legacy_{scope}_specifications.json"
    save(specification, specs)
    script(study, [PHASE + "/scripts/validacion_lock_predictions.py", "--scope", scope,
                   "--specifications", str(specification)])


def reuse_cut8_for_cut16(study):
    phase = study / PHASE
    original = phase / "resultados/externa_temporal_2019_designcut8"
    dest = phase / "resultados/diseno/2019_full_official/cut16"
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    records = []
    for k in range(1, 9):
        src = original / f"state_{k:02d}"
        proof = json.loads((src / "completion.json").read_text())
        if not proof["success"] or sha(src / "posterior_summary_and_draws.npz") != proof["posterior_output_sha256"]:
            raise RuntimeError("CUT8 state is not a completed verified fresh fit")
        shutil.copytree(src, dest / src.name)
        records.append({"state": k, "completion_sha256": sha(src / "completion.json"),
                        "conditional_target_sha256": json.loads((src / "fit_status.json").read_text())["conditional_target_sha256"]})
    save(phase / "validacion/legacy_cut16_first8_reuse.json", {"status": "VERIFIED_FRESH_CUT8_TARGETS_REUSED",
         "states": records, "performance_selection": False})


def compare_cut16(study):
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    phase = study / PHASE
    root8 = phase / "resultados/externa_temporal_2019_designcut8"
    root16 = phase / "resultados/diseno/2019_full_official/cut16"
    a = pd.read_csv(root8 / "district_predictions.csv", dtype={"ubigeo": str})
    b = pd.read_csv(root16 / "district_predictions.csv", dtype={"ubigeo": str})
    if not a.ubigeo.equals(b.ubigeo):
        raise RuntimeError("CUT8/CUT16 prediction supports differ")
    rows = []
    q = (0, .025, .5, .975, 1)
    quant = lambda v: {str(x): float(y) for x, y in zip(q, np.quantile(v, q))}
    for group, mask in (("all", np.ones(len(a), bool)), ("sampled", a.sampled.to_numpy(bool)),
                        ("unsampled", ~a.sampled.to_numpy(bool))):
        for endpoint in ("anemia", "stunting"):
            ma, mb = [d[f"p_mean_{endpoint}"].to_numpy()[mask] for d in (a, b)]
            wa, wb = [(d[f"upper95_{endpoint}"]-d[f"lower95_{endpoint}"]).to_numpy()[mask] for d in (a, b)]
            va, vb = [d[f"p_variance_{endpoint}"].to_numpy()[mask] for d in (a, b)]
            ba, bb = [d[f"between_state_variance_{endpoint}"].to_numpy()[mask] for d in (a, b)]
            rows.append({"group": group, "endpoint": endpoint, "districts": int(mask.sum()),
                "maximum_absolute_mean_difference_pp": float(100*np.max(np.abs(mb-ma))),
                "median_absolute_mean_difference_pp": float(100*np.median(np.abs(mb-ma))),
                "median_width_difference_sensitivity_minus_primary_pp": float(100*np.median(wb-wa)),
                "maximum_absolute_width_difference_pp": float(100*np.max(np.abs(wb-wa))),
                "spearman_posterior_mean": float(spearmanr(ma, mb).statistic),
                "primary_absolute_variance_probability_squared": quant(va),
                "sensitivity_absolute_variance_probability_squared": quant(vb),
                "primary_between_state_variance_share": quant(ba/va),
                "sensitivity_between_state_variance_share": quant(bb/vb)})
    target_equal = all(sha(root8/f"state_{k:02d}/posterior_summary_and_draws.npz") ==
                       sha(root16/f"state_{k:02d}/posterior_summary_and_draws.npz") for k in range(1, 9))
    if not target_equal:
        raise RuntimeError("The first eight conditional states were changed")
    save(root16 / "cut8_comparison.json", {"status": "PASS_FRESH_FIXED_STATE_DESCRIPTIVE_CUT16",
         "primary_state_count": 8, "sensitivity_state_count": 16,
         "same_first_eight_conditional_targets": True, "main_selected_by_performance": False,
         "comparisons": rows, "comparison_to_article_still_required": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--shared-private-root", type=Path)
    parser.add_argument("action", choices=("script", "compile", "cut40_generate", "cut40_workers", "cut40_fit_shard",
                                          "prepare_actual", "import2024", "lock", "reuse_cut8_for_cut16", "compare_cut16"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    study = args.study_root.resolve()
    setup(study)
    if args.action == "script":
        script(study, args.arguments)
    elif args.action == "compile":
        compile_models(study)
    elif args.action == "cut40_generate":
        cut40_generate(study, int(args.arguments[0]))
    elif args.action == "cut40_workers":
        cut40_workers(study, int(args.arguments[0]))
    elif args.action == "cut40_fit_shard":
        cut40_fit_shard(study, *map(int, args.arguments))
    elif args.action == "prepare_actual":
        prepare_actual(study, args.reference_root.resolve())
    elif args.action == "import2024":
        import2024(study, args.reference_root.resolve(), args.shared_private_root.resolve())
    elif args.action == "lock":
        lock(study, args.arguments[0])
    elif args.action == "reuse_cut8_for_cut16":
        reuse_cut8_for_cut16(study)
    elif args.action == "compare_cut16":
        compare_cut16(study)


if __name__ == "__main__":
    main()
