#!/usr/bin/env python3
"""Prepare and run the frozen experiments retained in Appendix A/Supplement.

This operational wrapper never edits a statistical source, seed, endpoint,
prior, sampler setting or diagnostic gate. Run directories must be outside
the public repository. ``prepare``/``preflight`` never fit a model; execution
requires the explicit ``run`` action and a selected stage.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PHASE = "fase2_2026-10-07"
STAGES = ("previous800", "cut40", "application2025", "external2024", "cut16")
SUFFIXES = {".py", ".R", ".stan", ".json", ".md", ".txt"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def copy_source_tree(source, target):
    """Only source files; binaries, environments and compile stamps stay out."""
    if not source.is_dir():
        raise FileNotFoundError(source)
    for original in source.rglob("*"):
        if (not original.is_file() or "__pycache__" in original.parts
                or original.suffix not in SUFFIXES or "compiled_source" in original.name):
            continue
        dest = target / original.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, dest)


def metadata_files(reference):
    out = "resultados/resultados_comparadores/"
    for name in ("protocol_freeze.json", "candidate_frozen_manifest.json",
                 "final_result_contract.json", "numerical_amendment_v2.json",
                 "candidate_frozen_manifest_v2.json", "protocol_numerical_v2_approved.json"):
        yield Path(out + name)
    for name in ("hogg_sampler_freeze.json", "hogg_strong_arms_freeze.json",
                 "hogg_spatial_freeze.json", "hogg_spatial_numerical_amendment.json"):
        yield Path("resultados/competidores_hogg") / name
    yield Path("resultados/competidores_gao/gao_frozen_code_manifest.json")
    for name in ("references.json", "sections.json", "competitor_sections.json",
                 "competitor_reference_style.json", "competitor_document_inputs.json"):
        yield Path("manuscrito") / name
    for name in ("beta_power_independent_audit.json", "competitor_protocol_independent_review.json",
                 "hogg_spatial_pilot_independent_review.json", "competitor_population_rng_independence_audit.json",
                 "hogg_spatial_numerical_initializer_independent_review.json",
                 "hogg_spatial_numerical_initializer_independent_evidence.json"):
        yield Path("validacion") / name
    yield Path(out + "validation/independent_final_audit_qa.json")
    yield Path(PHASE) / "protocol_freeze.json"
    yield Path(PHASE) / "validacion/diseno_protocol_freeze.json"
    yield Path(PHASE) / "study_decisions_before_external_responses.json"
    for rel in ("datos/geografia", "fuentes/external"):
        base = reference / PHASE / rel
        for p in base.glob("*"):
            if p.is_file() and p.suffix in {".csv", ".json"}:
                yield p.relative_to(reference)
    # Approved initializer proofs are source-only evidence, not old fit results.
    amendment = reference / "resultados/competidores_hogg/hogg_spatial_numerical_amendment.json"
    if amendment.exists():
        d = json.loads(amendment.read_text())
        for key in ("independent_audit_path", "numerical_pilot_path"):
            if key in d:
                rel = Path(d[key])
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError("Unsafe initializer evidence reference")
                yield rel


def check_sources(study):
    failures = []
    checked = []
    for rel in (Path("resultados/resultados_comparadores/protocol_freeze.json"),
                Path(PHASE) / "protocol_freeze.json",
                Path(PHASE) / "validacion/diseno_protocol_freeze.json"):
        record = json.loads((study / rel).read_text())
        for name, expected in record["source_sha256"].items():
            if rel.parts[0] == PHASE:
                path = study / PHASE / (name if name.startswith("scripts/") else "scripts/" + name)
                if not path.exists():
                    path = study / name
            else:
                path = (study / "resultados/simulaciones/source_snapshots/sparse_v2/models.py"
                        if name == "models.py" else study / "scripts/competitors" / name)
            if not path.is_file() or sha(path) != expected:
                failures.append(str(path.relative_to(study)))
            else:
                checked.append(str(path.relative_to(study)))
    if failures:
        raise RuntimeError("Frozen sources missing/different: " + ", ".join(sorted(set(failures))))
    return sorted(set(checked))


def prepare(args):
    study = args.run_root / "study"
    proof = args.run_root / "legacy_preparation.json"
    if proof.exists():
        checked = check_sources(study)
        record = json.loads(proof.read_text())
        for rel, digest in record["source_sha256"].items():
            if sha(study / rel) != digest:
                raise RuntimeError("Prepared source changed: " + rel)
        return study
    if study.exists():
        raise FileExistsError("Incomplete prepared workspace must be investigated: " + str(study))
    args.run_root.mkdir(parents=True, exist_ok=True)
    study.mkdir()
    source = args.source_root
    copy_source_tree(source / "scripts", study / "scripts")
    copy_source_tree(source / PHASE / "scripts", study / PHASE / "scripts")
    # The old portable package holds the public SAV preparation adapter.
    codigo = source / "codigo"
    if not codigo.exists():
        codigo = args.reference_root / "paquete_comparadores_publicacion_2026-10-06_entrega/codigo"
    copy_source_tree(codigo, study / "codigo")
    for branch in ("finite_v2", "sparse_v2"):
        rel = Path("resultados/simulaciones/source_snapshots") / branch
        snapshots = source / rel
        if not snapshots.exists():
            snapshots = args.reference_root / rel
        shutil.copytree(snapshots, study / rel,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for rel in metadata_files(args.reference_root):
        original = args.reference_root / rel
        if original.is_file():
            dest = study / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, dest)
    for name in ("processed/aux_all_districts.csv", "auxiliares/aux_all_districts.csv",
                 "provenance/source_manifest.json"):
        original = args.reference_root / "datos" / name
        if not original.exists():
            original = args.reference_root / "paquete_comparadores_publicacion_2026-10-06_entrega/datos" / name
        if original.exists():
            dest = study / "datos" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, dest)
    # Shared dependencies are linked only in this PRIVATE run directory.
    if args.cmdstan:
        dest = study / ".competitor_env/cmdstan/cmdstan-2.40.0"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(args.cmdstan, target_is_directory=True)
    rlib = args.r_lib or args.reference_root / ".Rlib"
    if rlib.exists():
        (study / ".Rlib").symlink_to(rlib.resolve(), target_is_directory=True)
    (study / "validacion").mkdir(exist_ok=True)
    (study / PHASE / "validacion").mkdir(exist_ok=True)
    sources = {str(p.relative_to(study)): sha(p) for p in study.rglob("*")
               if p.is_file() and p.suffix in SUFFIXES and ".competitor_env" not in p.parts}
    checked = check_sources(study)
    save(proof, {"status": "PREPARED_NO_MODEL_FITS", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "source_sha256": sources, "frozen_sources_checked": checked,
                 "private_records_copied": False, "finished_predictions_copied": False,
                 "reference_root": str(args.reference_root), "source_root": str(source),
                 "study_root": str(study), "previous_responses_are_reused_data": True,
                 "statistical_settings_changed": False,
                 "operational_adapters": ["external CmdStan/R dependency routes", "NumPy metadata serialization",
                    "fresh data preparation replaces copied processed results", "new runtime task paths and prediction locks"]})
    return study


def plan(args, study):
    worker = HERE / "legacy_worker.py"
    def helper(action, *rest):
        return [sys.executable, str(worker), "--study-root", str(study), "--reference-root", str(args.reference_root),
                "--shared-private-root", str(args.shared_private_root), action, *map(str, rest)]
    def script(rel, *rest):
        return helper("script", rel, *rest)
    c = "scripts/competitors/"
    p = PHASE + "/scripts/"
    aux = study / PHASE / "datos/geografia/aux_verified_2017_2018.csv"
    pre2019 = study / PHASE / "datos/private_external_prepared_2019_full"
    pre2024 = study / PHASE / "datos/private_external_prepared_2024_training"
    previous = [helper("compile"), script(c+"benchmark_prepare_final.py", "--workers", args.workers),
        script(c+"benchmark_run.py", "--tag", "final_primary_v2", "--workers", args.workers,
               "--scenarios", "binary_informative_mar", "sparse_srs_2025", "sparse_psu_2025", "--replications", *range(10000,10200)),
        script(c+"benchmark_run.py", "--tag", "final_structural_v2", "--workers", args.workers,
               "--scenarios", "sparse_srs_iid", "sparse_psu_bym2", "--replications", *range(20000,20100)),
        script(c+"gao_dispatch_final.py", "--tasks", study/"resultados/resultados_comparadores/task_plan/gao_all1800.csv",
               "--batch-label", "fresh_reproduction_gao", "--workers", args.workers),
        script(c+"benchmark_collect.py", "--tasks", study/"resultados/resultados_comparadores/task_plan/gao_all1800.csv", "--tag", "final_gao_evaluation_v1"),
        script(c+"hogg_benchmark_batch.py", "--tasks", study/"resultados/resultados_comparadores/task_plan/hogg_all1100.csv",
               "--tag", "fresh_reproduction_hogg", "--workers", args.workers),
        script("scripts/public_competitors/hogg_collect_tasks.py", study/"resultados/competidores_hogg/fresh_reproduction_hogg/task_output_contract.csv",
               study/"resultados/resultados_comparadores/task_plan/hogg_collector1100.csv"),
        script(c+"benchmark_collect.py", "--tasks", study/"resultados/resultados_comparadores/task_plan/hogg_collector1100.csv", "--tag", "final_hogg_evaluation_v1"),
        script(c+"benchmark_finalize.py"), script(c+"benchmark_spatial_evaluate.py", "--prepare-only")]
    previous += [script(c+"hogg_spatial_numeric.py", "--driver", "batch", "--start", 20000+25*k,
                        "--stop", 20000+25*(k+1), "--tag", f"spatial_sensitivity_{k:02d}") for k in range(4)]
    previous += [script(c+"benchmark_schedule_spatial_evaluation.py"),
                 script(c+"benchmark_validate_final_results.py"), script(c+"benchmark_boundary_audit.py", "--include-spatial"),
                 script(c+"benchmark_independent_final_audit.py")]
    cut40 = [helper("compile"), helper("cut40_workers", args.workers), script(p+"sim_design_diagnostics.py", "collect")]
    app2025 = [helper("prepare_actual"), script(p+"diseno_prepare_development.py", "--year", 2025,
               "--aux", aux, "--output", study/PHASE/"datos/diseno/2025_full_official"),
               script(p+"application2025_baseline.py"),
               script(p+"diseno_fit_cut.py", "--year", 2025, "--method", "cut8", "--prepared", study/PHASE/"datos/diseno/2025_full_official",
                      "--output", study/PHASE/"resultados/application2025_designcut8"), script(p+"phase2_application_summary.py")]
    external = [helper("compile"), helper("prepare_actual"),
        script(p+"validacion_prepare_training.py", "--year", 2019, "--aux", aux, "--output", pre2019)]
    for method in ("baseline", "hogg"):
        external.append(script(p+"validacion_fit_models.py", "--method", method, "--year", 2019,
            "--units", study/"datos/private_competitors/children_2019.csv", "--aux", aux, "--prepared", pre2019,
            "--output", study/PHASE/f"resultados/externa_temporal_2019_{method}", "--parallel-chains", 1))
    external += [script(p+"diseno_fit_cut.py", "--year", 2019, "--method", "cut8", "--prepared", pre2019,
                        "--output", study/PHASE/"resultados/externa_temporal_2019_designcut8"),
                 helper("lock", "temporal"), helper("import2024"),
                 script(p+"validacion_prepare_training.py", "--year", 2024, "--aux", aux, "--output", pre2024)]
    for method in ("baseline", "hogg"):
        external.append(script(p+"validacion_fit_models.py", "--method", method, "--year", 2024,
            "--units", study/PHASE/"datos/private_external/children_2024_training.csv", "--aux", aux, "--prepared", pre2024,
            "--output", study/PHASE/f"resultados/externa_territorial_2024_{method}", "--parallel-chains", 1))
    external += [script(p+"validacion_fit_design_reserved.py", "--prepared", pre2024, "--aux", aux,
                        "--output", study/PHASE/"resultados/externa_territorial_2024_designcut8", "--state", "all"),
                 helper("lock", "territorial"), script(p+"validacion_score_external.py")]
    cut16 = [helper("compile"), helper("reuse_cut8_for_cut16"),
        script(p+"diseno_fit_cut.py", "--year", 2019, "--method", "cut16", "--prepared", pre2019,
               "--output", study/PHASE/"resultados/diseno/2019_full_official/cut16"), helper("compare_cut16")]
    return {"previous800": previous, "cut40": cut40, "application2025": app2025, "external2024": external, "cut16": cut16}


def preflight(args, study):
    result = {"frozen_sources": len(check_sources(study)), "python": sys.executable,
              "cmdstan": str(args.cmdstan), "model_fits_dispatched": False}
    check = subprocess.run([sys.executable, "-c", "import numpy,pandas,scipy,cmdstanpy; import json; print(json.dumps({'numpy':numpy.__version__,'pandas':pandas.__version__,'scipy':scipy.__version__,'cmdstanpy':cmdstanpy.__version__}))"],
                           capture_output=True, text=True)
    result["python_dependencies"] = json.loads(check.stdout) if check.returncode == 0 else {"error": check.stderr[-2000:]}
    result["cmdstan_makefile_present"] = bool(args.cmdstan and (args.cmdstan / "makefile").is_file())
    rcode = "p<-c('haven','survey','jsonlite','digest','Matrix','INLA','readxl'); for(x in p)cat(x,if(requireNamespace(x,quietly=TRUE))as.character(packageVersion(x))else 'MISSING','\\n')"
    env = os.environ.copy()
    if (study / ".Rlib").exists():
        rcode = ".libPaths(c(" + json.dumps(str((study / ".Rlib").resolve())) + ",.libPaths()));" + rcode
    try:
        r = subprocess.run(["Rscript", "-e", rcode], env=env, capture_output=True, text=True)
        result["R_dependencies"] = r.stdout.splitlines()
        result["R_dependency_probe_exit_code"] = r.returncode
        result["R_probe_stderr"] = r.stderr[-1000:]
    except FileNotFoundError:
        result["R_dependencies"] = ["Rscript MISSING"]
    result["status"] = ("PASS_NO_FITS" if check.returncode == 0 and result["cmdstan_makefile_present"]
                        and not any("MISSING" in x for x in result["R_dependencies"]) else "DEPENDENCIES_INCOMPLETE_NO_FITS")
    save(args.run_root / "legacy_preflight.json", result)
    return result


def execute(args, commands):
    stages = STAGES if args.stage == "all" else (args.stage,)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"
    env["ENDES_STUDY_ROOT"] = str(args.run_root / "study")
    env["ENDES_EXPORT_PRIVATE_INPUTS"] = "1"
    if (args.run_root / "study/.Rlib").exists():
        env["ENDES_R_LIB"] = str((args.run_root / "study/.Rlib").resolve())
    for stage in stages:
        for index, command in enumerate(commands[stage]):
            receipt = args.run_root / "receipts" / stage / f"{index:03d}.json"
            if receipt.exists():
                old = json.loads(receipt.read_text())
                if old["command"] != command or old["exit_code"] != 0:
                    raise RuntimeError("Investigate previous incomplete/changed command: " + str(receipt))
                continue
            log = args.run_root / "logs" / stage / f"{index:03d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            print(json.dumps({"stage": stage, "command_index": index, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "log": str(log)}), flush=True)
            with log.open("x") as handle:
                result = subprocess.run(command, cwd=args.run_root / "study", env=env, stdout=handle, stderr=subprocess.STDOUT)
            save(receipt, {"command": command, "exit_code": result.returncode, "log_sha256": sha(log),
                           "operational_helper_sha256": {name: sha(HERE / name) for name in
                               ("legacy_runner.py", "legacy_worker.py")},
                           "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
            if result.returncode:
                raise RuntimeError("Command failed; preserve outputs and inspect " + str(log))
        save(args.run_root / f"legacy_{stage}_execution.json", {"status": "ALL_COMMANDS_EXITED_ZERO",
             "stage": stage, "commands": len(commands[stage]), "scientific_comparison_still_required": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cmdstan", type=Path)
    parser.add_argument("--r-lib", type=Path)
    parser.add_argument("--shared-private-root", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--action", choices=("prepare", "preflight", "plan", "run"), default="prepare")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    args = parser.parse_args()
    for key in ("reference_root", "source_root", "run_root", "cmdstan", "r_lib", "shared_private_root"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    repository = HERE.parent.resolve()
    if args.run_root == repository or args.run_root.is_relative_to(repository):
        raise ValueError("Private run root must be outside the public repository")
    if args.shared_private_root is None:
        args.shared_private_root = args.run_root.parent / "real_private_study"
    study = prepare(args)
    commands = plan(args, study)
    save(args.run_root / "legacy_commands.json", commands)
    if args.action == "preflight":
        print(json.dumps(preflight(args, study), indent=2))
    elif args.action == "run":
        execute(args, commands)
    else:
        print(json.dumps({"status": "PREPARED_NO_FITS", "plan": str(args.run_root / "legacy_commands.json"),
                          "commands_by_stage": {k: len(v) for k, v in commands.items()}, "workers": args.workers}, indent=2))


if __name__ == "__main__":
    main()
