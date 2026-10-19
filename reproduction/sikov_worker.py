"""Fresh-output orchestration of the unchanged Sikov/ICC statistical kernels.

The historical source and reference trees are read-only. This adapter changes
paths, retains chain files, and uses one externally compiled Stan executable.
It never changes formulas, random streams, the selected recipe, or fit gates.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

# CLI execution and helpers must share one initializer state.
sys.modules.setdefault("sikov_worker", sys.modules[__name__])

import numpy as np
import pandas as pd

_CTX = None
_CHAIN_OUTPUT = None
_BACKEND = None
_ORIGINAL_HMC_FIT = None
CONTEXTS = ("SAR_probability_srs", "SAR_logit_srs", "SAR_logit_psu_sparse", "IID_logit_psu_sparse")
METHODS = ("Sikov_FH", "Sikov_SAR", "Sikov_FH_BC", "Sikov_SAR_BC", "Sikov_SAR_k4", "Sikov_SAR_k8", "Sikov_SAR_haversine", "ICC", "Selected", "Sikov_SAR_Published", "Sikov_SAR_Published_BC")
RECIPE = {"family": "ICC_SAR_mixture", "icc_weight": .75}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x:
        x.tolist() if isinstance(x, np.ndarray) else x.item() if isinstance(x, np.generic) else str(x)) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def article_root(path):
    path = Path(path).resolve()
    if (path / "sikov_2026-10-10/scripts").is_dir():
        return path / "sikov_2026-10-10", path
    if (path / "scripts/comparison_simulation.py").is_file():
        return path, path.parent
    raise FileNotFoundError("Expected Endes1 study root or its sikov_2026-10-10 subdirectory: " + str(path))


def mapped_source(path, article, study):
    path = Path(path)
    if "sikov_2026-10-10" in path.parts:
        return article.joinpath(*path.parts[path.parts.index("sikov_2026-10-10") + 1:])
    if "estudio_publicacion" in path.parts:
        return study.joinpath(*path.parts[path.parts.index("estudio_publicacion") + 1:])
    if not path.is_absolute():
        return article / path
    raise ValueError("Unsupported historical provenance path: " + str(path))


def source_gate():
    """Check the actual imported 37 sources against both historical seals."""
    c = _CTX
    if c is None:
        raise RuntimeError("initialize() is required")
    if sha(c["reference"] / "protocol_freeze.json") != c["protocol_sha256"] or \
            sha(c["reference"] / "validacion/sikov_published_addendum_freeze.json") != c["addendum_sha256"]:
        raise RuntimeError("Historical scientific protocol changed during reproduction")
    identities = {**c["protocol"]["statistical_sources_sha256"], **c["addendum"]["additional_sources_sha256"]}
    if len(identities) != 37 or c["protocol"]["selected_recipe"] != RECIPE:
        raise RuntimeError("The 37-source/75:25 frozen scientific contract differs")
    bindings = {}
    for old, digest in identities.items():
        current = mapped_source(old, c["source"], c["study"])
        if not current.is_file() or sha(current) != digest:
            raise RuntimeError("Frozen statistical source mismatch: " + str(current))
        bindings[str(current)] = digest
    return bindings


def set_chain_output(out):
    global _CHAIN_OUTPUT
    _CHAIN_OUTPUT = Path(out).resolve() if out is not None else None


def _retained_hmc_fit(*args, **kwargs):
    if _CHAIN_OUTPUT is None:
        raise RuntimeError("Every NUTS fit requires a fresh explicit chain-output directory")
    kwargs.setdefault("output_dir", _CHAIN_OUTPUT)
    return _ORIGINAL_HMC_FIT(*args, **kwargs)


def _backend():
    global _BACKEND
    if _BACKEND is None:
        import cmdstanpy
        c = _CTX
        if not c["exe"].is_file():
            raise FileNotFoundError("Compile the shared Stan runtime before fitting: " + str(c["exe"]))
        cmdstanpy.set_cmdstan_path(str(c["cmdstan"]))
        model = cmdstanpy.CmdStanModel(stan_file=str(c["study"] / "scripts/competitors/beta_power_hybrid.stan"),
                                      exe_file=str(c["exe"]), compile=False)
        _BACKEND = c["modules"]["comparison_evaluate"].DiagonalMetric(model)
    _CTX["modules"]["comparison_evaluate"].hmc._MODEL = _BACKEND
    return _BACKEND.model


def initialize(source_root, reference_root, run_root, exe, cmdstan, chain_output=None):
    """Shared initializer for simulation and actual-data worker processes."""
    global _CTX, _ORIGINAL_HMC_FIT, _BACKEND
    source, study = article_root(source_root)
    reference, reference_study = article_root(reference_root)
    run = Path(run_root).resolve()
    if run == reference or run == source or run.is_relative_to(source) or run.is_relative_to(reference):
        raise ValueError("Fresh output must be outside both immutable source/reference article trees")
    for p in (study / "scripts/competitors", study / "scripts", source / "scripts"):
        sys.path.insert(0, str(p))
    names = ("comparison_simulation", "improve_development", "improve_native_development", "improve_select",
             "comparison_evaluate", "comparison_real", "sikov_published_fit", "sensitivity_sampled_graph_generate",
             "sensitivity_sampled_graph_score", "comparison_collect", "sikov_published_collect",
             "sikov_poverty_stratum_descriptive", "sikov_published_b300_sensitivity")
    modules = {name: importlib.import_module(name) for name in names}
    protocol = read_json(reference / "protocol_freeze.json")
    addendum = read_json(reference / "validacion/sikov_published_addendum_freeze.json")
    plan = read_json(reference / "validacion/sensitivity_sampled_graph_freeze.json")
    _CTX = dict(source=source, study=study, reference=reference, reference_study=reference_study, run=run,
                exe=Path(exe).resolve(), cmdstan=Path(cmdstan).resolve(), protocol=protocol, addendum=addendum,
                plan=plan, protocol_sha256=sha(reference / "protocol_freeze.json"),
                addendum_sha256=sha(reference / "validacion/sikov_published_addendum_freeze.json"), modules=modules)
    source_gate()
    for module in modules.values():
        if hasattr(module, "ROOT"):
            module.ROOT = run
        if hasattr(module, "STUDY"):
            module.STUDY = study
    generator = modules["sensitivity_sampled_graph_generate"]
    generator.DEST = run / "sensibilidad_sampled_graph"
    # An already frozen protocol is replayed. No new approval or prospective
    # scientific freeze is invented; only its source identity is revalidated.
    generator.gate = lambda: (source_gate() and plan,
        sha(reference / "validacion/sensitivity_sampled_graph_freeze.json"), protocol)
    modules["comparison_evaluate"].get_backend = _backend
    hmc = modules["comparison_evaluate"].hmc
    if _ORIGINAL_HMC_FIT is None:
        _ORIGINAL_HMC_FIT = hmc.fit
    hmc.fit = _retained_hmc_fit
    _BACKEND = None
    set_chain_output(chain_output)
    return _CTX


def worker_init(options):
    initialize(**options)


def semantic_npz(left, right, tolerance=0.):
    """Zip metadata is irrelevant; every array name, shape and value is checked."""
    a = dict(np.load(left, allow_pickle=False)); b = dict(np.load(right, allow_pickle=False))
    if set(a) != set(b):
        raise RuntimeError("NPZ fields differ: " + str(left))
    differences = {}
    for key in a:
        x, y = a[key], b[key]
        if x.shape != y.shape:
            raise RuntimeError("NPZ shape differs: " + key)
        if x.dtype.kind in "fc" and y.dtype.kind in "fc":
            local_tolerance = tolerance.get(key, 0.) if isinstance(tolerance, dict) else tolerance
            passed = np.allclose(x, y, atol=local_tolerance, rtol=0., equal_nan=True)
            finite = np.isfinite(x) & np.isfinite(y)
            difference = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else 0.
        else:
            passed = np.array_equal(x, y); difference = 0.
        differences[key] = difference
        if not passed:
            raise RuntimeError("Regenerated input differs from reference: " + key + " in " + str(left))
    return differences


def output_bindings(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob("*")) if p.is_file() and p.name != "completion.json"}


def receipt(folder, record):
    record = {**record, "source_hashes_sha256": source_gate(), "operational_adapter_sha256": sha(__file__),
              "execution_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json(Path(folder) / "completion.json", record)
    return record


def validate_completion(folder, expected_methods=None):
    folder = Path(folder); record = read_json(folder / "completion.json")
    if record.get("protocol_sha256") != _CTX["protocol_sha256"]:
        raise RuntimeError("Completion refers to a different frozen scientific protocol: " + str(folder))
    for name, digest in record.get("output_sha256", {}).items():
        if sha(folder / name) != digest:
            raise RuntimeError("Completed output changed: " + str(folder / name))
    for name, digest in record.get("chain_files_sha256", {}).items():
        if sha(folder / name) != digest:
            raise RuntimeError("Retained chain output changed")
    if expected_methods is not None and set(record["convergence"]) != set(expected_methods):
        raise RuntimeError("Completed method universe differs")
    return record


def setup():
    c = _CTX; run = c["run"]; m = c["modules"]
    run.mkdir(parents=True, exist_ok=True)
    for name in ("datos", "validacion", "evaluacion", "desarrollo", "sensibilidad_sampled_graph"):
        (run / name).mkdir(exist_ok=True)
    template = run / "datos/simulation_template"
    census_input = c["source"] / "datos/census_only_full_template.npz"
    census_areas = c["source"] / "datos/aux_sikov14_official.csv"
    for path in (census_input, census_areas):
        reference_path = c["reference"] / "datos" / path.name
        if sha(path) != sha(reference_path):
            raise RuntimeError("Source census input does not match the archived input: " + str(path))
    if not template.exists():
        m["comparison_simulation"].prepare_template(census_input, census_areas)
    # ARPACK's fixed-v0 eigenbasis can differ by machine-roundoff after a fresh
    # runtime install. Keep every raw/design input exact and report basis error.
    input_tolerance = {"basis": 1e-12}
    template_check = semantic_npz(template / "auxiliary.npz", c["reference"] / "datos/simulation_template/auxiliary.npz", input_tolerance)
    checks = []
    for phase, ids in (("development", range(8)), ("evaluation", range(30000, 30064))):
        prefix = "desarrollo" if phase == "development" else "evaluacion"
        for context in CONTEXTS:
            for rep in ids:
                target = run / prefix / "inputs" / context / f"rep_{rep:05d}"
                if not target.exists():
                    m["comparison_simulation"].generate(context, rep, phase)
                reference = c["reference"] / prefix / "inputs" / context / f"rep_{rep:05d}"
                diff = semantic_npz(target / "model_inputs/inputs.npz", reference / "model_inputs/inputs.npz", input_tolerance)
                # Hash-only examination of truth is permitted here. Fitting
                # processes never load evaluation_only arrays.
                if sha(target / "model_inputs/sample.csv") != sha(reference / "model_inputs/sample.csv"):
                    raise RuntimeError("Regenerated sample.csv differs")
                checks.append({"phase": phase, "context": context, "rep": rep, "model_input_array_maxdiff": diff,
                               "truth_value_read_for_comparison": False})
    for rep in range(30000, 30064):
        target = run / "sensibilidad_sampled_graph/inputs" / f"rep_{rep:05d}"
        if not target.exists():
            m["sensitivity_sampled_graph_generate"].generate(rep)
        reference = c["reference"] / "sensibilidad_sampled_graph/inputs" / f"rep_{rep:05d}"
        diff = semantic_npz(target / "model_inputs/inputs.npz", reference / "model_inputs/inputs.npz", input_tolerance)
        if sha(target / "model_inputs/sample.csv") != sha(reference / "model_inputs/sample.csv"):
            raise RuntimeError("Regenerated sampled-graph sample differs")
        checks.append({"phase": "sampled_graph", "rep": rep, "model_input_array_maxdiff": diff,
                       "truth_value_read_for_comparison": False})
    record = {"status": "PASS_REGENERATED_352_WORLD_INPUTS", "template_array_maxdiff": template_check,
              "worlds": {"development": 32, "evaluation": 256, "sampled_graph": 64}, "checks": checks,
              "historical_protocol_sha256": c["protocol_sha256"], "historical_addendum_sha256": c["addendum_sha256"],
              "census_input_sha256": {str(path): sha(path) for path in (census_input, census_areas)},
              "semantic_input_absolute_tolerances": input_tolerance, "all_other_input_arrays_exact": True,
              "sources_sha256": source_gate(), "adapter_sha256": sha(__file__),
              "operational_changes": ["output ROOT redirected", "shared compile=False CmdStan executable", "retained chain directory", "fresh execution receipts"],
              "kernels_or_seeds_or_numerical_gates_changed": False, "archived_predictions_used_for_fitting": False}
    write_json(run / "validacion/sikov_setup.json", record)
    print(json.dumps({"status": record["status"], "worlds": record["worlds"]}), flush=True)


def development_one(job):
    context, rep = job; c = _CTX; m = c["modules"]; run = c["run"]
    first = run / "desarrollo/recreated_laplace"; native = run / "desarrollo/recreated_native"
    target = native / "fit_results" / context / f"rep_{rep:05d}"
    if target.exists():
        raise FileExistsError("Preserve development fit, including any incomplete work: " + str(target))
    rows1, event1 = m["improve_development"].work((context, rep, str(first)))
    rows, event = m["improve_native_development"].work((context, rep, str(native), str(first), 100))
    write_json(target / "fresh_development_metrics.json", {"rows": rows, "event": event,
                                                          "laplace_rows": rows1, "laplace_event": event1})
    print(json.dumps({"stage": "development", "context": context, "rep": rep, "seconds": event["seconds"]}), flush=True)
    return event


def finalize_development():
    c = _CTX; m = c["modules"]; folder = c["run"] / "desarrollo/recreated_native"
    rows = []
    for context in CONTEXTS:
        for rep in range(8):
            path = folder / "fit_results" / context / f"rep_{rep:05d}/fresh_development_metrics.json"
            rows.extend(read_json(path)["rows"])
    from improve_candidates import CANDIDATES
    plan = {"phase": "development", "independent_final_outcomes_excluded": True,
            "candidate_methods": list(CANDIDATES), "scenarios": list(CONTEXTS),
            "development_replication_ids": list(range(8)), "candidate_recipes": CANDIDATES,
            "sources_sha256": source_gate(), "reference_B": 100}
    frame = pd.DataFrame(rows).sort_values(["scenario", "replication", "method", "outcome", "group"])
    selection, ranking, held = m["improve_select"].select(frame[frame.method.isin(CANDIDATES)], plan)
    if selection["selected_method"] != "ICC_SAR_w75" or selection["selected_recipe"] != RECIPE:
        raise RuntimeError("Fresh development did not reproduce the manuscript's frozen 75:25 selection")
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "replications.csv", index=False)
    ranking.to_csv(folder / "candidate_ranking.csv", index=False)
    held.to_csv(folder / "LOPO_selection_rule_metrics.csv", index=False)
    write_json(folder / "development_plan.json", plan)
    write_json(folder / "selected_recipe.json", selection)
    write_json(folder / "completion.json", {"complete": True, "completed_worlds": 32, "expected_worlds": 32,
        "source_hashes_sha256": source_gate(), "fresh_evaluation_outcomes_used": False,
        "matches_frozen_selected_recipe": True, "statistical_kernels_changed": False})
    print(json.dumps({"stage": "development_selection", "selected_method": selection["selected_method"]}), flush=True)
    return selection


def eval_one(context, rep, sampled=False):
    c = _CTX; run = c["run"]; m = c["modules"]
    if context not in CONTEXTS or rep not in range(30000, 30064):
        raise ValueError("Unplanned world")
    if sampled:
        source = run / "sensibilidad_sampled_graph/inputs" / f"rep_{rep:05d}"
        out = run / "sensibilidad_sampled_graph/fits" / f"rep_{rep:05d}"
        original, published = out / "original", out / "published"
        seed = int(np.random.SeedSequence([20261010, 973, rep]).generate_state(1)[0]) % (2**32 - 200000)
        pubseed = int(np.random.SeedSequence([20261010, 974, rep]).generate_state(1)[0])
    else:
        source = run / "evaluacion/inputs" / context / f"rep_{rep:05d}"
        original = run / "evaluacion/fits" / context / f"rep_{rep:05d}"
        published = run / "evaluacion/published_sar" / context / f"rep_{rep:05d}"
        out = original
        seed = int(np.random.SeedSequence([20261010, 811, CONTEXTS.index(context), rep]).generate_state(1)[0]) % (2**32 - 200000)
        pubseed = int(np.random.SeedSequence([20261010, 941, CONTEXTS.index(context), rep]).generate_state(1)[0])
    manifest = read_json(source / "manifest.json")
    if sha(source / "model_inputs/inputs.npz") != manifest["fit_inputs_sha256"] or sha(source / "model_inputs/sample.csv") != manifest["sample_sha256"]:
        raise RuntimeError("Regenerated fitting inputs changed")
    source_gate(); start = time.perf_counter()
    d = dict(np.load(source / "model_inputs/inputs.npz", allow_pickle=False)); units = pd.read_csv(source / "model_inputs/sample.csv")
    if (original / "completion.json").is_file():
        old = validate_completion(original, METHODS[:9])
    else:
        if original.exists():
            raise FileExistsError("Preserve incomplete original fit: " + str(original))
        set_chain_output(original / "hmc_chains")
        status = m["comparison_evaluate"].fits(d, units, original, seed, RECIPE)
        old = receipt(original, {"context": context, "replication": rep, "protocol_sha256": c["protocol_sha256"],
            "fit_inputs_sha256": manifest["fit_inputs_sha256"], "sample_sha256": manifest["sample_sha256"],
            "input_manifest_sha256": sha(source / "manifest.json"), "truth_read_by_fits": False,
            "seconds": time.perf_counter() - start, "seed": seed, "selected_recipe": RECIPE,
            "convergence": {k: v["converged"] for k, v in status.items()},
            "output_sha256": {p.name: sha(p) for p in original.iterdir() if p.is_file()},
            "chain_files_sha256": {str(p.relative_to(original)): sha(p) for p in (original / "hmc_chains").rglob("*") if p.is_file()},
            "runtime_executable_sha256": sha(c["exe"])})
    if (published / "completion.json").is_file():
        pub = validate_completion(published, METHODS[9:])
    else:
        if published.exists():
            raise FileExistsError("Preserve incomplete Published fit: " + str(published))
        tick = time.perf_counter(); _, _, status = m["sikov_published_fit"].fit_arrays(d, published, pubseed)
        pub = receipt(published, {"context": context, "replication": rep, "protocol_sha256": c["protocol_sha256"],
            "addendum_sha256": c["addendum_sha256"], "fit_inputs_sha256": manifest["fit_inputs_sha256"],
            "sample_sha256": manifest["sample_sha256"], "input_manifest_sha256": sha(source / "manifest.json"),
            "original_completion_sha256": sha(original / "completion.json"), "truth_read_by_fits": False,
            "seconds": time.perf_counter() - tick, "bootstrap_seed": pubseed,
            "convergence": dict.fromkeys(METHODS[9:], status["converged"]),
            "output_sha256": {p.name: sha(p) for p in published.iterdir() if p.is_file()}})
    convergence = {**old["convergence"], **pub["convergence"]}
    if sampled:
        receipt(out, {"context": "SAR_probability_sampled_graph_srs", "replication": rep,
            "protocol_sha256": c["protocol_sha256"], "addendum_sha256": c["addendum_sha256"],
            "truth_read_by_fits": False, "seconds": old["seconds"] + pub["seconds"], "convergence": convergence,
            "output_sha256": {"original/completion.json": sha(original / "completion.json"), "published/completion.json": sha(published / "completion.json")}})
    source_gate()
    result = {"stage": "sampled-one" if sampled else "eval-one", "context": context, "rep": rep,
              "all_converged": all(convergence.values()), "convergence": convergence, "seconds": time.perf_counter() - start}
    print(json.dumps(result), flush=True)
    if not result["all_converged"]:
        raise RuntimeError("Fit gates failed; every result was retained for diagnosis")
    return result


def b300_one(context, rep):
    c = _CTX; run = c["run"]
    if context not in CONTEXTS or rep not in (30000, 30001):
        raise ValueError("B300 subset is fixed to eight populations")
    # The original full-graph B300 contract requires the entire main prediction
    # cohort to close before this descriptive native refit. No truth is read.
    main_bindings = []
    for main_context in CONTEXTS:
        for main_rep in range(30000, 30064):
            for family, methods in (("fits", METHODS[:9]), ("published_sar", METHODS[9:])):
                main_folder = run / "evaluacion" / family / main_context / f"rep_{main_rep:05d}"
                main_record = validate_completion(main_folder, methods)
                if main_record["truth_read_by_fits"] is not False:
                    raise RuntimeError("Invalid closed main prediction role before B300")
                main_bindings.append({"path": str(main_folder / "completion.json"), "sha256": sha(main_folder / "completion.json")})
    original = run / "evaluacion/fits" / context / f"rep_{rep:05d}"
    published = run / "evaluacion/published_sar" / context / f"rep_{rep:05d}"
    validate_completion(original, METHODS[:9]); validate_completion(published, METHODS[9:])
    source = run / "evaluacion/inputs" / context / f"rep_{rep:05d}"
    d = dict(np.load(source / "model_inputs/inputs.npz", allow_pickle=False))
    folder = run / "evaluacion/b300" / context / f"rep_{rep:05d}"
    if folder.exists():
        raise FileExistsError("Preserve existing B300 work: " + str(folder))
    folder.mkdir(parents=True); tick = time.perf_counter()
    from sikov_bootstrap import fit_sikov_bootstrap
    from sikov_published_reference import fit_sikov_published_bootstrap
    keys = ("mean", "lower", "upper", "mse", "mse_plugin", "mse_delta", "mse_naive", "mse_bc", "naive_mcse", "bc_mcse")
    pubseed = int(np.random.SeedSequence([20261010, 975, CONTEXTS.index(context), rep]).generate_state(1)[0])
    convergence = {}
    for family in ("full_graph", "published"):
        values = []
        for j, endpoint in enumerate(("anemia", "stunting")):
            mask = np.isfinite(d["y"][:, j]) & np.isfinite(d["D"][:, j, j]) & (d["D"][:, j, j] > 0)
            common = dict(outcome=endpoint, covariate_names=d["covariate_names"].astype(str).tolist(), B=300, interval_estimator="naive")
            if family == "full_graph":
                values.append(fit_sikov_bootstrap(d["X_raw"], d["y"][mask, j], d["D"][mask, j, j], d["sample_idx"][mask], d["W"], d["strata"], spatial=True,
                    seed=202610100000 + CONTEXTS.index(context)*1000000 + rep*2 + j, **common))
            else:
                values.append(fit_sikov_published_bootstrap(d["X_raw"], d["y"][mask, j], d["D"][mask, j, j], d["sample_idx"][mask], d["coords"], d["strata"], seed=pubseed + 1009*j, **common))
        arrays = {k: np.column_stack([v[k] for v in values]) for k in keys}
        baseline = np.load((original / "Sikov_SAR.npz") if family == "full_graph" else (published / "Sikov_SAR_Published.npz"), allow_pickle=False)
        difference = float(np.max(abs(arrays["mean"] - baseline["mean"])))
        if difference > 1e-10:
            raise RuntimeError("B300 altered the closed B100 point estimator")
        np.savez_compressed(folder / (family + ".npz"), **arrays)
        status = {"converged": all(v["converged"] for v in values), "fits": values, "point_max_difference_B100": difference}
        if family == "published":
            status["independent_native_audit"] = c["modules"]["sikov_published_b300_sensitivity"].audit_native(d, status, arrays, 300)
        write_json(folder / (family + "_status.json"), status)
        convergence[family] = status["converged"]
    record = receipt(folder, {"context": context, "replication": rep, "B": 300, "protocol_sha256": c["protocol_sha256"],
        "truth_read_by_fits": False, "convergence": convergence, "seconds": time.perf_counter() - tick,
        "main_complete_before_B300": True, "main_prediction_completion_bindings": main_bindings,
        "source_input_sha256": sha(source / "model_inputs/inputs.npz"), "original_completion_sha256": sha(original / "completion.json"),
        "published_completion_sha256": sha(published / "completion.json"), "output_sha256": output_bindings(folder)})
    print(json.dumps({"stage": "b300-one", "context": context, "rep": rep, "seconds": record["seconds"], "convergence": convergence}), flush=True)
    if not all(convergence.values()):
        raise RuntimeError("B300 numerical gate failed; outputs retained")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=("setup", "development", "eval-one", "sampled-one", "b300-one", "score"))
    p.add_argument("--source-root", type=Path, required=True); p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True); p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--cmdstan", type=Path, required=True); p.add_argument("--context", choices=CONTEXTS)
    p.add_argument("--rep", type=int); p.add_argument("--workers", type=int, default=1); p.add_argument("--finalize", action="store_true")
    a = p.parse_args(argv)
    options = {k: getattr(a, k) for k in ("source_root", "reference_root", "run_root", "exe", "cmdstan")}
    initialize(**options)
    if a.stage == "setup":
        setup()
    elif a.stage == "development":
        if a.finalize:
            finalize_development()
        elif a.context is not None and a.rep is not None:
            if a.rep not in range(8):
                raise ValueError("Development IDs are 0..7")
            development_one((a.context, a.rep))
        elif a.context is not None or a.rep is not None:
            p.error("Development single-world execution requires both --context and --rep")
        else:
            if not 1 <= a.workers <= 4:
                raise ValueError("Laplace development allows one to four workers")
            jobs = [(c, r) for c in CONTEXTS for r in range(8)]
            with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers, initializer=worker_init, initargs=(options,)) as pool:
                list(pool.map(development_one, jobs))
            finalize_development()
    elif a.stage in ("eval-one", "sampled-one", "b300-one"):
        if a.rep is None or (a.context is None and a.stage != "sampled-one"):
            p.error("A declared --context/--rep is required")
        if a.stage == "b300-one":
            b300_one(a.context, a.rep)
        else:
            eval_one(a.context or CONTEXTS[0], a.rep, sampled=a.stage == "sampled-one")
    else:
        from sikov_score import score_all
        score_all(_CTX)


if __name__ == "__main__":
    main()
