"""Isolated full ICC refit using the public pre-outcome frozen census basis."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import sikov_worker as shared


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--context", choices=shared.CONTEXTS, default="SAR_logit_psu_sparse")
    p.add_argument("--rep", type=int, default=30000)
    a = p.parse_args(); cfg = json.loads(a.config.read_text())
    ctx = shared.initialize(cfg["source_root"], cfg["reference_root"], cfg["sikov_run_root"], cfg["beta_exe"], cfg["cmdstan"])
    out = Path(cfg["run_base"]) / "diagnostics" / ("canonical_full_ICC_"+a.context+"_"+str(a.rep))
    if out.exists():
        raise FileExistsError("Preserve an existing isolated diagnostic: "+str(out))
    out.mkdir(parents=True)
    inputs = ctx["run"] / "evaluacion/inputs" / a.context / f"rep_{a.rep:05d}/model_inputs"
    d = dict(np.load(inputs / "inputs.npz", allow_pickle=False)); units = pd.read_csv(inputs / "sample.csv")
    census_path = ctx["reference"] / "datos/simulation_template/auxiliary.npz"
    census = dict(np.load(census_path, allow_pickle=False))
    for key in ("X_raw", "X", "coords", "W", "strata"):
        if not np.array_equal(d[key], census[key]):
            raise RuntimeError("Fresh training/census geometry alignment differs: "+key)
    basis_difference = float(np.max(np.abs(d["basis"]-census["basis"])))
    # This is the previously frozen, publicly releasable census geometry,
    # not a learned posterior, initialization, chain or prediction cache.
    d["basis"] = census["basis"].copy()
    np.savez_compressed(out / "canonical_fit_inputs.npz", **d)
    seed = int(np.random.SeedSequence([20261010,811,shared.CONTEXTS.index(a.context),a.rep]).generate_state(1)[0]) % (2**32-200000)
    base = ctx["modules"]["comparison_evaluate"]
    plan = {"role":"isolated canonical-census ICC diagnostic, does not count in production cohort",
        "context":a.context,"replication":a.rep,"seed":seed,"sampler_config":base.SAMPLER,
        "source_census_geometry_sha256":shared.sha(census_path),"basis_entry_difference_from_fresh_ARPACK":basis_difference,
        "fresh_training_inputs_sha256":shared.sha(inputs/"inputs.npz"),"fresh_sample_sha256":shared.sha(inputs/"sample.csv"),
        "canonical_fit_inputs_sha256":shared.sha(out/"canonical_fit_inputs.npz"),
        "source_hashes_sha256":shared.source_gate(),"source_compiled_executable_sha256":shared.sha(cfg["beta_exe"]),
        "truth_or_scores_read":False,"old_fitted_values_or_MAP_or_metric_or_chains_used_for_fitting":False,
        "adapter_sha256":shared.sha(__file__)}
    shared.write_json(out/"diagnostic_plan.json",plan)
    shared.set_chain_output(out/"chains")
    base.get_backend(); start = time.perf_counter()
    fitted = base.hmc.fit(d["X"],d["y"],d["n_eff"],d["sample_idx"],units,d["basis"],seed=seed,sampler_config=base.SAMPLER)
    arrays = {"mean":fitted["p_mean"],"lower":fitted["lower95"],"upper":fitted["upper95"],"mse":fitted["p_variance"]}
    np.savez_compressed(out/"ICC.npz",**arrays)
    np.savez_compressed(out/"ICC_conditional_draws.npz",p_draws=fitted["p_draws"],anemia_marginal=fitted["marginal_draws"][0],stunting_marginal=fitted["marginal_draws"][1])
    shared.write_json(out/"ICC_status.json",{"converged":all(x["success"] and x["beta_shape_guard_cells"]==0 for x in fitted["status"]),"endpoints":fitted["status"]})
    archive = ctx["reference"] / "evaluacion/fits" / a.context / f"rep_{a.rep:05d}"
    compared = {}
    for filename in ("ICC.npz","ICC_conditional_draws.npz"):
        new=dict(np.load(out/filename,allow_pickle=False));old=dict(np.load(archive/filename,allow_pickle=False))
        if set(new)!=set(old):
            raise RuntimeError("Prediction/draw array universe differs: "+filename)
        compared[filename]={key:{"exact_numeric_identity":bool(np.array_equal(new[key],old[key])),
            "maximum_absolute_difference":float(np.max(np.abs(new[key]-old[key])))} for key in new}
    result={**plan,"status":"COMPLETED_ISOLATED_CANONICAL_FULL_ICC_REFIT","seconds":time.perf_counter()-start,
        "all_NUTS_endpoints_passed":all(x["success"] and x["beta_shape_guard_cells"]==0 for x in fitted["status"]),
        "predictions_and_draws_comparison":compared,
        "all_compared_arrays_exact":all(v["exact_numeric_identity"] for file in compared.values() for v in file.values()),
        "all_output_sha256":{str(path.relative_to(out)):shared.sha(path) for path in out.rglob("*") if path.is_file()}}
    shared.write_json(out/"diagnostic_completion.json",result)
    print(json.dumps({k:result[k] for k in ["status","seconds","all_NUTS_endpoints_passed","all_compared_arrays_exact"]}),flush=True)
    print(json.dumps(compared),flush=True)


if __name__ == "__main__":
    main()
