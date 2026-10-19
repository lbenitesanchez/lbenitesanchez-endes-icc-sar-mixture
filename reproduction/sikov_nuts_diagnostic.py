"""Isolated one-chain old/new-binary comparison with identical fixed inputs.

Never participates in the production cohort or scores population truth. Both
executables receive the same persisted Stan JSON, initial point, diagonal
inverse metric, chain id, seed and 1,000/1,000 warmup/sampling settings.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import cmdstanpy
import sikov_worker as shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--context", choices=shared.CONTEXTS, default="SAR_logit_psu_sparse")
    parser.add_argument("--rep", type=int, default=30000)
    parser.add_argument("--endpoint", choices=("anemia", "stunting"), default="stunting")
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    ctx = shared.initialize(cfg["source_root"], cfg["reference_root"], cfg["sikov_run_root"], cfg["beta_exe"], cfg["cmdstan"])
    out = Path(cfg["run_base"]) / "diagnostics" / ("fixed_input_binary_"+args.context+"_"+str(args.rep)+"_"+args.endpoint+args.output_suffix)
    if out.exists():
        raise FileExistsError("Preserve existing diagnostic work: "+str(out))
    out.mkdir(parents=True)
    from beta_power import CONFIG, EndpointPosterior, working_icc, cluster_effective_n
    inputs = ctx["run"] / "evaluacion/inputs" / args.context / f"rep_{args.rep:05d}/model_inputs"
    d = dict(np.load(inputs / "inputs.npz", allow_pickle=False)); units = pd.read_csv(inputs / "sample.csv")
    j = ("anemia", "stunting").index(args.endpoint)
    config = {**CONFIG, "implementation": "collapsed beta-power hyperposterior NUTS v1"}
    icc, working = working_icc(units, args.endpoint)
    n, _ = cluster_effective_n(units, args.endpoint, d["sample_idx"], d["n_eff"][:,j], icc)
    post = EndpointPosterior(d["X"], d["y"][:,j], n, d["sample_idx"], d["basis"], config).optimize()
    data = dict(A=len(d["X"]), P=post.X.shape[1], Q=post.B.shape[1], X=post.X, B=post.B,
                S=len(post.idx), idx=post.idx+1, successes=post.k, failures=post.n-post.k,
                **{k:config[k] for k in ("beta_sd", "intercept_sd", "spatial_sd", "log_kappa_mean", "log_kappa_sd")})
    metric = np.diag(post.cov).copy() if post.status["success"] and post.status["hessian_repairs"] == 0 else np.ones(len(post.mode))
    initial = dict(beta=post.mode[:post.X.shape[1]], gamma=post.mode[post.X.shape[1]:-1], log_kappa=post.mode[-1])
    fixed = out / "fixed_inputs"; fixed.mkdir()
    for name, values in (("data.json", data), ("initial.json", initial), ("diag_metric.json", {"inv_metric":metric})):
        cmdstanpy.write_stan_json(str(fixed/name), values)
    fixed_sha = {p.name:shared.sha(p) for p in fixed.iterdir()}
    seed = int(np.random.SeedSequence([20261010,811,shared.CONTEXTS.index(args.context),args.rep]).generate_state(1)[0]) % (2**32-200000) + 1009*j
    old_exe = ctx["reference_study"] / "scripts/competitors/beta_power_hybrid"
    new_exe = Path(cfg["beta_exe"])
    original_status = json.loads((ctx["reference"] / "evaluacion/fits" / args.context / f"rep_{args.rep:05d}/ICC_status.json").read_text())
    fresh_status = json.loads((ctx["run"] / "evaluacion/fits" / args.context / f"rep_{args.rep:05d}/ICC_status.json").read_text())
    settings = dict(seed=seed, chains=1, chain_ids=[1], parallel_chains=1, iter_warmup=1000,
        iter_sampling=1000, adapt_delta=.95, max_treedepth=12, metric="diag_e", save_warmup=False,
        sig_figs=18, show_progress=False, refresh=100)
    metadata = {"role": "isolated one-chain diagnostic; outside production cohort", "context": args.context,
        "replication": args.rep, "endpoint": args.endpoint, "truth_or_scores_read": False,
        "production_chains_changed": False, "production_samples_changed": False,
        "sampler_settings":settings, "fixed_input_sha256":fixed_sha, "working_ICC":working,
        "fresh_rebuilt_numerical_initialization":post.status,
        "archived_numerical_initialization":original_status["endpoints"][j]["numerical_initialization"],
        "production_fresh_numerical_initialization":fresh_status["endpoints"][j]["numerical_initialization"],
        "source_hashes_sha256":shared.source_gate(), "adapter_sha256":shared.sha(__file__)}
    shared.write_json(out / "diagnostic_plan.json", metadata)
    cmdstanpy.set_cmdstan_path(cfg["cmdstan"])
    runs = {}; frames = {}
    for label, executable in (("old",old_exe),("new",new_exe)):
        if {p.name:shared.sha(p) for p in fixed.iterdir()} != fixed_sha:
            raise RuntimeError("A fixed input changed between binary experiments")
        model = cmdstanpy.CmdStanModel(stan_file=str(ctx["study"] / "scripts/competitors/beta_power_hybrid.stan"),
                                      exe_file=str(executable), compile=False)
        start = time.perf_counter()
        fit = model.sample(data=str(fixed/"data.json"), inits=str(fixed/"initial.json"),
            inv_metric=str(fixed/"diag_metric.json"), output_dir=str(out/label), **settings)
        csv = Path(fit.runset.csv_files[0]); frame = pd.read_csv(csv, comment="#", float_precision="round_trip")
        frames[label] = frame
        runs[label] = {"executable":str(executable), "executable_sha256":shared.sha(executable),
            "csv":str(csv), "csv_sha256":shared.sha(csv), "seconds":time.perf_counter()-start,
            "samples":len(frame), "parameter_columns":frame.columns.tolist()}
        print(json.dumps({"diagnostic":label,"seconds":runs[label]["seconds"]}), flush=True)
    a, b = frames["old"], frames["new"]
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        raise RuntimeError("Binary experiments returned different sample universes")
    x, y = a.to_numpy(float), b.to_numpy(float); difference = np.abs(x-y)
    exact = np.array_equal(x,y)
    finite = np.isfinite(difference)
    first = np.argwhere(x != y)
    comparison = {"exact_numeric_draw_identity":bool(exact), "maximum_draw_absolute_difference":float(difference[finite].max()),
        "different_numeric_cells":int(np.count_nonzero(x != y)), "total_numeric_cells":int(x.size),
        "first_different_cell":None if not len(first) else {"sample_index":int(first[0,0]),"column":a.columns[int(first[0,1])]},
        "parameter_mean_max_absolute_difference":float(np.max(np.abs(x.mean(axis=0)-y.mean(axis=0)))),
        "column_mean_differences":{k:float(a[k].mean()-b[k].mean()) for k in a.columns},
        "single_chain_diagnostics_do_not_establish_production_convergence":True}
    result = {**metadata, "status":"COMPLETED_IDENTICAL_INPUT_OLD_NEW_BINARY_DIAGNOSTIC", "runs":runs,"comparison":comparison,
        "interpretation": "Binary replacement alone changed this chain with identical fixed inputs" if not exact else
            "Binary replacement alone did not change this chain; production inputs/initialization remain separate causes to investigate"}
    shared.write_json(out / "diagnostic_completion.json", result)
    print(json.dumps(comparison),flush=True)


if __name__ == "__main__":
    main()
