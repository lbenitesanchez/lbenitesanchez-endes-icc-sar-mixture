"""Fit calibrated beta-power point and fixed-weight CUT mixtures in phase2.

Every mixture state gets its own normalized conditional NUTS posterior.
Its probability stays fixed at 1/K regardless of likelihood normalization or
predictive performance. No old source, old posterior or held-out response is
modified. This runner is fail-closed until root's scientific protocol freeze.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from diseno_calibration import digest

PHASE = Path(__file__).resolve().parents[1]
STUDY = PHASE.parent
sys.path.insert(0, str(STUDY/"scripts/competitors"))
import beta_power_hmc as hmc
from beta_power_application_recovery import DiagonalMetric

SAMPLER = dict(metric="diag_e", adapt_delta=.95, retry_adapt_delta=.99,
               parallel_chains=1,
               initial_metric="diagonal of unrepaired positive MAP covariance; otherwise identity")


def freeze_gate():
    gate_path = PHASE/"validacion/diseno_protocol_freeze.json"
    if not gate_path.exists():
        raise RuntimeError("Root scientific freeze is absent; model dispatch forbidden")
    gate = json.loads(gate_path.read_text())
    if not gate.get("approved_for_model_dispatch", False):
        raise RuntimeError("Root has not approved design-model dispatch")
    for name, expected in gate["source_sha256"].items():
        source = PHASE/"scripts"/name
        if digest(source) != expected:
            raise RuntimeError("Frozen new source changed: "+name)
    original = json.loads((STUDY/"resultados/resultados_comparadores/protocol_freeze.json").read_text())
    for name, expected in original["source_sha256"].items():
        source = STUDY/"scripts"/name if name=="models.py" else STUDY/"scripts/competitors"/name
        if digest(source) != expected:
            raise RuntimeError("Frozen original kernel changed: "+name)
    return gate_path, gate


def conditional_fit(inputs, n, output, seed, label, config=None):
    freeze_gate()
    if not 0 < int(seed) < 2**32-104729-1009:
        raise ValueError("CmdStan seed and predeclared retry/endpoint offsets must fit uint32")
    output = Path(output)
    target=hashlib.sha256()
    for array in (inputs["X"],inputs["y"],n,inputs["sample_idx"],inputs["basis"]):
        value=np.ascontiguousarray(array)
        target.update(str(value.dtype).encode())
        target.update(json.dumps(value.shape).encode())
        target.update(value.tobytes())
    target.update(json.dumps(dict(seed=seed,sampler=SAMPLER,config={"cluster_adjustment":False,**(config or {})}),sort_keys=True).encode())
    target_sha256=target.hexdigest()
    if output.exists():
        completion = output/"completion.json"
        if completion.exists() and json.loads(completion.read_text()).get("success"):
            saved = np.load(output/"posterior_summary_and_draws.npz")
            status = json.loads((output/"fit_status.json").read_text())
            if status.get("conditional_target_sha256")!=target_sha256 or digest(output/"posterior_summary_and_draws.npz")!=json.loads(completion.read_text())["posterior_output_sha256"]:
                raise RuntimeError("Resume target or posterior bytes differ from completed conditional fit")
            return {key:saved[key] for key in saved.files}, status
        raise FileExistsError("Incomplete conditional output requires explicit investigation: "+str(output))
    output.mkdir(parents=True)
    current_model=hmc.get_model()
    if not isinstance(current_model,DiagonalMetric):
        hmc._MODEL=DiagonalMetric(current_model)
    start = time.perf_counter()
    fitted = hmc.fit(inputs["X"], inputs["y"], n, inputs["sample_idx"],
                     units=None, basis=inputs["basis"], seed=seed,
                     config={"cluster_adjustment":False, **(config or {})}, sampler_config=SAMPLER)
    status = dict(label=label, seed=seed, seconds=time.perf_counter()-start,
                  conditional_target_sha256=target_sha256,
                  end_points=fitted["status"], success=all(s["success"] and s["beta_shape_guard_cells"] == 0
                                                         for s in fitted["status"]),
                  original_outcome_prevalences_changed=False, prior_family_changed=False,
                  effective_n_fixed_within_this_conditional_fit=True)
    (output/"fit_status.json").write_text(json.dumps(status,indent=2))
    np.savez_compressed(output/"posterior_summary_and_draws.npz",
                        **{key:fitted[key] for key in ("p_mean","lower95","upper95","p_variance","p_draws")})
    (output/"completion.json").write_text(json.dumps(dict(success=status["success"],
          sampler_status_sha256=digest(output/"fit_status.json"),
          posterior_output_sha256=digest(output/"posterior_summary_and_draws.npz")),indent=2))
    if not status["success"]:
        raise RuntimeError("Conditional state failed the predeclared NUTS gates: "+label)
    return {key:fitted[key] for key in ("p_mean","lower95","upper95","p_variance","p_draws")}, status


def pool_equal_weight(states):
    """Law of total variance, equal marginal draw counts from every state."""
    count = min(len(s["p_draws"]) for s in states)
    means = np.stack([s["p_mean"] for s in states])
    variances = np.stack([s["p_variance"] for s in states])
    mean = means.mean(axis=0)
    variance = (variances+means**2).mean(axis=0)-mean**2
    # Deterministic thinning is computational only; a permutation spans chains
    # when retry draw counts differ. All marginal mean/variance summaries use
    # every conditional posterior draw.
    chosen=[]
    for k,state in enumerate(states):
        rng = np.random.default_rng(20261007081+k*1009)
        chosen.append(state["p_draws"][rng.permutation(len(state["p_draws"]))[:count]])
    draws = np.concatenate(chosen,axis=0)
    return dict(p_mean=mean,p_variance=np.maximum(variance,0),
                lower95=np.quantile(draws,.025,axis=0),upper95=np.quantile(draws,.975,axis=0),
                p_draws=draws,within_state_variance=variances.mean(axis=0),
                between_state_variance=np.var(means,axis=0),draws_per_state=count)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--year",type=int,choices=[2019,2025],required=True)
    parser.add_argument("--scope",choices=["full","territorial_training"],default="full")
    parser.add_argument("--method",choices=["point","cut8","cut16"],required=True)
    parser.add_argument("--prepared",type=Path,help="Optional frozen prepared directory within this new phase, for verified auxiliary variants")
    parser.add_argument("--output",type=Path,help="Optional new result directory within this phase")
    args=parser.parse_args()
    gate_path,gate=freeze_gate()
    if args.method=="cut16" and (args.year!=2019 or args.scope!="full" or not gate.get("cut16_sensitivity_approved",False)):
        raise RuntimeError("CUT16 is limited to explicitly approved2019 full sensitivity")
    source=args.prepared.resolve() if args.prepared else PHASE/f"datos/diseno/{args.year}_{args.scope}"
    if not source.is_relative_to(PHASE):
        raise ValueError("Prepared data must be under this new phase")
    prepared=json.loads((source/"preparation_manifest.json").read_text())
    if prepared["year"]!=args.year or prepared["scope"]!=args.scope or prepared["reserved_2024_read"]:
        raise RuntimeError("Prepared dataset violates declared development scope")
    inputs=dict(np.load(source/"fit_inputs.npz"))
    endpoints=[np.load(source/f"{name}_calibration.npz") for name in ("anemia","stunting")]
    if args.method=="point":
        n_states=[np.column_stack([v["central_effective_n"] for v in endpoints])]
    elif args.method=="cut8":
        n_states=[np.column_stack([v["mixture_effective_n"][k] for v in endpoints]) for k in range(8)]
    else:
        from diseno_calibration import CONFIG
        selected=np.random.default_rng(CONFIG["state_seed"]).permutation(CONFIG["replicates"])[:16]
        n_states=[np.column_stack([v["replicate_effective_n"][k] for v in endpoints]) for k in selected]
    output=args.output.resolve() if args.output else PHASE/f"resultados/diseno/{args.year}_{args.scope}/{args.method}"
    if not output.is_relative_to(PHASE):
        raise ValueError("New fit output must be under this new phase")
    output.mkdir(parents=True,exist_ok=True)
    if (output/"completion.json").exists():
        raise FileExistsError("Already completed output must not be overwritten")
    states=[];statuses=[]
    tick=time.perf_counter()
    base_seed=20261007+args.year*100+(10000 if args.scope=="territorial_training" else 0)
    for k,n in enumerate(n_states):
        label=f"{args.year}_{args.scope}_{args.method}_state{k+1:02d}"
        result,status=conditional_fit(inputs,n,output/f"state_{k+1:02d}",base_seed+k*100003,label)
        states.append(result);statuses.append(status)
        print(json.dumps(dict(completed_state=k+1,total_states=len(n_states),label=label,
                              success=status["success"],seconds=time.perf_counter()-tick)),flush=True)
    pooled=pool_equal_weight(states)
    areas=pd.read_csv(source/"area_ids.csv",dtype={"ubigeo":str,"prov_code":str,"dept_code":str})
    areas["sampled"]=np.isin(np.arange(len(areas)),inputs["sample_idx"])
    for j,name in enumerate(("anemia","stunting")):
        for key in ("p_mean","p_variance","lower95","upper95","within_state_variance","between_state_variance"):
            areas[f"{key}_{name}"]=pooled[key][:,j]
    areas.to_csv(output/"district_predictions.csv",index=False)
    np.savez_compressed(output/"pooled_posterior.npz",**{key:value for key,value in pooled.items() if isinstance(value,np.ndarray)})
    completion=dict(status="PASS_DESIGN_CALIBRATED_CONDITIONAL_POSTERIORS_AND_CUT",success=True,
                    year=args.year,scope=args.scope,method=args.method,conditional_states=len(states),
                    equal_state_weights=True,state_weights=[1/len(states)]*len(states),
                    conditioning_normalizers_not_compared=True,performance_selection=False,
                    all_states_passed_numerical_gates=True,total_seconds=time.perf_counter()-tick,
                    balanced_pooled_draws_per_state=pooled["draws_per_state"],
                    design_uncertainty="conditional PSU-score variance bootstrap; singleton, lower-stage and nonresponse components not propagated",
                    latent_ENDES_coverage_observable=False,
                    root_freeze_sha256=digest(gate_path),preparation_manifest_sha256=digest(source/"preparation_manifest.json"),
                    source_sha256=digest(Path(__file__)),conditional_status=statuses,
                    output_sha256={name:digest(output/name) for name in ("district_predictions.csv","pooled_posterior.npz")})
    (output/"completion.json").write_text(json.dumps(completion,indent=2))
    print(json.dumps(dict(complete=True,method=args.method,states=len(states),seconds=time.perf_counter()-tick)),flush=True)


if __name__=="__main__":
    main()
