"""Development-only finite candidate selection, with honest leave-world-out audit.

Reads metrics already scored by the development evaluator. This module never
opens final outcomes or changes source models. The plan lists every allowed
development population, method and scenario before selection. A completely
fresh evaluation must still assess the finally frozen selected method.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_cohort(records, plan):
    if plan.get("phase") != "development":
        raise ValueError("An explicit development plan is required")
    if not plan.get("independent_final_outcomes_excluded", False):
        raise ValueError("Final outcomes must be explicitly excluded by the plan")
    d = records.loc[records.group == "all"].copy()
    methods = plan["candidate_methods"]
    scenarios = plan["scenarios"]
    reps = plan["development_replication_ids"]
    if isinstance(reps, list):
        reps = {s: reps for s in scenarios}
    if set(d.method) != set(methods) or set(d.scenario) != set(scenarios):
        raise ValueError("Method/scenario cohort differs from development plan")
    if "phase" in d and not (d.phase == "development").all():
        raise ValueError("A nondevelopment metric row entered selection")
    keys = ["method", "scenario", "replication", "outcome"]
    if d.duplicated(keys).any():
        raise ValueError("Duplicate development metric cell")
    if not np.isfinite(d[["interval_score", "mse"]].to_numpy()).all():
        raise ValueError("All finite failures must be included; unresolved fits block selection")
    if not (d.mse >= 0).all():
        raise ValueError("Negative MSE")
    if not (d.areas == d.areas_requested).all():
        raise ValueError("Identical complete target-area support is required")
    expected = {(m, s, r, o) for m in methods for s in scenarios
                for r in reps[s] for o in ("anemia", "stunting")}
    actual = set(d[keys].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(f"Development cohort incomplete or contaminated; missing={len(expected-actual)}, extra={len(actual-expected)}")
    return d


def rank_candidates(d):
    per = d.groupby(["method", "scenario", "outcome"]).agg(
        interval_score=("interval_score", "mean"), mse=("mse", "mean"),
        coverage=("coverage", "mean"), convergence=("converged", "mean"))
    # Exactly equal context and endpoint weighting; no sample-size weighting.
    scores = per.groupby("method").mean().reset_index()
    scores["balanced_rmse"] = np.sqrt(scores.pop("mse"))
    return scores.sort_values(["interval_score", "balanced_rmse", "method"], kind="stable").reset_index(drop=True)


def select(records, plan):
    d = audit_cohort(records, plan)
    ranking = rank_candidates(d)
    choice = ranking.iloc[0].method
    # Holding out a complete finite population preserves endpoint covariance.
    # One world is one (scenario, replication) pair, not one area or endpoint.
    crossvalidated = []
    for (scenario, replication), held in d.groupby(["scenario", "replication"]):
        training = d.loc[~((d.scenario == scenario) & (d.replication == replication))]
        if training.groupby(["method", "scenario", "outcome"]).size().min() < 1:
            raise ValueError("Leave-population-out selection needs >=2 worlds per context")
        selected = rank_candidates(training).iloc[0].method
        test = held.loc[held.method == selected].copy()
        test["chosen_in_training"] = selected
        test["method"] = "LOPO_selection_rule"
        crossvalidated.append(test)
    held_results = pd.concat(crossvalidated, ignore_index=True)
    audit = rank_candidates(held_results).iloc[0].to_dict()
    return {"selected_method": str(choice), "phase": "development",
            "selection_criterion": "equal-context equal-endpoint IS95; RMSE then method name break exact ties",
            "selected_recipe": plan["candidate_recipes"][str(choice)],
            "development_only_no_superiority_claim": True,
            "LOPO_selection_rule_audit": audit,
            "LOPO_caveat": "audits the finite selection rule; does not replace fresh evaluation of the final selected fixed model",
            "no_interval_factor_tuning": True}, ranking, held_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh immutable development selection destination")
    records = pd.read_csv(args.records)
    plan = json.loads(args.plan.read_text())
    result, ranking, held = select(records, plan)
    result["source_sha256"] = {str(args.records): sha256(args.records),
                               str(args.plan): sha256(args.plan),
                               str(Path(__file__)): sha256(Path(__file__))}
    args.output.mkdir(parents=True)
    ranking.to_csv(args.output / "candidate_ranking.csv", index=False)
    held.to_csv(args.output / "LOPO_selection_rule_metrics.csv", index=False)
    (args.output / "selected_recipe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
