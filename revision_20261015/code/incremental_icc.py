"""Exploratory incremental value of the frozen mixture over ICC, without fitting.

Run with the scientific Python specified in the project README. All output is
aggregate synthetic-population evidence. Existing Holm3/Holm12 outputs are read
and audited, never overwritten. Inputs and the exploratory protocol are hashed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, t

CONTEXTS = ("SAR_probability_srs", "SAR_logit_srs", "SAR_logit_psu_sparse", "IID_logit_psu_sparse")
OUTCOMES = ("anemia", "stunting")
METHODS = ("Selected", "ICC")
MASKS = ("all", "observed_endpoint", "unsampled_common")
Z = float(norm.ppf(.975))
PRIMITIVES = ("interval_score", "mse", "bias", "coverage", "width", "mae")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def arrays(path):
    with np.load(path, allow_pickle=False) as value:
        return {key: value[key] for key in value.files}


def world_scores(run, protocol):
    """Validate completed synthetic predictions before reading their truth."""
    records, inputs = [], []
    for context in CONTEXTS:
        for rep in range(30000, 30064):
            source = run / "evaluacion/inputs" / context / f"rep_{rep:05d}"
            fit = run / "evaluacion/fits" / context / f"rep_{rep:05d}"
            manifest = json.loads((source / "manifest.json").read_text())
            proof = json.loads((fit / "completion.json").read_text())
            assert not proof["truth_read_by_fits"]
            assert proof["fit_inputs_sha256"] == sha(source / "model_inputs/inputs.npz")
            predictions = {}
            for method in METHODS:
                path = fit / (method + ".npz")
                assert sha(path) == proof["output_sha256"][path.name]
                predictions[method] = arrays(path)
            truth_path = source / "evaluation_only/truth.npz"
            assert sha(truth_path) == manifest["truth_sha256"]
            theta = arrays(truth_path)["theta"]
            data = arrays(source / "model_inputs/inputs.npz")
            observed = np.zeros_like(theta, dtype=bool)
            observed[data["sample_idx"]] = np.isfinite(data["y"])
            unsampled = ~np.isin(np.arange(len(theta)), data["sample_idx"])
            assert theta.shape == (120, 2) and int(unsampled.sum()) == 24
            for method, pred in predictions.items():
                mean, lo, hi = (pred[k] for k in ("mean", "lower", "upper"))
                assert mean.shape == lo.shape == hi.shape == theta.shape
                assert np.isfinite([mean, lo, hi]).all() and (lo <= hi).all()
                for j, outcome in enumerate(OUTCOMES):
                    masks = {"all": np.ones(len(theta), bool), "observed_endpoint": observed[:, j], "unsampled_common": unsampled}
                    for mask, eligible in masks.items():
                        assert eligible.any()
                        y, p, l, u = (a[eligible, j] for a in (theta, mean, lo, hi))
                        err = p-y
                        row = dict(context=context, replication=rep, method=method, outcome=outcome, mask=mask,
                                   districts=int(eligible.sum()), converged=bool(proof["convergence"][method]),
                                   interval_score=float(np.mean(u-l+40*np.maximum(l-y, 0)+40*np.maximum(y-u, 0))),
                                   mse=float(np.mean(err**2)), bias=float(err.mean()), mae=float(np.mean(abs(err))),
                                   coverage=float(np.mean((l <= y) & (y <= u))), width=float(np.mean(u-l)))
                        records.append(row)
            inputs.append({"context": context, "replication": rep,
                           "fit_completion_sha256": sha(fit / "completion.json"),
                           "model_inputs_sha256": sha(source / "model_inputs/inputs.npz"),
                           "truth_sha256": sha(truth_path),
                           "prediction_sha256": {m: sha(fit / (m+".npz")) for m in METHODS}})
    frame = pd.DataFrame(records)
    balanced = frame.groupby(["context", "replication", "method", "mask"], as_index=False)[list(PRIMITIVES)].mean()
    balanced["outcome"] = "endpoint_balanced"
    balanced["districts"] = np.nan
    balanced["converged"] = frame.groupby(["context", "replication", "method", "mask"]).converged.all().to_numpy()
    frame = pd.concat([frame, balanced], ignore_index=True)
    frame["population_rmse"] = np.sqrt(frame.mse)
    frame["population_coverage_distance95"] = abs(frame.coverage-.95)
    assert len(frame) == 256*2*3*3
    assert not frame.duplicated(["context", "replication", "method", "outcome", "mask"]).any()
    return frame, {"protocol_sha256": sha(protocol), "worlds": 256, "prediction_truth_manifest": inputs}


def balanced_mean(frame, value):
    return float(frame.assign(_v=value).groupby("context")._v.mean().mean())


def stratified_se(frame, influence):
    """Contexts are fixed equally weighted strata; population is the random unit."""
    local = frame[["context"]].copy(); local["_if"] = np.asarray(influence, float)
    groups = local.groupby("context")._if
    H = groups.ngroups
    assert groups.size().min() >= 2
    return float(np.sqrt(sum(v.var(ddof=1)/len(v) for _, v in groups))/H)


def functional_values(frame, primitive, function):
    x, y = frame[primitive+"_selected"].to_numpy(), frame[primitive+"_icc"].to_numpy()
    mx, my = balanced_mean(frame, x), balanced_mean(frame, y)
    if function == "linear":
        return mx, my, x, y, 1., 1.
    if function == "sqrt":
        assert mx > 0 and my > 0
        return np.sqrt(mx), np.sqrt(my), x, y, .5/np.sqrt(mx), .5/np.sqrt(my)
    if function == "distance95":
        return abs(mx-.95), abs(my-.95), x, y, float(np.sign(mx-.95)), float(np.sign(my-.95))
    raise ValueError(function)


def summarize_increment(frame):
    results = []
    specs = [(k, k, "linear") for k in PRIMITIVES]
    specs += [("RMSE_sqrt_mean_MSE", "mse", "sqrt"),
              ("mean_population_RMSE", "population_rmse", "linear"),
              ("aggregate_coverage_distance95", "coverage", "distance95"),
              ("mean_population_coverage_distance95", "population_coverage_distance95", "linear")]
    for outcome in (*OUTCOMES, "endpoint_balanced"):
        for mask in MASKS:
            sub = frame.loc[frame.outcome.eq(outcome) & frame["mask"].eq(mask)]
            pair = sub.loc[sub.method.eq("Selected")].merge(sub.loc[sub.method.eq("ICC")],
                on=["context", "replication", "outcome", "mask"], suffixes=("_selected", "_icc"), validate="one_to_one")
            for scope in ("equal_four_contexts", *CONTEXTS):
                p = pair if scope == "equal_four_contexts" else pair.loc[pair.context.eq(scope)]
                for metric, primitive, function in specs:
                    a, b, x, y, ga, gb = functional_values(p, primitive, function)
                    sea, seb = stratified_se(p, ga*x), stratified_se(p, gb*y)
                    se = stratified_se(p, ga*x-gb*y)
                    ratio = float(a/b) if abs(b) > 1e-14 and metric != "bias" else None
                    rse = stratified_se(p, ga*x/b - (a/b**2)*gb*y) if ratio is not None else None
                    item = dict(scope=scope, outcome=outcome, mask=mask, metric=metric,
                                population_pairs=len(p), fixed_contexts=p.context.nunique(),
                                selected=float(a), icc=float(b), selected_mcse=sea, icc_mcse=seb,
                                difference_selected_minus_icc=float(a-b), difference_mcse=se,
                                descriptive95_low=float(a-b-Z*se), descriptive95_high=float(a-b+Z*se),
                                ratio_selected_over_icc=ratio, relative_difference_percent=100*(ratio-1) if ratio is not None else None,
                                ratio_mcse=rse, relative_difference_percent_mcse=100*rse if rse is not None else None,
                                relative_descriptive95_low=100*(ratio-1-Z*rse) if rse is not None else None,
                                relative_descriptive95_high=100*(ratio-1+Z*rse) if rse is not None else None,
                                inference="exploratory_descriptive_no_pvalues_no_Holm")
                    if function == "distance95":
                        for label, values in (("selected", x), ("icc", y)):
                            c = balanced_mean(p, values); cs = stratified_se(p, values)
                            left, right = c-Z*cs, c+Z*cs
                            item[label+"_coverage_CI_spans95"] = bool(left <= .95 <= right)
                            item[label+"_distance95_bound_low"] = float(0 if left <= .95 <= right else min(abs(left-.95), abs(right-.95)))
                            item[label+"_distance95_bound_high"] = float(max(abs(left-.95), abs(right-.95)))
                        item["distance95_note"] = "Delta MCSE is local; absolute value is nonregular at 95. Use coverage interval and induced distance bounds when it spans95."
                    results.append(item)
    return pd.DataFrame(results)


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); out = np.empty(len(p))
    out[order] = np.minimum(1., np.maximum.accumulate(p[order]*np.arange(len(p), 0, -1)))
    return out


def audit_original_families(repo, out):
    """Independent formulas; no original tests or distributions are changed."""
    fresh = repo / "results_fresh/modern/resultados"
    frame = pd.read_csv(fresh / "published_sar_comparison/per_replication_scores.csv")
    original = pd.read_csv(fresh / "published_sar_comparison/paired_balanced_contrasts_Holm3.csv")
    a = frame.loc[frame.group.eq("all") & frame.method.eq("Selected")]
    records = []
    for ref in ("Sikov_FH", "Sikov_SAR", "Sikov_SAR_Published"):
        b = frame.loc[frame.group.eq("all") & frame.method.eq(ref)]
        pair = a.merge(b, on=["scenario", "replication", "outcome"], suffixes=("_a", "_b"), validate="one_to_one")
        for metric in ("interval_score", "mse", "mae", "coverage", "width"):
            d = pair.assign(delta=pair[metric+"_a"]-pair[metric+"_b"]).groupby(["scenario", "replication"]).delta.mean()
            estimate = float(d.groupby(level=0).mean().mean())
            se = float(np.sqrt(sum(d.loc[c].var(ddof=1)/len(d.loc[c]) for c in CONTEXTS))/4)
            pv = float(2*norm.sf(abs(estimate/se))) if se else float(estimate == 0)
            old = original.loc[original.reference.eq(ref) & original.metric.eq(metric)].iloc[0]
            values = dict(reference=ref, metric=metric, recomputed_difference=estimate, recomputed_mcse=se,
                          recomputed_p_two_sided=pv, mcse_unit="population after endpoint averaging", reference_distribution="standard_normal")
            values["max_absolute_error"] = float(max(abs(estimate-old.difference), abs(se-old.mcse), abs(pv-old.p_two_sided),
                                                       abs(estimate-Z*se-old.MC_CI95_low), abs(estimate+Z*se-old.MC_CI95_high)))
            records.append(values)
    raw = pd.DataFrame(records); primary = raw.loc[raw.metric.eq("interval_score")].copy()
    primary["recomputed_p_Holm3"] = holm(primary.recomputed_p_two_sided)
    checks3 = primary.merge(original.loc[original.metric.eq("interval_score"), ["reference", "p_Holm3"]], on="reference")
    e3 = float(max(raw.max_absolute_error.max(), abs(checks3.recomputed_p_Holm3-checks3.p_Holm3).max()))
    raw.to_csv(out / "Holm3_formula_recalculation.csv", index=False)
    original12 = pd.read_csv(fresh / "external_comparison_Holm12/paired_balanced_contrasts_Holm12.csv")
    rows12 = []
    for scope in ("temporal", "territorial"):
        data = pd.read_csv(fresh / ("external_extended_"+scope) / "district_scores.csv", dtype={"province": str, "ubigeo": str})
        for old in original12.loc[original12.scope.eq(scope)].itertuples():
            a = data.loc[data.method.eq("Selected"), ["ubigeo", "province", "outcome", old.metric]]
            b = data.loc[data.method.eq(old.reference), ["ubigeo", "outcome", old.metric]]
            pair = a.merge(b, on=["ubigeo", "outcome"], suffixes=("_a", "_b"), validate="one_to_one")
            pair["delta"] = pair[old.metric+"_a"]-pair[old.metric+"_b"]
            means = pair.groupby("outcome").delta.mean(); counts = pair.groupby("outcome").size()
            influence = pair.assign(_if=[.5*(v-means[o])/counts[o] for v, o in zip(pair.delta, pair.outcome)]).groupby("province")._if.sum()
            J = len(influence); diff = float(means.mean()); se = float(np.sqrt(J/(J-1)*sum(influence**2)))
            pv = float(2*t.sf(abs(diff/se), J-1)); crit = float(t.ppf(.975, J-1))
            error = max(abs(diff-old.difference), abs(se-old.cluster_standard_error), abs(pv-old.p_two_sided),
                        abs(diff-crit*se-old.unadjusted_t_ci_low), abs(diff+crit*se-old.unadjusted_t_ci_high))
            rows12.append(dict(scope=scope, reference=old.reference, metric=old.metric, difference=diff,
                               cluster_standard_error=se, province_clusters=J, t_df=J-1,
                               p_two_sided=pv, max_absolute_error=float(error), reference_distribution="Student_t_J_minus_1"))
    d12 = pd.DataFrame(rows12); d12["recomputed_p_Holm12"] = holm(d12.p_two_sided)
    matched = d12.merge(original12[["scope", "reference", "metric", "p_Holm12"]], on=["scope", "reference", "metric"])
    e12 = float(max(d12.max_absolute_error.max(), abs(matched.recomputed_p_Holm12-matched.p_Holm12).max()))
    d12.to_csv(out / "Holm12_formula_recalculation.csv", index=False)
    sources = ["comparison_collect.py", "sikov_published_collect.py", "align_external_score.py", "align_external_extended_score.py"]
    result = dict(status="PASS_UNCHANGED_ORIGINAL_FAMILY_AUDIT" if max(e3,e12)<1e-12 else "FAIL", Holm3_max_absolute_error=e3,
                  Holm12_max_absolute_error=e12, original_outputs_modified=False,
                  Holm3={"unit": "population; endpoints averaged first", "variance": "sum_h s_h^2/n_h divided by H^2, H=4 fixed contexts",
                         "test_distribution": "standard normal", "interval": "estimate +/-1.959963984540054 MCSE", "family": "3 primary IS95 contrasts only",
                         "MCSE_RMSE_ratio_column_caveat": "Original RMSE_ratio mcse stores MCSE of LOG ratio, not ratio. Its exponentiated interval is correctly on ratio scale. Primary IS95 is unaffected."},
                  Holm12={"unit": "province cluster with joint endpoint influence", "variance": "J/(J-1) sum_j U_j^2, U_j=.5 sum_e (S_je-n_je*mean_e)/N_e",
                          "test_distribution": "Student t with J-1 df", "family": "two scopes x three references x MSE/NLL =12", "not_MCSE": "This is a cluster sampling standard error, not simulation Monte Carlo error."},
                  source_sha256={s: sha(repo/"src/estudio_publicacion/sikov_2026-10-10/scripts"/s) for s in sources},
                  source_results_sha256={str(p.relative_to(repo)): sha(p) for p in [fresh/"published_sar_comparison/per_replication_scores.csv", fresh/"published_sar_comparison/paired_balanced_contrasts_Holm3.csv", fresh/"external_comparison_Holm12/paired_balanced_contrasts_Holm12.csv"]})
    write_json(out / "original_families_audit.json", result)
    assert result["status"].startswith("PASS")
    return result


def rank_development(d):
    per = d.groupby(["method", "scenario", "outcome"])[["interval_score", "mse", "coverage"]].mean()
    rank = per.groupby("method").mean().reset_index()
    rank["balanced_RMSE_sqrt_mean_MSE"] = np.sqrt(rank.pop("mse"))
    rank = rank.sort_values(["interval_score", "balanced_RMSE_sqrt_mean_MSE", "method"], kind="stable").reset_index(drop=True)
    rank.insert(0, "rank", np.arange(1, len(rank)+1))
    rank["gap_IS95_from_best"] = rank.interval_score-rank.interval_score.iloc[0]
    return rank


def development_audit(run, repo, out):
    source = run / "desarrollo/recreated_native"
    plan = json.loads((source / "development_plan.json").read_text())
    all_records = pd.read_csv(source / "replications.csv")
    d = all_records.loc[all_records.group.eq("all") & all_records.method.isin(plan["candidate_methods"])].copy()
    assert len(d) == 32*5*2 and not d.duplicated(["method", "scenario", "replication", "outcome"]).any()
    rank = rank_development(d)
    rank.to_csv(out / "Laplace_candidate_ranking.csv", index=False)
    d.to_csv(out / "Laplace_32_world_candidate_scores.csv", index=False)
    breakdown = d.groupby(["method", "scenario", "outcome"])[["interval_score", "mse", "bias", "coverage", "width"]].mean().reset_index()
    breakdown.to_csv(out / "Laplace_context_endpoint_grid.csv", index=False)
    loo = []; ranks = []
    for (context, rep), held in d.groupby(["scenario", "replication"]):
        train = d.loc[~(d.scenario.eq(context) & d.replication.eq(rep))]
        r = rank_development(train); winner, runner = r.iloc[0], r.iloc[1]
        ranks.extend(r.assign(held_context=context, held_replication=int(rep)).to_dict("records"))
        loo.append(dict(held_context=context, held_replication=int(rep), selected_method=winner.method,
                        runner_up=runner.method, training_IS95=float(winner.interval_score),
                        training_margin_best_to_second=float(runner.interval_score-winner.interval_score),
                        heldout_IS95=float(held.loc[held.method.eq(winner.method)].interval_score.mean()),
                        heldout_ICC_IS95=float(held.loc[held.method.eq("ICC")].interval_score.mean())))
    loof = pd.DataFrame(loo); loof.to_csv(out / "Laplace_LOPO_selection_and_margins.csv", index=False)
    pd.DataFrame(ranks).to_csv(out / "Laplace_LOPO_all_candidate_rankings.csv", index=False)
    original_rank = pd.read_csv(source / "candidate_ranking.csv")
    joined = rank.merge(original_rank, on="method", suffixes=("_new", "_old"))
    assert np.max(abs(joined.interval_score_new-joined.interval_score_old)) < 1e-12
    result = dict(status="PASS_ORIGINAL_LAPLACE_GRID_RECOVERED", worlds=32, contexts=4, worlds_per_context=8, candidates=plan["candidate_methods"],
                  recipes=plan["candidate_recipes"], selected_method=rank.iloc[0].method,
                  best_IS95=float(rank.interval_score.iloc[0]), second_method=rank.iloc[1].method,
                  best_second_absolute_margin=float(rank.interval_score.iloc[1]-rank.interval_score.iloc[0]),
                  best_second_relative_margin_percent=float(100*(rank.interval_score.iloc[1]/rank.interval_score.iloc[0]-1)),
                  LOPO_counts=dict(Counter(loof.selected_method)), LOPO_min_margin=float(loof.training_margin_best_to_second.min()),
                  LOPO_max_margin=float(loof.training_margin_best_to_second.max()),
                  integration="Laplace for development ICC and ICC_poverty; B100 parametric bootstrap native FH/SAR references",
                  approximation_caveat="LOPO audits a finite Laplace selection rule; does not assess NUTS ranking or prove the same optimum under NUTS.",
                  NUTS_replay_required=True, original_weights_unchanged=True,
                  source_sha256={name: sha(source/name) for name in ["development_plan.json", "replications.csv", "candidate_ranking.csv", "selected_recipe.json"]},
                  replay_inputs=[dict(context=c, replication=r, model_inputs_sha256=sha(run/"desarrollo/inputs"/c/f"rep_{r:05d}/model_inputs/inputs.npz"),
                                      sample_sha256=sha(run/"desarrollo/inputs"/c/f"rep_{r:05d}/model_inputs/sample.csv")) for c in CONTEXTS for r in range(8)])
    historical = repo/"src/estudio_publicacion/sikov_2026-10-10/desarrollo/improve_development_v4_native_rho999"
    hd = pd.read_csv(historical/"replications.csv")
    hd = hd.loc[hd.group.eq("all") & hd.method.isin(plan["candidate_methods"])]
    hr = rank_development(hd); hr.to_csv(out/"Archived_original_candidate_ranking.csv",index=False)
    hlo = []
    for (c,r), held in hd.groupby(["scenario","replication"]):
        ranking = rank_development(hd.loc[~(hd.scenario.eq(c)&hd.replication.eq(r))])
        hlo.append(dict(held_context=c,held_replication=int(r),selected_method=ranking.iloc[0].method,
                        runner_up=ranking.iloc[1].method,
                        training_margin_best_to_second=float(ranking.iloc[1].interval_score-ranking.iloc[0].interval_score)))
    pd.DataFrame(hlo).to_csv(out/"Archived_original_LOPO_selection_and_margins.csv",index=False)
    shutil.copyfile(historical/"replications.csv",out/"Archived_original_32_world_scores.csv")
    comparison=rank.merge(hr,on="method",suffixes=("_fresh","_archived"))
    comparison["IS95_difference_fresh_minus_archived"]=comparison.interval_score_fresh-comparison.interval_score_archived
    comparison.to_csv(out/"Laplace_fresh_archived_ranking_comparison.csv",index=False)
    result["historical_grid"]={"source_role":"Archived original numerical development; fresh rebuilt development above uses the same 32 input worlds.",
        "winner":str(hr.iloc[0].method),"runner_up":str(hr.iloc[1].method),
        "best_IS95":float(hr.iloc[0].interval_score),"best_second_absolute_margin":float(hr.iloc[1].interval_score-hr.iloc[0].interval_score),
        "LOPO_counts":dict(Counter(v["selected_method"] for v in hlo)),
        "source_sha256":{p.name:sha(p) for p in [historical/"replications.csv",historical/"candidate_ranking.csv",historical/"selected_recipe.json"]}}
    write_json(out / "Laplace_development_grid_audit.json", result)
    (out / "NUTS_development_replay_contract.md").write_text(
        "# NUTS development sensitivity contract\n\nExploratory only. Use all original 32 development populations, same inputs, covariates, priors, working ICC, seeds and four-chain gates as evaluation. Fit ICC and ICC_poverty with NUTS; retain original fresh B100 FH/SAR arrays after matching input hashes. Recompute ICC weights0.25,0.50,0.75 by the original mixture CDF algorithm. Freeze every prediction before scoring; rank the original five eligible candidates by equal-context/equal-endpoint IS95, then sqrt(mean MSE), then name. Report all candidate rankings and 32 leave-one-population-out margins. Do not replace the frozen75/25 recipe or original confirmatory families.\n\nInput hashes and five original recipes are in Laplace_development_grid_audit.json. Parent compute agent executes the separately registered NUTS replay and supplies its exact executable command in its own receipt. This audit launches no fitting.\n")
    return result


def figure(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 14.5pt at11.6in becomes approximately8.1pt at6.45in Word width.
    plt.rcParams.update({"font.size": 14.5, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 250})
    labels = {"equal_four_contexts": "Balanced overall", "SAR_probability_srs": "Prob. SAR / SRS", "SAR_logit_srs": "Logit SAR / SRS", "SAR_logit_psu_sparse": "Logit SAR / sparse", "IID_logit_psu_sparse": "Logit IID / sparse"}
    order = ["equal_four_contexts", *CONTEXTS]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), layout="constrained")
    for ax, metric, title, scale in zip(axes, ["interval_score", "RMSE_sqrt_mean_MSE"], ["95% interval score", "RMSE (percentage points)"], [1, 100]):
        d = summary.loc[summary.outcome.eq("endpoint_balanced") & summary["mask"].eq("all") & summary.metric.eq(metric)].set_index("scope").loc[order]
        x = scale*d.difference_selected_minus_icc.to_numpy(); err = Z*scale*d.difference_mcse.to_numpy()
        ax.axvline(0, color=".5", lw=1, ls="--")
        ax.errorbar(x, np.arange(5), xerr=err, fmt="o", color="#176b87", capsize=4, ms=6.5, lw=1.5)
        ax.set_yticks(np.arange(5), [labels[v] for v in order]); ax.invert_yaxis(); ax.set_title(title)
        ax.set_xlabel("Selected − ICC"); ax.grid(axis="x", alpha=.2)
    fig.suptitle("Exploratory comparison on the same 256 populations", fontsize=16)
    fig.savefig(out / "Selected_vs_ICC_incremental.png", bbox_inches="tight")
    fig.savefig(out / "Selected_vs_ICC_incremental.svg", bbox_inches="tight")
    plt.close(fig)


def english_report(summary, dev, audit, out):
    d = summary.loc[summary.scope.eq("equal_four_contexts") & summary.outcome.eq("endpoint_balanced") & summary["mask"].eq("all")]
    lines = ["# Exploratory incremental contribution of the frozen mixture over ICC", "", "This comparison was registered before this new scoring. It uses the same 256 existing fresh populations; it is exploratory and does not add a confirmatory test, change the 75/25 mixture or modify Holm3/Holm12. Four contexts receive equal weight; two endpoints are averaged inside each population. Population pairs, rather than district or endpoint rows, determine Monte Carlo error.", "", "| Metric | Selected | ICC | Selected minus ICC | MCSE of difference | Descriptive95% interval | Relative change |", "|---|---:|---:|---:|---:|---:|---:|"]
    for metric in ["interval_score", "RMSE_sqrt_mean_MSE", "mean_population_RMSE", "bias", "coverage", "width", "aggregate_coverage_distance95"]:
        v = d.loc[d.metric.eq(metric)].iloc[0]; scale = 100 if metric not in ["interval_score"] else 1
        name = metric + (" (percentage points)" if scale==100 else "")
        rel = "not meaningful" if pd.isna(v.relative_difference_percent) else f"{v.relative_difference_percent:.2f}%"
        lines.append(f"| {name} | {scale*v.selected:.5f} | {scale*v.icc:.5f} | {scale*v.difference_selected_minus_icc:.5f} | {scale*v.difference_mcse:.5f} | [{scale*v.descriptive95_low:.5f}, {scale*v.descriptive95_high:.5f}] | {rel} |")
    lines += ["", "IS95 penalizes both interval width and noncoverage. Lower IS95 or RMSE favours the mixture; lower distance from 95% coverage is closer to nominal. Increased coverage alone does not imply improved calibration. RMSE_sqrt_mean_MSE is the square root of balanced mean squared error. mean_population_RMSE averages the square root calculated separately in each endpoint-balanced population; these estimands differ.", "", "For a linear population statistic, MCSE is sqrt(sum_h s_h²/n_h)/H, with H fixed equally weighted contexts. Paired differences retain endpoint and method covariance. Nonlinear RMSE, relative changes and aggregate coverage distance use paired delta-method influence values centered within each context. Intervals are estimate±1.959964MCSE, descriptive and unadjusted. The absolute-distance function is nonregular at 95%; aggregate coverage intervals and their induced distance bounds are also supplied. No p values are computed.", "", "All context, endpoint and availability-mask results, including adverse contrasts, are retained in incremental_summary.csv. The unsampled-common mask means absence from the common sampled district set; observed-endpoint means a finite response for that endpoint. Neither mask is redefined after inspecting accuracy.", "", "## Original-family audit", "", f"Holm3 uses a standard-normal reference and population MCSE with fixed equal context weights. Independent recalculation maximum error:{audit['Holm3_max_absolute_error']:.3g}. Holm12 uses a province-cluster sandwich standard error and Student t with J−1 df (not simulation MCSE); maximum error:{audit['Holm12_max_absolute_error']:.3g}. Neither family is changed. The original RMSE_ratio row's mcse field is on the log-ratio scale; its exponentiated intervals are on the ratio scale. This labelling caveat does not affect the primary IS95 tests.", "", "## Original development rule", "", f"The original five-candidate Laplace grid used 32 worlds (8 per context). Winner:{dev['selected_method']}; runner-up:{dev['second_method']}; IS95 gap:{dev['best_second_absolute_margin']:.8f} ({dev['best_second_relative_margin_percent']:.3f}% relative to winner). LOPO selections:{dev['LOPO_counts']}; minimum training margin:{dev['LOPO_min_margin']:.8f}. Laplace development and NUTS evaluation use different integration approximations; this audit does not establish an identical NUTS optimum. A separately frozen NUTS development sensitivity is required and its results must remain exploratory.", "", "Source arrays, predictions, truth and protocol hashes are bound in incremental_provenance.json. This script fits no model, modifies no historical sources or results and exports no respondent records."]
    (out / "incremental_report_English.md").write_text("\n".join(lines)+"\n")


def builder_exports(summary, dev, audit, out):
    """Unrounded machine-readable claims and compact English author-review table."""
    metrics = ["interval_score", "RMSE_sqrt_mean_MSE", "mean_population_RMSE", "bias", "coverage", "width", "aggregate_coverage_distance95"]
    global_rows = summary.loc[summary.scope.eq("equal_four_contexts") & summary.metric.isin(metrics)].copy()
    def clean_records(frame):
        return json.loads(frame.to_json(orient="records", double_precision=15))
    # json float encoding below retains native precision; pandas to_json is used
    # only to turn missing values into JSON null in the large tabular export.
    def native_record(v):
        return {k: (None if pd.isna(x) else (x.item() if hasattr(x, "item") else x)) for k, x in v.items()}
    global_native = [native_record(v) for v in global_rows.to_dict("records")]
    write_json(out / "summary.json", {
        "format": "iaos-incremental-icc-summary-v1", "status": "PASS_EXPLORATORY_ANALYSIS",
        "exploratory": True, "confirmatory_families_changed": False, "population_pairs": 256,
        "metrics": metrics, "outcomes": [*OUTCOMES, "endpoint_balanced"], "masks": list(MASKS),
        "units": "Proportions throughout; multiply RMSE, bias, coverage, width and distance95 by100 for percentage points. IS95 remains on prevalence-proportion scale.",
        "global": global_native,
        "context_endpoint_mask_data_csv": "incremental_summary.csv",
        "figure_png": "Selected_vs_ICC_incremental.png", "figure_svg": "Selected_vs_ICC_incremental.svg",
        "Laplace_development": {k: dev[k] for k in ["selected_method", "second_method", "best_IS95", "best_second_absolute_margin", "best_second_relative_margin_percent", "LOPO_counts", "LOPO_min_margin", "LOPO_max_margin"]},
        "original_family_audit": {k: audit[k] for k in ["status", "Holm3_max_absolute_error", "Holm12_max_absolute_error"]}})
    compact = global_rows.loc[global_rows.outcome.eq("endpoint_balanced")].copy()
    compact["metric"] = pd.Categorical(compact.metric, categories=metrics, ordered=True)
    compact = compact.sort_values(["mask", "metric"])
    compact.to_csv(out / "compact_English_table.csv", index=False)
    get = lambda metric, scope="equal_four_contexts", mask="all", outcome="endpoint_balanced": native_record(summary.loc[
        summary.metric.eq(metric) & summary.scope.eq(scope) & summary["mask"].eq(mask) & summary.outcome.eq(outcome)].iloc[0].to_dict())
    claims = {"global_interval_score": get("interval_score"), "global_RMSE_sqrt_mean_MSE": get("RMSE_sqrt_mean_MSE"),
              "global_width": get("width"), "global_coverage": get("coverage"), "global_distance95": get("aggregate_coverage_distance95"),
              "global_bias": get("bias"), "context_interval_score": [get("interval_score", c) for c in CONTEXTS],
              "availability_interval_score": [get("interval_score", mask=m) for m in MASKS],
              "availability_RMSE_sqrt_mean_MSE": [get("RMSE_sqrt_mean_MSE", mask=m) for m in MASKS]}
    isv, rv, width, coverage, dist = (claims[k] for k in ["global_interval_score", "global_RMSE_sqrt_mean_MSE", "global_width", "global_coverage", "global_distance95"])
    narrative = {
        "format": "iaos-incremental-icc-narrative-v1", "exploratory": True, "raw_unrounded_claims": claims,
        "English_main_ready": (
            f"An exploratory paired comparison against ICC alone on the same 256 populations found balanced IS95 {100*(1-isv['ratio_selected_over_icc']):.2f}% lower and RMSE {100*(1-rv['ratio_selected_over_icc']):.2f}% lower for the frozen mixture. "
            f"Mean intervals were {width['relative_difference_percent']:.2f}% wider. Balanced coverage was {100*coverage['selected']:.2f}% for the mixture and {100*coverage['icc']:.2f}% for ICC, with respective absolute deviations from 95% of {100*dist['selected']:.2f} and {100*dist['icc']:.2f} percentage points. "
            "Thus the modest overall interval-score gain comes with wider intervals and greater aggregate overcoverage; it does not establish better nominal calibration or improvement in every context. No new confirmatory family is introduced."),
        "English_context_ready": "Endpoint-balanced IS95 differences favour the mixture in the two SRS contexts, favour ICC in sparse clustered logit-SAR, and are near zero in sparse clustered IID. Context-specific intervals are descriptive and unadjusted.",
        "English_methods_ready": "Population-paired summaries first average endpoints within each population, then contexts with equal weight. MCSE uses the sum of within-context population variances divided by sample sizes and by the squared number of contexts. Nonlinear RMSE and relative contrasts use paired delta-method influences. Reported 95% Monte Carlo intervals are descriptive; no new p values or multiplicity family are computed.",
        "English_development_ready": f"The original Laplace development grid selected 75% ICC/25% SAR in all 32 leave-one-population-out fits. The full-grid gap to the 50% ICC runner-up was {dev['best_second_absolute_margin']:.6f} IS95 ({dev['best_second_relative_margin_percent']:.2f}% of the winner score). These are Laplace selection diagnostics, not evidence that NUTS gives the same ranking.",
        "mandatory_caveats": ["All incremental comparisons are exploratory.", "Lower overall IS95 does not mean better nominal coverage calibration.", "Current comparison does not isolate an ICC or mixture effect under identical regressors.", "Holm3 normal-reference and Holm12 Student-t-reference families are unchanged.", "Original RMSE_ratio mcse field is log-ratio MCSE; primary interval-score comparisons are unaffected.", "NUTS development replay must be reported separately and cannot reselect the primary mixture."]}
    write_json(out / "narrative.json", narrative)
    names = {"interval_score": "IS95", "RMSE_sqrt_mean_MSE": "RMSE: sqrt(mean MSE), pp",
             "mean_population_RMSE": "Mean population RMSE, pp", "bias": "Signed bias, pp", "coverage": "Coverage, %",
             "width": "Mean interval width, pp", "aggregate_coverage_distance95": "Absolute aggregate coverage deviation, pp"}
    table = []
    for metric in metrics:
        v = get(metric); scale = 1 if metric == "interval_score" else 100
        f = lambda x: f"{scale*x:.4f}"
        rel = "—" if v["relative_difference_percent"] is None else f"{v['relative_difference_percent']:+.2f}"
        table.append([names[metric], f(v["selected"]), f(v["icc"]), f(v["difference_selected_minus_icc"]),
                      f(v["difference_mcse"]), f"[{f(v['descriptive95_low'])}, {f(v['descriptive95_high'])}]", rel])
    context_names={"SAR_probability_srs":"Probability SAR / SRS", "SAR_logit_srs":"Logit SAR / SRS",
                   "SAR_logit_psu_sparse":"Logit SAR / sparse PSU", "IID_logit_psu_sparse":"Logit IID / sparse PSU"}
    mask_names={"all":"All", "observed_endpoint":"Observed endpoint", "unsampled_common":"Outside common sample"}
    sup=[]
    for c in CONTEXTS:
        for outcome in (*OUTCOMES,"endpoint_balanced"):
            for mask in MASKS:
                iv=get("interval_score",c,mask,outcome); r=get("RMSE_sqrt_mean_MSE",c,mask,outcome)
                cv=get("coverage",c,mask,outcome); w=get("width",c,mask,outcome)
                sup.append([context_names[c],outcome.replace("_"," "),mask_names[mask],
                            f"{iv['selected']:.4f} / {iv['icc']:.4f}",
                            f"{iv['difference_selected_minus_icc']:+.4f} ({iv['difference_mcse']:.4f})",
                            f"{100*r['selected']:.3f} / {100*r['icc']:.3f}",
                            f"{100*cv['selected']:.2f} / {100*cv['icc']:.2f}",
                            f"{100*w['selected']:.2f} / {100*w['icc']:.2f}"])
    supplemental_paras = [narrative["English_methods_ready"],
        "The main RMSE functional is the square root of balanced mean population MSE, not the average of population RMSEs. Aggregate absolute coverage deviation is the absolute difference between the balanced coverage and0.95, not the average of population absolute deviations. Both alternatives are supplied separately in the machine-readable results. Delta-method MCSE for aggregate absolute coverage deviation is a local approximation and is nonregular at0.95; coverage intervals and induced bounds on the absolute deviation are also supplied.",
        narrative["English_context_ready"],
        "The original Holm3 primary tests use the standard-normal reference with population MCSE. Holm12 uses a province-cluster sampling standard error and Student t with J−1 degrees of freedom. Independent recalculation reproduced these families to absolute numerical error below4×10⁻¹⁶. The original RMSE-ratio mcse field is a log-ratio MCSE; its exponentiated interval is on the ratio scale. This labelling caveat does not affect the primary IS95 contrasts."]
    write_json(out/"editorial_contract.json", {
        "format":"iaos-incremental-editorial-contract-v1", "exploratory":True,
        "main_paragraphs":[narrative["English_main_ready"],narrative["English_context_ready"]],
        "supplement_paragraphs":supplemental_paras,
        "main_table":{"header":["Metric","Selected","ICC","Difference","MCSE","Descriptive95% interval","Relative change,%"],"rows":table},
        "supplement_table":{"header":["Context","Outcome","Domains","IS95 S / ICC","IS95 difference (MCSE)","RMSE S / ICC,pp","Coverage S / ICC,%","Width S / ICC,pp"],"rows":sup},
        "table_note":"Exploratory comparison on the same256 populations, with equal context and endpoint weights. Difference=Selected−ICC; ratios use ICC as denominator. MCSE is population based and paired. Intervals are descriptive and unadjusted; no p values or new Holm family. Values except IS95 are scaled to percentage points (coverage is percent). Signed-bias ratios have no stable interpretation and are omitted. Aggregate absolute coverage deviation is |balanced coverage−0.95|; its delta approximation is local.",
        "supplement_table_note":"Each context has64 populations. S denotes the frozen Selected mixture. Slash-separated values are Selected / ICC. Each row is an outcome and domain subset; endpoint-balanced rows average outcomes within each population. Positive IS95 differences favour ICC. All results are exploratory; the entire underlying450-row metric file, including mean population RMSE and absolute-deviation alternatives, is retained.",
        "figure_path":"revision_20261015/results/incremental_icc/Selected_vs_ICC_incremental.png",
        "figure":"revision_20261015/results/incremental_icc/Selected_vs_ICC_incremental.png",
        "figure_caption":"Exploratory Selected-minus-ICC differences in endpoint-balanced IS95 and RMSE on the same256 primary populations. Each context has64 independent populations; the global estimate gives all four contexts equal weight. Whiskers are paired, population-based, unadjusted descriptive95% Monte Carlo intervals. Negative values favour Selected. RMSE is sqrt(mean MSE), displayed in percentage points. No new p values, multiplicity family or mixture reselection is introduced.",
        "inputs":["revision_20261015/results/incremental_icc/incremental_summary.csv","revision_20261015/results/incremental_icc/summary.json","revision_20261015/results/incremental_icc/narrative.json","revision_20261015/results/incremental_icc/incremental_provenance.json"],
        "source_code":"revision_20261015/code/incremental_icc.py", "raw_precision_claims_file":"revision_20261015/results/incremental_icc/narrative.json"})


def score_csv_check(frame, repo):
    """Independent array-to-existing aggregate check; reads no extra responses."""
    old=pd.read_csv(repo/"results_fresh/modern/resultados/published_sar_comparison/per_replication_scores.csv")
    old=old.loc[old.method.isin(METHODS)].rename(columns={"scenario":"context","group":"mask"})
    fresh=frame.loc[frame.outcome.isin(OUTCOMES)]
    pair=fresh.merge(old,on=["context","replication","method","outcome","mask"],suffixes=("_fresh","_old"),validate="one_to_one")
    assert len(pair)==256*2*2*3
    error=max(float(abs(pair[k+"_fresh"]-pair[k+"_old"]).max()) for k in PRIMITIVES)
    assert error<1e-12
    return {"array_to_existing_aggregate_rows":len(pair),"max_absolute_metric_error":error,"passed":True}


def self_test():
    """Meaningful covariance/stratum/delta checks on small analytic fixtures."""
    f = pd.DataFrame({"context": ["a", "a", "b", "b"]})
    assert np.isclose(stratified_se(f, np.array([1.,3.,10.,14.])), np.sqrt(5)/2)
    assert stratified_se(f, np.array([1.,3.,10.,14.])-np.array([1.,3.,10.,14.])) == 0
    assert np.allclose(holm([.03,.01,.04]), [.06,.03,.06])
    # Gradients of the nonlinear global functionals must use global means,
    # rather than averaging gradients or roots calculated per population.
    mx,my=.09,.16;eps=1e-6
    assert np.isclose((np.sqrt(mx+eps)-np.sqrt(mx-eps))/(2*eps),.5/np.sqrt(mx),rtol=1e-8)
    fa=lambda x,y:np.sqrt(x)/np.sqrt(y)
    assert np.isclose((fa(mx+eps,my)-fa(mx-eps,my))/(2*eps),(.5/np.sqrt(mx))/np.sqrt(my),rtol=1e-8)
    assert np.isclose((fa(mx,my+eps)-fa(mx,my-eps))/(2*eps),-np.sqrt(mx)*(.5/np.sqrt(my))/(np.sqrt(my)**2),rtol=1e-8)
    assert not np.isclose(np.sqrt(np.mean([.01,.09])),np.mean(np.sqrt([.01,.09])))
    assert not np.isclose(abs(np.mean([.9,1.])-.95),np.mean(abs(np.array([.9,1.])-.95)))
    return {"fixed_context_variance_analytic": True, "paired_covariance_cancellation": True, "Holm_direct_example": True,
            "global_root_MSE_gradient_finite_difference":True,"RMSE_ratio_two_gradients_finite_difference":True,
            "root_mean_MSE_distinct_from_mean_RMSE":True,"aggregate_distance95_distinct_from_mean_distance":True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); repo = args.repo.resolve(); base = repo/"revision_20261015"
    out = args.output or base/"results/incremental_icc"
    protocol = base/"protocols/incremental_icc_exploratory_protocol.json"
    assert protocol.exists(), "Register protocol before scoring"
    # Check preserved study identity rather than the caller's checkout name.
    # Renaming or cloning the review repository must not block reproduction.
    baseline_path = repo/"expected/current_freeze/current_baseline.json"
    assert baseline_path.is_file(), "Preserved numerical baseline is required"
    assert json.loads(baseline_path.read_text()).get("format") == "iaos-current-fresh-baseline-v1"
    assert sha(repo/"docs/Articulo_IAOS.docx") == "60a88c7feb24a200a5ff806ffe15647899668f2543f035c451063095bd5a420d", "Preserved source document differs"
    out.resolve().relative_to(base.resolve())
    out.mkdir(parents=True, exist_ok=True)
    devout = base/"results/devgrid"; devout.mkdir(parents=True, exist_ok=True)
    auditout = base/"results/audit"; auditout.mkdir(parents=True, exist_ok=True)
    checks = self_test()
    frame, provenance = world_scores(args.run_root, protocol)
    provenance["array_to_existing_aggregate_check"] = score_csv_check(frame, repo)
    frame.to_csv(out/"per_population_Selected_ICC_metrics.csv", index=False)
    summary = summarize_increment(frame); summary.to_csv(out/"incremental_summary.csv", index=False)
    audit = audit_original_families(repo, auditout)
    dev = development_audit(args.run_root, repo, devout)
    figure(summary, out); english_report(summary, dev, audit, out); builder_exports(summary, dev, audit, out)
    provenance.update(status="PASS_EXPLORATORY_INCREMENTAL_ICC_ANALYSIS", source_sha256=sha(__file__), self_tests=checks,
                      population_score_rows=len(frame), aggregate_metric_rows=len(summary), no_new_models_fitted=True,
                      original_Holm_families_unchanged=True, development_selection_changed=False,
                      runtime={"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
                      outputs_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name!="incremental_provenance.json"})
    write_json(out/"incremental_provenance.json", provenance)
    print(json.dumps({"status": provenance["status"], "output": str(out), "development": str(devout), "audit": str(auditout), "self_tests": checks}, indent=2))


if __name__ == "__main__":
    main()
