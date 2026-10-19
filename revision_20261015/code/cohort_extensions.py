"""Exploratory closed-cohort summaries for sparse57 and NUTS development.

Consumes aggregate scores exported only after each entire prediction cohort is
locked. No model fit, truth access, reselection of the main mixture or original
Holm-family modification occurs in this adapter.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd

import incremental_icc as inc


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,np.floating)):
        value=value.item()
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def prepare_scores(path, expected_worlds, methods):
    frame=pd.read_csv(path)
    frame=frame.rename(columns={"scenario":"context","group":"mask"})
    required={"context","replication","method","outcome","mask","converged",*inc.PRIMITIVES}
    assert required.issubset(frame), required-set(frame)
    assert set(inc.CONTEXTS)==set(frame.context)
    assert set(methods).issubset(set(frame.method))
    frame=frame.loc[frame.method.isin(methods)&frame.outcome.isin(inc.OUTCOMES)&frame["mask"].isin(inc.MASKS)].copy()
    assert not frame.duplicated(["context","replication","method","outcome","mask"]).any()
    assert frame[[*inc.PRIMITIVES]].apply(np.isfinite).all().all()
    assert len(frame)==expected_worlds*len(methods)*2*3
    assert frame.groupby(["method","outcome","mask"]).size().eq(expected_worlds).all()
    assert frame.groupby(["context","method","outcome","mask"]).size().eq(expected_worlds//4).all()
    balanced=frame.groupby(["context","replication","method","mask"],as_index=False)[list(inc.PRIMITIVES)].mean()
    balanced["outcome"]="endpoint_balanced"
    frame=pd.concat([frame,balanced],ignore_index=True)
    frame["population_rmse"]=np.sqrt(frame.mse)
    frame["population_coverage_distance95"]=abs(frame.coverage-.95)
    return frame


def numerical_gates(records,methods):
    """Count failed recorded method/endpoint gates without deleting any case."""
    d=pd.read_csv(records)
    d=d.loc[d.method.isin(methods)&d.group.eq("all")&d.outcome.isin(inc.OUTCOMES)].copy()
    assert not d.empty,"No gate records cannot be treated as complete success"
    if d.converged.dtype!=bool:
        flags=d.converged.astype(str).str.casefold().map({"true":True,"false":False})
        assert flags.notna().all(),"Unknown convergence flag cannot be treated as success"
        d["converged"]=flags.astype(bool)
    pairs=d.groupby(["scenario","replication","method"]).converged.all()
    worlds=d.groupby(["scenario","replication"]).converged.all()
    failed=d.loc[~d.converged]
    return dict(recorded_gate_status_available=True,all_recorded_gates_passed=bool(d.converged.all()),
                population_worlds_total=len(worlds),worlds_with_any_failed_gate=int((~worlds).sum()),
                world_method_pairs_total=len(pairs),world_method_pairs_with_any_failed_gate=int((~pairs).sum()),
                method_endpoint_world_gate_cells_total=len(d),method_endpoint_world_gate_cells_failed=len(failed),
                failed_endpoint_world_cells_by_method={str(k):int(v)for k,v in failed.groupby("method").size().items()},
                finite_failed_predictions_retained=True,no_failed_case_deleted=True,
                unit_note="Method/endpoints are gate cells, not independent Monte Carlo replicates or additional populations.")


def sparse57(records, out):
    refs=["ICC","Sikov_FH","Sikov_SAR","Sikov_SAR_Published"]
    frame=prepare_scores(records,256,["Selected",*refs])
    gates=numerical_gates(records,["Selected",*refs])
    if "areas" in frame:
        assert frame.loc[frame["mask"].eq("all")&frame.outcome.isin(inc.OUTCOMES),"areas"].eq(120).all()
        assert frame.loc[frame["mask"].eq("unsampled_common")&frame.outcome.isin(inc.OUTCOMES),"areas"].eq(63).all()
    all_summaries=[]
    for ref in refs:
        sub=frame.loc[frame.method.isin(["Selected",ref])].copy()
        sub.loc[sub.method.eq(ref),"method"]="ICC"
        summary=inc.summarize_increment(sub)
        summary["reference_method"]=ref
        summary=summary.rename(columns={"icc":"reference","icc_mcse":"reference_mcse","difference_selected_minus_icc":"difference_selected_minus_reference","ratio_selected_over_icc":"ratio_selected_over_reference","icc_coverage_CI_spans95":"reference_coverage_CI_spans95","icc_distance95_bound_low":"reference_distance95_bound_low","icc_distance95_bound_high":"reference_distance95_bound_high"})
        all_summaries.append(summary)
    summaries=pd.concat(all_summaries,ignore_index=True)
    summaries.to_csv(out/"sparse57_exploratory_summary.csv",index=False)
    metrics=["interval_score","RMSE_sqrt_mean_MSE","mean_population_RMSE","bias","coverage","width","aggregate_coverage_distance95"]
    global_rows=summaries.loc[summaries.scope.eq("equal_four_contexts")&summaries.metric.isin(metrics)]
    compact=global_rows.loc[global_rows.outcome.eq("endpoint_balanced")]
    compact.to_csv(out/"sparse57_compact_English_table.csv",index=False)
    global_all=compact.loc[compact["mask"].eq("all")]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":14.5,"axes.spines.right":False,"axes.spines.top":False})
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.4),layout="constrained")
    labels={"ICC":"ICC only","Sikov_FH":"Fay–Herriot","Sikov_SAR":"Full-graph SAR","Sikov_SAR_Published":"Observed-domain SAR"}
    for ax,metric,scale,title in zip(axes,["interval_score","RMSE_sqrt_mean_MSE"],[1,100],["95% interval score","RMSE (percentage points)"]):
        d=global_all.loc[global_all.metric.eq(metric)].set_index("reference_method").loc[refs]
        ax.axvline(0,color=".5",ls="--",lw=1)
        ax.errorbar(scale*d.difference_selected_minus_reference,np.arange(4),xerr=inc.Z*scale*d.difference_mcse,fmt="o",capsize=3,color="#176b87")
        ax.set_yticks(np.arange(4),[labels[r]for r in refs]);ax.invert_yaxis();ax.set_title(title)
        ax.set_xlabel("Proposed mixture minus reference");ax.grid(axis="x",alpha=.2)
    fig.suptitle("Independent sensitivity with 47.5% observed districts",fontsize=16)
    fig.savefig(out/"sparse57_comparisons_English.png",dpi=250,bbox_inches="tight")
    fig.savefig(out/"sparse57_comparisons_English.svg",bbox_inches="tight");plt.close(fig)
    claims=clean(global_all.to_dict("records"))
    lines=["# Exploratory independent sparse57 cohort","","All 256 separately generated populations are retained, with equal four-context and two-endpoint balance. This sensitivity uses57 sampled districts out of 120; it does not enter the original256-population primary family. All intervals below are unadjusted descriptive population Monte Carlo intervals; no p values, new Holm family or weight reselection is introduced.","","| Reference | IS95 difference | MCSE | Relative IS95 change | RMSE difference (pp) | Relative RMSE change |","|---|---:|---:|---:|---:|---:|"]
    for ref in refs:
        a=global_all.loc[global_all.reference_method.eq(ref)&global_all.metric.eq("interval_score")].iloc[0]
        b=global_all.loc[global_all.reference_method.eq(ref)&global_all.metric.eq("RMSE_sqrt_mean_MSE")].iloc[0]
        lines.append(f"|{labels[ref]}|{a.difference_selected_minus_reference:.6f}|{a.difference_mcse:.6f}|{a.relative_difference_percent:.2f}%|{100*b.difference_selected_minus_reference:.4f}|{b.relative_difference_percent:.2f}%|")
    lines += ["","Full masks, endpoints and context-specific summaries are retained in sparse57_exploratory_summary.csv. sqrt(mean MSE) and mean population RMSE are distinct. Greater coverage does not alone mean improved calibration. Bias ratios are not reported. No original Holm3 or Holm12 result is changed."]
    lines += ["",f"Recorded numerical gates failed in {gates['worlds_with_any_failed_gate']} of 256 populations, {gates['world_method_pairs_with_any_failed_gate']} world-method pairs and {gates['method_endpoint_world_gate_cells_failed']} method-endpoint-world cells. All finite predictions from failed cases remain in these descriptive aggregates. Completion of the cohort is distinct from passing all numerical gates."]
    (out/"sparse57_report_English.md").write_text("\n".join(lines)+"\n")
    status="PASS_EXPLORATORY_AGGREGATE_ANALYSIS"if gates["all_recorded_gates_passed"]else"COMPLETE_EXPLORATORY_AGGREGATES_WITH_NUMERICAL_GATE_FAILURES"
    inc.write_json(out/"sparse57_summary.json",dict(format="iaos-sparse57-exploratory-summary-v1",status=status,population_pairs=256,worlds_per_context=64,sampled_fraction=57/120,exploratory=True,original_families_changed=False,numerical_gates=gates,records_sha256=inc.sha(records),code_sha256=inc.sha(__file__),all_masks_global=clean(global_rows.to_dict("records")),figure_png="sparse57_comparisons_English.png",csv="sparse57_exploratory_summary.csv",**{"global":claims}))
    return {"status":status,"worlds":256,"output":str(out),"numerical_gates":gates}


def nuts_development(records,out,laplace):
    methods=["ICC","ICC_poverty","ICC_SAR_w25","ICC_SAR_w50","ICC_SAR_w75"]
    frame=prepare_scores(records,32,methods)
    gates=numerical_gates(records,methods)
    d=frame.loc[frame["mask"].eq("all")&frame.outcome.isin(inc.OUTCOMES)].rename(columns={"context":"scenario"})
    ranking=inc.rank_development(d)
    ranking.to_csv(out/"NUTS_candidate_ranking.csv",index=False)
    d.to_csv(out/"NUTS_32_world_candidate_scores.csv",index=False)
    loorows=[];allr=[]
    for(context,rep),held in d.groupby(["scenario","replication"]):
        rank=inc.rank_development(d.loc[~(d.scenario.eq(context)&d.replication.eq(rep))])
        winner,runner=rank.iloc[0],rank.iloc[1]
        allr.extend(rank.assign(held_context=context,held_replication=int(rep)).to_dict("records"))
        loorows.append(dict(held_context=context,held_replication=int(rep),selected_method=winner.method,runner_up=runner.method,training_IS95=float(winner.interval_score),training_margin_best_to_second=float(runner.interval_score-winner.interval_score),heldout_selected_IS95=float(held.loc[held.method.eq(winner.method)].interval_score.mean())))
    loo=pd.DataFrame(loorows);loo.to_csv(out/"NUTS_LOPO_selection_and_margins.csv",index=False)
    pd.DataFrame(allr).to_csv(out/"NUTS_LOPO_all_candidate_rankings.csv",index=False)
    lr=pd.read_csv(laplace/"Laplace_candidate_ranking.csv")
    comparison=ranking.merge(lr,on="method",suffixes=("_NUTS","_Laplace"),validate="one_to_one")
    comparison["rank_change_NUTS_minus_Laplace"]=comparison.rank_NUTS-comparison.rank_Laplace
    comparison["IS95_change_NUTS_minus_Laplace"]=comparison.interval_score_NUTS-comparison.interval_score_Laplace
    comparison.to_csv(out/"NUTS_Laplace_grid_comparison.csv",index=False)
    status="PASS_EXPLORATORY_NUTS_DEVELOPMENT_GRID"if gates["all_recorded_gates_passed"]else"COMPLETE_DESCRIPTIVE_NUTS_GRID_WITH_NUMERICAL_GATE_FAILURES"
    dev=dict(format="iaos-NUTS-development-sensitivity-v1",status=status,worlds=32,worlds_per_context=8,eligible_candidates=methods,numerical_gates=gates,numerically_converged_grid_available=gates["all_recorded_gates_passed"],ranking_is_descriptive_only=True,selection_stability_claim_permitted=bool(gates["all_recorded_gates_passed"]),integration="four-chain NUTS with unchanged evaluated priors and gates; fresh native SAR B100 reused after input hash matches",best_method=str(ranking.iloc[0].method),second_method=str(ranking.iloc[1].method),best_IS95=float(ranking.iloc[0].interval_score),best_second_absolute_margin=float(ranking.iloc[1].interval_score-ranking.iloc[0].interval_score),best_second_relative_margin_percent=float(100*(ranking.iloc[1].interval_score/ranking.iloc[0].interval_score-1)),LOPO_counts=dict(Counter(loo.selected_method)),LOPO_min_margin=float(loo.training_margin_best_to_second.min()),LOPO_max_margin=float(loo.training_margin_best_to_second.max()),same_full_grid_winner_as_Laplace=bool(ranking.iloc[0].method==lr.sort_values("rank").iloc[0].method),frozen75_25_primary_recipe_changed=False,original_Holm_families_changed=False,exploratory=True,records_sha256=inc.sha(records),code_sha256=inc.sha(__file__),candidate_ranking=clean(ranking.to_dict("records")),comparison=clean(comparison.to_dict("records")))
    inc.write_json(out/"NUTS_development_grid_audit.json",dev)
    text=("# Exploratory NUTS development sensitivity\n\nAll 32 original development worlds and the original five eligible candidates are retained. Four-chain NUTS replaces the earlier Laplace integration for ICC and ICC_poverty; all mixtures retain their original fixed weights and CDF construction. No fresh evaluation truths select a recipe. This is an exploratory approximation sensitivity and does not reselect the main 75/25 strategy or alter Holm3/Holm12.\n\n"
          f"NUTS winner:{dev['best_method']}; runner-up:{dev['second_method']}; IS95 gap:{dev['best_second_absolute_margin']:.8f} ({dev['best_second_relative_margin_percent']:.3f}%). LOPO selections:{dev['LOPO_counts']}. Same full-grid winner as Laplace:{dev['same_full_grid_winner_as_Laplace']}. Minimum LOPO margin:{dev['LOPO_min_margin']:.8f}.\n\nRecorded gate failures affect {gates['worlds_with_any_failed_gate']} of 32 populations and {gates['method_endpoint_world_gate_cells_failed']} method-endpoint-world cells. All finite failures remain in the ranking; a ranking involving failed numerical gates cannot establish stability of weight selection. Even a fully converged sensitivity does not establish a universal optimum.\n\nAll candidate scores, Laplace/NUTS changes and held-population rankings are supplied as CSV, including any adverse ranking changes. The sampler convergence receipt remains separate from prediction accuracy.\n")
    (out/"NUTS_development_report_English.md").write_text(text)
    return {"status":dev["status"],"output":str(out),"best_method":dev["best_method"]}


def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["sparse57","nuts-development"],required=True);p.add_argument("--records",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--laplace",type=Path)
    a=p.parse_args();repo=Path(__file__).resolve().parents[2];a.output.resolve().relative_to(repo/"revision_20261015")
    a.output.mkdir(parents=True,exist_ok=True)
    result=sparse57(a.records,a.output)if a.mode=="sparse57"else nuts_development(a.records,a.output,a.laplace or repo/"revision_20261015/results/devgrid")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
