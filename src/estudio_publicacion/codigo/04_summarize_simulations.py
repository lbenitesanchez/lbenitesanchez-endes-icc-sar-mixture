#!/usr/bin/env python3
"""Publication tables, paired Monte Carlo contrasts and scientific figures."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "resultados" / "simulaciones"
spec=importlib.util.spec_from_file_location("simulation_summary",Path(__file__).with_name("03_simulation.py"))
sim=importlib.util.module_from_spec(spec);spec.loader.exec_module(sim)


def direct_metrics():
    records=[]
    X,coords,basis=sim.geometry()
    for scenario,cfg in sim.SCENARIOS.items():
        for rep in range(cfg["reps"]):
            inputfolder="inputs_finite_v2" if cfg["kind"]=="binary" else "inputs"
            path=OUT/inputfolder/scenario/f"rep_{rep:04d}.npz"
            data=np.load(path)
            idx=data["sample_idx"]
            truth=data["theta"][idx]
            D=data["D"];observed=data["y"]
            se=np.sqrt(np.maximum(np.diagonal(D,axis1=1,axis2=2),0))
            if cfg["kind"]=="gaussian":
                p=expit(observed)
                lo,hi=expit(observed-1.959963984540054*se),expit(observed+1.959963984540054*se)
                regenerated=sim.generate_gaussian(rep,scenario,X,coords,basis)
                if not np.array_equal(regenerated["sample_idx"],idx) or not np.allclose(regenerated["y"],observed,rtol=0,atol=1e-14):
                    raise ValueError("Seed reconstruction differs from frozen simulation input")
                critical=student_t.ppf(.975,regenerated["df"])[:,None]
                limits=[("direct_Wald",lo,hi),
                        ("direct_t_known_df",expit(observed-critical*se),expit(observed+critical*se))]
            else:
                p=observed
                lo,hi=np.maximum(p-1.959963984540054*se,0),np.minimum(p+1.959963984540054*se,1)
                limits=[("direct_Wald",lo,hi)]
            priority=(p>sim.THRESHOLDS).all(axis=1).astype(float)
            true_priority=(truth>sim.THRESHOLDS).all(axis=1).astype(float)
            for interval,lo,hi in limits:
                for t,outcome in enumerate(sim.OUTCOME_NAMES):
                    e=p[:,t]-truth[:,t]
                    records.append(dict(scenario=scenario,replication=rep,method="Direct",
                        D_mode="estimated",interval=interval,group="sampled",outcome=outcome,
                        bias=e.mean(),mse=np.mean(e**2),mae=np.mean(abs(e)),
                        coverage=np.mean((lo[:,t]<=truth[:,t])&(truth[:,t]<=hi[:,t])),
                        width=np.mean(hi[:,t]-lo[:,t]),priority_brier=np.mean((priority-true_priority)**2),
                        priority_accuracy=np.mean(priority==true_priority),fit_seconds=0,
                        converged=True,areas=len(idx)))
    return pd.DataFrame(records)


def paired_contrast(a,b,label):
    keys=["scenario","group","outcome","replication"]
    pair=a.merge(b,on=keys,suffixes=("_candidate","_reference"))
    rows=[]
    for key,p in pair.groupby(keys[:-1]):
        R=len(p)
        M1=p.mse_candidate.mean();M0=p.mse_reference.mean()
        logratio=.5*np.log(M1/M0)
        influence=.5*(p.mse_candidate/M1-p.mse_reference/M0)
        selog=influence.std(ddof=1)/np.sqrt(R)
        dc=p.coverage_candidate-p.coverage_reference
        dw=p.width_candidate-p.width_reference
        rows.append(dict(zip(keys[:-1],key),contrast=label,paired_replications=R,
            rmse_relative_change_pct=100*np.expm1(logratio),
            rmse_relative_change_lo=100*np.expm1(logratio-1.959963984540054*selog),
            rmse_relative_change_hi=100*np.expm1(logratio+1.959963984540054*selog),
            coverage_change_pp=100*dc.mean(),coverage_change_mcse_pp=100*dc.std(ddof=1)/np.sqrt(R),
            width_change_pp=100*dw.mean(),width_change_mcse_pp=100*dw.std(ddof=1)/np.sqrt(R)))
    return pd.DataFrame(rows)


def sparse_direct_metrics():
    records=[]
    for scenario in ("sparse_srs_2025","sparse_psu_2025"):
        for rep in range(500):
            data=np.load(OUT/"inputs_sparse_v2"/scenario/f"rep_{rep:04d}.npz")
            y=data["y"];truth=data["theta"][data["sample_idx"]]
            D=data["D"];se=np.sqrt(np.maximum(np.diagonal(D,axis1=1,axis2=2),0))
            lo=np.maximum(y-1.959963984540054*se,0);hi=np.minimum(y+1.959963984540054*se,1)
            both=np.isfinite(y).all(axis=1)
            predpriority=(y[both]>sim.THRESHOLDS).all(axis=1)
            truepriority=(truth[both]>sim.THRESHOLDS).all(axis=1)
            for t,outcome in enumerate(sim.OUTCOME_NAMES):
                mask=np.isfinite(y[:,t]);e=y[mask,t]-truth[mask,t]
                records.append(dict(scenario=scenario,replication=rep,method="Direct",D_mode="estimated",
                    interval="direct_Wald",group="sampled",outcome=outcome,bias=e.mean(),mse=np.mean(e**2),
                    mae=np.mean(abs(e)),coverage=np.mean((lo[mask,t]<=truth[mask,t])&(truth[mask,t]<=hi[mask,t])),
                    width=np.mean(hi[mask,t]-lo[mask,t]),priority_brier=np.mean(predpriority!=truepriority),
                    priority_accuracy=np.mean(predpriority==truepriority),fit_seconds=0,converged=True,areas=int(mask.sum())))
    return pd.DataFrame(records)


def tables(primary,bootstrap,sparse,ablation=None,srs_df_ablation=None):
    direct=direct_metrics()
    direct.to_csv(OUT/"replications_direct.csv",index=False)
    d_sparse=sparse_direct_metrics();d_sparse.to_csv(OUT/"replications_direct_sparse.csv",index=False)
    pieces=[]
    for experiment,records in (("primary",pd.concat([primary,direct],ignore_index=True)),
                               ("bootstrap_matched",bootstrap),
                               ("sparse_stress",pd.concat([sparse,d_sparse],ignore_index=True))):
        piece=sim.summarize(records);piece["experiment"]=experiment;pieces.append(piece)
    if ablation is not None:
        piece=sim.summarize(ablation);piece["experiment"]="median_ablation";pieces.append(piece)
    if srs_df_ablation is not None:
        piece=sim.summarize(srs_df_ablation);piece["experiment"]="srs_df_ablation";pieces.append(piece)
    summary=pd.concat(pieces,ignore_index=True)
    summary.to_csv(OUT/"summary_all_final.csv",index=False)
    table=summary[summary.sensitivity=="all_valid_predictions"].copy()
    for c in ("bias","bias_mcse","rmse","rmse_mcse","mae","width","width_mcse"):
        table[c+"_pp"]=100*table[c]
    table["coverage_pct"]=100*table.coverage
    table["coverage_mcse_pp"]=100*table.coverage_mcse
    table.to_csv(OUT/"publication_table.csv",index=False)
    contrasts=[]
    for method in sim.METHODS:
        a=primary[(primary.method==method)&(primary.D_mode=="smooth_biascorrected")]
        b=primary[(primary.method==method)&(primary.D_mode=="raw")]
        contrasts.append(paired_contrast(a,b,f"{method}: corrected GVF versus regularized raw D"))
    base=primary[(primary.method=="FH")&(primary.D_mode=="smooth_biascorrected")]
    for method in ("MFH","SFH","MSFH"):
        a=primary[(primary.method==method)&(primary.D_mode=="smooth_biascorrected")]
        contrasts.append(paired_contrast(a,base,f"{method} versus FH, corrected GVF"))
    for method in sim.METHODS:
        a=sparse[(sparse.method==method)&(sparse.D_mode=="smooth_biascorrected")]
        b=sparse[(sparse.method==method)&(sparse.D_mode=="raw")]
        contrasts.append(paired_contrast(a,b,f"{method}: corrected GVF versus regularized raw D"))
    base=sparse[(sparse.method=="FH")&(sparse.D_mode=="smooth_biascorrected")]
    for method in ("MFH","SFH","MSFH"):
        a=sparse[(sparse.method==method)&(sparse.D_mode=="smooth_biascorrected")]
        contrasts.append(paired_contrast(a,base,f"{method} versus FH, corrected GVF"))
    conditional=bootstrap[bootstrap.interval=="conditional"]
    conditional=conditional[conditional.method=="MSFH"]
    for kind in ("bootstrap_fixed_D","bootstrap_D"):
        a=bootstrap[bootstrap.interval==kind]
        contrasts.append(paired_contrast(a,conditional,f"{kind} versus conditional MSFH"))
    if ablation is not None:
        for method in sim.METHODS:
            a=primary[(primary.method==method)&(primary.D_mode=="smooth_biascorrected")]
            b=ablation[ablation.method==method]
            contrasts.append(paired_contrast(a,b,f"{method}: corrected GVF versus median GVF, matched 200"))
    if srs_df_ablation is not None:
        for method in sim.METHODS:
            a=srs_df_ablation[srs_df_ablation.method==method]
            b=sparse[(sparse.scenario=="sparse_srs_2025")&(sparse.replication<200)&
                     (sparse.method==method)&(sparse.D_mode=="smooth_biascorrected")]
            contrasts.append(paired_contrast(a,b,f"{method}: measured n-1 versus PSU/stratum df proxy, matched SRS 200"))
    contrast=pd.concat(contrasts,ignore_index=True)
    contrast.to_csv(OUT/"paired_contrasts.csv",index=False)
    metadata=pd.read_json(OUT/"metadata_full_v1.jsonl",lines=True)
    description=metadata.groupby("scenario").agg(
        replications=("replication","count"),mean_sampled_areas=("sampled_areas","mean"),
        boundary_rate=("boundary_rate","mean"),mean_outcome_missing_rate=("outcome_missing_rate","mean"),
        covariance_psd_projections=("psd_projection_count","sum"),
        area_without_one_outcome=("unmeasured_area_count","sum"))
    description.to_csv(OUT/"data_generation_diagnostics.csv")
    pd.read_json(OUT/"metadata_sparse_v2.jsonl",lines=True).drop(columns="fit_statuses").groupby("scenario").mean(
        numeric_only=True).to_csv(OUT/"sparse_generation_diagnostics.csv")
    return summary,contrast


def figures(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":13,
                        "axes.spines.top":False,"axes.spines.right":False})
    figdir=ROOT/"figuras";figdir.mkdir(parents=True,exist_ok=True)
    selected=summary[(summary.sensitivity=="all_valid_predictions")&
                     (summary.group=="all")&(summary.interval=="conditional")&
                     (summary.experiment=="primary")]
    scenarios=["g_r0_s0","g_r6_s0","g_r0_s5","g_r6_s5"]
    labels=["ρ=0, φ=0","ρ=.6, φ=0","ρ=0, φ=.5","ρ=.6, φ=.5"]
    ordered=[(m,d) for m in sim.METHODS for d in ("raw","smooth_biascorrected")]
    fig,axes=plt.subplots(1,2,figsize=(10.8,5.2),sharey=True,layout="constrained")
    for ax,outcome,title in zip(axes,sim.OUTCOME_NAMES,("Anemia","Stunting")):
        values=np.array([[float(selected[(selected.scenario==s)&(selected.method==m)&
                                         (selected.D_mode==d)&(selected.outcome==outcome)].coverage.iloc[0])*100
                          for s in scenarios] for m,d in ordered])
        im=ax.imshow(values,vmin=75,vmax=97.5,cmap="RdYlGn",aspect="auto")
        for i in range(len(ordered)):
            for j in range(len(scenarios)):
                ax.text(j,i,f"{values[i,j]:.1f}",ha="center",va="center",fontsize=9)
        ax.set_xticks(range(4),labels,rotation=20,ha="right")
        ax.set_yticks(range(8),[f"{m} · {'raw D' if d=='raw' else 'corrected GVF'}" for m,d in ordered])
        ax.set_title(title)
    fig.colorbar(im,ax=axes,label="Empirical coverage of 95% latent intervals (%)",shrink=.85)
    fig.savefig(figdir/"simulation_coverage.png",dpi=300)
    fig.savefig(figdir/"simulation_coverage.pdf")
    plt.close(fig)

    colors=dict(FH="#27548A",MFH="#348A59",SFH="#D58631",MSFH="#775098")
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.8),sharey=False,layout="constrained")
    for ax,outcome,title in zip(axes,sim.OUTCOME_NAMES,("Anemia","Stunting")):
        for method in sim.METHODS:
            block=selected[(selected.D_mode=="smooth_biascorrected")&
                           (selected.method==method)&(selected.outcome==outcome)].set_index("scenario").loc[scenarios]
            ax.errorbar(range(4),block.rmse*100,yerr=1.96*block.rmse_mcse*100,
                        color=colors[method],marker="o",label=method,capsize=3)
        ax.set_xticks(range(4),labels,rotation=20,ha="right")
        ax.set_ylabel("RMSE (percentage points)");ax.set_title(title)
        ax.grid(axis="y",alpha=.2)
    axes[1].legend(loc="lower left",ncol=2,frameon=False)
    fig.savefig(figdir/"simulation_rmse.png",dpi=300)
    fig.savefig(figdir/"simulation_rmse.pdf");plt.close(fig)

    boot=summary[(summary.sensitivity=="all_valid_predictions")&(summary.group=="all")&
                 (summary.scenario=="g_r6_s5")&(summary.method=="MSFH")&
                 (summary.D_mode=="smooth_biascorrected")]
    # Conditional primary and bootstrap results are summarized separately;
    # use the matched 200-population bootstrap experiment for this figure.
    bsummary=pd.read_csv(OUT/"summary_bootstrap_v1.csv")
    boot=bsummary[(bsummary.sensitivity=="all_valid_predictions")&(bsummary.group=="all")&
                  (bsummary.method=="MSFH")]
    kinds=["conditional","bootstrap_fixed_D","bootstrap_D"]
    names=["Conditional","Bootstrap\nD fixed","Bootstrap\nD reestimated"]
    fig,axes=plt.subplots(1,2,figsize=(9.8,3.7),layout="constrained")
    for outcome,color,title in zip(sim.OUTCOME_NAMES,("#27548A","#D58631"),("Anemia","Stunting")):
        block=boot[boot.outcome==outcome].set_index("interval").loc[kinds]
        delta=-.04 if outcome=="anemia" else .04
        axes[0].errorbar(np.arange(3)+delta,block.coverage*100,yerr=1.96*block.coverage_mcse*100,
                         marker="o",capsize=3,color=color,label=title)
        axes[1].errorbar(np.arange(3)+delta,block.width*100,yerr=1.96*block.width_mcse*100,
                         marker="o",capsize=3,color=color,label=title)
    axes[0].axhline(95,color="gray",linestyle="--",linewidth=1)
    axes[0].set_ylabel("Coverage (%)");axes[1].set_ylabel("Mean width (percentage points)")
    for ax in axes:
        ax.set_xticks(range(3),names);ax.grid(axis="y",alpha=.2)
    axes[0].legend(frameon=False)
    fig.savefig(figdir/"simulation_bootstrap.png",dpi=300)
    fig.savefig(figdir/"simulation_bootstrap.pdf");plt.close(fig)

    sparse=summary[(summary.sensitivity=="all_valid_predictions")&(summary.group=="all")&
                   (summary.interval=="conditional")&(summary.experiment=="sparse_stress")]
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout="constrained")
    for ax,scenario,title in zip(axes,("sparse_srs_2025","sparse_psu_2025"),
                                 ("Empirical counts · SRS","Empirical counts · PSU stress")):
        for t,outcome in enumerate(sim.OUTCOME_NAMES):
            for d,marker,delta in (("raw","x",-.1),("smooth_biascorrected","o",.1)):
                b=sparse[(sparse.scenario==scenario)&(sparse.outcome==outcome)&(sparse.D_mode==d)].set_index("method").loc[list(sim.METHODS)]
                ax.errorbar(np.arange(4)+delta,b.coverage*100,yerr=1.96*b.coverage_mcse*100,
                    color="#27548A" if t==0 else "#D58631",marker=marker,linestyle="none",capsize=3,
                    label=f"{outcome.title()} · {'raw' if d=='raw' else 'corrected'}")
        ax.axhline(95,color="gray",linestyle="--",linewidth=1)
        ax.set_xticks(range(4),sim.METHODS);ax.set_ylim(0,100);ax.set_title(title)
        ax.set_ylabel("Coverage of latent 95% interval (%)");ax.grid(axis="y",alpha=.2)
    axes[1].legend(loc="lower left",frameon=False,fontsize=11)
    fig.savefig(figdir/"simulation_sparse_coverage.png",dpi=300)
    fig.savefig(figdir/"simulation_sparse_coverage.pdf");plt.close(fig)

    binary_stress_figures(summary,figdir,colors)


def binary_stress_figures(summary,figdir=None,colors=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":13,
                        "axes.spines.top":False,"axes.spines.right":False})
    if figdir is None:
        figdir=ROOT/"figuras"
    figdir.mkdir(parents=True,exist_ok=True)
    if colors is None:
        colors=dict(FH="#27548A",MFH="#348A59",SFH="#D58631",MSFH="#775098")
    # Realistic binary stress tests include the unadjusted direct estimates.
    # Their common comparison population is sampled areas, not all districts.
    binary=summary[(summary.sensitivity=="all_valid_predictions")&(summary.group=="sampled")&
                   (summary.experiment.isin(["primary","sparse_stress"]))]
    cases=["binary_informative_mar","sparse_srs_2025","sparse_psu_2025"]
    titles=["Informative Poisson\n+ MAR","ENDES counts\nSRS","ENDES counts\nPSU stress"]
    methods=["Direct",*sim.METHODS]
    fig,axes=plt.subplots(2,3,figsize=(10,6.7),sharey=True,layout="constrained")
    for t,outcome in enumerate(sim.OUTCOME_NAMES):
        for col,(scenario,title) in enumerate(zip(cases,titles)):
            ax=axes[t,col]
            for j,method in enumerate(methods):
                if method=="Direct":
                    b=binary[(binary.scenario==scenario)&(binary.method==method)&
                             (binary.interval=="direct_Wald")&(binary.outcome==outcome)].iloc[0]
                    ax.errorbar(j,b.coverage*100,yerr=1.96*b.coverage_mcse*100,color="#555555",
                                marker="D",linestyle="none",capsize=3,label="Direct Wald")
                else:
                    for d,marker,offset in (("raw","x",-.08),("smooth_biascorrected","o",.08)):
                        b=binary[(binary.scenario==scenario)&(binary.method==method)&
                                 (binary.D_mode==d)&(binary.interval=="conditional")&
                                 (binary.outcome==outcome)].iloc[0]
                        ax.errorbar(j+offset,b.coverage*100,yerr=1.96*b.coverage_mcse*100,
                                    color=colors[method],marker=marker,linestyle="none",capsize=3)
            ax.axhline(95,color="gray",linestyle="--",linewidth=1)
            ax.set_xticks(range(5),methods,rotation=20);ax.set_ylim(0,100)
            ax.grid(axis="y",alpha=.2)
            if t==0:ax.set_title(title)
            if col==0:ax.set_ylabel(f"{outcome.title()} coverage (%)")
    from matplotlib.lines import Line2D
    axes[1,2].legend(handles=[Line2D([],[],color="#555555",marker="D",linestyle="none",label="Direct Wald"),
        Line2D([],[],color="black",marker="x",linestyle="none",label="SAE · regularized raw D"),
        Line2D([],[],color="black",marker="o",linestyle="none",label="SAE · corrected GVF")],
        loc="lower left",frameon=False,fontsize=11)
    fig.savefig(figdir/"stress_test_coverage.png",dpi=300)
    fig.savefig(figdir/"stress_test_coverage.pdf");plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(10,6.6),layout="constrained")
    for t,outcome in enumerate(sim.OUTCOME_NAMES):
        for method in methods:
            rows=[]
            for scenario in cases:
                b=binary[(binary.scenario==scenario)&(binary.outcome==outcome)&(binary.method==method)]
                b=b[(b.interval=="direct_Wald") if method=="Direct" else
                    ((b.interval=="conditional")&(b.D_mode=="smooth_biascorrected"))]
                rows.append(b.iloc[0])
            b=pd.DataFrame(rows)
            color="#555555" if method=="Direct" else colors[method]
            for col,metric in enumerate(("bias","rmse")):
                axes[t,col].errorbar(range(3),b[metric]*100,yerr=1.96*b[metric+"_mcse"]*100,
                    color=color,marker="D" if method=="Direct" else "o",label=method,capsize=3)
        axes[t,0].axhline(0,color="gray",linewidth=1,linestyle="--")
        for col,metric in enumerate(("bias","rmse")):
            axes[t,col].set_xticks(range(3),["Poisson + MAR","Sparse SRS","Sparse PSU"],rotation=12)
            axes[t,col].set_ylabel(f"{outcome.title()} {metric.upper() if metric=='rmse' else metric} (pp)")
            axes[t,col].grid(axis="y",alpha=.2)
    axes[0,1].legend(frameon=False,ncol=3,fontsize=11)
    fig.savefig(figdir/"stress_test_bias_rmse.png",dpi=300)
    fig.savefig(figdir/"stress_test_bias_rmse.pdf");plt.close(fig)


def main():
    for tag in ("full_v1","bootstrap_v1","finite_v2","sparse_v2","ablation_v2","srs_df_ablation"):
        status=json.loads((OUT/f"completion_{tag}.json").read_text())
        if not status["completed"]:
            raise RuntimeError(f"Experiment {tag} not complete")
    legacy=pd.read_csv(OUT/"replications_full_v1.csv")
    primary=pd.concat([legacy[legacy.scenario!="binary_informative_mar"],
                       pd.read_csv(OUT/"replications_finite_v2.csv")],ignore_index=True)
    primary.to_csv(OUT/"replications_primary_validated.csv",index=False)
    bootstrap=pd.read_csv(OUT/"replications_bootstrap_v1.csv")
    sparse=pd.read_csv(OUT/"replications_sparse_v2.csv")
    ablation_path=OUT/"replications_ablation_v2.csv"
    ablation=pd.read_csv(ablation_path) if ablation_path.exists() else None
    srs_df_ablation=pd.read_csv(OUT/"replications_srs_df_ablation.csv")
    # Summaries remain separate by experiment, even for matched identical seeds.
    summary,contrasts=tables(primary,bootstrap,sparse,ablation,srs_df_ablation)
    figures(summary)
    (OUT/"summary_completion.json").write_text(json.dumps(dict(
        completed=True,primary_replications=2400,bootstrap_external_replications=200,
        sparse_external_replications=1000,
        unique_validated_external_populations=3400,
        bootstrap_external_are_matched_subset=True,
        median_ablation_external_are_matched_subset=True,
        srs_df_ablation_external_are_matched_subset=True,
        srs_df_ablation_matched_replications=200,
        conditional_bootstrap_inner_requested=19800,
        variance_resampling_bootstrap_inner_requested=19800,
        summary_rows=len(summary),paired_contrast_rows=len(contrasts),
        diagnostic_experiments_excluded=["smoke","benchmark","bootstrap_benchmark"],
        conditional_bootstrap_repeats_not_double_counted=True),indent=2))
    manifest={"status":"complete","unique_validated_external_populations":3400,
        "primary_external_replications":3400,
        "external_counts_by_scenario":{s:c["reps"] for s,c in sim.SCENARIOS.items()},
        "sparse_external_counts":{"sparse_srs_2025":500,"sparse_psu_2025":500},
        "matched_evaluation_contexts":{"bootstrap_g_r6_s5":200,"median_ablation_g_r6_s5":200,
            "srs_df_ablation_sparse_srs_2025":200},
        "external_model_fits_primary_and_sparse":27200,
        "bootstrap_inner_requested":39600,
        "legacy_binary_full_v1_excluded":int(legacy[legacy.scenario=="binary_informative_mar"].replication.nunique()),
        "source_mapping":{"gaussian_and_misspecified":"source_snapshots/full_v1/models.py",
            "binary_informative_mar":"source_snapshots/finite_v2/models.py",
            "sparse_srs_2025_and_sparse_psu_2025":"source_snapshots/sparse_v2/models.py",
            "bootstrap":"source_snapshots/bootstrap_v1/models.py",
            "median_ablation":"source_snapshots/ablation_v2/models.py",
            "srs_df_ablation":"source_snapshots/srs_df_ablation/models.py"},
        "source_sha256":{},"failures":{},"interpretation":
            "Finite and PSU stress results determine limitations; Gaussian-only coverage cannot establish calibrated ENDES intervals"}
    for tag in ("full_v1","finite_v2","sparse_v2","bootstrap_v1","ablation_v2","srs_df_ablation"):
        config=json.loads((OUT/f"config_{tag}.json").read_text())
        manifest["source_sha256"][tag]=config.get("models_source_sha256",config.get("models_sha256"))
        fail=OUT/f"failures_{tag}.jsonl"
        manifest["failures"][tag]=sum(1 for _ in fail.open())
    statuses=list((OUT/"bootstrap_status"/"bootstrap_v1").glob("*/*.json"))
    manifest["bootstrap_inner_finite"]=sum(json.loads(p.read_text())["B_finite"] for p in statuses)
    manifest["bootstrap_inner_converged"]=sum(json.loads(p.read_text())["B_converged"] for p in statuses)
    (OUT/"validated_experiment_manifest.json").write_text(json.dumps(manifest,indent=2))
    (OUT/"final_experiment_manifest.json").write_text(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
