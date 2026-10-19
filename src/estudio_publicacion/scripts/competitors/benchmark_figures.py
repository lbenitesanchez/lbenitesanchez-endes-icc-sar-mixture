#!/usr/bin/env python3
"""Scientific figures from completed independent comparative evaluation only."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
from benchmark_inputs import ROOT,OUT,sha256
sys.path.insert(0,str(ROOT/'.pydeps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'BetaPowerHMCGraph':'Power beta (HMC)','FH':'FH','MFH':'MFH','SFH':'Spatial FH',
    'MSFH':'Multivariate spatial FH','GaoAuthorStrict':'Gao–Wakefield',
    'GaoCommonContinuityX':'Gao adapted','HoggTSLN':'Hogg TSLN'}
COLORS={'BetaPowerHMCGraph':'#0072B2','SFH':'#666666','GaoAuthorStrict':'#E69F00',
    'GaoCommonContinuityX':'#009E73','HoggTSLN':'#CC79A7'}
SCENARIOS={'binary_informative_mar':'Informative sampling',
    'sparse_srs_2025':'Sparse SRS','sparse_psu_2025':'Sparse PSU',
    'sparse_srs_iid':'SRS with iid area effects','sparse_psu_bym2':'PSU with BYM2 area effects'}
ENDPOINTS={'anemia':'Anaemia','stunting':'Stunting'}


def panel(summary,scenarios,methods,metric,multiplier,ylabel,output):
    fig,axes=plt.subplots(2,len(scenarios),figsize=(10,6.5),squeeze=False,sharey=True)
    for j,scenario in enumerate(scenarios):
        for i,outcome in enumerate(ENDPOINTS):
            ax=axes[i,j]
            subset=summary[(summary.scenario==scenario)&(summary.outcome==outcome)&(summary.group=='all')]
            for k,method in enumerate(methods):
                row=subset[subset.method==method]
                if len(row)!=1:raise ValueError(f'Expected one complete summary row {scenario}/{outcome}/{method}')
                row=row.iloc[0];value=row[metric]*multiplier;error=1.95996398454*row[metric+'_mcse']*multiplier
                ax.errorbar(k,value,yerr=error,fmt='o',color=COLORS[method],capsize=3,ms=6)
            ax.set_xticks(range(len(methods)));ax.set_xticklabels([LABELS[m]for m in methods],rotation=35,ha='right')
            if i==0:ax.set_title(SCENARIOS[scenario]);ax.tick_params(axis='x',labelbottom=False)
            if j==0:ax.set_ylabel(f'{ENDPOINTS[outcome]}\n{ylabel}')
            if metric=='coverage':ax.axhline(95,color='#444444',ls='--',lw=.8);ax.set_ylim(0,101)
            ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(output,dpi=300,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation-tag',default='final_evaluation');args=p.parse_args()
    folder=OUT/args.evaluation_tag;completion=json.loads((folder/'final_completion.json').read_text())
    if not completion['evaluation_completed']:raise RuntimeError('Final evaluation incomplete')
    summary=pd.read_csv(folder/'combined_summary.csv');scores=pd.read_csv(folder/'primary_balanced_interval_score.csv')
    figures=folder/'figures';figures.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size':12,'axes.titlesize':13,'axes.labelsize':12,'xtick.labelsize':12,'ytick.labelsize':12})
    methods=list(COLORS)
    panel(summary,list(SCENARIOS)[:3],methods,'coverage',100,'95% interval coverage (%)',figures/'primary_coverage.png')
    panel(summary,list(SCENARIOS)[:3],methods,'rmse',100,'RMSE (percentage points)',figures/'primary_RMSE.png')
    panel(summary,list(SCENARIOS)[3:],methods,'coverage',100,'95% interval coverage (%)',figures/'structural_coverage.png')
    panel(summary,list(SCENARIOS)[:3],methods,'interval_score',1,'95% interval score',figures/'primary_interval_score.png')
    if scores.primary_test_available.all():
        fig,ax=plt.subplots(figsize=(9,3.4))
        for k,row in scores.iterrows():
            ax.errorbar(row.interval_score_change,k,
                xerr=[[row.interval_score_change-row.change_ci95_lower],[row.change_ci95_upper-row.interval_score_change]],
                fmt='o',color=COLORS[row.reference],capsize=5,ms=7)
        ax.axvline(0,color='#444444',ls='--',lw=1)
        ax.set_yticks(range(3));ax.set_yticklabels([LABELS[m]for m in scores.reference])
        ax.set_xlabel('Power beta minus comparator: balanced 95% interval score')
        ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
        fig.tight_layout();fig.savefig(figures/'primary_balanced_contrasts.png',dpi=300,bbox_inches='tight');plt.close(fig)
    manifest=dict(summary_sha256=sha256(folder/'combined_summary.csv'),
        score_sha256=sha256(folder/'primary_balanced_interval_score.csv'),
        methods_displayed=methods,uncertainty='Normal 95% Monte Carlo error intervals between independent finite populations',
        figure_sha256={f.name:sha256(f)for f in figures.glob('*.png')},
        figure_source_sha256=sha256(Path(__file__)),interpretation='Scenario-specific simulation results; no universal ranking')
    (figures/'figure_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
