"""Publication figures from sealed aggregate scores; no fitting or selection."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
CONTEXTS=('SAR_probability_srs','SAR_logit_srs','SAR_logit_psu_sparse','IID_logit_psu_sparse')
NAMES=('Probability SAR, SRS','Logit SAR, SRS','Logit SAR, sparse PSUs','IID logit, sparse PSUs')
METHODS=('Sikov_FH','Sikov_SAR','Sikov_SAR_Published','ICC','Selected')
LABELS=('Sikov FH','Full-graph SAR','Published-scope SAR','ICC beta','Frozen 75/25 mixture')
COLORS=('#9e9e9e','#d7852d','#79599b','#527b9a','#184c3c')
REFERENCES=('Sikov_FH','Sikov_SAR','Sikov_SAR_Published')

def main():
    joint_path=ROOT/'validacion/joint_prediction_lock.json'
    joint=json.loads(joint_path.read_text())
    add_path=ROOT/'validacion/sikov_published_addendum_freeze.json'
    add=json.loads(add_path.read_text())
    proof=json.loads((ROOT/'validacion/sikov_published_evaluation_status.json').read_text())
    if (proof['status']!='PASS_COMPLETE_PRESPECIFIED_ADDITIVE_HOLM3_FAMILY'
            or not proof['all256_worlds_included'] or not proof['all_models_converged']
            or proof['selected_recipe_changed'] or not joint['approved_for_scoring']
            or joint['fresh_cohort_size']!=256 or joint['fresh_truth_read_before_joint']
            or joint['heldout2024responses_used_before_joint']
            or proof['protocol_sha256']!=SHA(ROOT/'protocol_freeze.json')
            or proof['addendum_sha256']!=SHA(add_path)
            or proof['joint_prediction_lock_sha256']!=SHA(joint_path)):
        raise RuntimeError('Complete joint-sealed expanded score comparison required')
    for record in joint['files']:
        if SHA(record['path'])!=record['sha256']:raise RuntimeError('Jointly sealed prediction changed')
    for name,value in add['additional_sources_sha256'].items():
        if SHA(name)!=value:raise RuntimeError('Frozen addendum source changed')
    source=ROOT/'resultados/published_sar_comparison'
    for name,value in proof['outputs_sha256'].items():
        if SHA(source/name)!=value:raise RuntimeError('Changed aggregate '+name)
    d=pd.read_csv(source/'summary.csv')
    d=d.loc[d.group.eq('all')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    out=ROOT/'resultados/figuras';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(11.5,7.8),layout='constrained')
    x=np.arange(4);w=.155
    for j,ep in enumerate(('anemia','stunting')):
        for k,metric in enumerate(('rmse','coverage')):
            ax=axes[j,k]
            for t,(method,label,color) in enumerate(zip(METHODS,LABELS,COLORS)):
                rows=d.loc[d.outcome.eq(ep)&d.method.eq(method)].set_index('scenario').loc[list(CONTEXTS)]
                scale=100.
                ax.bar(x+(t-2)*w,scale*rows[metric],w,color=color,label=label,
                       yerr=1.959963984540054*scale*rows[metric+'_mcse'],capsize=2,error_kw={'linewidth':.7})
            if metric=='coverage':ax.axhline(95,color='#333333',linestyle='--',linewidth=.9);ax.set_ylim(0,103)
            ax.set_title(('Anemia' if j==0 else 'Stunting')+(' — RMSE' if k==0 else ' — interval coverage'))
            ax.set_xticks(x,NAMES,rotation=20,ha='right',fontsize=8)
            ax.set_ylabel('Percentage points' if k==0 else 'Percent')
            ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncols=3,fontsize=9,frameon=False)
    fig.savefig(out/'matched_rmse_coverage.png',dpi=300)
    fig.savefig(out/'matched_rmse_coverage.pdf');plt.close(fig)
    raw=pd.read_csv(source/'paired_balanced_contrasts_Holm3.csv')
    projected=pd.read_csv(source/'physical_projection_contrasts.csv')
    fig,ax=plt.subplots(figsize=(8,4.2),layout='constrained')
    for t,(data,label,col) in enumerate([(raw,'Raw model predictions','#184c3c'),(projected,'Shared [0,1] projection','#527b9a')]):
        a=data.loc[data.metric.eq('interval_score')&data.reference.isin(REFERENCES)].set_index('reference').loc[list(REFERENCES)]
        y=np.arange(3)+(t-.5)*.17
        ax.errorbar(a.difference,y,xerr=np.vstack([a.difference-a.MC_CI95_low,a.MC_CI95_high-a.difference]),fmt='o',capsize=4,color=col,label=label)
    ax.axvline(0,color='#333333',linestyle='--',linewidth=.9)
    ax.set_yticks([0,1,2],['Versus Sikov FH','Versus full-graph SAR','Versus published-scope SAR'])
    ax.set_xlabel('Difference in balanced interval score (lower favours improvement)')
    ax.set_title('Frozen improvement on 256 fresh finite populations')
    ax.legend(frameon=False,fontsize=9);ax.grid(axis='x',alpha=.15)
    fig.savefig(out/'matched_interval_score.png',dpi=300);fig.savefig(out/'matched_interval_score.pdf');plt.close(fig)
    report={'score_manifest_sha256':SHA(ROOT/'validacion/sikov_published_evaluation_status.json'),'worlds':256,
      'protocol_sha256':SHA(ROOT/'protocol_freeze.json'),'addendum_sha256':SHA(add_path),
      'joint_prediction_lock_sha256':SHA(joint_path),'methods':METHODS,
      'errorbars':'paired/world Monte Carlo normal95%, descriptive intervals; primary tests use Holm3',
      'original_Holm2_results_preserved_separately':True,'exact_original_numeric_replication':False,
      'outputs_sha256':{p.name:SHA(p)for p in out.iterdir()if p.is_file()},'scientific_scales':'RMSE percentage points, coverage percent, interval score proportions'}
    (ROOT/'validacion/comparison_figures.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
