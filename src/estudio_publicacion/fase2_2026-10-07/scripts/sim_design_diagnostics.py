"""New finite-population design calibration diagnostics, independent of phase1.

Fitting reads sampled units only. Realized finite-population truth is written
to a separate evaluation branch and is opened only by the final collector.
No reserved ENDES2024 response is used by this component.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

PHASE=Path(__file__).resolve().parents[1]
STUDY=PHASE.parent
sys.path.insert(0,str(STUDY/'scripts'))
sys.path.insert(0,str(STUDY/'scripts/competitors'))
from models import spatial_basis
from beta_power import working_icc, cluster_effective_n
from diseno_calibration import calibrate, stratified_psu_counts, CONFIG
from diseno_fit_cut import conditional_fit, pool_equal_weight

CONTEXTS={'dense6':6,'sparse2':2}
POPULATIONS_PER_CONTEXT=20

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v))+'\n')

def gate():
    path=PHASE/'protocol_freeze.json'
    proof=json.loads(path.read_text())
    if not proof.get('approved_for_development_fits',False):raise RuntimeError('Scientific protocol is not frozen')
    expected=proof['source_sha256']['scripts/sim_design_diagnostics.py']
    if sha(__file__)!=expected:raise RuntimeError('Simulation source changed after freeze')
    return sha(path)

def template():
    rng=np.random.default_rng(np.random.SeedSequence([20261007,701]))
    coords=np.array([[-17+1.5*(a//10),-80+.5*(a%10)] for a in range(80)],float)
    raw=rng.normal(size=(80,6))
    raw[:,1]=.3*raw[:,0]+np.sqrt(.91)*raw[:,1]
    X=np.c_[np.ones(80),(raw-raw.mean(0))/raw.std(0)]
    B=spatial_basis(coords,rank=12)
    return X,B

def generate_world(context,rep):
    if context not in CONTEXTS or not 0<=rep<POPULATIONS_PER_CONTEXT:raise ValueError('Unplanned population')
    root=PHASE/f'datos/synthetic_design/{context}/rep_{rep:03d}'
    truthdir=PHASE/f'datos/synthetic_design_evaluation/{context}/rep_{rep:03d}'
    if root.exists() or truthdir.exists():raise FileExistsError('No overwriting generated populations')
    root.mkdir(parents=True);truthdir.mkdir(parents=True)
    rng=np.random.default_rng(np.random.SeedSequence([20261007,702,CONTEXTS[context],rep]))
    sampling_rng=np.random.default_rng(np.random.SeedSequence([20261007,703,CONTEXTS[context],rep]))
    X,B=template()
    beta_a=np.array([logit(.30),.35,.20,-.10,-.20,.10,.15])
    beta_s=np.array([logit(.13),.30,.15,-.10,-.15,.05,.10])
    eta_a=X@beta_a+B@rng.normal(0,.2,12)
    eta_s=X@beta_s+B@rng.normal(0,.2,12)
    m=CONTEXTS[context];M=120
    rows=[];frames=[];truth=[];population_sizes=[]
    for a in range(80):
        sigma=.35+.75*expit(X[a,1])
        u=rng.normal(size=(M,2))
        ba=sigma*u[:,0]
        bs=sigma*(.4*u[:,0]+np.sqrt(.84)*u[:,1])
        sizes=np.maximum(rng.poisson(20*np.exp(.35*ba-.5*.35**2*sigma**2)),1)
        selected=set(sampling_rng.choice(M,m,replace=False).tolist())
        total=np.zeros(2);N=0
        for p in range(M):
            n=int(sizes[p]);H=rng.binomial(1,expit(-.2+.3*X[a,1]+.3*ba[p]),n)
            z=rng.normal(size=(n,2));z[:,1]=.4*z[:,0]+np.sqrt(.84)*z[:,1]
            pa=expit(eta_a[a]+ba[p]+.6*(H-.5))
            ps=expit(eta_s[a]+bs[p]+.5*(H-.5))
            yy=np.c_[(norm.cdf(z[:,0])<pa).astype(int),(norm.cdf(z[:,1])<ps).astype(int)]
            total+=yy.sum(0);N+=n
            if p not in selected:continue
            key=f'{a:03d}:{p:03d}'
            frames.append({'psu_key':key,'stratum':a})
            # Known Bernoulli child inclusion depends on an outcome-related H.
            # No estimated response model or fabricated real-survey correction.
            q=np.where(H==1,.35,.80)
            take=np.flatnonzero(sampling_rng.uniform(size=n)<q)
            for k in take:
                w=(M/m)/q[k]
                rows.append({'area':a,'psu_key':key,'psu':key,'H':int(H[k]),
                    'y_anemia':int(yy[k,0]),'y_stunting':int(yy[k,1]),
                    'weight_anemia':w,'weight_stunting':w})
        truth.append(total/N);population_sizes.append(N)
    units=pd.DataFrame(rows);frame=pd.DataFrame(frames)
    units.to_csv(root/'sample.csv',index=False);frame.to_csv(root/'frame.csv',index=False)
    np.savez_compressed(root/'known_auxiliary.npz',X=X,basis=B)
    np.savez_compressed(truthdir/'realized_population_truth.npz',theta=np.asarray(truth),population_size=population_sizes)
    save(root/'generation.json',{'context':context,'rep':rep,'population_stream':[20261007,702,m,rep],
        'sampling_stream':[20261007,703,m,rep],'sampling_choices_do_not_change_remaining_population':True,
        'areas':80,'population_PSUs_per_area':M,'sampled_PSUs_per_stratum':m,
        'complete_frame_PSUs':len(frame),'measured_sample_children':len(units),
        'within_PSU_sampling':'independent Bernoulli probabilities0.35/0.80 by H, known inverse inclusion weights',
        'PSU_selection':'stratified simple random sampling without replacement, each district a stratum',
        'information_limit':'ultimate-cluster WR working variance has no separate lower-stage or finite-population-correction calculation; sub-sampling noise enters sampled PSU scores; evaluated against realized finite population',
        'finite_population_truth_read_by_fitting':False,'source_sha256':sha(__file__),
        'sample_sha256':sha(root/'sample.csv'),'frame_sha256':sha(root/'frame.csv'),
        'aux_sha256':sha(root/'known_auxiliary.npz'),
        'realized_truth_sha256':sha(truthdir/'realized_population_truth.npz')})

def fit_world(context,rep):
    freeze=gate();src=PHASE/f'datos/synthetic_design/{context}/rep_{rep:03d}'
    out=PHASE/f'resultados/design_simulation/{context}/rep_{rep:03d}'
    if (out/'completion.json').exists():raise FileExistsError('Completed population retained')
    out.mkdir(parents=True,exist_ok=True)
    generation=json.loads((src/'generation.json').read_text())
    for name,key in [('sample.csv','sample_sha256'),('frame.csv','frame_sha256'),('known_auxiliary.npz','aux_sha256')]:
        if sha(src/name)!=generation[key]:raise RuntimeError('Generated sampled input changed: '+name)
    frame=pd.read_csv(src/'frame.csv',dtype={'psu_key':str})
    units=pd.read_csv(src/'sample.csv',dtype={'psu_key':str,'psu':str})
    sampled=np.array(sorted(units.area.unique()),int)
    counts,_=stratified_psu_counts(frame)
    cal=[calibrate(frame,units,sampled,j,counts=counts) for j in ('anemia','stunting')]
    aux=dict(np.load(src/'known_auxiliary.npz'))
    inputs={**aux,'sample_idx':sampled,'y':np.column_stack([v['direct']['y'] for v in cal])}
    kish=np.column_stack([v['direct']['n_kish'] for v in cal])
    baseline=[];baseline_information={}
    for j,name in enumerate(('anemia','stunting')):
        rho,icc_information=working_icc(units,name)
        n,effective_information=cluster_effective_n(units,name,sampled,kish[:,j],rho)
        baseline.append(n)
        baseline_information[name]={'working_rho':rho,'icc_information':icc_information,
                                    'effective_n_information':effective_information}
    state_n=[np.column_stack([v['mixture_effective_n'][k] for v in cal]) for k in range(8)]
    tasks=[('baseline',np.column_stack(baseline)),('point',np.column_stack([v['central']['effective_n'] for v in cal]))]
    tasks += [(f'cut_state_{k:02d}',n) for k,n in enumerate(state_n)]
    model_inputs=out/'model_inputs';model_inputs.mkdir(exist_ok=True)
    np.savez_compressed(model_inputs/'effective_information.npz',
                        sample_idx=sampled,y=inputs['y'],kish=kish,
                        baseline_n=tasks[0][1],point_n=tasks[1][1],cut8_n=np.stack(state_n),
                        selected_replica_indices=np.stack([v['status']['mixture_state_indices'] for v in cal]))
    save(model_inputs/'baseline_icc_information.json',baseline_information)
    predictions={};statuses=[]
    seed=20261007+1000000*CONTEXTS[context]+rep*10000
    for k,(name,n) in enumerate(tasks):
        target=out/name
        try:
            result,status=conditional_fit(inputs,n,target,seed+k*1009,f'{context}_{rep:03d}_{name}')
        except RuntimeError as exc:
            # Retain finite final predictions even if convergence gates failed.
            # Integrity/gate/resume failures are never bypassed by saved output.
            if not str(exc).startswith('Conditional state failed the predeclared NUTS gates:'):raise
            if not (target/'posterior_summary_and_draws.npz').exists():raise
            status=json.loads((target/'fit_status.json').read_text())
            if status.get('success',True):raise
            data=np.load(target/'posterior_summary_and_draws.npz')
            result={key:data[key] for key in data.files}
            if not all(np.all(np.isfinite(result[key])) for key in ('p_mean','lower95','upper95','p_draws')):raise
        statuses.append(status);predictions[name]=result
        print(json.dumps({'context':context,'rep':rep,'conditional_task':name,'success':status['success']}),flush=True)
    mixture=pool_equal_weight([predictions[f'cut_state_{k:02d}'] for k in range(8)])
    final={'baseline':predictions['baseline'],'point':predictions['point'],'cut8':mixture}
    rows=[]
    for name,v in final.items():
        for a in range(80):
            for j,endpoint in enumerate(('anemia','stunting')):
                rows.append({'area':a,'method':name,'outcome':endpoint,'sampled':bool(a in sampled),
                    'mean':v['p_mean'][a,j],'lower95':v['lower95'][a,j],'upper95':v['upper95'][a,j]})
    pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False)
    save(out/'completion.json',{'complete':True,'all_conditional_states_passed':all(s['success'] for s in statuses),
        'context':context,'rep':rep,'protocol_sha256':freeze,'source_sha256':sha(__file__),
        'known_truth_opened':False,'conditional_tasks':statuses,'predictions_sha256':sha(out/'predictions.csv'),
        'effective_information_sha256':sha(model_inputs/'effective_information.npz'),
        'baseline_icc_information_sha256':sha(model_inputs/'baseline_icc_information.json'),
        'bootstrap_calibration_status':[v['status'] for v in cal]})

def collect():
    gate();rows=[];worlds=[]
    for context in CONTEXTS:
        for rep in range(POPULATIONS_PER_CONTEXT):
            out=PHASE/f'resultados/design_simulation/{context}/rep_{rep:03d}'
            comp=json.loads((out/'completion.json').read_text())
            if not comp['complete'] or sha(out/'predictions.csv')!=comp['predictions_sha256']:raise RuntimeError('Incomplete/unverified population')
            generated=PHASE/f'datos/synthetic_design/{context}/rep_{rep:03d}'
            expected=json.loads((generated/'generation.json').read_text())
            truthpath=PHASE/f'datos/synthetic_design_evaluation/{context}/rep_{rep:03d}/realized_population_truth.npz'
            if sha(truthpath)!=expected['realized_truth_sha256']:raise RuntimeError('Realized finite truth changed')
            truth=np.load(truthpath)['theta']
            predictions=pd.read_csv(out/'predictions.csv')
            for (method,outcome),d in predictions.groupby(['method','outcome'],sort=True):
                j=0 if outcome=='anemia' else 1
                d=d.sort_values('area');theta=truth[d.area.to_numpy(int),j]
                mean,lo,hi=[d[x].to_numpy(float) for x in ('mean','lower95','upper95')]
                width=hi-lo
                score=width+40*np.maximum(lo-theta,0)+40*np.maximum(theta-hi,0)
                rows.append({'context':context,'rep':rep,'method':method,'outcome':outcome,
                    'mse':np.mean((mean-theta)**2),'bias':np.mean(mean-theta),'coverage95':np.mean((lo<=theta)&(theta<=hi)),
                    'mean_width':width.mean(),'interval_score95':score.mean(),'areas':len(d),
                    'all_conditional_states_passed':comp['all_conditional_states_passed']})
            worlds.append({'context':context,'rep':rep,'completion_sha256':sha(out/'completion.json')})
    out=PHASE/'resultados/design_simulation/final';out.mkdir(exist_ok=True)
    d=pd.DataFrame(rows);d.to_csv(out/'population_metrics.csv',index=False)
    summaries=[]
    for (context,method,outcome),s in d.groupby(['context','method','outcome']):
        z={'context':context,'method':method,'outcome':outcome,'populations':len(s),'rmse':np.sqrt(s.mse.mean())}
        for col in ('mse','bias','coverage95','mean_width','interval_score95'):
            z[col]=s[col].mean();z[col+'_MCSE']=s[col].std(ddof=1)/np.sqrt(len(s))
        summaries.append(z)
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    paired=[]
    for context in CONTEXTS:
        for outcome in ('anemia','stunting'):
            subset=d.loc[(d.context==context)&(d.outcome==outcome)]
            for reference in ('baseline','point'):
                for metric in ('mse','bias','coverage95','mean_width','interval_score95'):
                    wide=subset.pivot(index='rep',columns='method',values=metric)
                    if len(wide)!=20 or wide[['cut8',reference]].isna().any().any():raise RuntimeError('Paired diagnostic cohort incomplete')
                    delta=wide.cut8-wide[reference]
                    paired.append({'context':context,'outcome':outcome,'reference':reference,'metric':metric,
                        'populations':len(delta),'cut8_minus_reference':delta.mean(),
                        'paired_MCSE':delta.std(ddof=1)/np.sqrt(len(delta)),
                        'scope':'unadjusted diagnostic Monte Carlo, excluded from primary external Holm8'})
    pd.DataFrame(paired).to_csv(out/'paired_diagnostics.csv',index=False)
    save(out/'completion.json',{'status':'COMPLETE_NEW_CALIBRATION_DIAGNOSTIC_COHORT','unique_populations':40,
        'cohort_scope':'20 dense6 and20 sparse2; diagnostic supplement, not confirmatory external family',
        'all40_completed':len(worlds)==40,'worlds':worlds,'source_sha256':sha(__file__),
        'phase1_populations_reused':False,'nonconverged_finite_predictions_retained':True,
        'primary_external_Holm_family_extended':False})

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','fit','collect'])
    p.add_argument('--context',choices=list(CONTEXTS));p.add_argument('--rep',type=int)
    a=p.parse_args();gate()
    if a.action=='collect':collect();return
    if a.context is None or a.rep is None:raise ValueError('Explicit planned context/rep required')
    if a.action=='generate':generate_world(a.context,a.rep)
    else:fit_world(a.context,a.rep)

if __name__=='__main__':main()
