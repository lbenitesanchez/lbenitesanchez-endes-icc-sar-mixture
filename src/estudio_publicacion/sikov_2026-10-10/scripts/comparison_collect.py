"""Score finite-population truth only after the complete prediction lock."""
from pathlib import Path
import argparse,datetime as dt,json
import numpy as np
import pandas as pd
from scipy.stats import norm
from comparison_simulation import ROOT,CONTEXTS,EVALUATION_IDS,sha,save
from comparison_evaluate import gate,METHODS
from improve_development import scores
from benchmark_metrics import summarize,paired_contrasts

def prediction_lock():
    proof,freeze=gate();items=[]
    for context in CONTEXTS:
        for rep in EVALUATION_IDS:
            folder=ROOT/'evaluacion/fits'/context/f'rep_{rep:05d}'
            p=folder/'completion.json'
            if not p.is_file():raise RuntimeError('Incomplete final cohort '+str(p))
            completion=json.loads(p.read_text())
            if completion['protocol_sha256']!=freeze or completion['truth_read_by_fits']:raise ValueError('Wrong final fit role')
            for name,value in completion['output_sha256'].items():
                if sha(folder/name)!=value:raise RuntimeError('Changed completed output '+str(folder/name))
            inputs=ROOT/'evaluacion/inputs'/context/f'rep_{rep:05d}/model_inputs'
            if sha(inputs.parent/'manifest.json')!=completion['input_manifest_sha256']:raise RuntimeError('Changed generation manifest')
            if sha(inputs/'inputs.npz')!=completion['fit_inputs_sha256'] or sha(inputs/'sample.csv')!=completion['sample_sha256']:raise RuntimeError('Fit inputs changed')
            if set(completion['convergence'])!=set(METHODS):raise ValueError('Methods changed')
            items.append({'context':context,'replication':rep,'completion_sha256':sha(p),'all_converged':all(completion['convergence'].values()),'outputs':completion['output_sha256']})
    lock={'protocol_sha256':freeze,'locked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'approved_for_scoring':True,'worlds':len(items),'all_expected_worlds':len(CONTEXTS)*len(EVALUATION_IDS),'prediction_items':items,'fresh_truth_read_before_lock':False,'convergence_failures_retained':sum(not x['all_converged'] for x in items)}
    out=ROOT/'validacion/fresh_prediction_lock.json'
    if out.exists():
        previous=json.loads(out.read_text())
        if previous['protocol_sha256']!=freeze or previous['prediction_items']!=items:raise RuntimeError('Existing prediction lock differs')
    else:save(out,lock)
    return lock

def balanced(frame,candidate,reference):
    d=frame.loc[frame.group.eq('all')].copy()
    keys=['scenario','replication','outcome']
    a=d.loc[d.method.eq(candidate)];b=d.loc[d.method.eq(reference)]
    p=a.merge(b,on=keys,suffixes=('_candidate','_reference'),validate='one_to_one')
    if len(p)!=len(CONTEXTS)*len(EVALUATION_IDS)*2:raise ValueError('Missing paired endpoints')
    rows=[]
    for metric in ('interval_score','mse','mae','coverage','width'):
        vals=p.assign(delta=p[metric+'_candidate']-p[metric+'_reference']).groupby(['scenario','replication']).delta.mean()
        delta=float(vals.groupby(level=0).mean().mean())
        variance=sum(float(vals.loc[c].var(ddof=1)/len(vals.loc[c])) for c in CONTEXTS)/len(CONTEXTS)**2
        se=float(np.sqrt(variance));pv=float(2*norm.sf(abs(delta/se))) if se>0 else float(delta==0)
        rows.append({'candidate':candidate,'reference':reference,'metric':metric,'paired_worlds':len(vals),'difference':delta,'mcse':se,'MC_CI95_low':delta-1.959963984540054*se,'MC_CI95_high':delta+1.959963984540054*se,'p_two_sided':pv})
    M1=float(p.mse_candidate.mean());M0=float(p.mse_reference.mean())
    delta_log=p.assign(influence=.5*(p.mse_candidate/M1-p.mse_reference/M0)).groupby(['scenario','replication']).influence.mean()
    se=np.sqrt(sum(float(delta_log.loc[c].var(ddof=1)/len(delta_log.loc[c])) for c in CONTEXTS)/len(CONTEXTS)**2)
    logratio=.5*np.log(M1/M0)
    rows.append({'candidate':candidate,'reference':reference,'metric':'RMSE_ratio','paired_worlds':len(delta_log),'difference':float(np.exp(logratio)),'mcse':float(se),'MC_CI95_low':float(np.exp(logratio-1.959963984540054*se)),'MC_CI95_high':float(np.exp(logratio+1.959963984540054*se)),'p_two_sided':float(2*norm.sf(abs(logratio/se))) if se>0 else float(logratio==0), 'mcse_log_ratio':float(se)})
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--lock-only',action='store_true');a=p.parse_args()
    lock=prediction_lock()
    if a.lock_only:print('All fresh predictions locked before truth access.');return
    rows=[];physical_rows=[]
    for context in CONTEXTS:
        for rep in EVALUATION_IDS:
            source=ROOT/'evaluacion/inputs'/context/f'rep_{rep:05d}';fit=ROOT/'evaluacion/fits'/context/f'rep_{rep:05d}'
            manifest=json.loads((source/'manifest.json').read_text())
            if sha(source/'evaluation_only/truth.npz')!=manifest['truth_sha256']:raise RuntimeError('Changed generation truth')
            truth=dict(np.load(source/'evaluation_only/truth.npz'));direct=dict(np.load(source/'model_inputs/inputs.npz',allow_pickle=False));completion=json.loads((fit/'completion.json').read_text())
            for method in METHODS:
                pred=dict(np.load(fit/(method+'.npz')))
                result=scores(truth,direct,pred,method,context,rep,[completion['convergence'][method]]*2,completion['seconds'])
                for row in result:row.update(phase='fresh_evaluation',protocol='same14_Sikov_parametricbootstrap100_ICC_NUTS_frozen')
                rows.extend(result)
                projected={**pred,**{k:np.clip(pred[k],0.,1.) for k in ('mean','lower','upper')}}
                sensitivity=scores(truth,direct,projected,method,context,rep,[completion['convergence'][method]]*2,completion['seconds'])
                for row in sensitivity:row.update(phase='fresh_evaluation_shared_physical_projection',protocol='all_methods_shared_projection_0_1_secondary_not_selected')
                physical_rows.extend(sensitivity)
    frame=pd.DataFrame(rows);out=ROOT/'resultados';out.mkdir(exist_ok=True)
    frame.to_csv(out/'fresh_replications.csv',index=False);summarize(frame).to_csv(out/'fresh_summary.csv',index=False)
    contrasts=[]
    for ref in METHODS:
        if ref=='Selected':continue
        contrasts.extend(balanced(frame,'Selected',ref))
    table=pd.DataFrame(contrasts)
    primary=table.metric.eq('interval_score')&table.reference.isin(['Sikov_FH','Sikov_SAR'])
    ps=table.loc[primary,'p_two_sided'].to_numpy();order=np.argsort(ps);adjusted=np.empty(2);adjusted[order]=np.minimum(1,np.maximum.accumulate(ps[order]*np.array([2,1])))
    table.loc[primary,'p_Holm2']=adjusted
    table.to_csv(out/'fresh_balanced_paired_contrasts.csv',index=False)
    physical=pd.DataFrame(physical_rows)
    physical.to_csv(out/'fresh_physical_projection_replications.csv',index=False)
    summarize(physical).to_csv(out/'fresh_physical_projection_summary.csv',index=False)
    pd.DataFrame([r for ref in ('Sikov_FH','Sikov_SAR','ICC') for r in balanced(physical,'Selected',ref)]).to_csv(out/'fresh_physical_projection_contrasts.csv',index=False)
    detailed=[]
    for ref in ('Sikov_FH','Sikov_SAR','ICC'):
        detailed.append(paired_contrasts(frame[frame.method.eq('Selected')],frame[frame.method.eq(ref)],'Selected_minus_'+ref))
    pd.concat(detailed).to_csv(out/'fresh_context_endpoint_paired_contrasts.csv',index=False)
    body={'worlds':lock['worlds'],'protocol_sha256':lock['protocol_sha256'],'prediction_lock_sha256':sha(ROOT/'validacion/fresh_prediction_lock.json'),'truth_access_only_after_complete_lock':True,'methods':METHODS,'all_method_world_pairs_represented':True,'all_models_converged':bool(frame.converged.all()),'primary_family':'two balanced IS95 comparisons, Holm2, entire256worlds, full120area universe','claims_scope':'Synthetic finite population comparison with harmonized faithfully reconstructed2023FH/SAR, not exact original numeric replication','source_sha256':sha(__file__),'outputs_sha256':{f.name:sha(f)for f in out.glob('fresh*.csv')},'primary_results':table.loc[primary].to_dict('records')}
    save(ROOT/'validacion/fresh_evaluation_status.json',body)
    print(json.dumps(body,indent=2))
if __name__=='__main__':main()
