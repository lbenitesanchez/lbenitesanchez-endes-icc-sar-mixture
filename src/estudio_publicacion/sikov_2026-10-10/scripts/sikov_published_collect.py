"""Additional reference prediction seal and separately expanded Holm3 family.

Original fits, candidate weights and Holm2 outputs are preserved. Scoring is
blocked until the joint seal covers all original/additional fresh and real fits.
"""
from pathlib import Path
import argparse,datetime as dt,json
import numpy as np
import pandas as pd
from sikov_published_fit import ROOT,ADDENDUM,CONTEXTS,IDS,gate,sha,save
from sikov_published_reference import METHOD,BC_METHOD
from comparison_collect import balanced
from improve_development import scores
from benchmark_metrics import summarize

def prediction_lock():
    add,addsha=gate();items=[]
    for context in CONTEXTS:
        for rep in IDS:
            folder=ROOT/'evaluacion/published_sar'/context/f'rep_{rep:05d}';completion=folder/'completion.json'
            if not completion.is_file():raise RuntimeError('Incomplete additional cohort '+str(completion))
            proof=json.loads(completion.read_text());old=ROOT/'evaluacion/fits'/context/f'rep_{rep:05d}/completion.json';source=ROOT/'evaluacion/inputs'/context/f'rep_{rep:05d}'
            if proof['protocol_sha256']!=add['base_protocol_sha256'] or proof['addendum_sha256']!=addsha or proof['truth_read_by_fits'] or proof['original_completion_sha256']!=sha(old):raise RuntimeError('Wrong additional seal role')
            if sha(source/'model_inputs/inputs.npz')!=proof['fit_inputs_sha256'] or sha(source/'manifest.json')!=proof['input_manifest_sha256']:raise RuntimeError('Changed additional training input')
            if set(proof['convergence'])!={METHOD,BC_METHOD}:raise RuntimeError('Additional methods changed')
            for filename,digest in proof['output_sha256'].items():
                if sha(folder/filename)!=digest:raise RuntimeError('Changed additional closed output '+filename)
            items.append({'context':context,'replication':rep,'completion_sha256':sha(completion),'all_converged':all(proof['convergence'].values()),'outputs':proof['output_sha256']})
    result={'protocol_sha256':add['base_protocol_sha256'],'addendum_sha256':addsha,'locked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'approved_for_scoring':False,'requires_complete_joint_prediction_lock':True,'worlds':len(items),'all_expected_worlds':256,'prediction_items':items,'fresh_truth_read_before_lock':False,'convergence_failures_retained':sum(not row['all_converged'] for row in items)}
    out=ROOT/'validacion/sikov_published_fresh_prediction_lock.json'
    if out.exists():
        previous=json.loads(out.read_text())
        if previous['protocol_sha256']!=result['protocol_sha256'] or previous['addendum_sha256']!=addsha or previous['prediction_items']!=items:raise RuntimeError('Existing additional prediction lock differs')
        return previous
    save(out,result);return result

def joint_gate():
    add,addsha=gate();path=ROOT/'validacion/joint_prediction_lock.json';joint=json.loads(path.read_text())
    if not joint.get('approved_for_scoring') or joint.get('protocol_sha256')!=add['base_protocol_sha256'] or joint.get('addendum_sha256')!=addsha or joint.get('heldout_outcomes_used',False):raise RuntimeError('Complete joint prediction lock required before scoring')
    for item in joint['files']:
        if sha(item['path'])!=item['sha256']:raise RuntimeError('Changed jointly sealed file '+item['path'])
    original=ROOT/'validacion/fresh_prediction_lock.json';published=ROOT/'validacion/sikov_published_fresh_prediction_lock.json'
    if joint.get('fresh_original_prediction_lock_sha256')!=sha(original) or joint.get('fresh_published_prediction_lock_sha256')!=sha(published):raise RuntimeError('Both complete fresh locks must be jointly bound')
    if joint.get('fresh_cohort_size')!=256 or joint.get('all_expected_worlds') is not True or joint.get('fresh_truth_read_before_joint') is not False or joint.get('heldout2024responses_used_before_joint') is not False:raise RuntimeError('Joint provenance flags incomplete')
    if json.loads(original.read_text())['worlds']!=256 or json.loads(published.read_text())['worlds']!=256:raise RuntimeError('Joint fresh cohort incomplete')
    return joint,sha(path)

def expanded_contrasts(frame):
    refs=('Sikov_FH','Sikov_SAR',METHOD)
    table=pd.DataFrame([row for reference in refs for row in balanced(frame,'Selected',reference)])
    primary=table.metric.eq('interval_score');p=table.loc[primary,'p_two_sided'].to_numpy();order=np.argsort(p);adjusted=np.empty(3);adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.array([3,2,1])));table.loc[primary,'p_Holm3']=adjusted
    return table

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lock-only',action='store_true');args=parser.parse_args();lock=prediction_lock()
    if args.lock_only:print('All additional fresh predictions sealed; joint lock still required before scoring.');return
    joint,jointsha=joint_gate();out=ROOT/'resultados/published_sar_comparison'
    if out.exists():raise FileExistsError('Preserve completed additional scoring outputs '+str(out))
    base=ROOT/'resultados/fresh_replications.csv';physicalbase=ROOT/'resultados/fresh_physical_projection_replications.csv';baseproof=json.loads((ROOT/'validacion/fresh_evaluation_status.json').read_text())
    if baseproof['protocol_sha256']!=joint['protocol_sha256'] or baseproof['outputs_sha256'][base.name]!=sha(base) or baseproof['outputs_sha256'][physicalbase.name]!=sha(physicalbase):raise RuntimeError('Original scoring seal differs')
    rows=[];physicalrows=[]
    for context in CONTEXTS:
        for rep in IDS:
            source=ROOT/'evaluacion/inputs'/context/f'rep_{rep:05d}';folder=ROOT/'evaluacion/published_sar'/context/f'rep_{rep:05d}';manifest=json.loads((source/'manifest.json').read_text());proof=json.loads((folder/'completion.json').read_text())
            truthpath=source/'evaluation_only/truth.npz'
            if sha(truthpath)!=manifest['truth_sha256']:raise RuntimeError('Changed finite-population truth')
            truth=dict(np.load(truthpath,allow_pickle=False));direct=dict(np.load(source/'model_inputs/inputs.npz',allow_pickle=False))
            for method in (METHOD,BC_METHOD):
                prediction=dict(np.load(folder/(method+'.npz'),allow_pickle=False))
                values=scores(truth,direct,prediction,method,context,rep,[proof['convergence'][method]]*2,proof['seconds'])
                for row in values:row.update(phase='fresh_evaluation_addendum',protocol='sampled_SAR2023_family_directed2024_extension_parametricbootstrap100')
                rows.extend(values)
                projected={**prediction,**{key:np.clip(prediction[key],0,1) for key in ('mean','lower','upper')}}
                values=scores(truth,direct,projected,method,context,rep,[proof['convergence'][method]]*2,proof['seconds'])
                for row in values:row.update(phase='fresh_evaluation_addendum_shared_physical_projection',protocol='all_methods_shared_projection_0_1_secondary_not_selected')
                physicalrows.extend(values)
    added=pd.DataFrame(rows);frame=pd.concat([pd.read_csv(base),added],ignore_index=True);physical=pd.concat([pd.read_csv(physicalbase),pd.DataFrame(physicalrows)],ignore_index=True)
    if len(frame)!=256*11*2*3 or frame[['scenario','replication','outcome','group','method']].duplicated().any():raise RuntimeError('Incomplete expanded comparison universe')
    refs=('Sikov_FH','Sikov_SAR',METHOD);table=expanded_contrasts(frame);primary=table.metric.eq('interval_score')
    out.mkdir();frame.to_csv(out/'per_replication_scores.csv',index=False);summarize(frame).to_csv(out/'summary.csv',index=False);table.to_csv(out/'paired_balanced_contrasts_Holm3.csv',index=False)
    physical.to_csv(out/'physical_projection_scores.csv',index=False);summarize(physical).to_csv(out/'physical_projection_summary.csv',index=False);pd.DataFrame([row for reference in refs for row in balanced(physical,'Selected',reference)]).to_csv(out/'physical_projection_contrasts.csv',index=False)
    body={'status':'PASS_COMPLETE_PRESPECIFIED_ADDITIVE_HOLM3_FAMILY','protocol_sha256':joint['protocol_sha256'],'addendum_sha256':joint['addendum_sha256'],'joint_prediction_lock_sha256':jointsha,'original_Holm2_preserved':True,'selected_recipe_changed':False,'all256_worlds_included':True,'additional_reference_frozen_before_truth_access':True,'all_models_converged':bool(frame.converged.all()),'expanded_primary_family':'3 balanced IS95 comparisons Selected versus FH/full-graph SAR/sampled-only SAR with declared directed non-sampled extension; Holm3','new_reference_exact_2023_neighbor_rule_known':False,'new_reference_non_sampled_extension_source':'2024 author preprint section2 option3','source_sha256':sha(__file__),'primary_results':table.loc[primary].to_dict('records'),'outputs_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'original_scored_output_sha256':{str(base):sha(base),str(physicalbase):sha(physicalbase)}}
    save(ROOT/'validacion/sikov_published_evaluation_status.json',body);print(json.dumps(body,indent=2))

if __name__=='__main__':main()
