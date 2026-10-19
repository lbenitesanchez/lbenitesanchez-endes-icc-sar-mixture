"""Independent additional-reference audit; only joint-sealed scores may be read."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
METHOD='Sikov_SAR_Published';BC=METHOD+'_BC';ENDPOINTS=('anemia','stunting')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def joint_gate():
    path=ROOT/'validacion/joint_prediction_lock.json';joint=json.loads(path.read_text());base=ROOT/'protocol_freeze.json';addpath=ROOT/'validacion/sikov_published_addendum_freeze.json';add=json.loads(addpath.read_text())
    if not joint.get('approved_for_scoring') or joint.get('fresh_cohort_size')!=256 or joint.get('all_expected_worlds') is not True or joint.get('fresh_truth_read_before_joint') is not False or joint.get('heldout2024responses_used_before_joint') is not False:
        raise RuntimeError('Complete provenance-safe joint lock required')
    if joint['protocol_sha256']!=sha(base) or joint['addendum_sha256']!=sha(addpath) or add['protocol_sha256']!=sha(base):raise RuntimeError('Joint protocol/addendum mismatch')
    for item in joint['files']:
        if sha(item['path'])!=item['sha256']:raise RuntimeError('Changed jointly sealed file')
    for block in ('statistical_sources_sha256','frozen_inputs_sha256','runtime_binary_sha256'):
        for path,digest in add.get(block,{}).items():
            if sha(path)!=digest:raise RuntimeError('Changed additional frozen file '+path)
    for label,filename in (('fresh_original_prediction_lock_sha256','fresh_prediction_lock.json'),('fresh_published_prediction_lock_sha256','sikov_published_fresh_prediction_lock.json')):
        if joint[label]!=sha(ROOT/'validacion'/filename):raise RuntimeError('Both complete fresh locks must be bound')
    return joint,add

def independent_triangular(d):
    from sikov_final_audit import graph
    matrices=[]
    for j in range(2):
        mask=np.isfinite(d['y'][:,j])&np.isfinite(d['D'][:,j,j])&(d['D'][:,j,j]>0);s=d['sample_idx'][mask]
        W=graph(d['coords'],d['strata'],6);WD=graph(d['coords'][s],d['strata'][s],6)
        W[s,:]=0;W[np.ix_(s,s)]=WD;matrices.append(W)
    return matrices

def audit_predictions(protocol):
    from sikov_final_audit import dense_gaussian_check
    joint,add=joint_gate();lockpath=ROOT/'validacion/sikov_published_fresh_prediction_lock.json';lock=json.loads(lockpath.read_text());expected={(c,r) for c in protocol['contexts'] for r in protocol['evaluation_ids']}
    if lock['worlds']!=256 or lock['fresh_truth_read_before_lock'] or lock['addendum_sha256']!=joint['addendum_sha256'] or {(i['context'],i['replication']) for i in lock['prediction_items']}!=expected:raise RuntimeError('Additional fresh prediction seal incomplete')
    jobs=[(item['context'],item['replication'],item) for item in lock['prediction_items']]+[(scope,None,None) for scope in ('temporal','territorial')]
    maxima={'mean':0.,'plugin':0.,'nll':0.,'directed_formula':0.,'sampled_covariance_relative':0.};boot_failed=boot_recovered=boot_requested=negative=boundaries=0;failures=[]
    for context,rep,item in jobs:
        if rep is None:
            folder=ROOT/'evaluacion/published_sar'/('real_'+context);old=ROOT/'evaluacion'/('real_'+context)/'completion.json';prepared=ROOT/'datos'/('real_2019_full' if context=='temporal' else 'real_2024_training');model=prepared/'fit_inputs.npz';manifest=prepared/'preparation_manifest.json';d=dict(np.load(model,allow_pickle=False));d['coords']=d['coords'][:,::-1]
        else:
            folder=ROOT/'evaluacion/published_sar'/context/f'rep_{rep:05d}';old=ROOT/'evaluacion/fits'/context/f'rep_{rep:05d}/completion.json';prepared=ROOT/'evaluacion/inputs'/context/f'rep_{rep:05d}';model=prepared/'model_inputs/inputs.npz';manifest=prepared/'manifest.json';d=dict(np.load(model,allow_pickle=False))
        completion=folder/'completion.json';proof=json.loads(completion.read_text())
        if proof['protocol_sha256']!=joint['protocol_sha256'] or proof['addendum_sha256']!=joint['addendum_sha256'] or proof['original_completion_sha256']!=sha(old) or proof['fit_inputs_sha256']!=sha(model):raise RuntimeError('Additional prediction provenance differs')
        if proof.get('truth_read_by_fits',False) or proof.get('heldout_outcomes_read_by_fits',False):raise RuntimeError('Additional fit read scoring outcomes')
        if rep is not None and (sha(completion)!=item['completion_sha256'] or proof['input_manifest_sha256']!=sha(manifest)):raise RuntimeError('Additional fresh completion/input changed')
        if rep is None and (proof['training_manifest_sha256']!=sha(manifest) or proof['original_predictions_sha256']!=sha(old.parent/'predictions.csv')):raise RuntimeError('Additional actual-data seal changed')
        for filename,digest in proof['output_sha256'].items():
            if sha(folder/filename)!=digest:raise RuntimeError('Additional output changed '+filename)
        p=dict(np.load(folder/(METHOD+'.npz'),allow_pickle=False));bc=dict(np.load(folder/(BC+'.npz'),allow_pickle=False));status=json.loads((folder/(METHOD+'_status.json')).read_text())
        if not np.array_equal(p['mean'],bc['mean']) or not np.array_equal(p['mse'],p['mse_naive']) or not np.array_equal(bc['mse'],p['mse_bc']):raise RuntimeError('Additional native/BC draw convention differs')
        if not np.isfinite(np.stack([p[k] for k in ('mean','lower','upper')])).all() or p['mean'].shape!=(len(d['X_raw']),2):raise RuntimeError('Additional area universe differs')
        if not np.allclose(bc['lower'],bc['mean']-1.959963984540054*np.sqrt(np.maximum(bc['mse_bc'],0)),atol=1e-12,rtol=0):raise RuntimeError('Additional BC sqrt convention differs')
        W=independent_triangular(d);check=dense_gaussian_check(d,status,'Sikov_SAR',p,W_override=W)
        for key in ('mean','plugin','nll'):maxima[key]=max(maxima[key],check['maximum_errors'][key])
        boot_failed+=check['bootstrap_failed'];boot_recovered+=check['bootstrap_recovered'];boot_requested+=check['bootstrap_requested'];boundaries+=check['rho_boundaries'];negative+=int((bc['mse_bc']<0).sum())
        if not status['converged']:failures.append([context,rep])
        for j,fit in enumerate(status['fits']):
            mask=np.isfinite(d['y'][:,j])&np.isfinite(d['D'][:,j,j])&(d['D'][:,j,j]>0);sampled=d['sample_idx'][mask]
            for sf in fit['fit']:
                ids=np.flatnonzero(d['strata']==sf['stratum']);s=np.searchsorted(ids,sampled[np.isin(sampled,ids)]);u=np.setdiff1d(np.arange(len(ids)),s);WW=W[j][np.ix_(ids,ids)];rho=sf['rho'];cols=[d['covariate_names'].tolist().index(c) for c in sf['covariates']];X=d['X_raw'][ids][:,cols];beta=np.asarray(sf['beta'])
                if np.any(WW[np.ix_(s,u)]):raise RuntimeError('Sampled-to-non-sampled links in published comparator')
                inv=np.linalg.inv(np.eye(len(ids))-rho*WW);C=inv@inv.T;invD=np.linalg.inv(np.eye(len(s))-rho*WW[np.ix_(s,s)]);CD=invD@invD.T
                relative=float(abs(C[np.ix_(s,s)]-CD).max())/max(float(abs(CD).max()),1.);maxima['sampled_covariance_relative']=max(maxima['sampled_covariance_relative'],relative)
                F=np.linalg.solve(np.eye(len(u))-rho*WW[np.ix_(u,u)],rho*WW[np.ix_(u,s)]);uhat=p['mean'][ids[s],j]-X[s]@beta;predicted=X[u]@beta+F@uhat
                if len(u):maxima['directed_formula']=max(maxima['directed_formula'],float(abs(predicted-p['mean'][ids[u],j]).max()))
        if rep is None:
            exported=pd.read_csv(folder/'predictions.csv',dtype={'ubigeo':str});codes=pd.read_csv(prepared/'area_frame.csv',dtype={'ubigeo':str}).ubigeo.tolist()
            if len(exported)!=2*len(codes) or exported[['ubigeo','method']].duplicated().any():raise RuntimeError('Additional real exported support differs')
            for method,arrays in ((METHOD,p),(BC,bc)):
                frame=exported[exported.method==method].set_index('ubigeo').loc[codes]
                for j,endpoint in enumerate(ENDPOINTS):
                    for prefix,key in (('p_','mean'),('lo95_','lower'),('hi95_','upper')):
                        if not np.allclose(frame[prefix+endpoint],arrays[key][:,j],atol=1e-12,rtol=0):raise RuntimeError('Additional real CSV changed')
    if maxima['directed_formula']>1e-8 or maxima['sampled_covariance_relative']>1e-11:raise RuntimeError('Published sampled graph/directed extension algebra failed')
    return {'additional_published_prediction_checks':{'fresh_worlds':256,'actual_scopes':2,'max_errors':maxima,'bootstrap_requested':boot_requested,'bootstrap_failed':boot_failed,'bootstrap_stationarity_recovered':boot_recovered,'rho_boundary_fits':boundaries,'negative_BC_MSE_cells':negative,'convergence_failures_retained':failures,'exact_2023_neighbor_rule_or_non_sampled_implementation_claimed':False,'joint_prediction_lock_sha256':sha(ROOT/'validacion/joint_prediction_lock.json'),'truth_values_read':False}}

def audit_scores(protocol):
    from sikov_final_audit import independent_balanced
    joint,add=joint_gate();out=ROOT/'resultados/published_sar_comparison';proof=json.loads((ROOT/'validacion/sikov_published_evaluation_status.json').read_text())
    if proof['status']!='PASS_COMPLETE_PRESPECIFIED_ADDITIVE_HOLM3_FAMILY' or proof['joint_prediction_lock_sha256']!=sha(ROOT/'validacion/joint_prediction_lock.json'):raise RuntimeError('Additional scoring seal differs')
    for filename,digest in proof['outputs_sha256'].items():
        if sha(out/filename)!=digest:raise RuntimeError('Additional scores changed')
    combined=pd.read_csv(out/'per_replication_scores.csv');physical=pd.read_csv(out/'physical_projection_scores.csv');base=pd.read_csv(ROOT/'resultados/fresh_replications.csv');keys=['scenario','replication','method','outcome','group'];metrics=('mse','mae','bias','interval_score','coverage','width')
    if len(combined)!=256*11*2*3 or combined[keys].duplicated().any():raise RuntimeError('Expanded score universe differs')
    old=combined[~combined.method.isin((METHOD,BC))].merge(base,on=keys,validate='one_to_one',suffixes=('_new','_old'))
    if len(old)!=len(base) or max(abs(old[m+'_new']-old[m+'_old']).max() for m in metrics)>1e-12:raise RuntimeError('Original scores changed in expanded family')
    rows=[];bounded=[]
    for context in protocol['contexts']:
        for rep in protocol['evaluation_ids']:
            source=ROOT/'evaluacion/inputs'/context/f'rep_{rep:05d}';manifest=json.loads((source/'manifest.json').read_text());truth=source/'evaluation_only/truth.npz'
            if sha(truth)!=manifest['truth_sha256']:raise RuntimeError('Additional score truth changed')
            theta=np.load(truth,allow_pickle=False)['theta'];d=dict(np.load(source/'model_inputs/inputs.npz',allow_pickle=False));obs=np.zeros(theta.shape,bool);obs[d['sample_idx']]=np.isfinite(d['y']);unsampled=~np.isin(np.arange(len(theta)),d['sample_idx'])
            for method in (METHOD,BC):
                p=dict(np.load(ROOT/'evaluacion/published_sar'/context/f'rep_{rep:05d}'/(method+'.npz'),allow_pickle=False))
                for j,endpoint in enumerate(ENDPOINTS):
                    for group,mask in (('all',np.ones(len(theta),bool)),('observed_endpoint',obs[:,j]),('unsampled_common',unsampled)):
                        target=theta[mask,j]
                        for bounded_flag,destination in ((False,rows),(True,bounded)):
                            mean,lo,hi=(p[k][mask,j] for k in ('mean','lower','upper'))
                            if bounded_flag:mean,lo,hi=(np.clip(v,0,1) for v in (mean,lo,hi))
                            error=mean-target;IS=hi-lo+40*np.maximum(lo-target,0)+40*np.maximum(target-hi,0)
                            destination.append(dict(scenario=context,replication=rep,method=method,outcome=endpoint,group=group,mse=np.mean(error**2),mae=np.mean(abs(error)),bias=np.mean(error),interval_score=np.mean(IS),coverage=np.mean((lo<=target)&(target<=hi)),width=np.mean(hi-lo)))
    maxerror=0.
    for own,reported in ((pd.DataFrame(rows),combined),(pd.DataFrame(bounded),physical)):
        matched=own.merge(reported[reported.method.isin((METHOD,BC))],on=keys,validate='one_to_one',suffixes=('_ind','_rep'))
        if len(matched)!=len(own):raise RuntimeError('Additional scoring domains omitted')
        maxerror=max(maxerror,max(float(abs(matched[m+'_ind']-matched[m+'_rep']).max()) for m in metrics))
    if maxerror>1e-12:raise RuntimeError('Independent additional finite-population scoring differs')
    reported=pd.read_csv(out/'paired_balanced_contrasts_Holm3.csv');primary=[];contrast_error=0.
    for reference in ('Sikov_FH','Sikov_SAR',METHOD):
        values=independent_balanced(combined,'Selected',reference,protocol['contexts'],protocol['evaluation_ids'])
        for row in values:
            actual=reported[(reported.reference==reference)&(reported.metric==row['metric'])]
            if len(actual)!=1:raise RuntimeError('Additional paired family incomplete')
            contrast_error=max(contrast_error,max(abs(float(actual.iloc[0][key])-row[key]) for key in ('difference','mcse','MC_CI95_low','MC_CI95_high','p_two_sided')))
            if row['metric']=='interval_score':primary.append({'reference':reference,**row})
    if contrast_error>1e-10:raise RuntimeError('Independent expanded paired MC covariance differs')
    p=np.array([row['p_two_sided'] for row in primary]);adjusted=[]
    for value in p:adjusted.append(min(1,max((3-int((p<t).sum()))*t for t in p if t<=value)))
    for row,adjustment in zip(primary,adjusted):
        row['p_Holm3']=adjustment;actual=reported[(reported.reference==row['reference'])&(reported.metric=='interval_score')].iloc[0]
        if abs(actual.p_Holm3-adjustment)>1e-12:raise RuntimeError('Expanded Holm3 differs')
    return {'additional_published_score_checks':{'status':'PASS_INDEPENDENT_ADDITIVE_HOLM3','score_rows_recomputed':len(rows),'score_max_absolute_error':maxerror,'paired_contrast_max_absolute_error':contrast_error,'original_Holm2_preserved':True,'BC_sensitivity_included':True,'selected_weights_changed':False,'expanded_primary_results':primary}}
