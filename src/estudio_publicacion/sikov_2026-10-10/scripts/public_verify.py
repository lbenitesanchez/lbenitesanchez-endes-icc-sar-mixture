"""Verify released files and recompute all closed synthetic comparison scores.

No truth file is hashed or loaded before all 256 prediction sets, completion
records and model-input identities have been verified. Never fits or selects.
"""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,tempfile
import numpy as np
import pandas as pd
from public_package import public_logical_path

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT.parent
PACKAGE=STUDY.parent
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ADDED_METHODS=('Sikov_SAR_Published','Sikov_SAR_Published_BC')
REFERENCES=('Sikov_FH','Sikov_SAR','Sikov_SAR_Published')
ADDENDUM_SHA='7822176f48357b1ee9bf0a926fa44b74bafd3fc9c481a1ece878a8d5bd3b4427'
POVERTY_PLAN_SHA='d28b814872726d6d0162c9cf1b717a0c6fa7c86c057a7b7c1a70d0c3b237b841'
POVERTY_OUTPUTS=('per_population_stratum_scores.csv','stratum_descriptive_summary.csv',
                 'selected_reference_descriptive_differences.csv')
SAMPLED_GRAPH_PLAN_SHA='0435d8b2167ffe1c275310027c6b5c81f2c462b8deca856eec7210bb57ff057e'
SAMPLED_GRAPH_CONTEXT='SAR_probability_sampled_graph_srs'
SAMPLED_GRAPH_OUTPUTS=('descriptive_replications.csv','descriptive_summary.csv','descriptive_paired_contrasts.csv',
    'physical_projection_replications.csv','physical_projection_summary.csv','physical_projection_paired_contrasts.csv')
PUBLISHED_B300_PLAN_SHA='d1c84f664bad288a4f5346891f1ec0e9321e810917ed62172ad898818dc0fc0a'
PUBLISHED_B300_METHODS=('Selected_B100','Published_B100','Published_B100_BC','Published_B300','Published_B300_BC')
PUBLISHED_B300_OUTPUTS=('per_population_scores.csv','summary.csv','selected_reference_differences.csv')

def portable_path(oldpath,proof):
    """Map retained original-host identities without using original-host files."""
    relative=Path(oldpath).relative_to(Path(proof['path_anchor']))
    if '..'in relative.parts:raise RuntimeError('Escaping retained source path')
    return public_logical_path(STUDY/relative,STUDY)

def frozen_metadata():
    proof=json.loads((ROOT/'protocol_freeze.json').read_text())
    addpath=ROOT/'validacion/sikov_published_addendum_freeze.json'
    add=json.loads(addpath.read_text());freeze=SHA(ROOT/'protocol_freeze.json');addsha=SHA(addpath)
    if (addsha!=ADDENDUM_SHA or add['protocol_sha256']!=freeze
            or add['base_protocol_sha256']!=freeze
            or not add['approved_for_additional_reference_fits']
            or add['accuracy_or_heldout_outcomes_read_before_addendum']
            or add['selection_performed_again'] or not add['base_protocol_and_selected_recipe_unchanged']
            or add['selected_recipe']!=proof['selected_recipe']
            or add['new_methods']!=list(ADDED_METHODS)
            or add['contexts']!=proof['contexts'] or add['evaluation_ids']!=proof['evaluation_ids']
            or add['fresh_worlds']!=proof['fresh_worlds'] or proof['fresh_worlds']!=256):
        raise RuntimeError('Changed prespecified addendum, cohort or selection')
    for document in (proof,add):
        for field in ('statistical_sources_sha256','additional_sources_sha256','base_kernel_sources_sha256'):
            for name,digest in document.get(field,{}).items():
                if SHA(portable_path(name,proof))!=digest:raise RuntimeError('Frozen portable source changed '+name)
    # All addendum inputs are aggregate source/contract certificates, no records.
    for name,digest in add['frozen_inputs_sha256'].items():
        if SHA(portable_path(name,proof))!=digest:raise RuntimeError('Frozen addendum certificate changed '+name)
    return proof,add,freeze,addsha

def safe_member(name):
    rel=PurePosixPath(name)
    if rel.is_absolute()or '..'in rel.parts or not rel.parts:
        raise RuntimeError('Unsafe released path '+name)
    return public_logical_path(PACKAGE/Path(*rel.parts),PACKAGE)

def is_truth(path):
    return 'evaluation_only'in path.parts and path.name=='truth.npz'

def compare(actual,expected,keys):
    if actual.duplicated(keys).any()or expected.duplicated(keys).any():
        raise ValueError('Duplicate score cell')
    a=actual.sort_values(keys).reset_index(drop=True)
    b=expected.sort_values(keys).reset_index(drop=True)
    if set(a.columns)!=set(b.columns)or len(a)!=len(b)or not a[keys].equals(b[keys]):
        raise ValueError('Unequal complete score schema or cohort')
    errors=[]
    numeric=set(b.select_dtypes(include=['number','bool']).columns)
    for col in b.columns:
        if col in numeric:
            av=a[col].to_numpy(float);bv=b[col].to_numpy(float)
            if (not np.array_equal(np.isnan(av),np.isnan(bv))
                    or not np.array_equal(np.isposinf(av),np.isposinf(bv))
                    or not np.array_equal(np.isneginf(av),np.isneginf(bv))):
                raise ValueError('Different missingness or infinities '+col)
            finite=np.isfinite(bv)
            if not np.allclose(av[finite],bv[finite],atol=1e-10,rtol=1e-9):
                raise ValueError('Different public metric '+col)
            if finite.any():errors.append(float(np.max(np.abs(av[finite]-bv[finite]))))
        else:
            if not a[col].fillna('<missing>').astype(str).equals(b[col].fillna('<missing>').astype(str)):
                raise ValueError('Different public label '+col)
    return max(errors,default=0.)

def verify_closed_predictions(lock,freeze,CONTEXTS,EVALUATION_IDS,METHODS):
    expected={(c,r)for c in CONTEXTS for r in EVALUATION_IDS}
    actual={(i['context'],i['replication'])for i in lock['prediction_items']}
    if (len(expected)!=256 or actual!=expected or len(lock['prediction_items'])!=256
            or lock.get('all_expected_worlds')!=256 or not lock.get('approved_for_scoring')
            or lock.get('fresh_truth_read_before_lock',True)
            or lock.get('convergence_failures_retained')!=0):
        raise RuntimeError('Partial, unapproved or failed closed synthetic cohort')
    closed=[]
    for item in lock['prediction_items']:
        context,rep=item['context'],item['replication']
        folder=ROOT/'evaluacion/fits'/context/f'rep_{rep:05d}'
        source=ROOT/'evaluacion/inputs'/context/f'rep_{rep:05d}'
        for path in [folder/'completion.json',source/'manifest.json',source/'model_inputs/inputs.npz',
                     source/'model_inputs/sample.csv']+[folder/name for method in METHODS
                                                       for name in (method+'.npz',method+'_status.json')]:
            public_logical_path(path,STUDY)
        if SHA(folder/'completion.json')!=item['completion_sha256']:
            raise RuntimeError('Changed archived completion')
        completion=json.loads((folder/'completion.json').read_text())
        if (completion['context']!=context or completion['replication']!=rep
                or completion['protocol_sha256']!=freeze or completion['truth_read_by_fits']
                or set(completion['convergence'])!=set(METHODS)
                or not all(completion['convergence'].values())
                or not item['all_converged']):
            raise RuntimeError('Invalid archived method/prediction role')
        for method in METHODS:
            for name in (method+'.npz',method+'_status.json'):
                digest=SHA(folder/name)
                if digest!=item['outputs'][name]or digest!=completion['output_sha256'][name]:
                    raise RuntimeError('Changed archived prediction '+name)
        record=json.loads((source/'manifest.json').read_text())
        if SHA(source/'manifest.json')!=completion['input_manifest_sha256']:
            raise RuntimeError('Closed input manifest changed')
        if record['phase']!='evaluation'or record['context']!=context or record['replication']!=rep:
            raise RuntimeError('Wrong released population input role')
        for file,key in [('model_inputs/inputs.npz','fit_inputs_sha256'),('model_inputs/sample.csv','sample_sha256')]:
            digest=SHA(source/file)
            if digest!=record[key]or digest!=completion[key]:
                raise RuntimeError('Changed closed synthetic model input')
        closed.append((context,rep,source,folder,record,completion))
    return closed

def verify_published_predictions(lock,joint,closed,freeze,addsha):
    expected={(c,r)for c,r,*_ in closed}
    if (len(lock['prediction_items'])!=256 or lock['worlds']!=256
            or {(i['context'],i['replication'])for i in lock['prediction_items']}!=expected
            or lock['protocol_sha256']!=freeze or lock['addendum_sha256']!=addsha
            or lock['fresh_truth_read_before_lock'] or lock['convergence_failures_retained']!=0
            or not lock['requires_complete_joint_prediction_lock']):
        raise RuntimeError('Incomplete or changed published-scope SAR prediction seal')
    bykey={(c,r):(source,original,record,completion)for c,r,source,original,record,completion in closed}
    for item in lock['prediction_items']:
        c,r=item['context'],item['replication'];source,original,_,_=bykey[c,r]
        folder=ROOT/'evaluacion/published_sar'/c/f'rep_{r:05d}'
        for path in [folder/'completion.json']+[folder/name for method in ADDED_METHODS
                                                for name in (method+'.npz',method+'_status.json')]:
            public_logical_path(path,STUDY)
        completion=json.loads((folder/'completion.json').read_text())
        if (SHA(folder/'completion.json')!=item['completion_sha256']
                or completion['protocol_sha256']!=freeze or completion['addendum_sha256']!=addsha
                or completion['original_completion_sha256']!=SHA(original/'completion.json')
                or completion['truth_read_by_fits'] or not item['all_converged']
                or set(completion['convergence'])!=set(ADDED_METHODS)
                or not all(completion['convergence'].values())
                or completion['fit_inputs_sha256']!=SHA(source/'model_inputs/inputs.npz')
                or completion['input_manifest_sha256']!=SHA(source/'manifest.json')):
            raise RuntimeError('Wrong closed published-scope SAR role')
        for method in ADDED_METHODS:
            for name in (method+'.npz',method+'_status.json'):
                digest=SHA(folder/name)
                if digest!=item['outputs'][name]or digest!=completion['output_sha256'][name]:
                    raise RuntimeError('Changed published-scope SAR prediction '+name)

def joint_file_inventory(joint,proof):
    """Validate the exact 16 allowed seal identities before hashing any member."""
    expected={ROOT/'protocol_freeze.json',ROOT/'validacion/sikov_published_addendum_freeze.json',
              ROOT/'validacion/fresh_prediction_lock.json',
              ROOT/'validacion/sikov_published_fresh_prediction_lock.json'}
    for scope in ('temporal','territorial'):
        for prefix in ('','published_sar/'):
            folder=ROOT/'evaluacion'/prefix/('real_'+scope)
            expected.update((folder/'completion.json',folder/'predictions.csv'))
        expected.update(ROOT/'validacion'/name for name in
            ('real_prediction_lock_'+scope+'.json','sikov_published_prediction_lock_'+scope+'.json'))
    expected={p.absolute()for p in expected}
    items=joint['files'];paths=[portable_path(item['path'],proof)for item in items]
    if any(is_truth(path)for path in paths):raise RuntimeError('Truth cannot be in the prediction seal')
    if len(items)!=16 or len(set(paths))!=16 or set(paths)!=expected:
        raise RuntimeError('Joint seal must contain exactly the 16 prescribed public identities')
    return {path:item['sha256']for path,item in zip(paths,items)}

def prediction_preflight():
    """Verify the full joint cohort using metadata/hash reads, never truth bytes."""
    # Read only metadata to reject a malicious extra path before any seal-member
    # SHA call, including the otherwise legitimate protocol and addendum paths.
    proof_header=json.loads((ROOT/'protocol_freeze.json').read_text())
    jointpath=ROOT/'validacion/joint_prediction_lock.json';joint=json.loads(jointpath.read_text())
    bound=joint_file_inventory(joint,proof_header)
    proof,add,freeze,addsha=frozen_metadata()
    if (not joint['approved_for_scoring'] or joint['protocol_sha256']!=freeze
            or joint['addendum_sha256']!=addsha or joint['fresh_cohort_size']!=256
            or joint['all_expected_worlds']is not True or joint['fresh_truth_read_before_joint']
            or joint['heldout2024responses_used_before_joint'] or joint.get('heldout_outcomes_used',False)
            or not joint['selected_recipe_unchanged']
            or any(joint['fresh_convergence_failures_retained'].values())):
        raise RuntimeError('Complete provenance-safe joint seal required')
    for path,digest in bound.items():
        if SHA(path)!=digest:raise RuntimeError('Changed jointly sealed artifact')
    required=[ROOT/'protocol_freeze.json',ROOT/'validacion/sikov_published_addendum_freeze.json']
    for key,name in (('fresh_original_prediction_lock_sha256','fresh_prediction_lock.json'),
                     ('fresh_published_prediction_lock_sha256','sikov_published_fresh_prediction_lock.json')):
        path=ROOT/'validacion'/name;required.append(path)
        if joint[key]!=SHA(path):raise RuntimeError('Joint fresh locks differ')
    if not set(required).issubset(bound):raise RuntimeError('Joint seal omits protocol/fresh lock identities')
    original=json.loads((ROOT/'validacion/fresh_prediction_lock.json').read_text())
    if original['protocol_sha256']!=freeze or original['worlds']!=256:raise RuntimeError('Changed original fresh lock')
    closed=verify_closed_predictions(original,freeze,proof['contexts'],proof['evaluation_ids'],proof['methods'])
    published=json.loads((ROOT/'validacion/sikov_published_fresh_prediction_lock.json').read_text())
    verify_published_predictions(published,joint,closed,freeze,addsha)
    for scope in ('temporal','territorial'):
        oldfolder=ROOT/'evaluacion'/('real_'+scope)
        pubfolder=ROOT/'evaluacion/published_sar'/('real_'+scope)
        for folder,methods,additional in ((oldfolder,proof['methods'],False),(pubfolder,ADDED_METHODS,True)):
            record=json.loads((folder/'completion.json').read_text())
            for path in (folder/'completion.json',folder/'predictions.csv'):
                if path.absolute()not in bound or SHA(path)!=bound[path.absolute()]:
                    raise RuntimeError('Both complete real scopes must be jointly sealed')
            if (record['protocol_sha256']!=freeze or record['heldout_outcomes_read_by_fits']
                    or record['predictions_sha256']!=SHA(folder/'predictions.csv')):
                raise RuntimeError('Wrong actual-data prediction role')
            needed=ADDED_METHODS if additional else ('Selected','Sikov_FH','Sikov_SAR')
            if not all(record['convergence'].get(m,False)for m in needed):raise RuntimeError('Actual primary convergence failure')
            if additional and (record['addendum_sha256']!=addsha
                    or record['original_completion_sha256']!=SHA(oldfolder/'completion.json')
                    or record['original_predictions_sha256']!=SHA(oldfolder/'predictions.csv')):
                raise RuntimeError('Actual published comparator changed original predictions')
            for method in methods:
                for name in (method+'.npz',method+'_status.json'):
                    public_logical_path(folder/name,STUDY)
                    if SHA(folder/name)!=record['output_sha256'][name]:raise RuntimeError('Actual prediction output changed')
        for filename in ('real_prediction_lock_'+scope+'.json','sikov_published_prediction_lock_'+scope+'.json'):
            path=ROOT/'validacion'/filename
            if path.absolute()not in bound:raise RuntimeError('Joint seal omits actual prediction lock')
    return {'proof':proof,'addendum':add,'protocol_sha256':freeze,'addendum_sha256':addsha,
            'joint':joint,'joint_sha256':SHA(jointpath),'closed':closed,
            'methods':list(proof['methods'])+list(ADDED_METHODS)}

def completed_result_gates(meta,external_dir=None,audit_path=None):
    """Require audited result certificates after prediction_preflight succeeds."""
    ext=external_dir or ROOT/'resultados/external_comparison_Holm12'
    audit=json.loads((audit_path or ROOT/'validacion/sikov_final_independent_audit.json').read_text())
    if (audit['status']!='PASS_COMPLETE_COMPARISON_INDEPENDENT_AUDIT'
            or audit['protocol_sha256']!=meta['protocol_sha256']
            or audit['prediction_lock_sha256']!=SHA(ROOT/'validacion/fresh_prediction_lock.json')
            or audit['additional_published_prediction_checks']['joint_prediction_lock_sha256']!=meta['joint_sha256']
            or audit['additional_published_prediction_checks']['fresh_worlds']!=256
            or audit['additional_published_prediction_checks']['actual_scopes']!=2
            or audit['additional_published_prediction_checks']['convergence_failures_retained']
            or audit['additional_published_score_checks']['status']!='PASS_INDEPENDENT_ADDITIVE_HOLM3'
            or audit['additional_published_score_checks']['selected_weights_changed']):
        raise RuntimeError('Complete expanded independent comparison audit required')
    base=json.loads((ROOT/'validacion/fresh_evaluation_status.json').read_text())
    expanded=json.loads((ROOT/'validacion/sikov_published_evaluation_status.json').read_text())
    if (base['worlds']!=256 or base['protocol_sha256']!=meta['protocol_sha256']
            or not base['all_models_converged']
            or not base['all_method_world_pairs_represented']
            or not base['truth_access_only_after_complete_lock']
            or expanded['status']!='PASS_COMPLETE_PRESPECIFIED_ADDITIVE_HOLM3_FAMILY'
            or not expanded['all256_worlds_included'] or not expanded['all_models_converged']
            or expanded['selected_recipe_changed'] or not expanded['original_Holm2_preserved']
            or not expanded['additional_reference_frozen_before_truth_access']
            or expanded['protocol_sha256']!=meta['protocol_sha256']
            or expanded['addendum_sha256']!=meta['addendum_sha256']
            or expanded['joint_prediction_lock_sha256']!=meta['joint_sha256']):
        raise RuntimeError('Expanded comparison status/provenance failed')
    for folder,report in ((ROOT/'resultados',base),(ROOT/'resultados/published_sar_comparison',expanded)):
        for name,digest in report['outputs_sha256'].items():
            if Path(name).name!=name or SHA(folder/name)!=digest:raise RuntimeError('Scored aggregate changed '+name)
    external=json.loads((ext/'external_final_summary.json').read_text())
    if (external['status']!='PASS_COMPLETE_SECONDARY_HOLM12_FAMILY'
            or external['protocol_sha256']!=meta['protocol_sha256']
            or external['addendum_sha256']!=meta['addendum_sha256']
            or not external['original_Holm8_family_reported_separately']
            or external['latent_coverage_claim_permitted']):raise RuntimeError('Complete reused external Holm12 required')
    for name,digest in external['input_sha256'].items():
        if SHA(portable_path(name,meta['proof']))!=digest:raise RuntimeError('External score scope changed')
    original_external=json.loads((ext.with_name('external_comparison')/'external_final_summary.json').read_text())
    if (original_external['status']!='PASS_COMPLETE_SECONDARY_HOLM8_FAMILY'
            or original_external['protocol_sha256']!=meta['protocol_sha256']):
        raise RuntimeError('Original separate external Holm8 family required')
    qa=json.loads((ROOT/'validacion/align_external_extended_independent_audit.json').read_text())
    if (qa['status']!='PASS_INDEPENDENT_EXTENDED_EXTERNAL_NUMERIC_AND_EXCLUSION_AUDIT'
            or qa['checks_passed']<=0 or qa['model_or_statistical_source_modified']
            or qa['individual_test_records_reopened']
            or qa['source_sha256']!=SHA(ROOT/'scripts/align_external_extended_audit.py')):
        raise RuntimeError('Extended independent external audit required')
    for name,digest in qa['native_csv_sha256'].items():
        if SHA(portable_path(name,meta['proof']))!=digest:raise RuntimeError('Audited external CSV changed')
    return {'base_status':base,'expanded_status':expanded,'audit':audit,'external_status':external,'external_QA':qa}

def recompute_external_contrasts(frames):
    """Independent paired province inference from public district aggregates.

    Fixed scope/metric/reference order also preserves the prespecified paired
    bootstrap RNG stream. No rows from the collected family define this oracle.
    """
    from scipy.stats import t
    if set(frames)!={'temporal','territorial'}:raise ValueError('Both external scopes required')
    records=[];endpoints=('anemia','stunting');methods={'Selected',*REFERENCES}
    for scope in ('temporal','territorial'):
        frame=frames[scope].copy()
        frame['ubigeo']=frame.ubigeo.astype(str);frame['province']=frame.province.astype(str)
        if (set(frame.method)!=methods or set(frame.outcome)!=set(endpoints)
                or frame.duplicated(['method','ubigeo','outcome']).any()
                or not frame.province.eq(frame.ubigeo.str[:4]).all()):
            raise ValueError('Wrong external aggregate comparator, domain or province identity')
        identities={method:set(map(tuple,frame.loc[frame.method.eq(method),['ubigeo','outcome']].to_numpy()))
                    for method in methods}
        if any(keys!=identities['Selected']for keys in identities.values()):
            raise ValueError('External strategies require identical endpoint/domain pairs')
        if not frame.direct.between(0,1).all():raise ValueError('Invalid external weighted direct proportion')
        direct=frame.loc[frame.method.eq('Selected'),['ubigeo','outcome','direct']]
        for method in REFERENCES:
            compare(frame.loc[frame.method.eq(method),['ubigeo','outcome','direct']],direct,['ubigeo','outcome'])
        # Recalculate every district metric from predictions and the aggregate
        # target; do not permit adulterated metric columns to define the oracle.
        for endpoint in endpoints:
            mask=frame.outcome.eq(endpoint);p=frame.loc[mask,'p_'+endpoint].to_numpy(float)
            y=frame.loc[mask,'direct'].to_numpy(float)
            if not np.isfinite(p).all():raise ValueError('Non-finite external prediction')
            q=np.clip(p,1e-12,1-1e-12)
            metrics={'MSE':(p-y)**2,'NLL':-(y*np.log(q)+(1-y)*np.log1p(-q))}
            for metric,value in metrics.items():
                if not np.allclose(frame.loc[mask,metric].to_numpy(float),value,atol=1e-12,rtol=1e-9):
                    raise ValueError('External district metric differs from prediction/target '+metric)
                frame.loc[mask,metric]=value
        codes=sorted(frame.ubigeo.unique());provinces=np.asarray([code[:4]for code in codes])
        labels=np.unique(provinces);J=len(labels)
        if J<2:raise ValueError('At least two external province clusters required')
        rng=np.random.default_rng(202610102024+(10000 if scope=='territorial'else 0))
        for metric in ('MSE','NLL'):
            values={method:frame.loc[frame.method.eq(method)].pivot(index='ubigeo',columns='outcome',values=metric)
                    .reindex(index=codes,columns=endpoints).to_numpy(float)for method in methods}
            if any(np.isinf(v).any()for v in values.values()):raise ValueError('Infinite external score')
            for reference in REFERENCES:
                delta=values['Selected']-values[reference]
                sums=np.zeros((J,2));counts=np.zeros((J,2))
                for k,province in enumerate(labels):
                    block=delta[provinces==province]
                    counts[k]=np.isfinite(block).sum(axis=0)
                    sums[k]=np.nansum(block,axis=0)
                totals=counts.sum(axis=0)
                if (totals==0).any():raise ValueError('No scored domains for an external endpoint')
                endpoint_mean=sums.sum(axis=0)/totals
                difference=float(endpoint_mean.mean())
                influence=.5*np.sum((sums-counts*endpoint_mean)/totals,axis=1)
                se=float(np.sqrt(J/(J-1)*np.sum(influence**2)))
                statistic=difference/se if se else(0. if difference==0 else np.copysign(np.inf,difference))
                pvalue=float(2*t.sf(abs(statistic),J-1));critical=float(t.ppf(.975,J-1))
                indices=rng.integers(0,J,size=(3999,J));denominator=counts[indices].sum(axis=1)
                if (denominator==0).any():raise ValueError('Paired bootstrap endpoint denominator vanished')
                boot=(sums[indices].sum(axis=1)/denominator).mean(axis=1)
                low,high=np.quantile(boot,[.025,.975])
                records.append({'scope':scope,'candidate':'Selected','reference':reference,'metric':metric,
                    'difference':difference,'endpoint_difference_anemia':float(endpoint_mean[0]),
                    'endpoint_difference_stunting':float(endpoint_mean[1]),'cluster_standard_error':se,
                    'province_clusters':J,'t_df':J-1,'p_two_sided':pvalue,
                    'unadjusted_t_ci_low':difference-critical*se,'unadjusted_t_ci_high':difference+critical*se,
                    'descriptive_bootstrap_ci_low':float(low),'descriptive_bootstrap_ci_high':float(high),
                    'bootstrap_repetitions':3999,'evaluated_anemia_domains':int(totals[0]),
                    'evaluated_stunting_domains':int(totals[1])})
    return pd.DataFrame(records)

def verify_external_family(family,recomputed):
    """Link all 12 keyed contrasts to the aggregate oracle before Holm."""
    keys=['scope','candidate','reference','metric']
    expected={(scope,'Selected',reference,metric)for scope in ('temporal','territorial')
              for reference in REFERENCES for metric in ('MSE','NLL')}
    for frame in (family,recomputed):
        if (len(frame)!=12 or frame.duplicated(keys).any()
                or set(map(tuple,frame[keys].to_numpy()))!=expected):
            raise ValueError('Incomplete or substituted external 12-contrast family')
    if set(family.columns)!=set(recomputed.columns)|{'p_Holm12','classification'}:
        raise ValueError('Wrong external collected family schema')
    # Difference, endpoint means, SE, df, p and both interval constructions must
    # match by identity; a self-consistent Holm adjustment cannot hide new rows.
    raw_error=compare(family[recomputed.columns],recomputed,keys)
    p=recomputed.p_two_sided.to_numpy(float)
    if not np.isfinite(p).all()or ((p<0)|(p>1)).any():raise ValueError('Invalid external oracle p-values')
    order=np.argsort(p,kind='stable');adjusted=np.empty(12)
    for rank,position in enumerate(order):
        adjusted[position]=min(1.,max((12-j)*p[order[j]]for j in range(rank+1)))
    oracle=recomputed.copy();oracle['p_Holm12']=adjusted
    oracle['classification']=np.where(adjusted>=.05,'difference_not_detected',
        np.where(oracle.difference<0,'candidate_improvement','candidate_deterioration'))
    return max(raw_error,compare(family,oracle,keys))

def poverty_supplement_gate(meta):
    """Validate the separate prespecified descriptive contract without truth IO."""
    from sikov_poverty_stratum_descriptive import gate_function_digests
    planpath=ROOT/'validacion/sikov_poverty_stratum_descriptive_plan.json'
    qapath=ROOT/'validacion/sikov_poverty_stratum_descriptive_qa.json'
    source=ROOT/'scripts/sikov_poverty_stratum_descriptive.py'
    statuspath=ROOT/'validacion/sikov_poverty_stratum_descriptive_status.json'
    for path in (planpath,qapath,source,statuspath):public_logical_path(path,STUDY)
    plan=json.loads(planpath.read_text());qa=json.loads(qapath.read_text());status=json.loads(statuspath.read_text())
    identities={**meta['proof']['statistical_sources_sha256'],**meta['addendum']['additional_sources_sha256']}
    if (SHA(planpath)!=POVERTY_PLAN_SHA
            or plan['status']!='FROZEN_DESCRIPTIVE_POVERTY_STRATUM_PLAN_BEFORE_ACCURACY'
            or plan['source_sha256']!=SHA(source)or plan['synthetic_QA_sha256']!=SHA(qapath)
            or plan['gate_function_text_sha256']!=gate_function_digests()
            or plan['statistical_source_identities_sha256']!=identities or len(identities)!=37
            or plan['protocol_sha256']!=meta['protocol_sha256']or plan['addendum_sha256']!=meta['addendum_sha256']
            or plan['fresh_worlds']!=256 or plan['methods']!=meta['methods']
            or plan['contexts']!=meta['proof']['contexts']or plan['evaluation_ids']!=meta['proof']['evaluation_ids']
            or plan['groups']!=['all','observed_endpoint','unsampled_common']
            or plan['hypothesis_tests_or_p_values']or plan['confidence_intervals_or_significance_claims']
            or plan['Holm_families_modified']or plan['selection_or_tuning_performed']
            or plan['models_or_predictions_modified']or plan['population_or_method_exclusion_permitted']
            or plan['truth_or_accuracy_read_by_this_postprocess_before_plan']
            or not plan['operational_package_inventory_is_not_frozen_by_this_plan']
            or qa['status']!='PASS_SYNTHETIC_POVERTY_STRATUM_DESCRIPTIVE_QA'
            or qa['source_sha256']!=SHA(source)or not all(qa['checks'].values())
            or not qa['synthetic_arrays_only']or qa['actual_accuracy_or_truth_read']):
        raise RuntimeError('Changed poverty-stratum descriptive plan/source/QA or scope')
    if (status['status']!='PASS_FROZEN_POVERTY_STRATUM_DESCRIPTIVE_SUPPLEMENT'
            or status['protocol_sha256']!=meta['protocol_sha256']or status['addendum_sha256']!=meta['addendum_sha256']
            or status['joint_prediction_lock_sha256']!=meta['joint_sha256']
            or status['independent_audit_sha256']!=SHA(ROOT/'validacion/sikov_final_independent_audit.json')
            or status['plan_sha256']!=SHA(planpath)or status['source_sha256']!=SHA(source)
            or status['synthetic_QA_sha256']!=SHA(qapath)or status['fresh_worlds']!=256
            or status['methods']!=meta['methods']or status['poverty_strata']!=[1,2,3]
            or status['per_population_rows']!=50688 or status['summary_rows']!=10395
            or status['paired_descriptive_rows']!=2835 or not status['all_planned_cells_retained']
            or not status['all_method_convergence_flags_preserved']or status['convergence_failures_retained']!=0
            or status['models_predictions_weights_or_Holm_families_modified']
            or status['tests_p_values_reselection_or_accuracy_based_tuning']
            or not status['truth_access_only_after_prediction_preflight_and_completed_result_gates']
            or not status['descriptive_nonconfirmatory_sensitivity']
            or not status['rmse_summary_is_mean_population_rmse']):
        raise RuntimeError('Incomplete or altered poverty descriptive supplement')
    expected_inputs={}
    for context,replication,inputdir,original,manifest,_ in meta['closed']:
        expected_inputs[(inputdir/'evaluation_only/truth.npz').absolute()]=manifest['truth_sha256']
        expected_inputs[(original/'completion.json').absolute()]=SHA(original/'completion.json')
        additional=ROOT/'evaluacion/published_sar'/context/f'rep_{replication:05d}'
        expected_inputs[(additional/'completion.json').absolute()]=SHA(additional/'completion.json')
    actual_inputs={portable_path(name,meta['proof']):digest for name,digest in status['inputs_sha256'].items()}
    if len(status['inputs_sha256'])!=len(expected_inputs)or actual_inputs!=expected_inputs:
        raise RuntimeError('Poverty supplement input identity differs from the 256 sealed worlds')
    folder=ROOT/'resultados/poverty_stratum_descriptive'
    expected_outputs={(folder/name).absolute()for name in POVERTY_OUTPUTS}
    actual_outputs={portable_path(name,meta['proof']):digest for name,digest in status['outputs_sha256'].items()}
    if len(status['outputs_sha256'])!=3 or set(actual_outputs)!=expected_outputs:
        raise RuntimeError('Unexpected poverty supplement output identity')
    for path,digest in actual_outputs.items():
        if SHA(path)!=digest:raise RuntimeError('Changed poverty descriptive CSV')
    correctionpath=ROOT/'validacion/sikov_poverty_stratum_label_correction.json'
    boundarypath=ROOT/'validacion/sikov_poverty_stratum_boundary_scope.json'
    for path in (correctionpath,boundarypath):public_logical_path(path,STUDY)
    correction=json.loads(correctionpath.read_text());boundary=json.loads(boundarypath.read_text())
    if (SHA(correctionpath)!='785ef7572f39fe76f0634765b566c178bbdd832fa1bf2d018d81ac8ab44794e6'
            or SHA(boundarypath)!='8d68284d43284f4b572a32438e32e7a517f4e946d3b85083b7f621f38ee706f8'
            or correction['classification_codes_models_predictions_weights_or_Holm_families_changed']
            or correction['accuracy_or_truth_read_before_correction']
            or not boundary['classification_stored_in_frozen_inputs_is_used_unchanged']
            or not boundary['no_domain_or_simulation_classification_changed']
            or boundary['exact_percent_boundary_counts']!={'30':0,'55':0}):
        raise RuntimeError('Changed prospective poverty label correction or fixed-support boundary scope')
    lineage=[]
    for field in ('invalid_original_artifacts_preserved_sha256','corrected_artifacts_sha256'):
        for name,digest in correction[field].items():
            path=public_logical_path(ROOT/name,STUDY)
            if SHA(path)!=digest:raise RuntimeError('Changed corrected or preserved poverty-label artifact')
            lineage.append(path)
    return {'plan':plan,'QA':qa,'status':status,'folder':folder,
            'paths':[planpath,qapath,source,statuspath,correctionpath,boundarypath,*lineage,*expected_outputs]}

def verify_poverty_tables(oracle,frame,summary,differences,contexts,replications,methods):
    """Recompute every declared poverty cell and covariance-aware MCSE table."""
    from sikov_poverty_stratum_descriptive import with_endpoint_balance,summarize_populations,paired_descriptive
    keys=['scenario','replication','method','outcome','poverty_stratum','group']
    expected={(context,replication,method,outcome,stratum,group)for context in contexts
              for replication in replications for method in methods for outcome in ('anemia','stunting')
              for stratum in (1,2,3)for group in ('all','observed_endpoint','unsampled_common')}
    for table in (oracle,frame):
        if len(table)!=len(expected)or set(map(tuple,table[keys].to_numpy()))!=expected or table.duplicated(keys).any():
            raise RuntimeError('Incomplete poverty world/method/endpoint/stratum/group universe')
    balanced=with_endpoint_balance(oracle)
    summary_keys=['method','outcome','poverty_stratum','group','scenario','metric']
    differences_keys=summary_keys+['reference']
    return max(compare(frame,oracle,keys),
               compare(summary,summarize_populations(balanced,contexts,replications),summary_keys),
               compare(differences,paired_descriptive(balanced,contexts,replications),differences_keys))

def sampled_graph_supplement_gate(meta,require_results=True):
    """Close all 64 supplemental predictions without loading any truth or draws.

    This gate is independent of the frozen primary 256-world function bodies.
    The private ICC draws remain omitted; their sealed hash and independent
    inversion audit are retained, while every public prediction is rehashed.
    """
    def read_json(path):
        public_logical_path(path,STUDY)
        return json.loads(path.read_text())
    planpath=ROOT/'validacion/sensitivity_sampled_graph_freeze.json'
    qapath=ROOT/'validacion/sensitivity_sampled_graph_prefit_QA.json'
    lockpath=ROOT/'validacion/sensitivity_sampled_graph_prediction_lock.json'
    auditpath=ROOT/'validacion/sensitivity_sampled_graph_prediction_audit.json'
    plan=read_json(planpath);qa=read_json(qapath);lock=read_json(lockpath);audit=read_json(auditpath)
    ids=list(range(30000,30064));methods=meta['methods'];context=SAMPLED_GRAPH_CONTEXT
    source_names={'scripts/sensitivity_sampled_graph_'+part+'.py'for part in ('generate','fit','audit','score','qa')}
    input_names={'datos/simulation_template/'+name for name in ('auxiliary.npz','manifest.json','area_ids.csv')}
    if (SAMPLED_GRAPH_PLAN_SHA is None or SHA(planpath)!=SAMPLED_GRAPH_PLAN_SHA
            or plan['status']!='FROZEN_PROSPECTIVE_DESCRIPTIVE_SENSITIVITY'
            or plan['protocol_sha256']!=meta['protocol_sha256']or plan['addendum_sha256']!=meta['addendum_sha256']
            or plan['context']!=context or plan['replications']!=ids or plan['worlds']!=64
            or plan['methods']!=methods or plan['selected_recipe']!=meta['proof']['selected_recipe']
            or plan['accuracy_or_truth_read_before_freeze']or plan['main_primary_family_membership']
            or plan['formal_p_values_or_new_Holm_family']
            or not plan['original_37_sources_256_worlds_and_Holm3_Holm12_unchanged']
            or not plan['no_recipe_or_model_selection_on_supplemental_results']
            or set(plan['sources_sha256'])!=source_names or set(plan['inputs_sha256'])!=input_names
            or plan['prefit_QA_path']!='validacion/sensitivity_sampled_graph_prefit_QA.json'
            or plan['prefit_QA_sha256']!=SHA(qapath)
            or qa['status']!='PASS_FICTITIOUS_PREFIT_SAMPLED_GRAPH_SENSITIVITY_QA' or not qa['passed']
            or qa['actual_final_truth_or_accuracy_read']or qa['actual_microdata_or_reserved_responses_read']
            or qa['QA_source_sha256']!=SHA(ROOT/'scripts/sensitivity_sampled_graph_qa.py')):
        raise RuntimeError('Changed prospective sampled-graph descriptive plan/source/QA or scope')
    paths=[planpath,qapath,lockpath,auditpath]
    for field in ('sources_sha256','inputs_sha256'):
        for name,digest in plan[field].items():
            path=public_logical_path(ROOT/name,STUDY)
            if SHA(path)!=digest:raise RuntimeError('Changed sampled-graph frozen source or template')
            paths.append(path)
    if (lock['status']!='PASS_COMPLETE_64_SUPPLEMENTAL_PREDICTION_SEAL' or lock['worlds']!=64
            or lock['protocol_sha256']!=meta['protocol_sha256']or lock['addendum_sha256']!=meta['addendum_sha256']
            or lock['sensitivity_plan_sha256']!=SHA(planpath)or lock['methods']!=methods
            or lock['all_expected_worlds']is not True or lock['fresh_truth_read_before_lock']
            or lock['approved_for_scoring']is not False
            or not lock['requires_principal_JOINT_and_independent_audit']or not lock['selected_recipe_unchanged']
            or len(lock['prediction_items'])!=64
            or {item['replication']for item in lock['prediction_items']}!=set(ids)
            or audit['status']!='PASS_INDEPENDENT_64_SUPPLEMENTAL_PREDICTION_AUDIT'
            or audit['prediction_lock_sha256']!=SHA(lockpath)or audit['sensitivity_plan_sha256']!=SHA(planpath)
            or audit['protocol_sha256']!=meta['protocol_sha256']or audit['addendum_sha256']!=meta['addendum_sha256']
            or audit['worlds']!=64 or audit['methods']!=methods or audit['truth_values_read']
            or not audit['no_worlds_or_methods_excluded']
            or audit['auditor_source_sha256']!=SHA(ROOT/'scripts/sensitivity_sampled_graph_audit.py')):
        raise RuntimeError('Incomplete or changed independent sampled-graph prediction seal')
    output_names={prefix+'/'+method+suffix for prefix,group in (('original',methods[:9]),('published',methods[9:]))
                  for method in group for suffix in ('.npz','_status.json')}
    private_draw='original/ICC_conditional_draws.npz';closed=[];failures=[]
    destination=ROOT/'sensibilidad_sampled_graph'
    for item in lock['prediction_items']:
        rep=item['replication'];source=destination/'inputs'/f'rep_{rep:05d}';fit=destination/'fits'/f'rep_{rep:05d}'
        filepaths=[source/'manifest.json',source/'model_inputs/inputs.npz',source/'model_inputs/sample.csv',
                   fit/'completion.json',*[fit/name for name in output_names]]
        for path in filepaths:public_logical_path(path,STUDY)
        record=read_json(fit/'completion.json');manifest=read_json(source/'manifest.json')
        if (SHA(fit/'completion.json')!=item['completion_sha256']
                or record['context']!=context or record['replication']!=rep or item['context']!=context
                or record['protocol_sha256']!=meta['protocol_sha256']or record['addendum_sha256']!=meta['addendum_sha256']
                or record['sensitivity_plan_sha256']!=SHA(planpath)or record['truth_read_by_fits']
                or record['selected_recipe']!=meta['proof']['selected_recipe']
                or record['original_fit_seed_stream']!=[20261010,973,rep]
                or record['original_fit_seed']!=int(np.random.SeedSequence([20261010,973,rep]).generate_state(1)[0])%(2**32-200000)
                or record['published_bootstrap_seed_stream']!=[20261010,974,rep]
                or record['published_bootstrap_seed']!=int(np.random.SeedSequence([20261010,974,rep]).generate_state(1)[0])
                or set(record['convergence'])!=set(methods)
                or item['all_converged']!=all(record['convergence'].values())
                or item['output_sha256']!=record['output_sha256']
                or set(record['output_sha256'])!=output_names|{private_draw}
                or manifest['context']!=context or manifest['replication']!=rep
                or manifest['phase']!='prospective_descriptive_sensitivity'
                or manifest['protocol_sha256']!=meta['protocol_sha256']or manifest['addendum_sha256']!=meta['addendum_sha256']
                or manifest['population_stream']!=[20261010,971,rep]or manifest['sampling_stream']!=[20261010,972,rep]
                or manifest['sensitivity_plan_sha256']!=SHA(planpath)
                or not manifest['truth_excluded_from_model_inputs']or not manifest['generative_graph_excluded_from_model_inputs']
                or record['input_manifest_sha256']!=SHA(source/'manifest.json')):
            raise RuntimeError('Changed supplemental world identity or prediction role')
        for filename,key in (('inputs.npz','fit_inputs_sha256'),('sample.csv','sample_sha256')):
            digest=SHA(source/'model_inputs'/filename)
            if digest!=manifest[key]or digest!=record[key]:raise RuntimeError('Changed shared supplemental model input')
        for name in output_names:
            if SHA(fit/name)!=record['output_sha256'][name]:raise RuntimeError('Changed public supplemental prediction or status')
        for method in methods:
            if not record['convergence'][method]:failures.append([rep,method])
        closed.append((rep,source,fit,manifest,record))
    if sorted(failures)!=sorted(lock['convergence_failures_retained'])or sorted(failures)!=sorted(audit['convergence_failures_retained']):
        raise RuntimeError('Supplemental convergence flags must be retained without exclusion')
    result={'plan':plan,'QA':qa,'lock':lock,'prediction_audit':audit,'closed':closed,
            'folder':destination/'resultados','paths':paths,'separate_worlds':64}
    if require_results:
        statuspath=ROOT/'validacion/sensitivity_sampled_graph_evaluation_status.json'
        resultpath=ROOT/'validacion/sensitivity_sampled_graph_result_audit.json'
        status=read_json(statuspath);result_audit=read_json(resultpath)
        bindings={'sensitivity_plan_sha256':SHA(planpath),'prediction_lock_sha256':SHA(lockpath),
                  'prediction_audit_sha256':SHA(auditpath),'joint_prediction_lock_sha256':meta['joint_sha256'],
                  'principal_independent_audit_sha256':SHA(ROOT/'validacion/sikov_final_independent_audit.json')}
        for report,expected_status in ((status,'PASS_COMPLETE_64_DESCRIPTIVE_SUPPLEMENTAL_SCORING'),
                (result_audit,'PASS_INDEPENDENT_64_DESCRIPTIVE_SUPPLEMENTAL_RESULT_AUDIT')):
            if (report['status']!=expected_status or any(report[key]!=value for key,value in bindings.items())
                    or report['protocol_sha256']!=meta['protocol_sha256']or report['addendum_sha256']!=meta['addendum_sha256']
                    or report['worlds']!=64 or report['methods']!=methods
                    or not report['no_worlds_or_methods_excluded']or not report['selected_recipe_unchanged']
                    or report['formal_p_values_or_new_Holm_family']or not report['original_256_and_Holm3_Holm12_unchanged']
                    or set(report['outputs_sha256'])!=set(SAMPLED_GRAPH_OUTPUTS)):
                raise RuntimeError('Changed sampled-graph descriptive scoring role, family or completion')
        if (status['context']!=context or not status['truth_access_only_after_all_64_and_main_comparison_closure']
                or not status['shared_projection_0_1_is_secondary']
                or status['source_sha256']!=SHA(ROOT/'scripts/sensitivity_sampled_graph_score.py')
                or result_audit['auditor_source_sha256']!=SHA(ROOT/'scripts/sensitivity_sampled_graph_audit.py')
                or result_audit['score_rows_per_scale']!=4224
                or result_audit['outputs_sha256']!=status['outputs_sha256']
                or max(result_audit['independent_max_errors'].values())>1e-11):
            raise RuntimeError('Independent supplemental result audit or numeric tolerance failed')
        for name,digest in status['outputs_sha256'].items():
            path=public_logical_path(result['folder']/name,STUDY)
            if SHA(path)!=digest:raise RuntimeError('Changed sampled-graph descriptive CSV')
            paths.append(path)
        paths.extend((statuspath,resultpath));result.update(status=status,result_audit=result_audit)
    return result

def published_b300_gate(meta):
    """Portable metadata/input/prediction gate; excludes runtime and ICC draws."""
    from sikov_poverty_stratum_descriptive import gate_function_digests
    names=('plan','qa','native_status','status')
    files={role:public_logical_path(ROOT/'validacion'/('sikov_published_b300_'+role+'.json'),STUDY)for role in names}
    plan,qa,native,status=(json.loads(files[role].read_text())for role in names)
    source=public_logical_path(ROOT/'scripts/sikov_published_b300_sensitivity.py',STUDY)
    dependency=ROOT/'scripts/sikov_poverty_stratum_descriptive.py'
    identities={**meta['proof']['statistical_sources_sha256'],**meta['addendum']['additional_sources_sha256']}
    jobs=[(context,rep)for context in meta['proof']['contexts']for rep in (30000,30001)]
    if (SHA(files['plan'])!=PUBLISHED_B300_PLAN_SHA
            or plan['status']!='FROZEN_PUBLISHED_B300_DESCRIPTIVE_PLAN_BEFORE_ACCURACY'
            or plan['source_sha256']!=SHA(source)or plan['QA_sha256']!=SHA(files['qa'])
            or plan['score_dependency_sha256']!=SHA(dependency)
            or plan['gate_function_text_sha256']!=gate_function_digests()
            or plan['statistical_source_identities_sha256']!=identities or len(identities)!=37
            or plan['protocol_sha256']!=meta['protocol_sha256']or plan['addendum_sha256']!=meta['addendum_sha256']
            or plan['contexts']!=meta['proof']['contexts']or plan['replications']!=[30000,30001]
            or plan['worlds']!=8 or plan['B']!=300 or not plan['primary_B100_unchanged']
            or plan['models']!=list(PUBLISHED_B300_METHODS)or plan['descriptive_references']!=list(PUBLISHED_B300_METHODS[1:])
            or plan['groups']!=['all','observed_endpoint','unsampled_common']
            or plan['Selected_recipe']!=meta['proof']['selected_recipe']
            or plan['new_Selected_with_Published_component']or plan['reselection_tuning_hypothesis_tests_p_values_or_Holm_changes']
            or plan['population_or_finite_convergence_failure_exclusions']or plan['actual_accuracy_or_truth_read_before_plan']
            or not plan['operational_package_inventory_and_gate_module_full_sha_not_frozen']
            or qa['status']!='PASS_SYNTHETIC_PUBLISHED_B300_DESCRIPTIVE_QA' or not all(qa['checks'].values())
            or qa['source_sha256']!=SHA(source)or qa['score_dependency_sha256']!=SHA(dependency)
            or not qa['synthetic_arrays_only']or qa['truth_or_final_accuracy_read']):
        raise RuntimeError('Changed Published B300 prospective plan/source/QA or scope')
    if (native['status']!='PASS_EIGHT_CLOSED_PUBLISHED_B300_NATIVE_CASES' or native['worlds']!=8 or native['B']!=300
            or native['plan_sha256']!=SHA(files['plan'])or native['source_sha256']!=SHA(source)
            or native['QA_sha256']!=SHA(files['qa'])or native['truth_values_read']or not native['all_cases_retained']
            or len(native['cases'])!=8 or [(item['context'],item['replication'])for item in native['cases']]!=jobs
            or not np.isfinite(native['points_unchanged_max_absolute_difference'])
            or native['points_unchanged_max_absolute_difference']>1e-10):
        raise RuntimeError('Eight complete Published B300 native closures required')
    core={(context,rep):(inputs,original,manifest,record)for context,rep,inputs,original,manifest,record in meta['closed']}
    closed=[];failures=[];bindings={};paths=[*files.values(),source,dependency]
    for item in native['cases']:
        context,rep=item['context'],item['replication'];inputs,original,manifest,old=core[context,rep]
        published=ROOT/'evaluacion/published_sar'/context/f'rep_{rep:05d}'
        folder=ROOT/'evaluacion/published_b300'/context/f'rep_{rep:05d}'
        for name in ('completion.json','predictions.npz','status.json'):public_logical_path(folder/name,STUDY)
        record=json.loads((folder/'completion.json').read_text())
        expected={'protocol_sha256':meta['protocol_sha256'],'addendum_sha256':meta['addendum_sha256'],
            'plan_sha256':SHA(files['plan']),'source_sha256':SHA(source),'QA_sha256':SHA(files['qa']),
            'input_manifest_sha256':SHA(inputs/'manifest.json'),'fit_inputs_sha256':SHA(inputs/'model_inputs/inputs.npz'),
            'original_completion_sha256':SHA(original/'completion.json'),'published_completion_sha256':SHA(published/'completion.json'),
            'published_B100_predictions_sha256':SHA(published/'Sikov_SAR_Published.npz')}
        if (any(record[key]!=value for key,value in expected.items())or record['context']!=context or record['replication']!=rep
                or record['B']!=300 or record['bootstrap_seed']!=int(np.random.SeedSequence([20261010,975,meta['proof']['contexts'].index(context),rep]).generate_state(1)[0])
                or record['truth_read_by_fits']or record['closed_point_max_absolute_difference']>1e-10
                or not record['naive_and_BC_use_same_draws']or not record['raw_negative_BC_kept']
                or set(record['output_sha256'])!={'predictions.npz','status.json'}
                or SHA(folder/'completion.json')!=item['completion_sha256']
                or record['independent_native_audit']!=item['independent_native_audit']
                or record['convergence_failure_preserved']!=bool(not record['all_converged'])):
            raise RuntimeError('Changed Published B300 closed case role, seed or native audit')
        for name,digest in record['output_sha256'].items():
            if SHA(folder/name)!=digest:raise RuntimeError('Changed closed Published B300 array/status')
        if not record['all_converged']:failures.append([context,rep])
        bindings[(inputs/'evaluation_only/truth.npz').absolute()]=manifest['truth_sha256']
        for directory in (original,published,folder):bindings[(directory/'completion.json').absolute()]=SHA(directory/'completion.json')
        paths.extend(folder/name for name in ('completion.json','predictions.npz','status.json'))
        closed.append((context,rep,inputs,original,published,folder,manifest,old,record))
    if failures!=native['convergence_failures_retained']:raise RuntimeError('Changed B300 retained convergence failures')
    expected={'protocol_sha256':meta['protocol_sha256'],'addendum_sha256':meta['addendum_sha256'],
        'plan_sha256':SHA(files['plan']),'source_sha256':SHA(source),'QA_sha256':SHA(files['qa']),
        'native_status_sha256':SHA(files['native_status']),'joint_prediction_lock_sha256':meta['joint_sha256'],
        'principal_independent_audit_sha256':SHA(ROOT/'validacion/sikov_final_independent_audit.json')}
    if (status['status']!='PASS_FROZEN_PUBLISHED_B300_DESCRIPTIVE_SENSITIVITY'
            or any(status[key]!=value for key,value in expected.items())or status['native_cases']!=native['cases']
            or status['worlds']!=8 or status['methods']!=list(PUBLISHED_B300_METHODS)
            or status['per_population_rows']!=240 or status['summary_rows']!=1575 or status['paired_difference_rows']!=1260
            or not status['truth_access_only_after_both_public_gates_and_principal_audit']
            or not status['all_worlds_and_convergence_flags_retained']or status['models_points_weights_primary_B100_or_Holm_modified']
            or status['hypothesis_tests_or_reselection']or status['new_Selected_mixture_with_Published_component']
            or not status['descriptive_nonconfirmatory_sensitivity']):
        raise RuntimeError('Changed Published B300 descriptive result role or scope')
    released_bindings={portable_path(name,meta['proof']):digest for name,digest in status['input_and_closed_prediction_bindings_sha256'].items()}
    if (len(status['input_and_closed_prediction_bindings_sha256'])!=32
            or len(released_bindings)!=32 or released_bindings!=bindings):raise RuntimeError('Wrong eight-world B300 input/prediction identity set')
    folder=ROOT/'resultados/published_b300_descriptive'
    expected_outputs={(folder/name).absolute()for name in PUBLISHED_B300_OUTPUTS}
    outputs={portable_path(name,meta['proof']):digest for name,digest in status['outputs_sha256'].items()}
    if len(status['outputs_sha256'])!=3 or len(outputs)!=3 or set(outputs)!=expected_outputs:raise RuntimeError('Wrong Published B300 descriptive output identities')
    for path,digest in outputs.items():
        if SHA(path)!=digest:raise RuntimeError('Changed Published B300 descriptive CSV')
    paths.extend(outputs)
    return {'plan':plan,'QA':qa,'native':native,'status':status,'closed':closed,'folder':folder,'paths':paths}


def verify_published_b300(gate):
    """Recompute fixed eight-case native algebra and all descriptive tables."""
    from sikov_published_b300_sensitivity import audit_native,bc_arrays,score_cells,paired_scores
    from sikov_poverty_stratum_descriptive import with_endpoint_balance,summarize_populations
    rows=[]
    for context,rep,inputs,original,published,folder,manifest,old,record in gate['closed']:
        d=dict(np.load(inputs/'model_inputs/inputs.npz',allow_pickle=False))
        arrays=dict(np.load(folder/'predictions.npz',allow_pickle=False))
        checked=audit_native(d,json.loads((folder/'status.json').read_text()),arrays,300)
        def numeric_identity(actual,expected):
            if isinstance(expected,dict):
                return isinstance(actual,dict)and set(actual)==set(expected)and all(numeric_identity(actual[k],v)for k,v in expected.items())
            if isinstance(expected,(int,float))and not isinstance(expected,bool):
                return np.isfinite(actual)and np.isclose(actual,expected,atol=1e-10,rtol=1e-9)
            return actual==expected
        if not numeric_identity(checked,record['independent_native_audit']):raise RuntimeError('Published B300 native dense algebra differs')
        primary=dict(np.load(published/'Sikov_SAR_Published.npz',allow_pickle=False))
        if not np.allclose(arrays['mean'],primary['mean'],atol=1e-10,rtol=0):raise RuntimeError('Published B300 points differ from B100')
        truthpath=public_logical_path(inputs/'evaluation_only/truth.npz',STUDY)
        if SHA(truthpath)!=manifest['truth_sha256']:raise RuntimeError('Changed closed eight-case truth')
        theta=np.load(truthpath,allow_pickle=False)['theta']
        predictions={PUBLISHED_B300_METHODS[0]:dict(np.load(original/'Selected.npz',allow_pickle=False)),
            PUBLISHED_B300_METHODS[1]:primary,PUBLISHED_B300_METHODS[2]:dict(np.load(published/'Sikov_SAR_Published_BC.npz',allow_pickle=False)),
            PUBLISHED_B300_METHODS[3]:arrays,PUBLISHED_B300_METHODS[4]:bc_arrays(arrays)}
        pub=json.loads((published/'completion.json').read_text())
        convergence=dict(zip(PUBLISHED_B300_METHODS,[old['convergence']['Selected'],pub['convergence']['Sikov_SAR_Published'],
            pub['convergence']['Sikov_SAR_Published_BC'],record['all_converged'],record['all_converged']]))
        rows.extend(score_cells(theta,d,predictions,convergence,context,rep))
    frame=pd.DataFrame(rows);balanced=with_endpoint_balance(frame)
    drop=['poverty_stratum','poverty_stratum_label'];contexts=gate['plan']['contexts'];reps=gate['plan']['replications']
    expected=(frame.drop(columns=drop),summarize_populations(balanced,contexts,reps).drop(columns=drop),
        paired_scores(balanced,contexts,reps).drop(columns=drop))
    keys=(['scenario','replication','method','outcome','group'],['method','outcome','group','scenario','metric'],
          ['method','outcome','group','scenario','metric','reference'])
    return max(compare(pd.read_csv(gate['folder']/name),oracle,key)
               for name,oracle,key in zip(PUBLISHED_B300_OUTPUTS,expected,keys))


def read_sampled_csv(path):
    """Decode the sole intentionally empty text field without changing numbers.

    The frozen scorer writes undefined_reason='' for defined estimates. Pandas
    reads that CSV cell as NaN by default. Preserve only this text column
    literally; numeric missingness and explicit nonempty reasons for undefined
    zero-MSE ratios stay unchanged, and forged 'NA'/'null' labels remain visible.
    """
    return pd.read_csv(path,converters={'undefined_reason':str})


def verify_sampled_graph_supplement(gate):
    """Recompute six separate descriptive CSVs after all 64 predictions close."""
    rows={False:[],True:[]};methods=gate['plan']['methods']
    for rep,source,fit,manifest,completion in gate['closed']:
        path=public_logical_path(source/'evaluation_only/truth.npz',STUDY)
        if SHA(path)!=manifest['truth_sha256']:raise RuntimeError('Changed separate sampled-graph truth')
        theta=np.load(path,allow_pickle=False)['theta'];inputs=dict(np.load(source/'model_inputs/inputs.npz',allow_pickle=False))
        for method in methods:
            prediction=dict(np.load(fit/('original'if method in methods[:9]else'published')/(method+'.npz'),allow_pickle=False))
            for projected in (False,True):
                rows[projected].extend(sampled_score_oracle(theta,inputs,prediction,method,rep,completion['convergence'][method],projected))
    errors=[]
    for projected in (False,True):
        prefix='physical_projection'if projected else'descriptive';frame=pd.DataFrame(rows[projected])
        if len(frame)!=4224:raise RuntimeError('Separate sampled-graph cohort requires all 4224 score cells')
        summary,contrasts=sampled_table_oracles(frame,methods)
        frames=((frame,'replications',['scenario','replication','method','outcome','group']),
                (summary,'summary',['method','outcome','group']),
                (contrasts,'paired_contrasts',['candidate','reference','group','metric']))
        for oracle,suffix,keys in frames:
            actual=read_sampled_csv(gate['folder']/(prefix+'_'+suffix+'.csv'))
            if any(column.startswith('p_')or 'Holm'in column for column in actual.columns):
                raise RuntimeError('The sampled-graph supplement cannot introduce p values or a Holm family')
            errors.append(compare(actual,oracle,keys))
    return max(errors)


def sampled_score_oracle(theta,inputs,prediction,method,rep,converged,physical=False):
    """Independent direct calculation of the six domain-mean scoring rules."""
    target=np.asarray(theta,float);index=np.asarray(inputs['sample_idx'],int)
    y=np.asarray(inputs['y'],float);n=len(target)
    values=np.stack([prediction[name]for name in ('mean','lower','upper')]).astype(float)
    if (target.shape!=(120,2)or not np.isfinite(target).all()or np.any((target<0)|(target>1))
            or values.shape!=(3,n,2)or not np.isfinite(values).all()or np.any(values[1]>values[2])
            or y.shape!=(len(index),2)or len(set(index))!=len(index)or np.any((index<0)|(index>=n))):
        raise ValueError('Incomplete independent sampled-graph target/prediction frame')
    seen=np.zeros((n,2),bool);seen[index]=np.isfinite(y)
    outside=np.ones(n,bool);outside[index]=False
    if physical:values=np.clip(values,0.,1.)
    rows=[]
    for endpoint,endpoint_name in enumerate(('anemia','stunting')):
        for group,mask in (('all',np.ones(n,bool)),('observed_endpoint',seen[:,endpoint]),('unsampled_common',outside)):
            if not mask.any():raise ValueError('Empty sampled-graph group cannot be omitted')
            point,lo,hi=values[:,mask,endpoint];truth=target[mask,endpoint];error=point-truth
            penalties=np.where(truth<lo,40*(lo-truth),np.where(truth>hi,40*(truth-hi),0.))
            rows.append(dict(scenario=SAMPLED_GRAPH_CONTEXT,replication=rep,method=method,outcome=endpoint_name,
                group=group,areas=int(mask.sum()),mse=float(error@error/len(error)),mae=float(np.abs(error).mean()),
                bias=float(error.mean()),interval_score=float((hi-lo+penalties).mean()),
                coverage=float(((lo<=truth)&(truth<=hi)).mean()),width=float((hi-lo).mean()),converged=bool(converged),
                point_outside_0_1=float(((point<0)|(point>1)).mean()),
                interval_outside_0_1=float(((lo<0)|(hi>1)).mean()),
                phase='prospective_descriptive_sampled_graph_sensitivity',shared_physical_projection=bool(physical)))
    return rows


def sampled_table_oracles(frame,methods):
    """Independent paired-world means, MCSE and log-ratio uncertainty."""
    metrics=('mse','mae','bias','interval_score','coverage','width');ids=list(range(30000,30064))
    keys=['scenario','replication','method','outcome','group'];groups=('all','observed_endpoint','unsampled_common')
    expected={(SAMPLED_GRAPH_CONTEXT,r,m,e,g)for r in ids for m in methods for e in ('anemia','stunting')for g in groups}
    if len(frame)!=4224 or frame.duplicated(keys).any()or set(map(tuple,frame[keys].to_numpy()))!=expected:
        raise ValueError('Exactly 64 separate worlds, 11 methods, 2 endpoints and 3 groups required')
    summaries=[];contrasts=[];Z=1.959963984540054
    def moments(a):return a.mean(axis=0),a.std(axis=0,ddof=1)/8.
    for group in groups:
        balanced={}
        for method in methods:
            base=frame[frame.group.eq(group)&frame.method.eq(method)]
            by_endpoint={e:base[base.outcome.eq(e)].set_index('replication').loc[ids]for e in ('anemia','stunting')}
            a=np.stack([part[list(metrics)].to_numpy(float)for part in by_endpoint.values()])
            balanced[method]=a.mean(axis=0)
            for label,worlds in [('anemia',a[0]),('stunting',a[1]),('balanced_anemia_stunting',balanced[method])]:
                mean,se=moments(worlds);rmse=np.sqrt(mean[0])
                selected=base if label=='balanced_anemia_stunting'else by_endpoint[label].reset_index()
                summaries.append(dict(method=method,outcome=label,group=group,worlds=64,rmse=float(rmse),
                    rmse_mcse=float(se[0]/(2*rmse))if rmse>0 else 0.,
                    **dict(zip(metrics,map(float,mean))),**dict(zip((m+'_mcse'for m in metrics),map(float,se))),
                    nonconverged_worlds_retained=int((~selected.groupby('replication').converged.all()).sum()),
                    descriptive_only=True))
        candidate=balanced['Selected']
        for method in methods:
            if method=='Selected':continue
            reference=balanced[method];mean,se=moments(candidate-reference)
            for j,metric in enumerate(metrics):
                contrasts.append(dict(candidate='Selected',reference=method,group=group,metric=metric,paired_worlds=64,
                    difference=float(mean[j]),mcse=float(se[j]),MC_CI95_low=float(mean[j]-Z*se[j]),
                    MC_CI95_high=float(mean[j]+Z*se[j]),descriptive_only=True,estimate_defined=True,undefined_reason='',
                    mcse_log_ratio=np.nan,interval_scope='Unadjusted Monte Carlo interval; no formal test or multiplicity claim'))
            m1,m0=candidate[:,0].mean(),reference[:,0].mean()
            item=dict(candidate='Selected',reference=method,group=group,metric='RMSE_ratio',paired_worlds=64,descriptive_only=True)
            if m1==0 or m0==0:
                item.update(difference=np.nan,mcse=np.nan,mcse_log_ratio=np.nan,MC_CI95_low=np.nan,MC_CI95_high=np.nan,
                    estimate_defined=False,undefined_reason='zero_candidate_or_reference_mean_MSE',
                    interval_scope='Ratio and its uncertainty not reported when either mean MSE is zero')
            else:
                logratio=.5*(np.log(m1)-np.log(m0));ratio=float(np.exp(logratio))
                influence=.5*(candidate[:,0]/m1-reference[:,0]/m0);logse=float(influence.std(ddof=1)/8.)
                item.update(difference=ratio,mcse=ratio*logse,mcse_log_ratio=logse,
                    MC_CI95_low=float(np.exp(logratio-Z*logse)),MC_CI95_high=float(np.exp(logratio+Z*logse)),
                    estimate_defined=True,undefined_reason='',
                    interval_scope='Unadjusted delta-method Monte Carlo interval on log RMSE ratio; no formal test')
            contrasts.append(item)
    return pd.DataFrame(summaries),pd.DataFrame(contrasts)

def verify_poverty_supplement(meta,gate):
    """Called after core prediction/result gates and core score recomputation."""
    from sikov_poverty_stratum_descriptive import cells
    rows=[]
    for context,replication,inputdir,original,manifest,completion in meta['closed']:
        truthpath=inputdir/'evaluation_only/truth.npz'
        if SHA(truthpath)!=manifest['truth_sha256']:raise RuntimeError('Changed poverty supplement finite-population truth')
        truth=dict(np.load(truthpath,allow_pickle=False));inputs=dict(np.load(inputdir/'model_inputs/inputs.npz',allow_pickle=False))
        additional=ROOT/'evaluacion/published_sar'/context/f'rep_{replication:05d}'
        published=json.loads((additional/'completion.json').read_text())
        predictions={method:dict(np.load((additional if method in ADDED_METHODS else original)/(method+'.npz'),
                                         allow_pickle=False))for method in meta['methods']}
        convergence={**completion['convergence'],**published['convergence']}
        rows.extend(cells(truth['theta'],inputs,predictions,convergence,context,replication))
    tables=[pd.read_csv(gate['folder']/name)for name in POVERTY_OUTPUTS]
    return verify_poverty_tables(pd.DataFrame(rows),*tables,meta['proof']['contexts'],
                                meta['proof']['evaluation_ids'],meta['methods'])

def main():
    manifest=json.loads((PACKAGE/'public_manifest.json').read_text())
    expected_members=set(manifest['files_sha256'])
    present={str(p.relative_to(PACKAGE))for p in PACKAGE.rglob('*')
             if p.is_file()and '__pycache__'not in p.parts
             and not any(part.startswith('.')for part in p.relative_to(PACKAGE).parts)}
    if present!=expected_members|{'public_manifest.json'}:
        raise RuntimeError('Released file inventory differs from manifest')
    meta=prediction_preflight()
    # The manifest cannot legitimize an arbitrary extra CSV/JSON/NPZ or private
    # file. Require the same positive inventory used by the archive builder.
    from public_package import expected_public_paths
    required,optional=expected_public_paths(meta['proof'],meta['addendum'])
    required_members={str(path.relative_to(PACKAGE))for path in required}|{'README.txt','requirements.txt'}
    allowed_members=required_members|{str(path.relative_to(PACKAGE))for path in optional}
    if not required_members.issubset(expected_members)or not expected_members.issubset(allowed_members):
        raise RuntimeError('Released manifest violates the exact positive public inventory')
    paths={name:safe_member(name)for name in expected_members}
    # First verify code, metadata and predictions; leave truth bytes untouched.
    for name,path in paths.items():
        if is_truth(path):continue
        if not path.is_file()or SHA(path)!=manifest['files_sha256'][name]:
            raise RuntimeError('Public artifact changed '+name)
    result_gates=completed_result_gates(meta)
    poverty_gate=poverty_supplement_gate(meta)
    sampled_gate=sampled_graph_supplement_gate(meta)
    b300_gate=published_b300_gate(meta)
    proof=meta['proof'];freeze=meta['protocol_sha256'];closed=meta['closed']
    if (manifest['protocol_sha256']!=freeze or proof['fresh_worlds']!=256
            or manifest['expected_fresh_worlds']!=256
            or manifest['addendum_sha256']!=meta['addendum_sha256']
            or manifest['joint_prediction_lock_sha256']!=meta['joint_sha256']
            or manifest['expected_methods']!=meta['methods']
            or manifest['expected_separate_sampled_graph_worlds']!=64
            or manifest['poverty_supplement_reuses_primary_worlds']!=256
            or manifest['supplemental_worlds_added_to_primary_family']
            or manifest['sampled_graph_plan_sha256']!=SAMPLED_GRAPH_PLAN_SHA
            or manifest['poverty_plan_sha256']!=POVERTY_PLAN_SHA
            or manifest['published_B300_plan_sha256']!=PUBLISHED_B300_PLAN_SHA
            or manifest['published_B300_reuses_primary_worlds']!=8
            or manifest['private_individual_records_exported']
            or manifest['stochastic_conditional_draws_exported']
            or manifest['real_units_children_households_exported']
            or manifest['executables_or_environments_exported']):
        raise RuntimeError('Wrong released protocol, cohort or privacy role')
    # Imports preserve frozen scoring formulas and never invoke a fitting gate.
    from comparison_simulation import CONTEXTS,EVALUATION_IDS
    from comparison_evaluate import METHODS
    from improve_development import scores
    from comparison_collect import balanced
    from benchmark_metrics import summarize,paired_contrasts
    if (list(CONTEXTS)!=proof['contexts']or list(EVALUATION_IDS)!=proof['evaluation_ids']
            or list(METHODS)!=proof['methods']):
        raise RuntimeError('Frozen cohort or method definitions differ')
    expected_truths={str((source/'evaluation_only/truth.npz').relative_to(PACKAGE))
                     for _,_,source,_,_,_ in closed}
    actual_truths={name for name,path in paths.items()if is_truth(path)}
    supplemental_truths={str((source/'evaluation_only/truth.npz').relative_to(PACKAGE))
                          for _,source,_,_,_ in sampled_gate['closed']}
    if len(expected_truths)!=256 or len(supplemental_truths)!=64 or actual_truths!=expected_truths|supplemental_truths:
        raise RuntimeError('Wrong complete released synthetic truth inventory')
    # Every planned prediction and its inputs passed above before ANY truth open.
    for name in actual_truths:
        if SHA(paths[name])!=manifest['files_sha256'][name]:
            raise RuntimeError('Released synthetic truth hash changed '+name)
    rows=[];physical=[];additional=[];additional_physical=[]
    for c,r,source,fit,record,completion in closed:
        truth_path=source/'evaluation_only/truth.npz'
        if SHA(truth_path)!=record['truth_sha256']:raise RuntimeError('Changed synthetic truth')
        truth=dict(np.load(truth_path,allow_pickle=False))
        direct=dict(np.load(source/'model_inputs/inputs.npz',allow_pickle=False))
        for method in METHODS:
            pred=dict(np.load(fit/(method+'.npz'),allow_pickle=False))
            args=(method,c,r,[completion['convergence'][method]]*2,completion['seconds'])
            raw_rows=scores(truth,direct,pred,*args)
            for row in raw_rows:
                row.update(phase='fresh_evaluation',protocol='same14_Sikov_parametricbootstrap100_ICC_NUTS_frozen')
            rows.extend(raw_rows)
            bounded={**pred,**{k:np.clip(pred[k],0.,1.)for k in ('mean','lower','upper')}}
            physical_rows=scores(truth,direct,bounded,*args)
            for row in physical_rows:
                row.update(phase='fresh_evaluation_shared_physical_projection',
                           protocol='all_methods_shared_projection_0_1_secondary_not_selected')
            physical.extend(physical_rows)
        published=ROOT/'evaluacion/published_sar'/c/f'rep_{r:05d}'
        pubproof=json.loads((published/'completion.json').read_text())
        for method in ADDED_METHODS:
            pred=dict(np.load(published/(method+'.npz'),allow_pickle=False))
            args=(method,c,r,[pubproof['convergence'][method]]*2,pubproof['seconds'])
            added=scores(truth,direct,pred,*args)
            for row in added:
                row.update(phase='fresh_evaluation_addendum',
                           protocol='sampled_SAR2023_family_directed2024_extension_parametricbootstrap100')
            additional.extend(added)
            bounded={**pred,**{k:np.clip(pred[k],0.,1.)for k in ('mean','lower','upper')}}
            projected_rows=scores(truth,direct,bounded,*args)
            for row in projected_rows:
                row.update(phase='fresh_evaluation_addendum_shared_physical_projection',
                           protocol='all_methods_shared_projection_0_1_secondary_not_selected')
            additional_physical.extend(projected_rows)
    raw=pd.DataFrame(rows);projected=pd.DataFrame(physical);results=ROOT/'resultados'
    errors=[];verified=[]
    def check(frame,filename,keys):
        errors.append(compare(frame,pd.read_csv(results/filename),keys));verified.append(filename)
    keys=['scenario','replication','outcome','group','method']
    check(raw,'fresh_replications.csv',keys)
    check(projected,'fresh_physical_projection_replications.csv',keys)
    summarykeys=['scenario','method','protocol','group','outcome']
    check(summarize(raw),'fresh_summary.csv',summarykeys)
    check(summarize(projected),'fresh_physical_projection_summary.csv',summarykeys)
    contrasts=pd.DataFrame([row for ref in METHODS if ref!='Selected'for row in balanced(raw,'Selected',ref)])
    primary=contrasts.metric.eq('interval_score')&contrasts.reference.isin(['Sikov_FH','Sikov_SAR'])
    ps=contrasts.loc[primary,'p_two_sided'].to_numpy()
    if len(ps)!=2:raise RuntimeError('Incomplete primary family')
    order=np.argsort(ps);adjusted=np.empty(2)
    adjusted[order]=np.minimum(1,np.maximum.accumulate(ps[order]*np.array([2,1])))
    contrasts.loc[primary,'p_Holm2']=adjusted
    contrastkeys=['candidate','reference','metric']
    check(contrasts,'fresh_balanced_paired_contrasts.csv',contrastkeys)
    projected_contrasts=pd.DataFrame([row for ref in ('Sikov_FH','Sikov_SAR','ICC')
                                      for row in balanced(projected,'Selected',ref)])
    check(projected_contrasts,'fresh_physical_projection_contrasts.csv',contrastkeys)
    detailed=pd.concat([paired_contrasts(raw[raw.method.eq('Selected')],raw[raw.method.eq(ref)],
                                        'Selected_minus_'+ref)
                        for ref in ('Sikov_FH','Sikov_SAR','ICC')],ignore_index=True)
    check(detailed,'fresh_context_endpoint_paired_contrasts.csv',
          ['scenario','group','outcome','contrast'])
    expanded=pd.concat([raw,pd.DataFrame(additional)],ignore_index=True)
    expanded_physical=pd.concat([projected,pd.DataFrame(additional_physical)],ignore_index=True)
    expanded_results=results/'published_sar_comparison'
    def check_expanded(frame,filename,keys):
        errors.append(compare(frame,pd.read_csv(expanded_results/filename),keys))
        verified.append('published_sar_comparison/'+filename)
    check_expanded(expanded,'per_replication_scores.csv',keys=['scenario','replication','outcome','group','method'])
    check_expanded(expanded_physical,'physical_projection_scores.csv',keys=['scenario','replication','outcome','group','method'])
    check_expanded(summarize(expanded),'summary.csv',summarykeys)
    check_expanded(summarize(expanded_physical),'physical_projection_summary.csv',summarykeys)
    from sikov_published_collect import expanded_contrasts
    check_expanded(expanded_contrasts(expanded),'paired_balanced_contrasts_Holm3.csv',contrastkeys)
    sensitivity=pd.DataFrame([row for ref in REFERENCES for row in balanced(expanded_physical,'Selected',ref)])
    check_expanded(sensitivity,'physical_projection_contrasts.csv',contrastkeys)
    external_frames={scope:pd.read_csv(results/('external_extended_'+scope)/'district_scores.csv',
                                       dtype={'ubigeo':str,'province':str})
                     for scope in ('temporal','territorial')}
    external_oracle=recompute_external_contrasts(external_frames)
    external_family=pd.read_csv(results/'external_comparison_Holm12/paired_balanced_contrasts_Holm12.csv')
    errors.append(verify_external_family(external_family,external_oracle))
    for scope in ('temporal','territorial'):
        scope_raw=pd.read_csv(results/('external_extended_'+scope)/'paired_balanced_contrasts_unadjusted.csv')
        errors.append(compare(scope_raw,external_oracle.loc[external_oracle.scope.eq(scope)],
                              ['scope','candidate','reference','metric']))
    # Recompute four-method external numeric contrasts from released district
    # aggregates only. The frozen auditor does not reopen individual responses.
    from align_external_extended_audit import audit as external_audit
    with tempfile.TemporaryDirectory(prefix='endes_public_external_QA_')as temporary:
        output=Path(temporary)/'audit.json'
        external_audit([results/'external_extended_temporal',results/'external_extended_territorial'],
                       results/'external_comparison_Holm12',output)
        external_check=json.loads(output.read_text())
    if external_check['checks_passed']!=result_gates['external_QA']['checks_passed']:
        raise RuntimeError('Released external numeric audit has different coverage')
    errors.append(verify_poverty_supplement(meta,poverty_gate))
    errors.append(verify_sampled_graph_supplement(sampled_gate))
    errors.append(verify_published_b300(b300_gate))
    verified.extend('poverty_stratum_descriptive/'+name for name in POVERTY_OUTPUTS)
    verified.extend('separate_sampled_graph/'+name for name in SAMPLED_GRAPH_OUTPUTS)
    verified.extend('published_b300_descriptive/'+name for name in PUBLISHED_B300_OUTPUTS)
    print(json.dumps({'status':'PASS_PUBLIC_INTEGRITY_AND_FULL_SYNTHETIC_SCORE_RECOMPUTATION',
      'files':len(manifest['files_sha256']),'worlds':256,'methods':len(meta['methods']),
      'addendum_sha256':meta['addendum_sha256'],'joint_prediction_lock_sha256':meta['joint_sha256'],
      'expanded_primary_Holm3_recomputed':True,'original_Holm2_recomputed_separately':True,
      'external_Holm12_numeric_checks_recomputed':external_check['checks_passed'],
      'external_Holm12_all12_rows_linked_to_recomputed_aggregates_before_Holm':True,
      'poverty_stratum_256_world_descriptive_tables_recomputed':True,
      'separate_sampled_graph_64_world_sensitivity_recomputed':True,
      'published_B300_fixed_eight_case_descriptive_tables_and_native_algebra_recomputed':True,
      'supplemental_worlds_added_to_primary_family':False,
      'recomputed_files':verified,'all_predictions_verified_before_first_truth_open':True,
      'maximum_numerical_error':max(errors),'microdata_accessed':False,
      'refits_or_reselection':False,'conditional_draws_omitted_as_declared':True},indent=2))

def b300_fixture_checks(checks,reject,write):
    """Artificial prospective seals and scores for the same fixed eight cases."""
    global PUBLISHED_B300_PLAN_SHA
    from sikov_published_b300_sensitivity import score_cells,paired_scores,MODELS,bc_arrays
    from sikov_poverty_stratum_descriptive import with_endpoint_balance,summarize_populations,gate_function_digests
    contexts=['fixture_C1','fixture_C2','fixture_C3','fixture_C4'];reps=[30000,30001];rows=[]
    for ci,context in enumerate(contexts):
        for rep in reps:
            theta=np.full((6,2),.4);inputs={'X_raw':np.ones((6,15)),'sample_idx':np.array([0,1,3,4]),'y':np.full((4,2),.4)}
            inputs['y'][0,0]=np.nan
            predictions={method:{'mean':theta+.01*j,'lower':theta-.05+ci*.002,'upper':theta+.06+(rep-30000)*.002}
                         for j,method in enumerate(MODELS)}
            rows.extend(score_cells(theta,inputs,predictions,dict.fromkeys(MODELS,True),context,rep))
    frame=pd.DataFrame(rows);balanced=with_endpoint_balance(frame)
    summary=summarize_populations(balanced,contexts,reps);differences=paired_scores(balanced,contexts,reps)
    assert len(frame)==240 and len(summary)==1575 and len(differences)==1260
    assert not any(c.startswith('p_')or 'CI'in c or 'Holm'in c for c in differences.columns)
    checks.append('published_B300_complete240_scores1575_summary1260_differences_without_tests')
    reject('published_B300_missing_world_summary_blocked',lambda:summarize_populations(balanced.iloc[:-1],contexts,reps))
    changed=differences.copy();changed.loc[0,'difference_Selected_B100_minus_reference']+=.01
    reject('published_B300_changed_paired_difference_blocked',lambda:compare(changed,differences,['method','outcome','poverty_stratum','group','scenario','metric','reference']))
    prior=PUBLISHED_B300_PLAN_SHA
    try:
        identities={str(ROOT/f'fake_statistical_{i}.py'):'retained_digest'for i in range(37)}
        proof={'contexts':contexts,'selected_recipe':{'family':'ICC_SAR_mixture','icc_weight':.75},
               'statistical_sources_sha256':identities,'path_anchor':str(STUDY)}
        meta={'proof':proof,'addendum':{'additional_sources_sha256':{}},'protocol_sha256':'fixture_base',
              'addendum_sha256':'fixture_add','joint_sha256':'fixture_joint','closed':[]}
        source=ROOT/'scripts/sikov_published_b300_sensitivity.py';source_sha=write(source,'fake_B300_source')
        dep=ROOT/'scripts/sikov_poverty_stratum_descriptive.py';dep_sha=write(dep,'fake_dependency')
        qa=dict(status='PASS_SYNTHETIC_PUBLISHED_B300_DESCRIPTIVE_QA',checks={'fake_arrays_only':True},
            source_sha256=source_sha,score_dependency_sha256=dep_sha,synthetic_arrays_only=True,truth_or_final_accuracy_read=False)
        qasha=write(ROOT/'validacion/sikov_published_b300_qa.json',qa)
        plan=dict(status='FROZEN_PUBLISHED_B300_DESCRIPTIVE_PLAN_BEFORE_ACCURACY',source_sha256=source_sha,QA_sha256=qasha,
            score_dependency_sha256=dep_sha,gate_function_text_sha256=gate_function_digests(),
            statistical_source_identities_sha256=identities,protocol_sha256='fixture_base',addendum_sha256='fixture_add',
            contexts=contexts,replications=reps,worlds=8,B=300,primary_B100_unchanged=True,models=list(PUBLISHED_B300_METHODS),
            descriptive_references=list(PUBLISHED_B300_METHODS[1:]),groups=['all','observed_endpoint','unsampled_common'],
            Selected_recipe=proof['selected_recipe'],new_Selected_with_Published_component=False,
            reselection_tuning_hypothesis_tests_p_values_or_Holm_changes=False,population_or_finite_convergence_failure_exclusions=False,
            actual_accuracy_or_truth_read_before_plan=False,operational_package_inventory_and_gate_module_full_sha_not_frozen=True)
        PUBLISHED_B300_PLAN_SHA=write(ROOT/'validacion/sikov_published_b300_plan.json',plan)
        cases=[];bindings={}
        for ci,context in enumerate(contexts):
            for rep in reps:
                inputs=ROOT/'evaluacion/inputs'/context/f'rep_{rep:05d}'
                original=ROOT/'evaluacion/fits'/context/f'rep_{rep:05d}'
                published=ROOT/'evaluacion/published_sar'/context/f'rep_{rep:05d}'
                output=ROOT/'evaluacion/published_b300'/context/f'rep_{rep:05d}'
                manifest={'truth_sha256':'fake_truth_not_opened'}
                manifestsha=write(inputs/'manifest.json',manifest);modelsha=write(inputs/'model_inputs/inputs.npz','fake_model_input')
                originalsha=write(original/'completion.json',{'fake_completion':True})
                pubsha=write(published/'completion.json',{'fake_completion':True})
                pubpredsha=write(published/'Sikov_SAR_Published.npz','fake_prediction')
                meta['closed'].append((context,rep,inputs,original,manifest,{}))
                outputs={name:write(output/name,'fake_prediction_or_status')for name in ('predictions.npz','status.json')}
                record=dict(protocol_sha256='fixture_base',addendum_sha256='fixture_add',plan_sha256=PUBLISHED_B300_PLAN_SHA,
                    source_sha256=source_sha,QA_sha256=qasha,input_manifest_sha256=manifestsha,fit_inputs_sha256=modelsha,
                    original_completion_sha256=originalsha,published_completion_sha256=pubsha,published_B100_predictions_sha256=pubpredsha,
                    context=context,replication=rep,B=300,bootstrap_seed=int(np.random.SeedSequence([20261010,975,ci,rep]).generate_state(1)[0]),
                    truth_read_by_fits=False,closed_point_max_absolute_difference=0.,naive_and_BC_use_same_draws=True,raw_negative_BC_kept=True,
                    output_sha256=outputs,independent_native_audit={'fixture_dense_check':True},all_converged=True,convergence_failure_preserved=False)
                cases.append(dict(context=context,replication=rep,completion_sha256=write(output/'completion.json',record),
                    independent_native_audit=record['independent_native_audit']))
                bindings[str(inputs/'evaluation_only/truth.npz')]=manifest['truth_sha256']
                for directory in (original,published,output):bindings[str(directory/'completion.json')]=SHA(directory/'completion.json')
        native=dict(status='PASS_EIGHT_CLOSED_PUBLISHED_B300_NATIVE_CASES',worlds=8,B=300,plan_sha256=PUBLISHED_B300_PLAN_SHA,
            source_sha256=source_sha,QA_sha256=qasha,truth_values_read=False,all_cases_retained=True,cases=cases,
            points_unchanged_max_absolute_difference=0.,convergence_failures_retained=[])
        nativepath=ROOT/'validacion/sikov_published_b300_native_status.json';nativesha=write(nativepath,native)
        auditsha=write(ROOT/'validacion/sikov_final_independent_audit.json',{'fake_complete_audit':True})
        outputs={str(ROOT/'resultados/published_b300_descriptive'/name):write(ROOT/'resultados/published_b300_descriptive'/name,'fake_closed_descriptive_CSV')
                 for name in PUBLISHED_B300_OUTPUTS}
        status=dict(status='PASS_FROZEN_PUBLISHED_B300_DESCRIPTIVE_SENSITIVITY',protocol_sha256='fixture_base',addendum_sha256='fixture_add',
            plan_sha256=PUBLISHED_B300_PLAN_SHA,source_sha256=source_sha,QA_sha256=qasha,native_status_sha256=nativesha,
            joint_prediction_lock_sha256='fixture_joint',principal_independent_audit_sha256=auditsha,native_cases=cases,worlds=8,
            methods=list(PUBLISHED_B300_METHODS),per_population_rows=240,summary_rows=1575,paired_difference_rows=1260,
            truth_access_only_after_both_public_gates_and_principal_audit=True,all_worlds_and_convergence_flags_retained=True,
            models_points_weights_primary_B100_or_Holm_modified=False,hypothesis_tests_or_reselection=False,
            new_Selected_mixture_with_Published_component=False,descriptive_nonconfirmatory_sensitivity=True,
            input_and_closed_prediction_bindings_sha256=bindings,outputs_sha256=outputs)
        write(ROOT/'validacion/sikov_published_b300_status.json',status)
        assert len(published_b300_gate(meta)['closed'])==8
        checks.append('published_B300_eight_public_native_seals_pass_without_truth_or_runtime_files')
        key=next(iter(bindings));path=Path(key);alias=str(path.parent)+'/./'+path.name
        write(ROOT/'validacion/sikov_published_b300_status.json',{**status,
            'input_and_closed_prediction_bindings_sha256':{**bindings,alias:bindings[key]}})
        reject('published_B300_aliased_extra_input_binding_blocked',lambda:published_b300_gate(meta))
        key=next(iter(outputs));path=Path(key);alias=str(path.parent)+'/./'+path.name
        write(ROOT/'validacion/sikov_published_b300_status.json',{**status,'outputs_sha256':{**outputs,alias:outputs[key]}})
        reject('published_B300_aliased_extra_output_binding_blocked',lambda:published_b300_gate(meta))
        write(ROOT/'validacion/sikov_published_b300_status.json',status)
        write(nativepath,{**native,'cases':cases[:-1]})
        reject('published_B300_seven_cases_blocked_before_any_truth_read',lambda:published_b300_gate(meta))
        write(nativepath,native);write(ROOT/'resultados/published_b300_descriptive/summary.csv','adulterated_table')
        reject('published_B300_changed_descriptive_CSV_blocked',lambda:published_b300_gate(meta))
    finally:PUBLISHED_B300_PLAN_SHA=prior


def supplement_fixture_checks(checks,reject,write):
    """Artificial arrays and 64 fabricated seals; no study outcome reads."""
    global SAMPLED_GRAPH_PLAN_SHA
    from sikov_poverty_stratum_descriptive import cells,with_endpoint_balance,summarize_populations,paired_descriptive
    methods=['Sikov_FH','Sikov_SAR','Sikov_FH_BC','Sikov_SAR_BC','Sikov_SAR_k4','Sikov_SAR_k8',
             'Sikov_SAR_haversine','ICC','Selected',*ADDED_METHODS]
    contexts=['fixture_C1','fixture_C2'];reps=list(range(4));rows=[]
    for ci,context in enumerate(contexts):
        for rep in reps:
            truth=np.full((9,2),.3)+np.arange(9)[:,None]*.005+rep*.002
            inputs={'X_raw':np.ones((9,15)),'strata':np.repeat([1,2,3],3),'sample_idx':np.array([0,1,3,4,6,7]),
                    'y':np.full((6,2),.3)}
            inputs['y'][0,1]=np.nan
            predictions={method:{'mean':truth+.005*(j+1)+ci*.002,
                'lower':truth-.05+rep*.001,'upper':truth+.06}for j,method in enumerate(methods)}
            rows.extend(cells(truth,inputs,predictions,dict.fromkeys(methods,True),context,rep))
    poverty=pd.DataFrame(rows);balanced=with_endpoint_balance(poverty)
    summary=summarize_populations(balanced,contexts,reps);paired=paired_descriptive(balanced,contexts,reps)
    assert verify_poverty_tables(poverty,poverty,summary,paired,contexts,reps,methods)==0
    checks.append('poverty_complete_strata_groups_endpoint_balance_and_MCSE_recomputed')
    reject('poverty_missing_stratum_group_cell_blocked',lambda:verify_poverty_tables(poverty,poverty.iloc[:-1],summary,paired,contexts,reps,methods))
    tampered=summary.copy();tampered.loc[0,'mcse']+=.01
    reject('poverty_adulterated_MCSE_blocked',lambda:verify_poverty_tables(poverty,poverty,tampered,paired,contexts,reps,methods))
    tampered=paired.copy();tampered.loc[0,'difference_Selected_minus_reference']+=.01
    reject('poverty_adulterated_paired_difference_blocked',lambda:verify_poverty_tables(poverty,poverty,summary,tampered,contexts,reps,methods))
    from sensitivity_sampled_graph_score import score_arrays,descriptive_summary,descriptive_contrasts
    rows=[];native=[]
    for rep in range(30000,30064):
        truth=np.full((120,2),.25)+np.arange(120)[:,None]*.001+(rep-30000)*.0001
        inputs={'sample_idx':np.arange(96),'y':np.full((96,2),.3)};inputs['y'][0,1]=np.nan
        for j,method in enumerate(methods):
            prediction={'mean':truth+.01*(j+1)+.001*np.sin(rep),'lower':truth-.04,'upper':truth+.055}
            converged=not(rep==30000 and method=='ICC')
            for physical in (False,True):
                a=pd.DataFrame(sampled_score_oracle(truth,inputs,prediction,method,rep,converged,physical))
                b=pd.DataFrame(score_arrays(truth,inputs,prediction,method,rep,converged,physical))
                assert compare(a,b,['scenario','replication','method','outcome','group'])<1e-10
                if not physical:rows.extend(a.to_dict('records'));native.extend(b.to_dict('records'))
    frame=pd.DataFrame(rows);a,b=sampled_table_oracles(frame,methods)
    assert compare(a,descriptive_summary(pd.DataFrame(native)),['method','outcome','group'])<1e-10
    assert compare(b,descriptive_contrasts(pd.DataFrame(native)),['candidate','reference','group','metric'])<1e-10
    checks.append('sampled64_independent_domain_scores_summary_MCSE_and_ratio_match_frozen_formulas')
    assert a.query('method == "ICC"').nonconverged_worlds_retained.eq(1).all()
    checks.append('sampled64_failed_finite_world_flags_retained_without_exclusion')
    reject('sampled64_missing_world_blocked',lambda:sampled_table_oracles(frame[frame.replication!=30063],methods))
    reject('sampled64_duplicated_cell_blocked',lambda:sampled_table_oracles(pd.concat([frame,frame.iloc[[0]]]),methods))
    zero=frame.copy();zero.loc[zero.method.eq('Selected'),'mse']=0.
    _,ratios=sampled_table_oracles(zero,methods)
    ratios=ratios[ratios.metric.eq('RMSE_ratio')]
    assert len(ratios)==30 and not ratios.estimate_defined.any()and ratios[['difference','mcse','mcse_log_ratio','MC_CI95_low','MC_CI95_high']].isna().all().all()
    checks.append('sampled64_zero_MSE_ratios_retained_undefined_without_invented_estimates')
    # Exercise actual CSV serialization, rather than only in-memory labels.
    contrast_keys=['candidate','reference','group','metric']
    csvpath=ROOT/'generated_sampled_contrasts.csv'
    b.to_csv(csvpath,index=False)
    roundtrip=read_sampled_csv(csvpath)
    assert compare(roundtrip,b,contrast_keys)<1e-10
    checks.append('sampled64_defined_blank_reason_csv_roundtrip_preserved')
    ratios.to_csv(csvpath,index=False)
    zero_roundtrip=read_sampled_csv(csvpath)
    assert compare(zero_roundtrip,ratios,contrast_keys)<1e-10
    checks.append('sampled64_explicit_zero_MSE_reason_and_numeric_NaN_csv_roundtrip_preserved')
    tampered=zero_roundtrip.copy();tampered.loc[tampered.index[0],'undefined_reason']=''
    reject('sampled64_missing_explicit_zero_MSE_reason_blocked_after_csv_read',
           lambda:compare(tampered,ratios,contrast_keys))
    tampered=roundtrip.copy();tampered.loc[tampered.index[0],'undefined_reason']='invented_reason'
    reject('sampled64_invented_defined_reason_blocked_after_csv_read',
           lambda:compare(tampered,b,contrast_keys))
    for token in ('NA','null'):
        tampered=roundtrip.copy();tampered.loc[tampered.index[0],'undefined_reason']=token
        tampered.to_csv(csvpath,index=False)
        literal_label=read_sampled_csv(csvpath)
        reject('sampled64_literal_'+token+'_reason_not_normalized_to_blank',
               lambda:compare(literal_label,b,contrast_keys))
    tampered=zero_roundtrip.copy();tampered.loc[tampered.index[0],'difference']=0.
    reject('sampled64_numeric_missingness_not_relaxed_by_label_csv_decode',
           lambda:compare(tampered,ratios,contrast_keys))
    # The complete positive supplemental seal is checked while every truth and
    # private draw is absent. Presence of those files is unnecessary for gating.
    original_plan_sha=SAMPLED_GRAPH_PLAN_SHA
    try:
        protocol,addendum='fixture_base','fixture_add';recipe={'family':'ICC_SAR_mixture','icc_weight':.75}
        source_hashes={'scripts/sensitivity_sampled_graph_'+name+'.py':write(ROOT/'scripts'/('sensitivity_sampled_graph_'+name+'.py'),'fake_source_'+name)
                       for name in ('generate','fit','audit','score','qa')}
        input_hashes={'datos/simulation_template/'+name:write(ROOT/'datos/simulation_template'/name,'fake_template_'+name)
                      for name in ('auxiliary.npz','manifest.json','area_ids.csv')}
        qapath=ROOT/'validacion/sensitivity_sampled_graph_prefit_QA.json'
        qasha=write(qapath,dict(status='PASS_FICTITIOUS_PREFIT_SAMPLED_GRAPH_SENSITIVITY_QA',passed=True,
            actual_final_truth_or_accuracy_read=False,actual_microdata_or_reserved_responses_read=False,
            QA_source_sha256=source_hashes['scripts/sensitivity_sampled_graph_qa.py']))
        plan=dict(status='FROZEN_PROSPECTIVE_DESCRIPTIVE_SENSITIVITY',protocol_sha256=protocol,addendum_sha256=addendum,
            context=SAMPLED_GRAPH_CONTEXT,replications=list(range(30000,30064)),worlds=64,methods=methods,selected_recipe=recipe,
            accuracy_or_truth_read_before_freeze=False,main_primary_family_membership=False,formal_p_values_or_new_Holm_family=False,
            original_37_sources_256_worlds_and_Holm3_Holm12_unchanged=True,no_recipe_or_model_selection_on_supplemental_results=True,
            sources_sha256=source_hashes,inputs_sha256=input_hashes,prefit_QA_path='validacion/sensitivity_sampled_graph_prefit_QA.json',prefit_QA_sha256=qasha)
        SAMPLED_GRAPH_PLAN_SHA=write(ROOT/'validacion/sensitivity_sampled_graph_freeze.json',plan)
        items=[]
        for rep in range(30000,30064):
            source=ROOT/'sensibilidad_sampled_graph/inputs'/f'rep_{rep:05d}'
            fit=ROOT/'sensibilidad_sampled_graph/fits'/f'rep_{rep:05d}'
            modelsha=write(source/'model_inputs/inputs.npz','fake_closed_model_input');samplesha=write(source/'model_inputs/sample.csv','fake_synthetic_sample')
            manifest=dict(context=SAMPLED_GRAPH_CONTEXT,replication=rep,phase='prospective_descriptive_sensitivity',
                protocol_sha256=protocol,addendum_sha256=addendum,sensitivity_plan_sha256=SAMPLED_GRAPH_PLAN_SHA,
                truth_excluded_from_model_inputs=True,generative_graph_excluded_from_model_inputs=True,
                population_stream=[20261010,971,rep],sampling_stream=[20261010,972,rep],
                fit_inputs_sha256=modelsha,sample_sha256=samplesha,truth_sha256='truth_never_read')
            manifestsha=write(source/'manifest.json',manifest)
            outputs={prefix+'/'+method+suffix:write(fit/(prefix+'/'+method+suffix),'fake_prediction_or_status')
                for prefix,group in (('original',methods[:9]),('published',methods[9:]))for method in group for suffix in ('.npz','_status.json')}
            outputs['original/ICC_conditional_draws.npz']='private_draw_never_read'
            completion=dict(context=SAMPLED_GRAPH_CONTEXT,replication=rep,protocol_sha256=protocol,addendum_sha256=addendum,
                sensitivity_plan_sha256=SAMPLED_GRAPH_PLAN_SHA,truth_read_by_fits=False,selected_recipe=recipe,
                convergence=dict.fromkeys(methods,True),output_sha256=outputs,input_manifest_sha256=manifestsha,
                fit_inputs_sha256=modelsha,sample_sha256=samplesha,original_fit_seed_stream=[20261010,973,rep],
                original_fit_seed=int(np.random.SeedSequence([20261010,973,rep]).generate_state(1)[0])%(2**32-200000),
                published_bootstrap_seed_stream=[20261010,974,rep],published_bootstrap_seed=int(np.random.SeedSequence([20261010,974,rep]).generate_state(1)[0]))
            items.append(dict(replication=rep,context=SAMPLED_GRAPH_CONTEXT,completion_sha256=write(fit/'completion.json',completion),all_converged=True,output_sha256=outputs))
        lock=dict(status='PASS_COMPLETE_64_SUPPLEMENTAL_PREDICTION_SEAL',protocol_sha256=protocol,addendum_sha256=addendum,
            sensitivity_plan_sha256=SAMPLED_GRAPH_PLAN_SHA,worlds=64,methods=methods,all_expected_worlds=True,approved_for_scoring=False,
            fresh_truth_read_before_lock=False,requires_principal_JOINT_and_independent_audit=True,selected_recipe_unchanged=True,
            prediction_items=items,convergence_failures_retained=[])
        lockpath=ROOT/'validacion/sensitivity_sampled_graph_prediction_lock.json';locksha=write(lockpath,lock)
        audit=dict(status='PASS_INDEPENDENT_64_SUPPLEMENTAL_PREDICTION_AUDIT',protocol_sha256=protocol,addendum_sha256=addendum,
            sensitivity_plan_sha256=SAMPLED_GRAPH_PLAN_SHA,prediction_lock_sha256=locksha,worlds=64,methods=methods,
            truth_values_read=False,no_worlds_or_methods_excluded=True,convergence_failures_retained=[],
            auditor_source_sha256=source_hashes['scripts/sensitivity_sampled_graph_audit.py'])
        auditpath=ROOT/'validacion/sensitivity_sampled_graph_prediction_audit.json';write(auditpath,audit)
        meta=dict(protocol_sha256=protocol,addendum_sha256=addendum,methods=methods,proof={'selected_recipe':recipe})
        assert len(sampled_graph_supplement_gate(meta,False)['closed'])==64
        checks.append('sampled64_all_public_predictions_and_inputs_sealed_without_truth_or_private_draw_files')
        changed={**lock,'prediction_items':items[:-1]};write(lockpath,changed)
        reject('sampled64_partial_prediction_seal_blocked_before_truth',lambda:sampled_graph_supplement_gate(meta,False))
        write(lockpath,lock);bad={**items[0],'context':'wrong_primary_context'};write(lockpath,{**lock,'prediction_items':[bad,*items[1:]]})
        write(auditpath,{**audit,'prediction_lock_sha256':SHA(lockpath)})
        reject('sampled64_substituted_primary_context_blocked_before_truth',lambda:sampled_graph_supplement_gate(meta,False))
        write(lockpath,lock);write(auditpath,audit)
        output=ROOT/'sensibilidad_sampled_graph/fits/rep_30000/original/Sikov_FH.npz'
        write(output,'adulterated_prediction')
        reject('sampled64_changed_public_prediction_blocked_before_truth',lambda:sampled_graph_supplement_gate(meta,False))
    finally:SAMPLED_GRAPH_PLAN_SHA=original_plan_sha


def self_test():
    """Fabricated hash/schema fixtures only; never use study truth or scores."""
    global ROOT,STUDY,SHA
    original_root,original_study,original_sha=ROOT,STUDY,SHA;checks=[]
    def reject(name,operation):
        try:operation()
        except (RuntimeError,ValueError,KeyError,FileNotFoundError):checks.append(name);return
        raise AssertionError('Fixture failed to reject '+name)
    try:
        with tempfile.TemporaryDirectory(prefix='endes_public_gate_fixture_')as temporary:
            STUDY=Path(temporary);ROOT=STUDY/'sikov';contexts=['C1','C2','C3','C4'];ids=list(range(64));methods=['A','B']
            freeze='fixture_base';addsha='fixture_add';items=[];pubitems=[]
            def write(path,value):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value)if isinstance(value,dict)else value)
                return original_sha(path)
            def guarded(path):
                if is_truth(Path(path)):raise AssertionError('Truth touched by prediction fixture')
                return original_sha(path)
            SHA=guarded
            for c in contexts:
                for r in ids:
                    source=ROOT/'evaluacion/inputs'/c/f'rep_{r:05d}'
                    fit=ROOT/'evaluacion/fits'/c/f'rep_{r:05d}'
                    a=write(source/'model_inputs/inputs.npz','generated_fixture_input')
                    b=write(source/'model_inputs/sample.csv','generated_fixture_sample')
                    record={'phase':'evaluation','context':c,'replication':r,'fit_inputs_sha256':a,'sample_sha256':b}
                    recsha=write(source/'manifest.json',record)
                    outputs={name:write(fit/name,'generated_fixture_prediction')
                             for m in methods for name in (m+'.npz',m+'_status.json')}
                    completion={'context':c,'replication':r,'protocol_sha256':freeze,'truth_read_by_fits':False,
                      'convergence':dict.fromkeys(methods,True),'fit_inputs_sha256':a,'sample_sha256':b,
                      'input_manifest_sha256':recsha,'output_sha256':outputs}
                    digest=write(fit/'completion.json',completion)
                    items.append({'context':c,'replication':r,'completion_sha256':digest,'all_converged':True,'outputs':outputs})
                    pub=ROOT/'evaluacion/published_sar'/c/f'rep_{r:05d}'
                    outs={name:write(pub/name,'generated_fixture_published_prediction')
                          for m in ADDED_METHODS for name in (m+'.npz',m+'_status.json')}
                    pc={'context':c,'replication':r,'protocol_sha256':freeze,'addendum_sha256':addsha,
                      'original_completion_sha256':digest,'truth_read_by_fits':False,
                      'convergence':dict.fromkeys(ADDED_METHODS,True),'fit_inputs_sha256':a,
                      'input_manifest_sha256':recsha,'output_sha256':outs}
                    pdigest=write(pub/'completion.json',pc)
                    pubitems.append({'context':c,'replication':r,'completion_sha256':pdigest,'all_converged':True,'outputs':outs})
            lock={'prediction_items':items,'all_expected_worlds':256,'approved_for_scoring':True,
                  'fresh_truth_read_before_lock':False,'convergence_failures_retained':0}
            closed=verify_closed_predictions(lock,freeze,contexts,ids,methods)
            checks.append('all256_original_predictions_pass_without_truth_read')
            publock={'prediction_items':pubitems,'worlds':256,'protocol_sha256':freeze,'addendum_sha256':addsha,
                     'fresh_truth_read_before_lock':False,'convergence_failures_retained':0,
                     'requires_complete_joint_prediction_lock':True}
            verify_published_predictions(publock,{},closed,freeze,addsha)
            checks.append('all256_additional_predictions_pass_without_truth_read')
            reject('255_original_worlds_blocked',lambda:verify_closed_predictions({**lock,'prediction_items':items[:-1]},freeze,contexts,ids,methods))
            reject('255_additional_worlds_blocked',lambda:verify_published_predictions({**publock,'prediction_items':pubitems[:-1]}, {},closed,freeze,addsha))
            reject('published_addendum_mismatch_blocked',lambda:verify_published_predictions(publock,{},closed,freeze,'wrong_addendum'))
            sample=pd.DataFrame({'key':['A','B'],'value':[1.,np.nan],'label':['x','y']})
            assert compare(sample,sample,['key'])==0;checks.append('numeric_nan_and_labels_preserved')
            reject('changed_score_label_blocked',lambda:compare(sample,sample.assign(label=['z','y']),['key']))
            reject('duplicate_score_cell_blocked',lambda:compare(pd.concat([sample,sample]),sample,['key']))
            reject('changed_missingness_blocked',lambda:compare(sample,sample.fillna(2.),['key']))
            # Joint seal identities are rejected without touching member bytes.
            joint_names=['protocol_freeze.json','validacion/sikov_published_addendum_freeze.json',
                         'validacion/fresh_prediction_lock.json','validacion/sikov_published_fresh_prediction_lock.json']
            for scope in ('temporal','territorial'):
                for prefix in ('','published_sar/'):
                    joint_names.extend('evaluacion/'+prefix+'real_'+scope+'/'+name
                                       for name in ('completion.json','predictions.csv'))
                joint_names.extend('validacion/'+name for name in
                    ('real_prediction_lock_'+scope+'.json','sikov_published_prediction_lock_'+scope+'.json'))
            joint={'files':[{'path':str(ROOT/name),'sha256':'fixture_digest'}for name in joint_names]}
            header={'path_anchor':str(STUDY)}
            assert len(joint_file_inventory(joint,header))==16
            checks.append('joint_exact16_identities_checked_without_hash_reads')
            extra={'path':str(ROOT/'evaluacion/inputs/C1/rep_00000/evaluation_only/truth.npz'),'sha256':'fixture_digest'}
            reject('joint_extra_truth_rejected_before_any_hash',lambda:joint_file_inventory({'files':joint['files']+[extra]},header))
            reject('joint_duplicate_rejected_before_any_hash',lambda:joint_file_inventory({'files':joint['files'][:-1]+[joint['files'][0]]},header))
            reject('joint_unexpected_path_substitution_rejected_before_any_hash',lambda:joint_file_inventory({'files':joint['files'][:-1]+[extra]},header))
            write(ROOT/'protocol_freeze.json',header)
            write(ROOT/'validacion/joint_prediction_lock.json',{'files':joint['files']+[extra]})
            def forbid_any_hash(path):raise AssertionError('A joint member was hashed before inventory rejection')
            SHA=forbid_any_hash
            try:reject('prediction_preflight_rejects_extra_before_even_protocol_hash',prediction_preflight)
            finally:SHA=guarded
            truth_fixture=ROOT/'evaluacion/inputs/C1/rep_00000/evaluation_only/truth.npz'
            write(truth_fixture,'generated_truth_fixture_never_read_by_gate')
            link=ROOT/'evaluacion/real_temporal/predictions.csv';link.parent.mkdir(parents=True,exist_ok=True)
            link.symlink_to(truth_fixture)
            reject('joint_prediction_symlink_to_truth_blocked_before_hash',lambda:joint_file_inventory(joint,header))
            write(ROOT/'validacion/joint_prediction_lock.json',joint)
            SHA=forbid_any_hash
            try:reject('preflight_prediction_symlink_to_truth_blocked_before_any_SHA',prediction_preflight)
            finally:SHA=guarded
            link.unlink();directory=link.parent;directory.rmdir();directory.symlink_to(truth_fixture.parent)
            reject('joint_symlink_ancestor_to_truth_folder_blocked_before_hash',lambda:joint_file_inventory(joint,header))
            directory.unlink()
            # The archive selector only enumerates explicit names. Even files
            # with otherwise permitted extensions cannot join through discovery.
            from public_package import expected_public_paths,select_public_files
            proof={**header,'contexts':contexts,'evaluation_ids':ids,'methods':methods,'statistical_sources_sha256':{}}
            add={'additional_sources_sha256':{},'frozen_inputs_sha256':{},'new_methods':list(ADDED_METHODS)}
            required,optional=expected_public_paths(proof,add,ROOT,STUDY)
            for path in required:
                if not path.exists():write(path,'generated_public_inventory_fixture')
            unexpected=[ROOT/folder/name for folder in ('resultados','validacion')
                        for name in ('unexpected.csv','unexpected.json','unexpected.npz')]
            unexpected.extend((ROOT/'resultados/nested/children.csv',ROOT/'validacion/private_external/check.json',
                               ROOT/'desarrollo/improve_development_v4_native_rho999/new_metrics.csv',
                               ROOT/'scripts/unreviewed_helper.py'))
            for path in unexpected:write(path,'generated_unexpected_file')
            selected=select_public_files(proof,add,ROOT,STUDY)
            assert selected==required and not selected.intersection(unexpected)
            checks.append('unexpected_CSV_JSON_NPZ_nested_files_and_helpers_not_exported')
            assert (ROOT/'validacion/align_poverty_vintage_source_provenance.json').absolute()in selected
            assert (ROOT/'validacion/comparison_application_figures_preparation_QA.json').absolute()in selected
            assert (ROOT/'fuentes/literature_update_2026-10-10.json').absolute()in selected
            assert (ROOT/'fuentes/journal_editorial_check_2026-10-10.json').absolute()in selected
            assert (ROOT/'fuentes/mixture_method_reference_2026-10-10.json').absolute()in selected
            assert (ROOT/'fuentes/historical_endpoint_and_survey_reference_2026-10-10.json').absolute()in selected
            assert (ROOT/'fuentes/solicitud_insumos_originales_Sikov.txt').absolute()in selected
            assert (ROOT/'validacion/report_bibliography_preparation_QA.json').absolute()in selected
            checks.append('poverty_literature_and_map_QA_explicitly_exported')
            missing=ROOT/'resultados/fresh_summary.csv';missing.unlink()
            reject('missing_required_public_artifact_blocked',lambda:select_public_files(proof,add,ROOT,STUDY))
            missing.symlink_to(truth_fixture)
            reject('public_expected_artifact_symlink_to_truth_not_exported',lambda:select_public_files(proof,add,ROOT,STUDY))
            missing.unlink()
            # Distinct artificial values for all scopes/references reveal row
            # swaps that preserve a 12-row count and a self-consistent Holm p.
            frames={}
            for scope_index,scope in enumerate(('temporal','territorial')):
                rows=[]
                for method_index,method in enumerate(('Selected',*REFERENCES)):
                    for province in range(4):
                        for district in range(2):
                            code=f'{province+1:04d}{district+1:02d}'
                            for endpoint_index,endpoint in enumerate(('anemia','stunting')):
                                y=.20+.04*province+.03*endpoint_index+.01*district+.015*scope_index
                                pa=.19+.018*province+.005*district+.035*method_index+.015*scope_index*method_index
                                ps=pa+.022;p=pa if endpoint=='anemia'else ps
                                rows.append({'ubigeo':code,'province':code[:4],'outcome':endpoint,'method':method,
                                             'direct':y,'p_anemia':pa,'p_stunting':ps,
                                             'MSE':(p-y)**2,'NLL':-(y*np.log(p)+(1-y)*np.log1p(-p))})
                frames[scope]=pd.DataFrame(rows)
            oracle=recompute_external_contrasts(frames)
            from align_external_extended_score import contrasts as frozen_external_contrasts
            frozen_fixture=pd.DataFrame([row for scope in ('temporal','territorial')
                for row in frozen_external_contrasts(frames[scope],scope,list(REFERENCES))])
            assert compare(oracle,frozen_fixture,['scope','candidate','reference','metric'])<1e-10
            checks.append('independent_external_formula_matches_frozen_scorer_on_artificial_aggregates')
            metric_changed={scope:frame.copy()for scope,frame in frames.items()}
            metric_changed['temporal'].loc[0,'MSE']+=.02
            reject('external_adulterated_district_metric_blocked',lambda:recompute_external_contrasts(metric_changed))
            def adjusted_family(raw):
                f=raw.copy();p=f.p_two_sided.to_numpy(float);order=np.argsort(p,kind='stable');a=np.empty(12)
                a[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(12,0,-1)))
                f['p_Holm12']=a;f['classification']=np.where(a>=.05,'difference_not_detected',
                    np.where(f.difference<0,'candidate_improvement','candidate_deterioration'))
                return f
            family=adjusted_family(oracle)
            assert verify_external_family(family.sample(frac=1,random_state=1),oracle)==0
            checks.append('all12_external_contrasts_linked_by_scope_reference_metric_before_Holm')
            for field in ('difference','p_two_sided','cluster_standard_error','unadjusted_t_ci_low',
                          'unadjusted_t_ci_high','descriptive_bootstrap_ci_low','descriptive_bootstrap_ci_high',
                          'endpoint_difference_anemia','t_df'):
                changed=oracle.copy();changed.loc[0,field]+=1 if field=='t_df'else .02
                reject('external_adulterated_'+field+'_blocked_even_with_recomputed_Holm',
                       lambda changed=changed:verify_external_family(adjusted_family(changed),oracle))
            numeric=list(oracle.select_dtypes(include=['number']).columns)
            swapped=oracle.copy();swapped.loc[[0,1],numeric]=swapped.loc[[1,0],numeric].to_numpy()
            reject('external_swapped_reference_family_blocked',lambda:verify_external_family(adjusted_family(swapped),oracle))
            swapped_scopes=family.copy();swapped_scopes['scope']=swapped_scopes.scope.map({'temporal':'territorial','territorial':'temporal'})
            reject('external_swapped_scope_family_blocked',lambda:verify_external_family(swapped_scopes,oracle))
            wrong_reference=family.copy();wrong_reference.loc[0,'reference']='unregistered_reference'
            reject('external_unregistered_reference_blocked',lambda:verify_external_family(wrong_reference,oracle))
            reject('external_duplicate_family_member_blocked',lambda:verify_external_family(pd.concat([family.iloc[:-1],family.iloc[[0]]]),oracle))
            reject('external_missing_family_member_blocked',lambda:verify_external_family(family.iloc[:-1],oracle))
            altered_holm=family.copy();altered_holm.loc[0,'p_Holm12']+=.03
            reject('external_changed_Holm_blocked_after_exact_raw_link',lambda:verify_external_family(altered_holm,oracle))
            supplement_fixture_checks(checks,reject,write)
            b300_fixture_checks(checks,reject,write)
    finally:ROOT,STUDY,SHA=original_root,original_study,original_sha
    print(json.dumps({'status':'PASS_GENERATED_FIXTURES_ONLY','checks':checks,
                     'study_truth_or_accuracy_read':False,'models_fit':False},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    self_test()if args.self_test else main()
