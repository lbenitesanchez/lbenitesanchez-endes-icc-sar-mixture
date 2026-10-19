"""Secondary reused-ENDES2024 evaluation after signed prediction locks.

No automatic model selection, no fitting, no truth access before all checks.
Balanced endpoint MSE and NLL contrasts maintain their covariance by clustering
both endpoints together within province. The complete two-scopes/two-references/
two-metrics family is adjusted with Holm8 only in the collection operation.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key,'1')
from pathlib import Path
import argparse,json,hashlib,datetime
import numpy as np
import pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT.parent;OLD=STUDY/'fase2_2026-10-07'
ENDPOINTS=['anemia','stunting'];BOOTSTRAPS=3999;EPS=1e-12
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify_lock(lockpath,protocol,predictions,intervals=None):
    lock=json.loads(lockpath.read_text())
    if not lock.get('approved_for_scoring',False)or lock.get('protocol_sha256')!=sha(protocol):
        raise RuntimeError('Unapproved prediction lock or protocol hash mismatch')
    expected=[]
    for key in ('predictions','files','locked_files'):
        rows=lock.get(key,[])
        if isinstance(rows,list):expected.extend(rows)
    def match(path,flatkey):
        actual=sha(path)
        if lock.get(flatkey)==actual:return
        if any(row.get('sha256',row.get('predictions_sha256'))==actual for row in expected):return
        raise RuntimeError('Input not identified by signed lock: '+str(path))
    match(predictions,'predictions_sha256')
    if intervals is not None:match(intervals,'intervals_sha256')
    return lock

def prediction_frame(path,methods,intervals=None):
    frame=pd.read_csv(path,dtype={'ubigeo':str})
    require=['ubigeo','method','p_anemia','p_stunting']
    if not set(require).issubset(frame.columns):raise ValueError('Missing prediction columns '+str(require))
    frame.ubigeo=frame.ubigeo.str.zfill(6)
    if frame[['ubigeo','method']].duplicated().any():raise ValueError('Duplicate prediction method/district')
    if not set(methods).issubset(set(frame.method)):raise ValueError('Comparator family absent')
    # Extra unselected candidates may be present but are never silently scored.
    frame=frame.loc[frame.method.isin(methods)].copy()
    aux=pd.read_csv(ROOT/'datos/aux_sikov14_official.csv',dtype={'ubigeo':str})
    support=set(aux.ubigeo)
    for method in methods:
        f=frame[frame.method.eq(method)]
        if len(f)!=1874 or set(f.ubigeo)!=support:raise ValueError('Each method must supply the complete identical1874-district frame; no intersection allowed')
        if not np.isfinite(f[['p_anemia','p_stunting']].to_numpy(float)).all():raise ValueError('Nonfinite prediction; comparison cannot drop a failing method/domain')
    if intervals is not None:
        with np.load(intervals,allow_pickle=False)as z:
            columns=['ubigeo','method']+[s+'_'+e for e in ENDPOINTS for s in ['lo95','hi95']]
            if not set(columns).issubset(z.files):raise ValueError('Intervals NPZ needs same district/method and four named limit columns')
            iv=pd.DataFrame({key:z[key]for key in columns})
        iv.ubigeo=iv.ubigeo.astype(str).str.zfill(6)
        frame=frame.merge(iv,on=['ubigeo','method'],how='left',validate='one_to_one')
    for endpoint in ENDPOINTS:
        columns=['lo95_'+endpoint,'hi95_'+endpoint]
        if any(c in frame for c in columns)and not all(c in frame for c in columns):raise ValueError('Incomplete interval pair')
        if all(c in frame for c in columns):
            if not np.isfinite(frame[columns].to_numpy(float)).all()or (frame[columns[0]]>frame[columns[1]]).any():raise ValueError('Invalid supplied interval limits')
    return frame

def direct_measurements(path,scope):
    # This is the first response-value read, after every prediction lock check.
    data=pd.read_csv(path,dtype={'ubigeo':str,'psu_key':str})
    if not data.year.eq(2024).all()or not data.de_facto.eq(1).all()or not data.age_months.between(6,59).all():
        raise ValueError('Different evaluation year/age/residence universe')
    data.ubigeo=data.ubigeo.str.zfill(6)
    held=data.ubigeo.str[:4].map(lambda p:int(hashlib.sha256(('ENDES-phase2-20261007|'+p).encode()).hexdigest()[:8],16)%5==0)
    if scope=='territorial'and not held.all():raise RuntimeError('Nonreserved province entered territorial scoring')
    rows=[]
    for endpoint,column in [('anemia','y_anemia_old'),('stunting','y_stunting')]:
        valid=data[column].isin([0,1])&data.weight.gt(0)&np.isfinite(data.weight)
        u=data.loc[valid];total=u.weight.groupby(u.ubigeo).sum()
        mean=(u.weight*u[column]).groupby(u.ubigeo).sum()/total
        count=u.groupby('ubigeo').size()
        for code in sorted(data.ubigeo.unique()):
            rows.append({'ubigeo':code,'province':code[:4],'outcome':endpoint,'direct':float(mean.get(code,np.nan)),
                         'weighted_observed_denominator':float(total.get(code,0)),'valid_positive_weight_records':int(count.get(code,0))})
    return pd.DataFrame(rows)

def balanced_pair(candidate,reference,provinces,rng):
    """Cluster ratio sandwich for .5mean(delta_anemia)+.5mean(delta_stunting)."""
    delta=np.asarray(candidate,float)-np.asarray(reference,float)
    if delta.ndim!=2 or delta.shape[1]!=2:raise ValueError('Two endpoint columns required')
    if np.isinf(delta).any():raise ValueError('Infinite score difference')
    labels,inverse=np.unique(provinces,return_inverse=True);J=len(labels)
    if J<2:raise ValueError('At least2 provinces needed')
    observed=np.isfinite(delta);counts=np.zeros((J,2));sums=np.zeros((J,2))
    for j in range(2):
        counts[:,j]=np.bincount(inverse,weights=observed[:,j].astype(float),minlength=J)
        sums[:,j]=np.bincount(inverse,weights=np.nan_to_num(delta[:,j],nan=0),minlength=J)
    totals=counts.sum(axis=0)
    if (totals==0).any():raise ValueError('No evaluable domains for an endpoint')
    endpoint_mean=sums.sum(axis=0)/totals;difference=float(endpoint_mean.mean())
    influence=.5*np.sum((sums-counts*endpoint_mean)/totals,axis=1)
    se=float(np.sqrt(J/(J-1)*np.dot(influence,influence)))
    stat=difference/se if se>0 else (0 if difference==0 else np.copysign(np.inf,difference))
    pvalue=float(2*t.sf(abs(stat),J-1));critical=float(t.ppf(.975,J-1))
    indices=rng.integers(0,J,size=(BOOTSTRAPS,J))
    den=counts[indices].sum(axis=1)
    if (den==0).any():raise ValueError('Resampled endpoint denominator vanished; cannot silently drop replicates')
    boot=(sums[indices].sum(axis=1)/den).mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
    return {'difference':difference,'endpoint_difference_anemia':float(endpoint_mean[0]),'endpoint_difference_stunting':float(endpoint_mean[1]),
            'cluster_standard_error':se,'province_clusters':J,'t_df':J-1,'p_two_sided':pvalue,
            'unadjusted_t_ci_low':difference-critical*se,'unadjusted_t_ci_high':difference+critical*se,
            'descriptive_bootstrap_ci_low':float(lo),'descriptive_bootstrap_ci_high':float(hi),'bootstrap_repetitions':BOOTSTRAPS,
            'evaluated_anemia_domains':int(totals[0]),'evaluated_stunting_domains':int(totals[1])}

def run(args):
    methods=[args.candidate,args.reference_fh,args.reference_sar]
    if len(set(methods))!=3:raise ValueError('Candidate/FH/SAR must be three distinct methods')
    lock=verify_lock(args.lock,args.protocol,args.predictions,args.intervals)
    pred=prediction_frame(args.predictions,methods,args.intervals)
    response=OLD/('datos/private_external/children_2024_scoring_only.csv'if args.scope=='temporal'else'datos/private_external/children_2024_test.csv')
    target=direct_measurements(response,args.scope)
    frame=set(pred.ubigeo);excluded=[];detail=[];summary=[]
    for endpoint in ENDPOINTS:
        td=target.loc[target.outcome.eq(endpoint)].copy()
        supported=td.ubigeo.isin(frame);observed=td.direct.notna()&td.weighted_observed_denominator.gt(0)
        for row in td.loc[~supported|~observed].to_dict('records'):
            excluded.append({**row,'reason':'unsupported_auxiliary_geography'if row['ubigeo']not in frame else'no_valid_positive_weight_outcome'})
        td=td.loc[supported&observed]
        f=pred.merge(td,on='ubigeo',validate='many_to_one')
        p=f['p_'+endpoint].to_numpy(float);y=f.direct.to_numpy(float);q=np.clip(p,EPS,1-EPS);bounded=np.clip(p,0,1)
        f['MSE']=(p-y)**2;f['NLL']=-(y*np.log(q)+(1-y)*np.log1p(-q))
        f['Brier']=y*(1-p)**2+(1-y)*p**2;f['MSE_bounded_sensitivity']=(bounded-y)**2
        f['outside_probability_range']=(p<0)|(p>1);f['NLL_clipped']=(p<EPS)|(p>1-EPS)
        if 'lo95_'+endpoint in f:
            f['noisy_direct_in_interval']=(f.direct>=f['lo95_'+endpoint])&(f.direct<=f['hi95_'+endpoint])
            f['interval_width_pp']=100*(f['hi95_'+endpoint]-f['lo95_'+endpoint])
        detail.append(f)
        for method in methods:
            part=f[f.method.eq(method)]
            row={'scope':args.scope,'method':method,'outcome':endpoint,'districts':len(part),'provinces':part.province.nunique(),
                 'RMSE_noisy_direct_pp':100*np.sqrt(part.MSE.mean()),'MSE':part.MSE.mean(),'Brier':part.Brier.mean(),'NLL':part.NLL.mean(),
                 'RMSE_bounded_sensitivity_pp':100*np.sqrt(part.MSE_bounded_sensitivity.mean()),
                 'outside_probability_range_count':int(part.outside_probability_range.sum()),'NLL_clipped_count':int(part.NLL_clipped.sum()),
                 'NLL_clipping_epsilon':EPS}
            if 'noisy_direct_in_interval'in part:
                row['noisy_direct_interval_containment_descriptive']=float(part.noisy_direct_in_interval.mean())
                row['mean_interval_width_pp']=float(part.interval_width_pp.mean())
                row['latent_interval_coverage_identified']=False
            summary.append(row)
    detail=pd.concat(detail,ignore_index=True)
    all_codes=sorted(detail.ubigeo.unique());provinces=np.array([c[:4]for c in all_codes]);contrasts=[]
    rng=np.random.default_rng(202610102024+(10000 if args.scope=='territorial'else 0))
    for metric in ['MSE','NLL']:
        candidate=detail.loc[detail.method.eq(args.candidate)].pivot(index='ubigeo',columns='outcome',values=metric).reindex(index=all_codes,columns=ENDPOINTS).to_numpy(float)
        for reference in [args.reference_fh,args.reference_sar]:
            ref=detail.loc[detail.method.eq(reference)].pivot(index='ubigeo',columns='outcome',values=metric).reindex(index=all_codes,columns=ENDPOINTS).to_numpy(float)
            value=balanced_pair(candidate,ref,provinces,rng)
            contrasts.append({'scope':args.scope,'candidate':args.candidate,'reference':reference,'metric':metric,**value})
    args.output.mkdir(parents=True,exist_ok=False)
    detail.to_csv(args.output/'district_scores.csv',index=False)
    pd.DataFrame(summary).to_csv(args.output/'endpoint_summary.csv',index=False)
    pd.DataFrame(contrasts).to_csv(args.output/'paired_balanced_contrasts_unadjusted.csv',index=False)
    pd.DataFrame(excluded).to_csv(args.output/'excluded_domains.csv',index=False)
    report={'status':'PASS_SECONDARY_REUSED2024_SCORING','scope':args.scope,'candidate':args.candidate,'references':[args.reference_fh,args.reference_sar],
            'scoring_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),'protocol_sha256':sha(args.protocol),
            'prediction_lock_sha256':sha(args.lock),'predictions_sha256':sha(args.predictions),'private_response_sha256':sha(response),
            'methods_supplied_same1874domain_frame':True,'no_favourable_prediction_intersection':True,'anemia_definition':'WHO2011 HC56<110, same as2019training',
            'universe':'de facto children6to59months; outcome-specific available-case directweightedratios',
            'balanced_endpoint_scores':'.5mean_district(anemia)+.5mean_district(stunting), preserving endpoint covariance with provincecluster sandwich/bootstrap',
            'inference':'two-sided Studentt provincecluster sandwich; descriptive3999 pairedprovince percentilebootstrap; Holm8 collection required',
            'prior2024_evaluation_observed_in_previous_phase':True,'independent_new_validation_claim_permitted':False,'latent_prevalence_or_coverage_revealed':False,
            'raw_unbounded_gaussian_predictions_score_primary_RMSE_and_Brier':True,'shared_NLL_clipping_epsilon':EPS,
            'clipping_not_hidden':True,'bounded_RMSE_sensitivity_separate':True,'contrasts':contrasts,'endpoint_summary':summary,
            'excluded_domain_endpoint_count':len(excluded),'private_unit_records_exported':False,'Holm8_pending_other_scope':True}
    (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:report[k]for k in ['status','scope','independent_new_validation_claim_permitted','Holm8_pending_other_scope']}),flush=True)

def collect(paths,output):
    reports=[json.loads(p.read_text())for p in paths]
    if len(reports)!=2 or {r['scope']for r in reports}!={'temporal','territorial'}:raise ValueError('Exactly both scopes needed')
    if len({r['protocol_sha256']for r in reports})!=1 or len({r['candidate']for r in reports})!=1 or len({tuple(r['references'])for r in reports})!=1:
        raise ValueError('Comparator family or protocol changed across scopes')
    rows=[c for r in reports for c in r['contrasts']];f=pd.DataFrame(rows)
    if len(f)!=8 or f[['scope','reference','metric']].duplicated().any():raise ValueError('Complete unique Holm8 family needed')
    p=f.p_two_sided.to_numpy(float)
    if not np.isfinite(p).all()or ((p<0)|(p>1)).any():raise ValueError('Invalid p-values')
    order=np.argsort(p,kind='stable');a=np.empty(8);a[order]=np.minimum(1,np.maximum.accumulate(p[order]*np.arange(8,0,-1)))
    f['p_Holm8']=a;f['classification']=np.where(a>=.05,'difference_not_detected',np.where(f.difference<0,'candidate_improvement','candidate_deterioration'))
    output.mkdir(parents=True,exist_ok=False);f.to_csv(output/'paired_balanced_contrasts_Holm8.csv',index=False)
    endpoint=pd.DataFrame([s for r in reports for s in r['endpoint_summary']]);endpoint.to_csv(output/'endpoint_summary_both_scopes.csv',index=False)
    report={'status':'PASS_COMPLETE_SECONDARY_HOLM8_FAMILY','family':'two scopes times FH/SARreferences times balancedMSE/NLL metrics',
            'protocol_sha256':reports[0]['protocol_sha256'],'candidate':reports[0]['candidate'],'classification_counts':f.classification.value_counts().to_dict(),
            'primary_scientific_success_not_decided_by_this_reused_secondary_dataset':True,'latent_coverage_claim_permitted':False,
            'input_sha256':{str(p):sha(p)for p in paths},'source_sha256':sha(Path(__file__))}
    (output/'external_final_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['temporal','territorial']);parser.add_argument('--predictions',type=Path)
    parser.add_argument('--lock',type=Path);parser.add_argument('--protocol',type=Path);parser.add_argument('--intervals',type=Path)
    parser.add_argument('--candidate',default='ICC');parser.add_argument('--reference-fh',default='SikovFH');parser.add_argument('--reference-sar',default='SikovSAR')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--collect',nargs=2,type=Path)
    args=parser.parse_args()
    if args.collect:collect(args.collect,args.output)
    elif not all([args.scope,args.predictions,args.lock,args.protocol]):parser.error('scope/predictions/lock/protocol required before scoring')
    else:run(args)
