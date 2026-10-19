"""Recalculate the additive four-method Holm12 external CSVs independently."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(scopes,collected,output):
    checks=[];records=[];reports=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    for path in scopes:
        report=json.loads((path/'summary.json').read_text());reports.append(report)
        frame=pd.read_csv(path/'district_scores.csv',dtype={'ubigeo':str,'province':str})
        summary=pd.read_csv(path/'endpoint_summary.csv')
        raw=pd.read_csv(path/'paired_balanced_contrasts_unadjusted.csv')
        methods=[report['candidate']]+report['references'];scope=report['scope']
        check(scope+' same four comparator methods',set(frame.method)==set(methods))
        check(scope+' every scored domain/outcome has all methods',frame.groupby(['ubigeo','outcome']).method.nunique().eq(4).all())
        check(scope+' each method/outcome/domain unique',not frame[['method','ubigeo','outcome']].duplicated().any())
        check(scope+' responses are valid weighted proportions',frame.direct.between(0,1).all()and frame.weighted_observed_denominator.gt(0).all())
        check(scope+' futureclinicaldefinition notmixed with old',report['anemia_definition']=='WHO2011 HC56<110, same as2019training')
        check(scope+' previous2024 reuse and noisy-direct targets disclosed',report['prior2024_evaluation_observed_in_previous_phase']and not report['independent_new_validation_claim_permitted']and not report['latent_prevalence_or_coverage_revealed'])
        for row in summary.to_dict('records'):
            part=frame.loc[frame.method.eq(row['method'])&frame.outcome.eq(row['outcome'])]
            p=part['p_'+row['outcome']].to_numpy(float);y=part.direct.to_numpy(float)
            mse=(p-y)**2;brier=y*(1-p)**2+(1-y)*p**2;q=np.minimum(np.maximum(p,1e-12),1-1e-12)
            nll=-(y*np.log(q)+(1-y)*np.log1p(-q))
            stem=scope+' '+row['method']+' '+row['outcome']
            check(stem+' original unbounded RMSE oracle',np.isclose(100*np.sqrt(np.mean(mse)),row['RMSE_noisy_direct_pp'],atol=1e-10,rtol=0))
            check(stem+' binary Brier oracle',np.isclose(np.mean(brier),row['Brier'],atol=1e-12,rtol=0))
            check(stem+' binary NLL oracle',np.isclose(np.mean(nll),row['NLL'],atol=1e-12,rtol=0))
            check(stem+' bounded RMSE separately recalculated',np.isclose(100*np.sqrt(np.mean((np.clip(p,0,1)-y)**2)),row['RMSE_bounded_sensitivity_pp'],atol=1e-10,rtol=0))
            check(stem+' outofrange and epsilon clippingcount',int(((p<0)|(p>1)).sum())==row['outside_probability_range_count']and int(((p<1e-12)|(p>1-1e-12)).sum())==row['NLL_clipped_count'])
            if 'noisy_direct_interval_containment_descriptive'in row:
                lo=part['lo95_'+row['outcome']].to_numpy(float);hi=part['hi95_'+row['outcome']].to_numpy(float)
                check(stem+' containment targets noisy direct',np.isclose(np.mean((y>=lo)&(y<=hi)),row['noisy_direct_interval_containment_descriptive'],atol=1e-12,rtol=0))
                check(stem+' confidence/credible width reported',np.isclose(100*np.mean(hi-lo),row['mean_interval_width_pp'],atol=1e-10,rtol=0))
        rng=np.random.default_rng(202610102024+(10000 if scope=='territorial'else 0))
        for row in raw.to_dict('records'):
            candidate=frame[frame.method.eq(row['candidate'])][['ubigeo','province','outcome',row['metric']]]
            reference=frame[frame.method.eq(row['reference'])][['ubigeo','outcome',row['metric']]]
            check(scope+' '+row['reference']+' '+row['metric']+' complete paired rows',set(map(tuple,candidate[['ubigeo','outcome']].to_numpy()))==set(map(tuple,reference[['ubigeo','outcome']].to_numpy())))
            paired=candidate.merge(reference,on=['ubigeo','outcome'],validate='one_to_one',suffixes=('_candidate','_reference'))
            paired['delta']=paired[row['metric']+'_candidate']-paired[row['metric']+'_reference']
            means=paired.groupby('outcome').delta.mean();n=paired.groupby('outcome').size();labels=sorted(paired.province.unique());J=len(labels)
            sums=np.zeros((J,2));counts=np.zeros((J,2));influences=np.zeros(J)
            for k,province in enumerate(labels):
                for j,endpoint in enumerate(['anemia','stunting']):
                    v=paired.loc[paired.province.eq(province)&paired.outcome.eq(endpoint),'delta']
                    sums[k,j]=v.sum();counts[k,j]=len(v)
                    influences[k]+=.5*float(np.sum(v-means[endpoint]))/n[endpoint]
            difference=float(.5*(means['anemia']+means['stunting']))
            se=float(np.sqrt(J/(J-1)*np.sum(influences**2)))
            statistic=difference/se if se else (0 if difference==0 else np.copysign(np.inf,difference))
            pvalue=float(2*t.sf(abs(statistic),J-1))
            indices=rng.integers(0,J,size=(3999,J));boot=(sums[indices].sum(axis=1)/counts[indices].sum(axis=1)).mean(axis=1)
            lo,hi=np.quantile(boot,[.025,.975]);stem=scope+' '+row['reference']+' '+row['metric']
            check(stem+' balanced difference oracle',np.isclose(difference,row['difference'],atol=1e-12,rtol=0))
            check(stem+' jointendpoint province sandwich oracle',np.isclose(se,row['cluster_standard_error'],atol=1e-12,rtol=0))
            check(stem+' two-sided Studentt oracle',np.isclose(pvalue,row['p_two_sided'],atol=1e-10,rtol=0))
            check(stem+' exact3999 jointprovince bootstrap',np.isclose(lo,row['descriptive_bootstrap_ci_low'],atol=1e-10,rtol=0)and np.isclose(hi,row['descriptive_bootstrap_ci_high'],atol=1e-10,rtol=0))
            records.append({'scope':scope,'reference':row['reference'],'metric':row['metric'],'recomputed_difference':difference,'recomputed_p':pvalue})
        excluded_path=path/'excluded_domains.csv'
        try:excluded=pd.read_csv(excluded_path,dtype={'ubigeo':str})
        except pd.errors.EmptyDataError:excluded=pd.DataFrame(columns=['ubigeo','outcome','reason'])
        check(scope+' exclusions agree with count',len(excluded)==report['excluded_domain_endpoint_count'])
        check(scope+' exclusion reasons prespecified',set(excluded.reason).issubset({'unsupported_auxiliary_geography','no_valid_positive_weight_outcome'}))
        check(scope+' excluded domain/endpoints never included',not set(map(tuple,excluded[['ubigeo','outcome']].to_numpy()))&set(map(tuple,frame[['ubigeo','outcome']].to_numpy())))
    family=pd.read_csv(collected/'paired_balanced_contrasts_Holm12.csv')
    check('complete unique family two scopes three references two metrics',len(family)==12 and set(family.scope)=={'temporal','territorial'}and not family[['scope','reference','metric']].duplicated().any())
    p=family.p_two_sided.to_numpy(float);order=np.argsort(p);a=np.empty(12)
    # Direct definition of Holm adjusted p: running maximum of descending
    # multiplicity weights applied to ordered original p-values.
    for k,position in enumerate(order):a[position]=min(1,max((12-j)*p[order[j]]for j in range(k+1)))
    check('Holm12 independent directdefinition oracle',np.allclose(a,family.p_Holm12,atol=1e-12,rtol=0))
    expected=np.where(a>=.05,'difference_not_detected',np.where(family.difference<0,'candidate_improvement','candidate_deterioration'))
    check('improvement/deterioration classification reproduces signs and Holm',np.array_equal(expected,family.classification))
    report={'status':'PASS_INDEPENDENT_EXTENDED_EXTERNAL_NUMERIC_AND_EXCLUSION_AUDIT','checks_passed':len(checks),'checks':checks,'recomputed_contrasts':records,
            'source_sha256':sha(Path(__file__)),'model_or_statistical_source_modified':False,'individual_test_records_reopened':False,
            'aggregated_scoring_CSVs_used_after_prediction_locks':True,'latent_coverage_claim_permitted':False,
            'native_csv_sha256':{str(path/file):sha(path/file)for path in scopes for file in ['district_scores.csv','endpoint_summary.csv','paired_balanced_contrasts_unadjusted.csv']}}
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:report[k]for k in ['status','checks_passed','model_or_statistical_source_modified']}),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scopes',nargs=2,type=Path,required=True);parser.add_argument('--collected',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.scopes,args.collected,args.output)
