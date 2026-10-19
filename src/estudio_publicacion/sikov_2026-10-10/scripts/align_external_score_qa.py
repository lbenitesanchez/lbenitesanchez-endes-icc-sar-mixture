"""Meaningful synthetic-only tests for locked secondary scoring/inference.

Never opens genuine2024 response files. Fake children are generated entirely
from fixed constants and public geographic codes, inside a temporary directory.
"""
from pathlib import Path
import tempfile,json,argparse,hashlib
import numpy as np
import pandas as pd
from scipy.stats import t
import align_external_score as score
ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name,condition):
    if not condition:raise AssertionError(name)
    checks.append(name)
def main():
    rng=np.random.default_rng(772)
    # Jointendpoint covariance is large: treating the two endpoint means as
    # independent understates the uncertainty. Verify against explicit cluster
    # influence arithmetic with different missingness denominators.
    delta=rng.normal(0,1,(24,2));delta[:,1]=delta[:,0]+rng.normal(0,.1,24)
    delta[1,1]=np.nan;delta[6,0]=np.nan
    province=np.repeat(['a','b','c','d','e','f'],4)
    result=score.balanced_pair(delta,np.zeros_like(delta),province,np.random.default_rng(22))
    mu=np.nanmean(delta,axis=0);J=6;influence=[]
    for label in np.unique(province):
        subset=delta[province==label]
        influence.append(sum(.5*np.nansum(subset[:,j]-mu[j])/np.isfinite(delta[:,j]).sum()for j in range(2)))
    oracle=float(np.sqrt(J/(J-1)*np.dot(influence,influence)))
    check('joint endpoint clustered SE matches independent influence oracle',abs(result['cluster_standard_error']-oracle)<1e-12)
    check('balanced endpoint difference respects missing denominators',abs(result['difference']-mu.mean())<1e-12)
    check('cluster Studentt pvalue independently reproduced',abs(result['p_two_sided']-2*t.sf(abs(mu.mean()/oracle),J-1))<1e-12)
    check('paired bootstraps exactly3999',result['bootstrap_repetitions']==3999)
    with tempfile.TemporaryDirectory(prefix='ENDES_synthetic_scoring_QA_')as temp:
        temp=Path(temp);score.OLD=temp/'fake_phase';(score.OLD/'datos/private_external').mkdir(parents=True)
        aux=pd.read_csv(ROOT/'datos/aux_sikov14_official.csv',dtype={'ubigeo':str})
        methods=['ICC','SikovFH','SikovSAR'];records=[];pred=[]
        for code in aux.ubigeo:
            for k in range(4):
                records.append({'year':2024,'ubigeo':code,'de_facto':1,'age_months':18,'weight':1,
                                'y_anemia_old':int(k==0),'y_stunting':int(k<2),'psu_key':'fake'})
            for method,p in zip(methods,[.28,.42,.47]):
                pred.append({'ubigeo':code,'method':method,'p_anemia':p,'p_stunting':p,
                             'lo95_anemia':p-.2,'hi95_anemia':p+.2,'lo95_stunting':p-.2,'hi95_stunting':p+.2})
        records=pd.DataFrame(records);prediction=pd.DataFrame(pred)
        # Honest raw-vs-bounded distinction and finite sharedNLL clipping.
        prediction.loc[1,'p_anemia']=1.25;prediction.loc[2,'p_stunting']=-.10
        pp=temp/'predictions.csv';prediction.to_csv(pp,index=False)
        protocol=temp/'protocol.json';protocol.write_text(json.dumps({'syntheticfixture':True}))
        lock=temp/'lock.json';lock.write_text(json.dumps({'approved_for_scoring':True,'protocol_sha256':score.sha(protocol),'predictions_sha256':score.sha(pp)}))
        bad=temp/'badlock.json';bad.write_text(json.dumps({'approved_for_scoring':True,'protocol_sha256':score.sha(protocol),'predictions_sha256':'0'*64}))
        try:score.verify_lock(bad,protocol,pp)
        except RuntimeError:check('incorrect prediction hash blocked before truth read',True)
        else:raise AssertionError('invalidlockaccepted')
        records.to_csv(score.OLD/'datos/private_external/children_2024_scoring_only.csv',index=False)
        held=records.ubigeo.str[:4].map(lambda p:int(hashlib.sha256(('ENDES-phase2-20261007|'+p).encode()).hexdigest()[:8],16)%5==0)
        records.loc[held].to_csv(score.OLD/'datos/private_external/children_2024_test.csv',index=False)
        reports=[]
        for scope in ['temporal','territorial']:
            output=temp/scope
            args=argparse.Namespace(scope=scope,predictions=pp,lock=lock,protocol=protocol,intervals=None,
                                    candidate='ICC',reference_fh='SikovFH',reference_sar='SikovSAR',output=output)
            score.run(args);report=json.loads((output/'summary.json').read_text());reports.append(output/'summary.json')
            check(scope+' complete four balanced contrasts',len(report['contrasts'])==4)
            check(scope+' explicitly secondary reused/no latent coverage',not report['independent_new_validation_claim_permitted']and not report['latent_prevalence_or_coverage_revealed'])
            summary=pd.read_csv(output/'endpoint_summary.csv')
            check(scope+' same endpoint support all methods',summary.groupby('outcome').districts.nunique().eq(1).all())
            if scope=='temporal':
                check('unbounded source predictions are counted and retained',summary.outside_probability_range_count.sum()==2)
                check('bounded RMSE is reported as separate sensitivity',(summary.RMSE_noisy_direct_pp>summary.RMSE_bounded_sensitivity_pp).any())
                check('finite common epsilon NLL with clippingcount',np.isfinite(summary.NLL).all()and summary.NLL_clipped_count.sum()==2)
                check('descriptive noisy-direct interval containment labelled',all('noisy_direct_interval_containment_descriptive'in s for s in report['endpoint_summary']))
        score.collect(reports,temp/'collected')
        family=pd.read_csv(temp/'collected/paired_balanced_contrasts_Holm8.csv')
        check('complete eight contrasts adjustment',len(family)==8 and np.isfinite(family.p_Holm8).all())
        check('Holm values never smaller than raw',np.all(family.p_Holm8>=family.p_two_sided-1e-15))
    report={'status':'PASS_SYNTHETIC_ONLY_EXTERNAL_SCORER_QA','checks_passed':len(checks),'checks':checks,
            'genuine2024_response_files_opened':False,'private_real_records_generated_or_exported':False,
            'scorer_sha256':score.sha(Path(score.__file__)),'test_source_sha256':score.sha(Path(__file__))}
    (ROOT/'validacion/align_external_score_synthetic_qa.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k]for k in ['status','checks_passed','genuine2024_response_files_opened']}),flush=True)
if __name__=='__main__':main()
