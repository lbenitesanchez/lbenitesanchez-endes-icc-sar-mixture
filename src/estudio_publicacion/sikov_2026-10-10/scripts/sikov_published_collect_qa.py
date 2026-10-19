"""Artificial paired-family and fail-closed joint-lock tests before freeze."""
import json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import sikov_published_collect as collector
from sikov_final_audit import independent_balanced
from sikov_published_fit import ROOT,sha

def main():
    rng=np.random.default_rng(20261010947);rows=[];references=('Sikov_FH','Sikov_SAR',collector.METHOD)
    for ci,context in enumerate(collector.CONTEXTS):
        for rep in collector.IDS:
            common=rng.normal();outcome=rng.uniform(.15,.3,2)
            for j,name in enumerate(('anemia','stunting')):
                for m,method in enumerate(('Selected',)+references):
                    value=outcome[j]+.003*m*(common+ci+j)
                    rows.append(dict(scenario=context,replication=rep,method=method,outcome=name,group='all',interval_score=value,mse=value,mae=value,coverage=value,width=value))
    frame=pd.DataFrame(rows);actual=collector.expanded_contrasts(frame);error=0.
    for reference in references:
        for row in independent_balanced(frame,'Selected',reference,collector.CONTEXTS,collector.IDS):
            compared=actual[(actual.reference==reference)&(actual.metric==row['metric'])].iloc[0]
            error=max(error,max(abs(row[k]-compared[k]) for k in ('difference','mcse','MC_CI95_low','MC_CI95_high','p_two_sided')))
    primary=actual[actual.metric=='interval_score'];p=primary.p_two_sided.to_numpy();expected=np.array([min(1,max((3-int((p<t).sum()))*t for t in p if t<=value)) for value in p]);holm_error=float(abs(expected-primary.p_Holm3.to_numpy()).max())
    checks={};oldroot=collector.ROOT;oldgate=collector.gate
    try:
        with tempfile.TemporaryDirectory(prefix='sikov_joint_gate_fixture_',dir=ROOT/'validacion') as tmp:
            root=Path(tmp);(root/'validacion').mkdir();collector.ROOT=root
            add={'base_protocol_sha256':'base_fixture'};collector.gate=lambda:(add,'add_fixture')
            original=root/'validacion/fresh_prediction_lock.json';published=root/'validacion/sikov_published_fresh_prediction_lock.json'
            original.write_text(json.dumps({'worlds':256}));published.write_text(json.dumps({'worlds':256}))
            joint={'approved_for_scoring':True,'protocol_sha256':'base_fixture','addendum_sha256':'add_fixture','heldout_outcomes_used':False,'fresh_cohort_size':256,'all_expected_worlds':True,'fresh_truth_read_before_joint':False,'heldout2024responses_used_before_joint':False,'fresh_original_prediction_lock_sha256':sha(original),'fresh_published_prediction_lock_sha256':sha(published),'files':[{'path':str(path),'sha256':sha(path)} for path in (original,published)]}
            target=root/'validacion/joint_prediction_lock.json';target.write_text(json.dumps(joint));collector.joint_gate();checks['complete_artificial_joint_contract_accepted']=True
            for key,value in [('approved_for_scoring',False),('all_expected_worlds',False),('fresh_cohort_size',255),('fresh_truth_read_before_joint',True),('heldout2024responses_used_before_joint',True),('fresh_published_prediction_lock_sha256','wrong')]:
                bad={**joint,key:value};target.write_text(json.dumps(bad))
                try:collector.joint_gate();checks['reject_'+key]=False
                except RuntimeError:checks['reject_'+key]=True
            target.write_text(json.dumps(joint));published.write_text(json.dumps({'worlds':255}))
            try:collector.joint_gate();checks['reject_changed_jointly_sealed_file']=False
            except RuntimeError:checks['reject_changed_jointly_sealed_file']=True
    finally:collector.ROOT=oldroot;collector.gate=oldgate
    checks.update(independent_correlated_endpoint_MCSE_matches=error<1e-12,independent_Holm3_matches=holm_error<1e-12,complete_primary_family_three_unique_refs=len(primary)==3 and set(primary.reference)==set(references))
    checks={key:bool(value) for key,value in checks.items()}
    result={'status':'PASS_ADDITIONAL_COLLECTOR_JOINT_GATE_AND_HOLM3_QA' if all(checks.values()) else 'FAIL','checks':checks,'paired_max_absolute_error':float(error),'Holm3_max_absolute_error':holm_error,'actual_training_or_heldout_values_read':False,'only_artificial_fixtures_used':True,'source_sha256':sha(__file__),'collector_sha256':sha(collector.__file__)}
    (ROOT/'validacion/sikov_published_collect_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if result['status']=='FAIL':raise RuntimeError('Additional collector QA failed')

if __name__=='__main__':main()
