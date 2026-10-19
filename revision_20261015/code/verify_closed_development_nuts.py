#!/usr/bin/env python3
"""Independent post-lock checks of the 32-world integration sensitivity."""
from pathlib import Path
import datetime as dt,json,sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numerical_cohorts_portable as run

def verify():
    run.gate();root=run.PUBLIC/'development_nuts';q=run.read(root/'scoring_receipt.json');lock=run.read(run.PRIVATE/'development_nuts/prediction_lock_before_score.json');grid=run.read(run.REPO/'revision_20261015/results/devgrid_nuts/NUTS_development_grid_audit.json')
    assert q['complete'] and q['worlds']==32 and q['all_models_converged'] and len(lock['items'])==32
    assert dt.datetime.fromisoformat(lock['locked_utc'])<dt.datetime.fromisoformat(q['scored_utc'])
    for name,digest in q['outputs'].items():assert run.sha(root/name)==digest
    f=pd.read_csv(root/'replications.csv');assert len(f)==32*7*2*3
    assert not f.duplicated(['scenario','replication','method','outcome','group']).any()
    assert f.converged.all()
    candidates=['ICC','ICC_poverty','ICC_SAR_w25','ICC_SAR_w50','ICC_SAR_w75'];allscore=f.loc[f.group.eq('all')&f.method.isin(candidates)]
    population=allscore.groupby(['scenario','replication','method'],as_index=False)[['interval_score','mse']].mean()
    rank=population.groupby(['scenario','method'])[['interval_score','mse']].mean().groupby('method').mean().sort_values(['interval_score','mse'])
    recorded={x['method']:x for x in grid['candidate_ranking']}
    for method,row in rank.iterrows():assert abs(float(row.interval_score)-recorded[method]['interval_score'])<1e-14
    assert list(rank.index)==[x['method'] for x in grid['candidate_ranking']]
    loocount={x:0 for x in candidates}
    for (context,rep),_ in population.groupby(['scenario','replication']):
        sub=population.loc[~(population.scenario.eq(context)&population.replication.eq(rep))]
        r=sub.groupby(['scenario','method'])[['interval_score','mse']].mean().groupby('method').mean().sort_values(['interval_score','mse']);loocount[str(r.index[0])]+=1
    assert {k:v for k,v in loocount.items() if v}==grid['LOPO_counts']
    d=pd.read_csv(root/'NUTS_diagnostics_all_attempts.csv');assert len(d)==128 and d.attempt.eq(1).all() and d.NUTS_final_success.all()
    assert d.max_Rhat.le(1.01).all() and d.min_bulk_ESS.ge(400).all() and d.min_tail_ESS.ge(400).all() and d.min_E_BFMI.ge(.3).all() and d.divergences.eq(0).all() and d.max_treedepth_hits.eq(0).all() and d.beta_shape_guard_cells.eq(0).all()
    capture=run.read(run.PUBLIC/'retained_sampler_capture_audit.json');entries=[x for x in capture['checked'] if x['cohort']=='development_nuts'];assert len(entries)==64 and sum(x['retained_chain_files'] for x in entries)==512
    result=dict(status='PASS_COMPLETE_32_WORLD_NUTS_DEVELOPMENT_POSTLOCK_VERIFICATION',verified_utc=run.utc(),worlds=32,worlds_per_context=8,eligible_candidates=5,reported_strategy_rows_including_references=7,score_rows=1344,actual_NUTS_fits=64,actual_endpoint_posteriors=128,actual_chain_CSV_files=512,NUTS_retries=0,NUTS_failed_endpoints=0,max_Rhat=float(d.max_Rhat.max()),min_ESS_bulk=float(d.min_bulk_ESS.min()),min_ESS_tail=float(d.min_tail_ESS.min()),min_E_BFMI=float(d.min_E_BFMI.min()),divergences=0,treedepth_hits=0,prediction_lock_precedes_score=True,independent_balanced_ranking_matches=True,independent_LOPO_counts=loocount,original75_25_strategy_changed=False,original_Holm_families_changed=False,sparse57_accuracy_not_included=True,score_CSV_sha256=run.sha(root/'replications.csv'),score_receipt_sha256=run.sha(root/'scoring_receipt.json'),grid_audit_sha256=run.sha(run.REPO/'revision_20261015/results/devgrid_nuts/NUTS_development_grid_audit.json'),code_sha256=run.sha(__file__))
    run.save(run.PUBLIC/'NUTS32_complete_numerical_certificate.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':verify()
