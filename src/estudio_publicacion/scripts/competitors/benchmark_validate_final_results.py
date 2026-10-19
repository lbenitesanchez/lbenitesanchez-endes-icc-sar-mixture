#!/usr/bin/env python3
"""Readiness checks for the locked independent benchmark, after consolidation.

These checks do not fit, select or modify statistical models. Negative method
comparisons pass; unavailable populations/inference remain explicit failures.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from benchmark_inputs import OUT,sha256
from benchmark_finalize import PRIMARY,STRUCTURAL,PRIMARY_METHODS,CANDIDATE,REFERENCES
from benchmark_primary_score import primary_contrast,holm


def check(condition,message,checks):
    if not bool(condition):raise AssertionError(message)
    checks.append(message)


def validate():
    final=OUT/'final_evaluation';done=json.loads((final/'final_completion.json').read_text())
    protocol=json.loads((OUT/'protocol_freeze.json').read_text())
    contract=json.loads((OUT/'final_result_contract.json').read_text())
    amendment=json.loads((OUT/'numerical_amendment_v2.json').read_text())
    checks=[]
    check(done['evaluation_completed'],'Final evaluator completed',checks)
    check(contract['final_tags'][:2]==['final_primary_v2','final_structural_v2'],'Only candidate v2 is canonical',checks)
    check(amendment['approved_for_dispatch'],'Numerical amendment approved',checks)
    check(done['protocol_sha256']==sha256(OUT/'protocol_freeze.json'),'Final protocol checksum matches',checks)
    for name,expected in done['output_sha256'].items():
        check(sha256(final/name)==expected,'Output checksum matches: '+name,checks)
    data=pd.read_csv(final/'combined_replications.csv')
    cells=['scenario','replication','method','protocol','group','outcome']
    check(not data.duplicated(cells).any(),'No duplicated evaluation cells',checks)
    planned={(scenario,int(rep))for scenario,reps in protocol['independent_seed_ids'].items()for rep in reps}
    observed=set(map(tuple,data[['scenario','replication']].drop_duplicates().to_numpy()))
    check(len(planned)==800 and planned==observed,'Exactly800 locked populations, no new or replacement worlds',checks)
    check(not data.run_tag.isin(['final_primary_v1']).any(),'Invalidated v1 rows excluded',checks)
    for tag in contract['final_tags']:
        folder=OUT/tag
        check(not (folder/'invalidated.json').exists(),'Canonical source not invalidated: '+tag,checks)
        check((folder/'completion.json').exists(),'Canonical source completed: '+tag,checks)
    expected=[]
    for scenario in PRIMARY+STRUCTURAL:
        count=200 if scenario in PRIMARY else 100
        ids=set(protocol['independent_seed_ids'][scenario])
        for method in PRIMARY_METHODS:
            for outcome in ['anemia','stunting']:
                part=data[(data.scenario==scenario)&(data.method==method)&
                    (data.group=='all')&(data.outcome==outcome)]
                check(len(part)==count and set(part.replication)==ids,
                    f'Full requested cohort {scenario}/{method}/{outcome}',checks)
                check(bool((part.areas==80).all()) and bool((part.areas_requested==80).all()),
                    f'Complete80-area target {scenario}/{method}/{outcome}',checks)
                expected.append(dict(scenario=scenario,method=method,outcome=outcome,
                    requested_replications=count,available_replications=len(part),
                    converged_replications=int(part.converged.sum()),
                    finite_nonconverged_retained=int((~part.converged).sum())))
    secondary=[]
    for method,scenario,ids in [('GaoAuthorStrictFullH','binary_informative_mar',set(range(10000,10200))),
        ('HoggTSLNFullH','binary_informative_mar',set(range(10000,10200))),
        ('HoggDesignPsiPSU','sparse_psu_2025',set(range(10000,10100)))]:
        ledger_tag='final_gao_evaluation_v1'if method.startswith('Gao')else'final_hogg_evaluation_v1'
        ledger=pd.DataFrame(json.loads((OUT/ledger_tag/'fits.json').read_text()))
        requested=ledger[(ledger.method==method)&(ledger.scenario==scenario)]
        check(len(requested)==len(ids) and set(requested.replication)==ids,
            'Additional arm exact requested cohort in failure-aware ledger: '+method,checks)
        for outcome in ['anemia','stunting']:
            part=data[(data.scenario==scenario)&(data.method==method)&
                (data.group=='all')&(data.outcome==outcome)]
            check(set(part.replication).issubset(ids) and bool((part.areas==80).all()),
                f'Additional arm uses planned worlds and80-area target {method}/{outcome}',checks)
            secondary.append(dict(method=method,scenario=scenario,outcome=outcome,
                requested_replications=len(ids),available_prediction_replications=len(part),
                unavailable_prediction_replications=len(ids)-len(part),
                all_requested_predictions_available=len(part)==len(ids)))
    primary=pd.read_csv(final/'primary_publication_table.csv')
    check(len(primary)==48 and set(primary.method)==set(PRIMARY_METHODS) and
        bool((primary.replications==200).all()) and bool((primary.mean_areas==80).all()),
        'Primary table48 all-area cells, each R200',checks)
    check(data[data.method.str.startswith('Direct')].group.eq('observed_endpoint').all(),
        'Direct controls evaluated only where observed',checks)
    scores=pd.read_csv(final/'primary_balanced_interval_score.csv')
    check(len(scores)==3 and set(scores.reference)==set(REFERENCES) and scores.primary_test_available.all(),
        'Exactly3 locked primary comparisons available',checks)
    recomputed=[]
    for reference in REFERENCES:
        score,paired=primary_contrast(data,CANDIDATE,reference,200)
        check(score['independent_paired_populations']==600 and len(paired)==600,
            'Primary pairing600 populations: '+reference,checks)
        saved=scores[scores.reference==reference].iloc[0]
        for key in ['interval_score_change','paired_mcse','change_ci95_lower','change_ci95_upper','p_two_sided']:
            np.testing.assert_allclose(float(saved[key]),float(score[key]),rtol=1e-11,atol=1e-13)
        recomputed.append(score)
    recomputed=pd.DataFrame(recomputed);recomputed['p_Holm3']=holm(recomputed.p_two_sided)
    for _,row in recomputed.iterrows():
        np.testing.assert_allclose(scores.loc[scores.reference==row.reference,'p_Holm3'],row.p_Holm3,rtol=1e-11,atol=1e-13)
    checks.append('Primary score/paired MCSE/CI/Holm3 reproduce from locked rows')
    pairs=pd.read_csv(final/'paired_contrasts.csv')
    for method,scenario,count in [('GaoAuthorStrictFullH','binary_informative_mar',200),
        ('HoggTSLNFullH','binary_informative_mar',200),('HoggDesignPsiPSU','sparse_psu_2025',100)]:
        part=pairs[(pairs.contrast==CANDIDATE+'_minus_'+method)&(pairs.scenario==scenario)&(pairs.group=='all')]
        for outcome in ['anemia','stunting']:
            available=next(row['available_prediction_replications']for row in secondary
                if row['method']==method and row['outcome']==outcome)
            cells=part[part.outcome==outcome]
            check((len(cells)==1 and int(cells.paired_replications.iloc[0])==available)if available else len(cells)==0,
                'Additional paired count equals available requested predictions: '+method+'/'+outcome,checks)
    spatial=OUT/'spatial_evaluation/completion.json'
    spatial_ready=None
    if spatial.exists():
        sd=json.loads(spatial.read_text());spatial_ready=bool(sd.get('evaluation_completed') and
            sd.get('all_requested_populations_represented') and sd.get('all_requested_candidate_pairs_available'))
        check(spatial_ready,'Separate spatial100 and candidate100 pairs available',checks)
        sp=pd.read_csv(spatial.parent/'paired_contrasts.csv');part=sp[sp.group=='all']
        check(len(part)==2 and bool((part.paired_replications==100).all()) and
            set(part.contrast)=={CANDIDATE+'_minus_HoggSpatial'},'Spatial secondary contrast exact100 pairs',checks)
    return dict(passed=True,finished_utc=datetime.now(timezone.utc).isoformat(),
        checks=checks,unique_populations=800,primary_cells=48,primary_Holm_family=3,
        primary_paired_populations_per_comparator=600,spatial_secondary_ready=spatial_ready,
        finite_nonconverged_predictions_are_retained=True,negative_comparative_results_are_not_failures=True,
        secondary_requested_and_available=secondary,
        secondary_unavailable_predictions_do_not_change_primary_cohort=True,
        endpoint_prediction_accounting=expected,
        protocol_sha256=sha256(OUT/'protocol_freeze.json'),
        numerical_amendment_sha256=sha256(OUT/'numerical_amendment_v2.json'),
        final_completion_sha256=sha256(final/'final_completion.json'),validator_sha256=sha256(Path(__file__)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');p.add_argument('--poll-seconds',type=int,default=30)
    args=p.parse_args()
    if args.poll_seconds<10:raise ValueError('Use a low-frequency observer')
    completion=OUT/'final_evaluation/final_completion.json'
    while args.wait and not completion.exists():time.sleep(args.poll_seconds)
    destination=OUT/'validation/final_result_readiness.json';destination.parent.mkdir(parents=True,exist_ok=True)
    try:proof=validate()
    except Exception as error:
        proof=dict(passed=False,error_type=type(error).__name__,error=str(error),
            checked_utc=datetime.now(timezone.utc).isoformat(),validator_sha256=sha256(Path(__file__)),
            no_fit_or_statistical_configuration_modified=True)
        destination.write_text(json.dumps(proof,indent=2));raise
    destination.write_text(json.dumps(proof,indent=2))
    print(json.dumps({k:v for k,v in proof.items()if k not in ['checks','endpoint_prediction_accounting']},indent=2))


if __name__=='__main__':main()
