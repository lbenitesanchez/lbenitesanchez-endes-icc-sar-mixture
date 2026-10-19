#!/usr/bin/env python3
"""Descriptive exact-boundary accounting after locked fits finish.

Finite population truth and exported prediction endpoints are compared exactly
to IEEE 0.0/1.0. Counts do not change scores, choose methods or define a new test.
They cannot identify the mathematical reason for a reported numerical endpoint.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from benchmark_inputs import OUT,sha256
from benchmark_collect import canonical_predictions

OUTCOMES=('anemia','stunting')


def masks(folder):
    truth=np.load(folder/'evaluation_only/truth.npz');direct=np.load(folder/'model_inputs/direct.npz')
    theta=truth['theta'];sampled=np.isin(np.arange(80),truth['sample_idx'])
    observed=np.zeros((80,2),bool);observed[direct['sample_idx']]=np.isfinite(direct['y'])
    return theta,{'all':np.ones((80,2),bool),
        'sampled_common':np.repeat(sampled[:,None],2,axis=1),
        'unsampled_common':np.repeat((~sampled)[:,None],2,axis=1),
        'observed_endpoint':observed}


def boundary_counts(theta,predictions,mask):
    p,lo,hi=[np.asarray(predictions[key])for key in ('mean','lower','upper')]
    available=mask&np.isfinite(p)&np.isfinite(lo)&np.isfinite(hi)
    zero=theta==0.;one=theta==1.
    covered=(lo<=theta)&(theta<=hi)
    counts=dict(requested_area_cells=int(mask.sum()),available_area_cells=int(available.sum()),
        truth_zero_requested=int((mask&zero).sum()),truth_one_requested=int((mask&one).sum()),
        truth_zero_available=int((available&zero).sum()),truth_one_available=int((available&one).sum()),
        truth_zero_covered=int((available&zero&covered).sum()),truth_one_covered=int((available&one&covered).sum()))
    for name,array in [('mean',p),('lower',lo),('upper',hi)]:
        counts[name+'_exact_zero']=int((available&(array==0.)).sum())
        counts[name+'_exact_one']=int((available&(array==1.)).sum())
    counts['interval_degenerate_zero']=int((available&(lo==0.)&(hi==0.)).sum())
    counts['interval_degenerate_one']=int((available&(lo==1.)&(hi==1.)).sum())
    counts['lower_zero_at_interior_truth']=int((available&(lo==0.)&~zero&~one).sum())
    counts['upper_one_at_interior_truth']=int((available&(hi==1.)&~zero&~one).sum())
    return counts


def fit_tasks(contract):
    tasks=[]
    for tag in contract['final_tags'][:2]:
        config=json.loads((OUT/tag/'config.json').read_text())
        for scenario in config['scenarios']:
            for rep in config['replications']:
                for method in config['methods']:
                    tasks.append(dict(scenario=scenario,replication=int(rep),method=method,
                        protocol='common_info_areaX_primary_df_graph',
                        output_dir=str(OUT/tag/'fit_results'/method/scenario/f'rep_{rep:05d}'),
                        experiment='candidate_and_direct_SAE_controls'))
    for name in ['gao_all1800.csv','hogg_collector1100.csv']:
        for task in pd.read_csv(OUT/'task_plan'/name).to_dict('records'):
            task['experiment']='external_native_or_additional_arm';tasks.append(task)
    spatial=OUT/'spatial_evaluation/completion.json'
    if spatial.exists():
        for task in pd.read_csv(OUT/'task_plan/hogg_spatial100.csv').to_dict('records'):
            task['experiment']='separate_spatial_component_sensitivity';tasks.append(task)
    frame=pd.DataFrame(tasks)
    if frame.duplicated(['scenario','replication','method','protocol']).any():
        raise ValueError('Duplicated boundary-audit fit tasks')
    return frame


def audit():
    protocol=json.loads((OUT/'protocol_freeze.json').read_text())
    contract=json.loads((OUT/'final_result_contract.json').read_text())
    if contract['final_tags'][:2]!=['final_primary_v2','final_structural_v2']:
        raise ValueError('Boundary audit requires approved candidate v2')
    if not (OUT/'final_evaluation/final_completion.json').exists():
        raise RuntimeError('Wait for final locked predictions before the descriptive audit')
    destination=OUT/'boundary_audit'
    if destination.exists():raise FileExistsError('Boundary audit is immutable')
    population=[];cache={}
    for scenario,ids in protocol['independent_seed_ids'].items():
        for rep in ids:
            folder=OUT/'inputs'/scenario/f'rep_{rep:05d}';theta,groups=masks(folder)
            cache[(scenario,int(rep))]=(theta,groups)
            for group,mask in groups.items():
                for t,name in enumerate(OUTCOMES):
                    use=mask[:,t];values=theta[:,t]
                    population.append(dict(scenario=scenario,replication=int(rep),group=group,outcome=name,
                        requested_area_cells=int(use.sum()),truth_exact_zero=int((use&(values==0.)).sum()),
                        truth_exact_one=int((use&(values==1.)).sum())))
    tasks=fit_tasks(contract);records=[];failures=[]
    for task in tasks.to_dict('records'):
        theta,groups=cache[(task['scenario'],int(task['replication']))]
        path=Path(task['output_dir'])/'predictions.csv'
        available_artifact=True
        try:predictions=canonical_predictions(path)
        except Exception as error:
            available_artifact=False
            predictions={key:np.full((80,2),np.nan)for key in ('mean','lower','upper')}
            failures.append(dict(scenario=task['scenario'],replication=int(task['replication']),
                method=task['method'],error=repr(error),output_dir=task['output_dir']))
        for group,mask in groups.items():
            if task['method'].startswith('Direct')and group!='observed_endpoint':continue
            for t,name in enumerate(OUTCOMES):
                pred={key:array[:,t]for key,array in predictions.items()}
                row=dict(scenario=task['scenario'],replication=int(task['replication']),
                    method=task['method'],protocol=task['protocol'],experiment=task['experiment'],
                    group=group,outcome=name,completed_prediction_artifact=available_artifact)
                row.update(boundary_counts(theta[:,t],pred,mask[:,t]));records.append(row)
    pop=pd.DataFrame(population);pred=pd.DataFrame(records)
    destination.mkdir(parents=True)
    pop.to_csv(destination/'population_replications.csv',index=False)
    pred.to_csv(destination/'prediction_replications.csv',index=False)
    popkeys=['scenario','group','outcome']
    popsummary=pop.groupby(popkeys).agg(requested_replications=('replication','size'),
        requested_area_cells=('requested_area_cells','sum'),
        truth_exact_zero=('truth_exact_zero','sum'),truth_exact_one=('truth_exact_one','sum'),
        replications_with_truth_zero=('truth_exact_zero',lambda x:int((x>0).sum())),
        replications_with_truth_one=('truth_exact_one',lambda x:int((x>0).sum()))).reset_index()
    popsummary['truth_zero_rate']=popsummary.truth_exact_zero/popsummary.requested_area_cells
    popsummary['truth_one_rate']=popsummary.truth_exact_one/popsummary.requested_area_cells
    popsummary.to_csv(destination/'population_boundaries.csv',index=False)
    predkeys=['scenario','method','protocol','experiment','group','outcome']
    countcols=[col for col in pred if col not in predkeys+['replication','completed_prediction_artifact']]
    summary=pred.groupby(predkeys)[countcols].sum().reset_index()
    counts=pred.groupby(predkeys).agg(requested_replications=('replication','size'),
        completed_prediction_artifacts=('completed_prediction_artifact','sum')).reset_index()
    summary=summary.merge(counts,on=predkeys,validate='one_to_one')
    summary['missing_area_cells']=summary.requested_area_cells-summary.available_area_cells
    summary['truth_zero_reported_coverage']=np.where(summary.truth_zero_available>0,
        summary.truth_zero_covered/summary.truth_zero_available,np.nan)
    summary['truth_one_reported_coverage']=np.where(summary.truth_one_available>0,
        summary.truth_one_covered/summary.truth_one_available,np.nan)
    summary.to_csv(destination/'prediction_boundaries.csv',index=False)
    (destination/'failures.json').write_text(json.dumps(failures,indent=2))
    completion=dict(completed=True,finished_utc=datetime.now(timezone.utc).isoformat(),
        analysis='Descriptive exact floating point boundary accounting; no new model selection, score, hypothesis family or rerun',
        exact_boundary_comparison='IEEE theta/lower/upper/mean ==0.0 or==1.0; no epsilon or clipping of truth',
        interpretation='Numerically reported endpoints do not establish structural posterior mass at0/1; counts alone cannot identify rounding, underflow or clipping causes',
        unique_populations=len(cache),requested_fit_tasks=len(tasks),prediction_read_failures=len(failures),
        spatial_secondary_included=(OUT/'spatial_evaluation/completion.json').exists(),
        nonconverged_finite_predictions_included=True,requested_failure_denominators_preserved=True,
        primary_score_and_Holm3_unchanged=True,additional_independent_populations=0,
        protocol_sha256=sha256(OUT/'protocol_freeze.json'),
        main_completion_sha256=sha256(OUT/'final_evaluation/final_completion.json'),
        source_sha256=sha256(Path(__file__)),
        output_sha256={path.name:sha256(path)for path in destination.iterdir()if path.is_file()})
    (destination/'completion.json').write_text(json.dumps(completion,indent=2));return completion


def self_check():
    theta=np.array([0.,1.,.5,0.]);pred=dict(mean=np.array([.1,.9,.5,np.nan]),
        lower=np.array([0.,.8,0.,np.nan]),upper=np.array([.2,1.,1.,np.nan]))
    counts=boundary_counts(theta,pred,np.ones(4,bool))
    assert counts['requested_area_cells']==4 and counts['available_area_cells']==3
    assert counts['truth_zero_requested']==2 and counts['truth_zero_available']==1
    assert counts['truth_zero_covered']==1 and counts['truth_one_covered']==1
    assert counts['lower_exact_zero']==2 and counts['upper_exact_one']==2
    assert counts['lower_zero_at_interior_truth']==1 and counts['upper_one_at_interior_truth']==1
    pred['lower'][0]=np.nextafter(0.,1.)
    assert boundary_counts(theta,pred,np.ones(4,bool))['truth_zero_covered']==0
    return dict(passed=True,exact_zero_distinguished_from_smallest_positive_float=True,
        missing_artifact_denominator_retained=True,interior_truth_boundary_endpoints_separated=True,
        no_actual_truth_or_performance_read=True,source_sha256=sha256(Path(__file__)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true')
    parser.add_argument('--include-spatial',action='store_true',help='Wait for the separate100 spatial predictions too')
    parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:
        proof=self_check();path=OUT/'validation/boundary_audit_qa.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2));return
    mainpath=OUT/'final_evaluation/final_completion.json';spatialpath=OUT/'spatial_evaluation/completion.json'
    while args.wait and (not mainpath.exists()or(args.include_spatial and not spatialpath.exists())):time.sleep(30)
    if args.include_spatial and not spatialpath.exists():raise RuntimeError('Spatial secondary predictions are still pending')
    print(json.dumps(audit(),indent=2))


if __name__=='__main__':main()
