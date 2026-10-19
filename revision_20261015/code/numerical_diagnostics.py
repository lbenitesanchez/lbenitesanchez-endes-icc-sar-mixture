#!/usr/bin/env python3
"""Export observed numerical diagnostics only; never reads simulation truth."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numerical_cohorts_portable as run

def safe_extreme(values,which):
    vals=[v for v in values if v is not None]
    return float((max if which=='max' else min)(vals)) if vals else None

def execute():
    run.gate();details=[];reml=[];summary=[]
    for cohort in ('development_nuts','sparse57'):
        methods=run.DEVMETHODS if cohort=='development_nuts' else run.METHODS
        for method in methods:
            worlds=0;endpoints=[];trials=[];method_reml=[]
            for _,context,rep in run.jobs_for(cohort):
                folder=run.PRIVATE/cohort/'fits'/context/f'rep_{rep:05d}';p=folder/(method+'_status.json')
                if not p.exists() or not (folder/(method+'_completion.json')).exists():continue
                worlds+=1;st=run.read(p)
                for j,end in enumerate(st.get('endpoints',[])):
                    final=end['sampler_trials'][-1];endpoints.append(end)
                    for t in end['sampler_trials']:
                        row=dict(cohort=cohort,context=context,replication=rep,method=method,endpoint=('anemia','stunting')[j],attempt=t['attempt'],is_final_attempt=t==final,max_Rhat=t['max_Rhat'],min_bulk_ESS=t['min_bulk_ESS'],min_tail_ESS=t['min_tail_ESS'],min_E_BFMI=min(t['E_BFMI']),divergences=t['divergences'],max_treedepth_hits=t['max_treedepth_hits'],passed=t['success'],beta_shape_guard_cells=end['beta_shape_guard_cells']);details.append(row);trials.append(row)
                for j,end in enumerate(st.get('fits',[])):
                    for fit in end.get('fit',[]):
                        row=dict(cohort=cohort,context=context,replication=rep,method=method,endpoint=('anemia','stunting')[j],stratum=fit['stratum'],converged=fit['converged'],raw_optimizer_converged=fit.get('raw_optimizer_converged'),stationarity_certificate_passed=fit.get('stationarity_certificate',{}).get('passed'),bootstrap_B=fit.get('bootstrap_B'),bootstrap_failed=fit.get('bootstrap_failed'),bootstrap_stationarity_recovered=fit.get('bootstrap_stationarity_recovered'),reused_exact_development_reference=st.get('reused_exactly',False));reml.append(row);method_reml.append(row)
            final=[x for x in trials if x['is_final_attempt']]
            summary.append(dict(cohort=cohort,method=method,planned_worlds=len(run.jobs_for(cohort)),observed_completed_methods=worlds,observed_NUTS_endpoints=len(endpoints),max_Rhat_final=safe_extreme([x['max_Rhat'] for x in final],'max'),min_ESS_bulk_final=safe_extreme([x['min_bulk_ESS'] for x in final],'min'),min_ESS_tail_final=safe_extreme([x['min_tail_ESS'] for x in final],'min'),min_E_BFMI_final=safe_extreme([x['min_E_BFMI'] for x in final],'min'),divergences_all_attempts=sum(x['divergences'] for x in trials) if trials else None,treedepth_hits_all_attempts=sum(x['max_treedepth_hits'] for x in trials) if trials else None,divergences_final=sum(x['divergences'] for x in final) if final else None,treedepth_hits_final=sum(x['max_treedepth_hits'] for x in final) if final else None,NUTS_retried_endpoints=sum(len(x['sampler_trials'])>1 for x in endpoints) if endpoints else None,NUTS_failed_final_endpoints=sum(not x['success'] for x in endpoints) if endpoints else None,REML_stratum_fits_observed=len(method_reml) if method_reml else None,REML_raw_optimizer_failures=sum(x['raw_optimizer_converged'] is False for x in method_reml) if method_reml else None,REML_final_failures=sum(not x['converged'] for x in method_reml) if method_reml else None,REML_bootstrap_failed=sum(x['bootstrap_failed'] or 0 for x in method_reml) if method_reml else None,REML_bootstrap_stationarity_recovered=sum(x['bootstrap_stationarity_recovered'] or 0 for x in method_reml) if method_reml else None))
    out=run.PUBLIC;out.mkdir(parents=True,exist_ok=True)
    for name,rows in (('observed_numerical_diagnostic_summary.csv',summary),('observed_NUTS_all_attempts.csv',details),('observed_REML_stratum_fits.csv',reml)):
        pd.DataFrame(rows).to_csv(out/name,index=False)
    run.save(out/'numerical_diagnostic_export_receipt.json',dict(exported_utc=run.utc(),NUTS_attempts_observed=len(details),REML_stratum_fits_observed=len(reml),simulation_truth_read=False,missing_diagnostics_are_unavailable_not_zero=True,thresholds_in_protocol=True,protocol_sha256=run.sha(run.PROTOCOL),outputs={name:run.sha(out/name) for name in ('observed_numerical_diagnostic_summary.csv','observed_NUTS_all_attempts.csv','observed_REML_stratum_fits.csv')}))
    print(json.dumps({'observed_NUTS_attempts':len(details),'observed_REML_fits':len(reml)}))
if __name__=='__main__':execute()
