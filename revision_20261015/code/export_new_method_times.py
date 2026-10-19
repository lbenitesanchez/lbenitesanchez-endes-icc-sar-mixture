#!/usr/bin/env python3
"""Observed stage costs from closed populations; no accuracy or truth access."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numerical_cohorts_portable as run

def export():
    run.gate();rows=[];mixtures=[]
    for cohort in ('development_nuts','sparse57'):
        methods=run.DEVMETHODS if cohort=='development_nuts' else run.METHODS
        for _,context,rep in run.jobs_for(cohort):
            folder=run.PRIVATE/cohort/'fits'/context/f'rep_{rep:05d}'
            if not (folder/'completion.json').exists():continue
            stages={}
            for method in methods:
                r=run.read(folder/(method+'_completion.json'));status=run.read(folder/(method+'_status.json'));sec=float(r['seconds']);assert np.isfinite(sec) and sec>=0;stages[method]=sec
                rows.append(dict(cohort=cohort,context=context,replication=rep,method=method,stage_wall_seconds=sec,reference_reused_exactly=bool(status.get('reused_exactly',False)),time_scope='saved prediction reuse and verification' if status.get('reused_exactly',False) else 'actual fresh fitting or mixture construction'))
            if cohort=='sparse57':
                total=sum(stages[k] for k in ('ICC','Sikov_SAR','Selected'))
                mixtures.append(dict(context=context,replication=rep,ICC_wall_seconds=stages['ICC'],full_SAR_B100_wall_seconds=stages['Sikov_SAR'],mixture_CDF_wall_seconds=stages['Selected'],mixture_strategy_sequential_wall_seconds=total,additional_wall_seconds_over_ICC=total-stages['ICC'],wall_ratio_mixture_over_ICC=total/stages['ICC']))
    pd.DataFrame(rows).to_csv(run.PUBLIC/'observed_method_stage_times.csv',index=False)
    pd.DataFrame(mixtures).to_csv(run.PUBLIC/'observed_sparse57_mixture_strategy_times.csv',index=False)
    summary=[]
    for (cohort,method),g in pd.DataFrame(rows).groupby(['cohort','method']):
        v=g.stage_wall_seconds.to_numpy();summary.append(dict(cohort=cohort,method=method,completed_worlds=len(v),mean=float(np.mean(v)),median=float(np.median(v)),minimum=float(v.min()),maximum=float(v.max()),units='wall seconds',exact_reference_reuse=bool(g.reference_reused_exactly.all())))
    addition=None
    if mixtures:
        d=pd.DataFrame(mixtures);addition=dict(completed_sparse57_worlds=len(d),ICC_median_wall_seconds=float(d.ICC_wall_seconds.median()),full_SAR_B100_median_wall_seconds=float(d.full_SAR_B100_wall_seconds.median()),mixture_CDF_median_wall_seconds=float(d.mixture_CDF_wall_seconds.median()),mixture_strategy_median_sequential_wall_seconds=float(d.mixture_strategy_sequential_wall_seconds.median()),median_additional_wall_seconds_over_ICC=float(d.additional_wall_seconds_over_ICC.median()),median_wall_ratio_mixture_over_ICC=float(d.wall_ratio_mixture_over_ICC.median()))
    result=dict(format='iaos-observed-method-stage-times-v1',observed_utc=run.utc(),scope='Only fully closed populations at the stated snapshot; numerical computation timing, not accuracy; unfinished populations are not runtime benchmarks. Concurrent unrelated workloads and heterogeneous contexts limit hardware-independent interpretation.',method_stage_summaries=summary,sparse57_mixture_additional_cost=addition,computational_cost_not_an_accuracy_score=True,simulation_truth_read=False,full_mixture_cost_definition='ICC fitting plus full-SAR B100 fitting plus mixture CDF construction, observed sequential stage wall times. FH and observed-SAR are comparison costs and excluded from the mixture cost.',development_reused_FH_SAR_not_charged_as_fresh_fitting=True,code_sha256=run.sha(__file__))
    run.save(run.PUBLIC/'observed_method_computational_times.json',result);print(json.dumps({'observed_utc':result['observed_utc'],'sparse57_mixture_additional_cost':addition}))
if __name__=='__main__':export()
