"""Hogg TSLN ENDES full and previously locked province-holdout fits.

Private child records are read in place and never copied into result files.
Outcomes use the WHO2024 anemia rule and HAZ stunting at ages6–59 months.
Only model-conditional summaries are emitted; no district ranking is produced.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import hogg_tsln as h


def load_completed_pair(output,units,areas):
    """Recover completed marginals after an API-only wrapper interruption.

    Reuses all posterior draws, checks the input identity and each final
    parameter diagnostic, and never refits an already completed endpoint.
    """
    status_path=output/'status.json'
    if not status_path.exists():return None
    status=json.loads(status_path.read_text())
    if not status.get('completed'):return None
    for label in ['anemia','stunting']:
        endpoint=status['outcomes'][label]
        if endpoint['input_sha256']!={'units':h._input_hash(units),'areas':h._input_hash(areas)}:
            raise ValueError('Cannot resume: completed input hash differs')
        attempt=len(status['all_attempts'][label])-1
        for stage in ['stage1','stage2']:
            summary=pd.read_csv(output/label/f'attempt_{attempt}'/f'{stage}_parameter_summary.csv',index_col=0)
            names=summary.index.to_series().astype(str)
            rows=~names.str.startswith(('lp__','eta[','effect_area[','linking_mean[','sampling_sd[',
                'log_sampling_sd_mean[','observation_variance[','theta[','prevalence['))
            values=summary.loc[rows,['R_hat','ESS_bulk','ESS_tail']].to_numpy(float)
            if not (np.isfinite(values).all() and values[:,0].max()<=1.01 and values[:,1:].min()>=400):
                raise ValueError('Cannot resume: final parameter diagnostic gate failed')
            dg=endpoint[stage];bfmi=np.asarray(dg['BFMI_per_chain'],float)
            if dg['divergences'] or dg['max_treedepth_hits'] or not np.isfinite(bfmi).all() or bfmi.min()<.3:
                raise ValueError('Cannot resume: sampler diagnostic gate failed')
    posterior=np.load(output/'posterior_draws.npz')
    # Endpoint CSVs preserve original all-draw summaries. Reconstruct canonical
    # columns directly rather than recomputing or pairing unequal draw counts.
    predictions=pd.DataFrame({'area':posterior['area'].astype(str)})
    result={}
    for label in ['anemia','stunting']:
        attempt=len(status['all_attempts'][label])-1
        frame=pd.read_csv(output/label/f'attempt_{attempt}'/'area_predictions.csv',dtype={'area':str})
        if not np.array_equal(frame.area,predictions.area):raise ValueError('Completed domain ordering differs')
        for target,source in [('p_mean','mean'),('lo95','lo95_ET'),('hi95','hi95_ET'),
                              ('p_median','median'),('lo95_HDI','lo95_HDI'),('hi95_HDI','hi95_HDI')]:
            predictions[target+'_'+label]=frame[source]
        result[label]=SimpleNamespace(draws=posterior[label])
    return {'predictions':predictions,'results':result,'status':status,'recovered_completed_pair':True}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--years',nargs='+',type=int,default=[2025,2019])
    p.add_argument('--modes',nargs='+',choices=['full','locked_train'],
                   default=['full','locked_train'])
    p.add_argument('--parallel-chains',type=int,default=1)
    p.add_argument('--resume-completed',action='store_true')
    args=p.parse_args()
    models=h.compile_models()
    cfg=h.MCMCConfig(parallel_chains=args.parallel_chains,phase='application')
    area_file=h.ROOT/'datos/private_competitors/area_aux.csv'
    area=pd.read_csv(area_file,dtype={'ubigeo':str})
    tasks=[]
    for year in args.years:
      child_file=h.ROOT/f'datos/private_competitors/children_{year}.csv'
      child=pd.read_csv(child_file,dtype={'ubigeo':str,'psu_key':str})
      score_file=h.ROOT/f'resultados/aplicacion_{year}_who2024/locked_test_area_scores.csv'
      locked=pd.read_csv(score_file,dtype={'ubigeo':str,'province':str})
      locked_provinces=sorted(locked.province.unique())
      for mode in args.modes:
        out=h.ROOT/f'resultados/competidores_hogg/endes_{year}_{mode}'
        unit=child if mode=='full' else child.loc[~child.ubigeo.str[:4].isin(locked_provinces)].copy()
        started=time.monotonic()
        result=load_completed_pair(out,unit,area) if args.resume_completed else None
        if result is None:result=h.fit_hogg_pair(unit,area,area_col='ubigeo',
            area_features=h.DEFAULT_FEATURES,population_N=None,config=cfg,
            output_dir=out,models=models,
            outcomes=('y_anemia_who2024','y_stunting'),weights=('weight','weight'),
            retry_on_diagnostics=True,retain_sampler_csv=True)
        predictions=result['predictions']
        predictions=predictions.rename(columns={'area':'ubigeo'})
        for endpoint,outcome in [('anemia','y_anemia_who2024'),('stunting','y_stunting')]:
          measured=unit.loc[unit[outcome].isin([0,1])]
          counts=measured.groupby('ubigeo').size()
          psu=measured.groupby('ubigeo')['psu_key'].nunique()
          sum_w=measured.groupby('ubigeo').weight.sum()
          sum_w2=measured.assign(w2=measured.weight**2).groupby('ubigeo').w2.sum()
          predictions['n_'+endpoint]=predictions.ubigeo.map(counts).fillna(0).astype(int)
          predictions['n_psu_'+endpoint]=predictions.ubigeo.map(psu).fillna(0).astype(int)
          predictions['n_eff_kish_'+endpoint]=predictions.ubigeo.map(sum_w**2/sum_w2)
          predictions['unsampled_'+endpoint]=predictions['n_'+endpoint]==0
          predictions['one_psu_'+endpoint]=predictions['n_psu_'+endpoint]==1
          predictions['n_le5_'+endpoint]=predictions['n_'+endpoint]<=5
          threshold=.30 if endpoint=='anemia' else .15
          predictions['conditional_probability_'+endpoint+'_gt_'+str(threshold)]=(result['results'][endpoint].draws>threshold).mean(axis=0)
        predictions['interval_interpretation']='model_conditional_posterior_not_uniformly_design_calibrated'
        predictions['diagnostic_use']='experimental_no_operational_district_ranking'
        predictions['province']=predictions.ubigeo.str[:4]
        predictions['locked_test_province']=predictions.province.isin(locked_provinces)
        predictions.to_csv(out/'predictions.csv',index=False)
        meta={'year':year,'mode':mode,'age_months':'6–59 inclusive',
            'population':'de_facto_eligible_children_HV103=1',
            'anemia_rule':'WHO2024','stunting_rule':'HAZ<-2',
            'features':h.DEFAULT_FEATURES,
            'biomass_semantic':'census proportion using dung/manure fuel, not total biomass',
            'individual_auxiliary_features':[],
            'district_population_N':'Unavailable; negligible-fraction FPC=1 assumed explicitly.',
            'input_child_sha256':hashlib.sha256(child_file.read_bytes()).hexdigest(),
            'input_area_sha256':hashlib.sha256(area_file.read_bytes()).hexdigest(),
            'locked_test_definition_sha256':hashlib.sha256(score_file.read_bytes()).hexdigest(),
            'locked_provinces':locked_provinces,
            'unit_records_before_train_filter':len(child),'unit_records_fit_frame':len(unit),
            'all_prediction_districts':len(area),
            'seconds_total':time.monotonic()-started+(result['status']['seconds_total'] if result.get('recovered_completed_pair') else 0),
            'seconds_this_invocation':time.monotonic()-started,
            'seconds_original_posterior_fits':result['status']['seconds_total'],
            'private_inputs_policy':'Read only; no child or household identifiers or records copied into outputs.',
            'limitations':'Published generic Hogg iid variance does not account for survey PSU/stratum clustering. Priors, S1 weighted model, S2 Gaussian variance and GVF assumptions define posterior uncertainty; simulations assess coverage. No paired-rule change distribution is inferred.',
            'success':result['status']['success'],
            'reused_completed_marginal_fits':result.get('recovered_completed_pair',False)}
        h.write_json(out/'application_manifest.json',meta)
        tasks.append(meta)
        print(json.dumps({'year':year,'mode':mode,'success':meta['success'],
                          'seconds':meta['seconds_total']}),flush=True)
    h.write_json(h.ROOT/'resultados/competidores_hogg/application_summary.json',tasks)


if __name__=='__main__':main()
