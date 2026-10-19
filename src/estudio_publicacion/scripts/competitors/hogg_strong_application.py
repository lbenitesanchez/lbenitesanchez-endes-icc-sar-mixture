"""Prespecified ENDES2025 age/sex S1 sensitivity, independent of frozen primary.

Survey-only age/sex features give this arm a higher information budget. S2
retains the same six district auxiliaries and original author-code GVF.
Private child records are read in place, never saved in result directories.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import pandas as pd
import hogg_tsln as h
import hogg_diagnostics as hd


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--modes',nargs='+',choices=['full','locked_train'],default=['full','locked_train'])
    parser.add_argument('--parallel-chains',type=int,default=1)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    child_path=h.ROOT/'datos/private_competitors/children_2025.csv'
    aux_path=h.ROOT/'datos/private_competitors/area_aux.csv'
    locked_path=h.ROOT/'resultados/aplicacion_2025_who2024/locked_test_area_scores.csv'
    child=pd.read_csv(child_path,dtype={'ubigeo':str,'psu_key':str})
    area=pd.read_csv(aux_path,dtype={'ubigeo':str})
    locked=pd.read_csv(locked_path,dtype={'ubigeo':str,'province':str})
    provinces=sorted(locked.province.unique())
    child,features,feature_metadata=hd.age_sex_features(child)
    cfg=h.MCMCConfig(parallel_chains=args.parallel_chains,phase='application_age_sex_sensitivity')
    preflight=[]
    for mode in args.modes:
        unit=child if mode=='full' else child.loc[~child.ubigeo.str[:4].isin(provinces)].copy()
        for outcome in ['y_anemia_who2024','y_stunting']:
            prep=h.prepare_hogg_data(unit,area,outcome=outcome,weight='weight',area_col='ubigeo',
                area_features=h.DEFAULT_FEATURES,unit_features=features,population_N=None)
            preflight.append({'mode':mode,**prep['metadata']})
    h.write_json(h.ROOT/'resultados/competidores_hogg/age_sex_preflight.json',
        {'features':feature_metadata,'tasks':preflight,'sampler':h.asdict(cfg)})
    if args.preflight_only:
        print(json.dumps(h._jsonable(preflight)),flush=True)
        return
    models=h.compile_models()
    tasks=[]
    for mode in args.modes:
        output=h.ROOT/f'resultados/competidores_hogg/endes_2025_{mode}_age_sex'
        unit=child if mode=='full' else child.loc[~child.ubigeo.str[:4].isin(provinces)].copy()
        started=time.monotonic()
        result=h.fit_hogg_pair(unit,area,area_col='ubigeo',area_features=h.DEFAULT_FEATURES,
            unit_features=features,population_N=None,config=cfg,output_dir=output,models=models,
            outcomes=('y_anemia_who2024','y_stunting'),weights=('weight','weight'),
            retry_on_diagnostics=True,retain_sampler_csv=True)
        alc=hd.annotate_pair(result,unit,area,area_col='ubigeo',area_features=h.DEFAULT_FEATURES,
            unit_features=features,population_N=None,outcomes=('y_anemia_who2024','y_stunting'),
            weights=('weight','weight'))
        pred=result['predictions'].rename(columns={'area':'ubigeo'})
        for endpoint,outcome in [('anemia','y_anemia_who2024'),('stunting','y_stunting')]:
            measured=unit.loc[unit[outcome].isin([0,1])]
            count=measured.groupby('ubigeo').size()
            psu=measured.groupby('ubigeo').psu_key.nunique()
            sw=measured.groupby('ubigeo').weight.sum()
            sw2=measured.assign(w2=measured.weight**2).groupby('ubigeo').w2.sum()
            pred['n_'+endpoint]=pred.ubigeo.map(count).fillna(0).astype(int)
            pred['n_psu_'+endpoint]=pred.ubigeo.map(psu).fillna(0).astype(int)
            pred['n_eff_kish_'+endpoint]=pred.ubigeo.map(sw**2/sw2)
            pred['unsampled_'+endpoint]=pred['n_'+endpoint]==0
            pred['one_psu_'+endpoint]=pred['n_psu_'+endpoint]==1
            pred['n_le5_'+endpoint]=pred['n_'+endpoint]<=5
            threshold=.30 if endpoint=='anemia' else .15
            pred[f'conditional_probability_{endpoint}_gt_{threshold}']=(result['results'][endpoint].draws>threshold).mean(axis=0)
        pred['province']=pred.ubigeo.str[:4]
        pred['locked_test_province']=pred.province.isin(provinces)
        pred['interval_interpretation']='model_conditional_posterior_not_uniformly_design_calibrated'
        pred['diagnostic_use']='experimental_no_operational_district_ranking'
        pred['information_budget']='higher_information_survey_age_sex_in_stage1'
        pred.to_csv(output/'predictions.csv',index=False)
        metadata={'year':2025,'mode':mode,'success':result['status']['success'],
            'features':feature_metadata,'district_features':h.DEFAULT_FEATURES,
            'unit_features':features,'sampling_D':'Original Hogg iid formula; PSU clustering remains unadjusted.',
            'modular_stage2':'Original author-code Gaussian summary, not joint non-Gaussian S1/S2 posterior.',
            'age_months':'6–59 inclusive','population':'de_facto eligible HV103=1',
            'anemia_rule':'WHO2024','stunting_rule':'HAZ<-2',
            'biomass_semantic':'Census dung/manure fuel proportion, not total biomass.',
            'district_population_N':'Unavailable; FPC=1 negligible-fraction assumption.',
            'locked_provinces':provinces,'unit_records_fit_frame':len(unit),
            'all_prediction_districts':len(area),'seconds_total':time.monotonic()-started,
            'input_sha256':{str(p.relative_to(h.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [child_path,aux_path,locked_path]},
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__),h.HERE/'hogg_diagnostics.py']},
            'ALC':alc,'privacy':'No child records or identifiers copied to outputs.',
            'interpretation':'Higher information sensitivity; no method-superiority or design-calibration claim from interval width.'}
        h.write_json(output/'application_manifest.json',metadata)
        tasks.append(metadata)
        print(json.dumps({'mode':mode,'success':metadata['success'],'seconds':metadata['seconds_total']}),flush=True)
    h.write_json(h.ROOT/'resultados/competidores_hogg/age_sex_application_summary.json',tasks)


if __name__=='__main__':main()
