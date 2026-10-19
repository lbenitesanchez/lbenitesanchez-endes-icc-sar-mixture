"""Separate S2 BYM2 ENDES sensitivity, reusing verified age/sex S1 draws."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import hogg_tsln as h
import hogg_diagnostics as hd
import hogg_spatial as sp


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--modes',nargs='+',choices=['full','locked_train'],default=['full','locked_train'])
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    child_path=h.ROOT/'datos/private_competitors/children_2025.csv'
    aux_path=h.ROOT/'datos/private_competitors/area_aux.csv'
    locked_path=h.ROOT/'resultados/aplicacion_2025_who2024/locked_test_area_scores.csv'
    child=pd.read_csv(child_path,dtype={'ubigeo':str,'psu_key':str})
    areas=pd.read_csv(aux_path,dtype={'ubigeo':str})
    locked=pd.read_csv(locked_path,dtype={'ubigeo':str,'province':str})
    provinces=sorted(locked.province.unique())
    child,features,feature_metadata=hd.age_sex_features(child)
    tasks=[];model=None
    for mode in args.modes:
        source=h.ROOT/f'resultados/competidores_hogg/endes_2025_{mode}_age_sex'
        source_status=json.loads((source/'status.json').read_text())
        if not source_status['success']:raise ValueError('Source age/sex posterior does not satisfy final numerical diagnostics')
        source_manifest=json.loads((source/'application_manifest.json').read_text())
        unit=child if mode=='full'else child.loc[~child.ubigeo.str[:4].isin(provinces)].copy()
        output=h.ROOT/f'resultados/competidores_hogg/endes_2025_{mode}_age_sex_s2_bym2'
        output.mkdir(exist_ok=True)
        edges=h.ROOT/f'resultados/competidores_gao/inputs/endes_2025_{mode}/adjacency_edges.csv'
        sources={};recoveries={}
        for endpoint,outcome in [('anemia','y_anemia_who2024'),('stunting','y_stunting')]:
            attempt=len(source_status['all_attempts'][endpoint])-1
            run=source/endpoint/f'attempt_{attempt}'
            codes,data,recovery=sp.recover_stage2_from_saved(run,unit,areas,outcome=outcome,
                weight='weight',area_col='ubigeo',area_features=h.DEFAULT_FEATURES,
                unit_features=features,population_N=None)
            sources[endpoint]=output/f'source_modular_summary_{endpoint}.npz'
            if sources[endpoint].exists():
                oldcodes,olddata=sp.load_modular_summary(sources[endpoint])
                if not np.array_equal(oldcodes,codes)or any(not np.array_equal(olddata[k],v)for k,v in data.items()):
                    raise ValueError('Existing recovered source summary differs; preserve it for audit')
            else:np.savez_compressed(sources[endpoint],area=codes,**data)
            recoveries[endpoint]=recovery
        metadata={'year':2025,'mode':mode,'information_budget':'Higher-information S1 survey age/sex plus six district predictors; spatial S2 only.',
            'S1_refit':False,'source_age_sex_manifest_sha256':hashlib.sha256((source/'application_manifest.json').read_bytes()).hexdigest(),
            'S1_recovery':recoveries,'age_sex_features':feature_metadata,'district_features':h.DEFAULT_FEATURES,
            'graph_sha256':hashlib.sha256(edges.read_bytes()).hexdigest(),
            'graph_definition':'Reused symmetric coordinate 8-nearest-neighbour graph; componentwise scaling/constraints, not border contiguity.',
            'privacy':'Private child records read only in place for identity reconstruction; no child rows or identifiers exported.',
            'interpretation':'Separate model-conditional spatial sensitivity; no Australian full-model replication, uniform design-coverage or operational ranking claim.',
            'ages':'6-59 months inclusive, de-facto eligible children, WHO2024 anemia and HAZ<-2 stunting',
            'source_input_sha256':source_manifest['input_sha256']}
        h.write_json(output/'source_reconstruction_manifest.json',metadata)
        if args.prepare_only:
            tasks.append({**metadata,'prepared_only':True});continue
        if model is None:model=sp.compile_spatial()
        start=time.monotonic()
        result=sp.fit_spatial_pair(sources,edges,output_dir=output,
            config=h.MCMCConfig(parallel_chains=1,phase='application_spatial_age_sex_sensitivity'),
            model=model,retry_on_diagnostics=True,retain_sampler_csv=True)
        pred=result['predictions'].rename(columns={'area':'ubigeo'})
        baseline=pd.read_csv(source/'predictions.csv',dtype={'ubigeo':str,'province':str})
        quality=[c for c in baseline if c.startswith(('n_','unsampled_','one_psu_','n_le5_'))or c in ['province','locked_test_province']]
        pred=pred.merge(baseline[['ubigeo']+quality],on='ubigeo',validate='one_to_one')
        for endpoint,threshold in [('anemia',.30),('stunting',.15)]:
            pred[f'conditional_probability_{endpoint}_gt_{threshold}']=(result['results'][endpoint]['draws']>threshold).mean(axis=0)
        pred['interval_interpretation']='model_conditional_posterior_not_uniformly_design_calibrated'
        pred['diagnostic_use']='experimental_no_operational_district_ranking'
        pred['information_budget']='higher_information_survey_age_sex_in_stage1_plus_spatial_stage2'
        pred.to_csv(output/'predictions.csv',index=False)
        metadata.update(success=result['status']['success'],all_prediction_districts=len(pred),
            seconds_spatial_fit=time.monotonic()-start,scope='S2 BYM2 component only, no S1 individual residual or benchmarking',
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),h.HERE/'hogg_spatial.py',h.HERE/'hogg_stage2_bym2.stan']})
        h.write_json(output/'application_manifest.json',metadata)
        tasks.append(metadata)
        print(json.dumps({'mode':mode,'success':metadata['success'],'seconds':metadata['seconds_spatial_fit']}),flush=True)
    h.write_json(h.ROOT/'resultados/competidores_hogg/spatial_age_sex_application_summary.json',tasks)


if __name__=='__main__':main()
