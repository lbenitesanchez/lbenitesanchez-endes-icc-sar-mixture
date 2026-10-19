"""Secondary full2025 application of the unchanged ICC beta reference."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
from pathlib import Path
import datetime as dt,json,time
import numpy as np
import pandas as pd
from validacion_fit_models import ROOT,STUDY,gate,sha,unit_dataset,assert_training_only,assert_prepared_training,load_binary
from diseno_fit_cut import hmc,DiagonalMetric

def main():
    protocol=gate(2025)
    aux=ROOT/'datos/geografia/aux_verified_2017_2018.csv'
    prepared=ROOT/'datos/diseno/2025_full_official'
    child=STUDY/'datos/private_competitors/children_2025.csv'
    out=ROOT/'resultados/application2025_baseline'
    if out.exists():raise FileExistsError('Preserve previous application outputs')
    assert sha(aux)=='cbec2fc142ec0ee06dd247230f0b2253e11af04d0e0e76e8374f30b456df7502'
    areas=pd.read_csv(aux,dtype={'ubigeo':str})
    units,excluded=unit_dataset(child,areas)
    assert_training_only(units,2025,child)
    p=np.load(prepared/'fit_inputs.npz')
    codes=pd.read_csv(prepared/'area_ids.csv',dtype={'ubigeo':str})
    assert np.array_equal(codes.ubigeo,areas.ubigeo)
    assert_prepared_training(p,units,areas,2025)
    config={'spatial_sd':.5,'cluster_adjustment':True}
    sampler={'metric':'diag_e','adapt_delta':.95,'retry_adapt_delta':.99,'parallel_chains':1}
    seed=20261007+50+2025
    dispatch={'status':'SECONDARY2025_FROZEN_FAMILY_DISPATCH_BEFORE_FIT',
              'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'protocol_sha256':protocol,
              'source_sha256':sha(__file__),'aux_sha256':sha(aux),'units_sha256':sha(child),
              'preparation_sha256':sha(prepared/'preparation_manifest.json'),
              'config':config,'sampler':sampler,'seed':seed,
              'primary_external_family_extended':False,'scope':'descriptive full2025 application, not heldout performance'}
    proof=ROOT/'validacion/application2025_baseline_pre_dispatch.json'
    with proof.open('x')as f:json.dump(dispatch,f,indent=2)
    out.mkdir(parents=True);started=time.perf_counter()
    hmc._MODEL=DiagonalMetric(load_binary('beta_power_hybrid'))
    fit=hmc.fit(p['X'],p['y'],p['kish_n'],p['sample_idx'],units,p['basis'],
                seed=seed,config=config,sampler_config=sampler,output_dir=out/'private_sampler')
    predictions=areas[['ubigeo']].copy()
    for j,name in enumerate(('anemia','stunting')):
        for key in ('p_mean','lower95','upper95','p_variance'):
            predictions[key+'_'+name]=fit[key][:,j]
    predictions.to_csv(out/'predictions.csv',index=False)
    status={**dispatch,'status':'COMPLETED_SECONDARY2025_ICC_BASELINE',
            'success':all(s['success']and s['beta_shape_guard_cells']==0 for s in fit['status']),
            'seconds':time.perf_counter()-started,'endpoints':fit['status'],
            'prediction_sha256':sha(out/'predictions.csv'),
            'excluded_no_auxiliary_records_by_domain':excluded,'reserved2024responses_read':False}
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    if not status['success']:raise RuntimeError('Final baseline2025 numerical gate failed')
    print(json.dumps({'status':status['status'],'success':True,'seconds':status['seconds']}),flush=True)

if __name__=='__main__':main()
