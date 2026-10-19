"""Matched actual-data fits, followed by an immutable external prediction seal."""
from pathlib import Path
import argparse,datetime as dt,json,time
import numpy as np
import pandas as pd
from comparison_simulation import ROOT,sha,save
from comparison_evaluate import gate,fits

def run(scope):
    proof,freeze=gate();year=2019 if scope=='temporal' else 2024
    prepared=ROOT/'datos'/('real_2019_full' if year==2019 else 'real_2024_training')
    manifest=json.loads((prepared/'preparation_manifest.json').read_text())
    if manifest['heldout_outcome_rows_read']:raise RuntimeError('Heldout training leakage')
    if year==2024 and manifest['protocol_sha256']!=freeze:raise RuntimeError('Training not bound to freeze')
    for name,digest in manifest['output_sha256'].items():
        if sha(prepared/name)!=digest:raise RuntimeError('Changed prepared source '+name)
    d=dict(np.load(prepared/'fit_inputs.npz',allow_pickle=False));d['X']=d['X_standardized']
    # Alignment records latitude/longitude; native Sikov kNN uses longitude/latitude.
    if manifest['coordinates_npz_order']!=['latitude','longitude']:raise RuntimeError('Unrecognized coordinate convention')
    d['coords']=d['coords'][:,::-1]
    unitspath=prepared/manifest['private_units_path'].split('/')[-2]/'units.csv'
    if sha(unitspath)!=manifest['private_units_sha256']:raise RuntimeError('Changed private training units')
    units=pd.read_csv(unitspath)
    out=ROOT/'evaluacion'/('real_'+scope)
    if out.exists():raise FileExistsError(out)
    before=time.perf_counter();status=fits(d,units,out,20261010+year,proof['selected_recipe'],sensitivity=True)
    ids=pd.read_csv(prepared/'area_frame.csv',dtype={'ubigeo':str})
    rows=[]
    for name in status:
        p=dict(np.load(out/(name+'.npz')))
        if p['mean'].shape!=(len(ids),2):raise ValueError('Unequal actual-data frames')
        for i,code in enumerate(ids.ubigeo):
            rows.append({'ubigeo':code,'method':name,'p_anemia':p['mean'][i,0],'p_stunting':p['mean'][i,1],'lo95_anemia':p['lower'][i,0],'hi95_anemia':p['upper'][i,0],'lo95_stunting':p['lower'][i,1],'hi95_stunting':p['upper'][i,1]})
    pred=out/'predictions.csv';pd.DataFrame(rows).to_csv(pred,index=False)
    gate()
    completion={'scope':scope,'year':year,'protocol_sha256':freeze,'all_converged':all(s['converged'] for s in status.values()),'convergence':{k:s['converged'] for k,s in status.items()},'heldout_outcomes_read_by_fits':False,'training_manifest_sha256':sha(prepared/'preparation_manifest.json'),'predictions_sha256':sha(pred),'seconds':time.perf_counter()-before,'output_sha256':{p.name:sha(p)for p in out.iterdir() if p.is_file()}}
    save(out/'completion.json',completion)
    lock={'scope':scope,'protocol_sha256':freeze,'approved_for_scoring':True,'predictions_sha256':sha(pred),'predictions_path':str(pred),'locked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'heldout_outcomes_used':False,'reuse_of_previously_evaluated_2024_disclosed':True,'all_converged':completion['all_converged'],'completion_sha256':sha(out/'completion.json')}
    save(ROOT/f'validacion/real_prediction_lock_{scope}.json',lock)
    print(json.dumps(completion,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=['temporal','territorial'],required=True);a=p.parse_args();run(a.scope)
