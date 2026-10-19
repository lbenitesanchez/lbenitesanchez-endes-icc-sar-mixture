"""Shared area-level training inputs and transparent reconstruction gaps.

Reads only development2019 responses. Held-out2024 data are inventoried by
path/schema from the frozen parser and hashes; no reserved response is loaded.
"""
from pathlib import Path
import hashlib,json,datetime
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT.parent;OLD=STUDY/'fase2_2026-10-07'
OUT=ROOT/'datos';VAL=ROOT/'validacion'
FEATURES=['Altitud','PisoTierra','Cemento','Agua','Elect','Desague','Internet','SIS','EsSalud','NoSeguro','Refrig','Castellano','Analfabet','Rural']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def graph(aux,k=6,metric='planar',within_stratum=True):
    coords=aux[['longitude','latitude']].to_numpy(float)
    if metric=='planar':d=cdist(coords,coords)
    elif metric=='haversine':
        lon,lat=np.deg2rad(coords).T
        q=np.sin((lat[:,None]-lat[None,:])/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin((lon[:,None]-lon[None,:])/2)**2
        d=6371.0088*2*np.arcsin(np.sqrt(np.clip(q,0,1)))
    else:raise ValueError(metric)
    np.fill_diagonal(d,np.inf)
    if within_stratum:
        s=aux.poverty_stratum.to_numpy(int);d[s[:,None]!=s[None,:]]=np.inf
    near=np.argsort(d,axis=1,kind='stable')[:,:k]
    assert np.isfinite(np.take_along_axis(d,near,axis=1)).all()
    W=np.zeros(d.shape,dtype=np.float32);W[np.arange(len(d))[:,None],near]=1/k
    assert np.allclose(W.sum(axis=1),1)
    return W

def main():
    auxpath=OUT/'aux_sikov14_official.csv';survey_path=OUT/'survey2019_harmonized_and_definition_sensitivities.csv'
    aux=pd.read_csv(auxpath,dtype={'ubigeo':str});survey=pd.read_csv(survey_path,dtype={'ubigeo':str})
    names=['intercept']+FEATURES;raw=aux[FEATURES].to_numpy(float)
    center=raw.mean(axis=0);scale=raw.std(axis=0)
    Xraw=np.column_stack([np.ones(len(aux)),raw]);Xstd=np.column_stack([np.ones(len(aux)),(raw-center)/scale])
    strata=aux.poverty_stratum.to_numpy(int)
    W6=graph(aux);W4=graph(aux,k=4);W8=graph(aux,k=8);W6geo=graph(aux,metric='haversine')
    Wglobal=graph(aux,within_stratum=False)
    graph_args=dict(W=W6,W_knn6_planar_stratum=W6,W_knn4_planar_stratum=W4,W_knn8_planar_stratum=W8,W_knn6_haversine_stratum=W6geo,W_knn6_planar_global=Wglobal)
    base=dict(ubigeo=aux.ubigeo.to_numpy(str),X=Xraw,X_raw=Xraw,X_standardized=Xstd,
              covariate_names=np.array(names),X_center=center,X_scale=scale,strata=strata,
              coords=aux[['latitude','longitude']].to_numpy(float),poverty=aux.poverty.to_numpy(float),**graph_args)
    np.savez_compressed(OUT/'aux_sikov14_common_inputs.npz',**base)
    outputs=['aux_sikov14_common_inputs.npz'];counts={}
    for universe in ['6_59','0_59']:
        observed=survey.loc[survey.universe.eq(universe)&survey.ubigeo.isin(aux.ubigeo)].set_index('ubigeo')
        matched=aux.ubigeo.isin(observed.index);idx=np.flatnonzero(matched);obs=observed.reindex(aux.ubigeo.iloc[idx])
        for name,growth in [('harmonized','stunting'),('hc70_p025','growth_hc70_p025'),('hc70_p10','growth_hc70_p10'),('hc5_p10','growth_hc5_p10'),('hc5_minus128','growth_hc5_minus128')]:
            y=obs[['y_anemia','y_'+growth]].to_numpy(float)
            D=np.zeros((len(idx),2,2));D[:,0,0]=obs.var_anemia;D[:,1,1]=obs['var_'+growth]
            D[:,0,1]=D[:,1,0]=obs['cov_anemia_'+growth]
            ne=obs[['n_eff_anemia','n_eff_'+growth]].to_numpy(float)
            n=obs[['n_anemia','n_'+growth]].to_numpy(int)
            outfile=f'fit_inputs_2019_{universe}_{name}.npz'
            np.savez_compressed(OUT/outfile,**base,sample_idx=idx,y=y,D=D,n_eff=ne,kish_n=ne,n=n,
                                observed_endpoint=np.isfinite(y),zero_variance_mask=np.diagonal(D,axis1=1,axis2=2)<=1e-15,
                                test_idx=np.array([],int),outcome_names=np.array(['anemia',growth]))
            outputs.append(outfile)
        counts[universe]={'matched_domains':len(idx),'observed_stratum_counts':{str(s):int(np.sum(strata[idx]==s))for s in [1,2,3]},
                         'domains_finite_both':int(np.isfinite(obs[['y_anemia','y_stunting']]).all(axis=1).sum()),
                         'zero_variance_anemia':int(obs.var_anemia.le(1e-15).sum()),'zero_variance_stunting':int(obs.var_stunting.le(1e-15).sum())}
    excluded=survey.loc[~survey.ubigeo.isin(aux.ubigeo)&survey.ubigeo.ne('PE')].copy()
    excluded.to_csv(OUT/'survey2019_unsupported_direct_only.csv',index=False);outputs.append('survey2019_unsupported_direct_only.csv')
    for universe in ['6_59','0_59']:
        obs=survey[survey.universe.eq(universe)&survey.ubigeo.isin(aux.ubigeo)].set_index('ubigeo')
        q=aux[['ubigeo','poverty_stratum']].merge(obs.reset_index(),on='ubigeo',how='left')
        q.to_csv(OUT/f'areas_2019_{universe}_aligned.csv',index=False);outputs.append(f'areas_2019_{universe}_aligned.csv')
    proof={'status':'PASS_ALIGNED_SHARED2019_INPUTS','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'source_sha256':{'aux':sha(auxpath),'survey2019':sha(survey_path),'R_private_intermediate':sha(STUDY/'datos/raw/analytic_private_2019.rds'),
                            'RECH0_2019':sha(STUDY/'datos/raw/ENDES2019/Modulo64/RECH0.SAV'),'RECH6_2019':sha(STUDY/'datos/raw/ENDES2019/Modulo74/RECH6.SAV'),
                            'RECH1_2019':sha(STUDY/'datos/raw/ENDES2019/Modulo64/RECH1.sav')},
           'outputs_sha256':{s:sha(OUT/s)for s in outputs},'survey_counts':counts,
           'graph_primary':'6NN planar lonlat degree distances, rebuilt independently inside each poverty stratum; row-standardized',
           'graph_sensitivity':['4NN planar within stratum','8NN planar within stratum','6NN haversine within stratum','6NN planar global'],
           'graph_exact_original_identified':False,'feature_names':names,'same_information_for_competitors':True,
           'harmonized_primary_definition':{'anemia':'HC56 altitude-adjusted haemoglobin in tenths g/dL,0<=HC56<999; anemiaHC56<110',
              'stunting':'HC70 WHOheightforage Zx100 in[-600,600];stuntingHC70<-200','universe':'de facto children6to59months; endpoint-specific available-case denominators'},
           'definition_sensitivities':['0to59months growth; anemia availability still6to59','HC70<qnorm(.025)x100','HC70<qnorm(.10)x100','HC5<qnorm(.10)x100','HC5<-128 (author supplied directos_endes.R default)'],
           'variance':'full-household3253PSUs/248strata first-stage with-replacement Taylor ratio covariance; zeros retained, no numerical variance repair silently applied',
           'independent_survey_checks':str((VAL/'align_survey2019_independent_survey_checks.csv').relative_to(ROOT)),
           'reproduction_level':'published FH/SAR model specification can be reconstructed; numerical replication is not established',
           'blocking_exact_replication_gaps':['Authors originalW construction/weights and exact fitted matrix not provided',
               'Authors poverty point predictor/version not provided; our786/888/200 strata differ from published618/671/585',
               'Original age/measurement/missingness definition code not provided; article percentile10 vs2.5 statements inconsistent',
               'Matched sample888 differs from published765districts',
               'Authors exact14covariate extraction denominators/category mappings not supplied'],
           'heldout2024_outcome_values_read':False,'microdata_exported_in_new_data_directory':False,'original_inputs_modified':False}
    (VAL/'align_inputs_manifest.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2))
    # Hashes and header schema only, never outcome values. Read exactly first
    # CSV header line to document scorer/training contracts before model lock.
    inventory=[]
    for relative,role in [('datos/private_external/children_2024_training.csv','territorial2024training'),
                          ('datos/private_external/children_2024_test.csv','territorial2024reservedtest'),
                          ('datos/private_external/children_2024_scoring_only.csv','temporal2024reservedscoring'),
                          ('datos/private_external/households_2024.csv','designframe2024')]:
        path=OLD/relative
        with path.open()as f:header=f.readline().strip()
        inventory.append({'path':str(path.relative_to(STUDY)),'role':role,'sha256':sha(path),'header_line':header,'data_rows_read':False})
    locks=[]
    for name in ['externa_prediction_lock_temporal.json','externa_prediction_lock_territorial.json','externa_first_reserved_open.json']:
        p=OLD/'validacion'/name;locks.append({'path':str(p.relative_to(STUDY)),'sha256':sha(p),'lock_metadata':json.loads(p.read_text())})
    event={'status':'PASS2024_PATH_SCHEMA_LOCK_INVENTORY_ONLY','files':inventory,'existing_locks':locks,
           'territorial_split_rule':'int(sha256(ENDES-phase2-20261007|+province)[:8],16)%5==0 defines heldoutprovinces',
           'reserved_outcome_values_loaded':False,'note':'Existing2024 evaluation already observed in previous phase; never treat it as fresh independent test for selecting new improvements. Freeze new models first, use fresh synthetic evaluation and disclose2024 reuse.'}
    (VAL/'align_2024_inventory_and_existing_locks.json').write_text(json.dumps(event,ensure_ascii=False,indent=2))
    print(json.dumps({'status':proof['status'],'survey_counts':counts,'outputs':len(outputs),'2024_outcome_values_read':False}),flush=True)

if __name__=='__main__':main()
