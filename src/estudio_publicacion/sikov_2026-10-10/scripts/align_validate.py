"""Check response alignment, full survey covariance and official predictors."""
from pathlib import Path
import json,hashlib,os
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT.parent;OUT=ROOT/'datos';VAL=ROOT/'validacion'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(name,condition,detail=None):
    checks.append({'name':name,'pass':bool(condition),'detail':detail})
    if not condition:raise AssertionError(name)
def main():
    p=np.load(OUT/'fit_inputs_2019_6_59_harmonized.npz')
    aux=pd.read_csv(OUT/'aux_sikov14_official.csv',dtype={'ubigeo':str})
    old=pd.read_csv(STUDY/'datos/processed/areas_2019_6_59.csv',dtype={'ubigeo':str}).set_index('ubigeo')
    idx=p['sample_idx'];y=p['y'];D=p['D'];codes=p['ubigeo']
    check('1874 auxiliary domains and 888 jointly observed',len(codes)==1874 and len(idx)==888)
    check('unique correctly padded district codes',len(set(codes))==1874 and all(len(s)==6 and s.isdigit()for s in codes))
    check('all14 official predictors finite',np.isfinite(p['X_raw']).all() and p['X_raw'].shape==(1874,15))
    check('common raw X names and values match official frame',np.allclose(aux[list(p['covariate_names'][1:])],p['X_raw'][:,1:],rtol=0,atol=1e-14))
    check('response-free standardization mean and scale',np.allclose(p['X_standardized'][:,1:].mean(axis=0),0,atol=1e-12)and np.allclose(p['X_standardized'][:,1:].std(axis=0),1,atol=1e-12))
    expected=old.reindex(codes[idx])
    for j,(yn,vn,nn)in enumerate([('y_anemia_old','var_anemia_old','n_eff_anemia_old'),('y_stunting','var_stunting','n_eff_stunting')]):
        check('2019 response exact prior audited reconstruction '+yn,np.allclose(y[:,j],expected[yn],rtol=0,atol=1e-14))
        check('2019 Taylor D exact prior audited reconstruction '+vn,np.allclose(D[:,j,j],expected[vn],rtol=0,atol=1e-14))
        check('Kish n exact prior audited reconstruction '+nn,np.allclose(p['n_eff'][:,j],expected[nn],rtol=0,atol=1e-10))
    check('shared covariance symmetric and positive semidefinite',np.allclose(D,D.swapaxes(1,2))and np.linalg.eigvalsh(D).min()>=-1e-12)
    check('zero sampling variances retained transparently',np.sum(D[:,0,0]<=1e-15)==448 and np.sum(D[:,1,1]<=1e-15)==497)
    check('poverty boundaries explicit midpoint proxy, published counts differ',aux.poverty_stratum.value_counts().sort_index().tolist()==[786,888,200])
    for name in ['W_knn6_planar_stratum','W_knn4_planar_stratum','W_knn8_planar_stratum','W_knn6_haversine_stratum','W_knn6_planar_global']:
        W=p[name];k=4 if 'knn4_'in name else 8 if 'knn8_'in name else 6
        check(name+' rowstd/no self/finite/nonnegative',np.allclose(W.sum(axis=1),1,atol=1e-6)and np.all(np.diag(W)==0)and np.isfinite(W).all()and np.all(W>=0))
        check(name+' expected neighbors',np.all(np.sum(W>0,axis=1)==k))
        if name.endswith('stratum'):
            a,b=np.where(W>0);check(name+' all edges stay within poverty stratum',np.all(p['strata'][a]==p['strata'][b]))
    qa=pd.read_csv(VAL/'align_survey2019_independent_survey_checks.csv')
    check('independent survey-package full-frame covariance8checks',len(qa)==8 and qa.max_abs_cov_difference.max()<1e-10)
    check('independent survey frame3253PSU248strata',qa.full_psus.eq(3253).all()and qa.full_strata.eq(248).all())
    inventory=json.loads((VAL/'align_2024_inventory_and_existing_locks.json').read_text())
    check('2024 training/test inventoried without loading response values',not inventory['reserved_outcome_values_loaded']and len(inventory['files'])==4)
    # Author compilation is a diagnostic only, not used as the common predictors.
    source=Path(os.environ.get('ENDES_AUX_ROOT', 'PRIVATE_INPUTS'))/'consolidado_nuevo_completo_con_directos.xlsx'
    orig=pd.read_excel(source);orig['ubigeo']=orig.Ubigeo.astype(int).astype(str).str.zfill(6);orig=orig.set_index('ubigeo').reindex(codes)
    comparison={}
    for feature,col in [('Cemento','PAREDES Ladrillo o bloque de cemento'),('Internet','Internet'),('SIS','Seg_SIS'),('EsSalud','Seg_EsSalud'),('NoSeguro','Seg_NoSeguro')]:
        delta=float(np.max(np.abs(aux[feature].to_numpy(float)-orig[col].to_numpy(float))))
        comparison[feature]={'maximum_absolute_difference_author_compilation':delta,'exact_to_1e-12':delta<1e-12}
        check('new official extraction '+feature+' agrees with author workbook',delta<1e-12,comparison[feature])
    report={'status':'PASS_ALIGNED_DATA_AND_OFFICIAL_SOURCE_CHECKS','checks':checks,'checks_passed':len(checks),
            'new_census_vs_author_workbook':comparison,'all_previous_data_readonly':True,
            'reserved_outcome_values_read':False,'exact_sikov_numerical_replication_claim_permitted':False,
            'source_sha256':sha(Path(__file__))}
    (VAL/'align_final_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:report[k]for k in ['status','checks_passed','reserved_outcome_values_read','exact_sikov_numerical_replication_claim_permitted']}),flush=True)
if __name__=='__main__':main()
