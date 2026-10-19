"""Read-only audit of locked training preparation, no reserved-test reads."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ANEMIA={1:['intercept','Altitud','Internet','Castellano'],2:['intercept','Altitud','Cemento','Agua','Internet','SIS','Castellano','Rural'],3:['intercept','Altitud','Cemento','Desague','SIS','Castellano']}
STUNTING={1:['intercept','Altitud','Refrig','Internet','Analfabet'],2:['intercept','Agua','Elect','Refrig','NoSeguro','Analfabet'],3:['intercept','Altitud','Elect','Refrig','SIS','Analfabet']}
def audit(directory,protocol):
    proof=json.loads((directory/'preparation_manifest.json').read_text());year=proof['year']
    if year==2024 and proof['protocol_sha256']!=sha(protocol):raise AssertionError('2024prep protocol differs')
    with np.load(directory/'fit_inputs.npz')as a:z={key:a[key]for key in a.files}
    upath=ROOT/proof['private_units_path']
    if not upath.is_relative_to(directory)or 'test' in upath.name.lower():raise AssertionError('Nontraining unit path')
    if sha(upath)!=proof['private_units_sha256']:raise AssertionError('Private units differ from preparation manifest')
    u=pd.read_csv(upath,dtype={'ubigeo':str,'psu_key':str,'psu':str,'prov_code':str})
    checks=[];details={}
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    check('common WHO2011 and children6–59 scope',u.year.eq(year).all()and u.age_months.between(6,59).all()and u.de_facto.eq(1).all()and proof['anemia_definition']=='WHO2011 HC56<110 in both2019 and2024')
    check('training cohort unique child identifiers',not u.child_id.duplicated().any())
    if year==2024:
        held=u.ubigeo.str[:4].map(lambda p:int(hashlib.sha256(('ENDES-phase2-20261007|'+p).encode()).hexdigest()[:8],16)%5==0)
        check('no heldout2024 province units',not held.any())
    check('all1874 support shared14 covariates',len(z['ubigeo'])==1874 and z['X_raw'].shape==(1874,15)and np.isfinite(z['X_raw']).all())
    check('private area index maps to public frame exactly',np.array_equal(z['ubigeo'][u.area.to_numpy(int)],u.ubigeo))
    for j,endpoint in enumerate(['anemia','stunting']):
        valid=u['y_'+endpoint].isin([0,1])&u.weight.gt(0)&np.isfinite(u.weight);v=u.loc[valid]
        sums=v.weight.groupby(v.area).sum();weighted=(v.weight*v['y_'+endpoint]).groupby(v.area).sum()/sums
        expected=weighted.reindex(z['sample_idx']).to_numpy(float);obs=np.isfinite(expected)
        check(endpoint+' weighted endpoint-specific ratios',np.allclose(expected,z['y'][:,j],atol=1e-12,rtol=0,equal_nan=True))
        check(endpoint+' missing responses do not become zeros',np.array_equal(np.isfinite(expected),np.isfinite(z['y'][:,j])))
        ne=(sums**2/(v.weight**2).groupby(v.area).sum()).reindex(z['sample_idx']).to_numpy(float)
        check(endpoint+' Kish n independently reproduced',np.allclose(ne,z['kish_n'][:,j],atol=1e-9,rtol=0,equal_nan=True))
        psus=v.groupby('area').psu_key.nunique().reindex(z['sample_idx']).to_numpy(float)
        strata=v.groupby('area').stratum.nunique().reindex(z['sample_idx']).to_numpy(float)
        df=np.where(obs,np.maximum(psus-strata,0),0.)
        check(endpoint+' df PSUminusrepresentedstrata',np.allclose(df,z['df_proxy'][:,j],atol=0,rtol=0,equal_nan=True))
        raw=z['D_raw'][:,j,j];padj=(expected*ne+.5)/(ne+1);binom=padj*(1-padj)/ne
        pos=obs&(raw>1e-15)&np.isfinite(binom)&(binom>0);deff=np.median(raw[pos]/binom[pos]);gvf=deff*binom
        work=df/(df+4)*raw+4/(df+4)*gvf;replace=(raw<=1e-15)|(expected<=0)|(expected>=1);work[replace]=gvf[replace]
        check(endpoint+' Dworking/GVF independently reproduced',np.allclose(work,z['D'][:,j,j],atol=1e-12,rtol=0,equal_nan=True))
        check(endpoint+' every responding domain has positiveD',np.isfinite(work[obs]).all()and(work[obs]>0).all())
        details[endpoint]={'responding_domains':int(obs.sum()),'missing_endpoint_domains':int((~obs).sum()),'rawDzero_responders':int(np.sum(obs&(raw<=1e-15))),
                           'median_design_effect':float(deff),'stratum_designs':[]}
        for model in ['FH','SAR']:
            for s in [1,2,3]:
                names=(ANEMIA if endpoint=='anemia'else STUNTING)[s].copy()
                if model=='SAR'and endpoint=='anemia'and s==3:names=['intercept','Cemento','SIS','Castellano']
                columns=[list(z['covariate_names']).index(name)for name in names]
                mask=obs&(z['strata'][z['sample_idx']]==s);matrix=z['X_raw'][z['sample_idx'][mask]][:,columns]
                rank=np.linalg.matrix_rank(matrix)
                check(endpoint+' '+model+' stratum'+str(s)+' published subset fullrank',rank==len(names))
                fullmatrix=z['X_standardized'][z['sample_idx'][mask]]
                details[endpoint]['stratum_designs'].append({'model':model,'stratum':s,'domains':int(mask.sum()),'columns':names,
                                                          'published_subset_rank':int(rank),'full14_plus_intercept_rank':int(np.linalg.matrix_rank(fullmatrix))})
    check('complete fixed census scale and q40 geographic basis',np.allclose(z['X_standardized'][:,1:].mean(axis=0),0,atol=1e-12)and z['basis'].shape==(1874,40))
    for name,digest in proof['output_sha256'].items():check('prepared output unchanged '+name,sha(directory/name)==digest)
    report={'status':'PASS_INDEPENDENT_COMMON_REAL_TRAINING_AUDIT','year':year,'scope':proof['scope'],'checks_passed':len(checks),'checks':checks,'endpoint_details':details,
            'private_units_sha256':sha(upath),'preparation_manifest_sha256':sha(directory/'preparation_manifest.json'),
            'protocol_sha256':sha(protocol),'reserved_test_outcome_values_read':False,'no_model_or_statistical_source_mutation':True,'auditor_source_sha256':sha(Path(__file__))}
    destination=ROOT/'validacion'/('align_real'+str(year)+'_training_independent_audit.json');destination.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:report[k]for k in ['status','year','checks_passed','reserved_test_outcome_values_read']}),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args();audit(args.directory.resolve(),args.protocol.resolve())
