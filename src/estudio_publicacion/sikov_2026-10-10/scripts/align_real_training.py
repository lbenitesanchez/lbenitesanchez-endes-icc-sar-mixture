"""Prepare same-information real fits using only authorized training responses.

2019 full development may be prepared before recipe freeze. 2024 territorial
training requires a frozen protocol with approved_for_real_training=true. Test
response paths are never accepted or opened. No model is fitted by this script.
"""
from pathlib import Path
import argparse,sys,hashlib,json,datetime
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT.parent;OLD=STUDY/'fase2_2026-10-07'
sys.path.insert(0,str(OLD/'scripts'))
from diseno_calibration import frame_from_households,taylor_contributions
sys.path.insert(0,str(STUDY/'scripts'))
from models import spatial_basis
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(year,output,protocol=None):
    output=output.resolve()
    if output.exists():raise FileExistsError(output)
    protocol_sha=None
    if year==2024:
        if protocol is None or not protocol.is_file():raise RuntimeError('2024 training preparation requires frozen protocol')
        frozen=json.loads(protocol.read_text())
        if not frozen.get('approved_for_real_training',False):raise RuntimeError('Frozen recipe has not approved2024 territorial training access')
        protocol_sha=sha(protocol)
    if year==2019:
        childpath=STUDY/'datos/private_competitors/children_2019.csv'
        hhpath=STUDY/'datos/private_competitors/households_2019.csv'
        scope='2019_full_development'
    else:
        childpath=OLD/'datos/private_external/children_2024_training.csv'
        hhpath=OLD/'datos/private_external/households_2024.csv'
        scope='2024_territorial_training_only'
    if 'test' in childpath.name or 'scoring' in childpath.name:raise RuntimeError('Test outcomes forbidden')
    auxpath=ROOT/'datos/aux_sikov14_official.csv'
    aux=pd.read_csv(auxpath,dtype={'ubigeo':str,'dept_code':str,'prov_code':str})
    units=pd.read_csv(childpath,dtype={'ubigeo':str,'psu_key':str,'prov_code':str})
    households=pd.read_csv(hhpath,dtype={'ubigeo':str,'psu_key':str})
    if not units.year.eq(year).all():raise ValueError('Wrong declared survey year')
    if not units.age_months.between(6,59).all()or not units.de_facto.eq(1).all():raise ValueError('Wrong common age/residence scope')
    if year==2024:
        held=units.ubigeo.str[:4].map(lambda province:int(hashlib.sha256(('ENDES-phase2-20261007|'+province).encode()).hexdigest()[:8],16)%5==0)
        if held.any():raise RuntimeError('Reserved province outcome leaked into training')
    frame=frame_from_households(households)
    lookup=dict(zip(aux.ubigeo,np.arange(len(aux))))
    units['area']=units.ubigeo.map(lookup)
    excluded=units.loc[units.area.isna()].groupby('ubigeo').size().to_dict()
    units=units.loc[units.area.notna()].copy();units['area']=units.area.astype(int)
    units['y_anemia']=units.y_anemia_old
    units['weight_anemia']=np.where(np.isfinite(units.y_anemia),units.weight,np.nan)
    units['weight_stunting']=np.where(np.isfinite(units.y_stunting),units.weight,np.nan)
    units['psu']=units.psu_key
    valid=(np.isfinite(units.y_anemia)|np.isfinite(units.y_stunting))&units.weight.gt(0)
    idx=np.sort(units.loc[valid,'area'].unique())
    results=[taylor_contributions(frame,units,ep,idx)for ep in ['anemia','stunting']]
    y=np.column_stack([r['y']for r in results]);kish=np.column_stack([r['n_kish']for r in results])
    n=np.column_stack([r['n_records']for r in results]);df=np.column_stack([r['df_proxy']for r in results])
    rawdiag=np.column_stack([r['variance']for r in results]);D_raw=np.zeros((len(idx),2,2))
    for j in range(2):D_raw[:,j,j]=rawdiag[:,j]
    for _,row_idx in frame.groupby('stratum',sort=True).indices.items():
        ii=np.asarray(row_idx);m=len(ii);ua=results[0]['scores'][ii];ub=results[1]['scores'][ii]
        D_raw[:,0,1]+=(m/(m-1)*(np.sum(ua*ub,axis=0)-ua.sum(axis=0)*ub.sum(axis=0)/m)if m>1 else np.sum(ua*ub,axis=0))
    D_raw[:,1,0]=D_raw[:,0,1]
    padj=(y*kish+.5)/(kish+1);binom=padj*(1-padj)/kish
    gvf=np.zeros_like(rawdiag);deff=[]
    for j in range(2):
        ok=np.isfinite(rawdiag[:,j])&(rawdiag[:,j]>1e-15)&np.isfinite(binom[:,j])&(binom[:,j]>0)
        if not ok.any():raise ValueError('Cannot estimate training-only median design effect')
        deff.append(float(np.median(rawdiag[ok,j]/binom[ok,j])))
        gvf[:,j]=deff[-1]*binom[:,j]
    shrink=df/(df+4.)
    working=shrink*rawdiag+(1-shrink)*gvf
    repair=(rawdiag<=1e-15)|(y<=0)|(y>=1)
    working=np.where(repair,gvf,working)
    # Each endpoint is fitted independently. The crosscovariance is retained
    # only as a documented moment diagnostic and is never used to infer a joint
    # bivariate outcome posterior.
    D=np.zeros_like(D_raw)
    for j in range(2):D[:,j,j]=working[:,j]
    denom=np.sqrt(rawdiag[:,0]*rawdiag[:,1]);corr=np.divide(D_raw[:,0,1],denom,out=np.zeros(len(idx)),where=denom>1e-15)
    corr=np.clip(corr,-.95,.95);D[:,0,1]=D[:,1,0]=corr*np.sqrt(working[:,0]*working[:,1])
    with np.load(ROOT/'datos/aux_sikov14_common_inputs.npz')as z:
        base={k:z[k]for k in z.files}
    if not np.array_equal(base['ubigeo'],aux.ubigeo):raise AssertionError('District order mismatch')
    basis=spatial_basis(aux[['latitude','longitude']].to_numpy(float),rank=40)
    if year==2019:
        old=np.load(ROOT/'datos/fit_inputs_2019_6_59_harmonized.npz')
        if not np.array_equal(idx,old['sample_idx'])or not np.allclose(y,old['y'],atol=1e-12,rtol=0):raise AssertionError('Prepared2019 ratios differ from independently validatedR')
        if not np.allclose(D_raw,old['D'],atol=1e-12,rtol=0):raise AssertionError('Prepared2019 full Taylor covariance differs from independently validatedR')
    output.mkdir(parents=True);private=output/'private';private.mkdir()
    keep=['child_id','household_id','year','ubigeo','prov_code','area','age_months','sex','weight','psu','stratum','psu_key','de_facto','y_anemia','y_stunting','weight_anemia','weight_stunting']
    units[keep].to_csv(private/'units.csv',index=False)
    aux.to_csv(output/'area_frame.csv',index=False)
    np.savez_compressed(output/'fit_inputs.npz',**base,y=y,D=D,D_raw=D_raw,D_working=D,GVF=gvf,
                        n_eff=kish,kish_n=kish,n=n,df_proxy=df,sample_idx=idx,basis=basis,
                        endpoint_names=np.array(['anemia','stunting']),test_idx=np.array([],int),zero_variance_mask=repair,
                        data_year=np.array(year),common_anemia_definition=np.array('WHO2011 HC56<110'))
    calibration=aux.iloc[idx][['ubigeo','poverty_stratum']].copy()
    for j,ep in enumerate(['anemia','stunting']):
        calibration['y_'+ep]=y[:,j];calibration['D_raw_'+ep]=rawdiag[:,j];calibration['D_working_'+ep]=working[:,j]
        calibration['df_proxy_'+ep]=df[:,j];calibration['GVF_'+ep]=gvf[:,j];calibration['n_kish_'+ep]=kish[:,j]
    calibration.to_csv(output/'area_calibration.csv',index=False)
    proof={'status':'PASS_COMMON_REAL_TRAINING_PREPARATION','year':year,'scope':scope,'protocol_sha256':protocol_sha,
           'source_sha256':{'children':sha(childpath),'full_household_design':sha(hhpath),'aux14':sha(auxpath),'script':sha(Path(__file__))},
           'private_units_path':str((private/'units.csv').relative_to(ROOT)),'private_units_sha256':sha(private/'units.csv'),
           'area_count':len(aux),'training_domains':len(idx),'training_children':len(units),'full_households':len(households),
           'fullPSUs':len(frame),'fullstrata':frame.stratum.nunique(),'heldout_outcome_rows_read':False,
           'same_covariate_info_all_methods':True,'feature_names':base['covariate_names'].tolist(),'coordinates_npz_order':['latitude','longitude'],
           'X_default':'raw intercept+14, X_standardized separately supplied','anemia_definition':'WHO2011 HC56<110 in both2019 and2024',
           'common_universe':'de_facto children6to59months; outcome-specific available-case denominators',
           'variance_policy':{'raw':'fullPSU stratified firststage Taylor','working':'aD_raw+(1-a)GVF, a=df_proxy/(df_proxy+4); zero/boundary replace byGVF',
                'GVF':'training positiveD median(D_raw/[padj(1-padj)/Kish]) times padj(1-padj)/Kish','padj':'(Kish*y+.5)/(Kish+1)',
                'df_proxy':'endpoint PSUs with measuredoutcome minus representedstrata, lowerboundedzero','median_design_effect':deff,
                'crosscovariance_scope':'retained moment diagnostic only; all endpointmodels independent'},
           'unsupported_direct_only':excluded,'previous_sources_modified':False,'private_microdata_must_be_excluded_from_public_packages':True,
           'output_sha256':{p.name:sha(p)for p in output.iterdir()if p.is_file()}}
    (output/'preparation_manifest.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2))
    print(json.dumps({k:proof[k]for k in ['status','year','scope','training_domains','fullPSUs','fullstrata','heldout_outcome_rows_read']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--year',type=int,choices=[2019,2024],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--protocol',type=Path)
    args=parser.parse_args();prepare(args.year,args.output,args.protocol)
