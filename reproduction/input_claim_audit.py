#!/usr/bin/env python3
"""Recompute Appendix A calibration and census numerical claims.

Only 2019 training records are read; no model fit or evaluation truth is used.
Real respondent rows stay external and are never written or printed. Census
counts are independently parsed from literal archived public HTML, then checked
against query JSON and the public district aggregate CSV.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

PHASE='fase2_2026-10-07'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def audit_calibration(source,reference,private,preparation):
    proof=json.loads(preparation.read_text())
    if not proof.get('fresh_microdata_preparation_from_SAV'):
        raise RuntimeError('Fresh SAV reconstruction evidence is required')
    path=source/PHASE/'scripts/diseno_calibration.py'
    module=load(path,'input_claim_unchanged_calibration')
    frozen=json.loads((source/PHASE/'protocol_freeze.json').read_text())
    relative='scripts/diseno_calibration.py'
    expected=frozen['source_sha256'].get(relative) or frozen['source_sha256'].get(PHASE+'/'+relative)
    if expected and sha(path)!=expected:raise RuntimeError('Frozen calibration source differs')
    child=private/'datos/private_competitors/children_2019.csv'
    household=private/'datos/private_competitors/households_2019.csv'
    aux=reference/PHASE/'datos/geografia/aux_verified_2017_2018.csv'
    areas=pd.read_csv(aux,dtype={'ubigeo':str})
    units=pd.read_csv(child,dtype={'ubigeo':str,'psu_key':str})
    if 'year' not in units or not units.year.eq(2019).all():
        raise RuntimeError('The numerical input audit is restricted to 2019 training')
    lookup=dict(zip(areas.ubigeo,np.arange(len(areas))))
    units['area']=units.ubigeo.map(lookup)
    units=units[units.area.notna()].copy();units['area']=units.area.astype(int)
    units['y_anemia']=units.y_anemia_who2024
    for endpoint in ('anemia','stunting'):
        units['weight_'+endpoint]=np.where(np.isfinite(units['y_'+endpoint]),units.weight,np.nan)
    frame=module.frame_from_households(pd.read_csv(household,dtype={'psu_key':str}))
    valid=(np.isfinite(units.y_anemia)|np.isfinite(units.y_stunting))&units.weight.gt(0)
    idx=np.sort(units.loc[valid,'area'].unique())
    counts,checks=module.stratified_psu_counts(frame)
    summaries=[]
    for endpoint in ('anemia','stunting'):
        result=module.calibrate(frame,units,idx,endpoint,counts=counts)
        direct,central=result['direct'],result['central']
        summaries.append({'endpoint':endpoint,'domains':len(idx),
            'only_one_observed_PSU':int((direct['n_psu']==1).sum()),
            'full_frame_singleton_component_domains':int((direct['singleton_contribution']>0).sum()),
            'n_calibration_greater_than_kish':int((central['effective_n']>direct['n_kish']).sum()),
            'central_to_kish_ratio_median':float(np.nanmedian(central['effective_n']/direct['n_kish'])),
            'central_to_kish_ratio_median_printed':f"{np.nanmedian(central['effective_n']/direct['n_kish']):.2f}"})
    return {'status':'FRESH_2019_WHO2024_CALIBRATION_CLAIMS_COMPUTED','endpoint_summary':summaries,
        'full_household_PSUs':len(frame),'full_household_strata':int(frame.stratum.nunique()),
        'full_household_singleton_strata':int(frame.groupby('stratum').size().eq(1).sum()),
        'unchanged_calibration_source_sha256':sha(path),'fresh_preparation_receipt_sha256':sha(preparation),
        'input_sha256':{'fresh_household_csv':sha(household),'fresh_training_child_csv':sha(child),
                       'verified_legacy_aux_csv':sha(aux)},
        'replicate_expectation_checks':checks,'WHO2024_scope_separate_from_WHO2011_Sikov14':True,
        'model_fits_dispatched':False,'heldout_or_synthetic_truth_read':False,'individual_records_exported':False}

def audit_census(source,reference):
    parser_path=source/PHASE/'scripts/geografia_redatam.py'
    parser=load(parser_path,'input_claim_literal_HTML_parser')
    base=reference/PHASE;files={}
    names=['census_age_months']+[f'census_sex_month_{m}'for m in range(6,12)]+[f'census_sex_year_{y}'for y in range(1,5)]
    parsed={}
    for name in names:
        query=base/'fuentes/geografia'/(name+'_query.json');q=json.loads(query.read_text())
        if len(q['tables'])!=1:raise ValueError('One district aggregate table required')
        table=q['tables'][0];path=base/table['path']
        if sha(path)!=table['sha256']:raise RuntimeError('Census table hash differs')
        reader=parser.Rows();reader.feed(path.read_text(encoding='utf-8-sig'))
        if reader.rows!=table['rows']:raise RuntimeError('Literal HTML rows differ from query JSON rows')
        header=next(row for row in reader.rows if 'Código'in row)
        code,domain=header.index('Código'),header.index('Distrito')
        columns=[i for i in range(domain+1,len(header))if header[i]]
        counts={}
        for row in reader.rows:
            if len(row)>code and re.fullmatch('[0-9]{6}',row[code]):
                if row[code]in counts:raise RuntimeError('Duplicate census district code')
                counts[row[code]]={header[i]:int(row[i].replace(' ',''))for i in columns}
        if len(counts)!=1874:raise RuntimeError('Historical district universe differs')
        parsed[name]=counts
        files[str(query.relative_to(base))]=sha(query);files[str(path.relative_to(base))]=sha(path)
    universes={tuple(sorted(rows))for rows in parsed.values()}
    if len(universes)!=1:raise RuntimeError('Census query district supports differ')
    districts=sorted(parsed['census_age_months']);records=[];identity_checks=0
    for district in districts:
        for month in range(6,12):
            if parsed['census_age_months'][district][f'Edad de {month} meses']!=sum(parsed[f'census_sex_month_{month}'][district].values()):
                raise RuntimeError('Exact month/sex identity fails')
            identity_checks+=1
        row={'ubigeo':district}
        for band,months,years in (('6_11',range(6,12),()),('12_23',(),(1,)),
                                 ('6_23',range(6,12),(1,)),('24_59',(),range(2,5)),
                                 ('6_59',range(6,12),range(1,5))):
            for sex,label in (('Hombre','male'),('Mujer','female')):
                row['population_'+band+'_'+label]=(sum(parsed[f'census_sex_month_{m}'][district][sex]for m in months)+
                                                    sum(parsed[f'census_sex_year_{y}'][district][sex]for y in years))
            row['population_'+band]=row['population_'+band+'_male']+row['population_'+band+'_female']
        if row['population_6_59']!=row['population_6_23']+row['population_24_59']:
            raise RuntimeError('Age-band additivity fails')
        records.append(row)
    fresh=pd.DataFrame(records).set_index('ubigeo').sort_index()
    expected_path=base/'datos/geografia/population_age_sex_2017_counts.csv'
    expected=pd.read_csv(expected_path,dtype={'ubigeo':str}).set_index('ubigeo').sort_index()
    if not fresh.index.equals(expected.index)or not np.array_equal(fresh.to_numpy(),expected[fresh.columns].to_numpy()):
        raise RuntimeError('Fresh literal HTML aggregate reconstruction differs from archived count CSV')
    return {'status':'PASS_LITERAL_HTML_CENSUS_AGE_SEX_RECONSTRUCTION','districts':len(fresh),
        'month_identity_checks':identity_checks,'all_month_identity_checks_pass':True,
        'national_band_counts':{band:int(fresh['population_'+band].sum())for band in ('6_11','12_23','6_23','24_59','6_59')},
        'public_aggregate_cells_verified':int(fresh.size),'literal_HTML_parser_source_sha256':sha(parser_path),
        'aggregate_count_reference_sha256':sha(expected_path),'source_queries':len(names),
        'source_sha256':files,'network_reacquisition_performed':False,
        'new_literal_source_parse_performed':True,'individual_records_read':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','reference-study-root','fresh-private-study-root','real-preparation-receipt','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result={'status':'FRESH_INPUT_CALIBRATION_AND_CENSUS_CLAIMS_AUDITED',
        'calibration':audit_calibration(args.source_root,args.reference_study_root,args.fresh_private_study_root,args.real_preparation_receipt),
        'census':audit_census(args.source_root,args.reference_study_root),'helper_sha256':sha(__file__),
        'model_fits_or_result_truth_reads_performed':False,'article_reproduction_confirmed':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'calibration_endpoints':result['calibration']['endpoint_summary'],
        'census_national_counts':result['census']['national_band_counts'],
        'census_identity_checks':result['census']['month_identity_checks'],'individual_records_exported':False},indent=2))

if __name__=='__main__':main()
