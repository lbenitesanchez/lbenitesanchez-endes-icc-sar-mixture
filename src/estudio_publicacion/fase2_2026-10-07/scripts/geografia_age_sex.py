#!/usr/bin/env python3
"""Exact 2017 census district child population cells, separate from frozen X.
Under-one ages are in completed months; ages1-4 are completed years. Single-
condition official REDATAM requests avoid a server rejection of complex filters.
These are 2017 census counts, never ENDES-year forecasts or interpolated totals.
"""
from pathlib import Path
import json,csv,hashlib,datetime
from geografia_helpers import counts,sha,ROOT,SRC,OUT,VAL,writecsv
months,q=counts('census_age_months')
M={m:counts(f'census_sex_month_{m}')[0]for m in range(6,12)}
Y={y:counts(f'census_sex_year_{y}')[0]for y in range(1,5)}
aux=OUT/'aux_verified_2017_2018.csv';initial_sha=sha(aux)
# Import of geografia_aux should not re-write the main X: protected caller guard.
assert initial_sha=='cbec2fc142ec0ee06dd247230f0b2253e11af04d0e0e76e8374f30b456df7502'
checks=[];out=[];compact=[]
for k in sorted(months):
 for m in range(6,12):
  src=months[k]['counts'][f'Edad de {m} meses'];got=sum(M[m][k]['counts'].values());assert src==got,(k,m,src,got)
  checks.append({'name':'exact_under_one_month_validation','ubigeo':k,'month':m,'unfiltered_month_sex_sum':got,'filtered_age_zero_month_count':src,'pass':True})
 row={'ubigeo':k,'census_reference_year':2017}
 for band,band_m,band_y in [('6_11',range(6,12),[]),('12_23',[],[1]),('6_23',range(6,12),[1]),('24_59',[],range(2,5)),('6_59',range(6,12),range(1,5))]:
  cells={sex:sum(M[m][k]['counts'][sex]for m in band_m)+sum(Y[y][k]['counts'][sex]for y in band_y)for sex in ('Hombre','Mujer')};total=sum(cells.values());row['population_'+band]=total
  for sex,cnt in cells.items():
   code='male'if sex=='Hombre'else'female';row['population_'+band+'_'+code]=cnt
   out.append({'ubigeo':k,'age_band_completed_months':band,'sex':code,'census_population_count':cnt,'reference_year':2017,'fraction_within_age_band':cnt/total if total else '', 'population_universe':'2017_census_population_in_official_CPV2017DI','current_year_population_status':'not_available_no_interpolation_no_projection'})
  if total:assert abs(sum(r['fraction_within_age_band']for r in out[-2:])-1)<1e-14
 assert row['population_6_23']==row['population_6_11']+row['population_12_23']
 assert row['population_6_59']==row['population_6_23']+row['population_24_59'];compact.append(row)
# Current and frozen covariate bytes must not change.
assert sha(aux)==initial_sha
writecsv(OUT/'population_age_sex_2017_long.csv',out);writecsv(OUT/'population_age_sex_2017_counts.csv',compact);writecsv(VAL/'geografia_age_month_consistency_checks.csv',checks)
files=[OUT/'population_age_sex_2017_long.csv',OUT/'population_age_sex_2017_counts.csv',VAL/'geografia_age_month_consistency_checks.csv']
report={'status':'PASS_EXACT_CENSUS_2017_CHILD_AGE_SEX_AGGREGATES','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'districts':1874,'long_rows':len(out),'month_identity_checks':len(checks),'all_month_identity_checks_pass':True,'all_joint_band_additivity_checks_pass':True,'interpolated_cells':0,'source_queries':11,'national_band_counts':{band:sum(r['population_'+band]for r in compact)for band in ('6_11','12_23','6_23','24_59','6_59')},'files':{str(p.relative_to(ROOT)):sha(p)for p in files},'frozen_aux_sha256_unchanged':initial_sha,'age_under_one_source':'Poblacio.C5P41M completed months0..11; verifies against separate Poblacio.C5P41=0 query in all districts','older_child_age_source':'Poblacio.C5P41 completed years1..4','sex_source':'Poblacio.SEXO: Hombre/Mujer','limitations':['2017 census composition may differ from ENDES2019/2024/2025 target populations.','No official exact district six-to-59month population projection for 2024/2025 was verified.','No poststratification to new17districts or boundary splits is performed.','Joint population age/sex margins are supplementary and do not change the frozen phase2 model specification.']}
(VAL/'geografia_age_sex_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:report[k]for k in ('status','districts','month_identity_checks','national_band_counts')},indent=2))
