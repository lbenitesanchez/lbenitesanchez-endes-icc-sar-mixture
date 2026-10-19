#!/usr/bin/env python3
"""Reconstruct official-source historical covariates and strict-code crosswalk.
Requires geografia_redatam.py, geografia_poverty.R, geografia_directorio.R and
native Poppler pdftotext -layout for the official 2019 directory. No ENDES reads.
"""
from pathlib import Path
import csv,json,hashlib,re,unicodedata,collections,datetime,math
ROOT=Path(__file__).resolve().parents[1];STUDY=ROOT.parent;SRC=ROOT/'fuentes/geografia';OUT=ROOT/'datos/geografia';VAL=ROOT/'validacion'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def readcsv(p):return list(csv.DictReader(p.open()))
def writecsv(p,rows):
 with p.open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def norm(x):return re.sub(r'\s+',' ',''.join(c for c in unicodedata.normalize('NFD',x.upper()) if not unicodedata.combining(c))).strip()

def counts(name):
 q=json.loads((SRC/(name+'_query.json')).read_text());assert len(q['tables'])==1
 t=q['tables'][0];assert sha(ROOT/t['path'])==t['sha256']
 rr=t['rows'];h=next(r for r in rr if 'Código' in r);ci=h.index('Código');di=h.index('Distrito');cols=[i for i in range(di+1,len(h))if h[i]]
 d={}
 for r in rr:
  if len(r)>ci and re.fullmatch('[0-9]{6}',r[ci]):
   vals={h[i]:int(r[i].replace(' ','')) for i in cols};assert all(v>=0 for v in vals.values());d[r[ci]]={'name':r[di],'counts':vals,'denominator':sum(vals.values())}
 assert len(d)==1874 and len(d)==len(set(d));return d,q

specs={'electricity':('census_electricity',['Sí tiene alumbrado eléctrico']),'sewer':('census_sewer',['Red pública de desagüe dentro de la vivienda','Red pública de desagüe fuera de la vivienda, pero dentro de la edificación']),'floor_earth':('census_floor',['Tierra']),'water_piped':('census_water',['Red pública dentro de la vivienda']),'biomass':('census_biomass',['Sí usa bosta, estiércol']),'refrigerator':('census_refrigerator',['Sí tiene refrigeradora o congeladora']),'gas_glp':('census_gas_glp',['Sí usa gas (balón GLP)']),'gas_natural':('census_gas_natural',['Sí usa gas natural']),'spanish':('census_spanish',['Castellano']),'illiteracy':('census_illiteracy',['No sabe leer y escribir'])}
old=readcsv(STUDY/'datos/processed/aux_all_districts.csv');old_by={r['ubigeo']:r for r in old};poverty={r['ubigeo']:r for r in readcsv(OUT/'poverty_official_2018.csv')};directory={r['ubigeo']:r for r in readcsv(OUT/'directorio_oficial_distritos_2025.csv')}
assert len(old_by)==1874 and set(old_by)==set(poverty) and set(old_by)<=set(directory)
data={};queries={}
for feature,(name,numerator)in specs.items():
 d,q=counts(name);assert set(d)==set(old_by);data[feature]=d;queries[feature]=q
# Exact one-to-one location verification: all 1874 official 2019 capital rows are
# recovered; all archived decimal coordinates equal their official DMS values.
pat=re.compile(r'(\d{1,3})[º°]\s*(\d{1,2})[’ʹ\']\s*(\d{1,2})[”"\u2033]');geo=[]
for page,txt in enumerate((SRC/'directorio_2019.txt').read_text().split('\f'),1):
 for lineno,ln in enumerate(txt.splitlines(),1):
  mm=list(pat.finditer(ln))
  if len(mm)==2:
   vals=[-(int(m[1])+int(m[2])/60+int(m[3])/3600)for m in mm];left=ln[:mm[0].start()];ac=re.search(r'(\d{1,3}(?:\s+\d{3})*|\d+)\s*$',left);assert ac
   geo.append({'page':page,'page_line':lineno,'line':ln,'longitude':vals[0],'latitude':vals[1],'altitude':int(ac[1].replace(' ',''))})
assert len(geo)==1874
match=[];aux=[];checks=[];diff=[];long=[]
for k in sorted(old_by):
 a=old_by[k];pv=poverty[k];gs=[g for g in geo if abs(float(a['latitude'])-g['latitude'])<1e-10 and abs(float(a['longitude'])-g['longitude'])<1e-10];assert len(gs)==1;kgeo=gs[0];assert abs(float(a['altitude'])-kgeo['altitude'])<1e-10
 names=data['electricity'][k]['name'];left,dist=names.split(', distrito: ',1);dep_prov=left.split(', ',1);dept=dep_prov[0];prov=dep_prov[1]if len(dep_prov)==2 else dep_prov[0]
 r={'ubigeo':k,'department':dept,'province':prov,'district':dist,'dept_code':k[:2],'prov_code':k[:4],'altitude':float(a['altitude']),'latitude':float(a['latitude']),'longitude':float(a['longitude']),'poverty':float(pv['poverty'])}
 r.update({'poverty_ci95_lower_pct':float(pv['poverty_ci95_lower_pct']),'poverty_ci95_upper_pct':float(pv['poverty_ci95_upper_pct']),'poverty_group_suffix':pv['district_group_suffix']})
 for ft,(n,nums)in specs.items():
  c=data[ft][k];den=c['denominator'];num=sum(c['counts'][v]for v in nums);assert den>0
  r[ft]=num/den;r[ft+'_count']=num;r[ft+'_denominator']=den
  long.extend({'ubigeo':k,'variable':ft,'category':v,'count':cnt,'denominator':den,'reference_year':2017}for v,cnt in c['counts'].items())
  if ft in a:
   delta=r[ft]-float(a[ft]);diff.append({'ubigeo':k,'variable':ft,'old':float(a[ft]),'verified':r[ft],'difference':delta})
 r['gas']=r['gas_glp']+r['gas_natural'];assert 0<=r['gas']<=1+1e-12
 r.update({'census_reference_year':2017,'census_extraction_status':'independently_reexecuted_official_CPV2017DI_counts_and_denominators','poverty_reference_year':2018,'poverty_publication_date':'2020-02-20','poverty_definition':pv['poverty_definition'],'coordinates_definition':'location_of_district_legal_capital_not_geometric_centroid','coordinates_reference_edition':2019,'coordinates_publication_date':'2019-03-27','coordinates_status':'exact_one_to_one_match_to_official_2019_directory_DMS_and_altitude','official_2025_code_status':'exact_same_UBIGEO_in_2025_directory','boundary_equivalence_status':'code_continuity_confirmed_polygon_boundary_equivalence_not_established','auxiliary_source_status':'independently_reconstructed_official_historical_sources'})
 match.append({'ubigeo':k,**kgeo,'old_latitude':a['latitude'],'old_longitude':a['longitude'],'old_altitude':a['altitude'],'source_pdf_sha256':sha(SRC/'directorio_2019.pdf')});aux.append(r)
 # Preserve the archived IEEE values if source-equivalent within 1e-14. This
 # avoids a spurious new fit whose sole change is decimal/export roundoff.
 for ft in ('poverty','illiteracy','electricity','spanish','biomass'):
  if abs(r[ft]-float(a[ft]))<1e-14:r[ft]=float(a[ft])
writecsv(OUT/'aux_verified_2017_2018.csv',aux);writecsv(OUT/'census_category_counts_2017.csv',long);writecsv(VAL/'geografia_covariate_numeric_differences.csv',diff);writecsv(VAL/'geografia_2019_capital_exact_matches.csv',match)
# Never crosswalk a newly created district by a similar name or inherited ratio.
xw=[];new_provinces={k[:4] for k in directory if k not in old_by}
for k in sorted(directory):
 d=directory[k];exists=k in old_by
 xw.append({'ubigeo_2025':k,'district_2025':d['district'],'ubigeo_historical_support':k if exists else '', 'historical_aux_available':exists,'match_method':'exact_UBIGEO' if exists else 'no_historical_match','status':'same_code_boundary_equivalence_unverified' if exists else 'new_district_no_aux_no_outcome_disaggregation','province_has_post2017_creation':k[:4]in new_provinces,'creation_legal_text':d['creation_legal_text'],'source_edition':2025,'can_transfer_2017_prevalence_to_new_district':False})
xw.append({'ubigeo_2025':'','district_2025':'MAZAMARI-PANGOA combined historical area','ubigeo_historical_support':'120699','historical_aux_available':False,'match_method':'no_exact_UBIGEO','status':'historical_combined_domain_direct_only_no_disaggregation','province_has_post2017_creation':False,'creation_legal_text':'','source_edition':2025,'can_transfer_2017_prevalence_to_new_district':False})
writecsv(OUT/'crosswalk_2017_2025_codes.csv',xw)
checks=[{'name':'all_1874_historical_codes_in_official_2025_list','pass':True},{'name':'all_1874_capital_coordinates_and_altitudes_exact_official_2019','pass':True},{'name':'all_10_census_questions_have_1874_count_rows','pass':True},{'name':'six_model_covariates_reconstructed_with_explicit_counts_and_denominators','pass':True},{'name':'new_17_districts_without_ratio_transfer','pass':sum(not r['historical_aux_available']and r['ubigeo_2025']!=''for r in xw)==17}]
report={'status':'PASS_OFFICIAL_HISTORICAL_AUXILIARIES_AND_STRICT_CODE_CROSSWALK','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'n_historical_districts':1874,'n_current_official_districts':1891,'n_new_since_historical_support':17,'n_census_category_count_rows':len(long),'n_queries':len(specs),'all_model_numeric_values_retained_as_source_equivalent':False,'max_covariate_reconstruction_difference':max(abs(d['difference'])for d in diff),'n_model_inputs':6,'all_checks_pass':all(c['pass']for c in checks),'checks':checks,'outputs':{str(p.relative_to(ROOT)):sha(p)for p in (OUT/'aux_verified_2017_2018.csv',OUT/'census_category_counts_2017.csv',OUT/'crosswalk_2017_2025_codes.csv',VAL/'geografia_2019_capital_exact_matches.csv',VAL/'geografia_covariate_numeric_differences.csv')},'sources':{'capital_coordinates':{'edition':2019,'publication_date':'2019-03-27','url':'https://www.gob.pe/institucion/inei/informes-publicaciones/3945124-directorio-nacional-de-municipalidades-provinciales-distritales-y-de-centros-poblados-2019','download_url':'https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Lib1653/Libro.pdf','sha256':sha(SRC/'directorio_2019.pdf')},'poverty':{'reference_year':2018,'publication_date':'2020-02-20','url':'https://www.gob.pe/institucion/inei/informes-publicaciones/3204872-mapa-de-pobreza-provincial-y-distrital-2018','sha256':sha(SRC/'poverty2018_anexo.xlsx'),'point_proxy':'arithmetic_midpoint_of_published_95pct_CI_not_official_published_point_estimate','grouped_districts':'preserved official Anexo2 district-group suffix; uncertainty not propagated through predictor'},'census':{'reference_year':2017,'official_service':'https://censos2017.inei.gob.pe/redatam/','base':'CPV2017DI','record_archives':[str((SRC/(specs[f][0]+'_query.json')).relative_to(ROOT))for f in specs]},'current_codes':{'edition':2025,'download_sha256':sha(SRC/'directorio_2025_excel.zip'),'historical_codes_only_used_for_temporal_model_inputs':True}},'limitations':['All census percentages are historical 2017, not current 2024/2025 values.','Coordinate information denotes capital locations, not district centroids or polygon adjacency.','Same UBIGEO does not establish unchanged polygon boundaries.','17 later district creations are retained without imputation or disaggregation.','120699 historical combined domain is retained as direct-only.','Poverty predictor is a published-CI midpoint proxy, and district grouped estimates and predictor uncertainty are retained as limitations.']}
(VAL/'geografia_aux_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:report[k]for k in ('status','n_historical_districts','n_current_official_districts','max_covariate_reconstruction_difference','all_checks_pass')},indent=2))
