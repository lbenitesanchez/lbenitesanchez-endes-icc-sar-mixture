"""Validate reconstructed aggregates against official tables and archive provenance.

Use bundled Python with pandas/openpyxl. No individual records are printed.
"""
from pathlib import Path
import hashlib,json,re,unicodedata,os
import numpy as np
import pandas as pd
import openpyxl

STUDY=Path(__file__).resolve().parents[1]
DATA=STUDY/'datos'; OUT=DATA/'processed'; RAW=DATA/'raw'; SOURCES=DATA/'fuentes'
SRC=Path(os.environ.get('ENDES_AUX_ROOT', 'PRIVATE_INPUTS'))
def norm(s):
    return re.sub(r'[^A-Z0-9]', '', unicodedata.normalize('NFD',str(s)).encode('ascii','ignore').decode().upper())
aux=pd.read_csv(OUT/'aux_all_districts.csv',dtype={'ubigeo':str})
# Restore leading zeroes after any external CSV consumers; codes always derive from UBIGEO.
aux['dept_code']=aux.ubigeo.str[:2];aux['prov_code']=aux.ubigeo.str[:4]
aux['geography_source_status']='exact_match_to_archived_public_source_csv'
aux['census_source_status']='author_declared_redatam_queries_not_independently_reexecuted'
aux.to_csv(OUT/'aux_all_districts.csv',index=False)
dept_names={norm(r.department):r.dept_code for r in aux[['department','dept_code']].drop_duplicates().itertuples()}
dept_names['PROVCONSTDELCALLAO']='07'
for y in [2019,2025]:
    d=pd.read_csv(OUT/f'areas_{y}_6_59.csv',dtype={'ubigeo':str,'dept_code':str,'prov_code':str,'domain':str})
    d['geography_source_status']=d.ubigeo.map(aux.set_index('ubigeo').geography_source_status)
    d['census_source_status']=d.ubigeo.map(aux.set_index('ubigeo').census_source_status)
    d.to_csv(OUT/f'areas_{y}_6_59.csv',index=False)

comparison=[]
w=openpyxl.load_workbook(RAW/'ENDES2025/Cuadros_Estadisticos/Cap. 10/Cap. 10_2025.XLSX',read_only=True,data_only=True)
for year in [2019,2025]:
    agg=pd.read_csv(OUT/f'aggregates_{year}.csv',dtype={'domain':str})
    for sheet,universe,outcome,pc,dc,nc,sec in [('10.14','6_59','anemia_old',33,121,122,34),('10.18','0_59','stunting',35,51,52,36)]:
        if year==2019:
            ff=pd.read_csv(SOURCES/f'official2019_sheet_{sheet}.csv',header=None)
            rows=list(ff.itertuples(index=False,name=None))
            pc,dc,nc,sec=(2,8,9,None) if outcome=='anemia_old' else (3,11,12,None)
        else:rows=list(w[sheet].values)
        for row in rows:
            label=str(row[0]);key=norm(label)
            code='PE' if key in ['TOTAL',f'TOTAL{year}'] else dept_names.get(key)
            if code is None:continue
            pred=agg[(agg.domain==code)&(agg.universe==universe)&(agg.level==('national' if code=='PE' else 'department'))]
            if len(pred)!=1:continue
            try: pv=float(row[pc]);nw=int(float(row[nc]));dw=float(row[dc])
            except (ValueError,TypeError):continue
            pr=pred.iloc[0];val=pr['y_'+outcome]*100
            entry=dict(year=year,official_table=sheet,geography=label,domain=code,universe=universe,outcome=outcome,
                official_prevalence_pct=pv,reconstructed_pct=val,difference_pp=val-pv,
                official_n=nw,reconstructed_n=int(pr['n_'+outcome]),count_difference=int(pr['n_'+outcome])-nw,
                official_weighted_den=dw,reconstructed_weighted_den=pr['weighted_den_'+outcome])
            if sec is not None:
                entry['official_se_pct']=row[sec]
                entry['reconstructed_se_pct']=np.sqrt(pr['var_'+outcome])*100
            comparison.append(entry)
bench=pd.DataFrame(comparison)
bench.to_csv(OUT/'official_benchmark_validation.csv',index=False)
assert bench.difference_pp.abs().max()<1e-9
assert bench.count_difference.abs().max()==0

geog=pd.read_csv(SOURCES/'ubigeo_distrito_github.csv',dtype={'inei':str})
m=aux.merge(geog,left_on='ubigeo',right_on='inei',suffixes=('_local','_source'),validate='one_to_one')
geogcheck={'n_local':len(aux),'n_source':len(geog),'n_matched':len(m),
    'max_abs_latitude_difference':float((m.latitude_local-m.latitude_source).abs().max()),
    'max_abs_longitude_difference':float((m.longitude_local-m.longitude_source).abs().max()),
    'max_abs_altitude_difference':float((m.altitude_local-m.altitude_source).abs().max()),
    'max_abs_poverty_difference':float((m.poverty-m.pct_pobreza_total).abs().max()),
    'poverty_reference_year':2018,
    'census_proportion_reference_year':'declared_census_2017_context_unverified_query_export',
    'census_status':'Author-supplied REDATAM compilation. Queries/denominator exports not independently reproduced.'}
(OUT/'auxiliary_source_validation.json').write_text(json.dumps(geogcheck,indent=2))
manifest=[]
urlmap={
 'ENDES2025_complementaria.zip':'https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Lib2089/informacion_complementaria.zip',
 'ENDES2025_libro.pdf':'https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Lib2089/libro.pdf',
 'ENDES2025_PPR.pdf':'https://proyectos.inei.gob.pe/endes/2025/ppr/Informe_Indicadores_de_Resultados_de_los_Programas_Presupuestales_ENDES_2025.pdf',
 'ENDES2019_modulo64.zip':'https://proyectos.inei.gob.pe/iinei/srienaho/descarga/SPSS/691-Modulo64.zip',
 'ENDES2019_modulo74.zip':'https://proyectos.inei.gob.pe/iinei/srienaho/descarga/SPSS/691-Modulo74.zip',
 'ENDES2019_Cap010.xls':'https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Endes2019/cuadros/Cap_010.xls',
 'ubigeo_distrito_github.csv':'https://raw.githubusercontent.com/jmcastagnetto/ubigeo-peru-aumentado/main/ubigeo_distrito.csv',
 'ubigeo_source_README.md':'https://raw.githubusercontent.com/jmcastagnetto/ubigeo-peru-aumentado/main/README.md',
}
for path in [*(RAW/n for n in ['ENDES2025_complementaria.zip','ENDES2019_modulo64.zip','ENDES2019_modulo74.zip']),
             *(SOURCES/n for n in ['ENDES2025_libro.pdf','ENDES2025_PPR.pdf','ENDES2019_Cap010.xls','ubigeo_distrito_github.csv','ubigeo_source_README.md']),
             RAW/'ENDES2019/Modulo64/RECH0.SAV',RAW/'ENDES2019/Modulo64/RECH1.sav',RAW/'ENDES2019/Modulo74/RECH6.SAV',
             *(RAW/'ENDES2025/Base_Datos'/n for n in ['RECH0_2025.SAV','RECH1_2025.SAV','RECH6_2025.SAV']),
             SRC/'consolidado_nuevo_completo_con_directos.xlsx',SRC/'consolidado_censo_ubigeo_actualizado.xlsx']:
    manifest.append({'path':str(path.relative_to(STUDY)) if path.is_relative_to(STUDY) else str(path),
     'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
     'source_url':urlmap.get(path.name),'acquired_date':'2026-10-05','individual_microdata':path.suffix.lower()=='.sav',
     'author_provided':path.is_relative_to(SRC)})
identical2019={name:hashlib.sha256((SRC/name).read_bytes()).hexdigest()==hashlib.sha256(path.read_bytes()).hexdigest()
    for name,path in [('RECH0.SAV',RAW/'ENDES2019/Modulo64/RECH0.SAV'),('RECH6.SAV',RAW/'ENDES2019/Modulo74/RECH6.SAV')]}
(OUT/'source_manifest.json').write_text(json.dumps({'sources':manifest,'2019_author_files_exact_official_hash_match':identical2019},indent=2,ensure_ascii=False))
checks=[]
for year in [2019,2025]:
    d=pd.read_csv(OUT/f'areas_{year}_6_59.csv')
    for rule in ['old','who']:
        a=d.var_anemia_old if rule=='old' else d.var_anemia_who2024
        b=d.var_stunting;c=d.cov_old_stunting if rule=='old' else d.cov_who_stunting
        ev=(a+b-np.sqrt((a-b)**2+4*c**2))/2
        assert ev.min()> -1e-12
        checks.append({'year':year,'rule':rule,'areas':len(d),'min_D_eigenvalue':float(ev.min()),
          'zero_variance_a':int((a<1e-15).sum()),'zero_variance_s':int((b<1e-15).sum()),
          'one_psu_domain':int(d.n_psu.eq(1).sum()),'raw_singular_D':int((a*b-c**2<=1e-14).sum()),
          'missing_any_anemia':int(d['missing_anemia_'+('old' if rule=='old' else 'who2024')].sum()),
          'missing_any_stunting':int(d.missing_stunting.sum())})
(OUT/'data_quality_checks.json').write_text(json.dumps({'official_benchmark_rows':len(bench),'max_absolute_benchmark_difference_pp':float(bench.difference_pp.abs().max()),'max_benchmark_count_difference':int(bench.count_difference.abs().max()),'domain_checks':checks},indent=2))
print(json.dumps({'official_benchmarks':len(bench),'max_difference_pp':float(bench.difference_pp.abs().max()),'all_counts_exact':bool(bench.count_difference.eq(0).all()),'geography_source_csv_matched':len(m),'2019_exact_official_hashes':identical2019},indent=2))
