#!/usr/bin/env python3
"""Independent raw HTML arithmetic audit for the frozen historical auxiliaries.
Uses regular-expression cell extraction rather than the producer's HTMLParser
or its parsed query JSON; does not fit models or touch the frozen covariate CSV.
"""
from pathlib import Path
import csv,json,re,html,hashlib,math,datetime,collections
R=Path(__file__).resolve().parents[1];P=R/'fuentes/geografia';O=R/'datos/geografia';V=R/'validacion';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a={x['ubigeo']:x for x in csv.DictReader((O/'aux_verified_2017_2018.csv').open())};checks=0;summaries=[];fail=[]
models={'electricity':('census_electricity',['Sí tiene alumbrado eléctrico']),'illiteracy':('census_illiteracy',['No sabe leer y escribir']),'spanish':('census_spanish',['Castellano']),'biomass':('census_biomass',['Sí usa bosta, estiércol'])}
# Actual archived HTML, independent of copied/parsed JSON cell values.
for feat,(name,positive)in models.items():
 p=P/(name+'_table_0.html');s=p.read_text(encoding='utf-8-sig');rr=[]
 for tr in re.findall(r'<tr\b[^>]*>(.*?)</tr>',s,re.S|re.I):
  cc=[re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>','',td))).strip()for td in re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>',tr,re.S|re.I)];rr.append(cc)
 h=next(t for t in rr if 'Código'in t);code=h.index('Código');cat=[i for i in range(h.index('Distrito')+1,len(h))if h[i]];pos=[h.index(n)for n in positive];seen=set()
 for row in rr:
  if len(row)>code and re.fullmatch('[0-9]{6}',row[code]):
   k=row[code];seen.add(k);vals=[int(row[j].replace(' ',''))for j in cat];den=sum(vals);num=sum(int(row[j].replace(' ',''))for j in pos);rec=a[k]
   for n,ok in [('raw_count',num==int(rec[feat+'_count'])),('raw_denominator',den==int(rec[feat+'_denominator'])),('model_ratio',abs(num/den-float(rec[feat]))<=1e-14)]:
    checks+=1
    if not ok:fail.append({'ubigeo':k,'feature':feat,'check':n})
 assert set(a)==seen
 summaries.append({'feature':feat,'raw_HTML_sha256':sha(p),'districts':len(seen),'checks':3*len(seen)})
# Exact official poverty interval calculation; rounded/export-represented
# historical proxy values remain byte-for-byte equal to the actual official CI.
pov={x['ubigeo']:x for x in csv.DictReader((O/'poverty_official_2018.csv').open())}
for k,r in a.items():
 lower=float(pov[k]['poverty_ci95_lower_pct']);upper=float(pov[k]['poverty_ci95_upper_pct']);checks+=1
 if abs(float(r['poverty'])-(lower+upper)/2)>3e-14:fail.append({'ubigeo':k,'check':'poverty_CI_midpoint'})
# The producer records each official PDF page/line; validate that line directly,
# recomputing its two DMS locations independently from scalar components.
page=(P/'directorio_2019.txt').read_text().split('\f');dm=re.compile(r'(\d+)[º°]\s*(\d+)[’ʹ\']\s*(\d+)[”"\u2033]')
for r in csv.DictReader((V/'geografia_2019_capital_exact_matches.csv').open()):
 k=r['ubigeo'];ln=page[int(r['page'])-1].splitlines()[int(r['page_line'])-1];v=dm.findall(ln);assert len(v)==2
 lon,lat=[-(int(d)*3600+int(m)*60+int(s))/3600 for d,m,s in v]
 for name,x in [('longitude',lon),('latitude',lat)]:
  checks+=1
  if abs(x-float(a[k][name]))>1e-10:fail.append({'ubigeo':k,'check':'official_DMS_'+name})
# Defensible field ownership and dates: no outcome/design/microdata fields.
assert not any(re.match('^(y_|var_|cov_|n_eff_|se_)',k) for k in next(iter(a.values())))
assert all(r['census_reference_year']=='2017'and r['poverty_reference_year']=='2018'and r['coordinates_reference_edition']=='2019'for r in a.values());checks+=len(a)
changes=[r for r in csv.DictReader((V/'geografia_covariate_numeric_differences.csv').open())if r['variable']in models and abs(float(r['difference']))>1e-14]
assert len(changes)==7;assert collections.Counter(r['variable']for r in changes)=={'biomass':5,'illiteracy':2};checks+=2
assert sha(O/'aux_verified_2017_2018.csv')=='cbec2fc142ec0ee06dd247230f0b2253e11af04d0e0e76e8374f30b456df7502';checks+=1
proof={'status':'PASS_INDEPENDENT_RAW_HTML_AND_OFFICIAL_PDF_AUXILIARY_AUDIT'if not fail else'FAIL','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'failures':fail,'raw_HTML_reconstruction':summaries,'n_aux_districts':len(a),'poverty_CSV_bound_export_roundoff_tolerance_pct':3e-14,'poverty_max_CSV_bound_export_roundoff_pct':max(abs(float(a[k]['poverty'])-(float(pov[k]['poverty_ci95_lower_pct'])+float(pov[k]['poverty_ci95_upper_pct']))/2)for k in a),'n_source_corrected_model_cells':len(changes),'corrected_cells':changes,'protected_aux_sha256_unchanged':sha(O/'aux_verified_2017_2018.csv'),'no_ENDES_outcomes_loaded':True,'source_script_sha256':sha(Path(__file__))}
(V/'geografia_independent_audit.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n');print(proof['status'],checks,'checks,',len(fail),'failures')
assert not fail
