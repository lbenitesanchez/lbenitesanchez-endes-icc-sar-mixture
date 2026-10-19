#!/usr/bin/env python3
"""Official REDATAM 2017 district aggregates. Never reads ENDES outcomes.
Network acquisition uses native curl's CA store (Python's bundled SSL chain does
not include the INEI intermediate). Records requests and archived tables/program.
"""
from pathlib import Path
import argparse, json, subprocess, re, html, hashlib, datetime, urllib.parse
from html.parser import HTMLParser
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'fuentes/geografia'
BASE='https://censos2017.inei.gob.pe/bininei/RpWebStats.exe/'
class Rows(HTMLParser):
 def __init__(self): super().__init__(convert_charrefs=True);self.rows=[];self.row=None;self.cell=None
 def handle_starttag(self,t,a):
  if t=='tr': self.row=[]
  if t in ('td','th') and self.row is not None:self.cell=[]
 def handle_data(self,d):
  if self.cell is not None:self.cell.append(d)
 def handle_endtag(self,t):
  if t in ('td','th') and self.cell is not None:
   self.row.append(' '.join(''.join(self.cell).split()));self.cell=None
  if t=='tr' and self.row is not None:self.rows.append(self.row);self.row=None

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fetch(url,p,fields=None):
 if p.exists() and p.stat().st_size>100:return {'cached':True,'sha256':digest(p)}
 args=['curl','-s','-f','-L','--max-time','120','--retry','2','--retry-delay','1']
 if fields:
  for k,v in fields: args.extend(['--data-urlencode',k+'='+v])
 args.extend([url,'-o',str(p)])
 z=subprocess.run(args,capture_output=True,text=True)
 if z.returncode:raise RuntimeError((url,z.returncode,z.stderr[-1000:]))
 return {'cached':False,'sha256':digest(p)}

def query(name,item,variables,filter='',endpoint='AreaList',extras=()):
 # Explicit selectors correspond to archived official forms.
 fields=[('MAIN','WebServerMain.inl'),('BASE','CPV2017DI'),('LANG','esp'),('CODIGO','XXUSUARIOXX'),('ITEM',item),('MODE','RUN'),('OUTPUT','Distrito'),('SELECTION','ALL'),('FORMAT','HTML'),('FORMAT','SPC'),('TOTCOL','on'),('UNIVERSE',''),('FILTER',filter),('TEXT_FILTER',filter),('Submit','Ejecutar')]
 fields.extend(('VARIABLE',v) for v in variables);fields.extend(extras)
 url=BASE+endpoint+'?';out=SOURCE/(name+'_result.html');fetch(url,out,fields)
 s=out.read_text(encoding='utf-8-sig');links=re.findall(r'<iframe[^>]+src="([^"]+)"',s,re.I)
 if not links or 'Error' in s:raise RuntimeError('No result tables '+name)
 tables=[]
 for i,l in enumerate(links):
  l=urllib.parse.quote(html.unescape(l),safe=':/?=&%~');p=SOURCE/f'{name}_table_{i}.html';fetch(l,p)
  rp=Rows();rp.feed(p.read_text(encoding='utf-8-sig'));rows=rp.rows
  tables.append({'path':str(p.relative_to(ROOT)),'sha256':digest(p),'rows':rows,'fetch_url':l})
 record={'name':name,'base':'CPV2017DI','reference_year':2017,'endpoint':url,'request_fields':fields,'request_result':str(out.relative_to(ROOT)),'result_sha256':digest(out),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tables':tables,'license':'Official publicly accessible aggregate statistics; no explicit standalone data license identified; attribution required; no individual records.'}
 (SOURCE/(name+'_query.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(name,'tables',len(tables),'rows',[len(t['rows']) for t in tables],flush=True)
 return record

QUERIES=[
 ('census_electricity','AREAVIV',['Vivienda.C2P11'],''),
 ('census_sewer','AREAVIV',['Vivienda.C2P10'],''),
 ('census_floor','AREAVIV',['Vivienda.C2P5'],''),
 ('census_water','AREAVIV',['Vivienda.C2P6'],''),
 ('census_biomass','AREAHOG',['Hogar.C3P16'],''),
 ('census_refrigerator','AREAHOG',['Hogar.C3P24'],''),
 ('census_gas_glp','AREAHOG',['Hogar.C3P12'],''),
 ('census_gas_natural','AREAHOG',['Hogar.C3P13'],''),
 ('census_spanish','AREAPOB',['Poblacio.C5P11'],''),
 ('census_illiteracy','AREAPOB',['Poblacio.C5P12'],''),
 ('census_age_months','AREAPOB',['Poblacio.C5P41M'],'Poblacio.C5P41=0'),
 ] + [(f'census_sex_month_{m}', 'AREAPOB', ['Poblacio.SEXO'], f'Poblacio.C5P41M={m}') for m in range(6,12)] + [(f'census_sex_year_{y}', 'AREAPOB', ['Poblacio.SEXO'], f'Poblacio.C5P41={y}') for y in range(1,5)]
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--only');ns=parser.parse_args();SOURCE.mkdir(parents=True,exist_ok=True)
 for q in QUERIES:
  if ns.only and q[0]!=ns.only:continue
  query(*q)
