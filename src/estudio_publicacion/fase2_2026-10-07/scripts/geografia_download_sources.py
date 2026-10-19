#!/usr/bin/env python3
"""Acquire official historical INEI sources and current-code audit sources.
No microdata, no outcomes. Preserves archives; checks every cached source SHA
against the acquisition manifest on subsequent reproduction.
"""
from pathlib import Path
import hashlib,json,subprocess,datetime,zipfile,re
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'fuentes/geografia'
SOURCES=[
 {'id':'capital_directory_2019','path':'directorio_2019.pdf','url':'https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Lib1653/Libro.pdf','landing':'https://www.gob.pe/institucion/inei/informes-publicaciones/3945124-directorio-nacional-de-municipalidades-provinciales-distritales-y-de-centros-poblados-2019','reference_edition':2019,'publication_date':'2019-03-27','purpose':'validate capital coordinates and altitude used as historical predictors'},
 {'id':'poverty_2018','path':'poverty2018_anexo.xlsx','url':'https://cdn.www.gob.pe/uploads/document/file/3340937/Anexo%20Estad%C3%ADstico.xlsx?v=1656708741','landing':'https://www.gob.pe/institucion/inei/informes-publicaciones/3204872-mapa-de-pobreza-provincial-y-distrital-2018','reference_edition':2018,'publication_date':'2020-02-20','purpose':'reconstruct published confidence-interval midpoint proxy and grouping metadata'},
 {'id':'district_codes_2024','path':'directorio_2024_excel.zip','url':'https://cdn.www.gob.pe/uploads/document/file/6407384/5612942-directorio-nacional-de-gobiernos-regionales-municipalidades-provinciales-distritales-y-de-centros-poblados-2024-cuadros-en-excel.zip?v=1732225977','landing':'https://www.gob.pe/institucion/inei/informes-publicaciones/5612942-directorio-nacional-de-gobiernos-regionales-municipalidades-provinciales-distritales-y-de-centros-poblados-2024','reference_edition':2024,'publication_date':'2024-07-04','purpose':'audit district-code continuity only; never prediction input'},
 {'id':'district_codes_2025','path':'directorio_2025_excel.zip','url':'https://cdn.www.gob.pe/uploads/document/file/7999764/6722617-directorio-nacional-de-gobiernos-regionales-municipalidades-provinciales-distritales-y-de-centros-poblados-2025-cuadros-en-excel.zip?v=1745955638','landing':'https://www.gob.pe/institucion/inei/informes-publicaciones/6722617-directorio-nacional-de-gobiernos-regionales-municipalidades-provinciales-distritales-y-de-centros-poblados-2025','reference_edition':2025,'publication_date':'2025-04-29','purpose':'audit district-code continuity only; never prediction input'}]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
P.mkdir(parents=True,exist_ok=True);manifest=P/'geografia_source_manifest.json';prior=json.loads(manifest.read_text())if manifest.exists()else None
for rec in SOURCES:
 p=P/rec['path'];cached=p.exists()
 if not cached:subprocess.run(['curl','-s','-f','-L','--max-time','90',rec['url'],'-o',str(p)],check=True)
 rec.update({'sha256':sha(p),'size_bytes':p.stat().st_size,'acquisition_utc':datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat(),'cached_this_run':cached,'license_status':'Publicly accessible official statistics; no explicit standalone data license identified; attribution to INEI; original publications and contact-containing workbooks retained as source evidence, not bundled as new analysis outputs.'})
 if prior:
  old=next(s for s in prior['sources']if s['id']==rec['id']);assert old['sha256']==rec['sha256'],('source changed',rec['id'])
 if p.suffix=='.zip':
  with zipfile.ZipFile(p)as z:
   bad=z.testzip();assert bad is None
   for n in z.namelist():
    # Only the municipality geography section, safely excluding path traversal.
    if '1.Directorio-MPyMD-' in n and not n.endswith('/'):
     dest=P/(p.stem)/n;dest=dest.resolve();assert (P/p.stem).resolve()in dest.parents
     if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(n))
# PDF text is a traceable extraction artifact; it preserves page breaks.
txt=P/'directorio_2019.txt'
if not txt.exists():subprocess.run(['pdftotext','-layout',str(P/'directorio_2019.pdf'),str(txt)],check=True)
rec={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':SOURCES,'census_source':{'reference_year':2017,'official_base':'CPV2017DI','landing':'https://censos2017.inei.gob.pe/redatam/','definitive_results_service_publication_evidence':'https://censo2017.inei.gob.pe/sistema-de-consulta-de-la-base-de-datos-de-los-censos-nacionales-se-publican-en-la-web-de-celadecepal/','service_publication_date_evidence':'2018-09-14','query_archive_glob':'census*_query.json','extraction_program':'scripts/geografia_redatam.py','counts_denominators_archive_included':True,'individual_microdata_downloaded':False},'PDF_extraction':{'path':str(txt.relative_to(ROOT)),'sha256':sha(txt),'method':'native Poppler pdftotext -layout; page breaks retained'},'temporal_scope':'Census2017 and poverty2018 published2020, capital-coordinate edition2019 are historical inputs available before2024. Directories2024/2025 validate codes and do not enter the forecast X; not a real-time2019 forecast because poverty publication is2020.'}
manifest.write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n');print('Official source manifest complete:',len(SOURCES),'downloads plus archived REDATAM requests.')
