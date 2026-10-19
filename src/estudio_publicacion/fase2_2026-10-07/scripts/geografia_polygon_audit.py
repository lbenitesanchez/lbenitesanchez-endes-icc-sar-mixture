#!/usr/bin/env python3
"""Audit official polygon downloads without modifying frozen model geography.
IDE web page labels the archive updated2023, but actual GPKG metadata is2026
and lacks Alto Trujillo. SDMR web layer exposes a limited152district subset.
These files cannot establish current whole-country boundary equivalence.
"""
from pathlib import Path
import sqlite3,struct,zipfile,csv,json,hashlib,datetime,collections
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'fuentes/geografia';O=ROOT/'datos/geografia';V=ROOT/'validacion'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d={r['ubigeo']:r for r in csv.DictReader((O/'directorio_oficial_distritos_2025.csv').open())}
c=sqlite3.connect(P/'ide_distrital_2023/DISTRITO.gpkg');meta=c.execute('select table_name,last_change,srs_id from gpkg_contents').fetchone();rows=c.execute('select ubigeo,nombdist,fuente,nombdep,nombprov from DISTRITO').fetchall();pol={r[0]:r for r in rows};assert len(pol)==len(rows)==1890
z=zipfile.ZipFile(P/'sdmr_distrital_published2020.zip');b=z.read(next(n for n in z.namelist()if n.endswith('.dbf')));n=struct.unpack_from('<I',b,4)[0];hs=struct.unpack_from('<H',b,8)[0];rs=struct.unpack_from('<H',b,10)[0];columns=[]
for off in range(32,hs-1,32):
 bb=b[off:off+32]
 if bb[0]==13:break
 columns.append((bb[:11].split(b'\0')[0].decode(),bb[16]))
records=[]
for i in range(n):
 rec=b[hs+i*rs:hs+(i+1)*rs];assert len(rec)==rs
 if rec[:1]==b'*':continue
 vals={};off=1
 for col,size in columns:vals[col]=rec[off:off+size].decode('latin1').strip();off+=size
 records.append(vals)
fields=['ubigeo','official_2025_name','in_ide_polygon_file','in_sdmr_polygon_subset','national_2024_2025_boundary_equivalence_verified','status'];out=[];sdset={r['UBIGEO']for r in records}
for k in sorted(d):
 out.append({'ubigeo':k,'official_2025_name':d[k]['district'],'in_ide_polygon_file':k in pol,'in_sdmr_polygon_subset':k in sdset,'national_2024_2025_boundary_equivalence_verified':False,'status':'polygon_present_edition_metadata_insufficient_for_temporal_equivalence'if k in pol else 'missing_from_downloaded_IDE_polygon_file'})
f=O/'polygon_coverage_audit_2024_2025.csv'
with f.open('w',newline='')as ff:w=csv.DictWriter(ff,fieldnames=fields);w.writeheader();w.writerows(out)
proof={'status':'AUDIT_COMPLETE_POLYGONS_NOT_ADOPTED_AS_MODEL_INPUT','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_aux_sha256_unchanged':sha(O/'aux_verified_2017_2018.csv'),'model_coordinates_or_adjacency_modified':False,'IDE':{'official_landing':'https://ide.inei.gob.pe/','official_download':'https://ide.inei.gob.pe/files/Distrito.rar','web_label':'Distrital (Actualizado al2023)','server_Last_Modified':'2026-03-06T20:13:47Z','gpkg_last_change':meta[1],'srs_id':meta[2],'records':1890,'source_field_unique':dict(collections.Counter(r[2]for r in rows)),'missing_official_2025_codes':sorted(set(d)-set(pol)),'missing_names':[d[k]['district']for k in sorted(set(d)-set(pol))],'license':'Free interactive access stated by official portal; no explicit standalone file license identified.'},'SDMR':{'official_landing':'https://sdmr.inei.gob.pe/layers/geonode:distrito','official_download':'https://sdmr.inei.gob.pe/layers/download/geonode%3Adistrito/ESRI%20Shapefile','publication_metadata':'2020-07-23T11:00','reference_edition_metadata':'INAI; abstract and purpose por definir','records':len(records),'national_directory_records':1891,'national_coverage':False,'geometry_used_for_models':False},'files':{str(p.relative_to(ROOT)):sha(p)for p in [P/'ide_distrital_labeled_2023.rar',P/'ide_distrital_2023/DISTRITO.gpkg',P/'sdmr_distrital_published2020.zip',f]},'conclusion':'Official polygon services were acquired and audited. Current boundary equivalence to2017/2018support is not established, and the frozen model retains2019capital coordinates; no arbitrary polygon transfer or mapping of split district ratios.'}
assert proof['model_aux_sha256_unchanged']=='cbec2fc142ec0ee06dd247230f0b2253e11af04d0e0e76e8374f30b456df7502'
(V/'geografia_polygon_audit.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n');print('Polygon coverage audit complete: IDE1890missingAltoTrujillo; SDMR152subset; frozen model untouched.')
