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

