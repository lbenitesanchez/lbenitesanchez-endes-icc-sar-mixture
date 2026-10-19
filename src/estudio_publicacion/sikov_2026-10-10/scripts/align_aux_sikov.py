"""Assemble common fourteen historical predictors, without reading outcomes."""
from pathlib import Path
import csv,json,re,hashlib,datetime
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT.parent
OLD=STUDY/'fase2_2026-10-07'
OUT=ROOT/'datos'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def counts(name,new=False):
    base=OUT/'census_sources' if new else OLD/'fuentes/geografia'
    path=base/(name+'_query.json')
    q=json.loads(path.read_text());assert len(q['tables'])==1
    table=q['tables'][0];source=(ROOT if new else OLD)/table['path']
    assert sha(source)==table['sha256']
    rows=table['rows'];h=next(r for r in rows if 'Código' in r)
    ci,di=h.index('Código'),h.index('Distrito')
    cc=[j for j in range(di+1,len(h))if h[j]]
    result={}
    for r in rows:
        if len(r)>ci and re.fullmatch(r'\d{6}',r[ci]):
            vals={h[j]:int(r[j].replace(' ',''))for j in cc}
            assert all(v>=0 for v in vals.values())
            assert sum(vals.values())>0
            result[r[ci]]=vals
    assert len(result)==1874
    return result,path

def main():
    auxpath=OLD/'datos/geografia/aux_verified_2017_2018.csv'
    aux=pd.read_csv(auxpath,dtype={'ubigeo':str,'dept_code':str,'prov_code':str})
    mapping={
        'PisoTierra':('census_floor',False,['Tierra']),
        'Cemento':('census_walls',True,['Ladrillo o bloque de cemento']),
        'Agua':('census_water',False,['Red pública dentro de la vivienda','Red pública fuera de la vivienda, pero dentro de la edificación']),
        'Elect':('census_electricity',False,['Sí tiene alumbrado eléctrico']),
        'Desague':('census_sewer',False,['Red pública de desagüe dentro de la vivienda','Red pública de desagüe fuera de la vivienda, pero dentro de la edificación']),
        'Internet':('census_internet',True,['Sí tiene conexión a internet']),
        'SIS':('census_sis',True,['Sí, afiliado al SIS']),
        'EsSalud':('census_essalud',True,['Sí, afiliado al EsSalud']),
        'NoSeguro':('census_noinsurance',True,['No se encuentra afiliado a ningún seguro']),
        'Refrig':('census_refrigerator',False,['Sí tiene refrigeradora o congeladora']),
        'Castellano':('census_spanish',False,['Castellano']),
        'Analfabet':('census_illiteracy',False,['No sabe leer y escribir']),
        'Rural':('census_rural',True,['Rural encuesta']),
    }
    # Validate exact category names before any proportions are written.
    output=aux[['ubigeo','department','province','district','dept_code','prov_code','latitude','longitude','poverty','poverty_ci95_lower_pct','poverty_ci95_upper_pct']].copy()
    output['Altitud']=aux.altitude/1000
    cache={};sources={str(auxpath.relative_to(STUDY)):sha(auxpath)};definition={}
    for name,(query,isnew,categories)in mapping.items():
        c,p=counts(query,isnew);cache[query]=c
        keys=set(next(iter(c.values())))
        if not set(categories).issubset(keys):
            raise ValueError((name,categories,keys))
        for code in output.ubigeo:
            assert set(c[code])==keys
        den=np.array([sum(c[code].values())for code in output.ubigeo],float)
        num=np.array([sum(c[code][category]for category in categories)for code in output.ubigeo],float)
        output[name]=num/den;output[name+'_count']=num.astype(int);output[name+'_denominator']=den.astype(int)
        sources[str(p.relative_to(STUDY))]=sha(p)
        definition[name]={'query':query,'numerator_categories':categories,'denominator':'sum of all returned nonblank response categories','unit':'proportion'}
    wall=cache['census_walls'];water=cache['census_water']
    output['Cemento_plus_piedra']=np.array([(wall[c]['Ladrillo o bloque de cemento']+wall[c]['Piedra o sillar con cal o cemento'])/sum(wall[c].values())for c in output.ubigeo])
    output['Agua_inside_only']=aux.water_piped
    # Rural definition matches the questionnaire/ENCAREA variable, also retain
    # census VAREA variant only if separately acquired, never conflate the two.
    output['poverty_stratum']=np.where(output.poverty<30,1,np.where(output.poverty<55,2,3))
    names=['Altitud','PisoTierra','Cemento','Agua','Elect','Desague','Internet','SIS','EsSalud','NoSeguro','Refrig','Castellano','Analfabet','Rural']
    assert np.isfinite(output[names+['latitude','longitude','poverty']].to_numpy(float)).all()
    assert ((output[names[1:]]>=0)&(output[names[1:]]<=1)).all().all()
    output.to_csv(OUT/'aux_sikov14_official.csv',index=False)
    proof={'status':'PASS_OFFICIAL_FOURTEEN_COMMON_PREDICTORS','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'source_sha256':sources,'output_sha256':sha(OUT/'aux_sikov14_official.csv'),'districts':len(output),
           'feature_names':['intercept']+names,'definitions':definition,'Altitud_definition':'official legal capital altitude in km',
           'article_predictor_description_ambiguities':['Cemento refers to cement walls but no category code is published; primary category brick/cement block; stone/cement alternative retained',
              'Agua public network primary inside plus outside building; inside-only alternative retained',
              'Internet is household questionnaire denominator, article says dwellings; original denominator not published',
              'Rural is survey-concept ENCAREA, article does not specify ENCAREA vs census-concept VAREA',
              'Analfabet is population age3plus responding to question, not adult15plus rate'],
           'poverty_definition':'arithmetic midpoint of official poverty2018 95percent CI, source available2020; exact authors point predictor unavailable',
           'poverty_stratum_counts':{str(k):int(v)for k,v in output.poverty_stratum.value_counts().sort_index().items()},
           'original_article_stratum_counts':[618,671,585],
           'ENDES_response_files_read':False,'covariates_selected_using_outcomes':False,'same_information_for_all_competitors':True}
    (ROOT/'validacion/align_aux_sikov_manifest.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2))
    print(json.dumps({k:proof[k]for k in ['status','districts','poverty_stratum_counts','output_sha256']}),flush=True)

if __name__=='__main__':main()
