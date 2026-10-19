"""Read-only standard-library verification of the author-review release."""
from pathlib import Path
import csv,hashlib,json,sys
HERE=Path(__file__).resolve()
PHASE=HERE.parents[1]
STUDY=PHASE.parent
ROOT=STUDY.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
    failures=[]
    for item in manifest['files']:
        p=ROOT/item['path']
        if not p.is_file()or sha(p)!=item['sha256']:failures.append(item['path'])
    protocol=json.loads((PHASE/'protocol_freeze.json').read_text())
    for name,value in protocol['source_sha256'].items():
        p=PHASE/name
        if not p.exists():p=STUDY/name
        if not p.exists()or sha(p)!=value:failures.append('frozen source:'+name)
    if sha(PHASE/'protocol_freeze.json')!=manifest['protocol_sha256']:failures.append('global protocol')
    excluded=json.loads((ROOT/'PRIVATE_INPUTS_REQUIRED.json').read_text())
    excluded_map={r['protocol_path']:r['sha256']for r in excluded}
    provided_inputs=0
    for name,digest in protocol['input_sha256'].items():
        p=STUDY/name.removeprefix('previous_study/')if name.startswith('previous_study/')else PHASE/name
        if p.is_file():
            provided_inputs+=1
            if sha(p)!=digest:failures.append('frozen included input:'+name)
        elif excluded_map.get(name)!=digest:failures.append('missing input without separate-acquisition record:'+name)
    if set(excluded_map)-set(protocol['input_sha256']):failures.append('unrecognized excluded input')
    for scope in ('temporal','territorial'):
        lock=json.loads((PHASE/f'validacion/externa_prediction_lock_{scope}.json').read_text())
        for item in lock['predictions']:
            for name,key in [('path','sha256'),('status_path','status_sha256')]:
                if sha(PHASE/item[name])!=item[key]:failures.append(scope+':'+item[name])
    with (PHASE/'resultados/externa_final/contrasts_Holm8.csv').open(newline='')as f:
        contrasts=list(csv.DictReader(f))
    if len(contrasts)!=8:failures.append('external family size')
    for item in manifest['files']:
        p=ROOT/item['path']
        if p.suffix=='.csv'and '/synthetic_'not in str(p):
            with p.open(newline='',encoding='utf-8-sig')as f:columns=next(csv.reader(f),[])
            forbidden={'child_id','household_id','HHID','HV001','HV021','psu_key','HC0','HC53','HC70'}
            if forbidden.intersection(columns):failures.append('respondent columns:'+item['path'])
    result={'status':'PASS_PUBLIC_AUTHOR_REVIEW_FILES'if not failures else'FAIL_PUBLIC_RELEASE',
            'files':len(manifest['files']),'frozen_sources':len(protocol['source_sha256']),
            'provided_frozen_inputs':provided_inputs,'separate_acquisition_inputs':len(excluded_map),
            'failures':failures,'private_microdata_expected_in_archive':False,
            'scope':'Integrity of provided files; raw-survey refitting requires separate lawful source acquisition.'}
    print(json.dumps(result,indent=2))
    if failures:sys.exit(1)
if __name__=='__main__':main()
