#!/usr/bin/env python3
"""Copy closed causal aggregate reports with explicit original SHA provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import re

DROP={'argv','command','commands','pid','process_id','stdout','stderr','log','logs'}
RECORD=re.compile(r'^(caseid|child_id|person_id|household_id|hhid|psu_key|hvidx|bidx)$',re.I)
PATH=re.compile(r'(?:/(?:Users|home|private|tmp|Volumes)/|[A-Za-z]:\\)',re.I)

def sanitize(value):
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            if key in DROP:continue
            if RECORD.fullmatch(key):raise ValueError('Unit-level field in aggregate evidence')
            public_key='LOCAL_REFERENCE_SHA256_'+hashlib.sha256(key.encode()).hexdigest()if PATH.search(key)else key
            result[public_key]=sanitize(item)
        return result
    if isinstance(value,list):return [sanitize(v)for v in value]
    if isinstance(value,str)and PATH.search(value):
        return 'LOCAL_REFERENCE_SHA256_'+hashlib.sha256(value.encode()).hexdigest()
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous public evidence versions')
    raw=a.input.read_bytes();original=json.loads(raw);public=sanitize(original)
    if not isinstance(public,dict):raise ValueError('Aggregate report object required')
    public['public_copy_provenance']={'original_closed_report_sha256':hashlib.sha256(raw).hexdigest(),
        'sanitizer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_statuses_unknowns_failures_and_numerical_values_preserved':True,
        'personal_machine_paths_and_operational_commands_removed':True,
        'original_report_or_prior_archive_changed':False}
    data=json.dumps(public,indent=2,allow_nan=False)
    if PATH.search(data):raise ValueError('Personal path in sanitized evidence')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(data+'\n')
    print(json.dumps({'original_sha256':hashlib.sha256(raw).hexdigest(),
        'public_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'original_report_modified':False}))

if __name__=='__main__':main()
