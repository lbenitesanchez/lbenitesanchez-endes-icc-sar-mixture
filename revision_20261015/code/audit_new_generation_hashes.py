#!/usr/bin/env python3
"""Verify all planned input and truth file hashes without loading truth arrays."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numerical_cohorts_portable as run

def audit():
    run.gate();rows=[]
    for cohort in ('development_nuts','sparse57'):
        for _,context,rep in run.jobs_for(cohort):
            root=(run.OLD/'sikov/desarrollo/inputs' if cohort=='development_nuts' else run.PRIVATE/'sparse57/evaluacion/inputs')/context/f'rep_{rep:05d}';m=run.read(root/'manifest.json')
            for filename,key in (('model_inputs/inputs.npz','fit_inputs_sha256'),('model_inputs/sample.csv','sample_sha256'),('evaluation_only/truth.npz','truth_sha256')):
                assert run.sha(root/filename)==m[key],(cohort,context,rep,filename)
            rows.append(dict(cohort=cohort,context=context,replication=rep,fit_inputs_sha256=m['fit_inputs_sha256'],sample_sha256=m['sample_sha256'],truth_sha256=m['truth_sha256'],hashes_match_generation_manifest=True))
    result=dict(status='PASS_ALL_288_PLANNED_INPUT_AND_TRUTH_FILE_IDENTITIES',audited_utc=run.utc(),planned_worlds=288,truth_arrays_loaded=False,accuracy_metrics_computed=False,hash_bytes_only=True,private_values_exported=False,checks=rows,protocol_sha256=run.sha(run.PROTOCOL),code_sha256=run.sha(__file__))
    run.save(run.PUBLIC/'planned_input_and_truth_hash_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
if __name__=='__main__':audit()
