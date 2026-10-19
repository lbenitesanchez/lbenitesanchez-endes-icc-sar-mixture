#!/usr/bin/env python3
"""Add provenance sidecars; preserve original generated data and manifests."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numerical_cohorts_portable as run

def execute():
    p=run.gate();rows=[]
    for context in run.CONTEXTS:
        for rep in range(50000,50064):
            root=run.PRIVATE/'sparse57/evaluacion/inputs'/context/f'rep_{rep:05d}';manifest=run.read(root/'manifest.json')
            record=dict(format='registered-sparse57-generation-provenance-v1',context=context,replication=rep,original_generation_manifest_sha256=run.sha(root/'manifest.json'),source_sha256_in_original_manifest_role='identity of unchanged source text before the registered substitutions; not an assertion that the original generator ran unchanged',original_generator_source_sha256=manifest['source_sha256'],registered_adapted_generator_text_sha256=p['adapted_generator_text_sha256'],registered_protocol_sha256=run.sha(run.PROTOCOL),population_stream=manifest['population_stream'],sampling_stream=manifest['sampling_stream'],observed_per_stratum=19,observed_total=57,unsampled_total=63,original_generation_manifest_modified=False,numerical_arrays_modified=False,truth_scored=False)
            run.save(root/'registered_generation_provenance.json',record)
            rows.append(dict(context=context,replication=rep,provenance_sha256=run.sha(root/'registered_generation_provenance.json'),manifest_sha256=record['original_generation_manifest_sha256']))
    run.save(run.PUBLIC/'generation_provenance_sidecar_audit.json',dict(status='PASS_256_PROVENANCE_SIDECARS',worlds=256,registered_protocol_sha256=run.sha(run.PROTOCOL),registered_adapted_generator_text_sha256=p['adapted_generator_text_sha256'],original_manifests_and_numeric_arrays_modified=False,truth_scored=False,sidecars=rows))
    print(json.dumps({'status':'PASS_256_PROVENANCE_SIDECARS','worlds':256,'numerical_arrays_modified':False}))
if __name__=='__main__':execute()
