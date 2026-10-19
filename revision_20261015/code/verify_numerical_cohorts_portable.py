#!/usr/bin/env python3
"""Check pure path routing, frozen function identities and spawned-worker config."""
from pathlib import Path
import argparse,ast,concurrent.futures as cf,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numerical_cohorts_portable as portable

def routing():
    return {key:str(getattr(portable,key)) for key in ('REPO','SOURCE','ARTICLE','PRIVATE','OLD','PUBLIC','PROTOCOL','EXE','CMDSTAN')}
def fingerprint(node):return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
def verify(check_runtime=False):
    manifest=portable.read(portable.PUBLIC/'numerical_cohorts_portability_manifest.json')
    source=Path(portable.__file__);assert portable.sha(source)==manifest['portable_runtime_source_sha256']
    functions={x.name:x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef)}
    for check in manifest['function_comparisons']:
        node=functions[check['function']]
        if check['function']=='main':
            for n in ast.walk(node):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='ArgumentParser':n.keywords=[k for k in n.keywords if k.arg!='parents']
        assert fingerprint(node)==check['original_function_AST_sha256'],check['function']
    with cf.ProcessPoolExecutor(max_workers=1) as pool:assert pool.submit(routing).result(timeout=30)==routing()
    assert '/Users/' not in source.read_text()
    assert len(portable.jobs_for('development_nuts'))==32 and len(portable.jobs_for('sparse57'))==256
    if check_runtime:portable.gate()
    result=dict(status='PASS_PORTABLE_ROUTING_AND_SCIENTIFIC_FUNCTION_IDENTITIES',verified_utc=portable.utc(),compared_functions=len(manifest['function_comparisons']),spawn_config_matches_parent=True,private_state_outside_repository=not portable.PRIVATE.is_relative_to(portable.REPO),registered_model_and_runtime_gate_checked=check_runtime,models_fitted=False,truth_read=False,portable_source_sha256=portable.sha(source),verification_source_sha256=portable.sha(__file__))
    portable.save(portable.PUBLIC/'portable_runtime_validation.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-runtime',action='store_true');a=parser.parse_args();verify(a.check_runtime)
