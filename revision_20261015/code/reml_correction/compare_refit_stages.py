#!/usr/bin/env python3
"""Truth-blind comparison of the immutable first and second fitting stages.

Production did not persist raw U/E matrices or their checksums. This audit
explicitly reconstructs standard-normal streams from recorded seeds, B and
stratum dimensions; it never describes those digests as recorded-run captures.
"""
from __future__ import annotations
import argparse
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cp(root,j):return root/j['output_relative_dir']/(j['method']+'_reml_correction_completion.json')
def equal(a,b):return bool(np.array_equal(np.asarray(a),np.asarray(b)))


def rng_calls(path):
    tree=ast.parse(path.read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='corrected_bootstrap')
    return [ast.dump(node,include_attributes=False)for node in ast.walk(function)
            if isinstance(node,ast.Call) and
            (ast.unparse(node.func)=='np.random.default_rng' or ast.unparse(node.func)=='rng.normal')]


def compare(repo,stage1,stage2,report):
    selection=read(stage2/'job_selection.json');inventory=read(stage1/'job_inventory.json')
    assert selection['truth_read']is False and selection['registered_inventory_sha256']==sha(stage1/'job_inventory.json')
    jobs={j['job_id']:j for j in inventory['jobs']}
    src=repo/'revision_20261015/code/reml_correction'
    base=src/'stable_reml_solver.py';second=src/'stable_reml_solver_stage2_v2.py'
    assert rng_calls(base)==rng_calls(second)
    counts=dict(jobs_compared=0,central_strata_compared=0,central_A_rho_changed=0,
                central_beta_changed=0,central_mean_arrays_changed=0,
                previously_retained_draws=0,previously_retained_draws_changed_A_rho=0,
                previously_retained_draws_now_rejected=0,newly_retained_draws=0,
                reconstructed_normal_matrices_compared=0,reconstructed_normal_matrix_mismatches=0)
    records=[]
    for job_id in selection['selected_job_ids']:
        j=jobs[job_id];c1=read(cp(stage1,j));c2=read(cp(stage2,j))
        assert c1['input_sha256']==c2['input_sha256']==j['input_sha256']
        assert c1['endpoint_seeds']==c2['endpoint_seeds']==j['endpoint_seeds']
        assert c1['bootstrap_B']==c2['bootstrap_B']==j['bootstrap_B']
        s1=read(stage1/j['output_relative_dir']/j['status_filename'])
        s2=read(stage2/j['output_relative_dir']/j['status_filename'])
        e1=s1['status'if j['development_status_schema']else'fits']
        e2=s2['status'if j['development_status_schema']else'fits']
        with (np.load(stage1/j['output_relative_dir']/j['prediction_filename'],allow_pickle=False)as p1,
              np.load(stage2/j['output_relative_dir']/j['prediction_filename'],allow_pickle=False)as p2):
            same_means=equal(p1['mean'],p2['mean'])
        counts['central_mean_arrays_changed']+=int(not same_means)
        item=dict(job_id=job_id,endpoint_seeds=j['endpoint_seeds'],bootstrap_B=j['bootstrap_B'],
                  input_sha256=j['input_sha256'],stage1_completion_sha256=sha(cp(stage1,j)),
                  stage2_completion_sha256=sha(cp(stage2,j)),central_mean_array_exactly_equal=same_means,
                  strata=[])
        for endpoint,(v1,v2)in enumerate(zip(e1,e2)):
            r1=np.random.default_rng(c1['endpoint_seeds'][endpoint]);r2=np.random.default_rng(c2['endpoint_seeds'][endpoint])
            for f1,f2 in zip(v1['fit'],v2['fit']):
                assert f1['stratum']==f2['stratum']
                same_params=f1['variance']==f2['variance'] and f1['rho']==f2['rho']
                same_beta=equal(f1['beta'],f2['beta'])
                counts['central_strata_compared']+=1;counts['central_A_rho_changed']+=int(not same_params)
                counts['central_beta_changed']+=int(not same_beta)
                hashes={}
                for name,dimension in (('U','n_areas'),('E','n_sampled')):
                    assert f1[dimension]==f2[dimension]
                    a=r1.normal(size=(j['bootstrap_B'],f1[dimension]));b=r2.normal(size=(j['bootstrap_B'],f2[dimension]))
                    h1=hashlib.sha256(a.tobytes(order='C')).hexdigest();h2=hashlib.sha256(b.tobytes(order='C')).hexdigest()
                    counts['reconstructed_normal_matrices_compared']+=1
                    counts['reconstructed_normal_matrix_mismatches']+=int(h1!=h2)
                    hashes[name]=dict(stage1_reconstructed_sha256=h1,stage2_reconstructed_sha256=h2,shape=list(a.shape))
                old=f1['bootstrap_draw_diagnostics'];new=f2['bootstrap_draw_diagnostics']
                assert len(old)==len(new)==j['bootstrap_B']
                changed=[];dropped=[];added=[];retained=0
                for d1,d2 in zip(old,new):
                    index=d1['draw_index_zero_based'];assert index==d2['draw_index_zero_based']
                    if d1['retained']:
                        retained+=1
                        if d1['variance']!=d2['variance'] or d1['rho']!=d2['rho']:changed.append(index)
                        if not d2['retained']:dropped.append(index)
                    elif d2['retained']:added.append(index)
                counts['previously_retained_draws']+=retained
                counts['previously_retained_draws_changed_A_rho']+=len(changed)
                counts['previously_retained_draws_now_rejected']+=len(dropped)
                counts['newly_retained_draws']+=len(added)
                item['strata'].append(dict(endpoint=('anemia','stunting')[endpoint],stratum=f1['stratum'],
                    central_A_rho_exactly_equal=same_params,central_beta_exactly_equal=same_beta,
                    previously_retained=retained,previously_retained_changed_A_rho_indices=changed,
                    previously_retained_now_rejected_indices=dropped,newly_retained_indices=added,
                    reconstructed_standard_normal_stream_checksums=hashes))
        counts['jobs_compared']+=1;records.append(item)
    assert counts['jobs_compared']==selection['selected_job_count']
    changes=sum(counts[k]for k in('central_A_rho_changed','central_beta_changed','central_mean_arrays_changed',
        'previously_retained_draws_changed_A_rho','previously_retained_draws_now_rejected','reconstructed_normal_matrix_mismatches'))
    value=dict(status='PASS_ONLY_PREVIOUSLY_REJECTED_DRAWS_ADDED'if changes==0 else'CHANGES_REQUIRE_NUMERICAL_REVIEW',
        truth_read=False,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        selection_sha256=sha(stage2/'job_selection.json'),base_solver_sha256=sha(base),stage2_solver_sha256=sha(second),
        RNG_call_expressions_and_order_identical_in_frozen_sources=True,
        checksum_evidence='Reconstructed standard-normal U/E streams from recorded seed, B and stratum dimensions; production did not persist raw U/E or captured-run checksums.',
        seed_B_input_identity_checked_for_every_selected_job=True,counts=counts,items=records)
    report.parent.mkdir(parents=True,exist_ok=True)
    with report.open('x')as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in value.items()if k!='items'},indent=2))
    return value


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','stage1-root','stage2-root','report'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();value=compare(a.repo.resolve(),a.stage1_root.resolve(),a.stage2_root.resolve(),a.report.resolve())
    if value['status']!='PASS_ONLY_PREVIOUSLY_REJECTED_DRAWS_ADDED':raise SystemExit(1)


if __name__=='__main__':main()
