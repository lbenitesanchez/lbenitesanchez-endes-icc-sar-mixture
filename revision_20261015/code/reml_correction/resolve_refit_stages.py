#!/usr/bin/env python3
"""Select fitting-diagnostic stage-two jobs and seal an explicit native overlay.

No population outcome or accuracy file is read. Stage one and two remain
immutable. The resolved tree has real directories and individual symlinks only;
the later score adapter may add mixtures and independent world completions.
"""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def save_new(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError('Preserve existing stage evidence: '+str(p))
    p.write_text(json.dumps(value,indent=2)+'\n')
def cp(root,j):return root/j['output_relative_dir']/(j['method']+'_reml_correction_completion.json')


def verify(root,j):
    p=cp(root,j);q=read(p)
    assert q['job_id']==j['job_id'] and q['input_sha256']==j['input_sha256']
    assert q['endpoint_seeds']==j['endpoint_seeds'] and q['bootstrap_B']==j['bootstrap_B']
    assert q['corrected'] and q['converged'] and q['truth_read']is False
    for rel,h in q['outputs_sha256'].items():assert sha(root/rel)==h,rel
    status=read(root/j['output_relative_dir']/j['status_filename'])
    endpoints=status['status'if j['development_status_schema']else'fits']
    failed=0
    for end in endpoints:
        assert end['converged']
        for sf in end['fit']:
            assert sf['stationarity_certificate']['passed'] and sf['stable_stationarity_certificate']['passed']
            if j['bootstrap_B']:
                assert sf['bootstrap_B']==j['bootstrap_B']
                assert sf['bootstrap_success']>=.95*j['bootstrap_B']
                assert sf['bootstrap_success']+sf['bootstrap_failed']==j['bootstrap_B']
                failed+=sf['bootstrap_failed']
    return q,failed


def plan_stage2(stage1,stage2):
    inventory=read(stage1/'job_inventory.json');jobs=inventory['jobs']
    assert len(jobs)==2780
    failure_paths=collections.defaultdict(list)
    for p in (stage1/'failures').glob('*.json'):
        q=read(p);failure_paths[q['job_id']].append(str(p.relative_to(stage1)))
    selected=[];pending=[];reasons={};phase1_bindings={}
    for j in jobs:
        p=cp(stage1,j)
        if p.is_file():
            q,failed=verify(stage1,j)
            phase1_bindings[j['job_id']]=dict(completion_relative_path=str(p.relative_to(stage1)),completion_sha256=sha(p))
            if failed:
                selected.append(j['job_id']);reasons[j['job_id']]=dict(reason='strict_bootstrap_refit_not_recovered',discarded_draws=failed)
        elif j['job_id']in failure_paths:
            selected.append(j['job_id']);reasons[j['job_id']]=dict(reason='method_exception_or_failed_central_gate',failure_receipts={p:sha(stage1/p)for p in failure_paths[j['job_id']]})
        else:pending.append(j['job_id'])
    if pending:raise RuntimeError(f'{len(pending)} stage-one jobs have not been attempted; second-stage applicability cannot be finalized')
    stage2.mkdir(parents=True,exist_ok=True)
    inventory_target=stage2/'job_inventory.json'
    if inventory_target.exists():assert sha(inventory_target)==sha(stage1/'job_inventory.json')
    else:inventory_target.write_bytes((stage1/'job_inventory.json').read_bytes())
    selection=dict(status='FITTING_ONLY_SECOND_STAGE_JOB_SELECTION',selected_utc=utc(),
                   registered_inventory_sha256=sha(stage1/'job_inventory.json'),
                   registered_jobs=len(jobs),selected_job_count=len(selected),selected_job_ids=selected,
                   selected_reasons=reasons,truth_read=False,
                   rule='Only completed methods with strictly rejected bootstrap draws, or attempted methods without an accepted central fit; no accuracy-based choice.',
                   models_graphs_B_endpoint_seeds_or_covariates_changed=False,
                   phase1_closed_checkpoint_bindings=phase1_bindings,
                   phase1_sources_unchanged=True)
    save_new(stage2/'job_selection.json',selection)
    print(json.dumps({k:v for k,v in selection.items()if k not in('selected_job_ids','selected_reasons','phase1_closed_checkpoint_bindings')},indent=2))


def link_file(destination,source,boundary):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.resolve().is_relative_to(boundary.resolve()):
        raise RuntimeError('Resolved output directories must be real, never links to external parents')
    if destination.is_symlink():
        assert destination.resolve()==source.resolve()
    elif destination.exists():raise FileExistsError('Refuse to replace resolved artifact: '+str(destination))
    else:destination.symlink_to(source.resolve())
    assert sha(destination)==sha(source)


def resolve(stage1,stage2,target):
    inventory_path=stage1/'job_inventory.json';inventory=read(inventory_path);jobs=inventory['jobs']
    selection=read(stage2/'job_selection.json');selected=set(selection['selected_job_ids'])
    assert selection['registered_inventory_sha256']==sha(inventory_path)
    assert len(jobs)==2780 and selection['registered_jobs']==2780
    assert selection['truth_read']is False
    dispatch_path=stage2/'stage2_dispatch_manifest.json'
    dispatch=read(dispatch_path)
    assert dispatch['truth_read']is False
    assert dispatch['global_inventory_sha256']==sha(inventory_path)
    assert dispatch['job_selection_sha256']==sha(stage2/'job_selection.json')
    assert dispatch['selected_job_count']==len(selected)
    freeze_path=stage2/'stage2_source_freeze.json'
    freeze=read(freeze_path)
    assert freeze['truth_read']is False and freeze['official_stage2_fit_started']is False
    comparison_path=stage2/'stage2_preservation_comparison.json'
    comparison=read(comparison_path)
    assert comparison['truth_read']is False and comparison['selection_sha256']==sha(stage2/'job_selection.json')
    lock_path=target/'prediction_lock_before_score.json'
    if lock_path.exists():raise FileExistsError('Resolved native lock is immutable: '+str(lock_path))
    target.mkdir(parents=True,exist_ok=True)
    invout=target/'job_inventory.json'
    if invout.exists():assert sha(invout)==sha(inventory_path)
    else:invout.write_bytes(inventory_path.read_bytes())
    # Preserve operational-filter evidence; each selected scientific job remains
    # byte-for-byte the registered global inventory entry.
    dispatch_inventories={}
    for p in stage1.glob('dispatch_*_jobs.json'):
        destination=target/p.name
        if destination.exists():assert sha(destination)==sha(p)
        else:destination.write_bytes(p.read_bytes())
        dispatch_inventories[p.name]=sha(p)
    items=[];outputs={};counts=collections.Counter();versions=collections.defaultdict(set)
    bootstrap_failed=0
    for j in jobs:
        label='stage2'if j['job_id']in selected else'stage1';root=stage2 if label=='stage2'else stage1
        q,failed=verify(root,j);bootstrap_failed+=failed
        counts[label]+=1
        for rel,h in q['outputs_sha256'].items():
            link_file(target/rel,root/rel,target);outputs[rel]=h
        cpath=cp(root,j);crel=str(cpath.relative_to(root))
        link_file(target/crel,cpath,target);outputs[crel]=sha(cpath)
        for key in ('solver_sha256','source_sha256','protocol_sha256'):
            versions[key].add(q[key])
        items.append(dict(job_id=j['job_id'],stage=label,
                          source_completion_relative_path=crel,completion_relative_path=crel,
                          completion_sha256=sha(cpath),outputs_sha256=q['outputs_sha256'],
                          solver_sha256=q['solver_sha256'],runner_sha256=q['source_sha256'],protocol_sha256=q['protocol_sha256']))
    manifest=dict(status='EXPLICIT_SAME_MODEL_STAGE_RESOLUTION',created_utc=utc(),truth_read=False,
                  registered_inventory_sha256=sha(inventory_path),second_stage_selection_sha256=sha(stage2/'job_selection.json'),
                  second_stage_selected_jobs=sorted(selected),planned_jobs=len(jobs),selected_job_counts=dict(counts),
                  allowed_stage_roots=[str(stage1),str(stage2)],source_versions={k:sorted(v)for k,v in versions.items()},
                  stage1_operational_dispatch_inventories_sha256=dispatch_inventories,
                  stage2_dispatch_manifest_sha256=sha(dispatch_path),
                  stage2_source_freeze_sha256=sha(freeze_path),
                  stage2_preservation_comparison_sha256=sha(comparison_path),
                  source_precedence='Stage two only for fitting-diagnostic selected jobs; all other native results retained from stage one.',
                  no_stage_native_file_overwritten=True,all_native_output_directories_are_real=True,
                  final_bootstrap_failed=bootstrap_failed,
                  bootstrap_gate='Every retained refit passes original and same-objective residual-contrast KKT tolerance1e-5; original minimum95percent retained.',
                  items=items)
    save_new(target/'resolution_manifest.json',manifest)
    lock=dict(status='ALL_CORRECTED_NATIVE_PREDICTIONS_LOCKED_BEFORE_RESCORING',complete=True,truth_read=False,
              locked_utc=utc(),planned=len(jobs),completed=len(jobs),remaining=0,
              all_closed=True,all_closed_converged=True,
              inventory_sha256=sha(inventory_path),resolution_manifest_sha256=sha(target/'resolution_manifest.json'),
              second_stage_selection_sha256=sha(stage2/'job_selection.json'),allowed_stage_roots=[str(stage1),str(stage2)],
              mixed_source_versions_explicit_resolution=True,source_versions=manifest['source_versions'],
              stage1_operational_dispatch_inventories_sha256=dispatch_inventories,
              stage2_dispatch_manifest_sha256=sha(dispatch_path),
              stage2_source_freeze_sha256=sha(freeze_path),
              stage2_preservation_comparison_sha256=sha(comparison_path),
              final_bootstrap_failed=bootstrap_failed,outputs_sha256=outputs,items=items)
    save_new(lock_path,lock)
    print(json.dumps({k:v for k,v in lock.items()if k not in('items','outputs_sha256')},indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=('plan-stage2','resolve'))
    ap.add_argument('--stage1-root',type=Path,required=True)
    ap.add_argument('--stage2-root',type=Path,required=True)
    ap.add_argument('--resolved-root',type=Path)
    a=ap.parse_args();s1=a.stage1_root.resolve();s2=a.stage2_root.resolve()
    if a.action=='plan-stage2':plan_stage2(s1,s2)
    else:
        if a.resolved_root is None:ap.error('--resolved-root is required for resolve')
        resolve(s1,s2,a.resolved_root.resolve())


if __name__=='__main__':main()
