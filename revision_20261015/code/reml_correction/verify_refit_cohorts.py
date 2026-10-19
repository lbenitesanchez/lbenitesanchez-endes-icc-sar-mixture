#!/usr/bin/env python3
"""Independent receipt/count/algebra checks of corrected REML jobs, no truth.

Exports only scalar diagnostics and aggregate bootstrap counts. Does not open
evaluation outcomes, chains, population responses, or microdata. A partial run
is reported as partial and cannot pass the final verification gate.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def certificate(cert):
    assert cert['tolerance']==1e-5 and cert['rho_bound']==.999
    g=cert['projected_gradient_max']
    assert bool(cert['passed'])==(g is not None and np.isfinite(g) and g<1e-5)
    return bool(cert['passed'])


def verify(repo,private,output,public):
    inventory=read(output/'job_inventory.json'); jobs=inventory['jobs']
    for rel,h in inventory['source_models_sha256'].items():assert sha(repo/rel)==h,rel
    assert len(jobs)==2780 and len({j['job_id']for j in jobs})==2780
    permitted={sha(output/'job_inventory.json'): {j['job_id']:j for j in jobs}}
    resolution=None;resolved_items={}
    if (output/'resolution_manifest.json').exists():
        resolution=read(output/'resolution_manifest.json')
        assert resolution['truth_read']is False and resolution['planned_jobs']==2780
        assert resolution['registered_inventory_sha256']==sha(output/'job_inventory.json')
        selected=read(Path(resolution['allowed_stage_roots'][1])/'job_selection.json')
        assert sha(Path(resolution['allowed_stage_roots'][1])/'job_selection.json')==resolution['second_stage_selection_sha256']
        assert set(selected['selected_job_ids'])==set(resolution['second_stage_selected_jobs'])
        stage2_root=Path(resolution['allowed_stage_roots'][1])
        assert sha(stage2_root/'stage2_dispatch_manifest.json')==resolution['stage2_dispatch_manifest_sha256']
        assert sha(stage2_root/'stage2_source_freeze.json')==resolution['stage2_source_freeze_sha256']
        assert sha(stage2_root/'stage2_preservation_comparison.json')==resolution['stage2_preservation_comparison_sha256']
        dispatch=read(stage2_root/'stage2_dispatch_manifest.json')
        assert dispatch['job_selection_sha256']==resolution['second_stage_selection_sha256']
        assert dispatch['global_inventory_sha256']==sha(output/'job_inventory.json')
        assert dispatch['selected_job_count']==len(selected['selected_job_ids']) and dispatch['truth_read']is False
        for rel,h in dispatch['sources_sha256'].items():assert sha(repo/rel)==h,rel
        for name,h in resolution['stage1_operational_dispatch_inventories_sha256'].items():assert sha(output/name)==h,name
        resolved_items={x['job_id']:x for x in resolution['items']}
    for p in output.glob('dispatch_*_jobs.json'):
        other=read(p);selection=other['operational_dispatch_selection']
        assert selection['registered_inventory_sha256']==sha(output/'job_inventory.json')
        assert selection['scientific_job_fields_changed'] is False
        for j in other['jobs']:assert j==next(x for x in jobs if x['job_id']==j['job_id'])
        permitted[sha(p)]={j['job_id']:j for j in other['jobs']}
    central=[];bootstraps=[];summaries=collections.defaultdict(collections.Counter)
    missing=[];records=[];output_catalogue={};failures=[]
    for j in jobs:
        folder=output/j['output_relative_dir']; cp=folder/(j['method']+'_reml_correction_completion.json')
        if not cp.is_file():missing.append(j['job_id']);continue
        try:
            q=read(cp);assert q['job_id']==j['job_id'] and q['input_sha256']==j['input_sha256']
            assert q['corrected'] and q['converged'] and not q['truth_read']
            assert q['endpoint_seeds']==j['endpoint_seeds'] and q['bootstrap_B']==j['bootstrap_B']
            assert permitted[q['job_inventory_sha256']][j['job_id']]==j
            if resolution:
                item=resolved_items[j['job_id']]
                assert item['completion_sha256']==sha(cp)
                assert item['stage']==('stage2'if j['job_id']in resolution['second_stage_selected_jobs']else'stage1')
                assert item['solver_sha256']==q['solver_sha256'] and item['runner_sha256']==q['source_sha256']
                assert item['protocol_sha256']==q['protocol_sha256']
                if item['stage']=='stage2':assert q['job_selection_sha256']==resolution['second_stage_selection_sha256']
            for rel,h in q['outputs_sha256'].items():
                assert sha(output/rel)==h,rel
                output_catalogue[rel]=h
            for rel,h in q['fitting_sources_sha256'].items():assert sha(repo/rel)==h,rel
            for key,hkey in [('input_path','input_sha256'),('original_prediction_path','original_prediction_sha256'),('original_status_path','original_status_sha256')]:
                assert sha(private/j[key])==j[hkey],j[key]
            p=dict(np.load(folder/j['prediction_filename'],allow_pickle=False))
            assert all(np.isfinite(p[k]).all()for k in ['mean','mse','lower','upper'])
            old=np.load(private/j['original_prediction_path'],allow_pickle=False)
            assert p['mean'].shape==old['mean'].shape and p['mean'].shape[1]==2
            assert np.allclose(p['lower'],p['mean']-1.959963984540054*np.sqrt(np.maximum(p['mse'],0)),rtol=0,atol=1e-12)
            assert np.allclose(p['upper'],p['mean']+1.959963984540054*np.sqrt(np.maximum(p['mse'],0)),rtol=0,atol=1e-12)
            assert np.array_equal(p['mse'],p['mse_naive'if j['bootstrap_B'] else'mse_delta'])
            st=read(folder/j['status_filename']); endpoints=st['status'if j['development_status_schema']else'fits']
            assert len(endpoints)==2
            s=summaries[(j['cohort'],j['method'])];s['jobs']+=1;s['elapsed_seconds']+=q['seconds']
            s['max_process_RSS_bytes']=max(s['max_process_RSS_bytes'],q['process_peak_RSS_bytes'])
            for endpoint,v in zip(('anemia','stunting'),endpoints):
                assert v['converged'] and len(v['fit'])==3
                for sf in v['fit']:
                    cert=sf['stationarity_certificate'];stable=sf['stable_stationarity_certificate']
                    assert certificate(cert) and certificate(stable)
                    metadata=sf['numerical_correction'];s['central_fits']+=1
                    s['central_rescued']+=int(metadata['repaired'])
                    oldcert=metadata['original_stationarity_certificate'];oldstable=metadata['original_stable_stationarity_certificate']
                    s['central_original_certificate_failed']+=int(not oldcert['passed'])
                    s['central_original_raw_success_certificate_failed']+=int(metadata['original_raw_optimizer_converged'] is True and not oldcert['passed'])
                    central.append(dict(cohort=j['cohort'],context=j['context'],replication=j['replication'],method=j['method'],endpoint=endpoint,stratum=sf['stratum'],
                        strict_converged=sf['converged'],original_raw_optimizer_converged=metadata['original_raw_optimizer_converged'],
                        original_certificate_passed=oldcert['passed'],original_projected_gradient=oldcert['projected_gradient_max'],
                        original_stable_certificate_passed=oldstable['passed'],
                        final_certificate_passed=cert['passed'],final_projected_gradient=cert['projected_gradient_max'],
                        final_stable_certificate_passed=stable['passed'],final_stable_projected_gradient=stable['projected_gradient_max'],
                        numerical_rescued=metadata['repaired'],original_variance=metadata['original_variance'],variance=sf['variance'],
                        original_rho=metadata['original_rho'],rho=sf['rho'],bootstrap_B=j['bootstrap_B']))
                    if not j['bootstrap_B']:continue
                    B=j['bootstrap_B'];draws=sf['bootstrap_draw_diagnostics'];assert len(draws)==B
                    assert [d['draw_index_zero_based']for d in draws]==list(range(B))
                    n=sum(bool(d['retained'])for d in draws)
                    assert n==sf['bootstrap_success'] and B-n==sf['bootstrap_failed'] and n>=.95*B
                    for d in draws:
                        passed=certificate(d['final_certificate']) and certificate(d['final_stable_certificate']) if d.get('final_certificate')and d.get('final_stable_certificate')else False
                        if d['retained']:assert passed and d['variance']>=0 and abs(d['rho'])<=.999
                    ns=sum(bool(d.get('repaired'))for d in draws)
                    failures_raw=sum(not d.get('original_certificate',{}).get('passed',False)for d in draws)
                    orbad=sum(d.get('raw_optimizer_converged')is True and not d.get('original_certificate',{}).get('passed',False)for d in draws)
                    assert sf['bootstrap_numerical_rescued']==ns
                    assert sf['bootstrap_strict_retained_certificates_passed']==n
                    s['bootstrap_requested']+=B;s['bootstrap_strict_retained']+=n;s['bootstrap_failed']+=B-n
                    s['bootstrap_rescued']+=ns;s['bootstrap_original_certificate_failed']+=failures_raw
                    s['bootstrap_original_raw_success_certificate_failed']+=orbad
                    bootstraps.append(dict(cohort=j['cohort'],context=j['context'],replication=j['replication'],method=j['method'],endpoint=endpoint,stratum=sf['stratum'],
                        bootstrap_B=B,retained=n,failed=B-n,numerical_rescued=ns,original_certificate_failed=failures_raw,original_raw_success_certificate_failed=orbad,
                        every_retained_fit_passed_both_certificates=True,original_min_success_fraction=.95))
            records.append(q)
        except BaseException as exc:failures.append(dict(job_id=j['job_id'],error=repr(exc)))
    report=dict(status='PASS_ALL_CORRECTED_METHODS_VERIFIED'if not missing and not failures else'INCOMPLETE_OR_FAILED_CORRECTED_METHOD_AUDIT',
                planned_jobs=len(jobs),verified_jobs=len(records),missing_jobs=len(missing),failed_jobs=len(failures),truth_read=False,
                original_source_models_changed=False,models_seeds_graphs_B_checked=True,
                central_fits_verified=len(central),bootstrap_strata_verified=len(bootstraps),
                bootstrap_requested=sum(r['bootstrap_B']for r in bootstraps),bootstrap_strict_retained=sum(r['retained']for r in bootstraps),
                bootstrap_failed=sum(r['failed']for r in bootstraps),
                source_versions={key:sorted({r[key]for r in records})for key in('source_sha256','solver_sha256','protocol_sha256')},
                inventory_sha256=sha(output/'job_inventory.json'),missing_job_ids=missing,failures=failures)
    if resolution:
        assert resolution['source_versions']==report['source_versions']
        report['explicit_stage_resolution_manifest_sha256']=sha(output/'resolution_manifest.json')
        report['source_precedence']=resolution['source_precedence']
        report['second_stage_selected_jobs']=len(resolution['second_stage_selected_jobs'])
    elif records and any(len(x)!=1 for x in report['source_versions'].values()):
        report['status']='FAIL_MIXED_FITTING_SOURCE_OR_PROTOCOL_VERSIONS'
    public.mkdir(parents=True,exist_ok=True)
    write_csv(public/'corrected_central_REML_stratum_fits.csv',central)
    write_csv(public/'corrected_bootstrap_stratum_summary.csv',bootstraps)
    rows=[dict(cohort=c,method=m,**dict(v))for(c,m),v in sorted(summaries.items())]
    allkeys=sorted({k for r in rows for k in r})
    rows=[{k:r.get(k,0)for k in allkeys}for r in rows]
    write_csv(public/'corrected_method_diagnostic_summary.csv',rows)
    report['outputs_sha256']={p.name:sha(p)for p in public.glob('corrected_*.csv')}
    (public/'corrected_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k not in('missing_job_ids','failures')},indent=2))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','private-run-root','output','public-output'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();report=verify(a.repo.resolve(),a.private_run_root.resolve(),a.output.resolve(),a.public_output.resolve())
    if report['status']!='PASS_ALL_CORRECTED_METHODS_VERIFIED':raise SystemExit(1)


if __name__=='__main__':main()
