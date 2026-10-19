#!/usr/bin/env python3
"""Regenerate/check legacy inputs with the versioned public orientation rule.

Original frozen source bytes, seeds and sampling rules are retained. An
explicit orientation correction is applied to the generated DGP basis; inputs
must match all archived observed samples/directs before any model is fitted.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import datetime as dt
import hashlib
import json
import sys
import time
import legacy_worker

WORKER={}


def initialize(study,orientation):
    legacy_worker.setup(Path(study))
    import benchmark_inputs as inputs
    reject_archive_population_caches(inputs.SIM)
    from legacy_orientation_adapter import install
    reports=install(inputs,orientation)
    import benchmark_validate_inputs as validation
    WORKER.update(inputs=inputs,validation=validation,reports=reports)


def reject_archive_population_caches(simulation_root):
    # The original exporter conditionally checks old theta/y caches if found.
    # Their complete absence is mandatory, so production cannot enter that
    # branch even if a user accidentally populates the new run directory.
    forbidden=[simulation_root/name for name in ('inputs_finite_v2','inputs_sparse_v2')]
    if any(path.exists() for path in forbidden):
        raise RuntimeError('Historical population caches are forbidden in the clean v2 workspace')


def generate(task):
    scenario,rep=task
    inputs=WORKER['inputs']
    folder=inputs.OUT/'inputs'/scenario/f'rep_{rep:05d}'
    if folder.exists():
        raise FileExistsError('Do not mix partial or previous generation outputs')
    inputs.export_replication(scenario,rep)
    result=WORKER['validation'].validate(folder)
    result['manifest_sha256']=inputs.sha256(folder/'manifest.json')
    proof=WORKER['reports'][-1]
    (folder/'orientation_generation_receipt.json').write_text(json.dumps(proof,indent=2)+'\n')
    return result


def prepare800(study,orientation,workers):
    legacy_worker.setup(study)
    import benchmark_prepare_final as original
    reject_archive_population_caches(study/'resultados/simulaciones')
    protocol=original.check_protocol()
    stages=original.task_plan(protocol)
    tick=time.monotonic();checks=[]
    import multiprocessing as mp
    for label,tasks in stages:
        stage=[]
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
                                 initializer=initialize,initargs=(str(study),str(orientation)))as pool:
            futures={pool.submit(generate,t):t for t in tasks}
            for future in as_completed(futures):
                row=future.result();stage.append(row);checks.append(row)
                if len(stage)%25==0 or len(stage)==len(tasks):
                    print(json.dumps({'stage':label,'completed':len(stage),'requested':len(tasks),
                                      'total_validated':len(checks),'seconds':time.monotonic()-tick}),flush=True)
        marker={'ready':True,'stage':label,'validated_populations':len(stage),'checks':stage,
                'authorizes_model_fits':False,'cross_archive_observed_input_identity_still_required':True,
                'protocol_sha256':original.sha256(original.OUT/'protocol_freeze.json'),
                'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
        (original.OUT/'task_plan'/f'{label}_READY.json').write_text(json.dumps(marker,indent=2)+'\n')
    assert len(checks)==800
    result={'passed':True,'independent_populations':800,'checks':checks,
            'authorizes_model_fits':False,'cross_archive_observed_input_identity_still_required':True,
            'protocol_sha256':original.sha256(original.OUT/'protocol_freeze.json'),
            'seconds_total':time.monotonic()-tick,'dispatcher_sha256':legacy_worker.sha(Path(__file__)),
            'original_dispatcher_sha256':original.sha256(Path(original.__file__)),
            'validation_source_sha256':original.sha256(Path(original.__file__).with_name('benchmark_validate_inputs.py')),
            'input_generation_correction':'Unspecified DGP eigenvector signs fixed by public immutable geometric anchors',
            'orientation_rule_sha256':legacy_worker.sha(orientation),
            'source_models_population_seeds_and_diagnostic_gates_unchanged':True,
            'historical_responses_truth_geometry_arrays_or_fits_loaded_in_production':False}
    (original.OUT/'input_validation_final800.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='checks'},indent=2))


def verify_identity(study,reference,output,catalogue=None):
    if catalogue:
        return verify_catalogue_identity(study,catalogue,output)
    from legacy_input_audit import audit_world,PRIMARY,STRUCTURAL
    suffix=Path('resultados/resultados_comparadores/inputs')
    jobs=[(s,r)for s in PRIMARY for r in range(10000,10200)]+[(s,r)for s in STRUCTURAL for r in range(20000,20100)]
    reports=[];failures=[];geometry_max=0.0
    for index,(scenario,rep)in enumerate(jobs,1):
        record=audit_world(study/suffix,reference/suffix,scenario,rep)
        observed=all(v['bytes_identical']for v in record['csv'].values())and record['direct_keys_match']and all(
                    v['exact_values']for v in record['direct_arrays'].values())
        geometry=record['geometry']['implied_covariance_Gram']
        if not observed or not geometry['within_roundoff']:
            failures.append({'scenario':scenario,'replication':rep,'observed_exact':observed,'geometry':geometry})
        geometry_max=max(geometry_max,geometry['maximum_absolute_error'])
        reports.append(record)
        if index%100==0:print(json.dumps({'input_identity_checked':index,'failures':len(failures)}),flush=True)
    result={'status':'PASS_800_OBSERVED_INPUTS_EXACT_FIT_GEOMETRY_ROUNDOFF'if not failures else'FAIL_INPUT_IDENTITY_NO_FITS_PERMITTED',
        'worlds':800,'primary_worlds':600,'structural_worlds':200,'csv_hashes_checked':4000,
        'failures':failures,'maximum_fit_geometry_Gram_difference':geometry_max,'world_records':reports,
        'no_archive_truth_population_outcomes_metrics_or_fitted_predictions_opened':True,
        'input_generation_correction_explicit':True,'model_fits_dispatched':False,
        'orientation_rule_sha256':legacy_worker.sha(Path(__file__).with_name('legacy_orientation_rule.json')),
        'time_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in ('world_records','failures')},indent=2))
    if failures:raise RuntimeError('Observed input identity failed; no fitting permitted')


def verify_catalogue_identity(study,catalogue_path,output):
    """Portable observed-input check: expected digests, geometry comparison only."""
    import numpy as np
    from legacy_orientation_adapter import array_sha
    from legacy_input_audit import PRIMARY,STRUCTURAL,CSV_FILES
    catalogue=json.loads(catalogue_path.read_text())
    if catalogue['format']!='legacy-observed-input-checksums-v1':raise ValueError('Unknown checksum catalogue')
    expected={(s,r)for s in PRIMARY for r in range(10000,10200)}|{(s,r)for s in STRUCTURAL for r in range(20000,20100)}
    received={(w['scenario'],w['replication'])for w in catalogue['worlds']}
    if received!=expected or len(catalogue['worlds'])!=800:raise ValueError('Catalogue must contain each frozen world once')
    if set(catalogue['csv_files'])!=set(CSV_FILES):raise ValueError('Unexpected CSV catalogue allowlist')
    base=study/'resultados/resultados_comparadores/inputs';records=[];failures=[];geometry_max=0.
    for index,world in enumerate(catalogue['worlds'],1):
        scenario=world['scenario'];rep=world['replication'];folder=base/scenario/f'rep_{rep:05d}'/'model_inputs'
        csv={name:legacy_worker.sha(folder/name)==world['csv'][name]for name in CSV_FILES}
        direct={}
        with np.load(folder/'direct.npz',allow_pickle=False)as values:
            key_match=set(values.files)==set(world['direct'])
            if not set(values.files).issubset(set(catalogue['direct_keys_allowlist'])):raise ValueError('Unexpected direct key')
            for key in values.files:
                value=values[key];target=world['direct'].get(key,{})
                direct[key]=(list(value.shape)==target.get('shape')and str(value.dtype)==target.get('dtype')and array_sha(value)==target.get('sha256'))
        reference=(catalogue_path.parent/world['fit_basis_reference']).resolve()
        if not reference.is_relative_to(catalogue_path.parent.resolve()):raise ValueError('Unsafe comparison geometry reference')
        old=np.load(reference,allow_pickle=False);fresh=np.load(folder/'fit_basis.npy',allow_pickle=False)
        if legacy_worker.sha(reference)!=world['fit_basis_sha256']:raise ValueError('Comparison geometry checksum changed')
        if old.shape!=fresh.shape:raise ValueError('Comparison fit geometry shape changed')
        gram=float(np.max(np.abs(old@old.T-fresh@fresh.T)));geometry_max=max(geometry_max,gram)
        passed=all(csv.values())and key_match and all(direct.values())and gram<=1e-12
        row={'scenario':scenario,'replication':rep,'csv_hashes_identical':csv,'direct_array_checksums_identical':direct,
             'direct_keys_match':key_match,'fit_geometry_Gram_maximum_absolute_error':gram,'passed':passed}
        records.append(row)
        if not passed:failures.append(row)
        if index%100==0:print(json.dumps({'catalogue_input_identity_checked':index,'failures':len(failures)}),flush=True)
    result={'status':'PASS_800_OBSERVED_INPUTS_EXACT_FIT_GEOMETRY_ROUNDOFF'if not failures else'FAIL_INPUT_IDENTITY_NO_FITS_PERMITTED',
        'worlds':800,'primary_worlds':600,'structural_worlds':200,'csv_hashes_checked':4000,'failures':failures,
        'maximum_fit_geometry_Gram_difference':geometry_max,'world_records':records,
        'expected_catalogue_sha256':legacy_worker.sha(catalogue_path),'verification_mode':'public observed-input checksums',
        'comparison_only_expected_fit_basis_never_provided_to_estimators':True,
        'no_archive_truth_population_outcomes_metrics_or_fitted_predictions_opened':True,
        'input_generation_correction_explicit':True,'model_fits_dispatched':False,
        'orientation_rule_sha256':legacy_worker.sha(Path(__file__).with_name('legacy_orientation_rule.json')),
        'time_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in ('world_records','failures')},indent=2))
    if failures:raise RuntimeError('Observed input checksum identity failed; no fitting permitted')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root',type=Path,required=True)
    p.add_argument('--reference-root',type=Path)
    p.add_argument('--input-catalogue',type=Path,help='Public observed-input checksums; replaces private model_inputs references')
    p.add_argument('--orientation-rule',type=Path,default=Path(__file__).with_name('legacy_orientation_rule.json'))
    p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('action',choices=('prepare800','verify-input-identity'))
    p.add_argument('--output',type=Path)
    a=p.parse_args();study=a.study_root.resolve()
    if a.action=='prepare800':prepare800(study,a.orientation_rule.resolve(),a.workers)
    else:verify_identity(study,a.reference_root.resolve()if a.reference_root else None,
                         a.output or study.parent/'legacy_v2_input_identity.json',
                         a.input_catalogue.resolve()if a.input_catalogue else None)


if __name__=='__main__':main()
