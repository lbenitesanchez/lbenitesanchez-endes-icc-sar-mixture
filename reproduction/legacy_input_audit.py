#!/usr/bin/env python3
"""Read-only audit of all 800 regenerated fit-facing input sets.

Never opens evaluation_only, finite-population truth, population outcomes,
generating covariance, fitted predictions or performance metrics. CSV bytes
are compared by SHA256; direct NPZ arrays and coordinate-derived fit bases
are compared semantically. All discrepancies remain in the certificate.
Geometry differences are reported separately from observed-sample inputs.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import numpy as np

PRIMARY=('binary_informative_mar','sparse_srs_2025','sparse_psu_2025')
STRUCTURAL=('sparse_srs_iid','sparse_psu_bym2')
CSV_FILES=('sample.csv','sample_all_available.csv','area.csv','population_X.csv','adjacency_edges.csv')
DIRECT_KEYS={'y','D','n_eff','sample_idx','df_original_proxy','df_primary'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def permitted(directory,name):
    if directory.name!='model_inputs' or name not in {*CSV_FILES,'direct.npz','fit_basis.npy'}:
        raise ValueError('Only explicitly permitted fit-facing inputs can be read')
    path=directory/name
    if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
        raise FileNotFoundError('Input missing or outside model_inputs: '+str(path))
    return path


def array_comparison(a,b):
    result={'fresh_shape':list(a.shape),'reference_shape':list(b.shape),
            'fresh_dtype':str(a.dtype),'reference_dtype':str(b.dtype)}
    if a.shape!=b.shape:
        return {**result,'same_shape':False,'exact_values':False,'within_roundoff':False}
    same_nan=np.array_equal(np.isnan(a),np.isnan(b)) if a.dtype.kind in 'fc' and b.dtype.kind in 'fc' else True
    exact=np.array_equal(a,b,equal_nan=True)
    numeric=a.dtype.kind in 'biufc' and b.dtype.kind in 'biufc'
    if numeric:
        finite=np.isfinite(a)&np.isfinite(b)
        maxerror=float(np.max(np.abs(a[finite].astype(float)-b[finite].astype(float))))if finite.any()else 0.0
        within=bool(np.allclose(a,b,atol=1e-12,rtol=1e-12,equal_nan=True))
        mismatches=~np.isclose(a,b,atol=0,rtol=0,equal_nan=True)
        result.update(maximum_absolute_error=maxerror,bitwise_different_values=int(mismatches.sum()))
    else:
        within=bool(exact)
    return {**result,'same_shape':True,'same_missingness':bool(same_nan),
            'exact_values':bool(exact),'within_roundoff':within}


def csv_semantics(a,b):
    """Only for differing CSV bytes; no row values are exposed in the report."""
    with a.open(newline='',encoding='utf-8-sig')as f:
        ar=list(csv.reader(f))
    with b.open(newline='',encoding='utf-8-sig')as f:
        br=list(csv.reader(f))
    result={'fresh_rows':len(ar)-1,'reference_rows':len(br)-1,'headers_equal':bool(ar and br and ar[0]==br[0])}
    if len(ar)!=len(br) or not result['headers_equal'] or any(len(x)!=len(y)for x,y in zip(ar,br)):
        return {**result,'same_shape':False,'semantically_equal':False}
    differences={};maximum=0.0;format_only=0
    for aa,bb in zip(ar[1:],br[1:]):
        for column,av,bv in zip(ar[0],aa,bb):
            if av==bv:
                continue
            try:
                x,y=float(av),float(bv)
                if np.isnan(x)and np.isnan(y):
                    format_only+=1
                    continue
                error=abs(x-y)
                maximum=max(maximum,error)
                if np.isclose(x,y,atol=1e-12,rtol=1e-12,equal_nan=True):
                    format_only+=1
                    continue
            except ValueError:
                pass
            differences[column]=differences.get(column,0)+1
    return {**result,'same_shape':True,'semantically_equal':not differences,
            'serialization_or_roundoff_cells':format_only,'maximum_absolute_numeric_difference':maximum,
            'different_cells_by_column':differences}


def audit_world(fresh,reference,scenario,rep):
    relative=Path(scenario)/f'rep_{rep:05d}'/'model_inputs'
    a=fresh/relative;b=reference/relative
    record={'scenario':scenario,'replication':rep,'csv':{},'direct_arrays':{},'geometry':{}}
    for name in CSV_FILES:
        af,bf=permitted(a,name),permitted(b,name)
        ah,bh=sha(af),sha(bf)
        record['csv'][name]={'fresh_sha256':ah,'reference_sha256':bh,'bytes_identical':ah==bh}
        if ah!=bh:
            record['csv'][name]['semantic_diagnostic']=csv_semantics(af,bf)
    af,bf=permitted(a,'direct.npz'),permitted(b,'direct.npz')
    with np.load(af,allow_pickle=False)as az,np.load(bf,allow_pickle=False)as bz:
        if not set(az.files).issubset(DIRECT_KEYS) or not set(bz.files).issubset(DIRECT_KEYS):
            raise ValueError('Unknown or forbidden array in direct.npz; never open truth arrays')
        record['direct_keys_match']=set(az.files)==set(bz.files)
        for key in sorted(set(az.files)&set(bz.files)):
            record['direct_arrays'][key]=array_comparison(az[key],bz[key])
    record['direct_container_sha256']={'fresh':sha(af),'reference':sha(bf),
                                      'container_hash_not_substituted_for_array_values':True}
    af,bf=permitted(a,'fit_basis.npy'),permitted(b,'fit_basis.npy')
    x,y=np.load(af,allow_pickle=False),np.load(bf,allow_pickle=False)
    comparison=array_comparison(x,y)
    geometry={'basis':comparison,'fresh_sha256':sha(af),'reference_sha256':sha(bf)}
    if x.shape==y.shape:
        geometry['implied_covariance_Gram']=array_comparison(x@x.T,y@y.T)
        if not comparison['exact_values']:
            # A sign/basis rotation can preserve the Gaussian field law; this
            # is a diagnostic only, never a repair or an input substitution.
            ux=np.linalg.svd(x,full_matrices=False)[0]
            uy=np.linalg.svd(y,full_matrices=False)[0]
            geometry['column_space_projector']=array_comparison(ux@ux.T,uy@uy.T)
    record['geometry']=geometry
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root',type=Path,required=True)
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();tick=time.monotonic()
    suffix=Path('resultados/resultados_comparadores/inputs')
    fresh=a.study_root.resolve()/suffix;reference=a.reference_root.resolve()/suffix
    jobs=[(s,r)for s in PRIMARY for r in range(10000,10200)]
    jobs +=[(s,r)for s in STRUCTURAL for r in range(20000,20100)]
    assert len(jobs)==800 and len(set(jobs))==800
    worlds=[];errors=[]
    for index,(scenario,rep)in enumerate(jobs,1):
        try:worlds.append(audit_world(fresh,reference,scenario,rep))
        except Exception as error:errors.append({'scenario':scenario,'replication':rep,'error':repr(error)})
        if index%100==0:
            print(json.dumps({'audited_requested_worlds':index,'complete_world_records':len(worlds),
                              'exceptions':len(errors),'seconds':time.monotonic()-tick}),flush=True)
    csv_differences=[{'scenario':w['scenario'],'replication':w['replication'],'file':name,
                      'semantic_diagnostic':v.get('semantic_diagnostic')}
                     for w in worlds for name,v in w['csv'].items()if not v['bytes_identical']]
    direct_differences=[{'scenario':w['scenario'],'replication':w['replication'],'array':name,'comparison':v}
                        for w in worlds for name,v in w['direct_arrays'].items()if not v['exact_values']]
    geometry_differences=[{'scenario':w['scenario'],'replication':w['replication'],'geometry':w['geometry']}
                          for w in worlds if not w['geometry']['basis']['exact_values']]
    exact=not errors and len(worlds)==800 and not csv_differences and not direct_differences and not geometry_differences
    numerical=(not errors and len(worlds)==800 and all(w['direct_keys_match']and
                all(v['within_roundoff']for v in w['direct_arrays'].values())for w in worlds) and
                all(v['semantic_diagnostic']['semantically_equal']for v in csv_differences))
    result={'status':'PASS_800_FIT_INPUTS_EXACT'if exact else'INPUT_DIFFERENCES_RETAINED'if numerical else'FAIL_INPUT_DIFFERENCES_OR_MISSING',
        'requested_worlds':800,'complete_world_records':len(worlds),'errors':errors,
        'csv_files_checked':len(worlds)*len(CSV_FILES),'csv_byte_differences':csv_differences,
        'direct_array_differences':direct_differences,'geometry_differences':geometry_differences,
        'observed_inputs_equal_within_explicit_roundoff':numerical,
        'all_input_values_and_geometry_exact':exact,'worlds':worlds,'seconds':time.monotonic()-tick,
        'time_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'one_CPU_thread':True,
        'evaluation_only_truth_population_outcomes_or_generating_covariance_opened':False,
        'predictions_draws_metrics_or_scores_opened':False,'models_fit_or_inputs_modified':False,
        'seeds_models_sampler_or_numerical_gates_changed':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in ('worlds','csv_byte_differences','direct_array_differences','geometry_differences')},indent=2))


if __name__=='__main__':main()
