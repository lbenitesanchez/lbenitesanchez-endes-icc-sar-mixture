#!/usr/bin/env python3
"""Declare the completed run's real geographic basis as a future input target.

Public targets contain exactly district codes, coordinates and the geographic
basis. Applying the lock is allowed only before any real fits. Other NPZ
members are preserved as opaque bytes: responses, weights and information
arrays are never interpreted, exported or changed by this adapter.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import zipfile
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ALLOW = {'ubigeo','coords','basis'}
SCOPES = {'temporal':'real_2019_full','territorial':'real_2024_training'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def array_sha(a):
    h=hashlib.sha256();h.update(a.dtype.str.encode());h.update(json.dumps(list(a.shape)).encode());h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def geometry(path,public=False):
    with np.load(path,allow_pickle=False)as z:
        if public and set(z.files)!=ALLOW:raise ValueError('Only the three geometry arrays may be public')
        if not ALLOW.issubset(z.files):raise ValueError('Prepared geometry fields are missing')
        result={k:z[k]for k in sorted(ALLOW)}
    codes,coords,b=result['ubigeo'],result['coords'],result['basis']
    if codes.dtype.kind not in 'US' or codes.ndim!=1 or len(codes)!=1874 or len(set(codes.tolist()))!=1874:
        raise ValueError('Unique public 1874-district order is required')
    if any(not re.fullmatch(r'\d{6}',str(x))for x in codes):raise ValueError('District keys must have six digits')
    if coords.shape!=(1874,2)or b.shape!=(1874,40)or coords.dtype!=np.float64 or b.dtype!=np.float64:
        raise ValueError('Current latitude/longitude coordinates or rank40 basis schema differs')
    if not np.isfinite(coords).all()or not np.isfinite(b).all():raise ValueError('Nonfinite geometry')
    if np.max(np.abs(b.mean(axis=0)))>1e-10 or not np.isclose(np.mean(np.sum(b*b,axis=1)),1,atol=1e-12,rtol=0):
        raise ValueError('The centered, unit mean-diagonal basis scale differs')
    return result

def graph(coords):
    # Exact unchanged source rule; no eigensolver or survey quantity is read.
    latlon=coords;m=len(coords)
    xy=np.column_stack((111.32*latlon[:,1]*np.cos(np.deg2rad(np.mean(latlon[:,0]))),111.32*latlon[:,0]))
    distance=cdist(xy,xy);np.fill_diagonal(distance,np.inf);k=min(8,m-1)
    ii=np.arange(m).repeat(k);jj=np.argpartition(distance,k-1,axis=1)[:,:k].ravel()
    w=coo_matrix((np.ones(len(ii)),(ii,jj)),shape=(m,m)).tocsr();w=w.maximum(w.T)
    degree=np.asarray(w.sum(axis=1)).ravel()
    ws=w.multiply((1/np.sqrt(degree))[:,None]).multiply((1/np.sqrt(degree))[None,:]).tocsr()
    old=hashlib.sha256()
    for a in (ws.indptr,ws.indices,ws.data):old.update(a.dtype.str.encode());old.update(a.tobytes())
    ws.sort_indices();canonical=hashlib.sha256()
    for a in (ws.indptr.astype('<i8'),ws.indices.astype('<i8'),ws.data.astype('<f8')):
        canonical.update(a.dtype.str.encode());canonical.update(a.tobytes())
    count,labels=connected_components(ws,directed=False)
    return {'kernel_graph_byte_sha256':old.hexdigest(),'canonical_graph_sha256':canonical.hexdigest(),
        'graph_shape':list(ws.shape),'graph_nnz':ws.nnz,'connected_components':int(count),
        'component_sizes':np.bincount(labels).tolist(),'coordinate_order':['latitude','longitude'],
        'native_SAR_runtime_coordinate_order':['longitude','latitude'],
        'source_neighbors':8,'distance_rule':'111.32*longitude*cos(mean_latitude),111.32*latitude',
        'neighbor_rule':'source np.argpartition(distance,k-1) without stable boundary-tie rule',
        'symmetrization':'maximum directed adjacency and its transpose',
        'normalization':'degree^-1/2 adjacency degree^-1/2',
        'canonical_graph_hash_dtypes':['<i8','<i8','<f8'],'eigensolver_executed_by_lock':False}

def export_target(run,source_root,output_root):
    run=run.resolve();source_root=source_root.resolve();output_root=output_root.resolve()
    if output_root.exists():raise FileExistsError('Preserve earlier geometry target versions')
    arrays=[];bindings={}
    for scope,folder in SCOPES.items():
        directory=run/'datos'/folder;p=directory/'fit_inputs.npz';manifest=directory/'preparation_manifest.json'
        m=read(manifest);completion=run/'evaluacion'/('real_'+scope)/'completion.json';done=read(completion)
        if m['output_sha256']['fit_inputs.npz']!=sha(p)or done['training_manifest_sha256']!=sha(manifest):
            raise ValueError('Actual fitting did not bind this prepared geometry')
        if m['coordinates_npz_order']!=['latitude','longitude'] or m['heldout_outcome_rows_read']is not False:
            raise ValueError('Wrong coordinate or preparation scope')
        arrays.append(geometry(p));bindings[scope]={'actual_fit_completion_sha256':sha(completion),
            'actual_preparation_manifest_sha256':sha(manifest),'actual_private_fit_input_sha256':sha(p)}
    if any(not np.array_equal(arrays[0][k],arrays[1][k])for k in ALLOW):raise ValueError('Actual temporal/territorial geometry differs')
    g=graph(arrays[0]['coords']);diagnostic=read(run/'real_basis_diagnostic.json')
    if g['kernel_graph_byte_sha256']!=diagnostic['graph_sha256']:
        raise ValueError('Reconstructed graph differs from completed-run diagnostic')
    output_root.mkdir(parents=True);p=output_root/'real_geometry.npz';np.savez(p,**arrays[0]);geometry(p,True)
    result={'format':'iaos-real-current-geographic-target-v1','status':'COMPLETED_RUN_GEOMETRY_TARGET_EXPORTED',
        'target_classification':'DECLARED_FIXED_CURRENT_FINITE_RANK_GEOGRAPHIC_PRIOR_TARGET',
        'target_NPZ_sha256':sha(p),'public_array_allowlist':sorted(ALLOW),
        'arrays':{k:{'shape':list(a.shape),'dtype':a.dtype.str,'C_contiguous_array_sha256':array_sha(a)}for k,a in arrays[0].items()},
        'graph':g,'basis_centering':'column means removed','basis_scaling':'mean diagonal of BBtranspose equals one',
        'basis_dimensions':[1874,40],'model_source_sha256':sha(source_root/'scripts/models.py'),
        'actual_completed_fit_provenance':bindings,'original_basis_diagnostic_sha256':sha(run/'real_basis_diagnostic.json'),
        'fixed_target_equals_actual_completed_run_basis':True,'lock_adapter_used_in_completed_run':False,
        'geometry_only_no_responses_weights_information_masks_or_draws_exported':True,
        'future_unlocked_eigensolver_may_produce_another_finite_rank_target':True,
        'model_kernels_projection_or_component_rule_modified':False,'new_fits_scores_or_eigensolver_executed':False,
        'helper_sha256':sha(__file__)}
    write(output_root/'real_geometry_manifest.json',result);return result

def reject_fits(run):
    for prefix in ('evaluacion/real_temporal','evaluacion/real_territorial',
                   'evaluacion/published_sar/real_temporal','evaluacion/published_sar/real_territorial'):
        if (run/prefix).exists():raise ValueError('Basis target must be applied before any real fit directory exists')
    if any((run/('real_fit_receipt_'+s+'.json')).exists()for s in SCOPES):raise ValueError('Real fit receipts already exist')

def replace_basis_npz(path,basis):
    """Replace one member, retain every other array's raw .npy bytes exactly."""
    before={}
    with zipfile.ZipFile(path)as z:
        names=z.namelist()
        if len(names)!=len(set(names))or'basis.npy'not in names:raise ValueError('Malformed prepared NPZ')
        for n in names:
            if '/'in n or not n.endswith('.npy'):raise ValueError('Unsafe prepared NPZ member')
            before[n]=z.read(n)
    stream=io.BytesIO();np.save(stream,basis,allow_pickle=False)
    after={**before,'basis.npy':stream.getvalue()};temporary=path.with_suffix('.basis-lock-new.npz')
    if temporary.exists():raise FileExistsError('Preserve previous basis replacement temp files')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED)as z:
        for n,data in after.items():z.writestr(n,data)
    with zipfile.ZipFile(temporary)as z:
        if any(z.read(n)!=data for n,data in before.items()if n!='basis.npy'):
            raise ValueError('A nongeometry input changed')
    return temporary,{n:hashlib.sha256(data).hexdigest()for n,data in before.items()if n!='basis.npy'}

def apply_target(run,source_root,target,output,check_only=False):
    run=run.resolve();source_root=source_root.resolve();target=target.resolve();output=output.resolve()
    if run.is_relative_to(source_root.parent.parent) or output.is_relative_to(source_root.parent.parent):
        raise ValueError('Future training inputs and receipts must remain outside the public checkout')
    reject_fits(run);manifest=read(target/'real_geometry_manifest.json')
    locked=geometry(target/'real_geometry.npz',True)
    if sha(target/'real_geometry.npz')!=manifest['target_NPZ_sha256']or sha(source_root/'scripts/models.py')!=manifest['model_source_sha256']:
        raise ValueError('Current geographic target or scientific model source changed')
    if set(manifest['public_array_allowlist'])!=ALLOW or set(manifest['arrays'])!=ALLOW:
        raise ValueError('Current target manifest must name exactly geometry arrays')
    for name,array in locked.items():
        descriptor=manifest['arrays'][name]
        if (descriptor['C_contiguous_array_sha256']!=array_sha(array)
            or descriptor['shape']!=list(array.shape)or descriptor['dtype']!=array.dtype.str):
            raise ValueError('Locked geographic array descriptor differs')
    if output.exists():raise FileExistsError('Basis-lock receipt must be new')
    preparation_path=run/'real_preparation_receipt.json';preparation=read(preparation_path)
    if preparation.get('fresh_microdata_preparation_from_SAV')is not True:
        raise ValueError('Future basis lock requires freshly reconstructed SAV training inputs')
    preparation_backup=run/'real_preparation_receipt_before_current_basis_lock.json'
    if preparation_backup.exists():raise FileExistsError('Preserve earlier pre-lock preparation receipt')
    plans=[]
    for scope,folder in SCOPES.items():
        p=run/'datos'/folder/'fit_inputs.npz';m=p.with_name('preparation_manifest.json');prepared=geometry(p)
        if read(m)['output_sha256']['fit_inputs.npz']!=sha(p):raise ValueError('Fresh preparation seal differs')
        if not np.array_equal(prepared['ubigeo'],locked['ubigeo'])or not np.array_equal(prepared['coords'],locked['coords']):
            raise ValueError('District order or coordinates differ from the fixed target')
        if graph(prepared['coords'])['canonical_graph_sha256']!=manifest['graph']['canonical_graph_sha256']:
            raise ValueError('Newly prepared graph differs; cannot silently change the target')
        backup=p.parent/'before_current_basis_lock'
        if backup.exists():raise FileExistsError('Preserve previous input-lock backups')
        plans.append((scope,p,m,backup))
    result={'format':'iaos-real-current-basis-lock-receipt-v1',
        'status':'PASS_CHECK_ONLY_BEFORE_FITS'if check_only else'PASS_FIXED_CURRENT_BASIS_APPLIED_BEFORE_FITS',
        'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'target_manifest_sha256':sha(target/'real_geometry_manifest.json'),
        'target_NPZ_sha256':sha(target/'real_geometry.npz'),'helper_sha256':sha(__file__),
        'only_basis_array_changed':True,'all_other_array_members_retained_byte_identically':True,
        'raw_response_weight_information_or_draw_arrays_interpreted':False,'fits_or_scores_executed':False,
        'used_in_historical_completed_run':False,'model_kernels_changed':False,'scopes':{}}
    if not check_only:
        # Complete all validations before any input is replaced; retain originals.
        for scope,p,m,backup in plans:
            temporary,unchanged=replace_basis_npz(p,locked['basis']);backup.mkdir()
            shutil.copyfile(p,backup/'fit_inputs.npz');shutil.copyfile(m,backup/'preparation_manifest.json')
            previous_npz,previous_m=sha(p),sha(m);temporary.replace(p);new=read(m)
            new['output_sha256']['fit_inputs.npz']=sha(p)
            new['declared_current_basis_target']={'target_manifest_sha256':result['target_manifest_sha256'],
                'previous_preparation_manifest_sha256':previous_m,'previous_fit_input_sha256':previous_npz,
                'only_basis_array_changed':True,'actual_source_target_is_versioned_and_disclosed':True}
            write(m,new)
            if not np.array_equal(geometry(p)['basis'],locked['basis']):raise ValueError('Applied basis identity differs')
            result['scopes'][scope]={'previous_private_fit_input_sha256':previous_npz,
                'new_private_fit_input_sha256':sha(p),'previous_preparation_manifest_sha256':previous_m,
                'new_preparation_manifest_sha256':sha(m),'unchanged_nonbasis_array_member_sha256':unchanged}
        output.parent.mkdir(parents=True,exist_ok=True);write(output,result)
        # This newly prepared run has not fitted a model. Its original SAV
        # receipt remains intact in a backup, while the working receipt states
        # the additional target explicitly before subsequent fit seals bind it.
        shutil.copyfile(preparation_path,preparation_backup)
        preparation['original_pre_lock_preparation_receipt_sha256']=sha(preparation_backup)
        preparation['declared_current_basis_lock']={'receipt_sha256':sha(output),
            'target_manifest_sha256':result['target_manifest_sha256'],
            'unlocked_eigensolver_comparison_fields_describe_original_pre_lock_generation':True,
            'current_working_basis_equals_versioned_completed_run_target':True,
            'survey_rows_auxiliary_direct_information_and_model_sources_changed':False,
            'basis_target_was_applied_before_any_fit':True}
        write(preparation_path,preparation)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=('export-target','apply'),required=True)
    for name in ('run-root','source-root','target-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.action=='export-target':
        if a.check_only or a.output:raise ValueError('Export uses a new target directory')
        result=export_target(a.run_root,a.source_root,a.target_root)
    else:
        if not a.output:raise ValueError('Apply requires a new private --output receipt path')
        result=apply_target(a.run_root,a.source_root,a.target_root,a.output,a.check_only)
    print(json.dumps({'status':result['status'],'new_fits_or_scores':False,'public_geometry_only':True}))

if __name__=='__main__':main()
