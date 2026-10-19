"""Explicit numerical orientation correction for the historical legacy DGP.

The original source recomputed an RBF eigenbasis without fixing column signs.
The same RNG coefficients applied to sign-flipped vectors generate different
seeded populations although the Gaussian field covariance is unchanged.
Only fixed public anchor/sign constants enter production: no historical B,
truth, observed response or fitted object is loaded by this adapter.
"""
from pathlib import Path
import hashlib
import json
import numpy as np


def array_sha(array):
    value=np.ascontiguousarray(array)
    h=hashlib.sha256()
    h.update(str(value.dtype).encode());h.update(json.dumps(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()


def orient(X,coords,basis,rule):
    if list(X.shape)!=rule['X_shape'] or list(coords.shape)!=rule['coordinates_shape'] or list(basis.shape)!=rule['basis_shape']:
        raise ValueError('DGP geometry shape differs from the fixed orientation contract')
    if array_sha(X)!=rule['X_sha256'] or array_sha(coords)!=rule['coordinates_sha256']:
        raise ValueError('Fixed DGP auxiliary/coordinate inputs differ')
    distances=np.sum((coords[:,None]-coords[None,:])**2,axis=2)
    kernel=np.exp(-distances/(2*.24**2))
    if array_sha(kernel)!=rule['DGP_RBF_K_sha256']:
        raise ValueError('DGP RBF covariance inputs differ')
    anchors=rule['anchors']
    if len(anchors)!=basis.shape[1] or len({a['row']for a in anchors})!=len(anchors):
        raise ValueError('Every column requires its own immutable anchor row')
    signs=[];margins=[]
    for col,anchor in enumerate(anchors):
        if anchor['column']!=col:
            raise ValueError('Ordered column/anchor mapping differs')
        value=float(basis[anchor['row'],col]);minimum=anchor['minimum_required_absolute_anchor']
        if not np.isfinite(value) or abs(value)<=minimum:
            raise ValueError('DGP anchor is unstable or near zero')
        signs.append(anchor['desired_sign']*(1 if value>0 else -1))
        margins.append(abs(value)-minimum)
    aligned=basis*np.asarray(signs)[None,:]
    error=float(np.max(np.abs(basis@basis.T-aligned@aligned.T)))
    if error>1e-12:
        raise ValueError('Sign orientation changed the Gaussian field covariance')
    return aligned,{'status':'PUBLIC_FIXED_ANCHOR_DGP_ORIENTATION_APPLIED',
        'multipliers':signs,'minimum_anchor_margin':min(margins),'Gram_change_maximum':error,
        'fresh_unoriented_basis_sha256':array_sha(basis),'fresh_oriented_basis_sha256':array_sha(aligned),
        'historical_geometry_array_read_in_production':False,'historical_truth_responses_or_fits_loaded':False}


def install(benchmark_inputs,rule_path):
    rule=json.loads(Path(rule_path).read_text())
    original_load=benchmark_inputs.load_module
    reports=[]
    def load_module(name,path):
        module=original_load(name,path)
        if Path(path).name in ('03_simulation.py','03b_sparse_simulation.py'):
            target=module.sim if hasattr(module,'sim')else module
            old_geometry=target.geometry
            def geometry(*args,**kwargs):
                X,coords,basis=old_geometry(*args,**kwargs)
                aligned,proof=orient(X,coords,basis,rule)
                reports.append(proof)
                return X,coords,aligned
            target.geometry=geometry
        return module
    benchmark_inputs.load_module=load_module
    return reports
