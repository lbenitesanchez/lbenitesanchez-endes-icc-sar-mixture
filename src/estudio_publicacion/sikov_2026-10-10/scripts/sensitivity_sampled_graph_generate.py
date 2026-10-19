"""Prospective descriptive sensitivity favourable to the sampled-only SAR.

This script copies the frozen probability-SAR/SRS population and survey design,
with independently declared new streams. Its sole generative structural change
is a triangular spatial graph: sampled-only W11 and zero W10. Fitters still
receive the original full-frame graph and all identical shared inputs.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
from pathlib import Path
import argparse, datetime as dt, hashlib, json
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.special import expit, logit, ndtr
from comparison_simulation import BETAS, NAMES
from sikov_published_reference import triangular_weights

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT.parent
DEST = ROOT / 'sensibilidad_sampled_graph'
CONTEXT = 'SAR_probability_sampled_graph_srs'
IDS = tuple(range(30000, 30064))
METHODS = ('Sikov_FH', 'Sikov_SAR', 'Sikov_FH_BC', 'Sikov_SAR_BC',
           'Sikov_SAR_k4', 'Sikov_SAR_k8', 'Sikov_SAR_haversine', 'ICC',
           'Selected', 'Sikov_SAR_Published', 'Sikov_SAR_Published_BC')
PLAN = ROOT / 'validacion/sensitivity_sampled_graph_freeze.json'
SCRIPT_NAMES = tuple('sensitivity_sampled_graph_' + name + '.py'
                     for name in ('generate', 'fit', 'audit', 'score', 'qa'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=lambda x:
        x.tolist() if isinstance(x, np.ndarray) else
        x.item() if isinstance(x, np.generic) else str(x)) + '\n')
    temporary.replace(path)


def local(path):
    """Resolve seals from the original host or a preserved public tree."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    elif 'sikov_2026-10-10' in p.parts:
        p = ROOT.joinpath(*p.parts[p.parts.index('sikov_2026-10-10') + 1:])
    elif 'estudio_publicacion' in p.parts:
        p = STUDY.joinpath(*p.parts[p.parts.index('estudio_publicacion') + 1:])
    else:
        raise ValueError('Unsupported sealed path ' + str(path))
    return p


def frozen_sources_gate():
    base = ROOT / 'protocol_freeze.json'
    addpath = ROOT / 'validacion/sikov_published_addendum_freeze.json'
    protocol = json.loads(base.read_text())
    add = json.loads(addpath.read_text())
    if not protocol['approved_for_fresh_fits']:
        raise RuntimeError('Base protocol not approved')
    if not add['approved_for_additional_reference_fits'] or \
            add['accuracy_or_heldout_outcomes_read_before_addendum']:
        raise RuntimeError('Pre-outcome additional reference required')
    if add['base_protocol_sha256'] != sha(base):
        raise RuntimeError('Changed base protocol')
    for record, blocks in ((protocol, ('statistical_sources_sha256',
            'frozen_inputs_sha256', 'runtime_binary_sha256')),
            (add, ('additional_sources_sha256', 'frozen_inputs_sha256',
                   'runtime_binary_sha256', 'base_kernel_sources_sha256'))):
        for block in blocks:
            for path, digest in record.get(block, {}).items():
                if sha(local(path)) != digest:
                    raise RuntimeError('Changed frozen scientific input ' + path)
    if protocol['selected_recipe'] != {'family': 'ICC_SAR_mixture',
                                       'icc_weight': .75}:
        raise RuntimeError('Original mixture recipe differs')
    return protocol, sha(base), sha(addpath)


def gate():
    protocol, base, add = frozen_sources_gate()
    plan = json.loads(PLAN.read_text())
    if plan.get('status') != 'FROZEN_PROSPECTIVE_DESCRIPTIVE_SENSITIVITY' or \
            plan.get('accuracy_or_truth_read_before_freeze') is not False:
        raise RuntimeError('Approved prospective sensitivity freeze required')
    if plan['protocol_sha256'] != base or plan['addendum_sha256'] != add:
        raise RuntimeError('Changed parent scientific seals')
    if plan['context'] != CONTEXT or plan['replications'] != list(IDS) or \
            plan['methods'] != list(METHODS):
        raise RuntimeError('Sensitivity universe changed')
    for path, digest in plan['sources_sha256'].items():
        if sha(local(path)) != digest:
            raise RuntimeError('Changed prospective sensitivity source ' + path)
    for path, digest in plan['inputs_sha256'].items():
        if sha(local(path)) != digest:
            raise RuntimeError('Changed prospective sensitivity template ' + path)
    if sha(local(plan['prefit_QA_path'])) != plan['prefit_QA_sha256']:
        raise RuntimeError('Changed prefit algebraic QA')
    return plan, sha(PLAN), protocol


def generate(rep):
    plan, plan_sha, _ = gate()
    if rep not in IDS:
        raise ValueError('Unplanned population stream')
    root = DEST / 'inputs' / f'rep_{rep:05d}'
    if root.exists():
        raise FileExistsError(root)
    a = dict(np.load(ROOT / 'datos/simulation_template/auxiliary.npz',
                     allow_pickle=False))
    X, strata, m = a['X_raw'], a['strata'], len(a['X_raw'])
    rng = np.random.default_rng(np.random.SeedSequence([20261010, 971, rep]))
    srng = np.random.default_rng(np.random.SeedSequence([20261010, 972, rep]))
    # Separate sampling RNG: declaring the support before the spatial draw does
    # not change the original population RNG sequence or later sampling design.
    sampled = np.sort(np.concatenate([srng.choice(np.flatnonzero(strata == g),
                         32, replace=False) for g in (1, 2, 3)]))
    Wgen = triangular_weights(a['coords'], sampled, strata, k=6)
    base = np.zeros((m, 2))
    for t, name in enumerate(('anemia', 'stunting')):
        for g in (1, 2, 3):
            mask = strata == g
            for key, b in BETAS[name][g].items():
                base[mask, t] += X[mask, NAMES.index(key)] * b
    base = np.clip(base, .03, .75)
    innov = rng.normal(size=(m, 2))
    innov[:, 1] = .4 * innov[:, 0] + np.sqrt(.84) * innov[:, 1]
    C = solve(np.eye(m) - .6 * Wgen, np.eye(m))
    # Identical normalization convention to the frozen probability-SAR DGP;
    # the denominator uses the new graph, not an endpoint or realized outcome.
    C /= np.sqrt(np.mean(np.sum(C * C, axis=1)))
    u = C @ innov
    p = np.clip(base + u * np.array([.07, .05]), .005, .90)
    J, M, L = 6, 60, 24
    ys, variances, kish, rows, truth, boundary = [], [], [], [], [], []
    for i in range(m):
        psu = rng.normal(size=(M, 2))
        psu[:, 1] = .5 * psu[:, 0] + np.sqrt(.75) * psu[:, 1]
        psu[:] = 0
        pp = expit(logit(p[i])[None, :] + psu)
        H = rng.binomial(1, .5, size=(M, L))
        z = rng.normal(size=(M, L, 2))
        z[:, :, 1] = .4 * z[:, :, 0] + np.sqrt(.84) * z[:, :, 1]
        yy = (ndtr(z) < pp[:, None, :]).astype(float)
        truth.append(yy.mean(axis=(0, 1)))
        if i not in sampled:
            continue
        selected = srng.choice(M, J, replace=False)
        num, den, sumsq = np.zeros((J, 2)), np.zeros((J, 2)), np.zeros(2)
        for j, k in enumerate(selected):
            q = np.where(H[k] == 1, .40, .80)
            children = np.flatnonzero(srng.uniform(size=L) < q)
            values = yy[k, children]
            obs = srng.uniform(size=values.shape) < np.array([.85, .90])
            ww = np.where(obs, (M / J) /
                          (q[children, None] * np.array([.85, .90])), 0.)
            num[j] = (ww * values).sum(0)
            den[j] = ww.sum(0)
            sumsq += (ww * ww).sum(0)
            for child, value, seen, w in zip(children, values, obs, ww):
                rows.append({'area': i, 'psu': f'{i}:{int(k)}', 'stratum': i,
                    'child_id': f'{i}:{int(k)}:{int(child)}',
                    'y_anemia': float(value[0]) if seen[0] else np.nan,
                    'y_stunting': float(value[1]) if seen[1] else np.nan,
                    'weight_anemia': float(w[0]) if seen[0] else np.nan,
                    'weight_stunting': float(w[1]) if seen[1] else np.nan})
        totals = den.sum(0)
        y = num.sum(0) / totals
        residual = num - den * y
        D = J / (J - 1) * (residual.T @ residual) / \
            (totals[:, None] * totals[None, :])
        n = totals * totals / sumsq
        ys.append(y); variances.append(D); kish.append(n)
        boundary.append((y == 0) | (y == 1))
    y, D, ne = np.asarray(ys), np.asarray(variances), np.asarray(kish)
    D_raw = D.copy()
    rawdiag = np.diagonal(D, axis1=1, axis2=2).copy()
    padj = (ne * y + .5) / (ne + 1)
    ref = padj * (1 - padj) / np.maximum(ne, 1)
    ratio = rawdiag / np.maximum(ref, 1e-12)
    for t in range(2):
        valid = (rawdiag[:, t] > 1e-15) & (y[:, t] > 0) & (y[:, t] < 1)
        deff = np.median(ratio[valid, t]) if valid.any() else 1.
        ref[:, t] *= max(deff, .05)
        mix = (J - 1) / (J - 1 + 4)
        DD = mix * rawdiag[:, t] + (1 - mix) * ref[:, t]
        DD = np.where((y[:, t] == 0) | (y[:, t] == 1), ref[:, t], DD)
        D[:, t, t] = np.maximum(DD, 1e-10)
    D[:, 0, 1] = D[:, 1, 0] = 0.
    if y.shape != (96, 2) or not np.isfinite(y).all() or \
            not np.isfinite(D).all() or not np.isfinite(ne).all() or \
            np.any(np.diagonal(D, axis1=1, axis2=2) <= 0):
        raise RuntimeError('Incomplete generated common SRS support; do not resample')
    fit, ev = root / 'model_inputs', root / 'evaluation_only'
    fit.mkdir(parents=True); ev.mkdir()
    np.savez_compressed(fit / 'inputs.npz', **{k: a[k] for k in
        ('X_raw', 'X', 'strata', 'coords', 'W', 'basis', 'covariate_names')},
        y=y, D=D, D_raw=D_raw, n_eff=ne, sample_idx=sampled)
    pd.DataFrame(rows).to_csv(fit / 'sample.csv', index=False)
    # Oracle and generative W are never supplied to an estimator.
    np.savez_compressed(ev / 'truth.npz', theta=np.asarray(truth),
                        sample_idx=sampled)
    save(root / 'manifest.json', {'context': CONTEXT, 'replication': rep,
        'phase': 'prospective_descriptive_sensitivity',
        'protocol_sha256': plan['protocol_sha256'],
        'addendum_sha256': plan['addendum_sha256'], 'sensitivity_plan_sha256': plan_sha,
        'population_stream': [20261010, 971, rep],
        'sampling_stream': [20261010, 972, rep], 'areas': m,
        'stratum_sizes': [40, 40, 40], 'sampled_stratum_sizes': [32, 32, 32],
        'population_PSUs_per_area': M, 'children_per_PSU': L,
        'sampled_areas': len(sampled), 'sampled_PSUs_per_area': J,
        'boundary_endpoints': int(np.sum(boundary)),
        'same_shared_X_y_D_n_psu_for_methods': True,
        'truth_excluded_from_model_inputs': True,
        'generative_graph_excluded_from_model_inputs': True,
        'generative_graph': 'W11 sampled-only within-stratum k6; W10 zero; non-sampled rows full-frame within-stratum k6',
        'generative_rho': .6, 'generative_W_sha256':
            hashlib.sha256(np.ascontiguousarray(Wgen).tobytes()).hexdigest(),
        'fitter_graph': 'Unchanged original full-frame W6 supplied to shared input; Published reconstructs its declared endpoint graph from coordinates',
        'published_coefficients_unchanged': True,
        'SAR_probability_clipping': 'base risk [0.03,.75], probability SAR [.005,.90]',
        'design_variance': 'Same frozen SRS ultimate-cluster Taylor and GVF stabilization, all from training sample',
        'source_sha256': sha(__file__), 'fit_inputs_sha256': sha(fit / 'inputs.npz'),
        'sample_sha256': sha(fit / 'sample.csv'), 'truth_sha256': sha(ev / 'truth.npz')})
    gate()
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', type=int, required=True)
    args = parser.parse_args()
    print(generate(args.rep))
