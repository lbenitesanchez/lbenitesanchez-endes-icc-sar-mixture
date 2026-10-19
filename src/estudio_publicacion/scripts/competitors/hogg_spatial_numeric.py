"""Explicit numerical initializer for the unchanged full-rank Hogg S2 target.

No likelihood, parameter support, prior, graph, draw count or diagnostic gate
is changed. This adapter affects only NUTS initial points and its initial
diagonal kinetic metric. The original Stan source, data and executable are
used byte for byte. Four independent
small perturbations prevent identical chain initial points.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import sys
for _name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[_name] = '1'
import numpy as np
import hogg_tsln as h


def make_initial_values(data, *, chains, seed):
    """Data-derived starts only; none of these caps constrain the posterior."""
    sampled = np.asarray(data['sampled_area'], dtype=int) - 1
    Z = np.asarray(data['Z'], dtype=float)
    theta = np.asarray(data['theta_s1'], dtype=float)
    L = np.asarray(data['L'], dtype=float)
    stable = np.asarray(data['stable'], dtype=int) - 1
    sd = np.asarray(data['sampling_sd_s1'], dtype=float).copy()
    if not len(stable) or not np.isfinite(sd[stable]).all() or (sd[stable] <= 0).any():
        raise ValueError('Finite positive stable sampling SDs are required for numerical initialization')
    logsd = np.log(sd[stable])
    omega = np.linalg.lstsq(L[stable], logsd, rcond=None)[0]
    residual = logsd - L[stable] @ omega
    # Bounded starting values are neither model bounds nor truncation priors.
    sigma_gvf = float(np.clip(np.sqrt(np.mean(residual**2)), .01, 1.5))
    omega_cov = sigma_gvf**2*np.linalg.inv(L[stable].T@L[stable] + .25*np.eye(L.shape[1]))
    omega_jitter = np.clip(np.sqrt(np.diag(omega_cov)), 1e-4, .1)
    unstable = np.asarray(data['unstable'], dtype=int) - 1
    sd[unstable] = np.exp(np.clip(L[unstable] @ omega + .5*sigma_gvf**2, -20., 5.))
    observation = np.asarray(data['model_variance_s1'], dtype=float) + sd**2
    X = np.column_stack([np.ones(len(sampled)), Z[sampled]])
    weights = 1. / (observation + .5**2)
    ridge = np.diag(np.r_[.01, np.repeat(.25, Z.shape[1])])
    coef = np.linalg.solve(X.T @ (weights[:, None]*X) + ridge,
                           X.T @ (weights*theta))
    error = theta - X @ coef
    residual_var = float(np.median(error**2) - np.median(observation))
    sigma_area = float(np.clip(np.sqrt(max(residual_var, .3**2)), .3, 2.))
    rho = 3.05 / (3.05+1.65)
    logit_rho = np.log(rho/(1-rho))
    rng = np.random.default_rng(seed + 770101)
    values = []
    for _ in range(chains):
        values.append({
            'intercept': float(coef[0] + rng.normal(0., .1)),
            'beta': coef[1:] + rng.normal(0., .1, len(coef)-1),
            'sigma_area': float(sigma_area*np.exp(rng.normal(0., .3))),
            'rho': float(1./(1.+np.exp(-(logit_rho+rng.normal(0., .3))))),
            'spatial_z': rng.normal(0., .1, int(data['R'])),
            'omega': omega + rng.normal(0., omega_jitter, len(omega)),
            'sigma_gvf': float(sigma_gvf*np.exp(rng.normal(0., .2))),
        })
    if not all(all(np.isfinite(v).all() for v in chain.values())
               and 0 < chain['rho'] < 1 and chain['sigma_area'] > 0
               and chain['sigma_gvf'] > 0 for chain in values):
        raise FloatingPointError('Invalid numerical starting point; preserve the original target and sampler')
    report = {'policy': 'Numerical initial points only; unchanged target and priors.',
        'initialization_seed': int(seed + 770101), 'chains': int(chains),
        'CAR_modes_initialized': int(data['R']), 'stable_areas': len(stable),
        'centering_rules': 'GVF least squares on stable log SD; ridge/GLS linking start; CAR coordinates near zero with independent small perturbations.',
        'initial_sigma_area_center': sigma_area,
        'initial_sigma_gvf_center': sigma_gvf, 'initial_rho_center': rho,
        'bounds_scope': 'The numerical clips bound only starting values and approximate initial variance weights. They never enter the Stan likelihood, priors, support or posterior computation.'}
    return values, report


def make_initial_metric(data, report):
    """Positive diagonal kinetic metric in Stan declaration order.

    Uses a local curvature approximation solely for initial NUTS scaling.
    Warmup continues to adapt its metric normally. The approximation never
    enters any density, prior or posterior summary.
    """
    sampled = np.asarray(data['sampled_area'], dtype=int) - 1
    stable = np.asarray(data['stable'], dtype=int) - 1
    unstable = np.asarray(data['unstable'], dtype=int) - 1
    L = np.asarray(data['L'], dtype=float)
    Z = np.asarray(data['Z'], dtype=float)[sampled]
    F = np.asarray(data['scaled_icar_factor'], dtype=float)[sampled]
    sd = np.asarray(data['sampling_sd_s1'], dtype=float).copy()
    sigma = report['initial_sigma_area_center']
    rho = report['initial_rho_center']
    sigma_gvf = report['initial_sigma_gvf_center']
    omega = np.linalg.lstsq(L[stable], np.log(sd[stable]), rcond=None)[0]
    sd[unstable] = np.exp(np.clip(L[unstable]@omega+.5*sigma_gvf**2, -20., 5.))
    iid = sigma**2*(1-rho)
    V = np.asarray(data['model_variance_s1'], dtype=float) + sd**2 + iid
    weight = 1./V
    # Order: intercept, beta[P], log(sigma), logit(rho), spatial_z[R],
    # omega[2], log(sigma_gvf), exactly as in the unchanged Stan declaration.
    curvature_intercept = 4./3. + weight.sum()
    curvature_beta = .25 + (weight[:, None]*Z**2).sum(axis=0)
    curvature_logsigma = sigma**2/2. + 2.*np.sum((iid/V)**2)
    curvature_logitrho = (3.05+1.65)*rho*(1-rho) + .5*np.sum((sigma**2*rho*(1-rho)/V)**2)
    curvature_spatial = 1. + sigma**2*rho*(weight[:, None]*F**2).sum(axis=0)
    curvature_omega = .25 + np.sum(L[stable]**2,axis=0)/sigma_gvf**2
    curvature_loggvf = .1 + 2.*len(stable)
    curvature = np.r_[curvature_intercept, curvature_beta,
        curvature_logsigma, curvature_logitrho, curvature_spatial,
        curvature_omega, curvature_loggvf]
    metric = np.clip(1./np.maximum(curvature, 1e-12), 1e-8, 1e2)
    if len(metric) != int(data['R'])+int(data['P'])+6 or not np.isfinite(metric).all() or (metric<=0).any():
        raise FloatingPointError('Invalid initial kinetic metric; preserve the original sampler')
    return metric


def install_initializer():
    """Install only on the unchanged whitened spatial Stan model."""
    original_sample = h._sample
    def numerical_sample(model, data, cfg, output, seed_offset=0):
        if Path(model.stan_file).stem != 'hogg_stage2_bym2':
            return original_sample(model, data, cfg, output, seed_offset)
        output.mkdir(parents=True, exist_ok=True)
        actual_seed = cfg.seed + seed_offset
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                values, report = make_initial_values(data, chains=cfg.chains, seed=actual_seed)
                metric = make_initial_metric(data, report)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
            # The original prior-only GVF target permits zero stable rows.
            # Numerical convenience must never reduce its support or turn a
            # valid original fit into a new initializer execution failure.
            h.write_json(output/'numerical_initialization_summary.json', {
                'fallback_to_original_initialization': True,
                'reason_type': type(error).__name__, 'reason': str(error),
                'target_and_priors_unchanged': True, 'config': h.asdict(cfg),
                'Stan_source_sha256': hashlib.sha256(Path(model.stan_file).read_bytes()).hexdigest(),
                'adapter_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
            return original_sample(model, data, cfg, output, seed_offset)
        serial = h._jsonable(values)
        report['inits_sha256'] = hashlib.sha256(json.dumps(serial, sort_keys=True).encode()).hexdigest()
        report['Stan_source_sha256'] = hashlib.sha256(Path(model.stan_file).read_bytes()).hexdigest()
        report['adapter_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report['initial_inverse_metric_sha256'] = hashlib.sha256(metric.tobytes()).hexdigest()
        report['initial_inverse_metric_dimension'] = len(metric)
        report['initial_inverse_metric_range'] = [float(metric.min()),float(metric.max())]
        report['metric_policy'] = 'Fixed positive initial diagonal curvature scaling only; all coordinates retained and normal warmup metric adaptation remains engaged. Unchanged Stan target, data and priors.'
        report['config'] = h.asdict(cfg)
        h.write_json(output/'numerical_initialization_summary.json', report)
        # Exact same data, executable and sampling arguments as h._sample,
        # with only explicit starting points and initial kinetic scaling added.
        return model.sample(data=h._jsonable(data), chains=cfg.chains,
            parallel_chains=cfg.parallel_chains, iter_warmup=cfg.warmup,
            iter_sampling=cfg.samples, seed=actual_seed, inits=serial,
            metric='diag_e', inv_metric=metric,
            adapt_delta=cfg.adapt_delta, max_treedepth=cfg.max_treedepth,
            show_progress=False, refresh=max(1000,cfg.warmup+cfg.samples),
            output_dir=str(output))
    h._sample = numerical_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver', choices=['pilot','scout','application','batch'], required=True)
    args, remainder = parser.parse_known_args()
    frozen_path = h.ROOT/'resultados/competidores_hogg/hogg_spatial_freeze.json'
    frozen = json.loads(frozen_path.read_text())
    if not frozen.get('approved_after_numerical_pilot'):
        raise ValueError('Original spatial target must already have passed its independent pilot')
    for name, expected in frozen['source_sha256'].items():
        if hashlib.sha256((h.HERE/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Original frozen spatial source changed: '+name)
    # Production calls require a separately audited numerical amendment. The
    # development pilot/scout cannot authorize themselves as final fits.
    if args.driver in ['application','batch']:
        amendment = h.ROOT/'resultados/competidores_hogg/hogg_spatial_numerical_amendment.json'
        approved = json.loads(amendment.read_text())
        if not approved.get('approved_after_independent_audit_and_pilot'):
            raise ValueError('Numerical amendment has not passed its audit and pilot')
        if approved['adapter_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError('Numerical adapter differs from the audited amendment')
    install_initializer()
    modules = {'pilot':'hogg_spatial_validate', 'scout':'hogg_spatial_scout',
               'application':'hogg_spatial_application', 'batch':'hogg_spatial_batch'}
    sys.argv = [modules[args.driver]] + remainder
    importlib.import_module(modules[args.driver]).main()


if __name__ == '__main__':
    main()
