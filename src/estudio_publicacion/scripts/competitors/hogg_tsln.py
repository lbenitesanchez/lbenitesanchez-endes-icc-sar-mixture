"""Hogg et al. (2024) generic two-stage logistic-normal comparator.

Both stages use NUTS. Stage 1 weighted Bernoulli terms with identical predictors
are summed exactly. Stage 2's Gaussian latent variables are integrated exactly,
then reconstructed conditionally. S1's weighted posterior uses NUTS; S2 retains
the published Gaussian measurement-summary model. No EB, Laplace approximation
to S1, or substitution of ENDES Taylor D for Hogg's sampling variance is used.
See hogg_design.md for the explicit author-code/paper-text fidelity profiles.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CMDSTAN = ROOT / '.competitor_env/cmdstan/cmdstan-2.40.0'
DEFAULT_FEATURES = ['altitude', 'poverty', 'illiteracy', 'electricity', 'spanish', 'biomass']
LOADED_PYTHON_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass
class MCMCConfig:
    chains: int = 4
    parallel_chains: int = 4
    warmup: int = 500
    samples: int = 1000
    adapt_delta: float = .95
    max_treedepth: int = 12
    seed: int = 20261005
    min_ess: int = 400
    model_rhat_limit: float = 1.01
    phase: str = 'validation'


@dataclass
class HoggResult:
    areas: np.ndarray
    draws: np.ndarray  # posterior draw x all area
    stage1: dict
    stage2_data: dict
    status: dict
    output_dir: Path

    @property
    def mean(self):
        return self.draws.mean(axis=0)

    @property
    def variance(self):
        return self.draws.var(axis=0, ddof=1)

    def predictions(self):
        lo, hi = np.quantile(self.draws, [.025, .975], axis=0)
        hdi_lo, hdi_hi = highest_density_interval(self.draws, .95)
        sampled = set(self.stage1['sampled_area_codes'])
        return pd.DataFrame({'area': self.areas, 'mean': self.mean,
                             'median': np.median(self.draws, axis=0),
                             'variance': self.variance, 'lo95_ET': lo, 'hi95_ET': hi,
                             'lo95_HDI': hdi_lo, 'hi95_HDI': hdi_hi,
                             'sampled': [x in sampled for x in self.areas]})


def highest_density_interval(draws, probability=.95):
    """Shortest contiguous empirical posterior interval, also report ET bands."""
    x = np.sort(np.asarray(draws), axis=0)
    span = int(np.floor(probability * len(x)))
    widths = x[span:] - x[:-span]
    idx = widths.argmin(axis=0)
    cols = np.arange(x.shape[1])
    return x[idx, cols], x[idx + span, cols]


def _jsonable(x):
    if isinstance(x, dict): return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return _jsonable(x.tolist())
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x) if np.isfinite(x) else None
    if isinstance(x, np.bool_): return bool(x)
    if isinstance(x, Path): return str(x)
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_jsonable(obj), ensure_ascii=False, indent=2,
                                    allow_nan=False))


def _read_frame(x, area_col):
    if isinstance(x, (str, Path)):
        return pd.read_csv(x, dtype={area_col: str})
    return x.copy()


def _input_hash(x):
    """Record provenance without copying private unit records into results."""
    if isinstance(x, (str, Path)):
        return hashlib.sha256(Path(x).read_bytes()).hexdigest()
    frame = x.copy()
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()


def prepare_hogg_data(units, areas, *, outcome='y', weight='weight', area_col='area',
                      area_features=None, unit_features=(), population_N=None):
    """Prepare endpoint-specific units and all-area auxiliary features.

    ``population_N`` is None (explicit negligible-fraction assumption), an area
    column name, a dict of area population sizes, or a length-M known vector.
    We never interpret arbitrarily normalized survey weights as population N.
    Outcome missingness is preserved until this endpoint is selected.
    """
    area_features = DEFAULT_FEATURES if area_features is None else list(area_features)
    units = _read_frame(units, area_col)
    areas = _read_frame(areas, area_col)
    units[area_col] = units[area_col].astype(str)
    areas[area_col] = areas[area_col].astype(str)
    if areas[area_col].duplicated().any(): raise ValueError('Duplicated auxiliary area')
    Zraw = areas[area_features].to_numpy(float)
    if not np.isfinite(Zraw).all(): raise ValueError('All prediction areas require finite auxiliaries')
    scale = Zraw.std(axis=0)
    active = scale > 1e-12
    if not active.any(): raise ValueError('At least one nonconstant area covariate is required')
    area_features = [x for x, keep in zip(area_features, active) if keep]
    centre = Zraw.mean(axis=0)[active]
    scale = scale[active]
    Z = (Zraw[:, active] - centre) / scale
    all_codes = areas[area_col].to_numpy(str)
    lookup = {code: i for i, code in enumerate(all_codes)}
    endpoint_valid = (units[outcome].isin([0, 1]) & np.isfinite(units[weight])
                      & (units[weight] > 0))
    dropped_missing = int((~endpoint_valid).sum())
    unmatched = endpoint_valid & ~units[area_col].isin(lookup)
    excluded_unmatched = int(unmatched.sum())
    u = units.loc[endpoint_valid & ~unmatched].copy()
    if len(u) == 0: raise ValueError('No observed endpoint in matched areas')
    u['_area_index'] = u[area_col].map(lookup).astype(int)
    # Stable ordering permits independent replication of each area aggregation.
    u = u.sort_values('_area_index', kind='stable').reset_index(drop=True)
    sampled_idx = np.sort(u['_area_index'].unique())
    sampled_map = {idx: i for i, idx in enumerate(sampled_idx)}
    unit_area = u['_area_index'].map(sampled_map).to_numpy(int)
    y = u[outcome].to_numpy(float)
    w = u[weight].to_numpy(float)
    X = Z[u['_area_index'].to_numpy(int)]
    x_names = list(area_features)
    unit_transform = {}
    for feature in unit_features:
        v = u[feature].to_numpy(float)
        if not np.isfinite(v).all(): raise ValueError('Missing survey covariate: ' + feature)
        # Fixed transformations use covariates only, with their units declared.
        sd = float(v.std())
        if sd <= 1e-12: continue
        mu = float(v.mean())
        X = np.column_stack([X, (v - mu) / sd])
        x_names.append(feature)
        unit_transform[feature] = {'centre': mu, 'scale': sd}
    # Author's S1 design is centred within the observed units. The all-area
    # standardization above makes weakly informative priors coherent in ENDES.
    unit_centre = X.mean(axis=0)
    X = X - unit_centre
    if np.linalg.matrix_rank(X) != X.shape[1]:
        raise ValueError('Rank-deficient stage-1 design; do not select columns using outcomes')
    n = np.bincount(unit_area, minlength=len(sampled_idx)).astype(int)
    sum_w = np.bincount(unit_area, weights=w, minlength=len(sampled_idx))
    sum_w2 = np.bincount(unit_area, weights=w*w, minlength=len(sampled_idx))
    neff = sum_w*sum_w/sum_w2
    direct = np.bincount(unit_area, weights=w * y, minlength=len(sampled_idx)) / sum_w
    normalized_area_weights = w * n[unit_area] / sum_w[unit_area]
    normalized_global_weights = w * len(u) / w.sum()
    if population_N is None:
        Ni = np.full(len(all_codes), np.inf)
        fpc_note = 'Negligible sampling fraction assumed; contemporary area population N unavailable.'
    elif isinstance(population_N, str):
        Ni = areas[population_N].to_numpy(float)
        fpc_note = 'Known area population sizes supplied as auxiliary totals.'
    elif isinstance(population_N, dict):
        Ni = np.array([population_N[code] for code in all_codes], float)
        fpc_note = 'Known area population sizes supplied as auxiliary totals.'
    else:
        Ni = np.asarray(population_N, float)
        fpc_note = 'Known area population sizes supplied as auxiliary totals.'
    if len(Ni) != len(all_codes) or (Ni[sampled_idx] <= n).any():
        raise ValueError('Population totals must exceed measured sample counts')
    fpc = 1 - n / Ni[sampled_idx]
    # Collapse only literally identical predictor rows and the same area.
    keys = np.column_stack([unit_area, X])
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    grouped_success = np.bincount(inverse, weights=normalized_global_weights * y)
    grouped_failure = np.bincount(inverse, weights=normalized_global_weights * (1 - y))
    s1_data = {'G': len(unique), 'm': len(sampled_idx), 'P': X.shape[1],
               'X': unique[:, 1:], 'area': unique[:, 0].astype(int) + 1,
               'weight_success': grouped_success, 'weight_failure': grouped_failure}
    return {'units': u, 'all_codes': all_codes, 'sampled_idx': sampled_idx,
            'sampled_codes': all_codes[sampled_idx], 'unit_area': unit_area,
            'y': y, 'w': w, 'area_weights': normalized_area_weights, 'X': X, 'Z': Z,
            'n': n, 'n_eff_kish':neff, 'direct': direct, 'fpc': fpc, 's1_data': s1_data,
            'metadata': {'outcome': outcome, 'weight': weight, 'unit_records_measured': len(u),
                         'excluded_missing_endpoint_or_weight': dropped_missing,
                         'excluded_unmatched_measured': excluded_unmatched,
                         'total_areas': len(all_codes), 'sampled_areas': len(sampled_idx),
                         'nonconstant_area_features': area_features, 's1_feature_names': x_names,
                         'area_covariate_centres': centre, 'area_covariate_scales': scale,
                         's1_observed_unit_covariate_centres': unit_centre,
                         'unit_feature_transform': unit_transform,
                         'fpc_assumption': fpc_note,
                         'clustering': 'PSU/stratum retained in input but generic published TSLN does not adjust this iid variance formula for clustering.',
                         'likelihood_groups': len(unique), 'likelihood_grouping': 'exact weighted-Bernoulli sufficient statistics'}}


def aggregate_stage1(prepared, intercept, beta, effect_area):
    """Hogg equations 4-8, draw by draw, with no boundary continuity adjustment."""
    X, ua = prepared['X'], prepared['unit_area']
    y, wa = prepared['y'], prepared['area_weights']
    n, direct = prepared['n'], prepared['direct']
    S, m = len(intercept), len(n)
    theta = np.empty((S, m)); gamma = np.full((S, m), np.nan)
    p_mean = np.empty((S, m)); psi = np.full((S, m), np.nan)
    vdirect = np.full(m, np.nan)
    # Chunk posterior computations to keep N=17k memory bounded.
    groups = [np.flatnonzero(ua == i) for i in range(m)]
    for i, rows in enumerate(groups):
        eta = intercept[:, None] + beta @ X[rows].T + effect_area[:, i, None]
        p = expit(eta)
        mu = p @ wa[rows] / n[i]
        if (mu <= 0).any() or (mu >= 1).any():
            raise FloatingPointError('Numerical logit saturation; do not silently clip S1')
        p_mean[:, i] = mu
        theta[:, i] = logit(mu)
        if n[i] == 1: continue  # Always binary-direct unstable; imputed in S2.
        factor = prepared['fpc'][i] / (n[i] * (n[i] - 1))
        data = np.sum(wa[rows] ** 2 * (y[rows] - direct[i]) ** 2)
        bias = (p - y[rows]) @ wa[rows] / n[i]
        bias_element = np.sum(wa[rows] ** 2 * (p - y[rows] - bias[:, None]) ** 2, axis=1)
        vdirect[i] = factor * data
        psi[:, i] = factor * (data + bias_element)
        gamma[:, i] = psi[:, i] / (mu * (1 - mu)) ** 2
    stable = (direct > 0) & (direct < 1)
    if not stable.any(): raise ValueError('No stable area available to fit the published GVF')
    if not np.isfinite(gamma[:, stable]).all() or (gamma[:, stable] <= 0).any():
        raise FloatingPointError('Invalid stable S1 sampling variance')
    return {'theta_draws': theta, 'gamma_draws': gamma, 'mu_draws': p_mean,
            'psi_draws': psi, 'direct_variance': vdirect, 'stable': stable,
            'single_measurement': n == 1,
            'sampled_area_codes': prepared['sampled_codes']}


def prepare_stage2(prepared, s1, *, profile='author_code', seed=20261006):
    """Author-code SD summary + QR prior, or clearly labelled paper-text profile."""
    if profile not in ['author_code', 'paper_text']: raise ValueError('Unknown fidelity profile')
    theta, gamma = s1['theta_draws'], s1['gamma_draws']
    selected = np.sort(np.random.default_rng(seed).choice(len(theta), len(theta) // 2, replace=False))
    chosen = theta[selected]
    model_var = theta.var(axis=0, ddof=1)
    if (model_var <= 0).any(): raise FloatingPointError('Zero S1 posterior uncertainty')
    sd = np.zeros(theta.shape[1])
    stable = s1['stable']
    if profile == 'author_code':
        sd[stable] = np.sqrt(gamma[:, stable]).mean(axis=0)
        Q, R = np.linalg.qr(prepared['Z'], mode='reduced')
        # Fix QR signs for deterministic reproduction across LAPACK backends.
        signs = np.where(np.diag(R) < 0, -1., 1.)
        Q = Q * signs
        R = signs[:, None] * R
        Zmodel = Q * np.sqrt(len(Q) - 1)
    else:
        sd[stable] = np.sqrt(gamma[:, stable].mean(axis=0))
        Zmodel = prepared['Z']
    data = {'M': len(prepared['all_codes']), 'm': theta.shape[1],
            'P': Zmodel.shape[1], 'Z': Zmodel,
            'sampled_area': prepared['sampled_idx'] + 1,
            'theta_s1': chosen.mean(axis=0), 'model_variance_s1': model_var,
            'sampling_sd_s1': sd, 'stable_n': int(stable.sum()),
            'stable': np.flatnonzero(stable) + 1, 'unstable_n': int((~stable).sum()),
            'unstable': np.flatnonzero(~stable) + 1,
            'L': np.column_stack([np.ones(theta.shape[1]), np.log(prepared['n'])])}
    return data, {'profile': profile, 's1_draws': len(theta), 'stage2_subset_size': len(selected),
                  'stage2_subset_indices': selected, 'theta_s1_selected_draws': chosen,
                  'sampling_sd_summary': 'mean(sqrt(gamma))' if profile == 'author_code' else 'sqrt(mean(gamma))',
                  'coefficient_prior_design': 'scaled thin QR basis, as author Stan code' if profile == 'author_code' else 'standardized original area covariates, as paper text',
                  'stage2_marginalization': 'exact Gaussian integration; mean of powered S1 likelihood retains full S1 posterior variance'}


def _backend():
    import cmdstanpy
    if not CMDSTAN.exists(): raise RuntimeError('Project CmdStan installation missing')
    # The installed toolchain cannot link SDK27's new TAPI architecture tags.
    # Select the compatible official SDK only for this Python process.
    compatible_sdk=Path('/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk')
    if compatible_sdk.exists(): os.environ.setdefault('SDKROOT',str(compatible_sdk))
    cmdstanpy.set_cmdstan_path(str(CMDSTAN))
    return cmdstanpy


def compile_models(include_reference=False):
    backend = _backend()
    names = ['hogg_stage1', 'hogg_stage2']
    if include_reference: names.append('hogg_stage2_full_reference')
    models={}
    for name in names:
        source=HERE/(name+'.stan')
        stamp=HERE/(name+'_compiled_source.json')
        before=hashlib.sha256(source.read_bytes()).hexdigest()
        cached=json.loads(stamp.read_text()) if stamp.exists() else {}
        model=backend.CmdStanModel(stan_file=str(source),
            force_compile=cached.get('stan_sha256')!=before)
        after=hashlib.sha256(source.read_bytes()).hexdigest()
        if before!=after:
            raise RuntimeError('Stan source changed during compilation; retry when source is stable: '+name)
        write_json(stamp,{'stan_sha256':after,
            'executable_sha256':hashlib.sha256(Path(model.exe_file).read_bytes()).hexdigest(),
            'cmdstan_version':'2.40.0','SDKROOT':os.environ.get('SDKROOT')})
        models[name]=model
    return models


def diagnose_fit(fit, cfg, summary=None):
    summary = fit.summary() if summary is None else summary
    # Generated predictions are stochastic posterior draws; inspect their
    # summaries separately from HMC parameter convergence.
    names = summary.index.to_series().astype(str)
    parameter_rows = ~names.str.startswith(('lp__', 'eta[', 'effect_area[', 'linking_mean[',
                                            'sampling_sd[', 'log_sampling_sd_mean[',
                                            'observation_variance[', 'theta[', 'prevalence['))
    ps = summary.loc[parameter_rows.values]
    # pandas reductions skip NaNs. Every sampled-parameter diagnostic must be
    # finite before taking extrema; an undefined tail ESS cannot pass the gate.
    diagnostic_values = ps[['R_hat', 'ESS_bulk', 'ESS_tail']].to_numpy(float)
    diagnostics_finite = bool(len(ps) > 0 and np.isfinite(diagnostic_values).all())
    rhat = float(ps.R_hat.max())
    bulk = float(ps.ESS_bulk.min())
    tail = float(ps.ESS_tail.min())
    mv = fit.method_variables()
    divergences = int(mv['divergent__'].sum())
    depth = int((mv['treedepth__'] >= cfg.max_treedepth).sum())
    energy = mv['energy__']
    bfmi = np.mean(np.diff(energy, axis=0) ** 2, axis=0) / np.var(energy, axis=0)
    success = (diagnostics_finite and rhat <= cfg.model_rhat_limit
               and bulk >= cfg.min_ess and tail >= cfg.min_ess
               and divergences == 0 and depth == 0
               and np.isfinite(bfmi).all() and np.min(bfmi) >= .3)
    return {'success': bool(success), 'max_parameter_Rhat': rhat,
            'all_parameter_diagnostics_finite': diagnostics_finite,
            'sampled_parameter_count': len(ps),
            'min_parameter_ESS_bulk': bulk, 'min_parameter_ESS_tail': tail,
            'divergences': divergences, 'max_treedepth_hits': depth, 'BFMI_per_chain': bfmi,
            'thresholds': {'Rhat': cfg.model_rhat_limit, 'ESS_bulk_and_tail': cfg.min_ess,
                           'BFMI': .3, 'divergence_or_depth_hits': 0},
            'draws': fit.num_draws_sampling * fit.chains}


def _sample(model, data, cfg, output, seed_offset=0):
    output.mkdir(parents=True, exist_ok=True)
    return model.sample(data=_jsonable(data), chains=cfg.chains,
                        parallel_chains=cfg.parallel_chains,
                        iter_warmup=cfg.warmup, iter_sampling=cfg.samples,
                        seed=cfg.seed + seed_offset, adapt_delta=cfg.adapt_delta,
                        max_treedepth=cfg.max_treedepth, show_progress=False,
                        refresh=max(1000, cfg.warmup + cfg.samples), output_dir=str(output))


def fit_hogg(units, areas, *, outcome='y', weight='weight', area_col='area',
             area_features=None, unit_features=(), population_N=None,
             profile='author_code', config=None, output_dir=None, models=None):
    """Reusable fit/predict API. Returns draws for every supplied known area."""
    cfg = MCMCConfig() if config is None else config
    output = ROOT / 'resultados/competidores_hogg/run' if output_dir is None else Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    p = prepare_hogg_data(units, areas, outcome=outcome, weight=weight, area_col=area_col,
                          area_features=area_features, unit_features=unit_features,
                          population_N=population_N)
    model = compile_models() if models is None else models
    start = time.monotonic()
    f1 = _sample(model['hogg_stage1'], p['s1_data'], cfg, output / 'stage1')
    summary1=f1.summary()
    d1 = diagnose_fit(f1, cfg, summary=summary1)
    summary1.to_csv(output / 'stage1_parameter_summary.csv')
    s1 = aggregate_stage1(p, f1.stan_variable('intercept'), f1.stan_variable('beta'),
                          f1.stan_variable('effect_area'))
    d2data, d2meta = prepare_stage2(p, s1, profile=profile, seed=cfg.seed + 21)
    stage1_time = time.monotonic() - start
    f2 = _sample(model['hogg_stage2'], d2data, cfg, output / 'stage2', seed_offset=1)
    summary2=f2.summary()
    d2 = diagnose_fit(f2, cfg, summary=summary2)
    summary2.to_csv(output / 'stage2_parameter_summary.csv')
    draws = f2.stan_variable('prevalence')
    if not np.isfinite(draws).all() or (draws <= 0).any() or (draws >= 1).any():
        raise FloatingPointError('Nonfinite or boundary posterior predictions')
    status = {'method': 'Hogg_TSLN_generic_' + profile,
              'completed': True, 'success': d1['success'] and d2['success'],
              'phase': cfg.phase, 'stage1': d1, 'stage2': d2,
              'seconds_stage1_plus_aggregation': stage1_time,
              'seconds_total': time.monotonic() - start,
              'config': asdict(cfg), 'input_metadata': p['metadata'],
              'input_sha256': {'units': _input_hash(units), 'areas': _input_hash(areas)},
              'model_metadata': {k: v for k, v in d2meta.items() if k != 'theta_s1_selected_draws'},
              'single_observation_domains': int(s1['single_measurement'].sum()),
              'stable_direct_domains': int(s1['stable'].sum()),
              'unstable_direct_domains_GVF': int((~s1['stable']).sum()),
              'single_observation_handling': 'Sampling variance undefined and not used; direct 0/1 domain receives jointly estimated GVF SD in S2.',
              'interpretation': 'Published generic nonspatial TSLN comparator, not Australian BYM2/benchmark case-study specification; posterior intervals require empirical coverage assessment.',
              'sampling_D': 'Hogg iid weighted variance formula, not ENDES first-stage Taylor covariance.',
              'uncertainty_summary_limit':'S2 uses the author modular diagonal Gaussian summary of S1, with model variance and sampling SD summaries; it is not a joint posterior integrating both-stage unit likelihoods or S1 cross-area covariance.',
              'loaded_python_source_sha256':LOADED_PYTHON_SOURCE_SHA256,
              'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [Path(__file__), HERE/'hogg_stage1.stan', HERE/'hogg_stage2.stan']}}
    result = HoggResult(p['all_codes'], draws, s1, d2data, status, output)
    result.predictions().to_csv(output / 'area_predictions.csv', index=False)
    np.savez_compressed(output / 'posterior_area_draws.npz', area=p['all_codes'], prevalence=draws)
    pd.DataFrame({'area': p['sampled_codes'], 'n_measured': p['n'],
                  'n_eff_kish':p['n_eff_kish'],
                  'direct': p['direct'], 'direct_variance_Hogg': s1['direct_variance'],
                  'single_observation_variance_undefined': s1['single_measurement'],
                  'direct_stable': s1['stable'], 'mu_s1_mean': s1['mu_draws'].mean(axis=0),
                  'theta_s1_mean_all_draws': s1['theta_draws'].mean(axis=0),
                  'theta_s1_posterior_sd': s1['theta_draws'].std(axis=0, ddof=1),
                  'sampling_sd_s1_used_stable': d2data['sampling_sd_s1']}).to_csv(output / 'stage1_area_summary.csv', index=False)
    write_json(output / 'status.json', status)
    return result


def fit_hogg_pair(units, areas, *, area_col='area', area_features=None,
                  population_N=None, unit_features=(), profile='author_code',
                  config=None, output_dir=None, models=None,
                  outcomes=('y_anemia', 'y_stunting'),
                  weights=('weight_anemia', 'weight_stunting'),
                  retain_sampler_csv=True, retry_on_diagnostics=False):
    """Canonical two-outcome benchmark adapter; two independent TSLN models.

    Unit records with different measurement availability are filtered separately.
    Joint cross-outcome posterior dependence is not estimated by these fits.
    ``retain_sampler_csv=False`` keeps diagnostics, parameter summaries, stdout,
    all-area posterior draws and hashes while removing redundant sampler CSVs.
    This optional storage policy is intended for Monte Carlo batches only.
    """
    cfg = MCMCConfig() if config is None else config
    output = ROOT/'resultados/competidores_hogg/pair' if output_dir is None else Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    start=time.monotonic()
    model=compile_models() if models is None else models
    results={}
    attempts={}
    for j,label in enumerate(['anemia','stunting']):
        endpoint_config=MCMCConfig(**{**asdict(cfg), 'seed':cfg.seed+100*j})
        results[label]=fit_hogg(units,areas,outcome=outcomes[j],weight=weights[j],
            area_col=area_col,area_features=area_features,population_N=population_N,
            unit_features=unit_features,profile=profile,config=endpoint_config,
            output_dir=output/label/'attempt_0',models=model)
        attempts[label]=[results[label].status]
        if retry_on_diagnostics and not results[label].status['success']:
            retry=MCMCConfig(**{**asdict(endpoint_config), 'warmup':1000,
                               'samples':2000, 'adapt_delta':.99})
            results[label]=fit_hogg(units,areas,outcome=outcomes[j],weight=weights[j],
                area_col=area_col,area_features=area_features,population_N=population_N,
                unit_features=unit_features,profile=profile,config=retry,
                output_dir=output/label/'attempt_1',models=model)
            attempts[label].append(results[label].status)
    a,s=results['anemia'],results['stunting']
    if not np.array_equal(a.areas,s.areas): raise ValueError('Endpoint area order differs')
    predictions=pd.DataFrame({'area':a.areas})
    for label,result in results.items():
        p=result.predictions()
        predictions['p_mean_'+label]=p['mean']
        predictions['lo95_'+label]=p['lo95_ET']
        predictions['hi95_'+label]=p['hi95_ET']
        predictions['p_median_'+label]=p['median']
        predictions['lo95_HDI_'+label]=p['lo95_HDI']
        predictions['hi95_HDI_'+label]=p['hi95_HDI']
    predictions.to_csv(output/'predictions.csv',index=False)
    np.savez_compressed(output/'posterior_draws.npz',area=a.areas,
                        anemia=a.draws,stunting=s.draws)
    status={'method':'Hogg_TSLN_generic_'+profile,'completed':True,
            'success':all(x.status['success'] for x in results.values()),
            'seconds_total':time.monotonic()-start,
            'outcomes':{k:v.status for k,v in results.items()},
            'all_attempts':attempts,
            'retry_on_diagnostics':retry_on_diagnostics,
            'retry_rule':'A failing endpoint is refit once at 1000 warmup/2000 samples/adapt_delta=.99, same four chains, priors, source and seed; all attempts retained. No performance-based retries or discarded failures.',
            'outcome_dependence':'Two independent endpoint fits; no inferred paired change or cross-outcome posterior covariance.',
            'interval_primary':'Equal-tail posterior 95%; HDI also retained.',
            'retain_sampler_csv':retain_sampler_csv}
    # Diagnostic retries can produce different marginal draw counts. All
    # reported marginal summaries retain every draw; the optional common-size
    # array is independently subsampled/permuted only for an API shape contract.
    common_n=min(len(a.draws),len(s.draws))
    common=[]
    for j,result in enumerate([a,s]):
        rng=np.random.default_rng(cfg.seed+4058+j)
        common.append(result.draws[rng.choice(len(result.draws),common_n,replace=False)])
    status['full_marginal_draw_counts']={'anemia':len(a.draws),'stunting':len(s.draws)}
    status['common_draw_array']='Independent deterministic subsampling/permutation to common size; full marginal arrays and all-draw interval summaries retained separately. Not a paired-rule change posterior.'
    if not retain_sampler_csv:
        hashes={}
        for label in results:
            for attempt in range(len(attempts[label])):
              for stage in ['stage1','stage2']:
                for path in (output/label/('attempt_'+str(attempt))/stage).glob('*.csv'):
                    hashes[str(path.relative_to(output))]=hashlib.sha256(path.read_bytes()).hexdigest()
                    path.unlink()
        status['removed_redundant_sampler_csv_sha256']=hashes
    write_json(output/'status.json',status)
    return {'predictions':predictions,'results':results,'status':status,
            'draws':np.stack(common,axis=-1),
            'marginal_draws':{'anemia':a.draws,'stunting':s.draws}}


def validate_exact_reductions(prepared, *, seed=4321):
    """Independent algebraic checks at arbitrary parameter values."""
    rng = np.random.default_rng(seed)
    sd = prepared['s1_data']
    beta = rng.normal(size=sd['P']); effect = rng.normal(size=sd['m']); intercept = .31
    eta_u = intercept + prepared['X'] @ beta + effect[prepared['unit_area']]
    w = prepared['w'] * len(prepared['w']) / prepared['w'].sum()
    y = prepared['y']
    original = np.sum(w * (y * -np.logaddexp(0, -eta_u) + (1-y) * -np.logaddexp(0, eta_u)))
    eta_g = intercept + sd['X'] @ beta + effect[sd['area'] - 1]
    collapsed = np.sum(sd['weight_success'] * -np.logaddexp(0, -eta_g)
                       + sd['weight_failure'] * -np.logaddexp(0, eta_g))
    unit_score = w * (y - expit(eta_u))
    group_score = sd['weight_success'] - (sd['weight_success'] + sd['weight_failure']) * expit(eta_g)
    gradient_unit = np.concatenate([[unit_score.sum()], prepared['X'].T @ unit_score,
                                   np.bincount(prepared['unit_area'], weights=unit_score,
                                               minlength=sd['m'])])
    gradient_group = np.concatenate([[group_score.sum()], sd['X'].T @ group_score,
                                    np.bincount(sd['area'] - 1, weights=group_score,
                                                minlength=sd['m'])])
    gradient_difference = float(np.max(np.abs(gradient_unit - gradient_group)))
    draws = rng.normal(.4, .6, size=100)
    s2 = draws.var(ddof=1); mu = draws.mean()
    grid = np.linspace(-1., 1., 21)
    full = np.array([norm.logpdf(draws, v, np.sqrt(s2)).mean() for v in grid])
    reduced = norm.logpdf(mu, grid, np.sqrt(s2))
    difference = full - reduced
    return {'weighted_bernoulli_loglik_abs_difference': float(abs(original-collapsed)),
            'weighted_bernoulli_full_gradient_max_abs_difference': gradient_difference,
            'powered_draw_likelihood_parameter_dependent_difference_range': float(np.ptp(difference)),
            'powered_draw_variance': float(s2),
            'sampling_plus_measurement_gaussian_convolution': 'N(theta_s1_mean | linking_mean, model_var + sampling_sd^2 + sigma_area^2)',
            'success': bool(abs(original-collapsed) < 1e-8 and gradient_difference < 1e-8
                            and np.ptp(difference) < 1e-12)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', required=True)
    parser.add_argument('--areas', required=True)
    parser.add_argument('--outcome', required=True)
    parser.add_argument('--weight', default='weight')
    parser.add_argument('--area-col', default='area')
    parser.add_argument('--features', nargs='+', default=DEFAULT_FEATURES)
    parser.add_argument('--unit-features', nargs='*', default=[])
    parser.add_argument('--population-N-col')
    parser.add_argument('--profile', choices=['author_code','paper_text'], default='author_code')
    parser.add_argument('--output', required=True)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--min-ess', type=int, default=400)
    parser.add_argument('--seed', type=int, default=20261005)
    parser.add_argument('--phase', default='validation')
    args = parser.parse_args()
    cfg = MCMCConfig(chains=args.chains, parallel_chains=args.chains,
                     warmup=args.warmup, samples=args.samples, min_ess=args.min_ess,
                     seed=args.seed, phase=args.phase)
    result = fit_hogg(args.units, args.areas, outcome=args.outcome, weight=args.weight,
                      area_col=args.area_col, area_features=args.features,
                      unit_features=args.unit_features, population_N=args.population_N_col,
                      profile=args.profile, config=cfg, output_dir=args.output)
    print(json.dumps({'status_file': str(result.output_dir/'status.json'),
                      'success': result.status['success'], 'seconds': result.status['seconds_total']}))


if __name__ == '__main__':
    main()
