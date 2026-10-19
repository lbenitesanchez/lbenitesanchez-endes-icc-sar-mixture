"""Strict, same-likelihood numerical correction for the frozen FH/SAR fitter.

The historical fitter is retained for every call. Its result is reused exactly
when the original projected-score certificate passes. Otherwise a deterministic
error-contrast variance/rho profile supplies a same-model recovery. Prediction,
plugin/delta MSE and the caller's bootstrap innovations are unchanged in form.
No population truth or accuracy score enters this module.

``install()`` patches both historical aliases and returns a context manager.
It is process-local and intended for sequential fits within each worker process.
``get_counters()`` returns aggregate diagnostics; individual metadata lives in
the returned fit's ``numerical_correction`` and ``stationarity_certificate``.
"""
from __future__ import annotations

import copy
import time
from collections import Counter
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve, eigh, null_space, solve_triangular
from scipy.optimize import brentq, minimize_scalar
from scipy.sparse.linalg import splu

import sikov_model as sm
import sikov_bootstrap as sb

VERSION = 'same_model_strict_REML_20261018_v2'
CERTIFICATE_TOLERANCE = 1e-5
RHO_BOUND = sm.RHO_BOUND
_ORIGINAL_FIT = sm._fit_stratum
_ORIGINAL_BOOTSTRAP = sb.fit_sikov_bootstrap
_ORIGINAL_CERTIFICATE = sb._certificate
_COUNTERS = Counter()
_INSTALLED = False


def reset_counters():
    _COUNTERS.clear()


def get_counters():
    return {'solver_version': VERSION, **dict(_COUNTERS)}


class ContrastProfile:
    """Exactly equivalent REML through orthonormal residual contrasts."""
    def __init__(self, Xall, y, D, s, W, spatial):
        self.s = np.asarray(s, int)
        self.y = np.asarray(y, float)
        self.scale = max(float(np.var(y)), float(np.mean(D)), 1e-5)
        self.spatial = bool(spatial)
        X = np.asarray(Xall)[self.s]
        self.Q = null_space(X.T)
        self.e = self.Q.T @ y
        self.Dp = self.Q.T @ (np.asarray(D)[:, None] * self.Q)
        self.LD = np.linalg.cholesky(self.Dp)
        self.white = solve_triangular(self.LD, self.e, lower=True, check_finite=False)
        self.constant = (np.linalg.slogdet(X.T @ X)[1]
                         + 2 * np.log(np.diag(self.LD)).sum()
                         + len(self.e) * np.log(2 * np.pi))
        self.W = sparse.csc_matrix(W)
        self.I = sparse.eye(len(W), format='csc')
        self.EQ = np.eye(len(W))[:, self.s] @ self.Q
        self.cache = {}
        self.variance_grid = np.r_[0., np.geomspace(1e-13, 100., 112)]

    def rho_data(self, rho):
        rho = float(rho) if self.spatial else 0.
        if rho not in self.cache:
            if self.spatial:
                fac = splu(self.I - rho * self.W)
                L = fac.solve(self.EQ, trans='T')
                dL = fac.solve(self.W.T @ L, trans='T')
                C = L.T @ L
                dC = dL.T @ L + L.T @ dL
            else:
                C = self.Q.T @ self.Q
                dC = np.zeros_like(C)
            temp = solve_triangular(self.LD, C, lower=True, check_finite=False)
            whiteC = solve_triangular(self.LD, temp.T, lower=True, check_finite=False).T
            ev, U = eigh((whiteC + whiteC.T) / 2, check_finite=False)
            if ev.min() < -1e-7 * max(1., ev.max()):
                raise np.linalg.LinAlgError('Residual covariance is not nonnegative')
            ev = np.maximum(ev, 0.)
            self.cache[rho] = (C, dC, ev, U.T @ self.white)
        return self.cache[rho]

    def evaluate(self, z, rho, gradient=False):
        C, dC, ev, t = self.rho_data(rho)
        den = 1. + float(z) * self.scale * ev
        loss = .5 * (self.constant + np.log1p(float(z) * self.scale * ev).sum()
                     + np.sum(t * t / den))
        if not gradient:
            return float(loss)
        ga = .5 * np.sum(self.scale * ev / den - t * t * self.scale * ev / den**2)
        gr = 0.
        if self.spatial:
            A = float(z) * self.scale
            fac = cho_factor(A * C + self.Dp, lower=True, check_finite=False)
            vi = cho_solve(fac, np.eye(len(self.e)), check_finite=False)
            ve = cho_solve(fac, self.e, check_finite=False)
            gr = .5 * (np.sum(vi * (A * dC).T) - ve @ (A * dC) @ ve)
        return float(loss), np.array([float(ga), float(gr)])

    def profile(self, rho):
        """Choose the best variance boundary or bracketed score root."""
        _, _, ev, t = self.rho_data(rho)
        den = 1. + self.variance_grid[:, None] * self.scale * ev
        g = .5 * np.sum(self.scale * ev / den - t * t * self.scale * ev / den**2, axis=1)
        zvalues = [0., 100.]
        for k in np.flatnonzero(g[:-1] * g[1:] < 0):
            a, b = self.variance_grid[k:k+2]
            zvalues.append(brentq(lambda z: self.evaluate(z, rho, True)[1][0],
                                 a, b, xtol=1e-16, rtol=1e-14, maxiter=100))
        zvalues.extend(self.variance_grid[np.flatnonzero(g == 0)].tolist())
        z = min(zvalues, key=lambda zz: self.evaluate(zz, rho))
        return float(z), self.evaluate(z, rho)


def _projected_gradient(z, rho, gradient, spatial):
    g = np.asarray(gradient, float).copy()
    if z <= 1e-8 and g[0] > 0:
        g[0] = 0.
    if z >= 100.-1e-8 and g[0] < 0:
        g[0] = 0.
    if not spatial:
        g[1] = 0.
    elif (rho <= -RHO_BOUND+1e-8 and g[1] > 0) or (rho >= RHO_BOUND-1e-8 and g[1] < 0):
        g[1] = 0.
    return float(np.max(np.abs(g)))


def _predict_at(Xall, y, D, s, W, spatial, A, rho):
    """Historical prediction and MSE algebra, evaluated at fixed parameters."""
    X = Xall[s]
    model = sm._SAR(W, s, spatial)
    C, dC, Call, diagC, dCall = model.covariance(rho, full=True)
    V = A*C + np.diag(D)
    fac = cho_factor(V, lower=True, check_finite=False)
    ViX = cho_solve(fac, X, check_finite=False)
    Viy = cho_solve(fac, y, check_finite=False)
    beta_factor = cho_factor(X.T @ ViX, lower=True, check_finite=False)
    beta = cho_solve(beta_factor, X.T @ Viy, check_finite=False)
    Q = cho_solve(beta_factor, np.eye(X.shape[1]), check_finite=False)
    residual = y-X@beta
    vir = cho_solve(fac, residual, check_finite=False)
    loss = .5*(2*np.log(np.diag(fac[0])).sum() + 2*np.log(np.diag(beta_factor[0])).sum()
               + residual@vir + (len(y)-X.shape[1])*np.log(2*np.pi))
    Gs = A*Call
    ViGs = cho_solve(fac, Gs.T, check_finite=False)
    K = ViGs.T
    mean = Xall@beta + K@residual
    g1 = A*diagC-np.sum(Gs*ViGs.T, axis=1)
    H = Xall-K@X
    g2 = np.sum((H@Q)*H, axis=1)
    plugin = np.maximum(g1+g2, 0)
    Vi = cho_solve(fac, np.eye(len(y)), check_finite=False)
    P = Vi-ViX@Q@ViX.T
    dVs = [C] + ([A*dC] if spatial and A > 1e-12 else [])
    dGs = [Call] + ([A*dCall] if spatial and A > 1e-12 else [])
    info = np.array([[.5*np.trace(P@a@P@b) for b in dVs] for a in dVs])
    covpar = np.linalg.pinv(info, rcond=1e-10)
    dK = [(dg-K@dv)@Vi for dg, dv in zip(dGs, dVs)]
    g3 = sum(covpar[i, j]*np.sum((a@V)*b, axis=1)
             for i, a in enumerate(dK) for j, b in enumerate(dK))
    mse_delta = np.maximum(plugin+2*np.maximum(g3, 0), 0)
    return {'mean': mean, 'mse_plugin': plugin, 'mse_delta': mse_delta,
            'variance': float(A), 'rho': float(rho), 'beta': beta, 'beta_cov': Q,
            'converged': False, 'message': 'Fixed same-objective residual-contrast profile; certificate determines acceptance',
            'reml_nll': float(loss), 'param_info_condition': float(np.linalg.cond(info)),
            '_model': model, '_sample_idx': np.asarray(s, int)}


def _rescue(Xall, y, D, s, W, spatial, original, problem=None):
    problem = problem or ContrastProfile(Xall, y, D, s, W, spatial)
    z0 = float(original['variance']) / problem.scale if original is not None else .5
    r0 = float(original['rho']) if spatial and original is not None else 0.
    baseline = problem.evaluate(z0, r0) if original is not None else float('inf')
    candidates = []

    def add(rho, source):
        z, value = problem.profile(float(rho))
        # A=0 leaves rho unidentified. Keep the historical exact FH boundary.
        if z == 0.:
            rho = 0.
        stable_gradient = problem.evaluate(z, rho, True)[1]
        stable_norm = _projected_gradient(z, rho, stable_gradient, spatial)
        cert = _safe_certificate(Xall, y, D, s, W, spatial, z*problem.scale, rho)
        entry = {'z': z, 'rho': float(rho), 'loss': value, 'certificate': cert,
                 'source': source, 'stable_projected_gradient_max': stable_norm}
        candidates.append(entry)
        return entry

    # Most flagged fits have a tiny variance-score error at the valid rho bound.
    quick = add(r0, 'fixed_original_rho_variance_profile')
    if (original is not None and quick['certificate']['passed'] and quick['loss'] <= baseline+1e-8
            and quick['stable_projected_gradient_max'] < CERTIFICATE_TOLERANCE):
        return quick, problem, baseline, len(candidates)
    if not spatial:
        return min(candidates, key=lambda x: x['loss']), problem, baseline, len(candidates)

    grid = np.unique(np.r_[np.linspace(-RHO_BOUND, RHO_BOUND, 25), r0,
                           RHO_BOUND-np.geomspace(1e-7, .1, 12),
                           -RHO_BOUND+np.geomspace(1e-7, .1, 12)])
    profiles = [problem.profile(float(r)) for r in grid]
    selected = {0, len(grid)-1, int(np.argmin([v[1] for v in profiles]))}
    for k in range(1, len(grid)-1):
        if profiles[k][1] < profiles[k-1][1] and profiles[k][1] < profiles[k+1][1]:
            selected.add(k)
            lo, hi = float(grid[k-1]), float(grid[k+1])
            opt = minimize_scalar(lambda r: problem.profile(float(r))[1], bounds=(lo, hi),
                                  method='bounded', options={'xatol': 1e-13, 'maxiter': 200})
            add(float(opt.x), 'refined_rho_profile')
            def score(r):
                z = problem.profile(float(r))[0]
                return problem.evaluate(z, r, True)[1][1]
            # Gradient-root polishing avoids an objective-tolerance stop on a ridge.
            gl, gh = score(lo), score(hi)
            if gl < 0 < gh:
                try:
                    rr = brentq(score, lo, hi, xtol=2e-14, rtol=1e-14, maxiter=150)
                    add(float(rr), 'profile_rho_score_root')
                except (ValueError, RuntimeError):
                    pass
    for k in sorted(selected):
        add(float(grid[k]), 'rho_grid_or_boundary')
    admissible = [x for x in candidates if x['certificate']['passed']
                  and x['stable_projected_gradient_max'] < CERTIFICATE_TOLERANCE
                  and x['loss'] <= baseline+1e-8]
    # Do not silently accept an inferior certified local minimum across a real gap.
    best_all = min(candidates, key=lambda x: x['loss'])
    if admissible:
        best = min(admissible, key=lambda x: x['loss'])
        if best['loss'] <= best_all['loss']+1e-8:
            return best, problem, baseline, len(candidates)
    return best_all, problem, baseline, len(candidates)


def _unavailable_certificate(exception):
    return {'projected_gradient_max': None, 'passed': False,
            'tolerance': CERTIFICATE_TOLERANCE, 'rho_bound': RHO_BOUND,
            'at_rho_boundary': None, 'gradient_scaled': None,
            'available': False, 'exception_type': type(exception).__name__,
            'exception': str(exception)}


def _safe_certificate(*args):
    try:
        cert = _ORIGINAL_CERTIFICATE(*args)
        if not np.isfinite(cert['projected_gradient_max']) or not np.isfinite(cert['gradient_scaled']).all():
            raise ValueError('Non-finite original certificate; no stationarity claim')
        return cert
    except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
        return _unavailable_certificate(exc)


def _stable_certificate(problem, A, rho, spatial):
    z = float(A)/problem.scale
    _, gradient = problem.evaluate(z, rho, True)
    norm = _projected_gradient(z, rho, gradient, spatial)
    return {'projected_gradient_max': norm, 'passed': bool(norm < CERTIFICATE_TOLERANCE),
            'tolerance': CERTIFICATE_TOLERANCE, 'rho_bound': RHO_BOUND,
            'at_rho_boundary': bool(spatial and abs(rho) >= RHO_BOUND-1e-8),
            'gradient_scaled': gradient[:2 if spatial else 1].tolist(),
            'objective': 'exact_same_REML_orthonormal_error_contrasts'}


def corrected_fit_stratum(Xall, y, D, sample_idx, W, spatial=True, tolerance=1e-9):
    started = time.perf_counter()
    _COUNTERS['fit_calls'] += 1
    _COUNTERS['spatial_fit_calls' if spatial else 'FH_fit_calls'] += 1
    # Invalid data/design are input errors, not grounds to invent a rescue.
    Xall = np.asarray(Xall, float); y = np.asarray(y, float); D = np.asarray(D, float)
    sample_idx = np.asarray(sample_idx, int); W = np.asarray(W, float)
    X = Xall[sample_idx]
    if len(y) <= X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError('stratum lacks sampled degrees of freedom or full-rank X')
    if np.any(D <= 0) or not np.isfinite(D).all():
        raise ValueError('sampling variances must be strictly positive; no silent GVF')
    problem = ContrastProfile(Xall, y, D, sample_idx, W, spatial)
    original = None; exception = None
    try:
        original = _ORIGINAL_FIT(Xall, y, D, sample_idx, W, spatial, tolerance)
    except (np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
        exception = exc
        _COUNTERS['original_fit_numerical_exception'] += 1
    native = bool(original['converged']) if original is not None else None
    if original is not None:
        cert = _safe_certificate(Xall, y, D, sample_idx, W, spatial,
                                 original['variance'], original['rho'])
        stable_cert = _stable_certificate(problem, original['variance'], original['rho'], spatial)
    else:
        cert = _unavailable_certificate(exception)
        stable_cert = _unavailable_certificate(exception)
    metadata = {'solver_version': VERSION, 'original_raw_optimizer_converged': native,
                'original_stationarity_certificate': copy.deepcopy(cert),
                'original_stable_stationarity_certificate': copy.deepcopy(stable_cert),
                'original_variance': float(original['variance']) if original is not None else None,
                'original_rho': float(original['rho']) if original is not None else None,
                'original_message': original['message'] if original is not None else None,
                'original_fit_exception_type': type(exception).__name__ if exception is not None else None,
                'original_fit_exception': str(exception) if exception is not None else None,
                'repaired': False}
    if cert['passed'] and stable_cert['passed']:
        _COUNTERS['original_certificate_passed'] += 1
        _COUNTERS['original_native_failed_both_certificates_passed'] += int(native is False)
        answer = original
    else:
        _COUNTERS['rescue_attempts'] += 1
        _COUNTERS['raw_success_certificate_failed'] += int(native is True and not cert['passed'])
        _COUNTERS['original_certificate_passed_stable_failed'] += int(cert['passed'] and not stable_cert['passed'])
        candidate, problem, baseline, n_candidates = _rescue(
            Xall, y, D, sample_idx, W, spatial, original, problem)
        cert = candidate['certificate']
        stable_cert = _stable_certificate(problem, candidate['z']*problem.scale, candidate['rho'], spatial)
        metadata.update(rescue_source=candidate['source'], rescue_candidates=n_candidates,
                        stable_original_nll=baseline if np.isfinite(baseline) else None,
                        stable_repaired_nll=candidate['loss'],
                        stable_nll_change=candidate['loss']-baseline if np.isfinite(baseline) else None,
                        stable_projected_gradient_max=candidate['stable_projected_gradient_max'])
        if (cert['passed'] and stable_cert['passed'] and candidate['loss'] <= baseline+1e-8):
            answer = _predict_at(Xall, y, D, sample_idx, W, spatial,
                                 candidate['z']*problem.scale, candidate['rho'])
            metadata['repaired'] = True
            _COUNTERS['rescue_passed'] += 1
        elif original is not None:
            answer = original
            cert = metadata['original_stationarity_certificate']
            stable_cert = metadata['original_stable_stationarity_certificate']
            _COUNTERS['rescue_failed'] += 1
        else:
            _COUNTERS['rescue_failed'] += 1
            raise RuntimeError('Historical numerical exception and strict same-objective rescue failed: '+str(exception))
    answer['raw_optimizer_converged'] = native
    answer['stationarity_certificate'] = cert
    answer['stable_stationarity_certificate'] = stable_cert
    answer['converged'] = bool(cert['passed'] and stable_cert['passed'])
    answer['numerical_correction'] = metadata
    _COUNTERS['final_certificate_failed'] += int(not cert['passed'])
    _COUNTERS['final_stable_certificate_failed'] += int(not stable_cert['passed'])
    _COUNTERS['final_variance_boundary'] += int(answer['variance'] == 0.)
    _COUNTERS['final_rho_boundary'] += int(spatial and abs(answer['rho']) >= RHO_BOUND-1e-8)
    _COUNTERS['elapsed_seconds'] += time.perf_counter()-started
    return answer


def corrected_bootstrap(X, y, D, sample_idx, W, strata=None, *, spatial=True,
                        outcome=None, covariate_names=None, B=100, seed=20261010,
                        interval_estimator='naive', innovations=None,
                        bootstrap_min_success=.95):
    """Historical B-draw/bootstrap algebra with strict refit acceptance.

    Every standard-normal draw is generated in the historical order, even if
    its refit fails. Individual numerical diagnostics contain no responses or
    population accuracy. No native success flag can bypass the certificate.
    """
    if B < 2:
        raise ValueError('bootstrap needs >=2 draws; final plan requires B>=100')
    if interval_estimator not in ('naive', 'bc'):
        raise ValueError('unknown estimator')
    before = get_counters()
    X = np.asarray(X, float); y = np.asarray(y, float); D = np.asarray(D, float)
    sample_idx = np.asarray(sample_idx, int); W = np.asarray(W, float); n = len(X)
    strata = np.ones(n, int) if strata is None else np.asarray(strata)
    if len(y) == n:
        y = y[sample_idx]
    if len(D) == n:
        D = D[sample_idx]
    initial = sm.fit_sikov(X, y, D, sample_idx, W, strata, spatial=spatial,
                          outcome=outcome, covariate_names=covariate_names,
                          uncertainty='delta')
    naive = np.full(n, np.nan); bc = np.full(n, np.nan)
    nse = np.full(n, np.nan); bcse = np.full(n, np.nan)
    rng = np.random.default_rng(seed); records = []
    for sf in initial['fit']:
        level = sf['stratum']; ids = np.flatnonzero(strata == level)
        obs = np.isin(sample_idx, ids); s = np.searchsorted(ids, sample_idx[obs])
        yy = y[obs]; DD = D[obs]
        cols = np.arange(X.shape[1]) if outcome is None else sm.select_columns(outcome, level, covariate_names, spatial)
        XX = X[ids][:, cols]; WW = W[np.ix_(ids, ids)].copy()
        sums = WW.sum(axis=1)
        WW = np.divide(WW, sums[:, None], out=np.zeros_like(WW), where=sums[:, None] > 0)
        A = sf['variance']; rho = sf['rho']; beta = np.asarray(sf['beta'])
        model = sm._SAR(WW, s, spatial)
        C, dC, Call, diagC, dCall = model.covariance(rho, full=True)
        V = A*C+np.diag(DD)
        fac = cho_factor(V, lower=True, check_finite=False)
        ViX = cho_solve(fac, XX[s], check_finite=False)
        Q = np.linalg.inv(XX[s].T @ ViX)
        beta_map = Q @ ViX.T
        K = A*Call @ cho_solve(fac, np.eye(len(s)), check_finite=False)
        oracle_map = (XX-K@XX[s]) @ beta_map+K
        factor = splu(sparse.eye(len(ids), format='csc')-rho*sparse.csc_matrix(WW)) if spatial else None
        sf['stationarity_certificate'] = _safe_certificate(XX, yy, DD, s, WW, spatial, A, rho)
        sf['converged'] = bool(sf['stationarity_certificate']['passed'] and sf['stable_stationarity_certificate']['passed'])
        if innovations is None:
            U = rng.normal(size=(B, len(ids))); E = rng.normal(size=(B, len(s)))
        else:
            ii = innovations[level]; U = np.asarray(ii['u']); E = np.asarray(ii['e'])
            if U.shape != (B, len(ids)) or E.shape != (B, len(s)):
                raise ValueError('bad innovation arrays')
        sq = []; bc_terms = []; failed = 0; recovered = 0; draw_diagnostics = []
        rescued = 0; original_certificate_failed = 0; original_raw_success_cert_failed = 0
        for b in range(B):
            innovation = np.sqrt(A)*U[b]
            latent = factor.solve(innovation) if spatial else innovation
            truth = XX@beta+latent
            boot_y = truth[s]+np.sqrt(DD)*E[b]
            row = {'draw_index_zero_based': b}
            try:
                boot = corrected_fit_stratum(XX, boot_y, DD, s, WW, spatial)
                meta = boot['numerical_correction']; old_cert = meta['original_stationarity_certificate']
                final_cert = boot['stationarity_certificate']
                valid = bool(boot['converged']) and np.isfinite(boot['mean']).all()
                recovered += int(valid and boot['raw_optimizer_converged'] is False)
                rescued += int(meta['repaired'])
                original_certificate_failed += int(not old_cert['passed'])
                original_raw_success_cert_failed += int(boot['raw_optimizer_converged'] is True and not old_cert['passed'])
                row.update(raw_optimizer_converged=boot['raw_optimizer_converged'],
                           original_certificate=old_cert, final_certificate=final_cert,
                           original_stable_certificate=meta['original_stable_stationarity_certificate'],
                           final_stable_certificate=boot['stable_stationarity_certificate'],
                           original_fit_exception=meta['original_fit_exception'],
                           repaired=meta['repaired'], variance=boot['variance'], rho=boot['rho'],
                           original_variance=meta['original_variance'], original_rho=meta['original_rho'],
                           stable_nll_change=meta.get('stable_nll_change'),
                           stable_projected_gradient_max=meta.get('stable_projected_gradient_max'),
                           rescue_source=meta.get('rescue_source'), retained=bool(valid))
                if valid:
                    sq.append((boot['mean']-truth)**2)
                    oracle = oracle_map@boot_y
                    bc_terms.append(-boot['mse_plugin']+(boot['mean']-oracle)**2)
                else:
                    failed += 1
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
                failed += 1
                _COUNTERS['bootstrap_refit_exceptions'] += 1
                row.update(retained=False, exception_type=type(exc).__name__, exception=str(exc))
            draw_diagnostics.append(row)
        successes = len(sq)
        if successes < bootstrap_min_success*B:
            raise RuntimeError(f'Only {successes}/{B} strictly certified bootstrap fits in stratum {level}')
        sq = np.asarray(sq); bc_terms = np.asarray(bc_terms)
        naive[ids] = sq.mean(axis=0)
        bc[ids] = 2*initial['mse_plugin'][ids]+bc_terms.mean(axis=0)
        nse[ids] = sq.std(axis=0, ddof=1)/np.sqrt(successes)
        bcse[ids] = bc_terms.std(axis=0, ddof=1)/np.sqrt(successes)
        sf.update(bootstrap_B=B, bootstrap_success=successes, bootstrap_failed=failed,
                  bootstrap_stationarity_recovered=recovered,
                  bootstrap_numerical_rescued=rescued,
                  bootstrap_original_certificate_failed=original_certificate_failed,
                  bootstrap_original_raw_success_certificate_failed=original_raw_success_cert_failed,
                  bootstrap_strict_retained_certificates_passed=successes,
                  bootstrap_draw_diagnostics=draw_diagnostics)
        records.append(sf)
    selected = naive if interval_estimator == 'naive' else bc
    se = np.sqrt(np.maximum(selected, 0)); z = 1.959963984540054
    after = get_counters()
    return {**initial, 'mse': selected, 'mse_naive': naive, 'mse_bc': bc,
            'naive_mcse': nse, 'bc_mcse': bcse,
            'lower': initial['mean']-z*se, 'upper': initial['mean']+z*se,
            'fit': records, 'converged': all(r['converged'] for r in records),
            'uncertainty': 'parametric_bootstrap_'+interval_estimator,
            'bootstrap_B': B, 'negative_bc_estimates': int((bc < 0).sum()),
            'BC_formula_reference': 'CRAN sae 1.3 pbmseSFH.R; Molina Salvati Pratesi (2009)',
            'bootstrap_partial_fit_policy': 'retain only original-KKT-certified refits if original success fraction is met; all draw diagnostics recorded',
            'numerical_correction': {
                'solver_version': VERSION,
                'acceptance': 'unchanged_original_KKT_certificate_required',
                'counters': {k: after.get(k, 0)-before.get(k, 0) for k in after if k != 'solver_version'},
                'innovations': 'same RNG, seed, U/E shapes and generation order as historical bootstrap',
                'bootstrap_acceptance': 'every retained refit passes original KKT; original B and minimum-success fraction retained'}}


class _Installation:
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        uninstall()


def install():
    global _INSTALLED
    if not _INSTALLED:
        sm._fit_stratum = corrected_fit_stratum
        sb._fit_stratum = corrected_fit_stratum
        sb.fit_sikov_bootstrap = corrected_bootstrap
        _INSTALLED = True
    return _Installation()


def uninstall():
    global _INSTALLED
    sm._fit_stratum = _ORIGINAL_FIT
    sb._fit_stratum = _ORIGINAL_FIT
    sb.fit_sikov_bootstrap = _ORIGINAL_BOOTSTRAP
    _INSTALLED = False
