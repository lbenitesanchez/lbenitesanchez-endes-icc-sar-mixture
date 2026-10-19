"""Second-stage same-model recovery of unresolved strict REML fits.

This module leaves the frozen stage-one source and outputs untouched. Every
call first uses stage one exactly. Only a final failure invokes additional
local profile-score root bracketing and a denser deterministic rho profile.
Both original and error-contrast KKT scores keep tolerance 1e-5. No evaluation
outcome or predictive-performance argument exists. Bootstrap algebra below is
copied verbatim from stage one except for the called fitter, solver version,
and the additional stage-two diagnostic field.
"""
from __future__ import annotations
import collections
import copy
import hashlib
import sys
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq, minimize_scalar
from scipy.sparse.linalg import splu
import sikov_model as sm
import sikov_bootstrap as sb
import stable_reml_solver as base

VERSION = 'same_model_strict_REML_20261018_stage2_v1'
BASE_SOLVER_SHA256 = 'cfd591cdfa2797dbb0b8b20f4a6fe2af7f21c52c9c83b782ded2df43efa52b6c'
CERTIFICATE_TOLERANCE = base.CERTIFICATE_TOLERANCE
RHO_BOUND = base.RHO_BOUND
_COUNTERS = base._COUNTERS
_STAGE2_COUNTERS = collections.Counter()
_safe_certificate = base._safe_certificate
_INSTALLED = False
_PREVIOUS = None


def reset_counters():
    base.reset_counters()
    _STAGE2_COUNTERS.clear()


def get_counters():
    return {**base.get_counters(), 'solver_version': VERSION, **dict(_STAGE2_COUNTERS)}


def _stage2_rescue(Xall, y, D, s, W, spatial, original):
    problem = base.ContrastProfile(Xall, y, D, s, W, spatial)
    baseline = (problem.evaluate(original['variance']/problem.scale, original['rho'])
                if original is not None else float('inf'))
    candidates = []

    def add(rho, source):
        rho = float(np.clip(rho, -RHO_BOUND, RHO_BOUND)) if spatial else 0.
        z, loss = problem.profile(rho)
        if z == 0.:
            rho = 0.
        certificate = base._safe_certificate(Xall, y, D, s, W, spatial, z*problem.scale, rho)
        stable = base._stable_certificate(problem, z*problem.scale, rho, spatial)
        item = {'z': z, 'rho': rho, 'loss': loss, 'certificate': certificate,
                'stable_certificate': stable, 'source': source}
        candidates.append(item)
        return item

    # Recreate stage one's fitting-only candidate, never its evaluation score.
    try:
        candidate, _, _, _ = base._rescue(Xall, y, D, s, W, spatial, original, problem)
        add(candidate['rho'], 'stage1_best_profile_candidate')
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        pass
    if original is not None:
        add(original['rho'], 'original_rho_variance_profile')
    if not spatial:
        add(0., 'FH_variance_profile')
    else:
        # Local derivative brackets fix a scalar objective stop on a very flat
        # ridge. Wide neighbour endpoints can have A=0 and score=0; therefore
        # shrinking local brackets, rather than only coarse grid endpoints,
        # are essential to distinguish the small interior minimum.
        centers = [x['rho'] for x in sorted(candidates,key=lambda x:x['loss'])[:3]]
        if not centers:
            centers = [0.]
        widths = np.geomspace(1e-10, .3, 36)
        for center in centers:
            for width in widths:
                lo = max(-RHO_BOUND, center-float(width))
                hi = min(RHO_BOUND, center+float(width))
                if not lo < hi:
                    continue
                def score(rho):
                    z = problem.profile(float(rho))[0]
                    return problem.evaluate(z, float(rho), True)[1][1]
                left, right = score(lo), score(hi)
                if left < 0 < right:
                    try:
                        root = brentq(score, lo, hi, xtol=5e-15, rtol=1e-14, maxiter=200)
                        add(root, 'local_profile_rho_score_root')
                    except (ValueError, RuntimeError, np.linalg.LinAlgError):
                        pass
        # Include boundaries explicitly and search a denser independent rho
        # profile. This is only reached for the few stage-one unresolved fits.
        grid = np.unique(np.r_[np.linspace(-RHO_BOUND,RHO_BOUND,201), centers,
                               RHO_BOUND-np.geomspace(1e-10,.15,50),
                               -RHO_BOUND+np.geomspace(1e-10,.15,50)])
        values = [problem.profile(float(r)) for r in grid]
        add(-RHO_BOUND, 'dense_profile_lower_boundary')
        add(RHO_BOUND, 'dense_profile_upper_boundary')
        add(float(grid[np.argmin([v[1] for v in values])]), 'dense_profile_grid_minimum')
        for k in range(1,len(grid)-1):
            if values[k][1] < values[k-1][1] and values[k][1] < values[k+1][1]:
                lo, hi = float(grid[k-1]), float(grid[k+1])
                opt = minimize_scalar(lambda rho:problem.profile(float(rho))[1],
                                      bounds=(lo,hi), method='bounded',
                                      options={'xatol':1e-14,'maxiter':300})
                local = add(float(opt.x), 'dense_profile_refined_local_minimum')
                def score(rho):
                    z = problem.profile(float(rho))[0]
                    return problem.evaluate(z,float(rho),True)[1][1]
                # Search locally around each refined minimum even when the
                # complete coarse bracket terminates on a zero-variance plateau.
                for width in np.geomspace(1e-10,max(hi-lo,1e-9),25):
                    a=max(lo,local['rho']-float(width));b=min(hi,local['rho']+float(width))
                    if a < b and score(a) < 0 < score(b):
                        try:
                            root=brentq(score,a,b,xtol=5e-15,rtol=1e-14,maxiter=200)
                            add(root,'dense_local_profile_rho_score_root')
                        except (ValueError,RuntimeError,np.linalg.LinAlgError):
                            pass
    if not candidates:
        return None, problem, baseline, 0
    best_all = min(candidates,key=lambda x:x['loss'])
    valid = [x for x in candidates if x['certificate']['passed']
             and x['stable_certificate']['passed'] and x['loss'] <= baseline+1e-8]
    if valid:
        best = min(valid,key=lambda x:x['loss'])
        if best['loss'] <= best_all['loss']+1e-8:
            return best, problem, baseline, len(candidates)
    return None, problem, baseline, len(candidates)


def corrected_fit_stratum(Xall, y, D, sample_idx, W, spatial=True, tolerance=1e-9):
    tick=time.perf_counter();before=base.get_counters()
    _STAGE2_COUNTERS['stage2_checked_fit_calls'] += 1
    answer=None;stage1_exception=None
    try:
        answer=base.corrected_fit_stratum(Xall,y,D,sample_idx,W,spatial,tolerance)
    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
        stage1_exception=exc
    after_stage1=base.get_counters()
    if answer is not None and answer['converged']:
        return answer
    _STAGE2_COUNTERS['stage2_rescue_attempts'] += 1
    Xall=np.asarray(Xall,float);y=np.asarray(y,float);D=np.asarray(D,float)
    sample_idx=np.asarray(sample_idx,int);W=np.asarray(W,float)
    if answer is not None:
        meta=copy.deepcopy(answer['numerical_correction'])
        original={'variance':meta['original_variance'],'rho':meta['original_rho']} if meta['original_variance'] is not None else None
        previous_original=copy.deepcopy(answer['stationarity_certificate'])
        previous_stable=copy.deepcopy(answer['stable_stationarity_certificate'])
    else:
        # A stage-one numerical exception is not converted into a fictitious
        # historical parameter vector or a native-success flag.
        try:
            original=base._ORIGINAL_FIT(Xall,y,D,sample_idx,W,spatial,tolerance)
        except (ValueError,RuntimeError,np.linalg.LinAlgError):
            original=None
        if original is None:
            previous_original=base._unavailable_certificate(stage1_exception)
            previous_stable=copy.deepcopy(previous_original)
            meta={'solver_version':base.VERSION,'original_raw_optimizer_converged':None,
                  'original_stationarity_certificate':copy.deepcopy(previous_original),
                  'original_stable_stationarity_certificate':copy.deepcopy(previous_stable),
                  'original_variance':None,'original_rho':None,'original_message':None,
                  'original_fit_exception_type':type(stage1_exception).__name__,
                  'original_fit_exception':str(stage1_exception),'repaired':False}
        else:
            prob=base.ContrastProfile(Xall,y,D,sample_idx,W,spatial)
            previous_original=base._safe_certificate(Xall,y,D,sample_idx,W,spatial,original['variance'],original['rho'])
            previous_stable=base._stable_certificate(prob,original['variance'],original['rho'],spatial)
            meta={'solver_version':base.VERSION,'original_raw_optimizer_converged':bool(original['converged']),
                  'original_stationarity_certificate':copy.deepcopy(previous_original),
                  'original_stable_stationarity_certificate':copy.deepcopy(previous_stable),
                  'original_variance':float(original['variance']),'original_rho':float(original['rho']),
                  'original_message':original['message'],'original_fit_exception_type':None,
                  'original_fit_exception':None,'repaired':False}
    candidate,problem,baseline,count=_stage2_rescue(Xall,y,D,sample_idx,W,spatial,original)
    stage2={'solver_version':VERSION,'stage1_solver_version':base.VERSION,
            'stage1_final_original_certificate':previous_original,
            'stage1_final_stable_certificate':previous_stable,
            'stage1_exception':str(stage1_exception) if stage1_exception is not None else None,
            'candidate_count':count,'recovered':candidate is not None}
    if candidate is not None:
        old_answer=answer
        answer=base._predict_at(Xall,y,D,sample_idx,W,spatial,
                                candidate['z']*problem.scale,candidate['rho'])
        answer['raw_optimizer_converged']=meta['original_raw_optimizer_converged']
        answer['stationarity_certificate']=candidate['certificate']
        answer['stable_stationarity_certificate']=candidate['stable_certificate']
        answer['converged']=True
        stage2.update(rescue_source=candidate['source'],stable_original_nll=baseline if np.isfinite(baseline) else None,
                      stable_repaired_nll=candidate['loss'],
                      stable_nll_change=candidate['loss']-baseline if np.isfinite(baseline) else None)
        meta.update(repaired=True,rescue_source='stage2_'+candidate['source'],
                    stable_original_nll=stage2['stable_original_nll'],
                    stable_repaired_nll=stage2['stable_repaired_nll'],
                    stable_nll_change=stage2['stable_nll_change'],
                    stable_projected_gradient_max=candidate['stable_certificate']['projected_gradient_max'])
        _STAGE2_COUNTERS['stage2_rescue_passed'] += 1
        # Overall counters refer to the final stage-two outcome. The previous
        # failure is retained explicitly in stage2 metadata and separate counts.
        base._COUNTERS['rescue_passed'] += 1
        failures_added=after_stage1.get('rescue_failed',0)-before.get('rescue_failed',0)
        if failures_added:
            base._COUNTERS['rescue_failed'] -= failures_added
        if after_stage1.get('rescue_attempts',0)==before.get('rescue_attempts',0):
            base._COUNTERS['rescue_attempts'] += 1
        for key in ('final_certificate_failed','final_stable_certificate_failed'):
            increment=after_stage1.get(key,0)-before.get(key,0)
            if increment:base._COUNTERS[key] -= increment
        if old_answer is not None:
            base._COUNTERS['final_variance_boundary'] -= int(old_answer['variance']==0.)
            base._COUNTERS['final_rho_boundary'] -= int(spatial and abs(old_answer['rho'])>=RHO_BOUND-1e-8)
        base._COUNTERS['final_variance_boundary'] += int(answer['variance']==0.)
        base._COUNTERS['final_rho_boundary'] += int(spatial and abs(answer['rho'])>=RHO_BOUND-1e-8)
    else:
        _STAGE2_COUNTERS['stage2_rescue_failed'] += 1
        if answer is None:
            raise RuntimeError('Stage-one exception and strict stage-two profile recovery failed: '+str(stage1_exception))
    meta['stage2_numerical_correction']=stage2
    answer['numerical_correction']=meta
    base_elapsed=after_stage1.get('elapsed_seconds',0)-before.get('elapsed_seconds',0)
    base._COUNTERS['elapsed_seconds'] += time.perf_counter()-tick-base_elapsed
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
                if meta.get('stage2_numerical_correction') is not None:
                    row['stage2_numerical_correction'] = meta['stage2_numerical_correction']
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
    def __enter__(self):return self
    def __exit__(self,*exc):uninstall()


def install():
    global _INSTALLED,_PREVIOUS
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()!=BASE_SOLVER_SHA256:
        raise RuntimeError('Frozen stage-one solver digest mismatch')
    if not _INSTALLED:
        published=sys.modules.get('sikov_published_reference')
        _PREVIOUS=(sm._fit_stratum,sb._fit_stratum,sb.fit_sikov_bootstrap,
                   published.fit_sikov_bootstrap if published is not None else None)
        sm._fit_stratum=corrected_fit_stratum
        sb._fit_stratum=corrected_fit_stratum
        sb.fit_sikov_bootstrap=corrected_bootstrap
        if published is not None:published.fit_sikov_bootstrap=corrected_bootstrap
        _INSTALLED=True
    return _Installation()


def uninstall():
    global _INSTALLED,_PREVIOUS
    if _INSTALLED:
        sm._fit_stratum,sb._fit_stratum,sb.fit_sikov_bootstrap,previous_published=_PREVIOUS
        published=sys.modules.get('sikov_published_reference')
        if published is not None:
            published.fit_sikov_bootstrap=(previous_published if previous_published is not None else sb.fit_sikov_bootstrap)
        _INSTALLED=False;_PREVIOUS=None
