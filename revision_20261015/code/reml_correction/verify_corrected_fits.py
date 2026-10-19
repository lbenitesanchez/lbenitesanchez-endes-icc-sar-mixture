#!/usr/bin/env python3
"""Independent receipt/output checker for the corrected REML/bootstrap run.

Reads training-file hashes, numerical status records and model predictions.
Never opens evaluation outcomes or raw training arrays. Partial checks are
explicitly incomplete. A full PASS requires every registered job and every
retained draw to satisfy both unchanged-tolerance score certificates.
"""
from __future__ import annotations
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

TOLERANCE = 1e-5
RHO_BOUND = .999
KEYS = ('mean', 'lower', 'upper', 'mse', 'mse_plugin', 'mse_delta',
        'mse_naive', 'mse_bc', 'naive_mcse', 'bc_mcse')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def certificate_check(cert, name, required=False):
    require(isinstance(cert, dict), name+': missing certificate')
    require(cert.get('tolerance') == TOLERANCE, name+': changed tolerance')
    require(cert.get('rho_bound') == RHO_BOUND, name+': changed rho bound')
    passed = cert.get('passed')
    require(isinstance(passed, bool), name+': ambiguous pass flag')
    value = cert.get('projected_gradient_max')
    gradient = cert.get('gradient_scaled')
    if value is None:
        require(not passed and cert.get('available') is False, name+': missing value treated as pass')
        require(bool(cert.get('exception_type')), name+': unavailable score lacks cause')
    else:
        require(math.isfinite(value) and value >= 0, name+': invalid projected score')
        require(passed == (value < TOLERANCE), name+': pass disagrees with strict score threshold')
        require(isinstance(gradient, list) and len(gradient) in (1, 2), name+': invalid score vector')
        require(np.isfinite(gradient).all(), name+': nonfinite unprojected score')
        require(value <= max(abs(x) for x in gradient)+1e-10,
                name+': projected score larger than raw score')
    if required:
        require(passed, name+': retained fit does not pass')
    return passed


def update_group(group, raw, old, old_stable, final, final_stable, repaired, retained, variance, rho):
    group['fits'] += 1
    group['raw_success'] += int(raw is True)
    group['raw_failure'] += int(raw is False)
    group['raw_unavailable'] += int(raw is None)
    group['original_certificate_passed'] += int(old['passed'])
    group['original_stable_certificate_passed'] += int(old_stable['passed'])
    group['original_certificate_passed_stable_failed'] += int(old['passed'] and not old_stable['passed'])
    group['original_certificate_failed_stable_passed'] += int(not old['passed'] and old_stable['passed'])
    group['raw_success_original_certificate_failed'] += int(raw is True and not old['passed'])
    group['repaired'] += int(repaired)
    group['final_original_certificate_passed'] += int(final['passed'])
    group['final_stable_certificate_passed'] += int(final_stable['passed'])
    group['retained'] += int(retained)
    group['not_retained'] += int(not retained)
    group['variance_zero'] += int(variance == 0.)
    group['rho_boundary'] += int(rho is not None and abs(rho) >= RHO_BOUND-1e-8)
    for label, cert in (('original', old), ('original_stable', old_stable),
                        ('final_original', final), ('final_stable', final_stable)):
        value = cert.get('projected_gradient_max')
        if value is not None:
            key = label+'_max_projected_gradient'
            group[key] = max(group.get(key, 0.), value)


def validate_fit_metadata(fit, label, *, central):
    """Validate flags/certificates and unchanged or rescued parameter identity."""
    if central:
        meta = fit['numerical_correction']
        raw = fit['raw_optimizer_converged']
        require(raw == meta['original_raw_optimizer_converged'], label+': raw flag overwritten')
        old = meta['original_stationarity_certificate']
        old_stable = meta['original_stable_stationarity_certificate']
        final = fit['stationarity_certificate']
        final_stable = fit['stable_stationarity_certificate']
        retained = bool(fit['converged'])
        repaired = bool(meta['repaired'])
        A, rho = fit['variance'], fit['rho']
        old_A, old_rho = meta['original_variance'], meta['original_rho']
    else:
        meta = fit
        raw = fit.get('raw_optimizer_converged')
        old = fit['original_certificate']
        old_stable = fit['original_stable_certificate']
        final = fit['final_certificate']
        final_stable = fit['final_stable_certificate']
        retained = bool(fit['retained'])
        repaired = bool(fit['repaired'])
        A, rho = fit['variance'], fit['rho']
        old_A, old_rho = fit['original_variance'], fit['original_rho']
    for cert, name in ((old, 'old'), (old_stable, 'old_stable'),
                       (final, 'final'), (final_stable, 'final_stable')):
        certificate_check(cert, label+'/'+name, required=retained and name.startswith('final'))
    require(raw in (None, True, False), label+': invalid native status')
    if central:
        require(retained == bool(final['passed'] and final_stable['passed']), label+': central acceptance is not both certificates')
    else:
        require(not retained or (final['passed'] and final_stable['passed']), label+': retained draw lacks both certificates')
    require(math.isfinite(A) and A >= 0 and math.isfinite(rho) and abs(rho) <= RHO_BOUND+1e-14,
            label+': invalid final covariance parameters')
    if not repaired:
        require(A == old_A and rho == old_rho, label+': certified point changed without repair flag')
        if retained:
            require(old['passed'] and old_stable['passed'], label+': unrepaired retention without old certificates')
    else:
        require(final['passed'] and final_stable['passed'], label+': repaired point remains uncertified')
        change = meta.get('stable_nll_change')
        require(change is None or (math.isfinite(change) and change <= 1e-8), label+': rescue worsens objective materially')
    if raw is None:
        require(old_A is None and old_rho is None, label+': unavailable original status has fabricated parameters')
        require(bool(meta.get('original_fit_exception')), label+': unavailable original fit has no exception')
    return raw, old, old_stable, final, final_stable, repaired, retained, A, rho


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='Private corrected run root')
    ap.add_argument('--private-run-root', type=Path, required=True)
    ap.add_argument('--inventory', type=Path)
    ap.add_argument('--protocol', type=Path)
    ap.add_argument('--report', type=Path)
    ap.add_argument('--stage2', action='store_true', help='Validate only the registered stage-two selection')
    ap.add_argument('--stage2-output', type=Path, help='Authoritative stage-two overlay, leaving stage one intact')
    ap.add_argument('--job-selection', type=Path, help='Truth-blind fitting-failure selection for stage two')
    ap.add_argument('--stage2-protocol', type=Path)
    ap.add_argument('--dispatch-inventory', type=Path, action='append', default=[],
                    help='Operational subset inventory; every scientific job dict must equal its global registered entry')
    ap.add_argument('--allow-partial', action='store_true')
    ap.add_argument('--job-id', action='append', help='Explicit diagnostic subset; never full completion')
    args = ap.parse_args()
    repo = args.repo.resolve(); output = args.output.resolve(); private = args.private_run_root.resolve()
    inventory_path = args.inventory or output/'job_inventory.json'
    protocol_path = args.protocol or repo/'revision_20261015/protocols/reml_correction_20261018.json'
    report_path = args.report or repo/'revision_20261015/results/reml_correction/corrected_fits_verification.json'
    plan = read(inventory_path); protocol = read(protocol_path)
    inventory_hash = sha(inventory_path); protocol_hash = sha(protocol_path)
    source_dir = repo/'revision_20261015/code/reml_correction'
    runner_hash = sha(source_dir/'refit_cohorts.py'); solver_hash = sha(source_dir/'stable_reml_solver.py')
    stage2_runner_hash = sha(source_dir/'refit_cohorts_stage2_v2.py') if args.stage2 or args.stage2_output else None
    stage2_solver_hash = sha(source_dir/'stable_reml_solver_stage2_v2.py') if args.stage2 or args.stage2_output else None
    stage2_protocol_path = (args.stage2_protocol or repo/'revision_20261015/protocols/reml_correction_stage2_20261018.json')
    stage2_protocol = read(stage2_protocol_path) if args.stage2 or args.stage2_output else None
    stage2_protocol_hash = sha(stage2_protocol_path) if stage2_protocol is not None else None
    if stage2_protocol is not None:
        require(stage2_protocol['base_protocol_sha256'] == protocol_hash, 'Stage-two amendment does not bind original correction protocol')
        require(stage2_protocol['stage1_solver_sha256'] == solver_hash, 'Stage-two amendment does not bind frozen stage-one solver')
        require(stage2_protocol['tolerance'] == TOLERANCE, 'Stage-two amendment changes score tolerance')
        require(stage2_protocol['data_graph_design_bounds_weights_B_seeds_changed'] is False,
                'Stage-two amendment changes scientific choices')
        require(stage2_protocol['no_new_random_streams_or_rerolled_rejected_draws'] is True,
                'Stage-two amendment does not preserve random streams')
    selection_hash = None; stage2_ids = set()
    if args.stage2 or args.stage2_output:
        require(args.job_selection is not None, 'Stage-two verification requires its registered fitting-only selection')
        selection = read(args.job_selection); selection_hash = sha(args.job_selection)
        require(selection['registered_inventory_sha256'] == inventory_hash, 'Stage-two selection changed inventory')
        require(selection['truth_read'] is False, 'Stage-two selection used evaluation truth')
        stage2_ids = set(selection['selected_job_ids'])
        require(len(stage2_ids) == len(selection['selected_job_ids']), 'Duplicate stage-two selection IDs')
        require(stage2_ids <= {j['job_id'] for j in plan['jobs']}, 'Stage-two selection contains unknown jobs')
    for rel, value in plan['source_models_sha256'].items():
        require(sha(repo/rel) == value, 'Historical source changed: '+rel)
    for rel, value in protocol['original_sources_sha256'].items():
        require(sha(repo/rel) == value, 'Protocol historical source changed: '+rel)
    jobs = plan['jobs']
    require(len(jobs) == len({j['job_id'] for j in jobs}), 'Inventory duplicate job IDs')
    global_jobs={j['job_id']:j for j in jobs}
    allowed_inventories={inventory_hash:set(global_jobs)}
    dispatch_inventory_bindings=[]
    for path in args.dispatch_inventory:
        dispatch=read(path);digest=sha(path)
        binding=dispatch['operational_dispatch_selection']
        require(binding['registered_inventory_sha256']==inventory_hash, 'Dispatch subset lacks registered inventory binding')
        require(binding['scientific_job_fields_changed'] is False, 'Dispatch subset declares scientific changes')
        require(dispatch['truth_read'] is False and dispatch['source_models_sha256']==plan['source_models_sha256'],
                'Dispatch subset changes truth access or source models')
        entries=dispatch['jobs'];ids={j['job_id'] for j in entries}
        require(len(ids)==len(entries), 'Dispatch subset contains duplicate jobs')
        require(ids<=set(global_jobs), 'Dispatch subset contains unregistered jobs')
        require(all(j==global_jobs[j['job_id']]for j in entries), 'Dispatch scientific job definition differs from global inventory')
        allowed_inventories[digest]=ids
        dispatch_inventory_bindings.append({'file_label':path.name,'sha256':digest,'jobs':len(entries),
                                            'registered_global_inventory_sha256':inventory_hash,
                                            'every_job_dict_identical_to_global':True})
    if args.stage2:
        jobs = [j for j in jobs if j['job_id'] in stage2_ids]
    if args.job_id:
        require(set(args.job_id) <= {j['job_id'] for j in jobs}, 'Unknown selected job ID')
        jobs = [j for j in jobs if j['job_id'] in args.job_id]
    groups = collections.defaultdict(collections.Counter)
    errors = []; remaining = []; job_summaries = []; hashed = {}

    def check_hash(path, expected):
        path = Path(path)
        actual = hashed.setdefault(str(path), sha(path)) if str(path) not in hashed else hashed[str(path)]
        require(actual == expected, 'Digest mismatch: '+path.name)

    for index, job in enumerate(jobs):
        source_stage = 2 if (args.stage2 or (args.stage2_output and job['job_id'] in stage2_ids)) else 1
        job_output = args.stage2_output.resolve() if source_stage == 2 and args.stage2_output else output
        expected_runner = stage2_runner_hash if source_stage == 2 else runner_hash
        expected_solver = stage2_solver_hash if source_stage == 2 else solver_hash
        expected_protocol = stage2_protocol_hash if source_stage == 2 else protocol_hash
        registered_protocol = stage2_protocol if source_stage == 2 else protocol
        completion = job_output/job['output_relative_dir']/(job['method']+'_reml_correction_completion.json')
        if not completion.is_file():
            remaining.append(job['job_id']); continue
        try:
            record = read(completion)
            require(record['job_id'] == job['job_id'], 'Completion identifies another job')
            require(record['corrected'] and record['converged'], 'Completion is not corrected and converged')
            require(record['truth_read'] is False, 'Fit claims outcome access')
            require(record['same_graph_model_seeds_B'] is True, 'Model/seed/B preservation flag missing')
            recorded_inventory=record['job_inventory_sha256']
            require(recorded_inventory in allowed_inventories and job['job_id'] in allowed_inventories[recorded_inventory],
                    'Inventory digest mismatch or job outside its registered operational subset')
            require(record['source_sha256'] == expected_runner, 'Runner version mismatch')
            require(record['solver_sha256'] == expected_solver, 'Solver version mismatch')
            require(record['protocol_sha256'] == expected_protocol, 'Protocol version mismatch')
            if source_stage == 2:
                require(record['base_solver_sha256'] == solver_hash, 'Stage-two base solver version mismatch')
                require(record['job_selection_sha256'] == selection_hash, 'Stage-two fitting selection mismatch')
            require(datetime.datetime.fromisoformat(record['created_utc']) >= datetime.datetime.fromisoformat(registered_protocol['registered_utc']),
                    'Fit completion precedes its registered correction protocol')
            require(record['input_sha256'] == job['input_sha256'], 'Input binding mismatch')
            require(record['endpoint_seeds'] == job['endpoint_seeds'], 'Endpoint seed mismatch')
            require(record['bootstrap_B'] == job['bootstrap_B'], 'Bootstrap B mismatch')
            expected_sources={
                str((source_dir/('refit_cohorts_stage2_v2.py' if source_stage==2 else 'refit_cohorts.py')).relative_to(repo)):expected_runner,
                str((source_dir/('stable_reml_solver_stage2_v2.py' if source_stage==2 else 'stable_reml_solver.py')).relative_to(repo)):expected_solver,
            }
            if source_stage==2:
                expected_sources[str((source_dir/'stable_reml_solver.py').relative_to(repo))]=solver_hash
            require(record['fitting_sources_sha256']==expected_sources, 'Fitting source catalogue missing or altered')
            for rel, value in record['fitting_sources_sha256'].items():
                check_hash(repo/rel, value)
            check_hash(private/job['input_path'], job['input_sha256'])
            check_hash(private/job['original_prediction_path'], job['original_prediction_sha256'])
            check_hash(private/job['original_status_path'], job['original_status_sha256'])
            for rel, value in record['outputs_sha256'].items():
                check_hash(job_output/rel, value)
            folder = job_output/job['output_relative_dir']
            status = read(folder/job['status_filename'])
            endpoints = status['status' if job['development_status_schema'] else 'fits']
            require(len(endpoints) == 2, 'Expected exactly two endpoint results')
            arrays = dict(np.load(folder/job['prediction_filename'], allow_pickle=False))
            expected_keys = KEYS if job['bootstrap_B'] else KEYS[:6]
            require(set(arrays) == set(expected_keys), 'Unexpected prediction arrays')
            require(all(a.ndim == 2 and a.shape[1] == 2 and np.isfinite(a).all() for a in arrays.values()),
                    'Nonfinite or malformed prediction array')
            require(all(a.shape == arrays['mean'].shape for a in arrays.values()), 'Prediction shape mismatch')
            require(np.all(arrays['mse'] >= 0), 'Primary MSE is negative')
            require(np.all(arrays['lower'] <= arrays['upper']), 'Interval bounds reversed')
            require(np.allclose(arrays['lower'], arrays['mean']-1.959963984540054*np.sqrt(arrays['mse']), rtol=0, atol=1e-13),
                    'Primary lower interval not generated from retained MSE')
            require(np.allclose(arrays['upper'], arrays['mean']+1.959963984540054*np.sqrt(arrays['mse']), rtol=0, atol=1e-13),
                    'Primary upper interval not generated from retained MSE')
            if (job['bootstrap_B'] and not job['development_status_schema']
                    and job['method'] in ('Sikov_FH','Sikov_SAR','Sikov_SAR_Published')):
                bc_path=folder/(job['method']+'_BC.npz')
                bc_status=read(folder/(job['method']+'_BC_status.json'))
                bc_arrays=dict(np.load(bc_path,allow_pickle=False))
                require(set(bc_arrays)==set(arrays), 'BC derived prediction keys differ')
                require(all(np.isfinite(a).all() for a in bc_arrays.values()), 'Nonfinite BC prediction')
                require(np.array_equal(bc_arrays['mse'],arrays['mse_bc']), 'BC MSE differs from same-draw estimate')
                require(all(np.array_equal(bc_arrays[k],arrays[k]) for k in arrays if k not in ('mse','lower','upper')),
                        'BC derived file alters primary draws or predictors')
                bc_se=np.sqrt(np.maximum(bc_arrays['mse'],0))
                require(np.allclose(bc_arrays['lower'],bc_arrays['mean']-1.959963984540054*bc_se,rtol=0,atol=1e-13), 'BC lower interval altered')
                require(np.allclose(bc_arrays['upper'],bc_arrays['mean']+1.959963984540054*bc_se,rtol=0,atol=1e-13), 'BC upper interval altered')
                require(bc_status['same_bootstrap_draws_as_primary'] is True and bc_status['converged'] is True,
                        'BC derived status does not bind primary draws')
                require(bc_status['negative_bc_MSE_cells']==int((bc_arrays['mse']<0).sum()), 'BC negative estimate count altered')
            central_count = 0; draw_count = 0; retained_count = 0; repaired_count = 0; stage2_repaired = 0
            cert_pass_stable_fail_count = 0
            for j, value in enumerate(endpoints):
                endpoint = ('anemia', 'stunting')[j]
                require(value['converged'] is True, 'Endpoint not converged')
                for key in expected_keys:
                    require(np.array_equal(arrays[key][:, j], np.asarray(value[key])), 'Status/prediction array mismatch: '+key)
                fits = value['fit']
                require(len(fits) == 3 and {f['stratum'] for f in fits} == {1, 2, 3}, 'Expected three poverty strata')
                require(sum(f['n_areas'] for f in fits) == arrays['mean'].shape[0], 'Area count mismatch')
                for fit in fits:
                    label = job['job_id']+'/'+endpoint+'/stratum'+str(fit['stratum'])
                    values = validate_fit_metadata(fit, label, central=True)
                    update_group(groups[(job['cohort'],job['method'],'central',endpoint,fit['stratum'])], *values)
                    central_count += 1; repaired_count += int(values[5])
                    stage2_info=fit['numerical_correction'].get('stage2_numerical_correction')
                    if stage2_info is not None:
                        require(source_stage == 2, label+': stage-two correction in stage-one source')
                        require(not (stage2_info['stage1_final_original_certificate']['passed'] and stage2_info['stage1_final_stable_certificate']['passed']),
                                label+': stage two applied to an already certified fit')
                        stage2_repaired += int(stage2_info['recovered'])
                    cert_pass_stable_fail_count += int(values[1]['passed'] and not values[2]['passed'])
                    B = job['bootstrap_B']
                    if B:
                        require(fit['bootstrap_B'] == B and value['bootstrap_B'] == B, label+': changed B')
                        draws = fit['bootstrap_draw_diagnostics']
                        require(len(draws) == B and {x['draw_index_zero_based'] for x in draws} == set(range(B)), label+': missing/repeated draws')
                        retained = 0; repaired = 0; old_bad = 0; raw_bad = 0; recovered = 0
                        for row in draws:
                            draw_count += 1
                            draw_label = label+'/draw'+str(row['draw_index_zero_based'])
                            if 'final_certificate' not in row:
                                require(row.get('retained') is False and bool(row.get('exception_type')), draw_label+': unrecorded numerical failure')
                                groups[(job['cohort'],job['method'],'bootstrap_exception',endpoint,fit['stratum'])]['fits'] += 1
                                continue
                            values = validate_fit_metadata(row, draw_label, central=False)
                            stage2_info=row.get('stage2_numerical_correction')
                            if stage2_info is not None:
                                require(source_stage == 2, draw_label+': stage-two correction in stage-one source')
                                require(not (stage2_info['stage1_final_original_certificate']['passed'] and stage2_info['stage1_final_stable_certificate']['passed']),
                                        draw_label+': stage two applied to an already certified fit')
                                stage2_repaired += int(stage2_info['recovered'])
                            update_group(groups[(job['cohort'],job['method'],'bootstrap',endpoint,fit['stratum'])], *values)
                            retained += int(values[6]); repaired += int(values[5])
                            old_bad += int(not values[1]['passed'])
                            raw_bad += int(values[0] is True and not values[1]['passed'])
                            recovered += int(values[6] and values[0] is False)
                            cert_pass_stable_fail_count += int(values[1]['passed'] and not values[2]['passed'])
                        require(retained == fit['bootstrap_success'], label+': success count disagrees with draws')
                        require(B-retained == fit['bootstrap_failed'], label+': failed count disagrees with draws')
                        require(retained >= math.ceil(.95*B), label+': original success fraction not met')
                        require(repaired == fit['bootstrap_numerical_rescued'], label+': rescue count disagrees with draws')
                        require(old_bad == fit['bootstrap_original_certificate_failed'], label+': original certificate count disagrees')
                        require(raw_bad == fit['bootstrap_original_raw_success_certificate_failed'], label+': raw-only acceptance count disagrees')
                        require(recovered == fit['bootstrap_stationarity_recovered'], label+': recovery count disagrees')
                        require(retained == fit['bootstrap_strict_retained_certificates_passed'], label+': strict acceptance count disagrees')
                        retained_count += retained; repaired_count += repaired
                    else:
                        require(fit['bootstrap_B'] == 0, label+': unexpected bootstrap in delta-only graph')
            telemetry = record['telemetry']
            require(telemetry['fit_calls'] == 6*(job['bootstrap_B']+1), 'Telemetry fit call count mismatch')
            require(telemetry.get('rescue_passed', 0) == repaired_count, 'Telemetry rescue count mismatch')
            require(telemetry.get('original_certificate_passed_stable_failed', 0) == cert_pass_stable_fail_count,
                    'Telemetry certificate-disagreement count mismatch')
            if source_stage == 2:
                require(telemetry['stage2_checked_fit_calls'] == telemetry['fit_calls'], 'Stage-two fit count mismatch')
                require(telemetry.get('stage2_rescue_passed', 0) == stage2_repaired, 'Stage-two rescue count mismatch')
            job_summaries.append(dict(job_id=job['job_id'],cohort=job['cohort'],method=job['method'],
                                      source_stage=source_stage,stage2_additional_repaired_fits=stage2_repaired,
                                      central_fits=central_count,bootstrap_attempts=draw_count,
                                      bootstrap_retained=retained_count,repaired_fits=repaired_count,
                                      original_certificate_passed_stable_failed=cert_pass_stable_fail_count,
                                      completion_sha256=sha(completion)))
        except (AssertionError, KeyError, ValueError, TypeError, OSError) as exc:
            errors.append({'job_id':job['job_id'],'error':str(exc)})
        if (index+1) % 100 == 0:
            print(json.dumps({'checked':index+1,'valid':len(job_summaries),'remaining':len(remaining),'errors':len(errors)}),flush=True)
    complete = not remaining and not args.job_id and (args.stage2 or len(jobs) == len(plan['jobs']))
    passed = complete and not errors
    aggregates=[]
    for key, values in sorted(groups.items()):
        aggregates.append(dict(zip(('cohort','method','phase','endpoint','stratum'),key),**dict(values)))
    status=('PASS_ALL_SELECTED_STAGE2_FITS_AND_RETAINED_BOOTSTRAPS' if passed and args.stage2 else
            'PASS_ALL_CORRECTED_FITS_AND_RETAINED_BOOTSTRAPS' if passed else
            'VALID_VERIFIED_SUBSET_NOT_COMPLETE' if not errors else 'FAILED_CORRECTED_FIT_VERIFICATION')
    report = dict(status=status,
                  verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  complete=complete,passed=passed,truth_read=False,raw_training_arrays_opened=False,
                  global_coverage_complete=complete and not args.stage2,
                  registered_jobs=len(plan['jobs']),checked_jobs=len(jobs),valid_jobs=len(job_summaries),
                  remaining_jobs=len(remaining),invalid_jobs=len(errors),
                  source_runner_sha256=runner_hash,source_solver_sha256=solver_hash,
                  stage2_source_runner_sha256=stage2_runner_hash,stage2_source_solver_sha256=stage2_solver_hash,
                  stage2_protocol_sha256=stage2_protocol_hash,stage2_selection_sha256=selection_hash,
                  stage2_selected_jobs=len(stage2_ids),
                  source_stage_job_counts=dict(collections.Counter(x['source_stage'] for x in job_summaries)),
                  inventory_sha256=inventory_hash,protocol_sha256=protocol_hash,
                  operational_dispatch_inventories=dispatch_inventory_bindings,
                  checker_sha256=sha(Path(__file__)),source_and_output_files_hashed=len(hashed),
                  final_score_tolerance=TOLERANCE,rho_bound=RHO_BOUND,
                  accepted_rule='original AND equivalent residual-contrast projected scores < 1e-5',
                  limits=['Stored score vectors and fitting-source identity are verified; this checker does not refit or recreate every bootstrap response.',
                          'KKT stationarity is a numerical certificate, not proof of a unique global optimum or statistical model adequacy.',
                          'No evaluation outcomes are opened; predictive comparisons require the separately sealed scoring stage.'],
                  aggregates=aggregates,job_summaries=job_summaries,errors=errors,remaining_job_ids=remaining)
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    csv_path=report_path.with_suffix('.csv')
    if aggregates:
        columns=list(dict.fromkeys(k for row in aggregates for k in row))
        with csv_path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(aggregates)
    print(json.dumps({k:report[k] for k in ('status','complete','registered_jobs','valid_jobs','remaining_jobs','invalid_jobs')},indent=2))
    if errors or (not complete and not args.allow_partial):
        sys.exit(1)


if __name__=='__main__':
    main()
