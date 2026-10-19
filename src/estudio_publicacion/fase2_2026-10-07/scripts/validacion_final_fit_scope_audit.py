"""Read-only final scope/chain/provenance audit; no external score arithmetic."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import numpy as np
import pandas as pd
from validacion_fit_models import ROOT, STUDY, sha, gate, fit_seed
from validacion_recover_hogg_status import diagnostic_audit, header
from diseno_fit_cut import SAMPLER


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def structural(path):
    return pd.read_csv(path, usecols=['child_id', 'household_id', 'year', 'ubigeo', 'prov_code',
                                     'psu_key', 'age_months', 'sex', 'de_facto'],
                       dtype={'child_id': str, 'household_id': str, 'ubigeo': str,
                              'prov_code': str, 'psu_key': str})


def test_province(code):
    return int(hashlib.sha256(('ENDES-phase2-20261007|' + code).encode()).hexdigest()[:8], 16) % 5 == 0


def main():
    protocol_sha = gate(2024)
    protocol = read(ROOT/'protocol_freeze.json')
    frozen_sources = {}
    for name, expected in protocol['source_sha256'].items():
        path = ROOT/name if (ROOT/name).exists() else STUDY/name
        require(sha(path) == expected, 'Frozen source changed: '+name)
        frozen_sources[name] = expected
    frozen_inputs = {}
    for name, expected in protocol['input_sha256'].items():
        path = STUDY/name.removeprefix('previous_study/') if name.startswith('previous_study/') else ROOT/name
        if not path.exists():
            path = STUDY/name
        require(sha(path) == expected, 'Frozen input changed: '+name)
        frozen_inputs[name] = expected
    locks = {}
    for scope in ('temporal', 'territorial'):
        path = ROOT/f'validacion/externa_prediction_lock_{scope}.json'
        locked = read(path)
        require(locked['protocol_sha256'] == protocol_sha and locked['approved_for_scoring'], 'Invalid lock')
        records = []
        for entry in locked['predictions']:
            prediction, status_path = ROOT/entry['path'], ROOT/entry['status_path']
            require(sha(prediction) == entry['sha256'], 'Locked map bytes changed')
            require(sha(status_path) == entry['status_sha256'], 'Locked status bytes changed')
            require(read(status_path)['success'], 'Locked model not successful')
            records.append({'method': entry['method'], 'prediction_sha256': sha(prediction),
                            'status_sha256': sha(status_path), 'districts': entry['districts']})
        locks[scope] = {'sha256': sha(path), 'locked_utc': locked['locked_utc'], 'exact_bytes_unchanged': True,
                        'predictions': records}
    first = read(ROOT/'validacion/externa_first_reserved_open.json')
    require(locks['temporal']['locked_utc'] < first['time_utc'] < locks['territorial']['locked_utc'],
            'Temporal lock/first reserved access/territorial lock order differs')
    prep_dir = ROOT/'datos/private_external_prepared_2024_training'
    prep = read(prep_dir/'preparation_manifest.json')
    for name, expected in prep['output_sha256'].items():
        require(sha(prep_dir/name) == expected, 'Prepared training input changed: '+name)
    for name, path in [('households', ROOT/'datos/private_external/households_2024.csv'),
                       ('children', ROOT/'datos/private_external/children_2024_training.csv'),
                       ('aux', ROOT/'datos/geografia/aux_verified_2017_2018.csv')]:
        require(sha(path) == prep['input_sha256'][name], 'Prepared raw input changed: '+name)
    require(prep['year']==2024 and prep['scope']=='territorial_training' and not prep['reserved_test_outcomes_read'],
            'Preparation does not isolate territorial training')
    train = structural(ROOT/'datos/private_external/children_2024_training.csv')
    test = structural(ROOT/'datos/private_external/children_2024_test.csv')
    cohort = structural(ROOT/'datos/private_external/children_2024_scoring_only.csv')
    for data in (train, test, cohort):
        require(data.child_id.is_unique and data.year.eq(2024).all() and data.de_facto.eq(1).all()
                and data.age_months.between(6,59).all() and data.sex.isin([1,2]).all(), 'Structural eligibility/key failure')
        require(data.prov_code.eq(data.ubigeo.str[:4]).all(), 'Province code differs from UBIGEO')
    require(not train.prov_code.map(test_province).any() and test.prov_code.map(test_province).all(), 'Hash-fold overlap')
    overlaps = {name: len(set(train[name]) & set(test[name])) for name in ('child_id','household_id','psu_key','prov_code')}
    require(not any(overlaps.values()), 'Train/test identifier/group overlap')
    require(set(cohort.child_id) == set(train.child_id) | set(test.child_id), 'Temporal cohort differs from full partition')
    households = pd.read_csv(ROOT/'datos/private_external/households_2024.csv', dtype={'household_id':str,'ubigeo':str,'psu_key':str})
    frame = pd.read_csv(ROOT/'datos/private_external_raw/household_frame_2024.csv', dtype={'household_id':str,'ubigeo':str,'psu_key':str})
    common = [name for name in frame.columns if name != 'household_id']
    require(households[common].equals(frame[common]), 'Full household design fields altered after responses opened')
    require(households.household_id.tolist()==[f'2024_H{i:05d}'for i in range(1,len(frame)+1)],
            'Anonymous household key does not follow the frozen parser row mapping')
    require(set(cohort.household_id) <= set(households.household_id), 'Child household outside full frame')
    require(len(households)==prep['full_households'] and households.psu_key.nunique()==prep['fullPSUs'], 'Frame counts differ')
    areas = pd.read_csv(ROOT/'datos/geografia/aux_verified_2017_2018.csv', dtype={'ubigeo':str})
    area_codes = set(areas.ubigeo)
    ids = pd.read_csv(prep_dir/'area_ids.csv',dtype={'ubigeo':str})
    require(np.array_equal(ids.ubigeo,areas.ubigeo), 'Prepared auxiliary order differs')
    inputs = dict(np.load(prep_dir/'fit_inputs.npz'))
    require(not ids.ubigeo.iloc[inputs['sample_idx']].str[:4].map(test_province).any(), 'Test province in fit sample_idx')
    require(len(inputs['test_idx'])==0, 'Heldout outcome indices in training NPZ')
    crosswalk = pd.read_csv(ROOT/'datos/geografia/crosswalk_2017_2025_codes.csv',dtype={'ubigeo_2025':str})
    new = crosswalk.loc[~crosswalk.historical_aux_available,'ubigeo_2025'].dropna().tolist()
    require(len(new)==17 and not (set(new)&area_codes), 'Post2017 creation exclusion differs')
    unmatched = {scope: sorted(set(data.ubigeo)-area_codes) for scope,data in [('training',train),('test',test),('full_cohort',cohort),('households',households)]}
    hogg_dir = ROOT/'resultados/externa_territorial_2024_hogg'
    hogg = read(hogg_dir/'status_recovered.json')
    recovery = read(hogg_dir/'serialization_recovery_manifest.json')
    require(recovery['original_wrapper_exit_code']==1 and recovery['no_refit'] and recovery['no_frozen_source_modified'], 'Hogg incident concealed')
    require(hogg['input_sha256']['units']==prep['input_sha256']['children'] and hogg['input_sha256']['aux']==prep['input_sha256']['aux'], 'Hogg fitting data differs')
    source, spatial = hogg['source_status'], hogg['spatial_status']
    hogg_audits = {}
    hogg_predictions = pd.read_csv(hogg_dir/'predictions.csv')
    for j, endpoint in enumerate(('anemia','stunting')):
        source_attempts = source['all_attempts'][endpoint]
        spatial_attempts = spatial['all_attempts'][endpoint]
        for kind, attempts, directory in [('canonical_source',source_attempts,hogg_dir/'age_sex_source'/endpoint),
                                          ('spatial',spatial_attempts,hogg_dir/'spatial'/endpoint)]:
            actual = sorted(path.name for path in directory.glob('attempt_*') if path.is_dir())
            require(actual == [f'attempt_{i}'for i in range(len(attempts))] and len(attempts)<=2,
                    'Hidden/discarded Hogg attempts')
        seed = fit_seed(2024,'hogg')+100*j
        audits = {}
        source_run = hogg_dir/'age_sex_source'/endpoint/f'attempt_{len(source_attempts)-1}'
        for stage, offset in [('stage1',0),('stage2',1)]:
            audits[stage], _ = diagnostic_audit(source_run,source['outcomes'][endpoint],stage,seed+offset)
        spatial_run = hogg_dir/'spatial'/endpoint/f'attempt_{len(spatial_attempts)-1}'
        audits['spatial_stage2'], draws = diagnostic_audit(spatial_run,spatial['outcomes'][endpoint],'stage2',seed,
                                                         spatial=True,with_prevalence=True)
        error = float(np.max(np.abs(draws.mean(axis=0)-hogg_predictions['p_mean_'+endpoint].to_numpy(float))))
        require(error<=1e-12,'Actual-chain spatial mean differs from locked map')
        hogg_audits[endpoint]={'canonical_attempt_count':len(source_attempts),'spatial_attempt_count':len(spatial_attempts),
                              'actual_chain_mean_max_difference':error,'stages':audits}
        for stage, audit in audits.items():
            require(audit['chains']==recovery['actual_retained_chains'][endpoint][stage]['chains'], 'Recovery CSV provenance changed')
    design_dir = ROOT/'resultados/externa_territorial_2024_designcut8'
    design = read(design_dir/'completion.json')
    require(design['state_weights']==[.125]*8 and design['conditional_states']==8 and not design['reserved_test_outcomes_read'], 'CUT scope/state weights changed')
    actual_states = sorted(p.name for p in design_dir.glob('state_*') if p.is_dir())
    require(actual_states==[f'state_{i:02d}'for i in range(1,9)],'Missing/added/discarded CUT states')
    calibrations=[np.load(prep_dir/f'{endpoint}_calibration.npz')for endpoint in ('anemia','stunting')]
    design_audits=[]
    for k in range(8):
        state=design_dir/f'state_{k+1:02d}';status=read(state/'fit_status.json');completion=read(state/'completion.json')
        require(sha(state/'fit_status.json')==completion['sampler_status_sha256'] and sha(state/'posterior_summary_and_draws.npz')==completion['posterior_output_sha256'], 'CUT posterior/status bytes changed')
        seed=20261007+2024*100+10000+k*100003
        n=np.column_stack([x['mixture_effective_n'][k]for x in calibrations])
        target=hashlib.sha256()
        for array in (inputs['X'],inputs['y'],n,inputs['sample_idx'],inputs['basis']):
            value=np.ascontiguousarray(array);target.update(str(value.dtype).encode());target.update(json.dumps(value.shape).encode());target.update(value.tobytes())
        target.update(json.dumps(dict(seed=seed,sampler=SAMPLER,config={'cluster_adjustment':False}),sort_keys=True).encode())
        require(status['seed']==seed and status['conditional_target_sha256']==target.hexdigest(), 'CUT conditional parameters/data target differs')
        require(status['success'] and all(x['success'] and x['beta_shape_guard_cells']==0 for x in status['end_points']), 'CUT numerical failure hidden')
        design_audits.append({'state':k+1,'seed':seed,'target_sha256':target.hexdigest(),
            'fit_status_sha256':sha(state/'fit_status.json'),'posterior_sha256':completion['posterior_output_sha256'],
            'endpoint_trial_counts':[len(x['sampler_trials'])for x in status['end_points']]})
    baseline=read(ROOT/'resultados/externa_territorial_2024_baseline/status.json')
    require(baseline['input_sha256']==hogg['input_sha256'] and baseline['preparation_manifest_sha256']==sha(prep_dir/'preparation_manifest.json'), 'Baseline common data differs')
    require(all(x['success'] and x['beta_shape_guard_cells']==0 for x in baseline['endpoints']), 'Baseline gate failed')
    result={'status':'PASS_FINAL_EXTERNAL_FIT_SCOPE_AND_CHAIN_PROVENANCE','audited_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'audit_source_sha256':sha(Path(__file__)),'protocol_sha256':protocol_sha,
        'unchanged_frozen_sources':frozen_sources,'unchanged_frozen_inputs':frozen_inputs,'prediction_locks':locks,
        'chronology':{'temporal_lock_before_first_reserved_access':True,'first_reserved_open_utc':first['time_utc'],
                      'territorial_lock_after_first_access':True},
        'structural_scope':{'training_children':len(train),'test_children':len(test),'full_temporal_cohort':len(cohort),
            'identifier_overlap':overlaps,'hash_fold_rule_verified':True,'training_only_sample_domains':len(inputs['sample_idx']),
            'full_households':len(households),'full_PSUs':households.psu_key.nunique(),'full_strata':households.stratum.nunique(),
            'before_response_household_design_fields_exactly_preserved':True,
            'raw_household_key_replaced_by_frozen_sequential_anonymous_key':True,
            'test_outcome_columns_read_by_this_audit':False},
        'geography':{'historical_prediction_frame_districts':len(areas),'post2017_unverified_creation_codes_excluded_from_model_frame':new,
                    'unmatched_observed_codes_by_scope':unmatched,'boundary_equivalence_verified':False,
                    'no_parent_prevalence_transfer_to_new_districts':True,'code_crosswalk_2025_metadata_not_numeric_feature':True},
        'hogg_actual_retained_chain_count':24,'hogg_actual_chain_audits':hogg_audits,
        'hogg_serialization_incident_preserved':True,'hogg_recovery_manifest_sha256':sha(hogg_dir/'serialization_recovery_manifest.json'),
        'CUT8_conditional_target_audits':design_audits,'CUT8_state_selection_or_weight_change':False,
        'CUT8_trace_storage_limit':'Frozen conditional runner retains posterior arrays, actual sampler diagnostics and target hashes, but removes temporary raw hyperparameter CSV; 24-CSV direct audit applies to Hogg, not CUT.',
        'baseline_endpoint_trial_counts':[len(x['sampler_trials'])for x in baseline['endpoints']],
        'no_hidden_or_discarded_Hogg_attempts':True,'no_missing_or_discarded_CUT_states':True,
        'external_scores_recomputed':False,'external_score_audit_owned_by':'fase2_geografia'}
    destination=ROOT/'validacion/externa_final_fit_scope_chain_audit.json'
    with destination.open('x')as file:json.dump(result,file,indent=2)
    print(json.dumps({'status':result['status'],'audit_sha256':sha(destination),
                      'Hogg_retained_chains':24,'overlap':overlaps,'scope_counts':[len(train),len(test),len(cohort)]}),flush=True)


if __name__=='__main__':
    main()
