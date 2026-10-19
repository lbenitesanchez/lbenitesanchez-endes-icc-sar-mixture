"""Create the immutable internal phase2 scientific freeze before model dispatch."""
from pathlib import Path
import datetime as dt
import hashlib
import json

PHASE=Path(__file__).resolve().parents[1]
STUDY=PHASE.parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write_new(p, obj):
    with Path(p).open('x') as f: json.dump(obj,f,indent=2); f.write('\n')

def main():
    if (PHASE/'protocol_freeze.json').exists(): raise FileExistsError('Scientific freeze already exists')
    if (PHASE/'datos/private_external_raw/RECH6_2024.SAV').exists():
        raise RuntimeError('Reserved responses already extracted')
    old=read(STUDY/'resultados/resultados_comparadores/protocol_freeze.json')
    sources={}
    for name, expected in old['source_sha256'].items():
        rel='scripts/'+name if name=='models.py' else 'scripts/competitors/'+name
        assert sha(STUDY/rel)==expected, rel
        sources[rel]=expected
    design=read(PHASE/'validacion/diseno_task_status.json')['source_sha256']
    for name, expected in design.items():
        assert sha(PHASE/'scripts'/name)==expected,name
        sources['scripts/'+name]=expected
    external=read(PHASE/'validacion/externa_pre_freeze_sources_ready.json')['sources']
    for rel, expected in external.items():
        assert sha(PHASE/rel)==expected,rel
        sources[rel]=expected
    for name in ['hogg_tsln.py','hogg_diagnostics.py','hogg_spatial.py',
                 'hogg_spatial_numeric.py','hogg_stage1.stan','hogg_stage2.stan',
                 'hogg_stage2_bym2.stan']:
        p=STUDY/'scripts/competitors'/name
        sources[str(p.relative_to(STUDY))]=sha(p)
    for name in ['beta_power_application_recovery.py','beta_power_application.py']:
        sources['scripts/competitors/'+name]=sha(STUDY/'scripts/competitors'/name)
    for name in ['sim_design_diagnostics.py','root_freeze_phase2.py']:
        sources['scripts/'+name]=sha(PHASE/'scripts'/name)
    aux='datos/geografia/aux_verified_2017_2018.csv'
    assert sha(PHASE/aux)=='cbec2fc142ec0ee06dd247230f0b2253e11af04d0e0e76e8374f30b456df7502'
    inputs={aux:sha(PHASE/aux)}
    for rel in ['datos/geografia/crosswalk_2017_2025_codes.csv',
                'fuentes/external/province_partition_before_responses.csv',
                'fuentes/external/external_2024_acquisition_manifest.json',
                'fuentes/external/external_validation_protocol_draft.json',
                'study_decisions_before_external_responses.json']:
        inputs[rel]=sha(PHASE/rel)
    for p in sorted((PHASE/'datos/private_external_prepared_2019_full').iterdir()):
        if p.is_file(): inputs[str(p.relative_to(PHASE))]=sha(p)
    for rel in ['datos/private_competitors/children_2019.csv',
                'datos/private_competitors/households_2019.csv']:
        inputs['previous_study/'+rel]=sha(STUDY/rel)
    prior_zip=STUDY/'ENDES_comparadores_publicacion_2026-10-06_entrega.zip'
    assert sha(prior_zip)=='fe3bd1aa449987c35f611cbfc7321deb64ce2b3bb5f3bda723a7e2daf779bd2f'
    qa_names=['diseno_numerical_qa.json','diseno_survey_validation_summary.json',
              'diseno_auxiliary_qa.json','diseno_sim_source_qa.json',
              'diseno_official_aux_preparation_validation.json',
              'externa_scorer_arithmetic_qa.json','externa_wrapper_scope_qa.json',
              'externa_parser_syntax_hash_qa.json','geografia_aux_validation.json',
              'root_prefreeze_numerical_review.json','root_common_2019_model_inputs_review.json',
              'externa_cmdstan_seed_offset_qa.json']
    evidence={}
    for name in qa_names:
        p=PHASE/'validacion'/name
        proof=read(p)
        assert 'PASS' in str(proof.get('status','')), name
        evidence['validacion/'+name]=sha(p)
    decisions=read(PHASE/'study_decisions_before_external_responses.json')
    external_protocol=read(PHASE/'fuentes/external/external_validation_protocol_draft.json')
    numerical=read(PHASE/'scripts/diseno_protocol.json')
    numerical['status']='FROZEN_BEFORE_NEW_FITS'
    protocol={'status':'INTERNAL_SCIENTIFIC_FREEZE_BEFORE_NEW_MODEL_FITS_AND_RESERVED2024_RESPONSES',
              'frozen_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'authorization':'Human requested applying improvements1,2,3 including applications and simulations',
              'approved_for_development_fits':True,'approved_for_reserved_outcome_access':False,
              'registration_scope':'Internally timestamped and hashed; not a public preregistration or complete aggregate blinding',
              'source_sha256':sources,'input_sha256':inputs,'QA_sha256':evidence,
              'decisions':decisions,'external_protocol':external_protocol,'design_protocol':numerical,
              'method_label':'Variance-informed modular beta-power CUT8; working design approximation',
              'common_model_preparation':'datos/private_external_prepared_2019_full',
              'reserved_access_rule':'Append-only approval binding this immutable protocol SHA after all three2019 predictions are locked',
              'scoring_rule':'Both temporal and territorial triple prediction locks required before any2024 performance scores',
              'CUT16_sensitivity':'2019full only; first8 exact conditional targets may be reused after target and output hash verification; fit remaining8 unchanged seeds',
              'simulation_failure_policy':'No new seed replacement or favorable cohort reduction; finite diagnostically failed predictions retained, unavailable worlds explicitly reported, no full-cohort success claim with missing output',
              'runtime':'At most6 active serial-chain workers: design2, external2, simulation2',
              'previous_delivery_sha256':sha(prior_zip)}
    write_new(PHASE/'protocol_freeze.json',protocol)
    design_gate={'status':'ROOT_APPROVED_SOURCE_AND_MODEL_DISPATCH',
                 'approved_for_model_dispatch':True,'cut16_sensitivity_approved':True,
                 'source_sha256':design,'global_protocol_sha256':sha(PHASE/'protocol_freeze.json'),
                 'official_aux_sha256':inputs[aux],
                 'common_2019_preparation_manifest_sha256':inputs['datos/private_external_prepared_2019_full/preparation_manifest.json'],
                 'scope':'No2024 responses until append-only access approval; all original16 kernels unchanged'}
    write_new(PHASE/'validacion/diseno_protocol_freeze.json',design_gate)
    print(json.dumps({'status':'FROZEN','source_count':len(sources),
                      'protocol_sha256':sha(PHASE/'protocol_freeze.json'),
                      'design_gate_sha256':sha(PHASE/'validacion/diseno_protocol_freeze.json'),
                      'reserved2024responses_opened':False}))

if __name__=='__main__':main()
