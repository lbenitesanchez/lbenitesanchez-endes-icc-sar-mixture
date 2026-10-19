"""Assemble a NEW comparator review package only after final release gates.

This script never edits the earlier package, statistical source, finished result
tree or documents. Its default mode is a non-mutating preflight. Assembly and
ZIP creation require --assemble and a current release-gate JSON.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'paquete_publicacion'
OUT=ROOT/'resultados/resultados_comparadores'
PERSONAL_PREFIX=str(Path.home())+'/'
DOCS=['Manuscript_ENDES_IAOS.docx','Supplement_ENDES_IAOS.docx','Cover_letter_IAOS.docx',
      'Title_page_author_confirmation.docx','Resultados_y_guia_de_publicacion.docx']
TEXT={'.py','.r','.stan','.md','.txt','.json','.jsonl','.csv','.sh'}
FROZEN=['resultados/resultados_comparadores/protocol_freeze.json',
        'resultados/resultados_comparadores/candidate_frozen_manifest.json',
        'resultados/competidores_gao/gao_frozen_code_manifest.json',
        'resultados/competidores_hogg/hogg_sampler_freeze.json',
        'resultados/competidores_hogg/hogg_strong_arms_freeze.json']
PUBLIC_COPY_RECORDS={}


def public_provenance_module():
    path=ROOT/'scripts/public_competitors/public_copy_provenance.py'
    spec=importlib.util.spec_from_file_location('public_copy_provenance',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(s):
    # Metadata only. Statistical source and frozen manifests are byte-exact.
    return s.replace(str(ROOT)+'/', '').replace(PERSONAL_PREFIX, 'local_user/')


def copy(src,dst,*,exact=False):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not exact and src.suffix.lower() in TEXT:
        original=src.read_text();sanitized=clean(original)
        dst.write_text(sanitized,encoding='utf-8')
        if original!=sanitized:
            PUBLIC_COPY_RECORDS[str(dst.resolve())]=(src.resolve(),dst.resolve())
    else:shutil.copy2(src,dst)


def frozen_checks(destination=None):
    folder=ROOT if destination is None else destination
    protocol=json.loads((folder/FROZEN[0]).read_text())
    checks=[]
    candidate_rel=protocol.get('candidate_frozen_manifest_file',FROZEN[1])
    checks.append({'path':candidate_rel,'passed':sha(folder/candidate_rel)==protocol['candidate_frozen_manifest_sha256']})
    for name,expected in protocol['source_sha256'].items():
        f=folder/('scripts/models.py'if name=='models.py'else'scripts/competitors/'+name)
        checks.append({'path':str(f.relative_to(folder)),'passed':sha(f)==expected})
    for key,rel in zip(['gao','hogg','hogg_strong'],FROZEN[2:]):
        manifest=folder/rel
        checks.append({'path':rel,'passed':sha(manifest)==protocol['competitor_freeze_manifest_sha256'][key]})
        obj=json.loads(manifest.read_text())
        files=obj.get('files',[])
        files+= [{'file':'scripts/competitors/'+n,'sha256':v}for n,v in obj.get('source_sha256',{}).items()]
        for item in files:checks.append({'path':item['file'],'passed':sha(folder/item['file'])==item['sha256']})
    spatial_rel='resultados/competidores_hogg/hogg_spatial_freeze.json'
    spatial=json.loads((folder/spatial_rel).read_text())
    if not spatial.get('approved_after_numerical_pilot'):raise RuntimeError('Spatial numerical freeze has not passed')
    for name,expected in spatial['source_sha256'].items():
        rel='scripts/competitors/'+name
        checks.append({'path':rel,'passed':sha(folder/rel)==expected})
    amendment_rel='resultados/competidores_hogg/hogg_spatial_numerical_amendment.json'
    if (folder/amendment_rel).exists():
        amendment=json.loads((folder/amendment_rel).read_text())
        if not amendment.get('approved_after_independent_audit_and_pilot'):
            raise RuntimeError('Optional spatial initializer amendment is not yet approved')
        rel='scripts/competitors/hogg_spatial_numeric.py'
        checks.append({'path':rel,'passed':sha(folder/rel)==amendment['adapter_sha256']})
        review=json.loads((folder/'validacion/hogg_spatial_numerical_initializer_independent_review.json').read_text())
        checks.append({'path':'validacion/hogg_spatial_numerical_initializer_independent_review.json',
                       'passed':str(review.get('status','')).startswith('PASS')and not review.get('required_findings_open')
                       and review['adapter_sha256']==amendment['adapter_sha256']})
        checks.append({'path':spatial_rel,'passed':sha(folder/spatial_rel)==amendment['original_spatial_freeze_sha256']})
        for path_key,sha_key in [('independent_audit_path','independent_audit_sha256'),
                                 ('numerical_pilot_path','numerical_pilot_sha256')]:
            rel=Path(amendment[path_key])
            if rel.is_absolute():raise RuntimeError('Spatial amendment evidence paths must be portable')
            checks.append({'path':str(rel),'passed':sha(folder/rel)==amendment[sha_key]})
        contract_rel='resultados/resultados_comparadores/spatial_result_contract.json'
        contract=json.loads((folder/contract_rel).read_text())
        if contract.get('execution_contract_version')!=2 or contract['fixed_population_ids']!=list(range(20000,20100)):
            raise RuntimeError('Spatial execution contract must retain its100 matched population IDs')
        execution=contract['numerical_execution_amendment']
        for path,expected in [(amendment_rel,execution['amendment_sha256']),
                              ('scripts/competitors/hogg_spatial_numeric.py',execution['adapter_sha256']),
                              (contract['spatial_freeze'],contract['spatial_freeze_sha256']),
                              (contract['task_file'],contract['task_sha256']),
                              ('scripts/competitors/benchmark_spatial_evaluate.py',contract['evaluator_sha256']),
                              ('scripts/competitors/benchmark_metrics.py',contract['methods_source_sha256'])]:
            if path==contract['task_file'] and destination is not None:
                checks.append(public_provenance_module().verify_public_reference(folder,path,expected))
            else:checks.append({'path':path,'passed':sha(folder/path)==expected})
        archive_rel='resultados/resultados_comparadores/spatial_execution_contract_archive/v1_before_initializer_adapter'
        history=folder/archive_rel
        preservation=json.loads((history/'preservation.json').read_text())
        for name,expected in preservation['files_sha256'].items():
            if name not in {'spatial_result_contract.json','spatial_evaluation_queue_manifest.json','benchmark_spatial_evaluate.py'}:
                raise RuntimeError('Unexpected spatial execution archive entry')
            checks.append({'path':archive_rel+'/'+name,'passed':sha(history/name)==expected})
        checks.append({'path':archive_rel+'/spatial_result_contract.json',
                       'passed':sha(history/'spatial_result_contract.json')==execution['original_result_contract_sha256']})
    if not all(row['passed']for row in checks):raise RuntimeError('Frozen code or manifest differs')
    return checks


def final_tags():
    contract=json.loads((OUT/'final_result_contract.json').read_text())
    tags=contract['final_tags']
    if any((OUT/tag/'invalidated.json').exists()for tag in tags):
        raise RuntimeError('Invalidated numerical runs cannot be released as final results')
    return tags


def release_checks(gate_path):
    gate=json.loads(gate_path.read_text())
    if not gate.get('scientific_review_passed')or not gate.get('all_pages_inspected'):
        raise RuntimeError('Final scientific review and visual inspection of every document page are required')
    for name in DOCS:
        if gate.get('document_sha256',{}).get(name)!=sha(ROOT/'manuscrito'/name):
            raise RuntimeError('Document release gate is stale: '+name)
    path=OUT/'final_evaluation/final_completion.json'
    done=json.loads(path.read_text())
    if not done.get('evaluation_completed')or not done.get('all_requested_populations_represented'):
        raise RuntimeError('Final independent simulation evaluation is incomplete')
    if not done.get('primary_Holm_family_available')or done.get('unique_observed_populations')!=800:
        raise RuntimeError('The locked primary comparison family and800 populations are required')
    if gate.get('final_completion_sha256')!=sha(path):raise RuntimeError('Final result gate is stale')
    spatial=OUT/'spatial_evaluation/completion.json'
    spatial_done=json.loads(spatial.read_text())
    if not spatial_done.get('evaluation_completed')or not spatial_done.get('all_requested_populations_represented')or not spatial_done.get('all_requested_candidate_pairs_available'):
        raise RuntimeError('The matched100-world spatial sensitivity is incomplete')
    if gate.get('spatial_completion_sha256')!=sha(spatial):raise RuntimeError('Spatial result gate is stale')
    boundary=json.loads((OUT/'boundary_audit/completion.json').read_text())
    if not boundary.get('completed')or boundary.get('unique_populations')!=800 or boundary.get('additional_independent_populations')!=0:
        raise RuntimeError('The descriptive boundary accounting must preserve the800-world cohort')
    if not boundary.get('primary_score_and_Holm3_unchanged')or boundary.get('main_completion_sha256')!=sha(path):
        raise RuntimeError('Boundary accounting differs from the locked final results')
    spatial_amendment=ROOT/'resultados/competidores_hogg/hogg_spatial_numerical_amendment.json'
    if spatial_amendment.exists()and gate.get('hogg_spatial_numerical_amendment_sha256')!=sha(spatial_amendment):
        raise RuntimeError('Optional spatial initializer amendment gate is stale')
    sensitivity=OUT/'application_sensitivity_v2/completion.json'
    sensitivity_done=json.loads(sensitivity.read_text())
    if not sensitivity_done.get('completed')or not sensitivity_done.get('all_requested_jobs_recorded')or not sensitivity_done.get('all_endpoints_converged')or sensitivity_done.get('jobs')!=7:
        raise RuntimeError('All7 declared application sensitivities must pass the fixed gates')
    if gate.get('application_sensitivity_completion_sha256')!=sha(sensitivity):raise RuntimeError('Application sensitivity gate is stale')
    amendment=json.loads((OUT/'numerical_amendment_v2.json').read_text())
    if not amendment.get('approved_for_dispatch')or amendment.get('statistical_configuration_changed'):
        raise RuntimeError('Same-target numerical amendment requires completed approval')
    if set(done['prediction_artifact_runs'])!=set(final_tags()):
        raise RuntimeError('Completion and amended final-result contract differ')
    original=ROOT/amendment['original_protocol']
    if sha(original)!=amendment['original_protocol_sha256']:
        raise RuntimeError('Original archived freeze differs')
    return gate,done


def assemble(destination,archive,gate_path):
    gate,completion=release_checks(gate_path);original_checks=frozen_checks()
    if destination.exists()or archive.exists():raise FileExistsError('Use new package and archive names')
    if destination.is_relative_to(BASE.resolve())or archive.is_relative_to(destination):
        raise ValueError('The new package must be outside the legacy tree and its ZIP must be outside the new tree')
    PUBLIC_COPY_RECORDS.clear()
    shutil.copytree(BASE,destination,ignore=shutil.ignore_patterns('__pycache__','*.pyc','package_manifest.json'))
    source_records=[]
    # Preserve all scientific kernels and frozen-generation snapshots exactly.
    for name in ['models.py','02_application.py','06_build_documents.py','11_export_unit_inputs.R','11_application_comparison.py','12_competitor_documents.py',
                 '13_competitor_figures.py','14_package_competitors.py']:
        src=ROOT/'scripts'/name;dst=destination/'scripts'/name
        copy(src,dst,exact=True);source_records.append({'path':str(dst.relative_to(destination)),
             'original_sha256':sha(src),'public_sha256':sha(dst),'adapted':False})
    for sub in ['competitors','public_competitors']:
        for src in (ROOT/'scripts'/sub).iterdir():
            if sub=='competitors'and src.name=='benchmark_final_zip_audit.py':
                continue  # External workspace-only release auditor; not a reproduction stage.
            if src.is_file()and src.suffix.lower()in {'.py','.stan','.r','.md','.txt'}:
                dst=destination/'scripts'/sub/src.name;copy(src,dst,exact=True)
                source_records.append({'path':str(dst.relative_to(destination)),
                     'original_sha256':sha(src),'public_sha256':sha(dst),'adapted':False})
    for rel in FROZEN:copy(ROOT/rel,destination/rel,exact=True)
    for name in ['numerical_amendment_v2.json','final_result_contract.json','candidate_frozen_manifest_v2.json',
                 'protocol_numerical_v2_approved.json','spatial_result_contract.json','secondary_arm_result_contract.json']:
        copy(OUT/name,destination/'resultados/resultados_comparadores'/name,exact=True)
    spatial_archive=OUT/'spatial_execution_contract_archive/v1_before_initializer_adapter'
    for name in ['preservation.json','spatial_result_contract.json','spatial_evaluation_queue_manifest.json','benchmark_spatial_evaluate.py']:
        src=spatial_archive/name
        copy(src,destination/src.relative_to(ROOT),exact=True)
    archive_folder=OUT/'protocol_archive/v1_before_numerical_repair'
    for name in ['protocol_freeze.json','candidate_frozen_manifest.json','candidate_queue_manifest.json']:
        src=archive_folder/name
        if src.exists():copy(src,destination/src.relative_to(ROOT),exact=name!='candidate_queue_manifest.json')
    historical=OUT/'source_snapshots/final_primary_v1'
    old_names=json.loads((archive_folder/'protocol_freeze.json').read_text())['source_sha256']
    for name,expected in old_names.items():
        src=historical/name
        if not src.exists():src=ROOT/('scripts/models.py'if name=='models.py'else'scripts/competitors/'+name)
        if sha(src)!=expected:raise RuntimeError('Invalidated source snapshot differs: '+name)
        copy(src,destination/'historical_source_snapshots/v1_invalidated'/name,exact=True)
    (destination/'historical_source_snapshots/v1_invalidated/README.txt').write_text(
        'Historical v1 source only. Numerical evaluation was invalidated after12completed populations.\n'
        'Do not use these sources/results for final inference. Approved v2 reruns all800 identical samples/IDs.\n')
    for src in (ROOT/'resultados/competidores_hogg').glob('*spatial*freeze*.json'):
        copy(src,destination/'resultados/competidores_hogg'/src.name,exact=True)
    spatial_amendment=ROOT/'resultados/competidores_hogg/hogg_spatial_numerical_amendment.json'
    if spatial_amendment.exists():copy(spatial_amendment,destination/spatial_amendment.relative_to(ROOT),exact=True)
    for src in (ROOT/'resultados/competidores_hogg/numeric_adapter_history').glob('*.py'):
        copy(src,destination/src.relative_to(ROOT),exact=True)
    for tag in ['finite_v2','sparse_v2']:
        for src in (ROOT/'resultados/simulaciones/source_snapshots'/tag).iterdir():
            if src.is_file()and src.suffix in {'.py','.csv'}:
                copy(src,destination/'resultados/simulaciones/source_snapshots'/tag/src.name,exact=True)
    # Public raw reconstruction can optionally emit private local intermediates.
    data_code=destination/'codigo/01_prepare_data.R'
    old=data_code.read_text();needle=' # The public reconstruction does not persist individual-level intermediates.'
    if old.count(needle)!=1:raise RuntimeError('Expected portable reconstruction guard changed')
    replacement=(' # Optional local intermediates for unit-level comparators; never redistribute.\n'
                 " if(Sys.getenv('ENDES_EXPORT_PRIVATE_INPUTS')=='1'){\n"
                 "  private_dir<-file.path(package_root,'datos/raw');dir.create(private_dir,recursive=TRUE,showWarnings=FALSE)\n"
                 "  saveRDS(list(household=hh,child=child),file.path(private_dir,paste0('analytic_private_',year,'.rds')))\n"
                 ' }')
    data_code.write_text(old.replace(needle,replacement))
    source_records.append({'path':'codigo/01_prepare_data.R','original_sha256':sha(BASE/'codigo/01_prepare_data.R'),
        'public_sha256':sha(data_code),'adapted':True,'adaptation':'Explicit optional privateRDS output; no statistical calculation changed'})
    copy(ROOT/'scripts/public_competitors/README_COMPETIDORES_ES_EN.md',destination/'README_COMPETIDORES_ES_EN.md')
    for name in DOCS:copy(ROOT/'manuscrito'/name,destination/'manuscrito'/name,exact=True)
    for name in ['references.json','sections.json']:copy(ROOT/'manuscrito'/name,destination/'manuscrito'/name,exact=True)
    for src in (ROOT/'manuscrito').glob('competitor*.json'):copy(src,destination/'manuscrito'/src.name)
    # Only district/aggregate survey outputs, never group statistics, raw draws,
    # Stan fitting-data JSON, endpoint summaries or unit exports.
    for folder in ['final_evaluation','application_comparison_final_v1','spatial_evaluation','boundary_audit']:
        for src in (OUT/folder).iterdir():
            if src.is_file()and src.suffix in {'.csv','.json'}:copy(src,destination/'resultados/resultados_comparadores'/folder/src.name)
    allowed={'district_predictions.csv','province_test_predictions.csv','test_predictions.csv','test_summary.csv','fit_status.json',
             'test_fit_status.json','manifest.json','predictions.csv','application_manifest.json',
             'numerical_initialization_summary.json'}
    for parent in [OUT/'application_final_candidate_v2',ROOT/'resultados/competidores_gao',ROOT/'resultados/competidores_hogg']:
        for src in parent.rglob('*'):
            if src.is_file()and src.name in allowed and ('endes_' in str(src)or parent.name.startswith('application_final_candidate')):
                copy(src,destination/src.relative_to(ROOT))
    # Declared application sensitivities use the same private-unit protection.
    plan_folder=OUT/'application_sensitivity_v2'
    for name in ['declared_plan.json','queue_status.json','completion.json','descriptive_summary.csv','summary_manifest.json']:
        src=plan_folder/name
        if src.exists():copy(src,destination/src.relative_to(ROOT))
    for parent in OUT.glob('application_*_v2'):
        if parent.name.startswith('application_comparison'):continue
        for src in parent.rglob('*'):
            if src.is_file()and src.name in allowed:copy(src,destination/src.relative_to(ROOT))
    # All fit-facing final synthetic predictions and diagnostic/failure contracts
    # are indexed, while raw chains, fitting-data JSON and compiled binaries stay out.
    synthetic_contract={'predictions.csv','status.json','diagnostics.json','task_execution.json',
                        'execution_failure.json','worker_failure.json','failure.json','numerical_initialization_summary.json'}
    synthetic_parents=[*[OUT/tag/'fit_results'for tag in final_tags() if tag.startswith('final_primary')or tag.startswith('final_structural')],
                       OUT/'external_predictions/gao',ROOT/'resultados/competidores_hogg']
    final_ids=set(range(10000,10200))|set(range(20000,20100))
    for parent in synthetic_parents:
        for src in parent.rglob('*'):
            rep_ids=[int(x[4:])for x in src.parts if x.startswith('rep_')and x[4:].isdigit()]
            if src.is_file()and src.name in synthetic_contract and rep_ids and rep_ids[-1]in final_ids:
                copy(src,destination/src.relative_to(ROOT))
    # The preceding statistical sensitivity plan is retained only as a hashed
    # historical specification needed by the v2 queue, never as final predictions.
    historic_plan=OUT/'application_sensitivity_v1/declared_plan.json'
    copy(historic_plan,destination/historic_plan.relative_to(ROOT))
    # Numerical pilot summaries and deterministic audits only. Raw CSV chains,
    # private fitting JSON, temporary unit inputs and compiled binaries stay out.
    for parent in [OUT/'validation/beta_power_hybrid_v2',ROOT/'resultados/competidores_hogg/spatial_pilot',
                   ROOT/'resultados/competidores_hogg/spatial_numeric_pilot']:
        for src in parent.rglob('*.json'):
            if src.name in {'completion.json','status.json','progress.json','declared_plan.json','posterior_equivalence.json',
                            'numerical_initialization_summary.json'}:
                copy(src,destination/src.relative_to(ROOT))
    for src in (OUT/'validation/final_numerical_rep10012').iterdir():
        if src.is_file()and src.suffix in {'.py','.json'}and ('audit'in src.name or src.name in {'capture.json','independent_stable_gamma_ratio_checks.py'}):
            copy(src,destination/src.relative_to(ROOT))
    for tag in final_tags():
        for name in ['config.json','completion.json','fits.json','failures.json','failures.jsonl','replications.csv','summary.csv']:
            src=OUT/tag/name
            if src.exists():copy(src,destination/src.relative_to(ROOT))
    for src in (OUT/'task_plan').glob('*.csv'):copy(src,destination/src.relative_to(ROOT))
    # Wholly synthetic inputs/predictions are safe; retain estimator/evaluator separation.
    plan=json.loads((OUT/'protocol_freeze.json').read_text())['independent_seed_ids']
    for scenario,seeds in plan.items():
        for seed in seeds:
            folder=OUT/'inputs'/scenario/f'rep_{seed:05d}'
            for src in folder.rglob('*'):
                if src.is_file()and src.suffix in {'.csv','.npy','.npz','.json'}:copy(src,destination/src.relative_to(ROOT))
    for rel in ['input_validation_final800.json']:
        copy(OUT/rel,destination/'resultados/resultados_comparadores'/rel)
    for name in ['spatial_evaluator_qa.json','final_result_readiness.json','boundary_audit_qa.json',
                 'secondary_accounting_qa.json','secondary_zero_availability_qa.json',
                 'independent_final_audit_qa.json','final_audit_observer.json']:
        src=OUT/'validation'/name
        if src.exists():copy(src,destination/src.relative_to(ROOT))
    for parent in [ROOT/'fuentes/competidores/gao_wakefield',ROOT/'fuentes/competidores']:
        for src in parent.glob('*manifest.json'):copy(src,destination/src.relative_to(ROOT))
    # Bibliographic verification metadata only; primary full texts remain
    # obtainable at the recorded official URLs and are not redistributed.
    conceptual_sources=ROOT/'fuentes/competidores/conceptual_novelty/sources_manifest.json'
    if conceptual_sources.exists():copy(conceptual_sources,destination/conceptual_sources.relative_to(ROOT))
    for src in (ROOT/'validacion').iterdir():
        if src.is_file()and src.suffix in {'.json','.txt','.py'}and (any(k in src.name for k in ['beta_power','competitor_protocol','competitor_narrative','hogg_spatial','competitor_manuscript','competitor_builder','competitor_package'])
                or src.name in {'competitor_final_independent_audit.json','competitor_population_rng_independence_audit.json',
                    'competitor_final_document_visual_review.json','competitor_final_figure_visual_review.json'}):
            copy(src,destination/'validacion'/src.name)
    recipe_history=ROOT/'validacion/public_recipe_source_history/pre_final_arithmetic_addition_20261006'
    for name in ['14_package_competitors.py','reproduce.py','preservation.json']:
        src=recipe_history/name
        copy(src,destination/src.relative_to(ROOT),exact=True)
    # Read-only final execution evidence is distinct from the arithmetic audit.
    # These contain accounting/diagnostics, not unit inputs or sampler draws.
    for name in ['final_execution_audit.json','final_execution_endpoint_diagnostics.csv',
                 'spatial_final_execution_summary.json','spatial_final_endpoint_diagnostics.csv',
                 'spatial_application_final_diagnostics.json','spatial_application_graph_order_check.json']:
        src=ROOT/'resultados/competidores_hogg'/name
        if src.exists():copy(src,destination/src.relative_to(ROOT))
    for src in (ROOT/'figuras').glob('competitor*'):
        if src.suffix in {'.png','.svg','.pdf'}:copy(src,destination/'figuras'/src.name,exact=True)
    copy(gate_path,destination/'validacion/competitor_publication_release_gate.json')
    # Update the earlier component indexes for the single optional private I/O
    # change. Historical reconstruction evidence retains its original snapshot.
    data_index=destination/'datos/provenance/data_component_manifest.json'
    data_manifest=json.loads(data_index.read_text())
    changes=[]
    for record in data_manifest['files']:
        current=destination/record['path']
        if sha(current)!=record['sha256']:
            changes.append({'path':record['path'],'historical_sha256':record['sha256'],
                            'current_sha256':sha(current)})
            record['sha256']=sha(current);record['bytes']=current.stat().st_size
    data_manifest['subsequent_IO_change']={'numeric_reconstruction_unchanged':True,
        'change':'Explicit opt-in privateRDS export for locally reproduced unit-level competitors',
        'historical_validation_snapshot_preserved':True,'source_changes':changes}
    data_index.write_text(json.dumps(data_manifest,indent=2))
    helper=destination/'codigo/refresh_public_provenance.py'
    spec=importlib.util.spec_from_file_location('legacy_component_refresh',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.refresh_public_manifests(destination)
    for cache in destination.rglob('__pycache__'):
        shutil.rmtree(cache)
    provenance=public_provenance_module()
    transformations=[provenance.transformation_record(src,dst,ROOT,dst.relative_to(destination))
                     for src,dst in PUBLIC_COPY_RECORDS.values()]
    (destination/provenance.MANIFEST).write_text(json.dumps({
        'scope':'Operational path redaction only; historical contracts and approved evidence retain original checksums.',
        'source_original_bytes_checked_before_redaction':True,
        'source_frozen_validation':original_checks,
        'original_checksums_are_not_public_byte_checksums':True,
        'statistical_sources_and_approved_evidence_preserved_exactly':True,
        'files':sorted(transformations,key=lambda row:row['path'])},indent=2))
    provenance.verify_manifest(destination)
    checked=frozen_checks(destination)
    forbidden=[];personal=[]
    for src in destination.rglob('*'):
        if not src.is_file():continue
        rel=src.relative_to(destination)
        if src.suffix.lower()in {'.sav','.rds','.zip','.pyc'}or any(p in rel.parts for p in ['private_competitors','.competitor_env','.Rlib','__pycache__']):forbidden.append(str(rel))
        if src.suffix.lower()in TEXT and provenance.private_user_path_hits(src.read_text(errors='replace')):
            personal.append(str(rel))
    if forbidden or personal:raise RuntimeError(f'Privacy/portability audit failed: {forbidden+personal}')
    manifest=dict(status='final_local_review_package',unique_new_independent_populations=800,
        earlier_phase_primary_populations=3400,counts_not_pooled_between_phases=True,
        numerical_amendment='numerical_amendment_v2.json; same density,800identical samples rerun',
        hogg_spatial_initialization='Explicit approved starts/initial-diagonal-metric amendment'if spatial_amendment.exists()else'Original frozen initialization',
        original_numerical_execution='v1 invalidated; source/freeze records retained, results excluded',
        individual_survey_records_redistributed=False,raw_sampler_or_fitting_data_redistributed=False,
        public_operational_copy_provenance='public_copy_transformations.json; original and public checksums distinguished',
        source_records=source_records,frozen_sources_checked=checked,
        author_identity_and_declarations='Require human author confirmation',journal_submission='Not submitted',
        files=[dict(path=str(p.relative_to(destination)),bytes=p.stat().st_size,sha256=sha(p))
               for p in sorted(destination.rglob('*'))if p.is_file()])
    (destination/'package_manifest.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6)as z:
        for p in sorted(destination.rglob('*')):
            if p.is_file():z.write(p,arcname='ENDES_comparadores/'+str(p.relative_to(destination)))
    with zipfile.ZipFile(archive)as z:
        if z.testzip()is not None:raise RuntimeError('Archive integrity failure')
    audit=dict(archive=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,
        archive_integrity_passed=True,privacy_audit_passed=True,frozen_code_preserved=True,
        release_gate_sha256=sha(gate_path),all_file_hashes='package_manifest.json')
    (ROOT/'validacion/competitor_package_validation.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--assemble',action='store_true')
    p.add_argument('--gate',type=Path,default=ROOT/'validacion/competitor_publication_release_gate.json')
    p.add_argument('--destination',type=Path,default=ROOT/'paquete_comparadores_publicacion')
    p.add_argument('--archive',type=Path,default=ROOT/'ENDES_comparadores_publicacion_2026-10-05.zip')
    a=p.parse_args()
    checks=frozen_checks()
    if not a.assemble:
        print(json.dumps(dict(preflight_frozen_code_passed=True,checked=len(checks),assembly_executed=False,
             final_gate_required={'scientific_review_passed':True,'all_pages_inspected':True,
                 'document_sha256':{n:'currentSHA256'for n in DOCS},'final_completion_sha256':'currentSHA256',
                 'spatial_completion_sha256':'currentSHA256','application_sensitivity_completion_sha256':'currentSHA256',
                 'hogg_spatial_numerical_amendment_sha256':'currentSHA256 if optional approved amendment is used'}),indent=2));return
    assemble(a.destination.resolve(),a.archive.resolve(),a.gate.resolve())


if __name__=='__main__':main()
