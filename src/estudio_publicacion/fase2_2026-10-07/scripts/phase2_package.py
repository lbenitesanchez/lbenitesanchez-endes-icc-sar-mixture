"""Build a distinct release after numerical and rendered-document checks."""
from pathlib import Path
import csv,datetime as dt,hashlib,json,shutil,subprocess,sys,zipfile
PHASE=Path(__file__).resolve().parents[1];STUDY=PHASE.parent
STAGE=PHASE/'revision_autores_2026-10-07'
ARCHIVE=STUDY/'ENDES_mejoras_1_2_3_2026-10-07_revision_autores.zip'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def main():
    if STAGE.exists()or ARCHIVE.exists():raise FileExistsError('Preserve previously built release')
    for name in ['phase2_results_independent_audit.json','phase2_document_visual_qa.json']:
        if not read(PHASE/'validacion'/name)['status'].startswith('PASS'):raise RuntimeError('Release gate missing:'+name)
    protocol=read(PHASE/'protocol_freeze.json')
    if sha(STUDY/'ENDES_comparadores_publicacion_2026-10-06_entrega.zip')!=protocol['previous_delivery_sha256']:
        raise RuntimeError('Previous immutable delivery changed')
    STAGE.mkdir()
    copied={}
    def add(source,destination=None):
        source=Path(source)
        if not source.is_file():raise FileNotFoundError(source)
        relative=Path(destination)if destination else Path('estudio_publicacion')/source.relative_to(STUDY)
        target=STAGE/relative
        if str(relative)in copied:return
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        copied[str(relative)]={'path':str(relative),'sha256':sha(target),'bytes':target.stat().st_size}
    # Portable statistical sources and original editorial helpers, no binaries,
    # installed environments, tokens, caches or author clone contact material.
    for f in (STUDY/'scripts').rglob('*'):
        if '__pycache__'not in f.parts and f.suffix in {'.py','.R','.stan','.json','.md'}:
            if 'compiled_source'not in f.name:add(f)
    add(STUDY/'scripts/public_competitors/requirements_competitors.txt')
    # Historical auxiliary reference is needed to reconstruct the extraction
    # comparison and original ordering; fitting uses the corrected phase file.
    add(STUDY/'datos/processed/aux_all_districts.csv')
    for f in (PHASE/'scripts').rglob('*'):
        if '__pycache__'not in f.parts and 'validacion_models'not in f.parts and f.suffix in {'.py','.R','.stan','.json','.txt'}:
            add(f)
    for key in protocol['source_sha256']:
        source=PHASE/key
        if not source.exists():source=STUDY/key
        add(source)
    for filename in ['references.json','competitor_sections.json','sections.json','competitor_reference_style.json','competitor_document_inputs.json']:
        add(STUDY/'manuscrito'/filename)
    for f in (PHASE/'manuscrito').iterdir():
        if f.suffix in {'.docx','.json'}:add(f)
    for f in (PHASE/'figuras').iterdir():
        if f.suffix in {'.png','.svg'}:add(f)
    add(PHASE/'protocol_freeze.json')
    add(PHASE/'study_decisions_before_external_responses.json')
    add(PHASE/'fuentes/external/province_partition_before_responses.csv')
    for f in (PHASE/'datos/geografia').iterdir():
        if f.suffix in {'.csv','.json','.txt'}:add(f)
    # Census tables and provenance only; complete municipal directories have
    # names/contact details and are represented by official URLs and hashes.
    for folder in ['metodologia','geografia','external']:
        for f in (PHASE/'fuentes'/folder).rglob('*'):
            if not f.is_file():continue
            # Complete extracted directory text contains municipal contacts,
            # just like the original PDF/XLSX. Keep derived codes and provenance.
            if f.suffix=='.txt'and 'directorio'in f.name.lower():continue
            if f.suffix in {'.json','.bib','.txt'}:add(f)
            elif f.suffix=='.html'and ('census'in f.name or 'redatam'in f.name):add(f)
    for f in (PHASE/'validacion').rglob('*.json'):
        if 'diseno_pre_freeze_history'not in f.parts and 'rendered'not in f.parts:add(f)
    result_names={
       'externa_final':['completion.json','summary.csv','contrasts_Holm8.csv'],
       'externa_temporal_2019_baseline':['predictions.csv','status.json'],
       'externa_temporal_2019_hogg':['predictions.csv','status_recovered.json','serialization_recovery_manifest.json'],
       'externa_temporal_2019_designcut8':['district_predictions.csv','completion.json'],
       'externa_territorial_2024_baseline':['predictions.csv','status.json'],
       'externa_territorial_2024_designcut8':['predictions.csv','completion.json'],
       'application2025_baseline':['predictions.csv','status.json'],
       'application2025_designcut8':['district_predictions.csv','completion.json'],
       'diseno/2019_full_official/cut16':['district_predictions.csv','completion.json','cut8_comparison.json']}
    hogg=PHASE/'resultados/externa_territorial_2024_hogg'
    result_names['externa_territorial_2024_hogg']=['predictions.csv',('status.json'if(hogg/'status.json').exists()else'status_recovered.json')]
    if (hogg/'serialization_recovery_manifest.json').exists():result_names['externa_territorial_2024_hogg'].append('serialization_recovery_manifest.json')
    for folder,names in result_names.items():
        for name in names:add(PHASE/'resultados'/folder/name)
    add(PHASE/'resultados/application2025_summary.csv')
    for f in (PHASE/'resultados/design_simulation/final').iterdir():
        if f.suffix in {'.csv','.json'}:add(f)
    for context in ('dense6','sparse2'):
        for rep in range(20):
            world=PHASE/f'resultados/design_simulation/{context}/rep_{rep:03d}'
            for name in ['completion.json','predictions.csv']:add(world/name)
            for f in world.glob('*/fit_status.json'):add(f)
            for f in world.glob('*/completion.json'):add(f)
            for branch in ('synthetic_design','synthetic_design_evaluation'):
                for f in (PHASE/f'datos/{branch}/{context}/rep_{rep:03d}').iterdir():add(f)
    oldresults=STUDY/'resultados/resultados_comparadores/final_evaluation'
    for f in oldresults.iterdir():
        if f.suffix in {'.csv','.json'}:add(f)
    for f in (STUDY/'resultados/resultados_comparadores').glob('*manifest*.json'):add(f)
    add(STUDY/'resultados/resultados_comparadores/protocol_freeze.json')
    config=read(STUDY/'manuscrito/competitor_document_inputs.json')
    for key in ('spatial_summary_path','spatial_paired_path','spatial_completion_path'):
        add(STUDY/config[key])
    # Keep every locked status byte-identical in the archive.
    for scope in ('temporal','territorial'):
        for item in read(PHASE/f'validacion/externa_prediction_lock_{scope}.json')['predictions']:
            add(PHASE/item['path']);add(PHASE/item['status_path'])
    private_inputs=[]
    for name,value in protocol['input_sha256'].items():
        source=(STUDY/name.removeprefix('previous_study/')if name.startswith('previous_study/')else PHASE/name)
        if str(Path('estudio_publicacion')/source.relative_to(STUDY))not in copied:
            private_inputs.append({'protocol_path':name,'sha256':value,'reason':'Separate original-source acquisition required; no raw respondent file distributed.'})
    readme=('ENDES — revisión de autores, mejoras 1, 2 y 3\n\n'
        'Versión analítica y editorial de octubre de 2026. No enviada ni aceptada.\n'
        'Los cinco documentos Word están en estudio_publicacion/fase2_2026-10-07/manuscrito.\n'
        'Se verificó Statistical Journal of the IAOS, SCImago2025 Q3, categoría Statistics, Probability and Uncertainty.\n'
        'La cohorte nueva CUT8 es adversa frente a ICC en RMSE e interval score; no atribuirle ganancias del experimento previo800.\n'
        'Las ocho comparaciones externas2024 incluyen todos los signos y usan una familia Holm fija.\n\n'
        'VERIFICACIÓN SIN INSTALAR DEPENDENCIAS\n'
        'Desde esta carpeta: python3 estudio_publicacion/fase2_2026-10-07/scripts/verify_public_release.py\n'
        'Comprueba hashes de archivos,45 fuentes estadísticas congeladas y ambos locks predictivos.\n'
        'RELEASE_MANIFEST.json contiene hashes yPRIVATE_INPUTS_REQUIRED.json enumera fuentes reservadas excluidas.\n\n'
        'REPRODUCCIÓN COMPLETA\n'
        'Los modelos requieren Python3.12, NumPy, pandas, SciPy, CmdStanPy1.3 y CmdStan2.40.\n'
        'Las extracciones requieren R4.6.1, haven, survey4.5, jsonlite, digest y readxl.\n'
        'Adquiera los módulos originales oficiales INEI2019,2024y2025 por separado. Mantenga las rutas relativas declaradas y verifique sus hashes.\n'
        'Se incluyen las muestras e insumos sintéticos completos de los 40 mundos y su verdad finita agregada. La población individual completa es regenerable por semillas, pero no se archiva. Las cadenas privadas, JSON individuales y entornos instalados se excluyen.\n'
        'El auxiliar legado datos/processed/aux_all_districts.csv se conserva para reconstruir el orden y la auditoría de extracción. Contiene los siete valores previos corregidos; todos los nuevos ajustes utilizan datos/geografia/aux_verified_2017_2018.csv en la fase2.\n'
        'El protocolo y todos los archivos completados son inmutables: una nueva reproducción se ejecuta en una copia de trabajo distinta, preservando esta entrega.\n'
        'El wrapper Hogg conserva un error original al escribir arrays NumPy en JSON; el controlador operacional recupera metadata sólo tras gates/cadenas/mapas comprobados.\n'
        'Los scores externos se recalculan sólo después de ambos locks; no alterar cohortes, semillas, estados o familias por resultados.\n\n'
        'ANTES DE ENVIAR\n'
        'Los autores confirman nombre/orden, afiliaciones, contribuciones, ética y permiso de uso secundario, financiación, conflictos, exclusividad y repositorio.\n'
        'No se distribuyen registros de niños/hogares ni directorios municipales con contactos. Los mapas son predicciones en capitales legales, no polígonos de límites.\n')
    f=STAGE/'README_ES.txt';f.write_text(readme);copied[f.name]={'path':f.name,'sha256':sha(f),'bytes':f.stat().st_size}
    f=STAGE/'PRIVATE_INPUTS_REQUIRED.json';save(f,private_inputs);copied[f.name]={'path':f.name,'sha256':sha(f),'bytes':f.stat().st_size}
    for item in copied.values():
        p=STAGE/item['path']
        if p.suffix=='.csv'and '/synthetic_'not in item['path']:
            with p.open(newline='',encoding='utf-8-sig')as stream:columns=next(csv.reader(stream),[])
            forbidden={'child_id','household_id','HHID','HV001','HV021','psu_key','HC0','HC53','HC70'}
            if forbidden.intersection(columns):raise RuntimeError('Respondent columns in candidate release:'+item['path'])
    manifest={'status':'AUTHOR_REVIEW_REPRODUCIBILITY_PACKAGE_NOT_SUBMITTED','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'protocol_sha256':sha(PHASE/'protocol_freeze.json'),'files':list(sorted(copied.values(),key=lambda r:r['path'])),
        'excluded_categories':['raw surveys','household/person identifiers','real district response-score detail','sampler CSV/NPZ real responses',
                               'municipal contact directories','compiled binaries and environments','internal rendered document previews'],
        'exact_refitting_requires_original_sources':True,'author_approval_pending':True}
    save(STAGE/'RELEASE_MANIFEST.json',manifest)
    public_verifier=STAGE/'estudio_publicacion'/PHASE.name/'scripts/verify_public_release.py'
    result=subprocess.run([sys.executable,str(public_verifier)],cwd=STAGE,capture_output=True,text=True)
    if result.returncode:raise RuntimeError('Staged public verification failed:'+result.stdout+result.stderr)
    with zipfile.ZipFile(ARCHIVE,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6)as z:
        for f in sorted(STAGE.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(STAGE))
    with zipfile.ZipFile(ARCHIVE)as z:
        broken=z.testzip()
        if broken:raise RuntimeError('ZIP CRC failure:'+broken)
        for item in manifest['files']:
            if hashlib.sha256(z.read(item['path'])).hexdigest()!=item['sha256']:raise RuntimeError('ZIP manifest mismatch')
        members=len(z.namelist())
    proof={'status':'PASS_NEW_ZIP_CRC_SHA_PRIVACY_STAGED_VERIFICATION','archive':str(ARCHIVE),
        'archive_sha256':sha(ARCHIVE),'bytes':ARCHIVE.stat().st_size,'members':members,
        'public_verification':json.loads(result.stdout),'previous_archive_unchanged':True,
        'manuscript_submission_status':'not submitted; author declarations pending'}
    save(PHASE/'validacion/phase2_archive_release_validation.json',proof)
    ARCHIVE.with_suffix('.sha256').write_text(proof['archive_sha256']+'  '+ARCHIVE.name+'\n')
    print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
