"""Read the closed author-review ZIP after its successful release gate.

No extraction, mutation of the staged package, model fit, or raw-response read.
All checks use the standard library and the actual bytes inside the ZIP.
The independent proof is written once, outside the archive.
"""
from pathlib import Path, PurePosixPath
import csv
import datetime as dt
import hashlib
import io
import json
import re
import stat
import zipfile

PHASE = Path(__file__).resolve().parents[1]
STUDY = PHASE.parent
ARCHIVE = STUDY / 'ENDES_mejoras_1_2_3_2026-10-07_revision_autores.zip'
PREFIX = 'estudio_publicacion/'
PHASE_PREFIX = PREFIX + PHASE.name + '/'
FROZEN_PROTOCOL_SHA256 = 'c9d2b4eb7ad1f7acc928aa82124b226620f5a1dc170ec82371a63bbd0ec301d2'
METHODS = {'BetaPowerExisting', 'BetaDesignUncertainty', 'HoggAgeSexSpatial'}
WORDS = {
    'Cover_letter_IAOS_fase2.docx', 'Manuscript_ENDES_IAOS_fase2.docx',
    'Resultados_y_guia_de_publicacion_fase2.docx',
    'Supplement_ENDES_IAOS_fase2.docx', 'Title_page_author_confirmation_fase2.docx',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    gate_path = PHASE / 'validacion/phase2_archive_release_validation.json'
    if not gate_path.is_file():
        raise RuntimeError('Wait for the completed package release proof; no ZIP access yet.')
    gate = json.loads(gate_path.read_text())
    if gate.get('status') != 'PASS_NEW_ZIP_CRC_SHA_PRIVACY_STAGED_VERIFICATION':
        raise RuntimeError('Package release proof has not passed; no ZIP access yet.')
    if Path(gate['archive']).resolve() != ARCHIVE.resolve() or not ARCHIVE.is_file():
        raise RuntimeError('Release proof does not identify the expected existing ZIP.')
    before = sha(ARCHIVE)
    if before != gate['archive_sha256']:
        raise RuntimeError('Closed archive differs from its completed release proof.')
    destination = PHASE / 'validacion/phase2_archive_independent_audit.json'
    if destination.exists():
        raise FileExistsError('Preserve the independent archive proof.')
    checks = []

    def check(name, passed, details=None):
        checks.append({'name': name, 'passed': bool(passed), 'details': details})

    with zipfile.ZipFile(ARCHIVE, 'r') as z:
        infos = z.infolist()
        names = [f.filename for f in infos]
        members = set(names)
        check('ZIP CRC for every member', z.testzip() is None)
        check('Unique member names', len(names) == len(members))
        check('No directory or symlink entries', all(not f.is_dir() and
              not stat.S_ISLNK(f.external_attr >> 16) for f in infos))
        check('Safe relative archive paths', all(not PurePosixPath(n).is_absolute() and
              '..' not in PurePosixPath(n).parts and '\\' not in n for n in names))

        def contents(name):
            if name not in members:
                raise FileNotFoundError('Required ZIP entry: ' + name)
            return z.read(name)

        def read(name):
            return json.loads(contents(name))

        manifest = read('RELEASE_MANIFEST.json')
        entries = manifest['files']
        manifest_names = [r['path'] for r in entries]
        check('Manifest entries are unique', len(manifest_names) == len(set(manifest_names)))
        check('ZIP contains exactly the manifest files and manifest itself',
              members == set(manifest_names) | {'RELEASE_MANIFEST.json'})
        for entry in entries:
            data = contents(entry['path'])
            check('Manifest SHA/length ' + entry['path'],
                  digest(data) == entry['sha256'] and len(data) == entry['bytes'])
        check('Completed gate records actual member count and archive size',
              gate['members'] == len(names) and gate['bytes'] == ARCHIVE.stat().st_size)
        check('Review package remains not submitted',
              manifest['status'] == 'AUTHOR_REVIEW_REPRODUCIBILITY_PACKAGE_NOT_SUBMITTED' and
              manifest['author_approval_pending'] is True and
              manifest['exact_refitting_requires_original_sources'] is True)

        protocol_path = PHASE_PREFIX + 'protocol_freeze.json'
        protocol = read(protocol_path)
        protocol_sha = digest(contents(protocol_path))
        check('Global scientific protocol exact bytes', protocol_sha == FROZEN_PROTOCOL_SHA256 ==
              sha(PHASE / 'protocol_freeze.json') == manifest['protocol_sha256'])
        check('Exactly45 frozen scientific sources', len(protocol['source_sha256']) == 45)
        for name, expected in protocol['source_sha256'].items():
            key = PHASE_PREFIX + name
            if key not in members:
                key = PREFIX + name
            check('Frozen scientific source ' + name, digest(contents(key)) == expected)

        excluded = read('PRIVATE_INPUTS_REQUIRED.json')
        excluded_names = [r['protocol_path'] for r in excluded]
        excluded_map = {r['protocol_path']: r['sha256'] for r in excluded}
        check('Exactly11 unique separate-acquisition input records',
              len(excluded) == len(excluded_map) == 11)
        provided = []
        omitted = []
        for name, expected in protocol['input_sha256'].items():
            key = (PREFIX + name.removeprefix('previous_study/') if
                   name.startswith('previous_study/') else PHASE_PREFIX + name)
            if key in members:
                provided.append(name)
                check('Frozen included input ' + name, digest(contents(key)) == expected)
                check('Included frozen input absent from private requirement records', name not in excluded_map)
            else:
                omitted.append(name)
                check('Omitted input declared with exact frozen hash ' + name,
                      excluded_map.get(name) == expected)
        check('Exactly6 public inputs and11 omitted inputs with no extra records',
              len(provided) == 6 and len(omitted) == 11 and
              set(omitted) == set(excluded_names))

        lock_proofs = []
        for scope in ('temporal', 'territorial'):
            key = PHASE_PREFIX + f'validacion/externa_prediction_lock_{scope}.json'
            lock = read(key)
            check(scope + ' lock exact local bytes/protocol',
                  digest(contents(key)) == sha(PHASE / f'validacion/externa_prediction_lock_{scope}.json') and
                  lock['protocol_sha256'] == protocol_sha)
            check(scope + ' lock has exactly three specified methods',
                  len(lock['predictions']) == 3 and
                  {r['method'] for r in lock['predictions']} == METHODS)
            for entry in lock['predictions']:
                for field, hash_key in [('path', 'sha256'), ('status_path', 'status_sha256')]:
                    data = contents(PHASE_PREFIX + entry[field])
                    check(scope + ' locked ' + field + ' ' + entry['method'],
                          digest(data) == entry[hash_key] == sha(PHASE / entry[field]))
                check(scope + ' retained successful status ' + entry['method'],
                      read(PHASE_PREFIX + entry['status_path']).get('success') is True)
            lock_proofs.append({'scope': scope, 'lock_sha256': digest(contents(key)),
                                'methods': sorted(METHODS)})

        word_names = {PurePosixPath(n).name for n in members
                      if n.startswith(PHASE_PREFIX + 'manuscrito/') and n.endswith('.docx')}
        check('Exactly five expected Word documents', word_names == WORDS)
        word_proofs = []
        for name in sorted(WORDS):
            data = contents(PHASE_PREFIX + 'manuscrito/' + name)
            check('Exact final Word bytes ' + name, digest(data) == sha(PHASE / 'manuscrito' / name))
            with zipfile.ZipFile(io.BytesIO(data)) as doc:
                check('Word internal CRC/document XML ' + name,
                      doc.testzip() is None and 'word/document.xml' in doc.namelist())
            word_proofs.append({'name': name, 'sha256': digest(data), 'bytes': len(data)})
        visual_key = PHASE_PREFIX + 'validacion/phase2_document_visual_qa.json'
        check('Visual QA proof retained with exact bytes and PASS',
              read(visual_key)['status'].startswith('PASS') and
              digest(contents(visual_key)) == sha(PHASE / 'validacion/phase2_document_visual_qa.json'))

        conditional_count = 0
        synthetic_samples = 0
        tasks = ['baseline', 'point'] + [f'cut_state_{k:02d}' for k in range(8)]
        for context in ('dense6', 'sparse2'):
            for rep in range(20):
                world = PHASE_PREFIX + f'resultados/design_simulation/{context}/rep_{rep:03d}/'
                comp = read(world + 'completion.json')
                check(f'Completed synthetic world {context}/{rep}', comp['complete'] is True and
                      comp['protocol_sha256'] == protocol_sha and
                      digest(contents(world + 'predictions.csv')) == comp['predictions_sha256'])
                check(f'All ten synthetic task names {context}/{rep}',
                      {PurePosixPath(n).parent.name for n in members if n.startswith(world) and
                       n.count('/') == world.count('/') + 1 and n.endswith('/completion.json')} == set(tasks))
                for task, world_status in zip(tasks, comp['conditional_tasks']):
                    conditional_count += 1
                    task_key = world + task + '/completion.json'
                    status_key = world + task + '/fit_status.json'
                    task_completion = read(task_key)
                    status = read(status_key)
                    check(f'Synthetic task completion/diagnostics {context}/{rep}/{task}',
                          digest(contents(status_key)) == task_completion['sampler_status_sha256'] and
                          status == world_status and task_completion['success'] == status['success'] and
                          digest(contents(task_key)) == sha(PHASE / task_key.removeprefix(PHASE_PREFIX)))
                data_root = PHASE_PREFIX + f'datos/synthetic_design/{context}/rep_{rep:03d}/'
                generation = read(data_root + 'generation.json')
                for name, hash_key in [('sample.csv', 'sample_sha256'), ('frame.csv', 'frame_sha256'),
                                       ('known_auxiliary.npz', 'aux_sha256')]:
                    check(f'Synthetic input hash {context}/{rep}/{name}',
                          digest(contents(data_root + name)) == generation[hash_key])
                truth_key = PHASE_PREFIX + f'datos/synthetic_design_evaluation/{context}/rep_{rep:03d}/realized_population_truth.npz'
                check(f'Finite population truth hash {context}/{rep}',
                      digest(contents(truth_key)) == generation['realized_truth_sha256'])
                sampled = list(csv.DictReader(io.StringIO(contents(data_root + 'sample.csv').decode())))
                frame = list(csv.DictReader(io.StringIO(contents(data_root + 'frame.csv').decode())))
                check(f'Synthetic sample and selected frame counts {context}/{rep}',
                      len(sampled) == generation['measured_sample_children'] and
                      len(frame) == generation['complete_frame_PSUs'])
                synthetic_samples += 1
        check('Exactly400 synthetic task completions and40 samples inspected',
              conditional_count == 400 and synthetic_samples == 40)

        raw_flags = []
        csv_flags = []
        contact_flags = []
        json_record_flags = []
        forbidden = {'child_id', 'household_id', 'person_id', 'HHID', 'HV001', 'HV021', 'HV022',
                     'psu_key', 'HC0', 'HC1', 'HC53', 'HC57A', 'HC57B', 'HC70',
                     'y_anemia', 'y_anemia_who2024', 'y_stunting', 'weight_anemia', 'weight_stunting'}
        id_keys = {'child_id', 'household_id', 'person_id', 'HHID', 'HV001', 'HV021',
                   'HC0', 'HC53', 'HC57A', 'HC57B', 'HC70'}

        def individual_values(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in id_keys and not isinstance(item, (int, float, bool)):
                        yield key
                    yield from individual_values(item)
            elif isinstance(value, list):
                for item in value:
                    yield from individual_values(item)

        inspected_non_synthetic_csv = 0
        inspected_text_metadata = 0
        for name in names:
            data = contents(name)
            p = PurePosixPath(name)
            suffix = p.suffix.lower()
            synthetic_data = bool(re.search(r'/datos/synthetic_design(?:_evaluation)?/(dense6|sparse2)/rep_\d{3}/', name))
            if (suffix in {'.sav', '.dta', '.por', '.pdf', '.xlsx', '.xls', '.zip', '.exe', '.dylib', '.so'} or
                '/datos/private_' in name or '/validacion_models/' in name or
                '/.competitor_env/' in name or '/.Rlib/' in name or '/__pycache__/' in name or
                '_compiled_source.json' in name or (suffix == '.npz' and not synthetic_data) or
                (suffix == '.txt' and 'directorio' in p.name.lower()) or
                data[:4] in {b'\x7fELF', b'\xfe\xed\xfa\xce', b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa\xcf', b'\xcf\xfa\xed\xfe'}):
                raw_flags.append(name)
            if suffix == '.csv' and not synthetic_data:
                inspected_non_synthetic_csv += 1
                columns = next(csv.reader(io.StringIO(data.decode('utf-8-sig'))), [])
                if forbidden.intersection(columns) or {'lp__', 'divergent__', 'treedepth__'}.intersection(columns):
                    csv_flags.append(name)
            if suffix in {'.txt', '.html', '.bib', '.json'} and ('/fuentes/' in name or '/validacion/' in name):
                inspected_text_metadata += 1
                text = data.decode('utf-8', errors='replace')
                if re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', text) or any(
                        x in text for x in ('Teléfono', 'TELÉFONO', 'Correo electrónico', 'CORREO ELECTRÓNICO')):
                    contact_flags.append(name)
            if suffix == '.json' and list(individual_values(json.loads(data))):
                json_record_flags.append(name)
        check('No selected raw/private/sampler/contact/executable paths or binary signatures', not raw_flags, raw_flags)
        check('No respondent or sampler CSV columns outside generated synthetic input', not csv_flags, csv_flags)
        check('No municipal contact patterns in source/validation text metadata', not contact_flags, contact_flags)
        check('No individual-record keyed arrays/strings in JSON', not json_record_flags, json_record_flags)
        check('Original dependency requirements and historical auxiliary-only reference included',
              PREFIX + 'scripts/public_competitors/requirements_competitors.txt' in members and
              PREFIX + 'datos/processed/aux_all_districts.csv' in members)
        contrasts = list(csv.DictReader(io.StringIO(contents(PHASE_PREFIX +
                        'resultados/externa_final/contrasts_Holm8.csv').decode())))
        check('Exactly specified eight external contrasts without duplicated cells', len(contrasts) == 8 and
              {(r['scope'], r['outcome'], r['reference']) for r in contrasts} ==
              {(s, o, m) for s in ('temporal', 'territorial') for o in ('anemia', 'stunting')
               for m in ('BetaPowerExisting', 'HoggAgeSexSpatial')})
        check('All adverse external findings retained',
              sum(r['candidate_significantly_better'].lower() == 'true' for r in contrasts) == 0 and
              sum(r['candidate_significantly_worse'].lower() == 'true' for r in contrasts) == 4)
        for key in ['phase2_simulation_independent_audit.json', 'phase2_external_independent_audit.json',
                    'phase2_results_independent_audit.json', 'geografia_public_release_static_review.json']:
            target = PHASE_PREFIX + 'validacion/' + key
            check('Independent scientific/static proof exact bytes ' + key,
                  digest(contents(target)) == sha(PHASE / 'validacion' / key) and
                  read(target)['status'].startswith('PASS'))
        manifest_sha = digest(contents('RELEASE_MANIFEST.json'))

    after = sha(ARCHIVE)
    check('Archive unchanged while read and equal to release SHA', before == after == gate['archive_sha256'])
    previous = STUDY / 'ENDES_comparadores_publicacion_2026-10-06_entrega.zip'
    check('Original previous delivery archive unchanged', sha(previous) == protocol['previous_delivery_sha256'])
    sidecar = ARCHIVE.with_suffix('.sha256')
    check('Published SHA sidecar identifies this archive', sidecar.is_file() and
          sidecar.read_text().split()[0] == before and ARCHIVE.name in sidecar.read_text())
    result = {
        'status': 'PASS_INDEPENDENT_CLOSED_ZIP_CRC_SHA_FROZEN_INPUTS_LOCKS_WORD_PRIVACY' if
                  all(c['passed'] for c in checks) else 'FAIL_INDEPENDENT_CLOSED_ZIP_AUDIT',
        'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'audit_source_sha256': sha(__file__), 'archive': str(ARCHIVE), 'archive_sha256': before,
        'archive_bytes': ARCHIVE.stat().st_size, 'archive_members': len(names),
        'manifest_sha256': manifest_sha, 'manifest_files': len(entries),
        'release_gate_sha256': sha(gate_path), 'protocol_sha256': protocol_sha,
        'frozen_scientific_sources': 45, 'public_frozen_inputs': provided,
        'separate_acquisition_input_count': len(omitted), 'prediction_locks': lock_proofs,
        'Word_documents': word_proofs, 'synthetic_worlds': synthetic_samples,
        'synthetic_conditional_completion_count': conditional_count,
        'privacy_schema_and_text_checks': {'non_synthetic_csv_files': inspected_non_synthetic_csv,
                                          'source_validation_text_files': inspected_text_metadata},
        'checks': checks, 'failure_count': sum(not c['passed'] for c in checks),
        'scope': 'Actual closed ZIP integrity, exact source/input/lock/document bytes and inspected archive privacy; no stage or archive mutation.',
        'limitations': ['No fresh-environment model refitting or full raw-survey reproduction was performed.',
                       'Word bytes and retained visual-QA proof were checked; this auditor did not perform new visual rendering.',
                       'CSV schemas, JSON value types, binary signatures and selected text patterns were inspected; no formal anonymisation theorem is asserted.',
                       'Exact refitting still requires separately acquired original sources and declared local dependencies.',
                       'Hash integrity is distinct from author approval, submission and journal acceptance.'],
        'mutations': ['This independent proof only; staged release and ZIP remain untouched.'],
    }
    with destination.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'archive_sha256': before, 'checks': len(checks),
                      'failure_count': result['failure_count'],
                      'failures': [c for c in checks if not c['passed']]}))
    if result['failure_count']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
