"""Independently read the exact final ZIP; never assemble, fit, or edit documents.

The archive, indexed contents, present study artifacts, frozen kernels, explicit
public path-redaction ledger, and metadata-only public guards are checked. All
execution occurs in a disposable extraction after its source hashes are checked.
Synthetic individual records are permitted only in the locked simulation inputs.
"""
from pathlib import Path, PurePosixPath, PureWindowsPath
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'ENDES_comparadores/'
OUT = 'resultados/resultados_comparadores/'
DOCS = ['Manuscript_ENDES_IAOS.docx', 'Supplement_ENDES_IAOS.docx',
        'Cover_letter_IAOS.docx', 'Title_page_author_confirmation.docx',
        'Resultados_y_guia_de_publicacion.docx']
TEXT = {'.py', '.r', '.stan', '.md', '.txt', '.json', '.jsonl', '.csv', '.sh', '.svg', '.xml'}
FORBIDDEN_SUFFIXES = {'.sav', '.zsav', '.dta', '.rds', '.rda', '.rdata', '.pyc', '.pyo',
    '.so', '.dylib', '.dll', '.exe', '.o', '.a', '.lib', '.whl', '.tar', '.gz', '.tgz', '.zip'}
FORBIDDEN_PARTS = {'private_competitors', '.competitor_env', '.rlib', '__pycache__',
    '.git', '.cmdstan', 'site-packages', 'node_modules', 'hogg_author_repository'}
PRIVATE_COLUMNS = {'child_id', 'household_id', 'person_id', 'hhid', 'hh_id', 'unit_id',
    'caseid', 'hvidx', 'bidx', 'y_anemia', 'y_stunting', 'weight_anemia', 'weight_stunting'}
BINARY_MAGIC = {b'\x7fELF', b'\xfe\xed\xfa\xce', b'\xce\xfa\xed\xfe',
    b'\xfe\xed\xfa\xcf', b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca'}
HISTORICAL_DISTRICT_CSV = re.compile(
    r'resultados/aplicacion_(?:2019|2025)_(?:old|who2024)/'
    r'(?:district_predictions_(?:FH|MFH|SFH|MSFH|SELECTED)|model_input_all_districts_used)\.csv')


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value):
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    return bool(value) and not (posix.is_absolute() or windows.drive or windows.root
        or '..' in posix.parts or '..' in windows.parts or '\\' in value or '\x00' in value
        or posix.as_posix() != value or value == '.')


def private_paths(text):
    value = re.sub(r'https?://[^\s\"\'<>]+', '', text).replace('\\\\', '\\')
    return bool(re.search(r'/(?:Users|home)/[^/\s\"\'<>]+/|[A-Za-z]:\\Users\\[^\\\s\"\'<>]+\\', value))


def security_findings(relative, data, reviewed_base=None):
    """Inspect actual member bytes and CSV schema, not filename assertions alone."""
    path = PurePosixPath(relative)
    lower_parts = {part.lower() for part in path.parts}
    findings = []
    if path.suffix.lower() in FORBIDDEN_SUFFIXES or lower_parts & FORBIDDEN_PARTS:
        findings.append('private_data_library_clone_or_archive')
    if data[:4] in BINARY_MAGIC or data[:2] == b'MZ':
        findings.append('compiled_executable_magic')
    if re.search(r'(^|/)(?:posterior_draws|stage[12]_data|stan_data|fitting_data|data\.Rdump|analytic_private_)', relative, re.I):
        findings.append('raw_sampler_or_private_fitting_input')
    # Earlier Gaussian fit arrays are district-level summaries, not raw chains.
    synthetic_array = (relative.startswith(OUT+'inputs/') or bool(re.fullmatch(
        r'resultados/simulaciones/inputs(?:_finite_v2|_sparse_v2)?/[^/]+/rep_\d+\.npz', relative)))
    district_array = bool(re.fullmatch(
        r'resultados/aplicacion_(?:2019|2025)_(?:old|who2024)/(?:fit_(?:FH|MFH|SFH|MSFH)\.npz|model_input_values_used\.npz|fixed_graph_spectral_basis_q40\.npy)', relative))
    if path.suffix.lower() in {'.npz', '.npy'} and not (synthetic_array or district_array):
        findings.append('binary_array_outside_approved_synthetic_or_district_inputs')
    if path.suffix.lower() in TEXT:
        text = data.decode('utf-8', errors='replace')
        if private_paths(text):
            findings.append('private_workstation_path')
        if path.suffix.lower() == '.csv':
            reader = csv.DictReader(io.StringIO(text))
            header = reader.fieldnames or []
            names = {column.strip().lstrip('\ufeff').lower() for column in header}
            strong_private = PRIVATE_COLUMNS-{'y_anemia', 'y_stunting'}
            if strong_private & names and not relative.startswith(OUT+'inputs/'):
                findings.append('individual_survey_schema_outside_synthetic_inputs')
            if {'y_anemia', 'y_stunting'} & names and not relative.startswith(OUT+'inputs/'):
                aggregate = ({'n_stunting', 'var_stunting', 'n_eligible', 'year', 'level', 'domain', 'universe'} <= names)
                aggregate = aggregate and (relative.startswith('datos/processed/')
                    or relative.startswith('resultados/aplicacion_')
                    or relative == 'resultados/simulaciones/source_snapshots/sparse_v2/areas_2025_6_59.csv')
                if aggregate:
                    rows = list(reader)
                    keys = [(r['year'], r['level'], r['domain'], r['universe']) for r in rows]
                    aggregate = len(keys) == len(set(keys))
                    # Exactly the24 demonstrated historical district tables:
                    # unsampled rows may lack survey domain/year, but retain
                    # unique geographic codes. No other CSV gets this exception.
                    if not aggregate and HISTORICAL_DISTRICT_CSV.fullmatch(relative) and reviewed_base is not None:
                        source = Path(reviewed_base)/relative
                        forbidden_unit = strong_private | {'weight', 'sampling_weight',
                            'age_months', 'sex', 'psu', 'stratum', 'hv001', 'hv002', 'hv003',
                            'v001', 'v002', 'v003', 'hw1', 'hw70', 'hc57'}
                        codes = [r.get('ubigeo', '') for r in rows]
                        aggregate = (source.is_file() and sha(source) == sha_bytes(data)
                            and not forbidden_unit & names and bool(codes)
                            and len(codes) == len(set(codes))
                            and all(re.fullmatch(r'[0-9]{6}', code) for code in codes))
                if not aggregate:
                    findings.append('outcome_records_without_unique_aggregate_domain_schema')
            if any(c in {'lp__', 'treedepth__', 'stepsize__', 'divergent__'} for c in header):
                findings.append('raw_sampler_chain')
    if path.suffix.lower() == '.docx':
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as document:
                if document.testzip() is not None or 'word/document.xml' not in document.namelist():
                    findings.append('invalid_docx_container')
                if any('vbaproject' in n.lower() or '/embeddings/' in n.lower() for n in document.namelist()):
                    findings.append('embedded_document_code_or_object')
        except zipfile.BadZipFile:
            findings.append('invalid_docx_container')
    return findings


def inventory(archive, extracted=None, reviewed_base=None):
    """Fail closed before extraction on unsafe names, links, duplicate entries."""
    contents, issues, privacy = {}, [], []
    with zipfile.ZipFile(archive) as container:
        members = container.infolist()
        names = [info.filename for info in members]
        if len(names) != len(set(names)) or len(names) != len(set(n.casefold() for n in names)):
            raise ValueError('Duplicate or case-insensitively colliding ZIP paths')
        for info in members:
            if info.is_dir():
                raise ValueError('Final ZIP inventory must contain indexed files only')
            name = info.filename
            if not name.startswith(PREFIX) or not safe_relative(name[len(PREFIX):]):
                raise ValueError('Unsafe path or wrong archive root: '+name)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or info.flag_bits & 1:
                raise ValueError('Symlink or encrypted ZIP entry: '+name)
        corrupt = container.testzip()
        if corrupt is not None:
            raise ValueError('ZIP CRC failure: '+corrupt)
        for info in members:
            relative = info.filename[len(PREFIX):]
            data = container.read(info)
            contents[relative] = {'bytes': len(data), 'sha256': sha_bytes(data)}
            privacy.extend({'path': relative, 'finding': finding}
                           for finding in security_findings(relative, data, reviewed_base))
            if extracted is not None:
                target = Path(extracted)/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        if 'package_manifest.json' not in contents:
            raise ValueError('Missing package_manifest.json')
        manifest = json.loads(container.read(PREFIX+'package_manifest.json'))
    records = manifest.get('files', [])
    indexed = [row.get('path', '') for row in records]
    if len(indexed) != len(set(indexed)) or any(not safe_relative(p) for p in indexed):
        issues.append('unsafe_or_duplicate_manifest_paths')
    if set(indexed) != set(contents)-{'package_manifest.json'}:
        issues.append('manifest_inventory_not_exact')
    for row in records:
        if contents.get(row.get('path')) != {'bytes': row.get('bytes'), 'sha256': row.get('sha256')}:
            issues.append('manifest_file_mismatch:'+str(row.get('path')))
    return {'manifest': manifest, 'contents': contents, 'issues': issues,
            'privacy_findings': privacy, 'archive_crc_passed': True}


class Review:
    def __init__(self):
        self.checks = []

    def check(self, name, condition, detail=None):
        self.checks.append({'check': name, 'passed': bool(condition), 'detail': detail})

    def passed(self):
        return all(row['passed'] for row in self.checks)


def verify_transformation(root, package, entry):
    relative = entry['source_relative_path']
    if not safe_relative(relative) or not safe_relative(entry['path']):
        return False
    source, public = root/relative, package/entry['path']
    if not source.is_file() or not public.is_file():
        return False
    if sha(source) != entry['original_sha256'] or sha(public) != entry['public_sha256']:
        return False
    expected = source.read_text().replace(str(root)+'/', '').replace(str(Path.home())+'/', 'local_user/')
    return expected.encode('utf-8') == public.read_bytes()


def actual_current_checks(root, package, inv, review):
    manifest = inv['manifest']
    review.check('manifest phase and publication status',
        manifest.get('unique_new_independent_populations') == 800
        and manifest.get('earlier_phase_primary_populations') == 3400
        and manifest.get('counts_not_pooled_between_phases') is True
        and manifest.get('individual_survey_records_redistributed') is False
        and manifest.get('raw_sampler_or_fitting_data_redistributed') is False
        and manifest.get('journal_submission') == 'Not submitted')
    ledger = json.loads((package/'public_copy_transformations.json').read_text())
    transformed = {row['path']: row for row in ledger['files']}
    review.check('unique transformation paths', len(transformed) == len(ledger['files']))
    for row in ledger['files']:
        review.check('exact declared path redaction '+row['path'], verify_transformation(root, package, row))
    def current(relative, expected=None):
        public, source = package/relative, root/relative
        actual = sha(public) if public.is_file() else None
        wanted = expected or (sha(source) if source.is_file() else None)
        if actual == wanted and wanted is not None:
            okay, mode = True, 'exact_current_bytes'
        else:
            entry = transformed.get(relative)
            okay = bool(entry and entry['original_sha256'] == wanted
                        and entry['public_sha256'] == actual and verify_transformation(root, package, entry))
            mode = 'verified_explicit_path_redaction' if okay else 'mismatch'
        review.check('current artifact '+relative, okay, {'original_sha256': wanted,
                     'public_sha256': actual, 'mode': mode})
    gate_relative = 'validacion/competitor_publication_release_gate.json'
    current(gate_relative)
    gate = json.loads((package/gate_relative).read_text())
    review.check('visual and scientific release gate', gate.get('all_pages_inspected') is True
        and gate.get('scientific_review_passed') is True)
    for name in DOCS:
        current('manuscrito/'+name, gate['document_sha256'][name])
        review.check('current local DOCX '+name, sha(root/'manuscrito'/name) == gate['document_sha256'][name])
    for row in gate['evidence']:
        review.check('safe gate evidence '+row['path'], safe_relative(row['path']))
        if safe_relative(row['path']):
            current(row['path'], row['sha256'])
    mapping = {'final_completion_sha256': OUT+'final_evaluation/final_completion.json',
        'spatial_completion_sha256': OUT+'spatial_evaluation/completion.json',
        'application_sensitivity_completion_sha256': OUT+'application_sensitivity_v2/completion.json',
        'hogg_spatial_numerical_amendment_sha256': 'resultados/competidores_hogg/hogg_spatial_numerical_amendment.json',
        'builder_sha256': 'scripts/12_competitor_documents.py',
        'build_manifest_sha256': 'manuscrito/competitor_manuscript_build_manifest.json'}
    for key, relative in mapping.items():
        current(relative, gate[key])
    kernel_manifest = json.loads((root/(OUT+'source_snapshots/numerical_v2_master/source_manifest.json')).read_text())
    protocol = json.loads((package/(OUT+'protocol_freeze.json')).read_text())
    review.check('exactly16 frozen kernels in same protocol', len(kernel_manifest['source_sha256']) == 16
                 and protocol['source_sha256'] == kernel_manifest['source_sha256'])
    for name, expected in kernel_manifest['source_sha256'].items():
        current('scripts/models.py' if name == 'models.py' else 'scripts/competitors/'+name, expected)
    requested = {(scenario, replication) for scenario, replications in protocol['independent_seed_ids'].items()
                 for replication in replications}
    review.check('800 fixed synthetic population identifiers', len(requested) == 800)
    expected_inputs = set()
    for scenario, replication in sorted(requested):
        relative = OUT+f'inputs/{scenario}/rep_{replication:05d}'
        metadata = json.loads((package/relative/'manifest.json').read_text())
        review.check('synthetic-only manifest '+relative, metadata.get('synthetic') is True
            and metadata['scenario'] == scenario and metadata['replication'] == replication
            and metadata.get('truth_available_to_estimators') is False)
        expected_inputs.update(str(f.relative_to(root)) for f in (root/relative).rglob('*') if f.is_file())
    actual_inputs = {p for p in inv['contents'] if p.startswith(OUT+'inputs/')}
    review.check('complete fixed synthetic input inventory', expected_inputs == actual_inputs)
    for relative in sorted(expected_inputs):
        current(relative)
    folders = ['final_evaluation', 'application_comparison_final_v1', 'spatial_evaluation', 'boundary_audit']
    for folder in folders:
        source_folder = root/(OUT+folder)
        expected_paths = {str(f.relative_to(root)) for f in source_folder.iterdir()
                          if f.is_file() and f.suffix in {'.csv', '.json'}}
        actual_paths = {p for p in inv['contents'] if p.startswith(OUT+folder+'/')}
        review.check('complete current aggregate folder '+folder, expected_paths == actual_paths)
        for relative in sorted(expected_paths):
            current(relative)
    figures = {str(f.relative_to(root)) for f in (root/'figuras').glob('competitor*')
               if f.is_file() and f.suffix in {'.png', '.svg', '.pdf'}}
    public_figures = {p for p in inv['contents'] if p.startswith('figuras/competitor')}
    review.check('complete current comparator figures', figures == public_figures)
    for relative in sorted(figures):
        current(relative)
    readme = package/'README_COMPETIDORES_ES_EN.md'
    source_readme = root/'scripts/public_competitors/README_COMPETIDORES_ES_EN.md'
    review.check('current corrected README bytes', readme.is_file() and sha(readme) == sha(source_readme))
    text = readme.read_text()
    review.check('README19 simulation20 application25 references',
                 '19 etapas de' in text and '20 etapas de aplicación' in text and '25 referencias' in text)
    source_records = manifest['source_records']
    review.check('all source records agree with inventory', all(
        inv['contents'].get(row['path'], {}).get('sha256') == row['public_sha256']
        and (row.get('adapted') or row['original_sha256'] == row['public_sha256'])
        for row in source_records))
    for relative in ['scripts/14_package_competitors.py',
        'scripts/public_competitors/public_copy_provenance.py',
        'scripts/public_competitors/reproduce.py', 'codigo/verify_aggregate_inputs.py',
        'validacion/competitor_manuscript_final_word_check.py']:
        original = root/relative if (root/relative).exists() else root/'paquete_publicacion'/relative
        review.check('trusted public guard source '+relative, (package/relative).is_file()
                     and sha(package/relative) == sha(original))
    # The inherited binary arrays must be the same reviewed district aggregates
    # or archived synthetic inputs. District shapes and keys are read explicitly.
    import numpy as np
    district_arrays = []
    for relative in inv['contents']:
        if relative.endswith(('.npz', '.npy')) and not relative.startswith(OUT+'inputs/'):
            original = root/'paquete_publicacion'/relative
            review.check('inherited reviewed array bytes '+relative, original.is_file()
                         and sha(package/relative) == sha(original))
            if relative.startswith('resultados/aplicacion_'):
                path = package/relative
                if path.suffix == '.npz':
                    with np.load(path, allow_pickle=False) as arrays:
                        shape_ok = all(not arrays[k].dtype.hasobject and all(d <= 1874 for d in arrays[k].shape)
                            and not re.search(r'draw|chain|child|household|caseid', k, re.I) for k in arrays.files)
                        district_arrays.append({'path': relative, 'keys': list(arrays.files),
                            'shapes': {k: list(arrays[k].shape) for k in arrays.files}})
                else:
                    array = np.load(path, allow_pickle=False)
                    shape_ok = not array.dtype.hasobject and array.shape == (1874, 40)
                    district_arrays.append({'path': relative, 'shape': list(array.shape)})
                review.check('district-only binary array schema '+relative, shape_ok)
    review.check('earlier district arrays explicitly inspected', len(district_arrays) > 0, district_arrays)
    historical_csv = []
    for relative in inv['contents']:
        if HISTORICAL_DISTRICT_CSV.fullmatch(relative):
            with (package/relative).open() as stream:
                rows = list(csv.DictReader(stream))
            codes = [r['ubigeo'] for r in rows]
            source = root/'paquete_publicacion'/relative
            okay = len(rows) == len(set(codes)) == 1874 and sha(source) == sha(package/relative)
            historical_csv.append({'path': relative, 'rows': len(rows), 'unique_ubigeo': len(set(codes)),
                                   'same_reviewed_BASE_sha256': sha(source) == sha(package/relative)})
            review.check('exact historical district CSV scope '+relative, okay)
    review.check('all24 historical district CSV checked', len(historical_csv) == 24, historical_csv)
    return gate


PORTABLE_CODE = r'''
import importlib.util,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);sys.dont_write_bytecode=True
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
assembly=load('public_zip_assembly_reader',root/'scripts/14_package_competitors.py')
checked=assembly.frozen_checks(root)
bridge=load('public_zip_path_reader',root/'scripts/public_competitors/public_copy_provenance.py')
provenance=bridge.verify_manifest(root)
contract=json.loads((root/'resultados/resultados_comparadores/spatial_result_contract.json').read_text())
reference=bridge.verify_public_reference(root,contract['task_file'],contract['task_sha256'])
aggregates=load('public_zip_aggregate_reader',root/'codigo/verify_aggregate_inputs.py');aggregates.main()
words=load('public_zip_word_reader',root/'validacion/competitor_manuscript_final_word_check.py');words.main()
word_reports=list((root/'validacion/independent_final_word_checks').glob('*.json'))
word_report=[json.loads(p.read_text()) for p in word_reports if not p.name.endswith('_documents.json')]
assert len(word_report)==1 and word_report[0]['status']=='PASS', 'Portable Word reader did not pass'
result={'frozen_checks':checked,'provenance':provenance,'spatial_reference':reference,
 'aggregate_hash_reader_passed':True,'word_reader_status':word_report[0]['status'],
 'word_reader_checks':len(word_report[0]['checks']),'fits_executed':0,
 'document_builds_or_renders':0,'network_requests':0}
pathlib.Path(sys.argv[2]).write_text(json.dumps(result,indent=2))
print(json.dumps({'portable_guards_passed':True,'frozen_checks':len(checked),'word_checks':len(word_report[0]['checks'])}))
'''


def audit(archive, root, report_path):
    review, archive = Review(), Path(archive).resolve()
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'REQUIRED_CORRECTION',
        'scope': 'Exact completed ZIP, present artifact hashes, privacy/integrity and portable read-only guards; no fitting, assembly or document modification.',
        'auditor_sha256': sha(__file__), 'archive': str(archive), 'archive_sha256': sha(archive),
        'archive_bytes': archive.stat().st_size, 'fits_executed': 0, 'documents_modified': 0}
    report['synthetic_fixture_passes_used_as_archive_evidence'] = False
    report['external_auditor_classification_repair'] = {
        'scope': 'Only24 historical district CSV, exact reviewed BASE bytes and aggregate schema with unique six-digit UBIGEO; no unit IDs/weights/PSU/age/sex exemption.',
        'first_attempt': 'validacion/final_zip_auditor_history/attempt_1_before_missing_domain_classification_repair',
        'reason': 'Unobserved districts may lack survey domain/year fields; first reader produced false privacy positives.',
        'ZIP_modified': False, 'statistical_sources_or_documents_modified': False,
        'archive_auditor_source_is_previous_snapshot': None}
    try:
        assembly_record = json.loads((root/'validacion/competitor_package_validation.json').read_text())
        review.check('completed assembly record identifies exact ZIP',
            assembly_record['archive'] == archive.name and assembly_record['sha256'] == report['archive_sha256']
            and assembly_record['bytes'] == report['archive_bytes'] and assembly_record['archive_integrity_passed'])
        with tempfile.TemporaryDirectory(prefix='endes_final_zip_readonly_') as directory:
            package = Path(directory)/'package'; package.mkdir()
            inv = inventory(archive, package, root/'paquete_publicacion')
            review.check('ZIP CRC and exact manifest inventory', not inv['issues'], inv['issues'])
            review.check('actual member privacy and binary inspection', not inv['privacy_findings'], inv['privacy_findings'])
            report['package_manifest_sha256'] = inv['contents']['package_manifest.json']['sha256']
            report['indexed_files'] = len(inv['contents'])-1
            report['auditor_in_archive_sha256'] = inv['contents'].get('scripts/competitors/benchmark_final_zip_audit.py', {}).get('sha256')
            report['external_auditor_classification_repair']['archive_auditor_source_is_previous_snapshot'] = bool(
                report['auditor_in_archive_sha256'] and report['auditor_in_archive_sha256'] != report['auditor_sha256'])
            report['external_auditor_classification_repair']['external_reader_distributed_in_this_archive'] = report['auditor_in_archive_sha256'] is not None
            report['uncompressed_bytes'] = sum(row['bytes'] for row in inv['contents'].values())
            actual_current_checks(root, package, inv, review)
            # Never execute any archive source when an identity/security check failed.
            if review.passed():
                result_path = Path(directory)/'portable_result.json'
                environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
                environment.pop('ENDES_EXPORT_PRIVATE_INPUTS', None)
                executed = subprocess.run([sys.executable, '-I', '-B', '-c', PORTABLE_CODE,
                    str(package), str(result_path)], cwd=directory, env=environment,
                    capture_output=True, text=True, timeout=240)
                review.check('portable public read-only guards executed', executed.returncode == 0,
                    {'returncode': executed.returncode, 'stdout': executed.stdout[-5000:], 'stderr': executed.stderr[-5000:]})
                if result_path.exists():
                    report['portable_guard_result'] = json.loads(result_path.read_text())
            else:
                report['portable_guards_skipped_due_to_failed_preconditions'] = True
        review.check('ZIP bytes unchanged throughout independent review', sha(archive) == report['archive_sha256'])
    except Exception as error:
        review.check('audit completed without exception', False, type(error).__name__+': '+str(error))
    report['checks'] = review.checks
    report['failed_checks'] = [row['check'] for row in review.checks if not row['passed']]
    report['passed'] = review.passed()
    report['status'] = 'PASS' if report['passed'] else 'REQUIRED_CORRECTION'
    report['limitations'] = ['Hash/source/portable metadata checks do not rerun statistical estimation.',
        'Root performed the separate rendered-page review; this auditor does not inspect page images.',
        'This is a local author-review package; author confirmation and journal submission remain pending.']
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({'status': report['status'], 'archive_sha256': report['archive_sha256'],
        'checks': len(review.checks), 'failed': report['failed_checks'], 'evidence': str(report_path)}, indent=2))
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--report', type=Path, default=ROOT/'validacion/competitor_final_zip_independent_audit.json')
    args = parser.parse_args()
    return audit(args.archive, args.root.resolve(), args.report.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
