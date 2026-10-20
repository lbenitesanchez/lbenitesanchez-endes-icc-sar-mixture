"""Check source files and the exact approved CSV bundle; no model fitting."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'.py', '.r', '.stan', '.cpp', '.h', '.hpp', '.json', '.md', '.txt', '.sha256', '.cff'}
SPECIAL = {'.gitignore', '.gitattributes'}
PERSONAL = re.compile(r'/(?:Users|home|Volumes)/[A-Za-z0-9_-]{2,}(?:/|$)')
SECRET = re.compile(r'(?:ghp_|github_pat_|sk-proj-|xoxb-|xoxp-)[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
PAPER = ['Dear' + ' Editor', 'We present for' + ' author review', 'District health statistics require' + ' useful uncertainty']
SKIP = {'.git', '__pycache__', '.venv', 'venv', '.pytest_cache'}


def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(part in SKIP for part in p.relative_to(ROOT).parts) and p.name != '.DS_Store')


def approved_csvs():
    """Only the 35 frozen, reviewed input identities may be distributed as CSV."""
    contract = json.loads((ROOT / 'simplified/input_hashes.json').read_text())
    registered = contract['external_inputs_sha256']
    if not isinstance(registered, dict) or len(registered) != 35:
        raise ValueError('The approved bundle must contain exactly 35 CSVs')
    approved = {}
    for relative, digest in registered.items():
        path = PurePosixPath(relative)
        if (path.is_absolute() or '\\' in relative or any(part in ('', '.', '..') for part in relative.split('/'))
                or not relative.startswith(('datos/', 'referencias/')) or path.suffix != '.csv'
                or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)):
            raise ValueError('Invalid registered CSV identity: ' + relative)
        approved['simplified/inputs/' + relative] = digest
    return approved


def main():
    problems = []
    all_files = files()
    try:
        approved = approved_csvs()
    except (KeyError, TypeError, ValueError, OSError) as error:
        approved = {}
        problems.append(('simplified/input_hashes.json', str(error)))
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if path.is_symlink() and not any(part in SKIP for part in relative.parts):
            problems.append((relative.as_posix(), 'symlink'))
    present = {p.relative_to(ROOT).as_posix() for p in all_files}
    for relative in sorted(set(approved) - present):
        problems.append((relative, 'missing approved CSV'))
    for p in all_files:
        relative = p.relative_to(ROOT).as_posix()
        if p.is_symlink(): problems.append((relative, 'symlink'))
        if p.suffix.lower() not in ALLOWED and p.name not in SPECIAL and relative not in approved:
            problems.append((relative, 'unapproved artifact extension'))
        data = p.read_bytes()
        if relative in approved and hashlib.sha256(data).hexdigest() != approved[relative]:
            problems.append((relative, 'approved CSV hash differs'))
        try: text = data.decode('utf-8')
        except UnicodeDecodeError:
            problems.append((relative, 'binary payload')); continue
        if PERSONAL.search(text): problems.append((relative, 'personal filesystem location'))
        if SECRET.search(text): problems.append((relative, 'possible credential'))
        if any(marker in text for marker in PAPER): problems.append((relative, 'embedded document prose'))
        if p.suffix == '.py':
            try: ast.parse(text, filename=relative)
            except SyntaxError as error: problems.append((relative, str(error)))
        if p.suffix == '.json':
            try: json.loads(text)
            except ValueError as error: problems.append((relative, str(error)))
    manifest = json.loads((ROOT / 'provenance/release_files.json').read_text())
    expected = manifest['files_sha256']
    actual = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in all_files if p.relative_to(ROOT).as_posix() not in manifest['excluded_from_self_index']}
    if actual != expected:
        problems.append(('provenance/release_files.json', 'source inventory differs'))
    if problems:
        print(json.dumps({'status': 'FAIL', 'findings': problems}, indent=2)); raise SystemExit(1)
    print(json.dumps({'status': 'PASS_PUBLIC_REPRODUCTION_RELEASE', 'files': len(all_files),
                      'indexed_files': len(expected), 'approved_derived_csv_files': len(approved),
                      'private_microdata_or_submission_documents_included': False,
                      'model_fits_run_by_this_check': False}))


if __name__ == '__main__':
    main()
