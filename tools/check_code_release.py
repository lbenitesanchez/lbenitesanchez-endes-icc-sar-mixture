"""Check the code-only release; no data access or model fitting."""
import ast
import hashlib
import json
from pathlib import Path
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


def main():
    problems = []
    all_files = files()
    for p in all_files:
        relative = p.relative_to(ROOT).as_posix()
        if p.is_symlink(): problems.append((relative, 'symlink'))
        if p.suffix.lower() not in ALLOWED and p.name not in SPECIAL: problems.append((relative, 'non-code artifact extension'))
        data = p.read_bytes()
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
    print(json.dumps({'status': 'PASS_CODE_ONLY_RELEASE', 'files': len(all_files), 'indexed_files': len(expected), 'data_or_documents_included': False, 'model_fits_run_by_this_check': False}))


if __name__ == '__main__':
    main()
