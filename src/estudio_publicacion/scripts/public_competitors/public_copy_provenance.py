"""Trace public path redaction without rewriting historical checksums.

Operational metadata only: no statistical imports, fits, or assertion that
redacted bytes match the original archived checksum.
"""
from pathlib import Path, PureWindowsPath
import argparse
import csv
import hashlib
import io
import json
import re

MANIFEST = 'public_copy_transformations.json'


def private_user_path_hits(text):
    # An official HTTP URL may legitimately contain /home/...; it is not a
    # workstation directory. Normalize JSON-escaped Windows separators only.
    stripped = re.sub(r'https?://[^\s\"\'<>]+', '', text)
    stripped = stripped.replace('\\\\', '\\')
    return re.findall(r'/(?:Users|home)/[^/\s\"\'<>]+/|[A-Za-z]:\\Users\\[^\\\s\"\'<>]+\\', stripped)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    return sha_bytes(Path(path).read_bytes())


def csv_identity(text):
    """Hash every non-path field as literal text, preserving order and IDs."""
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames
    if not fields or len(fields) != len(set(fields)):
        raise ValueError('CSV must have unique column names')
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError('CSV row width differs from its header')
    paths = [name for name in fields if re.search(r'(^|_)(path|file|dir)$', name)
             or name in {'model_inputs', 'extension_file', 'area_input'}]
    statistical = [name for name in fields if name not in paths]
    canonical = json.dumps({'columns': statistical,
        'rows': [[row[name] for name in statistical] for row in rows]},
        ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return {'columns': fields, 'path_columns': paths,
            'statistical_columns': statistical, 'rows': len(rows),
            'statistical_content_sha256': sha_bytes(canonical)}, rows


def verify_redacted_paths(root, rows, changed_columns):
    root = Path(root).resolve()
    for row in rows:
        for column in changed_columns:
            value = row[column]
            if not value:
                continue
            path = Path(value)
            windows = PureWindowsPath(value)
            if (path.is_absolute() or windows.drive or windows.root or '..' in path.parts
                or '..' in windows.parts
                or not (root/path).resolve().is_relative_to(root)):
                raise ValueError('Redacted operational paths must resolve within the public package')


def transformation_record(source, public, source_root, public_path):
    source_bytes = Path(source).read_bytes()
    public_bytes = Path(public).read_bytes()
    record = {'path': str(public_path),
        'source_relative_path': str(Path(source).resolve().relative_to(Path(source_root).resolve())),
        'original_sha256': sha_bytes(source_bytes), 'public_sha256': sha_bytes(public_bytes),
        'transformation': 'Literal analysis-root/home prefixes removed from operational paths only',
        'original_bytes_match_public': False,
        'original_validation_scope_preserved': True,
        'model_source_or_statistical_numeric_calculation_changed': False}
    if Path(source).suffix.lower() == '.csv':
        before, old_rows = csv_identity(source_bytes.decode('utf-8'))
        after, new_rows = csv_identity(public_bytes.decode('utf-8'))
        if before != after:
            raise ValueError('Public CSV redaction changed a non-path field, row order, or schema')
        changed = sorted({name for left, right in zip(old_rows, new_rows)
                          for name in before['columns'] if left[name] != right[name]})
        if not set(changed).issubset(before['path_columns']):
            raise ValueError('Public CSV changes include non-path columns')
        record['CSV_identity'] = before
        record['changed_columns'] = changed
    return record


def verify_public_reference(root, relative, expected_original_sha256):
    """Validate exact bytes or an explicitly recorded path-only CSV copy."""
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Public reference must be a safe relative path')
    path = (root / rel).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Public reference escapes its package')
    actual = file_sha(path)
    if actual == expected_original_sha256:
        return {'path': str(rel), 'passed': True, 'validation_mode': 'exact_original_bytes',
            'original_bytes_match_public': True, 'expected_original_sha256': expected_original_sha256,
            'actual_public_sha256': actual}
    manifest = json.loads((root / MANIFEST).read_text())
    entries = [entry for entry in manifest['files'] if entry['path'] == str(rel)]
    if len(entries) != 1:
        raise ValueError('A unique public path-transformation record is required')
    entry = entries[0]
    if entry['original_sha256'] != expected_original_sha256 or entry['public_sha256'] != actual:
        raise ValueError('Original/public provenance checksums do not match their distinct records')
    if path.suffix.lower() != '.csv' or 'CSV_identity' not in entry:
        raise ValueError('Only verified operational CSV path transformations can bridge this reference')
    identity, rows = csv_identity(path.read_text())
    if identity != entry['CSV_identity']:
        raise ValueError('Public CSV statistical/ID/seed/method fields differ from the original copy')
    if not set(entry['changed_columns']).issubset(identity['path_columns']):
        raise ValueError('Transformation declares a non-path change')
    verify_redacted_paths(root, rows, entry['changed_columns'])
    if rel.name == 'hogg_spatial100.csv':
        expected_ids = list(range(20000, 20100))
        expected_paths = [f'resultados/competidores_hogg/spatial_sensitivity_{(rep-20000)//25:02d}'
                          f'/structural_bym2/rep_{rep:05d}' for rep in expected_ids]
        if ([row['replication'] for row in rows] != list(map(str, expected_ids))
            or any(row['scenario'] != 'sparse_psu_bym2' or row['method'] != 'HoggSpatial'
                   or row['protocol'] != 'separate_S2_BYM2_component_sensitivity_v1'
                   or row['diagnostic_style'] != 'hogg' for row in rows)
            or [row['output_dir'] for row in rows] != expected_paths):
            raise ValueError('Public spatial tasks differ from the fixed100-world operational layout')
    return {'path': str(rel), 'passed': True, 'validation_mode': 'verified_path_only_public_copy',
        'original_bytes_match_public': False, 'expected_original_sha256': expected_original_sha256,
        'actual_public_sha256': actual, 'statistical_content_sha256': identity['statistical_content_sha256'],
        'historical_contract_unchanged': True}


def verify_manifest(root):
    root = Path(root).resolve()
    manifest = json.loads((root / MANIFEST).read_text())
    for entry in manifest['files']:
        rel = Path(entry['path'])
        if rel.is_absolute() or '..' in rel.parts or not (root / rel).resolve().is_relative_to(root):
            raise ValueError('Unsafe transformed-file reference')
        if file_sha(root / rel) != entry['public_sha256']:
            raise ValueError('Public copy differs from its own checksum: ' + str(rel))
        if 'CSV_identity' in entry:
            identity, rows = csv_identity((root / rel).read_text())
            if identity != entry['CSV_identity']:
                raise ValueError('Public statistical CSV columns differ: ' + str(rel))
            if not set(entry['changed_columns']).issubset(identity['path_columns']):
                raise ValueError('Transformation declares a non-path change')
            verify_redacted_paths(root, rows, entry['changed_columns'])
    return {'passed': True, 'transformed_files': len(manifest['files']),
            'original_checksum_values_not_relabelled_as_public': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    result = verify_manifest(args.root)
    contract = json.loads((args.root / 'resultados/resultados_comparadores/spatial_result_contract.json').read_text())
    result['spatial_original_contract_reference'] = verify_public_reference(
        args.root, contract['task_file'], contract['task_sha256'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
