"""Acquire official reserved ENDES2024 bytes without reading child responses.

Before protocol freeze this program inspects only ZIP member metadata. It does
not extract or parse SAV files. Public INEI aggregate snippets seen incidentally
in source discovery are disclosed in the acquisition manifest.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import urllib.request
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Lib2016/Informacion_Complementaria_Endes_2024.zip'
LANDING = 'https://www.gob.pe/institucion/inei/informes-publicaciones/6813623-peru-encuesta-demografica-y-de-salud-familiar-endes-2024'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    dest = ROOT / 'datos/private_external_raw/ENDES2024_complementaria.zip'
    metadata = ROOT / 'fuentes/external'
    metadata.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if args.download and not dest.exists():
        temporary = dest.with_suffix('.zip.part')
        # System curl uses the native trusted certificate store. Certificate
        # verification remains enabled; no insecure TLS option is permitted.
        headers_path = metadata / 'download_http_headers.txt'
        run = subprocess.run(['/usr/bin/curl', '--fail', '--location', '--retry', '2',
                              '--connect-timeout', '30', '--max-time', '1200',
                              '--dump-header', str(headers_path), '--output', str(temporary),
                              '--write-out', '%{url_effective}', URL], check=True, text=True, stdout=subprocess.PIPE)
        final_url = run.stdout.strip()
        headers = headers_path.read_text()
        temporary.replace(dest)
        (metadata / 'download_http_metadata.json').write_text(json.dumps({'requested_url': URL, 'final_url': final_url, 'headers': headers}, indent=2))
    if not dest.exists():
        raise FileNotFoundError(dest)
    with zipfile.ZipFile(dest) as archive:
        members = [{'name': item.filename, 'uncompressed_bytes': item.file_size, 'compressed_bytes': item.compress_size,
                    'crc32': format(item.CRC, '08x')} for item in archive.infolist()]
        wanted = [item for item in members if any(name in item['name'].upper() for name in ['RECH0_', 'RECH1_', 'RECH6_', 'DICCIONARIO', 'DICIONARIO', 'FICHA'])]
    manifest = {'status': 'RESERVED_ARCHIVE_ACQUIRED_NO_CHILD_RESPONSES_PARSED', 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'survey_edition': 2024, 'official_publication_date': '2025-05-28', 'official_landing_url': LANDING,
                'archive_url': URL, 'archive_relative_path': str(dest.relative_to(ROOT)), 'archive_sha256': sha(dest),
                'archive_bytes': dest.stat().st_size, 'members': members, 'relevant_members': wanted,
                'child_outcomes_read': False, 'child_SAV_extracted': False, 'household_frame_read': False,
                'exposure_disclosure': 'Source search incidentally displayed public national 2024 values for anemia ages6-35 and stunting ages0-59. No reserved district/child outcomes or model scores inspected. Edition2024 was selected by parent before this search.',
                'privacy': 'Raw archive and individual records remain private; never include in public manuscript package.'}
    output = metadata / 'external_2024_acquisition_manifest.json'
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in ['status', 'archive_sha256', 'archive_bytes']}), flush=True)


if __name__ == '__main__':
    main()
