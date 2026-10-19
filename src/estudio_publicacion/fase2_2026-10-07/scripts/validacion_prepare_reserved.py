"""Guarded acquisition-to-private-preparation entry point; no score calculation."""
from pathlib import Path
import datetime as dt
import json
import subprocess
import zipfile
from validacion_fit_models import ROOT, sha, gate


def main():
    protocol_sha = gate(2024)
    archive = ROOT / 'datos/private_external_raw/ENDES2024_complementaria.zip'
    expected = json.loads((ROOT / 'fuentes/external/external_2024_acquisition_manifest.json').read_text())['archive_sha256']
    if sha(archive) != expected:
        raise RuntimeError('Reserved archive differs from prefrozen source')
    event = ROOT / 'validacion/externa_first_reserved_open.json'
    if event.exists():
        raise RuntimeError('Reserved responses have already been opened; preserve the original event and outputs')
    info = {'event': 'first_reserved_child_response_file_extraction',
            'time_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'protocol_sha256': protocol_sha,
            'archive_sha256': expected, 'prediction_scores_computed': False}
    event.write_text(json.dumps(info, indent=2))
    with zipfile.ZipFile(archive) as z:
        for name in ['RECH1_2024.SAV', 'RECH6_2024.SAV']:
            data = z.read('2. BD ENDES 2024/' + name)
            (ROOT / 'datos/private_external_raw' / name).write_bytes(data)
    subprocess.run(['/usr/local/bin/Rscript', str(Path(__file__).with_name('validacion_parse_reserved.R')), str(ROOT)], check=True, cwd=ROOT)
    print(json.dumps({'status': 'reserved_inputs_prepared_privately', 'outcome_scores_computed': False}), flush=True)


if __name__ == '__main__':
    main()
