"""Additional official CPV2017DI aggregates needed by Sikov Table 3.1.

Acquisition is independent of ENDES responses. Reuses audited parser without
modifying any previous source or output. Each independent request is archived.
"""
from pathlib import Path
import importlib.util
import argparse

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT.parent
source = STUDY / 'fase2_2026-10-07/scripts/geografia_redatam.py'
spec = importlib.util.spec_from_file_location('archived_official_query', source)
query_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(query_module)
query_module.ROOT = ROOT
query_module.SOURCE = ROOT / 'datos/census_sources'
query_module.SOURCE.mkdir(parents=True, exist_ok=True)
QUERIES = [
    ('census_walls', 'AREAVIV', ['Vivienda.C2P3']),
    ('census_internet', 'AREAHOG', ['Hogar.C3P213']),
    ('census_sis', 'AREAPOB', ['Poblacio.C5P81']),
    ('census_essalud', 'AREAPOB', ['Poblacio.C5P82']),
    ('census_noinsurance', 'AREAPOB', ['Poblacio.C5P86']),
    ('census_rural', 'AREAVIV', ['Vivienda.ENCAREA']),
]
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only')
    args = parser.parse_args()
    requests = [q for q in QUERIES if not args.only or args.only == q[0]]
    # REDATAM's XXUSUARIOXX requests share a remote temporary slot. Parallel
    # requests can overwrite each other's table links, so acquisition is serial.
    for request in requests:
        result = query_module.query(*request)
        print(result['name'], result['retrieved_utc'], flush=True)
