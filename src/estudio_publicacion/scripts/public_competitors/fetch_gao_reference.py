"""Optional pinned author reference for Gao fidelity checks; stdlib only.

No download occurs unless explicitly executing this helper. It fetches only the
MIT license and single author models.R file, not a repository clone or data.
"""
from pathlib import Path
import hashlib,json,urllib.request

ROOT=Path(__file__).resolve().parents[2]

def main():
    folder=ROOT/'fuentes/competidores/gao_wakefield'
    manifest=json.loads((folder/'sources_manifest.json').read_text())
    rows=manifest.get('sources',manifest.get('files',[]))
    if not rows:raise ValueError('Pinned source manifest is missing')
    wanted={'author_code/LICENSE','author_code/analysis/models.R'}
    selected=[row for row in rows if row['file']in wanted]
    if len(selected)!=2:raise ValueError('Required two reference records not found')
    for row in selected:
        target=folder/row['file'];target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():payload=target.read_bytes()
        else:
            with urllib.request.urlopen(row['url'],timeout=120)as response:payload=response.read()
        if len(payload)!=row['bytes']or hashlib.sha256(payload).hexdigest()!=row['sha256']:
            raise ValueError('Pinned reference checksum differs: '+row['file'])
        if not target.exists():target.write_bytes(payload)
        print('Verified author reference: '+row['file'])

if __name__=='__main__':main()
