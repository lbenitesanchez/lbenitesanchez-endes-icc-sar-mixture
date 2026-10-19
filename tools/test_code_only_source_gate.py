"""Exercise strict source compatibility without data or statistical fits."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reproduction'))
from code_only_sources import historical_source_matches


def main():
    digest = lambda x: hashlib.sha256(x).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        repo = Path(directory)
        (repo / 'provenance').mkdir()
        target = repo / 'module.py'
        original, portable = b'original source', b'portable path source'
        old = digest(original)
        manifest = repo / 'provenance/code_only_sources.json'
        record = {'original_sha256': old, 'public_sha256': digest(portable), 'policy': 'portable_paths_only'}
        manifest.write_text(json.dumps({'historical_sources': {'module.py': record}}))
        target.write_bytes(original)
        assert historical_source_matches(target, old)
        target.write_bytes(portable)
        assert historical_source_matches(target, old)
        assert not historical_source_matches(target, digest(b'wrong historical identity'))
        target.write_bytes(b'unrecorded statistical change')
        assert not historical_source_matches(target, old)
        target.unlink()
        assert not historical_source_matches(target, old)
        record['policy'] = 'editorial_source_omitted'
        record['public_sha256'] = None
        manifest.write_text(json.dumps({'historical_sources': {'module.py': record}}))
        assert historical_source_matches(target, old)
        assert not historical_source_matches(target, digest(b'wrong omitted identity'))
        assert not historical_source_matches(repo / 'unknown.py', old)
        target.write_bytes(b'an unexpected replacement')
        assert not historical_source_matches(target, old)
    # Check the actual archived cohort protocol against the published source map.
    protocol = json.loads((ROOT / 'revision_20261015/protocols/new_computation_protocol_v1.json').read_text())
    for relative, expected in protocol['original_sources_sha256'].items():
        assert historical_source_matches(ROOT / relative, expected), relative
    print('PASS strict source compatibility fixtures and all archived cohort source identities')


if __name__ == '__main__':
    main()
