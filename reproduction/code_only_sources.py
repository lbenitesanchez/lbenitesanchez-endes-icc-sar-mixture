"""Explicit compatibility of the public code-only source distribution.

No statistical source may change without a recorded exact original/public hash.
Only documented editorial sources may be absent. Scientific model, seed,
likelihood, graph, inference and bootstrap gates are otherwise unchanged.
"""
from pathlib import Path
import hashlib
import json


def historical_source_matches(target, expected):
    target = Path(target).resolve()
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if target.is_file() and digest(target) == expected:
        return True
    for repo in (target.parent, *target.parents):
        manifest = repo / 'provenance/code_only_sources.json'
        if manifest.is_file():
            relative = target.relative_to(repo).as_posix()
            record = json.loads(manifest.read_text()).get('historical_sources', {}).get(relative)
            if not record or record.get('original_sha256') != expected:
                return False
            if record.get('policy') == 'editorial_source_omitted':
                return not target.exists()
            if record.get('policy') in ('portable_paths_only', 'numerical_table_formatters_only', 'tables_only_contract'):
                return target.is_file() and digest(target) == record.get('public_sha256')
            return False
    return False
