#!/usr/bin/env python3
"""Verify distributed aggregate files by SHA-256; no third-party packages."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "datos/provenance/aggregate_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for r in manifest["files"]:
        path = ROOT / r["path"]
        if not path.exists() or path.stat().st_size != r["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != r["sha256"]:
            failures.append(r["path"])
    if failures:
        raise SystemExit("Aggregate files differ: " + ", ".join(failures))
    print(f"Verified {len(manifest['files'])} frozen aggregate/data files. No microdata required.")


if __name__ == "__main__":
    main()
