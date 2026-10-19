#!/usr/bin/env python3
"""Optional INEI acquisition using the Python standard library.

Run only when raw-data reconstruction is wanted. Downloaded archives and SAV
files contain individual records and must not be redistributed with results.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datos/provenance/source_manifest.json"
ARCHIVES = {
    "ENDES2019_modulo64.zip": "ENDES2019",
    "ENDES2019_modulo74.zip": "ENDES2019",
    "ENDES2025_complementaria.zip": "ENDES2025",
}
DOCUMENTS = {"ENDES2025_libro.pdf", "ENDES2025_PPR.pdf", "ENDES2019_Cap010.xls"}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def acquire(record, cache, verify_only):
    path = cache / Path(record["path"]).name
    if path.exists():
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Cached hash differs: {path.name}. Preserve and investigate that file before retrying.")
        print(f"Verified cache: {path.name}", flush=True)
        return path, "verified_existing_cache"
    if verify_only:
        raise FileNotFoundError(f"Missing cache: {path.name}; verify-only never downloads.")
    request = urllib.request.Request(record["source_url"], headers={"User-Agent": "ENDES-reproducible-research/1.0"})
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, prefix=path.name + ".", suffix=".partial", delete=False) as f:
            tmp = Path(f.name)
            with urllib.request.urlopen(request, timeout=120) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    f.write(block)
        if tmp.stat().st_size != record["bytes"] or sha256(tmp) != record["sha256"]:
            raise ValueError(f"Official response differs from the archived size/hash: {path.name}; no unverified archive accepted.")
        tmp.replace(path)
        print(f"Downloaded and verified: {path.name}", flush=True)
        return path, "downloaded_verified"
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def extract_safe(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            name = member.filename.replace("\\", "/")
            target = (destination / name).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Unsafe archive member rejected: {name}")
        z.extractall(destination)


def verify_extracted(records, raw):
    checked = 0
    for record in records:
        if Path(record["path"]).suffix.lower() != ".sav":
            continue
        relative = Path(record["path"]).relative_to("datos/raw")
        path = raw / relative
        if not path.exists() or sha256(path) != record["sha256"]:
            raise ValueError(f"Extracted required module differs or is missing: {relative}")
        checked += 1
    print(f"Verified {checked} required SAV modules; individual records were not printed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "datos/cache_microdatos")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "datos/raw")
    parser.add_argument("--extract", action="store_true", help="Extract verified archives to the raw directory.")
    parser.add_argument("--verify-only", action="store_true", help="Check existing cache; never access the network.")
    parser.add_argument("--list", action="store_true", help="List official resources without network access or filesystem changes.")
    parser.add_argument("--documents", action="store_true", help="Also acquire official book/PPR/2019 table; not needed for core reconstruction.")
    args = parser.parse_args()
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))["sources"]
    names = set(ARCHIVES) | (DOCUMENTS if args.documents else set())
    selected = [r for r in records if Path(r["path"]).name in names]
    if len(selected) != len(names):
        raise ValueError("Acquisition metadata is incomplete.")
    if args.list:
        print(json.dumps(selected, ensure_ascii=False, indent=2))
        return
    args.cache.mkdir(parents=True, exist_ok=True)
    log = []
    for record in selected:
        path, status = acquire(record, args.cache, args.verify_only)
        if args.extract and path.name in ARCHIVES:
            extract_safe(path, args.raw_dir / ARCHIVES[path.name])
        log.append({"filename": path.name, "sha256": record["sha256"], "source_url": record["source_url"], "status": status,
                    "contains_individual_microdata": path.suffix.lower() == ".zip"})
    if args.extract:
        verify_extracted(records, args.raw_dir)
    (args.cache / "acquisition_log.json").write_text(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "resources": log}, indent=2), encoding="utf-8")
    print("Keep the cache and raw modules local; the public package contains only aggregates.")


if __name__ == "__main__":
    main()
