#!/usr/bin/env python3
"""Create a new aggregate-only analysis workspace; refuse existing destinations."""
import argparse
import datetime
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    if destination.exists():
        raise SystemExit("Destination already exists; choose a new directory. No existing analysis was overwritten.")
    if destination.is_relative_to(ROOT):
        raise SystemExit("Choose a directory outside the distributed package, usually a sibling directory.")
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "codigo", destination / "codigo", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("processed", "auxiliares", "provenance"):
        shutil.copytree(ROOT / "datos" / name, destination / "datos" / name)
    for name in ("requirements.txt", "README_DATOS_ES_EN.md", "README_ANALISIS_ES_EN.md", "reproducir_analisis.sh"):
        shutil.copy2(ROOT / name, destination / name)
    (destination / "resultados/simulaciones").mkdir(parents=True)
    (destination / "fresh_workspace.json").write_text(json.dumps({
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "fresh_directory": True,
        "copied_inputs": ["codigo", "datos/processed", "datos/auxiliares", "datos/provenance"],
        "contains_raw_microdata": False,
        "existing_results_copied": False,
        "instruction": "Run only inside this fresh workspace; published results are preserved in the distributed package."}, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
