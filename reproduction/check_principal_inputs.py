"""Verify locally acquired principal inputs without exporting survey records.

This checks the six archived raw-file identities and the two public district
auxiliaries. Passing this check does not supply the additional scientific
reference trees or certify a completed preparation or model fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(data_root):
    root = Path(data_root).expanduser().resolve()
    if root.is_relative_to(REPO) or REPO.is_relative_to(root):
        raise ValueError("The principal input root must be separate from the Git checkout")
    catalogue = json.loads((REPO / "expected/private_csv_checksums.json").read_text())
    raw = catalogue["raw_sav_checksums"]
    if len(raw) != 6:
        raise ValueError("Expected the six recorded 2019/2024 raw files")
    checked, problems = [], []
    for relative, expected in sorted(raw.items()):
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Invalid raw-file catalogue path")
        path = root / name
        if not path.is_file():
            problems.append({"path": relative, "reason": "missing file"})
        elif path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            problems.append({"path": relative, "reason": "archived size or SHA-256 differs"})
        else:
            checked.append(relative)
    review = json.loads((REPO / "provenance/principal_data_review.json").read_text())
    auxiliary = review["approved_auxiliary_files"]
    for relative, expected in sorted(auxiliary.items()):
        path = REPO / relative
        if not path.is_file() or sha(path) != expected:
            problems.append({"path": relative, "reason": "public auxiliary missing or changed"})
    return {"status": "FAIL_PRINCIPAL_INPUT_IDENTITIES" if problems else "PASS_PRINCIPAL_INPUT_IDENTITIES_ONLY",
            "verified_raw_files": len(checked), "expected_raw_files": 6,
            "public_auxiliary_files": len(auxiliary), "findings": problems,
            "individual_records_exported": False, "data_files_modified": False,
            "reference_prerequisites_validated": False, "model_fits_run": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="External study root containing the six SAV files at the recorded relative paths")
    args = parser.parse_args()
    try:
        result = verify(args.data_root)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, "Error: " + str(error) + "\n")
    print(json.dumps(result, indent=2))
    if result["findings"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
