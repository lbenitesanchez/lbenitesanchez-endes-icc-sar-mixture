#!/usr/bin/env python3
"""Compare numeric reconstruction with published aggregates, allowing CSV rounding."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = {
    "areas_2019_6_59.csv": ["ubigeo"], "areas_2025_6_59.csv": ["ubigeo"],
    "aggregates_2019.csv": ["level", "domain", "universe"],
    "aggregates_2025.csv": ["level", "domain", "universe"],
    "national_summary.csv": ["year", "level", "domain", "universe"],
    "sample_flow.csv": ["year"], "who2024_official_validation.csv": ["year"],
    "variance_survey_validation.csv": ["year", "level", "domain", "universe"],
    "unmatched_geography.csv": ["year", "ubigeo"], "aux_all_districts.csv": ["ubigeo"],
}


def read(path, keys):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in result:
            raise ValueError(f"Duplicate keys in {path.name}: {key}")
        result[key] = row
    return fields, result


def equal(a, b, atol, rtol):
    if a == b or {a.upper(), b.upper()} <= {"", "NA", "NAN"}:
        return True, 0.0
    try:
        x, y = float(a), float(b)
    except ValueError:
        return a.upper() == b.upper(), 0.0
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y), 0.0
    return math.isclose(x, y, abs_tol=atol, rel_tol=rtol), abs(x-y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstructed", type=Path, default=ROOT / "datos/reconstruidos")
    parser.add_argument("--reference", type=Path, default=ROOT / "datos/processed")
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--rtol", type=float, default=1e-11)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    checks = []
    differences = []
    for filename, keys in KEYS.items():
        fields, ref = read(args.reference / filename, keys)
        candidate_fields, candidate = read(args.reconstructed / filename, keys)
        missing_fields = set(fields)-set(candidate_fields)
        if set(ref) != set(candidate) or missing_fields:
            differences.append({"file": filename, "missing_columns": sorted(missing_fields), "keys_equal": set(ref)==set(candidate)})
            continue
        maximum = 0.0
        n = 0
        for key, row in ref.items():
            for column in fields:
                ok, delta = equal(row[column], candidate[key][column], args.atol, args.rtol)
                maximum = max(maximum, delta)
                n += 1
                if not ok and len(differences) < 30:
                    differences.append({"file": filename, "key": key, "column": column, "absolute_difference": delta})
        checks.append({"file": filename, "rows": len(ref), "cells_compared": n, "max_numeric_absolute_difference": maximum})
    report = {"passed": not differences, "atol": args.atol, "rtol": args.rtol, "checks": checks, "differences": differences,
              "interpretation": "Numerical/label reconstruction check; matching ratios do not validate census extraction or district model accuracy."}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
