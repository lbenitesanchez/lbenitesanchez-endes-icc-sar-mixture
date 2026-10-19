"""Independent locked external score arithmetic; run only after final completion.

No model/scoring-module imports, no fitting, no selection or new exclusions.
All private2024 response reads occur after final completion and both exact
three-method locks have passed the hash gates.
"""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

from pathlib import Path
from collections import defaultdict
import csv
import datetime as dt
import hashlib
import json
import math
import statistics as stats
import numpy as np
from scipy.stats import t as student_t

PHASE = Path(__file__).resolve().parents[1]
OUT = PHASE / "resultados/externa_final"
METHODS = ("BetaPowerExisting", "BetaDesignUncertainty", "HoggAgeSexSpatial")
ENDPOINTS = ("anemia", "stunting")
REFERENCES = ("BetaPowerExisting", "HoggAgeSexSpatial")
RESPONSES = {"anemia": "y_anemia_who2024", "stunting": "y_stunting"}
SCOPES = ("temporal", "territorial")
FLOOR = 1e-12
BOOTSTRAPS = 3999
ABS_TOLERANCE = 2e-11


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def code(value):
    value = str(value).zfill(6)
    check(len(value) == 6 and value.isdigit(), "Invalid six-digit geographic identifier")
    return value


def held_province(province):
    value = hashlib.sha256(("ENDES-phase2-20261007|" + province).encode()).hexdigest()
    return int(value[:8], 16) % 5 == 0


def percentile(values, q):
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def cluster_contrast(differences, provinces, rng):
    """Scalar cluster sums and hand-interpolated paired bootstrap percentiles."""
    labels = sorted(set(provinces))
    J, n = len(labels), len(differences)
    check(J >= 2 and n > 0, "Insufficient province clusters")
    mean = stats.fmean(differences)
    buckets = defaultdict(list)
    for value, label in zip(differences, provinces):
        buckets[label].append(value)
    counts = [len(buckets[x]) for x in labels]
    totals = [math.fsum(buckets[x]) for x in labels]
    centered = [s - count * mean for s, count in zip(totals, counts)]
    se = math.sqrt(J * math.fsum(x * x for x in centered) / (J - 1)) / n
    statistic = mean / se if se > 0 else 0 if mean == 0 else math.copysign(math.inf, mean)
    p = float(2 * student_t.sf(abs(statistic), J - 1))
    critical = float(student_t.ppf(.975, J - 1))
    # Same frozen seed/label order; arithmetic does not reuse scorer outputs.
    draws = rng.integers(0, J, size=(BOOTSTRAPS, J))
    bootstrap = [math.fsum(totals[int(j)] for j in draw) / sum(counts[int(j)] for j in draw) for draw in draws]
    return {"difference": mean, "cluster_standard_error": se,
            "province_clusters": J, "t_df": J - 1, "p_two_sided": p,
            "unadjusted_t_ci_low": mean - critical * se,
            "unadjusted_t_ci_high": mean + critical * se,
            "descriptive_bootstrap_ci_low": percentile(bootstrap, .025),
            "descriptive_bootstrap_ci_high": percentile(bootstrap, .975),
            "bootstrap_repetitions": BOOTSTRAPS}


def holm8(contrasts):
    check(len(contrasts) == 8, "Exactly eight fixed NLL contrasts required")
    ordered = sorted(enumerate(contrasts), key=lambda pair: pair[1]["p_two_sided"])
    running_maximum = 0.0
    for position, (_, item) in enumerate(ordered):
        p = item["p_two_sided"]
        check(math.isfinite(p) and 0 <= p <= 1, "Invalid primary family p-value")
        running_maximum = max(running_maximum, (8 - position) * p)
        item["p_Holm8"] = min(1., running_maximum)
        item["candidate_significantly_better"] = item["difference"] < 0 and item["p_Holm8"] < .05
        item["candidate_significantly_worse"] = item["difference"] > 0 and item["p_Holm8"] < .05


def main():
    # Neither private responses nor exposed score CSVs are opened before this.
    completion_path = OUT / "completion.json"
    if not completion_path.is_file():
        raise RuntimeError("External audit not authorized to read results/responses before externa_final/completion.json")
    completion = read(completion_path)
    check(completion["status"] == "COMPLETE_RESERVED_EXTERNAL_VALIDATION", "External collection incomplete")
    protocol_path = PHASE / "protocol_freeze.json"
    protocol = read(protocol_path)
    protocol_sha = sha(protocol_path)
    check(completion["protocol_sha256"] == protocol_sha, "External completion binds another protocol")
    score_path = PHASE / "scripts/validacion_score_external.py"
    check(sha(score_path) == protocol["source_sha256"]["scripts/validacion_score_external.py"], "Frozen scorer source changed")
    check(completion["source_sha256"] == sha(score_path), "External completion scorer hash changed")
    access = read(PHASE / "validacion/externa_reserved_access_approval.json")
    check(access.get("approved_for_reserved_outcome_access") is True and access["protocol_sha256"] == protocol_sha, "Reserved-access approval missing")
    locks = {}
    prediction_maps = {}
    prediction_digests = {}
    for scope in SCOPES:
        lock_path = PHASE / f"validacion/externa_prediction_lock_{scope}.json"
        lock = read(lock_path)
        check(lock["scope"] == scope and lock.get("approved_for_scoring") is True, "Exact scope prediction lock missing")
        check(lock["protocol_sha256"] == protocol_sha, "Prediction lock protocol mismatch")
        check(len(lock["predictions"]) == 3 and {x["method"] for x in lock["predictions"]} == set(METHODS), "Three-method family changed")
        maps = {}
        for item in lock["predictions"]:
            path, status = PHASE / item["path"], PHASE / item["status_path"]
            check(sha(path) == item["sha256"] and sha(status) == item["status_sha256"], "Locked prediction/status bytes changed")
            check(read(status).get("success") is True, "Locked model status not successful")
            rows = csv_rows(path)
            area_key, mean_prefix = item.get("area_key", "ubigeo"), item.get("mean_prefix", "p_mean_")
            data = {}
            for row in rows:
                area = code(row[area_key])
                check(area not in data, "Duplicate prediction geography")
                data[area] = {outcome: number(row[mean_prefix + outcome]) for outcome in ENDPOINTS}
                check(all(math.isfinite(p) and 0 <= p <= 1 for p in data[area].values()), "Invalid locked predicted probability")
            maps[item["method"]] = data
        check(all(set(maps[m]) == set(maps[METHODS[0]]) for m in METHODS), "Unequal supplied geographic supports")
        locks[scope] = lock
        prediction_maps[scope] = maps
        prediction_digests[scope] = {"lock_sha256": sha(lock_path), "locked_maps": lock["predictions"]}
    check({x["scope"] for x in completion["inputs"]} == set(SCOPES), "Completed input scopes changed")
    input_records = {x["scope"]: x for x in completion["inputs"]}
    for scope in SCOPES:
        check(input_records[scope]["prediction_lock_sha256"] == prediction_digests[scope]["lock_sha256"], "Completed scoring uses a different prediction lock")
    # Both locks and final completion now passed. Private response reads begin.
    district_metrics, summaries, contrasts, excluded, scope_proofs = [], [], [], [], []
    for scope in SCOPES:
        response_path = PHASE / ("datos/private_external/children_2024_scoring_only.csv" if scope == "temporal" else "datos/private_external/children_2024_test.csv")
        check(sha(response_path) == input_records[scope]["response_sha256"], "Scoring response snapshot changed")
        responses = csv_rows(response_path)
        buckets = defaultdict(list)
        for row in responses:
            check(number(row["year"]) == 2024 and number(row["de_facto"]) == 1 and 6 <= number(row["age_months"]) <= 59, "Scoring population changed")
            area = code(row["ubigeo"])
            if scope == "territorial":
                check(held_province(area[:4]), "Training/nonreserved province entered territorial test")
            buckets[area].append(row)
        maps = prediction_maps[scope]
        support = set(maps[METHODS[0]])
        rng = np.random.default_rng(202610072024 + (10000 if scope == "territorial" else 0))
        eligible_counts = {}
        for outcome in ENDPOINTS:
            endpoint_rows = {method: [] for method in METHODS}
            eligible = []
            for area in sorted(buckets):
                valid = [(number(row["weight"]), number(row[RESPONSES[outcome]])) for row in buckets[area]
                         if number(row[RESPONSES[outcome]]) in (0., 1.) and math.isfinite(number(row["weight"])) and number(row["weight"]) > 0]
                total_weight = math.fsum(w for w, _ in valid)
                positive_records = len(valid)
                direct = math.fsum(w * y for w, y in valid) / total_weight if total_weight > 0 else math.nan
                if area not in support or total_weight <= 0:
                    excluded.append({"scope": scope, "outcome": outcome, "ubigeo": area,
                        "province": area[:4], "reason": "unverified_auxiliary_geography" if area not in support else "no_valid_positive_weight_outcome",
                        "weighted_observed_denominator": total_weight, "valid_positive_weight_records": positive_records})
                    continue
                eligible.append(area)
                for method in METHODS:
                    p = maps[method][area][outcome]
                    q = max(FLOOR, min(1 - FLOOR, p))
                    # Directly sum weighted binary observations, independently
                    # of the published mean-direct score shortcut.
                    nll = math.fsum(w * (-math.log(q) if y == 1 else -math.log1p(-q)) for w, y in valid) / total_weight
                    brier = math.fsum(w * (p - y) ** 2 for w, y in valid) / total_weight
                    record = {"scope": scope, "outcome": outcome, "method": method, "ubigeo": area,
                        "province": area[:4], "direct": direct, "weighted_observed_denominator": total_weight,
                        "valid_positive_weight_records": positive_records, "binary_NLL": nll,
                        "Brier_binary_expectation": brier, "squared_error_against_noisy_direct": (p - direct) ** 2,
                        "numerical_clipping_applied": p < FLOOR or p > 1 - FLOOR}
                    district_metrics.append(record)
                    endpoint_rows[method].append(record)
            eligible_counts[outcome] = len(eligible)
            check(eligible, "Empty frozen eligibility cohort")
            for method in METHODS:
                rows = endpoint_rows[method]
                summaries.append({"scope": scope, "outcome": outcome, "method": method,
                    "districts": len(rows), "provinces": len({x["province"] for x in rows}),
                    "binary_NLL": stats.fmean(x["binary_NLL"] for x in rows),
                    "Brier_binary_expectation": stats.fmean(x["Brier_binary_expectation"] for x in rows),
                    "RMSE_noisy_direct_pp": 100 * math.sqrt(stats.fmean(x["squared_error_against_noisy_direct"] for x in rows)),
                    "clipped_cells": sum(x["numerical_clipping_applied"] for x in rows)})
            candidate = endpoint_rows["BetaDesignUncertainty"]
            for reference in REFERENCES:
                references = endpoint_rows[reference]
                check([x["ubigeo"] for x in candidate] == [x["ubigeo"] for x in references] == eligible, "Paired equal-district eligibility changed")
                differences = [c["binary_NLL"] - r["binary_NLL"] for c, r in zip(candidate, references)]
                inference = cluster_contrast(differences, [x["province"] for x in candidate], rng)
                contrasts.append({"scope": scope, "outcome": outcome, "candidate": "BetaDesignUncertainty",
                                  "reference": reference, "districts": len(candidate), **inference})
        scope_proofs.append({"scope": scope, "response_sha256": sha(response_path), "input_child_records": len(responses),
                             "observed_districts": len(buckets), "locked_prediction_districts": len(support),
                             "eligible_districts_by_outcome": eligible_counts})
    holm8(contrasts)
    check(len(contrasts) == 8 and len(summaries) == 12, "Fixed contrast/summary family changed")
    failures, maximum = [], {}
    def compare(filename, independent, keys, numeric, flags=()):
        rows = csv_rows(OUT / filename)
        key = lambda row: tuple(str(row[x]) for x in keys)
        expected, published = {key(x): x for x in independent}, {key(x): x for x in rows}
        check(len(expected) == len(independent) and len(published) == len(rows) and set(expected) == set(published), "Reported eligibility/grid differs: " + filename)
        deltas = []
        for ident, row in published.items():
            for col in numeric:
                delta = abs(number(row[col]) - expected[ident][col])
                deltas.append(delta)
                # Large raw weighted totals also allow normal floating summation
                # tolerance; score/p/CI comparisons retain strict absolute bound.
                tolerance = max(ABS_TOLERANCE, 2e-13 * abs(expected[ident][col])) if col == "weighted_observed_denominator" else ABS_TOLERANCE
                if not math.isfinite(delta) or delta > tolerance:
                    failures.append({"file": filename, "key": ident, "column": col, "absolute_difference": delta})
            for col in flags:
                if (str(row[col]).lower() == "true") != expected[ident][col]:
                    failures.append({"file": filename, "key": ident, "column": col, "kind": "boolean_mismatch"})
        maximum[filename] = max(deltas, default=0)
    compare("district_scores.csv", district_metrics, ["scope", "outcome", "method", "ubigeo"],
            ["direct", "weighted_observed_denominator", "valid_positive_weight_records", "binary_NLL", "Brier_binary_expectation", "squared_error_against_noisy_direct"], ["numerical_clipping_applied"])
    compare("summary.csv", summaries, ["scope", "outcome", "method"],
            ["districts", "provinces", "binary_NLL", "Brier_binary_expectation", "RMSE_noisy_direct_pp", "clipped_cells"])
    contrast_columns = ["districts", "difference", "cluster_standard_error", "province_clusters", "t_df", "p_two_sided", "p_Holm8",
                        "unadjusted_t_ci_low", "unadjusted_t_ci_high", "descriptive_bootstrap_ci_low", "descriptive_bootstrap_ci_high", "bootstrap_repetitions"]
    compare("contrasts_Holm8.csv", contrasts, ["scope", "outcome", "candidate", "reference"], contrast_columns,
            ["candidate_significantly_better", "candidate_significantly_worse"])
    compare("excluded_district_endpoints.csv", excluded, ["scope", "outcome", "ubigeo", "reason"], ["weighted_observed_denominator", "valid_positive_weight_records"])
    check(all(sha(OUT / name) == expected for name, expected in completion["output_sha256"].items()), "Reported score bytes changed after completion")
    findings = {"significantly_better_contrasts": sum(x["candidate_significantly_better"] for x in contrasts),
                "significantly_worse_contrasts": sum(x["candidate_significantly_worse"] for x in contrasts),
                "negative_point_NLL_differences": sum(x["difference"] < 0 for x in contrasts),
                "positive_point_NLL_differences": sum(x["difference"] > 0 for x in contrasts)}
    proof = {"status": "PASS_INDEPENDENT_EXTERNAL_BINARY_SCORES_PROVINCE_T_HOLM8" if not failures else "FAIL",
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "audit_source_sha256": sha(__file__),
        "frozen_scorer_source_sha256": sha(score_path), "protocol_sha256": protocol_sha,
        "external_completion_sha256": sha(completion_path), "both_prediction_locks_verified_before_response_and_score_access": True,
        "locked_prediction_sources": prediction_digests, "scopes": scope_proofs,
        "district_score_rows": len(district_metrics), "excluded_district_endpoint_rows": len(excluded),
        "independent_summaries": summaries, "independent_contrasts_exactly8": contrasts,
        "independent_performance_findings": findings, "maximum_absolute_differences": maximum,
        "absolute_numeric_tolerance": ABS_TOLERANCE, "weighted_total_relative_tolerance": 2e-13,
        "comparison_failures": failures, "reported_score_file_sha256": {name: sha(OUT / name) for name in completion["output_sha256"]},
        "inference_definition": {"binary_NLL_Brier": "Positive finite survey weights normalized inside each district and endpoint; directly recomputed from binary individual responses.",
            "district_difference": "Equal-district mean of paired candidate-minus-reference binary-NLL scores.",
            "sandwich_SE": "sqrt[J/(J-1)*sum_j(sum_i_in_j(delta_i-mean_delta))^2]/N_districts",
            "t_reference": "Two-sided Student t, df=J-1; approximate province-cluster inference.",
            "Holm_family": "Exactly2scopes*2endpoints*2references=8 binary-NLL contrasts; monotone step-down adjustment.",
            "bootstrap_limits": "3999 paired-province resamples using frozen seeds and linear percentile interpolation; descriptive unadjusted limits.",
            "eligibility": "Only supplied geographic frame and at least one binary valid outcome with positive finite weight; endpoint-specific; no sample-size or favorable score filters."},
        "integrity": {"fits_or_score_sources_modified": False, "new_seeds_fits_or_favorable_subsets": False,
                      "private_child_records_exposed_in_proof": False, "latent_truth_or_latent_coverage_claimed": False}}
    destination = PHASE / "validacion/phase2_external_independent_audit.json"
    with destination.open("x") as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: proof[k] for k in ("status", "district_score_rows", "excluded_district_endpoint_rows", "independent_performance_findings", "maximum_absolute_differences")}))


if __name__ == "__main__":
    main()
