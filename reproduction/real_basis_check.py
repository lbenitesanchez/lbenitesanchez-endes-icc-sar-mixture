"""Diagnose an archived-versus-fresh spectral basis without fitting responses."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scipy
from scipy.sparse.csgraph import connected_components
from real_worker import locations, source_gate, write_json, sha


def diagnose(source_root, reference_root, run_root):
    study, source = locations(source_root)
    refstudy, reference = locations(reference_root)
    source_gate(study, source, refstudy)
    sys.path.insert(0, str(study / "scripts"))
    models = importlib.import_module("models")
    aux = pd.read_csv(Path(run_root) / "datos/aux_sikov14_official.csv")
    coords = aux[["latitude", "longitude"]].to_numpy(float)
    captured = {}
    original = models.eigsh
    def capture(matrix, *args, **kwargs):
        values, vectors = original(matrix, *args, **kwargs)
        captured.update(matrix=matrix, values=values, vectors=vectors, kwargs=kwargs)
        return values, vectors
    models.eigsh = capture
    try:
        fresh = models.spatial_basis(coords, rank=40)
        repeated = models.spatial_basis(coords, rank=40)
    finally:
        models.eigsh = original
    with np.load(reference / "datos/real_2019_full/fit_inputs.npz", allow_pickle=False) as data:
        archived = data["basis"]
    with np.load(Path(run_root) / "datos/real_2019_full/fit_inputs.npz", allow_pickle=False) as data:
        prepared = data["basis"]
    ws = captured["matrix"]
    component_count, component_labels = connected_components(ws, directed=False)
    values = captured["values"]
    vectors = captured["vectors"]
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    spectral_residual = ws @ vectors - vectors * values
    graph_hash = hashlib.sha256()
    for array in (ws.indptr, ws.indices, ws.data):
        graph_hash.update(array.dtype.str.encode())
        graph_hash.update(array.tobytes())
    covfresh, covold = fresh @ fresh.T, archived @ archived.T
    # Centering removes unknown constant offsets. Restore the least-squares
    # constant for each archived column and report its eigen-residual under
    # the freshly reconstructed graph. This is diagnostic only.
    retained_values = values[1:]
    ones = np.ones(len(coords))
    residuals = []
    for j, value in enumerate(retained_values):
        r = ws @ archived[:, j] - value * archived[:, j]
        direction = ws @ ones - value * ones
        offset = -float(np.dot(direction, r) / np.dot(direction, direction))
        residuals.append(float(np.max(np.abs(r + offset * direction))))
    result = {"status": "BASIS_DIFFERENCE_DISCLOSED" if not np.array_equal(fresh, archived) else "EXACT_BASIS_REPRODUCTION",
        "source_model_sha256": sha(study / "scripts/models.py"),
        "source_aux_sha256": sha(Path(run_root) / "datos/aux_sikov14_official.csv"),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "graph_sha256": graph_hash.hexdigest(), "graph_shape": list(ws.shape), "graph_nnz": ws.nnz,
        "kernel_eigensolver_v0": "linspace(.1,1,m)", "kernel_eigensolver_tolerance": captured["kwargs"]["tol"],
        "captured_eigenvalues_descending": values.tolist(),
        "minimum_retained_eigenvalue_gap": float(np.min(np.abs(np.diff(values[1:])))),
        "omitted_first_to_first_retained_eigenvalue_gap": float(values[0] - values[1]),
        "connected_components": int(component_count),
        "connected_component_sizes": np.bincount(component_labels).tolist(),
        "omitted_eigenvector_unique": bool(component_count == 1),
        "eigenvalue_rank_boundary": [float(values[-2]), float(values[-1])],
        "fresh_max_abs_eigen_residual": float(np.max(np.abs(spectral_residual))),
        "archived_centered_basis_restored_constant_max_eigen_residual": max(residuals),
        "repeated_calls_exact": bool(np.array_equal(fresh, repeated)),
        "prepared_basis_matches_unchanged_source_exactly": bool(np.array_equal(fresh, prepared)),
        "max_abs_basis_difference": float(np.max(np.abs(fresh - archived))),
        "max_abs_covariance_difference": float(np.max(np.abs(covfresh - covold))),
        "relative_frobenius_covariance_difference": float(np.linalg.norm(covfresh - covold) / np.linalg.norm(covold)),
        "raw_response_read": False, "models_fitted": False,
        "old_basis_substituted": False, "difference_hidden_or_relabeled_exact": False,
        "interpretation": "All raw parsers, direct survey quantities and auxiliary arrays are checked separately. Fresh basis is retained unchanged. A disconnected normalized graph has repeated leading eigenvalue 1, so the kernel's omitted first eigenvector is not uniquely identified. The basis differs from archived values; no exact input-reproduction claim is made for it. Reproduction of the article must be assessed after fitting and scoring, including published rounding and diagnostics."}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.source_root, args.reference_root, args.run_root)
    write_json(args.report, report)
    print(json.dumps({key: report[key] for key in ("status", "repeated_calls_exact", "max_abs_basis_difference", "relative_frobenius_covariance_difference")}), flush=True)


if __name__ == "__main__":
    main()
