#!/usr/bin/env python3
"""Check portable core/sparse imports without production fits or network access."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    sparse = load("portable_sparse", CODE / "03b_sparse_simulation.py")
    model = sparse.sim.simulation_models()
    assert Path(model.__file__).resolve() == (CODE / "models.py").resolve()
    checks = {"sibling_model_source_resolved": True, "production_fits_run": False, "network_access": False}
    for scenario in sparse.SCENARIOS:
        data = sparse.generate(0, scenario)
        assert data["y"].shape[1] == 2
        assert data["D"].shape == (len(data["y"]), 2, 2)
        assert np.isfinite(data["theta"]).all()
        checks[scenario] = {"observed_domains": len(data["y"]), "passed": True}
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
