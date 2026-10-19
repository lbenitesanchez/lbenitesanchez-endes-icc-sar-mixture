"""Fixed 75% ICC beta + 25% SAR mixture using the unchanged original function.

ICC and SAR must cover the same districts and endpoints in the same order.
The SAR component is full-graph with bootstrap MSE. This entry checks its source
without requiring the external aggregate/reference CSVs used for presentation.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from support.runtime import REPO, verify_sources, _external

KERNEL = "src/estudio_publicacion/sikov_2026-10-10/scripts/improve_candidates.py"


def mix_components(icc, sar):
    """Apply the original mixture-CDF algorithm with ICC weight fixed at 0.75."""
    verify_sources(required=[KERNEL])
    source = REPO / KERNEL
    spec = importlib.util.spec_from_file_location("iaos_english_mixture_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mixture_prediction(icc, sar, icc_weight=0.75, alpha=0.05)


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icc", type=Path, required=True,
                        help="NPZ with p_mean, p_variance, and p_draws or marginal_draws_0/1")
    parser.add_argument("--sar", type=Path, required=True, help="NPZ with mean and mse on the same support")
    parser.add_argument("--output", type=Path, help="New external NPZ output file")
    parser.add_argument("--salida", type=Path, dest="output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output is None:
        parser.error("Provide --output")
    try:
        icc_path = _external(args.icc, "--icc")
        sar_path = _external(args.sar, "--sar")
        output = _external(args.output, "--output")
        if output.suffix.lower() != ".npz":
            raise ValueError("--output must end in .npz")
        if output.exists():
            raise ValueError("An existing output file will not be overwritten")
        def load(path):
            with np.load(path, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}
        icc, sar = load(icc_path), load(sar_path)
        marginal = sorted((key for key in icc if key.startswith("marginal_draws_")),
                          key=lambda key: int(key.rsplit("_", 1)[1]))
        if marginal:
            icc["marginal_draws"] = [icc[key] for key in marginal]
        result = mix_components(icc, sar)
        status = result.pop("status")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **result, status_json=json.dumps(status))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, "Error: " + str(error) + "\n")
    print("Saved the fixed 75/25 mixture to " + str(output))


# Compatibility for callers of the previous interface.
combinar_componentes = mix_components


if __name__ == "__main__":
    main()
