"""Propuesta75% ICC beta +25% SAR mediante la función original, sin reajuste.

ICC ySAR deben cubrir los mismos distritos, desenlaces y orden. La componente
SAR es full-graph con MSE bootstrap. Esta entrada comprueba la fuente original;
no exige las tablas de referencia utilizadas por los programas de presentación.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from soporte.runtime import REPO, validar_fuentes, _external

KERNEL = "src/estudio_publicacion/sikov_2026-10-10/scripts/improve_candidates.py"


def combinar_componentes(icc, sar):
    """Aplicar el algoritmo original de la CDF con peso ICC fijo0.75."""
    validar_fuentes(required=[KERNEL])
    source = REPO / KERNEL
    spec = importlib.util.spec_from_file_location("iaos_simple_mixture_kernel", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mixture_prediction(icc, sar, icc_weight=0.75, alpha=0.05)


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icc", type=Path, required=True, help="NPZ con p_mean,p_variance,p_draws o marginal_draws_0/1")
    parser.add_argument("--sar", type=Path, required=True, help="NPZ con mean,mse sobre el mismo soporte")
    parser.add_argument("--salida", type=Path, required=True, help="NPZ nuevo fuera del repositorio")
    args = parser.parse_args()
    try:
        icc_path = _external(args.icc, "--icc")
        sar_path = _external(args.sar, "--sar")
        output = _external(args.salida, "--salida")
        if output.suffix.lower() != ".npz":
            raise ValueError("--salida debe terminar en .npz")
        if output.exists():
            raise ValueError("No se sobrescribe una salida existente")
        def load(path):
            with np.load(path, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}
        icc, sar = load(icc_path), load(sar_path)
        marginal = sorted((key for key in icc if key.startswith("marginal_draws_")),
                          key=lambda key: int(key.rsplit("_", 1)[1]))
        if marginal:
            icc["marginal_draws"] = [icc[key] for key in marginal]
        result = combinar_componentes(icc, sar)
        status = result.pop("status")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **result, status_json=json.dumps(status))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, "Error: " + str(error) + "\n")
    print("Mezcla75/25 guardada en " + str(output))


if __name__ == "__main__":
    main()
