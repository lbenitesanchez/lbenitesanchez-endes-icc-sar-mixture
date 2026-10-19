"""Verificar fuentes e insumos externos sin crear salidas ni ejecutar modelos."""
from pathlib import Path
import argparse
import json
import os
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from soporte.runtime import validar_insumos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("IAOS_SIMPLE_DATA_ROOT"),
                        help="Carpeta externa con datos/ y referencias/; o IAOS_SIMPLE_DATA_ROOT")
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("Indique --data-root o IAOS_SIMPLE_DATA_ROOT")
    try:
        result = validar_insumos(args.data_root)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, "Error: " + str(error) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
