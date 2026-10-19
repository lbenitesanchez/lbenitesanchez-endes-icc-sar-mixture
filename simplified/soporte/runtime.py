"""Entradas simples por tabla/figura; fuentes y datos se comprueban antes de ejecutar.

Los adaptadores originales del paquete sencillo se ejecutan sin cambios en una
vista temporal externa: se copian fuentes verificadas y datos derivados. No se
crean enlaces, no se ajustan modelos y ninguna salida se escribe en el checkout.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.resolve()
MARKER = ".iaos_simplified_output.json"
MARKER_KIND = "endes-icc-sar-mixture:simplified:aggregate-regeneration-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relative(value):
    if not isinstance(value, str) or "\\" in value or any(x in ("", ".", "..") for x in value.split("/")):
        raise ValueError("Ruta relativa no válida: " + str(value))
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError("Se requiere una ruta relativa: " + value)
    return path


def _file(root, relative):
    root = Path(root).resolve()
    path = root / _relative(relative)
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Archivo ausente, enlazado o fuera del árbol esperado: " + relative)
    return path


def _external(path, label):
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(REPO) or REPO.is_relative_to(path):
        raise ValueError(label + " debe estar fuera del repositorio")
    return path


def _contract():
    manifest = _read(ROOT / "insumos_sha256.json")
    if set(manifest) != {"sources_sha256", "external_inputs_sha256"}:
        raise ValueError("Contrato de fuentes/insumos desconocido")
    if len(manifest["sources_sha256"]) != 18 or len(manifest["external_inputs_sha256"]) != 35:
        raise ValueError("El contrato debe contener18fuentes y35insumos externos")
    for collection in manifest.values():
        for relative, digest in collection.items():
            _relative(relative)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Huella SHA-256 no válida: " + relative)
    if any(not p.startswith(("datos/", "referencias/")) for p in manifest["external_inputs_sha256"]):
        raise ValueError("Los insumos externos deben estar en datos/ o referencias/")
    return manifest


def validar_fuentes(required=None):
    """Comprobar fuentes del repositorio, sin leer datos ni crear salidas."""
    contract = _contract()["sources_sha256"]
    names = sorted(contract) if required is None else list(required)
    for relative in names:
        if relative not in contract:
            raise ValueError("Fuente no registrada: " + relative)
        if sha(_file(REPO, relative)) != contract[relative]:
            raise RuntimeError("Fuente original modificada: " + relative)
    return {"status": "PASS", "fuentes_verificadas": len(names)}


def validar_insumos(data_root):
    """Verificar18fuentes y35CSV externos; no escribe ningún archivo."""
    data_root = _external(data_root, "--data-root")
    if not data_root.is_dir():
        raise ValueError("--data-root no es una carpeta existente")
    sources = validar_fuentes()
    inputs = _contract()["external_inputs_sha256"]
    for relative, digest in sorted(inputs.items()):
        if sha(_file(data_root, relative)) != digest:
            raise RuntimeError("Insumo externo modificado: " + relative)
    return {"status": "PASS", **{k: v for k, v in sources.items() if k != "status"},
            "insumos_externos_verificados": len(inputs), "ajustes_de_modelos_ejecutados": False}


def _output_root(data_root, output_root):
    data_root = _external(data_root, "--data-root")
    output_root = _external(output_root, "--output-root")
    if output_root.is_relative_to(data_root) or data_root.is_relative_to(output_root):
        raise ValueError("Las carpetas de datos y salidas deben ser independientes, sin solaparse")
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("--output-root debe ser una carpeta")
    if output_root.exists() and any(path.is_symlink() for path in output_root.rglob("*")):
        raise ValueError("La carpeta de salidas contiene enlaces simbólicos; utilice una carpeta sin enlaces")
    expected = {"kind": MARKER_KIND, "inputs_manifest_sha256": sha(ROOT / "insumos_sha256.json")}
    if output_root.exists() and any(output_root.iterdir()):
        marker = output_root / MARKER
        if not marker.is_file() or marker.is_symlink() or _read(marker) != expected:
            raise ValueError("La carpeta de salidas tiene archivos ajenos a esta ejecución; utilice una carpeta nueva")
    return output_root, expected


@contextmanager
def _view(data_root, output_root):
    """Copiar los53archivos verificados a una vista temporal, sin enlaces."""
    contract = _contract()
    with tempfile.TemporaryDirectory(prefix=".iaos_view_", dir=output_root) as temporary:
        view = Path(temporary)
        count = 0
        for base, entries, prefix in (
            (REPO, contract["sources_sha256"], "soporte/originales"),
            (data_root, contract["external_inputs_sha256"], ""),
        ):
            for relative, digest in sorted(entries.items()):
                source = _file(base, relative)
                if sha(source) != digest:
                    raise RuntimeError("Archivo cambiado durante la preparación: " + relative)
                destination = view / prefix / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if destination.is_symlink() or sha(destination) != digest:
                    raise RuntimeError("La copia temporal no conserva la huella: " + relative)
                count += 1
        if count != 53:
            raise RuntimeError("Vista temporal incompleta")
        yield view


def _csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.reader(stream))


def _objects():
    objects = _read(ROOT / "mapa_resultados.json")
    if len(objects) != 20 or len({x["id"] for x in objects}) != 20:
        raise ValueError("Se requieren las20entradas actuales de la propuesta")
    for obj in objects:
        _relative(obj["folder"])
        if obj["kind"] not in ("Tabla", "Figura"):
            raise ValueError("Tipo de resultado desconocido")
        if obj["kind"] == "Tabla":
            _relative(obj["reference_csv"])
    return objects


def _generate(view, obj, output_root):
    out = output_root / obj["folder"] / "salida"
    if not out.resolve().is_relative_to(output_root):
        raise ValueError("Ruta de salida fuera del directorio autorizado")
    out.mkdir(parents=True, exist_ok=True)
    if obj["kind"] == "Tabla":
        from soporte import table_adapter
        evidence = table_adapter.generate(view, obj["id"], out)
        destination = out / ("Tabla_" + obj["number"] + ".csv")
        actual, expected = _csv(destination), _csv(view / obj["reference_csv"])
        if actual != expected:
            raise RuntimeError("La tabla no coincide con la referencia externa: " + obj["id"])
        comparison = {"celdas_identicas": sum(map(len, expected)),
                      "filas": len(expected), "referencia_usada_solo_para_verificar": True}
    else:
        from soporte import figure_adapter
        evidence = figure_adapter.generate(view, obj["id"], out)
        destination = out / ("Figura_" + obj["number"] + ".png")
        if evidence["numeric_artist_fingerprint"]["sha256"] != obj["expected_numeric_artist_sha256"]:
            raise RuntimeError("Los valores dibujados difieren de la figura registrada: " + obj["id"])
        comparison = {"datos_dibujados_identicos_a_edicion_actual": True,
                      "png_identico_en_bytes": sha(destination) == obj["original_sha256"],
                      "imagen_original_sha256": obj["original_sha256"]}
    result = {"id": obj["id"], "status": "PASS", "nivel": "regeneracion_desde_insumos_derivados",
              "reajuste_de_modelos": False, "fuentes_originales_modificadas": False,
              "insumos_distribuidos_en_git": False, "vista_temporal_sin_enlaces": True,
              "salida": str(destination.relative_to(output_root)),
              "comparacion": comparison, "detalle": evidence}
    (out / "verificacion.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(obj["id"] + ": PASS -> " + str(destination.relative_to(output_root)))
    return result


def regenerar(object_ids, data_root, output_root):
    """Regenerar resultados seleccionados; entradas y salidas permanecen externas."""
    objects = _objects()
    selected = [x for x in objects if x["id"] in set(object_ids)]
    if not selected or len(selected) != len(set(object_ids)):
        raise ValueError("Selección de resultados desconocida o vacía")
    data_root = _external(data_root, "--data-root")
    output_root, marker = _output_root(data_root, output_root)
    checks = validar_insumos(data_root)
    # Todas las rutas, fuentes e insumos se validaron antes de crear salidas.
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / MARKER).write_text(json.dumps(marker, indent=2) + "\n")
    with _view(data_root, output_root) as view:
        results = [_generate(view, obj, output_root) for obj in selected]
    report = {"status": "PASS", "entradas": checks, "resultados": len(results),
              "reajuste_de_modelos": False, "fuentes_originales_modificadas": False,
              "vista_temporal_eliminada": True, "detalle": results}
    (output_root / "VERIFICACION_EJECUCION.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def main(object_id=None):
    objects = _objects()
    parser = argparse.ArgumentParser(description="Regenerar tablas/figuras desde insumos externos certificados; no reajusta modelos.")
    parser.add_argument("--data-root", type=Path, default=os.environ.get("IAOS_SIMPLE_DATA_ROOT"),
                        help="Carpeta externa con datos/ y referencias/; o IAOS_SIMPLE_DATA_ROOT")
    parser.add_argument("--output-root", type=Path, default=os.environ.get("IAOS_SIMPLE_OUTPUT_ROOT"),
                        help="Carpeta externa nueva o administrada; o IAOS_SIMPLE_OUTPUT_ROOT")
    if object_id is None:
        parser.add_argument("--solo", choices=["articulo", "suplemento"] + [x["id"] for x in objects])
    args = parser.parse_args()
    if args.data_root is None or args.output_root is None:
        parser.error("Indique --data-root y --output-root, o sus variables de entorno")
    if object_id is not None:
        ids = [object_id]
    else:
        ids = [x["id"] for x in objects if not args.solo or x["id"] == args.solo
               or x["folder"].startswith(args.solo + "/")]
    try:
        report = regenerar(ids, args.data_root, args.output_root)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, "Error: " + str(error) + "\n")
    print(f"Completado: {report['resultados']} resultados verificados; ningún modelo reajustado.")


if __name__ == "__main__":
    main()
