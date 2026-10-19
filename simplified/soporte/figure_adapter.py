"""Regenerar las figuras actuales desde agregados corregidos, sin ajustar modelos.

Se ejecutan únicamente los bloques de dibujo de los programas originales.
Las comprobaciones de ajustes pertenecen al análisis de origen: este adaptador
no inventa ni sustituye sus locks y no representa una reestimación desde cero.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORIGINALS = Path("soporte/originales")
SCRIPTS = Path("src/estudio_publicacion/sikov_2026-10-10/scripts")
REVISION = Path("revision_20261015/code")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helper(root):
    path = root / ORIGINALS / REVISION / "terminology_20261018/render_proposal_figures.py"
    spec = importlib.util.spec_from_file_location("iaos_plot_identity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def selected_plot(source, kind, helper):
    """Seleccionar el dibujo original; excluir el ajuste, gates y escrituras."""
    if kind not in ("interval", "sparse57"):
        return helper.plot_nodes(source, kind)
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == ("sparse57" if kind == "sparse57" else "main"))
    if kind == "interval":
        start = next(i for i, node in enumerate(function.body) if helper.assignment(node, "raw"))
        end = next(i for i, node in enumerate(function.body) if helper.assignment(node, "report"))
    else:
        start = next(i for i, node in enumerate(function.body)
                     if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                     and ast.unparse(node.value.func) == "plt.rcParams.update")
        end = next(i for i, node in enumerate(function.body)
                   if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                   and ast.unparse(node.value.func) == "fig.savefig")
    body = [copy.deepcopy(node) for node in function.body[start:end] if not (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) in ("fig.savefig", "plt.close"))]
    return ast.Module(body=body, type_ignores=[])


def generate(root, object_id, output_dir):
    """Crear PNG y SVG para Figura_1/2/3/S1/S2 y devolver sus comprobantes.

    root es la carpeta sencilla. output_dir es el directorio de salida de la
    tabla/figura elegida. Ningún fichero de entrada se modifica.
    """
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    requested_id = object_id
    object_id = object_id.replace(" ", "_")
    for prefix in ("Articulo_", "Suplemento_"):
        if object_id.startswith(prefix):
            object_id = object_id[len(prefix):]
    aliases = {"figure1": "Figura_1", "figure2": "Figura_2", "figure3": "Figura_3",
               "figureS1": "Figura_S1", "figureS2": "Figura_S2"}
    object_id = aliases.get(object_id, object_id)
    helper, helper_path = load_helper(root)
    base = root / "datos/sikov"
    inputs = []
    replacements = {}
    extra = {}
    if object_id in ("Figura_1", "Figura_S1"):
        kind = "principal" if object_id == "Figura_1" else "interval"
        source = root / ORIGINALS / SCRIPTS / "comparison_figures.py"
        folder = base / "resultados/published_sar_comparison"
        extra.update(source=folder)
        if kind == "principal":
            inputs = [folder / "summary.csv"]
        else:
            inputs = [folder / "paired_balanced_contrasts_Holm3.csv",
                      folder / "physical_projection_contrasts.csv"]
            extra["REFERENCES"] = ("Sikov_FH", "Sikov_SAR", "Sikov_SAR_Published")
            replacements = {
                "Frozen improvement on 256 fresh finite populations": "Proposed mixture on 256 finite populations",
                "Difference in balanced interval score (lower favours improvement)":
                    "Difference in balanced interval score (lower favours proposed mixture)"}
    elif object_id == "Figura_2":
        kind = "incremental"
        source = root / ORIGINALS / REVISION / "incremental_icc.py"
        inputs = [root / "datos/incremental/incremental_summary.csv"]
        extra.update(summary=pd.read_csv(inputs[0]))
        replacements = {"Selected − ICC": "Proposed mixture − ICC"}
    elif object_id == "Figura_3":
        kind = "sparse57"
        source = root / ORIGINALS / REVISION / "cohort_extensions.py"
        inputs = [root / "datos/sparse57/sparse57_exploratory_summary.csv"]
        frame = pd.read_csv(inputs[0])
        metrics = ["interval_score", "RMSE_sqrt_mean_MSE", "mean_population_RMSE",
                   "bias", "coverage", "width", "aggregate_coverage_distance95"]
        selected = frame.loc[frame.scope.eq("equal_four_contexts") & frame.metric.isin(metrics)
                             & frame.outcome.eq("endpoint_balanced") & frame["mask"].eq("all")]
        refs = ["ICC", "Sikov_FH", "Sikov_SAR", "Sikov_SAR_Published"]
        if len(selected) != len(refs) * len(metrics):
            raise ValueError("La sensibilidad sparse57 no contiene todos los agregados esperados")
        extra.update(global_all=selected, refs=refs, inc=SimpleNamespace(Z=helper.Z))
    elif object_id == "Figura_S2":
        kind = "map"
        source = root / ORIGINALS / SCRIPTS / "comparison_application_figures.py"
        inputs = [base / "resultados/aplicacion2019/matched_application2019_districts.csv",
                  root / "datos/mapa/area_frame.csv",
                  root / "datos/mapa/district_availability_for_map.csv"]
        data, areas, availability = [pd.read_csv(path, dtype={"ubigeo": str}) for path in inputs]
        if areas.ubigeo.duplicated().any() or len(areas) != 1874:
            raise ValueError("El mapa necesita el marco histórico de 1,874 distritos")
        mask = availability.set_index("ubigeo").loc[areas.ubigeo]
        ordered = {method: group.set_index("ubigeo").loc[areas.ubigeo]
                   for method, group in data.groupby("method")}
        for method in helper.METHODS:
            if method not in ordered:
                raise ValueError("Falta el método del mapa: " + method)
            for coordinate in ("latitude", "longitude"):
                if not np.array_equal(ordered[method][coordinate].to_numpy(), areas[coordinate].to_numpy()):
                    raise ValueError("Coordenadas incompatibles en el mapa")
        flags = {}
        for outcome in ("anemia", "stunting"):
            values = mask[outcome + "_observed"]
            if values.dtype != bool:
                values = values.astype(str).str.lower().map({"true": True, "false": False})
                if values.isna().any():
                    raise ValueError("Máscara de observación desconocida")
            flags[outcome] = values.to_numpy(bool)
        extra.update(ordered=ordered, ENDPOINTS=("anemia", "stunting"),
                     lon=areas.longitude.to_numpy(), lat=areas.latitude.to_numpy(), observed=flags)
        replacements = {": selected mixture": ": Proposed mixture",
                        ": mixture minus sampled-graph SAR": ": Proposed mixture minus sampled-graph SAR"}
    else:
        raise ValueError("Figura desconocida: " + object_id)

    before = selected_plot(source, kind, helper)
    after = helper.ReplaceText(replacements).visit(copy.deepcopy(before))
    before_ast_sha = hashlib.sha256(ast.dump(before, include_attributes=False).encode()).hexdigest()
    after_ast_sha = hashlib.sha256(ast.dump(after, include_attributes=False).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric = []
    produced = []
    for updated, nodes in ((False, before), (True, after)):
        plt.rcdefaults()
        labels = ("Sikov FH", "Full-graph SAR", "Published-scope SAR", "ICC beta",
                  "Proposed mixture" if updated else "Frozen 75/25 mixture")
        env = dict(np=np, pd=pd, plt=plt, Path=Path, ROOT=root, out=output_dir,
                   CONTEXTS=helper.CONTEXTS, NAMES=helper.NAMES, METHODS=helper.METHODS,
                   COLORS=helper.COLORS, Z=helper.Z, LABELS=labels)
        env.update(extra)
        exec(compile(ast.fix_missing_locations(nodes), str(source), "exec"), env)
        fig = env["fig"]
        # sparse57 guardó el PNG antes de cualquier canvas.draw adicional.
        # Conservar ese orden evita variaciones diminutas de constrained_layout.
        if not (updated and kind == "sparse57"):
            numeric.append(helper.artist_fingerprint(fig))
        if updated:
            kwargs = {"bbox_inches": "tight"} if kind in ("incremental", "sparse57") else {}
            for extension in ("png", "svg"):
                path = output_dir / (object_id + "." + extension)
                fig.savefig(path, dpi=250 if kind in ("incremental", "sparse57") else 300, **kwargs)
                produced.append(path)
        if updated and kind == "sparse57":
            numeric.append(helper.artist_fingerprint(fig))
        plt.close(fig)
    if numeric[0] != numeric[1]:
        raise ValueError("El cambio de texto alteró los valores dibujados")
    evidence = dict(status="PASS_REGENERATED_FROM_CORRECTED_AGGREGATES", object_id=requested_id,
                    refitted_models=False, copied_existing_pixels=False,
                    source={str(source.relative_to(root)): sha(source),
                            str(helper_path.relative_to(root)): sha(helper_path)},
                    inputs={str(path.relative_to(root)): sha(path) for path in inputs},
                    output_files={path.name: sha(path) for path in produced},
                    numeric_artist_fingerprint=numeric[1],
                    numeric_artists_unchanged_by_labels=True,
                    plotting_ast_before_sha256=before_ast_sha,
                    plotting_ast_after_sha256=after_ast_sha,
                    label_replacements=replacements,
                    provenance_scope="Dibujo desde agregados certificados; no reproduce por sí solo ajustes ni bootstrap.")
    (output_dir / (object_id + "_comprobante.json")).write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return evidence
