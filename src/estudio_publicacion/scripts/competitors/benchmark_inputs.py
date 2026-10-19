#!/usr/bin/env python3
"""Capture unit data from frozen generators without altering random draws.

Fit-facing files never contain population outcomes or prevalence truth. The
callbacks are inserted into an in-memory AST, leaving archived source intact.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from benchmark_structural import STRUCTURAL_SCENARIOS,latent_structural,structural_spec

ROOT = Path(os.environ.get("ENDES_STUDY_ROOT",Path(__file__).resolve().parents[2]))
SIM = ROOT / "resultados" / "simulaciones"
OUT = ROOT / "resultados" / "resultados_comparadores"
PRIMARY_SCENARIOS = ("binary_informative_mar", "sparse_srs_2025", "sparse_psu_2025")
SCENARIOS = PRIMARY_SCENARIOS+STRUCTURAL_SCENARIOS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CaptureCalls(ast.NodeTransformer):
    def __init__(self, target):
        self.target = target

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if any(isinstance(t, ast.Name) and t.id == self.target for t in node.targets):
            call = ast.Expr(ast.Call(ast.Name("__capture_units", ast.Load()),
                                    [ast.Call(ast.Name("locals", ast.Load()), [], [])], []))
            return [node, ast.copy_location(call, node)]
        return node

    def visit_Return(self, node):
        node.value = ast.Call(ast.Name("__capture_finish", ast.Load()),
                              [node.value, ast.Call(ast.Name("locals", ast.Load()), [], [])], [])
        return node


class SRSBranchAlias(ast.NodeTransformer):
    """Add a structural scenario to the archived SRS branch in memory only."""
    def visit_Compare(self,node):
        node=self.generic_visit(node)
        if (isinstance(node.left,ast.Name) and node.left.id=="scenario"
            and len(node.ops)==1 and isinstance(node.ops[0],ast.Eq)
            and len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant)
            and node.comparators[0].value=="sparse_srs_2025"):
            node.ops=[ast.In()]
            node.comparators=[ast.Tuple(elts=[ast.Constant("sparse_srs_2025"),ast.Constant("sparse_srs_iid")],ctx=ast.Load())]
        return node


class UnitCapture:
    def __init__(self, scenario):
        self.scenario = scenario
        self.populations = {}
        self.units = []
        self.selected_domains = None

    def population(self, area, outcomes, X, H=None, G=None, L=None):
        if area in self.populations:
            return
        self.populations[area] = dict(outcomes=np.asarray(outcomes).reshape(-1, 2).copy(),
                                      X=np.asarray(X[area]).copy(),
                                      H=None if H is None else np.asarray(H).copy(), G=G, L=L)

    def append(self, area, children, values, observed, pi, response, psus, H=None, design=None):
        n = len(children)
        record = pd.DataFrame(dict(area=np.repeat(area, n),
            child_id=[f"{area}:{int(x)}" for x in children],
            psu=[f"{area}:{int(x)}" for x in psus], stratum=np.repeat(area, n),
            pi_sample=np.broadcast_to(pi, (n,)),
            pi_response_anemia=np.broadcast_to(np.asarray(response)[..., 0], (n,)),
            pi_response_stunting=np.broadcast_to(np.asarray(response)[..., 1], (n,))))
        for t, name in enumerate(("anemia", "stunting")):
            response_t = record[f"pi_response_{name}"].to_numpy()
            record[f"y_{name}"] = np.where(observed[:, t], values[:, t], np.nan)
            record[f"weight_{name}"] = np.where(observed[:, t], 1 / (record.pi_sample * response_t), np.nan)
        record["base_weight"] = 1 / record.pi_sample
        if design:
            for key, value in design.items():
                record[key] = value
        if H is not None:
            record["H"] = H
        self.units.append(record)

    def capture(self, v):
        if self.scenario == "binary_informative_mar":
            i = v["i"]
            if not self.populations:
                for a in range(v["m"]):
                    self.population(a, v["outcomes"][a], v["X"], v["H"][a])
            take = np.flatnonzero(v["take"])
            self.append(i, take, v["outcomes"][i, take], v["observe"][take],
                        v["inclusion"][i, take], v["resp"][i, take], take, v["H"][i, take],
                        design=dict(sampling_kind="poisson",pi_stage1=v["inclusion"][i,take],
                                    pi_stage2=1.,N_stage1=v["population_n"],N_stage2=np.nan,
                                    n_stage1=len(take),n_stage2=1))
        elif self.scenario in ("sparse_srs_2025","sparse_srs_iid"):
            a = v["area"]
            self.population(a, v["outcomes"], v["X"])
            take = v["take"]
            self.append(a, take, v["values"], v["observed"],
                        v["k"][a] / v["N"], v["response_probability"][a], take,
                        design=dict(sampling_kind="srs",pi_stage1=v["k"][a]/v["N"],
                                    pi_stage2=1.,N_stage1=v["N"],N_stage2=np.nan,
                                    n_stage1=v["k"][a],n_stage2=1))
        else:
            a = v["area"]
            self.population(a, v["outcomes"], v["X"], G=v["G"], L=v["L"])
            ids = v["psu"] * v["L"] + v["children"]
            self.append(a, ids, v["values"], v["observed"],
                        v["J"] / v["G"] * v["nj"] / v["L"],
                        v["response_probability"][a], np.repeat(v["psu"], len(ids)),
                        design=dict(sampling_kind="two_stage_srs",pi_stage1=v["J"]/v["G"],
                                    pi_stage2=v["nj"]/v["L"],N_stage1=v["G"],N_stage2=v["L"],
                                    n_stage1=v["J"],n_stage2=v["nj"]))

    def finish(self, data, v):
        if self.scenario == "binary_informative_mar":
            self.selected_domains = np.asarray(v["sampled_domains"])
        else:
            self.selected_domains = np.asarray(sorted(v["selected_areas"]))
        return data


def generate_captured(scenario, rep):
    sparse = scenario != "binary_informative_mar"
    tag = "sparse_v2" if sparse else "finite_v2"
    source = SIM / "source_snapshots" / tag / ("03b_sparse_simulation.py" if sparse else "03_simulation.py")
    module = load_module(f"competitor_frozen_{tag}", source)
    capture = UnitCapture(scenario)
    module.__dict__["__capture_units"] = capture.capture
    module.__dict__["__capture_finish"] = capture.finish
    function = "generate" if sparse else "generate_binary"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    node = CaptureCalls("observed" if sparse else "observe").visit(node)
    if scenario=="sparse_srs_iid":node=SRSBranchAlias().visit(node)
    modified = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(modified, str(source), "exec"), module.__dict__)
    previous_template = os.environ.get("ENDES_SPARSE_TEMPLATE")
    original_latent=None
    try:
        if sparse:
            os.environ["ENDES_SPARSE_TEMPLATE"] = str(source.with_name("areas_2025_6_59.csv"))
            if scenario in STRUCTURAL_SCENARIOS:
                # Each freshly loaded sparse module owns its own helper object.
                # Restore the function even on exception; never patch archive files.
                module.SCENARIOS=module.SCENARIOS+STRUCTURAL_SCENARIOS
                original_latent=module.sim.latent_truth
                coords=module.sim.geometry()[1]
                module.sim.latent_truth=lambda rng,X,basis,cfg:latent_structural(rng,X,coords,scenario)
            data = module.generate(rep, scenario)
        else:
            data = module.generate_binary(rep, scenario, *module.geometry())
    finally:
        if original_latent is not None:module.sim.latent_truth=original_latent
        if previous_template is None:
            os.environ.pop("ENDES_SPARSE_TEMPLATE", None)
        else:
            os.environ["ENDES_SPARSE_TEMPLATE"] = previous_template
    assert len(capture.populations) == 80
    recomputed = np.stack([capture.populations[a]["outcomes"].mean(axis=0) for a in range(80)])
    np.testing.assert_array_equal(recomputed, data["theta"])
    provenance=dict(generator_source=str(source), generator_sha256=sha256(source),
                    capture_source_sha256=sha256(Path(__file__)),
                    template_sha256=sha256(source.with_name("areas_2025_6_59.csv")) if sparse else None,
                    generator_metadata=data.get("metadata",{}))
    if scenario in STRUCTURAL_SCENARIOS:
        provenance.update(structural_source_sha256=sha256(Path(__file__).with_name("benchmark_structural.py")),
                          structural_specification=structural_spec(scenario,data["coords"]))
    return data, capture, provenance


def export_replication(scenario, rep, destination=None):
    folder = Path(destination) if destination else OUT / "inputs" / scenario / f"rep_{rep:05d}"
    if folder.exists():
        raise FileExistsError(folder)
    data, capture, provenance = generate_captured(scenario, rep)
    original_folder = "inputs_finite_v2" if scenario == "binary_informative_mar" else "inputs_sparse_v2"
    archived = SIM / original_folder / scenario / f"rep_{rep:04d}.npz"
    comparisons = {}
    if archived.exists():
        saved = np.load(archived)
        for key in ("theta", "y", "D", "n_eff", "sample_idx"):
            assert data[key].shape == saved[key].shape
            if np.issubdtype(data[key].dtype, np.integer):
                equal = np.array_equal(data[key], saved[key])
            else:
                equal = np.allclose(data[key], saved[key], rtol=0, atol=0, equal_nan=True)
            comparisons[key] = bool(equal)
            if not equal:
                raise AssertionError(f"Frozen regeneration differs for {scenario} {rep} {key}")
    folder.mkdir(parents=True)
    fitdir = folder / "model_inputs";fitdir.mkdir()
    evaldir = folder / "evaluation_only";evaldir.mkdir()
    X = data["X"]; coords = data["coords"]
    sample = pd.concat(capture.units, ignore_index=True)
    sample = sample[sample.area.isin(capture.selected_domains)].copy()
    sample["primary_support"] = sample.area.isin(data["sample_idx"])
    sample["X1"] = X[sample.area, 1];sample["X2"] = X[sample.area, 2]
    sample.to_csv(fitdir / "sample_all_available.csv", index=False)
    sample[sample.primary_support].to_csv(fitdir / "sample.csv", index=False)
    cells = []
    sizes = []
    for a, pop in capture.populations.items():
        N = len(pop["outcomes"]);sizes.append(N)
        h = pop["H"]
        if h is None:
            cells.append(dict(area=a, cell_id=f"{a}:0", population_weight=N, X1=X[a, 1], X2=X[a, 2]))
        else:
            for value in np.unique(h):
                cells.append(dict(area=a, cell_id=f"{a}:{int(value)}", population_weight=int(np.sum(h == value)),
                                  X1=X[a, 1], X2=X[a, 2], H=int(value)))
    population_X = pd.DataFrame(cells)
    assert not any(c.startswith("y_") or c.startswith("theta") for c in population_X.columns)
    population_X.to_csv(fitdir / "population_X.csv", index=False)
    area = pd.DataFrame(dict(area=np.arange(80), X1=X[:, 1], X2=X[:, 2],
                             coord_x=coords[:, 0], coord_y=coords[:, 1], N_population=sizes,
                             sampled_common=np.isin(np.arange(80), data["sample_idx"]),
                             sampling_selected=np.isin(np.arange(80), capture.selected_domains)))
    area.to_csv(fitdir / "area.csv", index=False)
    # Generating covariance is evaluation-only. Estimators receive a separate
    # basis made from public coordinates and the frozen application algorithm.
    np.save(evaldir / "generating_basis.npy",data["basis"])
    geometry_models=load_module("benchmark_geometry_models",SIM/"source_snapshots"/"sparse_v2"/"models.py")
    np.save(fitdir / "fit_basis.npy",geometry_models.spatial_basis(coords,rank=12,neighbors=8))
    distance = np.sqrt(np.sum((coords[:, None] - coords[None, :])**2, axis=2))
    edge = np.transpose(np.where(np.triu(np.isclose(distance, 1/8, atol=1e-12), 1)))
    pd.DataFrame(edge, columns=["area1", "area2"]).to_csv(fitdir / "adjacency_edges.csv", index=False)
    direct = {k: data[k] for k in ("y", "D", "n_eff", "sample_idx")}
    if "df" in data:
        direct["df_original_proxy"] = data["df"]
    direct["df_primary"] = np.maximum(data["n_eff"]-1, 1) if scenario in ("sparse_srs_2025","sparse_srs_iid") else data.get("df", np.maximum(data["n_eff"]-1, 1))
    np.savez_compressed(fitdir / "direct.npz", **direct)
    np.savez_compressed(evaldir / "truth.npz", theta=data["theta"],
                        sample_idx=data["sample_idx"], sampling_selected=capture.selected_domains)
    np.savez_compressed(evaldir / "population_outcomes.npz",
                        **{f"area_{a:03d}": p["outcomes"].astype(np.uint8) for a, p in capture.populations.items()})
    manifest = dict(scenario=scenario,replication=rep,synthetic=True,primary_support="stored common complete-pair support for finite, partial endpoints retained for sparse",
                    secondary_sample_file="sample_all_available.csv retains original partial pairs",
                    truth_available_to_estimators=False, population_X_is_exact_feature_cell_compression=True,
                    unit_features=["X1", "X2"] + (["H"] if "H" in sample else []),
                    information_equal_features=["X1", "X2"], adjacency="known rook-grid; zero-based area ids",
                    fit_basis="rank12 symmetric8NN graph spectral basis from coordinates only; no generating covariance",
                    saved_input_exact_comparison=comparisons, new_independent_population=not archived.exists(),
                    sampled_children=len(sample),primary_sampled_children=int(sample.primary_support.sum()),
                    eligible_population=sum(sizes),**provenance)
    manifest["input_sha256"] = {p.name: sha256(p) for p in fitdir.iterdir() if p.is_file()}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def add_coordinate_fit_basis(folder):
    """Add coordinate-only geometry to existing development inputs atomically.

    Original CSV/NPZ/manifest and generating-basis files remain archived. This
    additional provenance artifact labels old oracle geometry as diagnostic.
    """
    folder=Path(folder);fitdir=folder/"model_inputs"
    area=pd.read_csv(fitdir/"area.csv").sort_values("area")
    coords=area[["coord_x","coord_y"]].to_numpy(float)
    source=SIM/"source_snapshots"/"sparse_v2"/"models.py"
    geometry=load_module("benchmark_geometry_models",source)
    basis=geometry.spatial_basis(coords,rank=12,neighbors=8)
    path=fitdir/"fit_basis.npy"
    if path.exists():np.testing.assert_allclose(np.load(path),basis,rtol=0,atol=1e-12)
    else:
        temporary=fitdir/"fit_basis.new.npy";np.save(temporary,basis);temporary.replace(path)
    note=dict(fit_basis="rank12 symmetric8NN graph spectral; coordinates only",
              fit_basis_sha256=sha256(path),coordinates_file_sha256=sha256(fitdir/"area.csv"),
              geometry_source_sha256=sha256(source),outcomes_used=False,
              prior_generating_basis="Historical spatial_basis.npy is development oracle diagnostic only; never selected by graph final runner")
    (folder/"coordinate_geometry.json").write_text(json.dumps(note,indent=2))
    return note


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--replications", nargs="+", type=int, default=[0])
    args=parser.parse_args()
    for scenario in args.scenarios:
        for rep in args.replications:
            print(json.dumps(export_replication(scenario, rep)), flush=True)


if __name__ == "__main__":
    main()
