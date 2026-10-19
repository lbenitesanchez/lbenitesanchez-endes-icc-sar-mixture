"""Explicit common-information extension, not the published Gao estimator.

Imports the frozen primary variance/continuity preparation without editing it.
Only model_inputs/direct.npz is read; simulation truth is never accessed.
"""
from pathlib import Path
import sys, argparse, json, hashlib
import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY / "scripts"))
from models import prepare_observations


def prepare_synthetic(indir, outfile):
    indir, outfile = Path(indir), Path(outfile)
    if indir.name != "model_inputs":
        raise ValueError("Only blinded model_inputs permitted")
    a = np.load(indir / "direct.npz")
    area = pd.read_csv(indir / "area.csv", dtype={"area":str})
    ob = prepare_observations(a["y"], a["D"], a["n_eff"],
        transform="logit", D_mode="smooth_biascorrected", df=a["df_primary"])
    rows = []
    for j, name in enumerate(("anemia", "stunting")):
        for i, idx in enumerate(a["sample_idx"]):
            rows.append({"area":area.area.iloc[int(idx)], "outcome":name,
                         "z":ob["z"][i,j], "var_z":ob["D"][i,j,j]})
    outfile.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outfile, index=False)
    outfile.with_suffix(".json").write_text(json.dumps({
        "classification":"Explicit common-continuity and regularized-D extension; not published Gao",
        "D_mode":"smooth_biascorrected", "df":"exported df_primary; SRS uses n-1 matched baseline",
        "input_file":"model_inputs/direct.npz", "input_sha256":hashlib.sha256((indir/"direct.npz").read_bytes()).hexdigest(),
        "primary_models_source_sha256":hashlib.sha256((STUDY/"scripts/models.py").read_bytes()).hexdigest(),
        "truth_access":False,"preparation":ob["metadata"]},indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model_inputs"); p.add_argument("output_csv")
    args=p.parse_args();prepare_synthetic(args.model_inputs,args.output_csv)
