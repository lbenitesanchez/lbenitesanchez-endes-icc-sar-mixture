#!/usr/bin/env python3
"""Verify common-sample identities and estimator/evaluator separation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_inputs import OUT,SCENARIOS,PRIMARY_SCENARIOS,sha256


def validate(folder):
    metadata=json.loads((folder/"manifest.json").read_text())
    fit=folder/"model_inputs";ev=folder/"evaluation_only"
    units=pd.read_csv(fit/"sample.csv");area=pd.read_csv(fit/"area.csv")
    aux=pd.read_csv(fit/"population_X.csv");direct=np.load(fit/"direct.npz")
    truth=np.load(ev/"truth.npz");population=np.load(ev/"population_outcomes.npz")
    assert not any(col.startswith(("y_","theta")) for col in aux.columns)
    assert not any(col.startswith("theta") for col in area.columns)
    assert not units.child_id.duplicated().any()
    np.testing.assert_array_equal(aux.groupby("area").population_weight.sum(),area.N_population)
    reconstructed=[];ne=[]
    for a in direct["sample_idx"]:
        subset=units[units.area==a];yd=[];nd=[]
        for name in ("anemia","stunting"):
            seen=subset[f"y_{name}"].notna();w=subset.loc[seen,f"weight_{name}"].to_numpy()
            y=subset.loc[seen,f"y_{name}"].to_numpy()
            yd.append(np.sum(w*y)/w.sum() if len(w) else np.nan)
            nd.append(w.sum()**2/np.sum(w*w) if len(w) else 0.)
        reconstructed.append(yd);ne.append(nd)
    np.testing.assert_allclose(reconstructed,direct["y"],rtol=0,atol=1e-14,equal_nan=True)
    np.testing.assert_allclose(ne,direct["n_eff"],rtol=1e-13,atol=1e-12)
    for a in range(80):
        pop=population[f"area_{a:03d}"]
        np.testing.assert_array_equal(pop.mean(axis=0),truth["theta"][a])
        subset=units[units.area==a]
        ids=subset.child_id.str.split(":").str[-1].astype(int).to_numpy()
        for j,name in enumerate(("anemia","stunting")):
            seen=subset[f"y_{name}"].notna().to_numpy()
            np.testing.assert_array_equal(subset.loc[seen,f"y_{name}"],pop[ids[seen],j])
    for name,expected in metadata["input_sha256"].items():
        assert sha256(fit/name)==expected,(folder,name)
    if metadata["scenario"] in PRIMARY_SCENARIOS and metadata["replication"]<20:
        assert len(metadata["saved_input_exact_comparison"])==5
        assert all(metadata["saved_input_exact_comparison"].values())
    return dict(scenario=metadata["scenario"],replication=metadata["replication"],
        primary_children=len(units),archived_exact_match=metadata["saved_input_exact_comparison"],
        weighted_point_and_Kish_reconstruction=True,population_truth_exact=True,
        observed_child_outcomes_exact=True,truth_excluded_from_fit_inputs=True,input_hashes_passed=True)


def main():
    checks=[validate(folder) for scenario in SCENARIOS
            for folder in sorted((OUT/"inputs"/scenario).glob("rep_*"))]
    result=dict(passed=True,inputs=len(checks),checks=checks,
                evaluator_source_sha256=sha256(Path(__file__)))
    (OUT/"input_validation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({"passed":True,"inputs":len(checks)}))


if __name__=="__main__":main()
