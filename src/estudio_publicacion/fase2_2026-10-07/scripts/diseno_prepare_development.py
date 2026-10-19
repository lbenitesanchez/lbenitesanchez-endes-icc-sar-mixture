"""Prepare real development calibration inputs without model fitting or scoring."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from diseno_calibration import CONFIG, calibrate, digest, frame_from_households, stratified_psu_counts, write_summary

PHASE = Path(__file__).resolve().parents[1]
STUDY = PHASE.parent
sys.path.insert(0, str(STUDY/"scripts/competitors"))
sys.path.insert(0, str(STUDY/"scripts"))
from models import spatial_basis


def load_dataset(year, aux_path=None):
    if year not in (2019, 2025):
        raise ValueError("Development preparation forbids reserved outcomes")
    spec = importlib.util.spec_from_file_location("phase2_development_dataset", STUDY/"scripts/02_application.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if aux_path is None:
        data = module.dataset(year, "who2024")
    else:
        aux_path=Path(aux_path).resolve()
        if not aux_path.is_relative_to(PHASE):
            raise ValueError("New official auxiliary must be under this phase")
        aux=pd.read_csv(aux_path,dtype={"ubigeo":str,"dept_code":str,"prov_code":str})
        if not aux.ubigeo.str.fullmatch(r"\d{6}").all() or aux.ubigeo.duplicated().any():
            raise ValueError("Auxiliary UBIGEO must be unique six-digit codes")
        forbidden=[c for c in aux if c.startswith(("y_","var_","cov_","n_eff_","se_"))
                   or c in ("direct_anemia","direct_stunting","weight","psu_key","household_id","child_id")]
        if forbidden:
            raise ValueError("Outcome/survey-only columns cannot define census auxiliary eligibility: "+str(forbidden))
        for field,nchar in (("dept_code",2),("prov_code",4)):
            if field not in aux:
                aux[field]=aux.ubigeo.str[:nchar]
            if not np.array_equal(aux[field].to_numpy(str),aux.ubigeo.str[:nchar].to_numpy(str)):
                raise ValueError("Auxiliary parent geography disagrees with its UBIGEO")
        features=module.FEATURES
        required=features+["latitude","longitude"]
        if not set(required).issubset(aux.columns):
            raise ValueError("New auxiliary lacks prespecified features: "+str(set(required)-set(aux.columns)))
        good=np.all(np.isfinite(aux[required].to_numpy(float)),axis=1)
        excluded=aux.loc[~good].copy()
        aux=aux.loc[good].sort_values("ubigeo").reset_index(drop=True)
        source=STUDY/f"datos/processed/areas_{year}_6_59.csv"
        areas=pd.read_csv(source,dtype={"ubigeo":str,"dept_code":str,"prov_code":str})
        columns=[c for c in areas if c not in aux or c=="ubigeo"]
        allareas=aux.merge(areas[columns],on="ubigeo",how="left",validate="one_to_one")
        unsupported=areas.loc[~areas.ubigeo.isin(aux.ubigeo)].copy()
        raw=allareas[features].to_numpy(float)
        center=raw.mean(axis=0);scale=raw.std(axis=0)
        if np.any(scale<=0):
            raise ValueError("Prespecified auxiliary has a constant covariate")
        X=np.column_stack((np.ones(len(raw)),(raw-center)/scale))
        all_y=allareas[["y_anemia_who2024","y_stunting"]].to_numpy(float)
        idx=np.flatnonzero(np.any(np.isfinite(all_y),axis=1))
        observed=allareas.iloc[idx];y=all_y[idx]
        D=np.zeros((len(idx),2,2))
        D[:,0,0]=observed.var_anemia_who2024
        D[:,1,1]=observed.var_stunting
        D[:,0,1]=D[:,1,0]=observed.cov_who_stunting
        ne=observed[["n_eff_anemia_who2024","n_eff_stunting"]].to_numpy(float)
        df=np.column_stack((np.maximum(observed.n_psu_anemia_who2024.to_numpy(float)-observed.n_strata_anemia_who2024.to_numpy(float),1),
                            np.maximum(observed.n_psu_stunting.to_numpy(float)-observed.n_strata_stunting.to_numpy(float),1)))
        manifest=dict(year=year,rule="who2024",features=features,X_center=center.tolist(),X_scale=scale.tolist(),
                      auxiliary_path=str(aux_path.relative_to(PHASE)),auxiliary_sha256=digest(aux_path),
                      census_areas=len(allareas),sampled_areas=len(idx),unsupported_survey_codes=unsupported.ubigeo.tolist(),
                      excluded_auxiliary_codes=excluded.ubigeo.tolist(),
                      auxiliary_eligibility_uses_only_census_covariates_and_coordinates=True,
                      survey_outcome_columns_used_to_filter_auxiliary=False,
                      geography_split_or_response_dependent_imputation_applied=False,
                      direct_source_sha256=digest(source))
        data=allareas,X,y,D,ne,idx,df,unsupported,manifest
    allareas, X, y, D, ne, idx, df, unsupported, manifest = data
    households = pd.read_csv(STUDY/f"datos/private_competitors/households_{year}.csv", dtype={"ubigeo":str})
    units = pd.read_csv(STUDY/f"datos/private_competitors/children_{year}.csv", dtype={"ubigeo":str})
    lookup = dict(zip(allareas.ubigeo, np.arange(len(allareas))))
    units["area"] = units.ubigeo.map(lookup)
    units = units.loc[units.area.notna()].copy()
    units["area"] = units.area.astype(int)
    units["y_anemia"] = units.y_anemia_who2024
    units["psu"] = units.psu_key
    for name in ("anemia", "stunting"):
        units[f"weight_{name}"] = np.where(np.isfinite(units[f"y_{name}"]), units.weight, np.nan)
    frame = frame_from_households(households)
    return data, frame, units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, choices=[2019,2025], required=True)
    parser.add_argument("--scope", choices=["full","territorial_training"], default="full")
    parser.add_argument("--aux",type=Path,help="Verified updated auxiliary CSV under this new phase")
    parser.add_argument("--output",type=Path,help="Optional new preparation output under this phase")
    args = parser.parse_args()
    data, frame, units = load_dataset(args.year,args.aux)
    allareas,X,y,D,ne,idx,df,unsupported,manifest = data
    out = args.output.resolve() if args.output else PHASE/f"datos/diseno/{args.year}_{args.scope}{'_official' if args.aux else ''}"
    if not out.is_relative_to(PHASE):
        raise ValueError("Prepared output must be under new phase")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    original_idx = idx.copy()
    test_idx = np.array([], dtype=int)
    if args.scope == "territorial_training":
        fold_path = STUDY/f"resultados/aplicacion_{args.year}_who2024/validation_folds.csv"
        folds = pd.read_csv(fold_path,dtype={"ubigeo":str}).set_index("ubigeo").loc[allareas.iloc[idx].ubigeo]
        held = folds.locked_test.to_numpy(bool)
        test_idx = idx[held]
        take = np.flatnonzero(~held)
        idx,y,D,ne = idx[take],y[take],D[take],ne[take]
        units = units.loc[units.area.isin(idx)].copy()
        if set(units.area) & set(test_idx):
            raise AssertionError("Held-out outcome records passed preparation boundary")
    counts, replicate_checks = stratified_psu_counts(frame)
    basis = spatial_basis(allareas[["latitude","longitude"]].to_numpy(float), rank=40)
    np.savez_compressed(out/"fit_inputs.npz", X=X, basis=basis, sample_idx=idx, y=y,
                        kish_n=ne, D=D, test_idx=test_idx)
    allareas[["ubigeo","prov_code","dept_code"]].to_csv(out/"area_ids.csv", index=False)
    unsupported.to_csv(out/"unsupported_direct_only.csv",index=False)
    (out/"auxiliary_model_metadata.json").write_text(json.dumps(manifest,indent=2))
    checks=[]
    for j,name in enumerate(("anemia","stunting")):
        result = calibrate(frame, units, idx, name, counts=counts)
        dy = float(np.nanmax(abs(result["direct"]["y"]-y[:,j])))
        dn = float(np.nanmax(abs(result["direct"]["n_kish"]-ne[:,j])))
        dv = float(np.nanmax(abs(result["direct"]["variance"]-D[:,j,j])))
        if dy>1e-10 or dn>1e-8 or dv>1e-10:
            raise AssertionError((name,dy,dn,dv))
        result["status"]["replicate_checks"] = replicate_checks
        result["status"].update(year=args.year, scope=args.scope, forbidden_reserved_year_read=False,
                                heldout_training_overlap=0)
        write_summary(result,out,name)
        checks.append(dict(outcome=name,max_direct_difference=dy,max_Kish_difference=dn,
                           max_Taylor_variance_difference=dv,
                           singleton_domains=int(np.sum(result["direct"]["singleton_contribution"]>0)),
                           central_n_finite=int(np.sum(np.isfinite(result["central"]["effective_n"]))),
                           boundary_count=int(np.sum(np.isin(result["direct"]["y"],[0,1])))))
    input_paths = [STUDY/f"datos/private_competitors/households_{args.year}.csv",
                   STUDY/f"datos/private_competitors/children_{args.year}.csv",
                   STUDY/f"datos/processed/areas_{args.year}_6_59.csv",
                   args.aux.resolve() if args.aux else STUDY/"datos/processed/aux_all_districts.csv"]
    proof = dict(status="PASS_DEVELOPMENT_DESIGN_CALIBRATION_PREPARATION", year=args.year, scope=args.scope,
                 model_fits_dispatched=False, config=CONFIG, original_sampled_domains=len(original_idx),
                 training_domains=len(idx), heldout_domains=len(test_idx), comparisons=checks,
                 full_household_design_used=True, reserved_2024_read=False,
                 auxiliary_metadata=manifest,
                 input_sha256={str(p.relative_to(STUDY)):digest(p) for p in input_paths},
                 source_sha256={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name("diseno_calibration.py")]})
    (out/"preparation_manifest.json").write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof,indent=2))


if __name__ == "__main__":
    main()
