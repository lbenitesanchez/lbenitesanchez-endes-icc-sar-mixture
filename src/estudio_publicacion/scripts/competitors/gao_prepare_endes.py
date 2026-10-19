"""Blinded aggregate ENDES inputs for the locked Gao application comparison.

Stage 1 with constant district X was verified to equal the original direct
estimates. This export uses that equality, and never exposes test outcomes to
the linking model. Shared continuity/GVF is fitted only on the training areas.
"""
from pathlib import Path
import sys, importlib.util, json, hashlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
STUDY=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(STUDY/"scripts"))
from models import prepare_observations
spec=importlib.util.spec_from_file_location("endes_primary_app",STUDY/"scripts/02_application.py")
app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app)


def export(year):
    areas,X,y,D,ne,idx,df,unsupported,manifest=app.dataset(year,"who2024")
    folds=pd.read_csv(STUDY/f"resultados/aplicacion_{year}_who2024/validation_folds.csv",dtype={"ubigeo":str})
    folds=folds.set_index("ubigeo")
    test=folds.locked_test.reindex(areas.ubigeo.iloc[idx]).fillna(False).to_numpy(bool)
    for split in ("full","locked_train"):
        train=np.ones(len(idx),bool) if split=="full" else ~test
        ob=prepare_observations(y[train],D[train],ne[train],transform="logit",
                               D_mode="smooth_biascorrected",df=df[train])
        out=STUDY/f"resultados/competidores_gao/inputs/endes_{year}_{split}"
        out.mkdir(parents=True,exist_ok=True)
        a=pd.DataFrame({"area":areas.ubigeo,"latitude":areas.latitude,"longitude":areas.longitude})
        for j,x in enumerate(app.FEATURES):a[x]=X[:,j+1]
        a.to_csv(out/"area.csv",index=False)
        xy=np.c_[111.32*a.longitude.to_numpy()*np.cos(np.deg2rad(a.latitude.mean())),
                 111.32*a.latitude.to_numpy()]
        d=cdist(xy,xy);np.fill_diagonal(d,np.inf)
        jj=np.argpartition(d,7,axis=1)[:,:8]
        pd.DataFrame({"area1":np.repeat(a.area.to_numpy(),8),
                      "area2":a.area.to_numpy()[jj.ravel()]}).to_csv(out/"adjacency_edges.csv",index=False)
        for j,name in enumerate(("anemia","stunting")):
            est=np.full(len(a),np.nan);var=est.copy();z=est.copy();vz=est.copy()
            est[idx[train]]=y[train,j];var[idx[train]]=D[train,j,j]
            z[idx[train]]=ob["z"][:,j];vz[idx[train]]=ob["D"][:,j,j]
            pd.DataFrame({"area":a.area,"est":est,"var":var,"z":z,"var_z":vz}).to_csv(out/f"estimates_{name}.csv",index=False)
        (out/"input_manifest.json").write_text(json.dumps({
            "year":year,"split":split,"test_areas_masked":int(test.sum()) if split=="locked_train" else 0,
            "six_features":app.FEATURES,"stage1":"Verified identity: constant-area-X LGREG equals direct estimator and variance",
            "stage1_working_model_scope":"Area-constant working predictions cancel; no test model parameters are supplied",
            "strict_model":"Published author filter and BYM2 intercept only",
            "extension_model":"Primary continuity/GVF observations and six area covariates in linking model; explicit extension",
            "extension_training_variance_preparation":ob["metadata"],
            "covariate_populations":"Historical district characteristics known; no contemporary child age-sex margins asserted",
            "unsupported_direct_only_codes":unsupported.ubigeo.tolist(),
            "source_manifest":manifest,
            "source_models_sha256":hashlib.sha256((STUDY/"scripts/models.py").read_bytes()).hexdigest()},indent=2))


if __name__=="__main__":
    for year in (2019,2025):export(year)
