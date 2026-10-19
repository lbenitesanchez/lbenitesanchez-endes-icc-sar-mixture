"""Verify updated-auxiliary routing without asserting an extraction is official."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from diseno_calibration import digest
from diseno_prepare_development import PHASE,STUDY,load_dataset


def main():
    old,frame,units=load_dataset(2019)
    original=old[0]
    aux=pd.read_csv(STUDY/"datos/processed/aux_all_districts.csv",dtype={"ubigeo":str,"dept_code":str,"prov_code":str})
    checks=[]
    def check(name,ok,detail=None):
        checks.append(dict(name=name,passed=bool(ok),detail=detail))
        if not ok:
            raise AssertionError(name)
    with tempfile.TemporaryDirectory(prefix="diseno_auxiliary_qa_",dir=PHASE/"validacion") as temp:
        path=Path(temp)/"aux_fixture.csv"
        aux.to_csv(path,index=False)
        new,newframe,newunits=load_dataset(2019,path)
        newareas=new[0]
        check("auxiliary-only permutation preserves set of target codes",set(original.ubigeo)==set(newareas.ubigeo))
        protected=["y_anemia_who2024","y_stunting","var_anemia_who2024","var_stunting","cov_who_stunting","n_eff_anemia_who2024","n_eff_stunting"]
        left=original.set_index("ubigeo").sort_index()[protected].to_numpy(float)
        right=newareas.set_index("ubigeo").sort_index()[protected].to_numpy(float)
        check("auxiliary routing preserves exact prevalence Kish and Taylor inputs",np.array_equal(left,right,equal_nan=True))
        check("full household PSU frame unaffected by auxiliary routing",frame.equals(newframe))
        leftx=pd.DataFrame(old[1],index=original.ubigeo).sort_index().to_numpy()
        rightx=pd.DataFrame(new[1],index=newareas.ubigeo).sort_index().to_numpy()
        check("X scaling invariant to metadata-only row ordering",np.max(abs(leftx-rightx))<1e-12,
              dict(max_difference=float(np.max(abs(leftx-rightx)))))
        # Explicit negative fixtures stop forbidden survey data becoming an
        # outcome-dependent census-domain eligibility filter.
        for field in ("y_anemia","var_stunting","direct_anemia","weight","child_id"):
            bad=aux.copy();bad[field]=0
            bad.to_csv(path,index=False)
            rejected=False
            try:
                load_dataset(2019,path)
            except ValueError:
                rejected=True
            check("auxiliary rejects survey-only column "+field,rejected)
        for mutation,label in ((lambda d: pd.concat([d,d.iloc[:1]],ignore_index=True),"duplicate UBIGEO"),
                               (lambda d: d.assign(ubigeo="123"),"non-six-digit UBIGEO"),
                               (lambda d: d.assign(dept_code="99"),"discordant parent geography")):
            mutation(aux.copy()).to_csv(path,index=False)
            rejected=False
            try:
                load_dataset(2019,path)
            except ValueError:
                rejected=True
            check("auxiliary rejects "+label,rejected)
    proof=dict(status="PASS_AUXILIARY_INPUT_ROUTING_AND_LEAKAGE_GUARDS",checks_count=len(checks),checks=checks,
               official_new_auxiliary_extraction_verified_by_this_test=False,
               reserved_2024_read=False,model_fits_dispatched=False,
               source_sha256={name:digest(PHASE/"scripts"/name) for name in ("diseno_auxiliary_qa.py","diseno_prepare_development.py")})
    out=PHASE/"validacion/diseno_auxiliary_qa.json"
    out.write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof,indent=2))


if __name__=="__main__":
    main()
