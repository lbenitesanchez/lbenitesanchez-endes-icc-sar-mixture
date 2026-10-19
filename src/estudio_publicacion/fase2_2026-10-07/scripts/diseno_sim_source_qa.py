"""Closed-form baseline fixtures and root simulation structural source review."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from diseno_calibration import digest
import sim_design_diagnostics as simulation

PHASE=Path(__file__).resolve().parents[1]


def main():
    checks=[]
    def check(name,ok,detail=None):
        checks.append(dict(name=name,passed=bool(ok),detail=detail))
        if not ok:
            raise AssertionError(name)
    # Exact extreme: all within-PSU outcomes identical, both outcomes with
    # distinct district means. ICC=1; cluster-adjusted Kish n becomes effective
    # PSU count, including deliberately unequal PSU weights.
    rows=[]
    for a in range(5):
        for p in range(2):
            for child in range(2):
                rows.append(dict(area=a,psu=f"{a}:{p}",y_anemia=float(p),y_stunting=float(1-p),
                                 weight_anemia=float(1 if p==0 else 4),weight_stunting=float(1 if p==0 else 4)))
    units=pd.DataFrame(rows)
    idx=np.arange(5)
    kish=np.repeat(100/34,5)
    je=np.repeat(100/68,5)
    for outcome in ("anemia","stunting"):
        rho,details=simulation.working_icc(units,outcome)
        n,estimated_je=simulation.cluster_effective_n(units,outcome,idx,kish,rho)
        check("perfect clustered outcome ICC1 "+outcome,abs(rho-1)<1e-15)
        check("perfect ICC reduces Kish to effective PSU count "+outcome,np.max(abs(n-je))<1e-15)
        check("unequal PSU weight concentration retained "+outcome,np.array_equal(estimated_je,je))
    # Exact extreme: each PSU has one zero and one one; between-PSU mean
    # variance zero while within-PSU variance positive. Working ICC truncates0.
    balanced=units.copy()
    balanced["y_anemia"]=np.tile([0.,1.],len(balanced)//2)
    rho,details=simulation.working_icc(balanced,"anemia")
    n,_=simulation.cluster_effective_n(balanced,"anemia",idx,kish,rho)
    check("zero between-PSU variance working ICC0",rho==0)
    check("zero ICC keeps original Kish count",np.array_equal(n,kish))
    # A PSU-poor area cannot identify its own ICC and must not modify the
    # pooled working ICC of the eligible multi-PSU training areas.
    singleton=units.iloc[:2].copy();singleton["area"]=99
    singleton["psu"]="99:one"
    mutated=singleton.copy();mutated["y_anemia"]=1-mutated.y_anemia
    rho1,_=simulation.working_icc(pd.concat([balanced,singleton]),"anemia")
    rho2,_=simulation.working_icc(pd.concat([balanced,mutated]),"anemia")
    check("unidentifiable singleton area excluded from pooled ICC",rho1==rho2==0)
    X,B=simulation.template()
    check("fixed simulation design dimensions",X.shape==(80,7) and B.shape==(80,12))
    check("six fixed synthetic covariates centered and scaled",np.max(abs(X[:,1:].mean(0)))<1e-14 and np.max(abs(X[:,1:].std(0)-1))<1e-14)
    check("spatial basis field scale unchanged",abs(np.mean(np.sum(B**2,axis=1))-1)<1e-14)
    covariance=np.array([[1,.4],[.4,1]])
    L=np.array([[1.,0],[.4,np.sqrt(.84)]])
    check("specified correlated PSU and child Gaussian construction is PSD",np.max(abs(L@L.T-covariance))<1e-15 and np.linalg.eigvalsh(covariance).min()>0)
    check("two planned diagnostic cohorts total40 independent worlds",simulation.POPULATIONS_PER_CONTEXT==20 and simulation.CONTEXTS=={"dense6":6,"sparse2":2})
    proof=dict(status="PASS_PRE_FREEZE_SIMULATION_BASELINE_AND_TEMPLATE_QA",checks_count=len(checks),checks=checks,
               prevalence_worlds_generated=0,models_fitted=0,reserved_2024_read=False,
               root_source_sha256=digest(PHASE/"scripts/sim_design_diagnostics.py"),
               review_scope="baseline ICC/count correction, geometry and covariance constructions; source stream separation and wording recommendations sent to root separately",
               source_sha256=digest(Path(__file__)))
    output=PHASE/"validacion/diseno_sim_source_qa.json"
    output.write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof,indent=2))


if __name__=="__main__":
    main()
