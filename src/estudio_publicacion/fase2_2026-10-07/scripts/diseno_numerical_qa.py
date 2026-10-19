"""Independent identities and negative fixtures for new design calibration."""
import itertools
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from diseno_calibration import (calibrate, digest, gvf_stabilize,
    replicated_centered_variance, stratified_psu_counts, taylor_contributions)

PHASE=Path(__file__).resolve().parents[1]


def main():
    checks=[]
    def record(name,passed,detail=None):
        checks.append(dict(name=name,passed=bool(passed),detail=detail))
        if not passed:
            raise AssertionError(name)
    # Enumerate every ordered ordinary-bootstrap sample, independently of
    # production RNG or multinomial draws. Known unbiased sample variance
    # m/(m-1)*SS is the reference, not production Taylor code.
    for m in (2,3,4):
        score=np.column_stack((np.linspace(-.3,.7,m),np.linspace(.9,-.1,m)**2))
        frame=pd.DataFrame(dict(psu_key=[f"p{i}" for i in range(m)],stratum=np.repeat(1,m)))
        samples=list(itertools.product(range(m),repeat=m))
        counts=np.array([np.bincount(sample,minlength=m) for sample in samples])
        actual,roundoff=replicated_centered_variance(frame,score,counts)
        reference=m/(m-1)*np.sum((score-score.mean(axis=0))**2,axis=0)
        record(f"exact bootstrap expectation m{m}",np.max(abs(actual.mean(axis=0)-reference))<1e-13,
               dict(reference=reference.tolist(),mean=actual.mean(axis=0).tolist(),ordered_samples=len(samples)))
        record(f"positive bootstrap variance m{m}",np.all(actual>=0))
        if m==2:
            record("two PSU bootstrap zero-or-double identity",
                   np.all(np.minimum(np.abs(actual),np.abs(actual-2*reference))<1e-13))
            record("two PSU variance uncertainty not degenerate",np.std(actual[:,0])>0)
    singleton_frame=pd.DataFrame(dict(psu_key=["solo"],stratum=[99]))
    single,roundoff=replicated_centered_variance(singleton_frame,np.array([[.125,.3]]),np.ones((5,1),dtype=int))
    record("full singleton Taylor fixed component",np.array_equal(single,np.repeat([[.125**2,.3**2]],5,axis=0)))
    bad=False
    try:
        replicated_centered_variance(singleton_frame,np.array([[.125]]),np.zeros((3,1),dtype=int))
    except ValueError:
        bad=True
    record("default refuses unidentified singleton resampling",bad)
    # Closed-form full-stratum Taylor variance with empty domain PSU holes.
    frame=pd.DataFrame(dict(psu_key=["a","b","c","d"],stratum=[1,1,2,2]))
    units=pd.DataFrame(dict(area=[0,0,0,0],psu_key=["a","a","c","c"],
                            y_anemia=[1.,0.,1.,0.],weight_anemia=[1.,1.,2.,1.]))
    d=taylor_contributions(frame,units,"anemia",[0])
    # Direct p=3/5; PSU scores: -1/25,0,+1/25,0.
    expected=2*((-1/25)**2/2+(1/25)**2/2)
    record("full-frame zero-score PSU holes retain Taylor component",abs(d["variance"][0]-expected)<1e-15,
           dict(actual=d["variance"][0],expected=expected))
    record("score sum zero",abs(d["full_score_sums"][0])<1e-15)
    rejected=False
    try:
        taylor_contributions(frame,units.assign(psu_key="unknown"),"anemia",[0])
    except ValueError:
        rejected=True
    record("unknown child PSU rejected",rejected)
    counts,summary=stratified_psu_counts(frame,dict(replicates=4096,mixture_states=8))
    record("bootstrap counts preserve full-stratum PSU draw size",np.all(counts[:,:2].sum(axis=1)==2) and np.all(counts[:,2:].sum(axis=1)==2))
    record("two PSU bootstrap samples both empty and double multiplicity",np.any(counts[:,0]==0) and np.any(counts[:,0]==2))
    # Stabilization fixture includes true sample zero/one and domain df0.
    y=np.array([.1,.2,.3,.4,.5,.6,.7,.8,0.,1.])
    kish=np.arange(10.,20.)
    psueff=np.arange(2.,12.)
    psu=np.arange(2,12)
    dfs=psu-1;dfs[-2:]=0
    v=np.r_[y[:8]*(1-y[:8])/kish[:8]*1.4,0.,0.]
    stable=gvf_stabilize(y,kish,psueff,psu,dfs,v)
    record("all observed boundary effective counts strictly positive",np.all(np.isfinite(stable["effective_n"])) and np.all(stable["effective_n"]>0))
    record("sample zero and one never changed",y[-2]==0 and y[-1]==1)
    record("sample zero and one get GVF-only variance",np.array_equal(stable["alpha"][-2:],np.zeros(2)))
    record("interior design variance matching identity",np.max(abs(stable["effective_n"][:8]*stable["stabilized_variance"][:8]-y[:8]*(1-y[:8])))<1e-15)
    # Generate ten stable independent training domains plus an excluded domain.
    rows=[];frame_rows=[]
    for a in range(11):
        for p in range(4):
            key=f"{a}:{p}";frame_rows.append(dict(psu_key=key,stratum=a))
            for child in range(5):
                rows.append(dict(area=a,psu_key=key,y_anemia=float((child+p+a)%4==0),weight_anemia=1.+.1*child))
    train=pd.DataFrame(rows);full=pd.DataFrame(frame_rows)
    selected=np.arange(10)
    cfg=dict(replicates=32,mixture_states=8)
    fit1=calibrate(full,train,selected,"anemia",cfg)
    changed=train.copy()
    changed.loc[changed.area==10,"y_anemia"]=1-changed.loc[changed.area==10,"y_anemia"]
    fit2=calibrate(full,changed,selected,"anemia",cfg)
    record("held-out outcome mutation cannot change calibration",np.array_equal(fit1["mixture_effective_n"],fit2["mixture_effective_n"]))
    record("CUT state indices fixed and reproducible",fit1["status"]["mixture_state_indices"]==fit2["status"]["mixture_state_indices"])
    record("replicates nondegenerate with variable PSU scores",np.max(np.std(fit1["replicated_variance"],axis=0))>0)
    from diseno_fit_cut import pool_equal_weight
    one=dict(p_mean=np.array([[.2]]),p_variance=np.array([[.01]]),p_draws=np.full((4096,1,1),.2))
    two=dict(p_mean=np.array([[.8]]),p_variance=np.array([[.01]]),p_draws=np.full((8192,1,1),.8))
    pooled=pool_equal_weight([one,two])
    record("CUT pooled mean equal external weights",abs(pooled["p_mean"][0,0]-.5)<1e-15)
    record("CUT pooled variance law includes between-state variance",abs(pooled["p_variance"][0,0]-.1)<1e-15)
    record("CUT balances states when retry draws differ",pooled["draws_per_state"]==4096 and np.sum(pooled["p_draws"]==.2)==4096 and np.sum(pooled["p_draws"]==.8)==4096)
    # Functional regression: two successive conditional states must not nest
    # the metric adapter. Mock sample receives a1D diagonal every time, and no
    # CmdStan executable, model or outcome fit is run in this fixture.
    import diseno_fit_cut as runner
    metric_calls=[]
    class MockRawModel:
        def sample(self,**kwargs):
            metric=np.asarray(kwargs["inv_metric"])
            if metric.ndim!=1:
                raise AssertionError("Mock raw sampler must receive diagonal vector")
            metric_calls.append(metric.tolist())
            return None
    original_model=runner.hmc._MODEL
    original_get_model=runner.hmc.get_model
    original_fit=runner.hmc.fit
    original_gate=runner.freeze_gate
    mock_raw=MockRawModel()
    def mock_get_model():
        return runner.hmc._MODEL if runner.hmc._MODEL is not None else mock_raw
    def mock_fit(X,y,n,idx,**kwargs):
        runner.hmc.get_model().sample(inv_metric=np.diag([1.,2.,3.]))
        return dict(p_mean=np.array([[.4,.6]]),lower95=np.array([[.2,.4]]),upper95=np.array([[.6,.8]]),
                    p_variance=np.array([[.01,.01]]),p_draws=np.repeat([[[.4,.6]]],16,axis=0),
                    status=[dict(success=True,beta_shape_guard_cells=0),dict(success=True,beta_shape_guard_cells=0)])
    try:
        runner.hmc._MODEL=None
        runner.hmc.get_model=mock_get_model
        runner.hmc.fit=mock_fit
        runner.freeze_gate=lambda: (None,{"mock_only_not_authority":True})
        mock_inputs=dict(X=np.array([[1.]]),y=np.array([[.5,.5]]),sample_idx=np.array([0]),basis=np.zeros((1,0)))
        with tempfile.TemporaryDirectory(prefix="phase2_mock_metric_") as temporary:
            for state in (1,2):
                runner.conditional_fit(mock_inputs,np.array([[10.,10.]]),Path(temporary)/str(state),state,f"mock{state}")
        record("two conditional calls wrap metric only once",metric_calls==[[1.,2.,3.],[1.,2.,3.]])
        record("metric adapter not recursively wrapped",isinstance(runner.hmc._MODEL,runner.DiagonalMetric) and not isinstance(runner.hmc._MODEL.model,runner.DiagonalMetric))
        oversized_seed_rejected=False
        try:
            runner.conditional_fit(mock_inputs,np.array([[10.,10.]]),Path(temporary)/"bad_seed",20261007000,"badseed")
        except ValueError:
            oversized_seed_rejected=True
        record("CmdStan unsigned32 seed guard",oversized_seed_rejected)
    finally:
        runner.hmc._MODEL=original_model
        runner.hmc.get_model=original_get_model
        runner.hmc.fit=original_fit
        runner.freeze_gate=original_gate
    proof=dict(status="PASS_DESIGN_CENTERED_PSU_BOOTSTRAP_AND_CUT_NUMERICAL_QA",
               checks_count=len(checks),checks=checks,model_fits_dispatched=False,
               reserved_2024_responses_read=False,
               source_sha256={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name("diseno_calibration.py"),Path(__file__).with_name("diseno_fit_cut.py")]})
    path=PHASE/"validacion/diseno_numerical_qa.json"
    path.write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof,indent=2))


if __name__=="__main__":
    main()
