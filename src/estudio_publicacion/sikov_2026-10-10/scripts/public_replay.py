"""Portable eleven-method re-fit of one released synthetic world.

Archived predictions and statistical source files remain immutable. Compilation
and replay outputs occur outside the release. No scoring truth or private survey
response is opened, and the selected 75/25 recipe is not reselected.
"""
from pathlib import Path
import argparse,json,hashlib,os,shutil
import numpy as np
import pandas as pd
import comparison_evaluate as base
from comparison_simulation import ROOT,STUDY,CONTEXTS,EVALUATION_IDS
from sikov_published_fit import fit_arrays
from public_verify import prediction_preflight,frozen_metadata,completed_result_gates,sampled_graph_supplement_gate,SAMPLED_GRAPH_CONTEXT
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--context',choices=(*CONTEXTS,SAMPLED_GRAPH_CONTEXT),required=True)
    p.add_argument('--rep',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.rep not in EVALUATION_IDS:raise ValueError('Only the closed fresh synthetic cohort')
    meta=prediction_preflight();completed_result_gates(meta)
    gate=sampled_graph_supplement_gate(meta,require_results=False)
    proof=meta['proof'];supplemental=a.context==SAMPLED_GRAPH_CONTEXT
    if supplemental:
        source=ROOT/'sensibilidad_sampled_graph/inputs'/f'rep_{a.rep:05d}'
    else:source=ROOT/'evaluacion/inputs'/a.context/f'rep_{a.rep:05d}'
    manifest=json.loads((source/'manifest.json').read_text())
    for name,key in [('inputs.npz','fit_inputs_sha256'),('sample.csv','sample_sha256')]:
        if sha(source/'model_inputs'/name)!=manifest[key]:raise RuntimeError('Released input mismatch '+name)
    output=a.output.resolve()
    if output.exists():raise FileExistsError('Reproduction must use a new output folder')
    if output.is_relative_to(STUDY.parent.resolve()):
        raise ValueError('Replay and compiler outputs must be outside the immutable release')
    output.mkdir(parents=True);runtime=output/'runtime';runtime.mkdir()
    for name in ('beta_power_hybrid.stan','beta_power_stable_functions.stan'):
        original=STUDY/'scripts/competitors'/name
        shutil.copyfile(original,runtime/name)
        if sha(original)!=sha(runtime/name):raise RuntimeError('Portable Stan copy changed')
    backend={}
    def portable_backend():
        import cmdstanpy
        local=STUDY/'.competitor_env/cmdstan/cmdstan-2.40.0'
        if local.is_dir():cmdstanpy.set_cmdstan_path(str(local))
        sdk=Path('/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk')
        if sdk.exists():os.environ.setdefault('SDKROOT',str(sdk))
        model=cmdstanpy.CmdStanModel(stan_file=str(runtime/'beta_power_hybrid.stan'))
        base.hmc._MODEL=base.DiagonalMetric(model)
        backend['executable']=Path(model.exe_file)
        return model
    # Operational path adaptation only. The copied frozen Stan program, sampler
    # configuration, posterior, seeds and convergence criteria remain identical.
    base.get_backend=portable_backend
    d=dict(np.load(source/'model_inputs/inputs.npz',allow_pickle=False))
    units=pd.read_csv(source/'model_inputs/sample.csv')
    stream=[20261010,973,a.rep]if supplemental else[20261010,811,CONTEXTS.index(a.context),a.rep]
    seed=int(np.random.SeedSequence(stream).generate_state(1)[0])%(2**32-200000)
    status=base.fits(d,units,output/'original',seed,proof['selected_recipe'])
    pubstream=[20261010,974,a.rep]if supplemental else[20261010,941,CONTEXTS.index(a.context),a.rep]
    pubseed=int(np.random.SeedSequence(pubstream).generate_state(1)[0])
    _,_,pubstatus=fit_arrays(d,output/'published_scope',pubseed)
    frozen_metadata()
    summary={'role':'portable eleven-method synthetic reproduction',
      'protocol_sha256':meta['protocol_sha256'],'addendum_sha256':meta['addendum_sha256'],
      'joint_prediction_lock_sha256':meta['joint_sha256'],
      'context':a.context,'replication':a.rep,'truth_read':False,'private_survey_records_read':False,
      'separate_sampled_graph_descriptive_sensitivity':supplemental,
      'supplemental_plan_sha256':sha(ROOT/'validacion/sensitivity_sampled_graph_freeze.json')if supplemental else None,
      'selected_recipe':proof['selected_recipe'],'selection_performed_again':False,
      'runtime_executable_sha256':sha(backend['executable']),
      'compiled_executable_path':str(backend['executable']),
      'convergence':{**{k:v['converged']for k,v in status.items()},
                     **dict.fromkeys(meta['addendum']['new_methods'],pubstatus['converged'])},
      'outputs_sha256':{str(p.relative_to(output)):sha(p)for p in output.rglob('*')if p.is_file()},
      'original_release_files_modified':False,
      'portability_limit':'new compiler/platform can cause floating-point or MCMC differences; diagnostic gates retained'}
    (output/'reproduction_completion.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if not all(summary['convergence'].values()):raise RuntimeError('Replayed numerical gates failed; results retained for diagnosis')
if __name__=='__main__':main()
