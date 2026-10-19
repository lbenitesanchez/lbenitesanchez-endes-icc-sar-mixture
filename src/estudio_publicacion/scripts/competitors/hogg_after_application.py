"""Dependency-only one-core handoff: age/sex -> spatial pilot -> scout -> Hogg.

The waiting coordinator uses no fitting core. Every fit is run serially after
the age/sex application exits. No scores, coverage results or truth are read.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'resultados/competidores_hogg'
HERE=ROOT/'scripts/competitors'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--age-process',type=int,required=True)
    p.add_argument('--lane',default='lane01')
    args=p.parse_args()
    state=RESULT/'one_core_handoff.json'
    events=[]
    def record(phase,**kwargs):
        event={'phase':phase,'utc':datetime.now(timezone.utc).isoformat(),**kwargs}
        events.append(event);state.write_text(json.dumps(events,indent=2))
        print(json.dumps(event),flush=True)
    def run(script,arguments,log):
        with(RESULT/log).open('w')as stream:
            r=subprocess.run([sys.executable,str(HERE/script)]+arguments,cwd=ROOT,
                stdout=stream,stderr=subprocess.STDOUT,check=False)
        return r.returncode
    record('waiting_for_age_sex_application',age_process=args.age_process,
           resource_rule='One existing ENDES lane only; does not add a fitting core.')
    while True:
        try:os.kill(args.age_process,0)
        except ProcessLookupError:break
        time.sleep(15)
    summary=RESULT/'age_sex_application_summary.json'
    if not summary.exists():
        record('age_application_missing_completion_summary',action='Manual audit required; no silent refit or dependent fit started.')
        return
    age=json.loads(summary.read_text())
    if len(age)!=2 or not all(row['success']for row in age):
        record('age_application_did_not_pass_final_gates',action='Manual audit required; preserve every attempt.')
        return
    record('spatial_pilot_started',age_summary_sha256=sha(summary))
    code=run('hogg_spatial_validate.py',[],'spatial_pilot.log')
    proof=RESULT/'spatial_pilot/posterior_equivalence.json'
    passed=code==0 and proof.exists()and json.loads(proof.read_text()).get('passed',False)
    record('spatial_pilot_finished',returncode=code,passed=passed)
    if passed:
        names=['hogg_tsln.py','hogg_spatial.py','hogg_stage2_bym2.stan',
            'hogg_stage2_bym2_full_reference.stan','hogg_spatial_validate.py',
            'hogg_spatial_application.py','hogg_spatial_batch.py','hogg_spatial_scout.py',
            'hogg_spatial_design.md']
        frozen={'approved_after_numerical_pilot':True,'utc':datetime.now(timezone.utc).isoformat(),
            'source_sha256':{name:sha(HERE/name)for name in names},
            'pilot_proof_sha256':sha(proof),
            'independent_algebra_review_sha256':sha(ROOT/'validacion/hogg_spatial_independent_review.json'),
            'priors':'sigma HalfNormal(0,2), rho Beta(3.05,1.65), original author-code QR-beta/intercept/GVF priors.',
            'initial_sampler':{'chains':4,'parallel_chains':1,'warmup':500,'retained':1000,
                'adapt_delta':.95,'maximum_tree_depth':12},
            'numerical_gates':'All sampled-parameter finite Rhat<=1.01, bulk/tail ESS>=400, no divergences/depth hits, every chain BFMI>=.3.',
            'fixed_retry':'One failing endpoint retry at 1000 warmup, 2000 retained, adapt_delta=.99; same four chains, seed, priors and graph.',
            'graph':'Full M-C constrained ICAR modes with componentwise geometric-variance scaling; no low-rank shortcut.',
            'scope':'Separate 100 existing structural BYM2 worlds plus 2025 age/sex full/locked S2 sensitivities; no modification to primary Holm3, generic Hogg1100 or Australian full-case claim.',
            'S1':'Reuse the identical verified modular summary; no refit or evaluation truth.',
            'scout_policy':'Single short application chain only measures runtime. No scout draws enter final summaries; priors/graph are not tuned.'}
        path=RESULT/'hogg_spatial_freeze.json'
        if path.exists():raise FileExistsError('Preserve existing spatial freeze rather than overwrite it')
        path.write_text(json.dumps(frozen,indent=2))
        record('spatial_source_and_sampler_frozen',freeze_sha256=sha(path))
        preparation=run('hogg_spatial_application.py',['--modes','full','--prepare-only'],
                        'spatial_full_source_preparation.log')
        if preparation==0:
            record('full_1874_runtime_scout_started')
            scout=run('hogg_spatial_scout.py',[
                '--source-npz',str(RESULT/'endes_2025_full_age_sex_s2_bym2/source_modular_summary_anemia.npz'),
                '--edges',str(ROOT/'resultados/competidores_gao/inputs/endes_2025_full/adjacency_edges.csv'),
                '--output',str(RESULT/'spatial_runtime_scout_1874')],'spatial_runtime_scout.log')
            record('full_1874_runtime_scout_finished',returncode=scout,
                policy='Runtime observation only; no posterior or model tuning.')
        else:record('spatial_source_preparation_failed',returncode=preparation)
    # A separate spatial pilot failure cannot invalidate or retune the already
    # frozen nonspatial experiment; its first shard can still make progress.
    record('first_primary_Hogg_shard_started',lane=args.lane)
    code=run('hogg_queue.py',['--lane',args.lane,'--max-shards','1'],
             'final_hogg_'+args.lane+'.log')
    record('first_primary_Hogg_shard_finished',returncode=code)


if __name__=='__main__':main()
