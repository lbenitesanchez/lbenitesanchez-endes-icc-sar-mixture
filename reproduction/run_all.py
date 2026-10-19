#!/usr/bin/env python3
"""Execute the full clean model cohort and retain honest completion receipts.

The config and run directory are private, external to the public repository.
Prepare inputs and development first with the worker CLIs. This controller
will not reuse historical predictions or label running fits as reproduced.
"""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
CONTEXTS = ('SAR_probability_srs', 'SAR_logit_srs', 'SAR_logit_psu_sparse', 'IID_logit_psu_sparse')

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    base, root = Path(cfg['run_base']), Path(cfg['sikov_run_root'])
    if base.resolve().is_relative_to(HERE.parent):
        raise ValueError('Runtime and private records must be outside the repository')
    python = cfg['python_new']
    common = ['--source-root', cfg['source_root'], '--reference-root', cfg['reference_root'],
              '--run-root', str(root), '--exe', cfg['beta_exe'], '--cmdstan', cfg['cmdstan']]
    environment = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        environment[name] = '1'
    environment['SDKROOT'] = cfg.get('sdkroot', '/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk')
    environment['MACOSX_DEPLOYMENT_TARGET'] = cfg.get('deployment_target', '26.0')
    lock = threading.Lock()
    state = dict(status='RUNNING', article_reproduction_confirmed=False,
                 started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 selected_recipe={'family':'ICC_SAR_mixture', 'icc_weight':.75},
                 expected={'primary':256, 'sampled_graph':64, 'B300':8, 'real_scopes':2,
                           'legacy_reference':800, 'legacy_cut_diagnostics':40},
                 completed={}, commands=[], kernels_or_seeds_changed=False)
    state_file = base/'progress.json'

    def command(label, cmd):
        receipt = base/'controller_receipts'/(label+'.json')
        log = base/'logs/controller'/(label+'.log')
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old['command'] != cmd or old['exit_code'] != 0:
                raise RuntimeError('Investigate an existing failed or changed command: '+str(receipt))
            return old
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with log.open('x') as output:
            process = subprocess.Popen(cmd, env=environment, stdout=output, stderr=subprocess.STDOUT)
            with lock:
                state.setdefault('running',{})[label]=dict(pid=process.pid,log=str(log),started_epoch=started)
                save(state_file,state)
            exit_code=process.wait()
        record = dict(label=label, command=cmd, exit_code=exit_code,
                      seconds=time.time()-started, log=str(log),
                      log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                      finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        save(receipt, record)
        with lock:
            state['running'].pop(label,None)
            state['commands'].append(record)
            state['updated_utc'] = record['finished_utc']
            save(state_file, state)
        return record

    def batch(stage, jobs, workers):
        failures=[]
        def execute(job):
            context, rep = job
            label=f'{stage}_{context}_{rep}'
            cmd=[python, str(HERE/'sikov_worker.py'), '--stage', stage, *common, '--rep',str(rep)]
            if context:
                cmd += ['--context',context]
            result=command(label,cmd)
            with lock:
                state['completed'][stage]=state['completed'].get(stage,0)+1
                if result['exit_code']:
                    failures.append(label)
                save(state_file,state)
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(execute,jobs):
                pass
        if failures:
            raise RuntimeError('All scheduled worlds retained; failures require investigation: '+', '.join(failures))

    # Every following fit uses the newly regenerated inputs. Development can
    # never silently choose another model using the held-out evaluation worlds.
    setup=json.loads((root/'validacion/sikov_setup.json').read_text())
    development=json.loads((root/'desarrollo/recreated_native/completion.json').read_text())
    preparation=json.loads((root/'real_preparation_receipt.json').read_text())
    if setup['status']!='PASS_REGENERATED_352_WORLD_INPUTS' or not development['matches_frozen_selected_recipe']:
        raise RuntimeError('Verified fresh inputs and the frozen development selection are required')
    if not preparation['fresh_microdata_preparation_from_SAV']:
        raise RuntimeError('Actual data must be reconstructed from SAV before fitting')
    save(state_file,state)

    def real():
        failures=[]
        def fit(scope):
            record=command('real_'+scope,[python,str(HERE/'real_worker.py'),'--action','fit',*common,'--scope',scope])
            if record['exit_code']:
                failures.append(scope)
            with lock:
                state['completed']['real_scopes']=state['completed'].get('real_scopes',0)+1
                save(state_file,state)
        with futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(fit,('temporal','territorial')))
        if failures:
            raise RuntimeError('Fresh actual-data failures retained: '+', '.join(failures))

    def simulations():
        # Interleave contexts; the unchanged sampler uses one chain process at
        # a time per world. The worker count is an operational resource limit.
        jobs=[(context,rep) for rep in range(30000,30064) for context in CONTEXTS]
        batch('eval-one',jobs,int(cfg.get('sim_workers',6)))
        batch('b300-one',[(c,r) for r in (30000,30001) for c in CONTEXTS],int(cfg.get('sim_workers',6)))
        batch('sampled-one',[(None,r) for r in range(30000,30064)],4)

    def legacy_pipeline():
        gate=cfg.get('legacy_setup_gate')
        if gate:
            gate=Path(gate)
            while not gate.is_file():
                time.sleep(5)
            proof=json.loads(gate.read_text())
            if not proof.get('complete'):
                raise RuntimeError('The initial legacy preparation did not close')
        legacy=[python,str(HERE/'legacy_runner.py'), '--source-root',cfg['source_root'],
                '--reference-root',cfg['reference_root'],'--run-root',cfg['legacy_run_root'],
                '--cmdstan',cfg['cmdstan'],'--workers',str(cfg.get('legacy_workers',2)),
                '--r-lib',cfg['r_lib'],'--action','run','--stage','all']
        return command('legacy_all',legacy)

    try:
        with futures.ThreadPoolExecutor(max_workers=3) as pool:
            sim = pool.submit(simulations)
            actual = pool.submit(real)
            old_future=pool.submit(legacy_pipeline)
            sim.result()
            actual.result()
            old=old_future.result()
            if old['exit_code']:
                raise RuntimeError('Legacy rerun incomplete; inspect its preserved receipts')
        real_scored=command('real_score',[python,str(HERE/'real_worker.py'),'--action','score',*common])
        if real_scored['exit_code']:
            raise RuntimeError('Fresh actual-data scoring or comparison failed; outputs retained')
        scored=command('sikov_score',[python,str(HERE/'sikov_worker.py'),'--stage','score',*common])
        if scored['exit_code']:
            raise RuntimeError('Fresh scoring or article comparison failed; outputs retained')
        state['status']='FITS_AND_MAIN_SCORING_FINISHED_COMPARISON_REVIEW_REQUIRED'
        state['article_reproduction_confirmed']=False
        state['legacy_numerical_and_article_comparison_pending']=True
    except BaseException as error:
        state['status']='INCOMPLETE_REQUIRES_INVESTIGATION'
        state['error']=repr(error)
        save(state_file,state)
        raise
    finally:
        state['updated_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        save(state_file,state)

if __name__=='__main__':
    main()
