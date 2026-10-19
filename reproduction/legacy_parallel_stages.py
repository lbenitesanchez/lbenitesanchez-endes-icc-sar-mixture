#!/usr/bin/env python3
"""Execute canonical future stages behind the stopped original-controller barrier.

Commands, worker counts, seeds and frozen helpers are read unchanged from the
original plan. Real execution receipts let the original controller skip these
already completed commands after resumption. No successful receipt is written
before an actual zero process exit. Failures remain for review.
"""
from pathlib import Path
import argparse,datetime as dt,json,os,subprocess,sys
import legacy_v2_runner as frozen
import legacy_runner as original

def barrier(pid,run):
    line=subprocess.check_output(['ps','-p',str(pid),'-o','pid=,stat=,command='],text=True).strip()
    parts=line.split(None,2)
    if parts[0]!=str(pid)or'T'not in parts[1]or'legacy_v2_runner.py'not in parts[2]or str(run)not in parts[2]:
        raise RuntimeError('Original top-controller pause/identity barrier is not in place')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--controller-pid',type=int,required=True)
    p.add_argument('--stages',nargs='+',choices=('cut40','application2025','external2024','cut16'),required=True)
    a=p.parse_args();run=a.run_root.resolve();barrier(a.controller_pid,run)
    gate=json.loads((run/'legacy_v2_input_identity.json').read_text())
    if gate['status']!='PASS_800_OBSERVED_INPUTS_EXACT_FIT_GEOMETRY_ROUNDOFF'or gate['failures']:raise RuntimeError('Exact 800-input gate required')
    original.check_sources(run/'study')
    contract=json.loads((run/'legacy_v2_generation_contract.json').read_text())
    if contract['operational_helper_sha256']!=frozen.hashes():raise RuntimeError('Frozen operational sources changed')
    commands=json.loads((run/'legacy_commands.json').read_text())
    env=os.environ.copy()
    for name in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):env[name]='1'
    env['ENDES_STUDY_ROOT']=str(run/'study');env['ENDES_EXPORT_PRIVATE_INPUTS']='1'
    env['ENDES_R_LIB']=str((run/'study/.Rlib').resolve())
    for stage in a.stages:
        owner=run/'parallel_stage_owners'/f'{stage}.json';owner.parent.mkdir(exist_ok=True)
        with owner.open('x')as handle:json.dump({'pid':os.getpid(),'controller_barrier_pid':a.controller_pid,'helper_sha256':original.sha(Path(__file__))},handle)
        for index,command in enumerate(commands[stage]):
            barrier(a.controller_pid,run)
            receipt=run/'receipts'/stage/f'{index:03d}.json'
            if receipt.exists():
                prior=json.loads(receipt.read_text())
                if prior['command']!=command or prior['exit_code']!=0 or prior['operational_helper_sha256']!=frozen.hashes():raise RuntimeError('Existing receipt differs or failed')
                continue
            # Canonical compiler calls are identical across stages. Reuse a
            # completed fresh compiler receipt; avoid shared-stamp write races.
            if command[-1]=='compile':
                candidates=list((run/'receipts').glob('*/*.json'))
                prior=next((path for path in candidates if json.loads(path.read_text()).get('command')==command and
                    json.loads(path.read_text()).get('exit_code')==0),None)
                if prior:
                    record=json.loads(prior.read_text());record.update(reused_fresh_identical_compile_receipt_sha256=original.sha(prior),
                        canonical_command_was_executed_before=True,parallel_stage_adapter_sha256=original.sha(Path(__file__)))
                    original.save(receipt,record);continue
            log=run/'logs'/stage/f'{index:03d}.log';log.parent.mkdir(parents=True,exist_ok=True)
            print(json.dumps({'parallel_stage':stage,'command_index':index,'log':str(log)}),flush=True)
            with log.open('x')as handle:result=subprocess.run(command,cwd=run/'study',env=env,stdout=handle,stderr=subprocess.STDOUT)
            original.save(receipt,{'command':command,'exit_code':result.returncode,'log_sha256':original.sha(log),
                'operational_helper_sha256':frozen.hashes(),'parallel_stage_adapter_sha256':original.sha(Path(__file__)),
                'controller_pause_barrier_pid':a.controller_pid,'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
            if result.returncode:raise RuntimeError('Canonical command failed; preserve outputs and investigate '+str(log))
        original.save(run/f'legacy_{stage}_execution.json',{'status':'ALL_COMMANDS_EXITED_ZERO','stage':stage,'commands':len(commands[stage]),
            'input_identity_certificate_sha256':original.sha(run/'legacy_v2_input_identity.json'),
            'scientific_comparison_still_required':True,'parallel_stage_adapter_sha256':original.sha(Path(__file__)),
            'canonical_original_commands_unchanged':True})

if __name__=='__main__':main()
