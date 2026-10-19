#!/usr/bin/env python3
"""Operational, reversible Gao queue assistance without source modifications.

Pause only the two named original R parents. Exclude their active tasks plus
a conservative next-row fence. Preserve completed/failure artifacts. The
unchanged dispatcher fits remaining tasks with four separate R processes.
Resume originals ONLY after every assisted task passes their own skip guard.
On any helper/metadata/gate failure they remain paused for explicit review.
"""
from pathlib import Path
import argparse,csv,datetime as dt,hashlib,json,os,re,shlex,signal,subprocess,sys

HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def identity(row):return (row['scenario'],row['replication'],row['method'])
def file_hashes(folder):
    return {name:sha(folder/name)for name in ('predictions.csv','diagnostics.json','failure.json')if(folder/name).is_file()}
def ps(pid):
    return subprocess.check_output(['ps','-p',str(pid),'-o','pid=,ppid=,stat=,command='],text=True).strip()
def process_matches(pid,parent,expected,stopped=False):
    line=ps(pid);parts=line.split(None,3)
    if(parts[0]!=str(pid)or parts[1]!=str(parent)or parts[3]!=expected or
       (stopped and 'T'not in parts[2])):raise RuntimeError('Exact PID/PPID/argv/state identity differs')
    return line
def skipped_by_original(row):
    folder=Path(row['output_dir'])
    if not(folder/'predictions.csv').is_file()or not(folder/'diagnostics.json').is_file():return False
    old=json.loads((folder/'diagnostics.json').read_text());endpoints=old.get('endpoints',{})
    endpoints=list(endpoints.values())if isinstance(endpoints,dict)else endpoints
    return(old.get('variant')==row['variant']and old.get('stage2_mode')==row['stage2_mode']and
           bool(endpoints)and all(e.get('INLA_ok')is True for e in endpoints))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--worker-pids',type=int,nargs=2,required=True)
    p.add_argument('--dispatcher-pid',type=int,required=True)
    p.add_argument('--operation-root',type=Path,required=True)
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--shared-private-root',type=Path,required=True)
    p.add_argument('--execute-pause-and-assist',action='store_true')
    a=p.parse_args();run=a.run_root.resolve();study=run/'study';out=a.operation_root.resolve()
    if out.is_relative_to(HERE.parent)or out.exists():raise ValueError('Use a fresh private operation directory')
    out.mkdir(parents=True)
    gate=json.loads((run/'legacy_v2_input_identity.json').read_text())
    if gate.get('status')!='PASS_800_OBSERVED_INPUTS_EXACT_FIT_GEOMETRY_ROUNDOFF'or gate.get('failures')!=[]:raise RuntimeError('Exact fresh 800-input gate required')
    taskfile=study/'resultados/resultados_comparadores/task_plan/gao_all1800.csv'
    tasks=list(csv.DictReader(taskfile.open()));assert len(tasks)==1800 and len({identity(r)for r in tasks})==1800
    dispatch=study/'resultados/resultados_comparadores/dispatch_logs/fresh_reproduction_gao'
    original_processes={str(pid):ps(pid)for pid in a.worker_pids}
    expected_commands={}
    for index,(pid,line)in enumerate(original_processes.items(),1):
        command=line.split(None,3)[3];tokens=shlex.split(command)
        suffix=['--no-echo','--no-restore','--file='+str(study/'scripts/competitors/gao_worker.R'),
                '--args',str(dispatch/f'tasks_worker{index}.csv')]
        if(str(a.dispatcher_pid)!=line.split()[1]or len(tokens)!=6 or Path(tokens[0]).name!='R'or tokens[1:]!=suffix):
            raise RuntimeError('PID ownership/worker command differs; no signals sent')
        expected_commands[int(pid)]=command
    source_files={str(path.relative_to(study)):sha(path)for path in (study/'scripts/competitors').glob('gao*')if path.is_file()and path.suffix in('.py','.R')}
    state={'status':'PLAN_ONLY_NO_SIGNALS','original_worker_processes':original_processes,
        'dispatcher_pid':a.dispatcher_pid,'full_task_sha256':sha(taskfile),'source_sha256':source_files,
        'input_identity_certificate_sha256':sha(run/'legacy_v2_input_identity.json'),
        'operational_helper_sha256':sha(Path(__file__)),'new_R_workers':4,
        'sources_models_seeds_and_diagnostic_gates_changed':False,'original_controller_retired':False}
    save(out/'operation.json',state)
    if not a.execute_pause_and_assist:print(json.dumps(state,indent=2));return
    # Never send a signal to the controller, dispatcher, native INLA children,
    # or unrelated projects. The R parents cannot advance their loaded queues.
    paused=[]
    for pid in a.worker_pids:
        process_matches(pid,a.dispatcher_pid,expected_commands[pid])
        os.kill(pid,signal.SIGSTOP);paused.append(pid)
        state.update(status='ORIGINAL_R_PARENTS_PAUSED_PREPARING_INVENTORY',paused_worker_pids=paused)
        save(out/'operation.json',state)
    stopped={str(pid):process_matches(pid,a.dispatcher_pid,expected_commands[pid],True)for pid in a.worker_pids}
    state['paused_processes']=stopped;save(out/'operation.json',state)
    protected=[];active=[]
    for index,pid in enumerate(a.worker_pids,1):
        log=dispatch/f'worker{index}.log'
        matches=re.findall(r'^TASK\s+(\d+)\s+(\d+)\s+(.+)$',log.read_text(),re.M)
        if not matches:raise RuntimeError('Active task cannot be resolved; originals remain paused')
        ordinal,total,directory=matches[-1];shard=list(csv.DictReader((dispatch/f'tasks_worker{index}.csv').open()))
        row=shard[int(ordinal)-1]
        if total!=str(len(shard))or directory.strip()!=row['output_dir']:raise RuntimeError('Log/loaded queue mismatch; originals remain paused')
        active.append({'pid':pid,'queue_position':int(ordinal),'task':identity(row),'output_dir':row['output_dir']})
        protected.append(row)
        if int(ordinal)<len(shard):protected.append(shard[int(ordinal)])
    protected_ids={identity(r)for r in protected};before={};pending=[];explicit_failures=[]
    for row in tasks:
        folder=Path(row['output_dir']);files=file_hashes(folder)
        if identity(row)in protected_ids:continue
        if 'predictions.csv'in files and 'diagnostics.json'in files:
            if not skipped_by_original(row):raise RuntimeError('Existing pair fails original skip guard; remain paused')
            before[row['output_dir']]=files
        elif 'failure.json'in files:before[row['output_dir']]=files;explicit_failures.append(identity(row))
        else:
            # No unprotected partial file may be overwritten. A paused worker
            # could only own its last TASK; unexpected partials require review.
            if folder.exists()and any(folder.iterdir()):raise RuntimeError('Unexpected unowned partial output; originals remain paused')
            pending.append(row)
    state.update(status='ORIGINAL_R_PARENTS_PAUSED_PENDING_HELPER',paused_processes=stopped,
        active_tasks=active,protected_task_ids=list(protected_ids),preserved_completed_or_failure_outputs=len(before),
        preexisting_explicit_failures=list(explicit_failures),assisted_tasks=len(pending),paused_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    save(out/'operation.json',state);save(out/'preserved_output_fingerprints.json',before)
    # Original workers would reattempt failure-only tasks; therefore they are
    # never resumed automatically when a preexisting/new failure is present.
    if explicit_failures:raise RuntimeError('Failure-only output cannot be skipped by originals; remain paused for coordination')
    shard=out/'assisted_tasks.csv'
    with shard.open('w',newline='')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(tasks[0]));writer.writeheader();writer.writerows(pending)
    label='runtime_assistance_'+out.name
    command=[sys.executable,str(HERE/'legacy_worker.py'),'--study-root',str(study),
        '--reference-root',str(a.reference_root.resolve()),'--shared-private-root',str(a.shared_private_root.resolve()),
        'script','scripts/competitors/gao_dispatch_final.py','--tasks',str(shard),'--batch-label',label,'--workers','4']
    state.update(command=command,assisted_task_sha256=sha(shard),status='ASSISTANCE_RUNNING_ORIGINALS_PAUSED')
    save(out/'operation.json',state);print(json.dumps({'assisted_tasks':len(pending),'active_original_tasks':active,'protected_fence_tasks':len(protected_ids),'new_workers':4}),flush=True)
    with(out/'assistance.log').open('x')as handle:
        result=subprocess.run(command,cwd=study,stdout=handle,stderr=subprocess.STDOUT)
    checks={'helper_exit_code':result.returncode,'all_assisted_tasks_satisfy_original_skip_guard':all(skipped_by_original(row)for row in pending),
        'preserved_finished_outputs_byte_identical':all(file_hashes(Path(folder))==files for folder,files in before.items()),
        'source_files_byte_identical':all(sha(study/name)==value for name,value in source_files.items()),
        'full_task_plan_byte_identical':sha(taskfile)==state['full_task_sha256']}
    state.update(status='ASSISTANCE_FAILED_ORIGINALS_STILL_PAUSED',completion_checks=checks,
                 finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    save(out/'operation.json',state)
    if result.returncode or not all(value for key,value in checks.items()if key!='helper_exit_code'):
        raise RuntimeError('No automatic SIGCONT: preserve failures and request coordination')
    for pid in a.worker_pids:
        process_matches(pid,a.dispatcher_pid,expected_commands[pid],True)
    for pid in a.worker_pids:os.kill(pid,signal.SIGCONT)
    state.update(status='ALL_ASSISTED_TASKS_PASS_ORIGINAL_R_PARENTS_RESUMED',
                 original_R_worker_pids_resumed=a.worker_pids,numerical_failures_retried=False,
                 original_active_tasks_were_not_dispatched_by_helper=True)
    save(out/'operation.json',state);print(json.dumps({'status':state['status'],'checks':checks},indent=2),flush=True)

if __name__=='__main__':main()
