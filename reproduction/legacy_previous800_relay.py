#!/usr/bin/env python3
"""Honest operational handoff of the unfinished legacy previous800 tail.

The original controller stays stopped while canonical side-stage writers run.
Its first five fresh receipts and terminal Gao R-worker certificate are adopted
explicitly. New commands have separate real argv, logs and receipts. Hogg uses
four workers and a new immutable copy of byte-identical scientific sources and
already freshly compiled executables. No completed fit is dispatched again.
"""
from pathlib import Path
import argparse,csv,datetime as dt,json,os,shutil,signal,subprocess,sys,time
import legacy_runner as original
import legacy_v2_runner as frozen
from legacy_accelerate_gao import skipped_by_original

HERE=Path(__file__).resolve().parent
NAMES=('hogg_tsln.py','hogg_diagnostics.py','hogg_design_extension.py',
       'hogg_stage1.stan','hogg_stage2.stan','hogg_stage2_design_extension.stan',
       'hogg_stage1','hogg_stage2','hogg_stage2_design_extension',
       'hogg_design.md','hogg_strong_design.md','hogg_benchmark_batch.py')
SIDE_STAGES=('cut40','application2025','external2024','cut16')

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(path):return json.loads(path.read_text())
def ps(pid):
    result=subprocess.run(['ps','-p',str(pid),'-o','pid=,ppid=,stat=,command='],capture_output=True,text=True)
    return result.stdout.strip() if result.returncode==0 else ''

def gate(run):
    args=argparse.Namespace(run_root=run,orientation_rule=HERE/'legacy_orientation_rule.json')
    path=frozen.require_identity(args)
    strict=read(run/'legacy_public_strict_input_identity.json')
    if strict.get('status')!=frozen.IDENTITY_STATUS or strict.get('worlds')!=800 or strict.get('failures'):
        raise RuntimeError('Public strict 800-input identity certificate is required')
    original.check_sources(run/'study')
    return {'private_identity_sha256':original.sha(path),
            'public_strict_identity_sha256':original.sha(run/'legacy_public_strict_input_identity.json')}

def barrier(pid,expected):
    line=ps(pid);parts=line.split(None,3)
    if len(parts)!=4 or parts[0]!=str(pid) or 'T'not in parts[2] or parts[3]!=expected:
        raise RuntimeError('Exact original controller identity/stopped barrier is absent')
    return line

def make_snapshot(run,destination):
    core=run/'study/scripts/competitors';transfer=read(run/'legacy_v2_compilation_transfer.json')
    for name in ('hogg_stage1','hogg_stage2','hogg_stage2_design_extension'):
        stamp=read(core/(name+'_compiled_source.json'))
        if (stamp.get('stan_sha256')!=original.sha(core/(name+'.stan')) or
            stamp.get('executable_sha256')!=original.sha(core/name) or
            transfer['fresh_executable_sha256'][name]!=original.sha(core/name)):
            raise RuntimeError('Fresh compilation binding differs: '+name)
    if destination.exists():
        manifest=read(destination/'snapshot_manifest.json')
        if set(manifest['source_sha256'])!=set(NAMES):raise RuntimeError('Snapshot inventory differs')
        for name,expected in manifest['source_sha256'].items():
            if original.sha(destination/name)!=expected or original.sha(core/name)!=expected:
                raise RuntimeError('Snapshot/source bytes differ: '+name)
        return manifest
    destination.mkdir(parents=True)
    for name in NAMES:shutil.copy2(core/name,destination/name)
    manifest={'source_sha256':{name:original.sha(destination/name)for name in NAMES},
        'core_scientific_frozen_records':{name:original.sha(run/'study/resultados/competidores_hogg'/name)
            for name in ('hogg_sampler_freeze.json','hogg_strong_arms_freeze.json')},
        'utc':now(),'implementation':'Independent implementation; no author repository or unlicensed source copied.',
        'backend':'CmdStan2.40.0 exactNUTS; unchanged modularGaussianS2 integrated analytically.',
        'operational_classification':'BYTE_IDENTICAL_SOURCE_AND_FRESH_EXECUTABLE_SNAPSHOT_NO_SHARED_COMPILATION',
        'fresh_compilation_transfer_certificate_sha256':original.sha(run/'legacy_v2_compilation_transfer.json'),
        'new_snapshot_copied_fitted_outputs_or_draws':False}
    original.save(destination/'snapshot_manifest.json',manifest)
    return manifest

def adopt_prefix(run,commands):
    adopted=[]
    for index in range(5):
        path=run/'receipts/previous800'/f'{index:03d}.json';record=read(path)
        if record['command']!=commands[index] or record['exit_code']!=0 or record['operational_helper_sha256']!=frozen.hashes():
            raise RuntimeError('Fresh original prefix receipt differs or failed')
        log=run/'logs/previous800'/f'{index:03d}.log'
        if original.sha(log)!=record['log_sha256']:raise RuntimeError('Adopted original log changed')
        adopted.append({'original_index':index,'receipt_sha256':original.sha(path),'exit_code':0,
                        'classification':'ADOPT_COMPLETED_FRESH_ORIGINAL_COMMAND_RECEIPT'})
    return adopted

def gao_terminal(run,operation):
    state=read(operation/'operation.json')
    if state.get('status')!='ALL_ASSISTED_TASKS_PASS_ORIGINAL_R_PARENTS_RESUMED':return None
    path=run/'study/resultados/resultados_comparadores/dispatch_logs/fresh_reproduction_gao/execution.json'
    if not path.exists():return None
    record=read(path)
    if (record.get('completed_execution')is not True or record.get('worker_exit_codes')!=[0,0] or
        record.get('requested_fits')!=1800 or record.get('completed_prediction_artifacts')!=1800 or
        record.get('explicit_failure_artifacts')!=0 or record.get('missing_artifacts')!=0):
        raise RuntimeError('Original Gao terminal certificate is incomplete or failed; no tail is launched')
    line=ps(state['dispatcher_pid'])
    if line and 'Z'not in line.split(None,3)[2]:return None
    tasks=list(csv.DictReader((run/'study/resultados/resultados_comparadores/task_plan/gao_all1800.csv').open()))
    if len(tasks)!=1800 or not all(skipped_by_original(row)for row in tasks):
        raise RuntimeError('All 1800 Gao pairs must satisfy the original integrity/convergence guard')
    return {'classification':'ADOPT_FRESH_GAO_TERMINAL_R_WORKER_EXIT_CERTIFICATE',
        'terminal_execution_certificate_sha256':original.sha(path),'assistance_certificate_sha256':original.sha(operation/'operation.json'),
        'R_worker_exit_codes':[0,0],'requested_fits':1800,'complete_valid_prediction_pairs':1800,
        'original_python_dispatcher_has_terminated':True,
        'original_controller_process_exit_and_command_005_receipt_observed':False}

def prepare(a):
    run=a.run_root.resolve();op=run/'previous800_relay';op.mkdir(exist_ok=True)
    binding=read(run/'controller_parallel_barrier.json')
    expected=binding['before'].split(None,3)[3]
    barrier(a.controller_pid,expected);inputs=gate(run);frozen.verify_versions()
    plan=read(run/'legacy_commands.json');commands=plan['previous800']
    prefix=adopt_prefix(run,commands);snapshot=make_snapshot(run,op/'source_snapshot')
    batch=run/'study/resultados/competidores_hogg/fresh_reproduction_hogg'
    if batch.exists():raise RuntimeError('Hogg final batch already exists; no redispatch allowed')
    tail=[list(command)for command in commands[6:]]
    index=tail[1].index('--workers');tail[1][index+1]='4'
    tail[1]+=['--source-snapshot',str(op/'source_snapshot')]
    record={'classification':'OPERATIONAL_CONTROLLER_HANDOFF_UNCHANGED_MODELS_SEEDS_AND_GATES',
        'prepared_utc':now(),'original_controller_pid':a.controller_pid,'original_controller_command':expected,
        'original_plan_sha256':original.sha(run/'legacy_commands.json'),'operational_relay_sha256':original.sha(Path(__file__)),
        'frozen_operational_helper_sha256':frozen.hashes(),'input_identity':inputs,'adopted_prefix':prefix,
        'new_tail_commands':tail,'tail_original_indices':list(range(6,20)),
        'Hogg_worker_count_actually_requested':4,'source_snapshot_sha256':original.sha(op/'source_snapshot/snapshot_manifest.json'),
        'frozen_snapshot_source_sha256':snapshot['source_sha256'],'new_controller_does_not_claim_original_controller_exit_zero':True}
    path=op/'relay_plan.json'
    if path.exists():
        prior=read(path)
        for key in record:
            if key!='prepared_utc' and prior.get(key)!=record[key]:raise RuntimeError('Prepared relay contract changed')
        return run,op,prior
    original.save(path,record);return run,op,record

def execute(a):
    run,op,plan=prepare(a)
    owner=op/'execution_owner.json'
    with owner.open('x')as handle:json.dump({'pid':os.getpid(),'started_utc':now(),'relay_plan_sha256':original.sha(op/'relay_plan.json')},handle)
    terminal=None
    print(json.dumps({'status':'WAITING_FOR_ORIGINAL_GAO_TERMINAL_ALL_1800','tail_commands':14,'Hogg_workers':4}),flush=True)
    while terminal is None:
        barrier(a.controller_pid,plan['original_controller_command'])
        terminal=gao_terminal(run,a.gao_operation_root.resolve())
        if terminal is None:time.sleep(15)
    original.save(op/'adopted_gao_terminal.json',terminal)
    env=os.environ.copy()
    for name in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):env[name]='1'
    env['ENDES_STUDY_ROOT']=str(run/'study');env['ENDES_EXPORT_PRIVATE_INPUTS']='1'
    env['ENDES_R_LIB']=str((run/'study/.Rlib').resolve())
    for index,command in zip(plan['tail_original_indices'],plan['new_tail_commands']):
        barrier(a.controller_pid,plan['original_controller_command']);gate(run)
        if original.sha(Path(__file__))!=plan['operational_relay_sha256']:raise RuntimeError('Live relay bytes changed')
        receipt=run/'receipts/previous800_relay'/f'{index:03d}.json'
        if receipt.exists():raise RuntimeError('Do not rerun an existing relay command; inspect receipt first')
        log=run/'logs/previous800_relay'/f'{index:03d}.log';log.parent.mkdir(parents=True,exist_ok=True)
        print(json.dumps({'relay_command_index':index,'actual_workers':4 if index==7 else None,'started_utc':now()}),flush=True)
        started=time.monotonic()
        with log.open('x')as handle:result=subprocess.run(command,cwd=run/'study',env=env,stdout=handle,stderr=subprocess.STDOUT)
        original.save(receipt,{'command':command,'exit_code':result.returncode,'log_sha256':original.sha(log),
            'operational_helper_sha256':frozen.hashes(),'relay_helper_sha256':plan['operational_relay_sha256'],
            'relay_plan_sha256':original.sha(op/'relay_plan.json'),'seconds':time.monotonic()-started,'finished_utc':now()})
        if result.returncode:raise RuntimeError('Relay command failed; preserve all results and inspect '+str(log))
    original.save(run/'legacy_previous800_execution.json',{
        'status':'ALL_COMMANDS_EXITED_ZERO','stage':'previous800','commands':20,'actually_executed_new_tail_commands':14,
        'adopted_fresh_prefix_commands':5,'adopted_fresh_Gao_terminal_worker_certificate':True,
        'input_identity_certificate_sha256':plan['input_identity']['private_identity_sha256'],
        'relay_plan_sha256':original.sha(op/'relay_plan.json'),'adopted_gao_terminal_sha256':original.sha(op/'adopted_gao_terminal.json'),
        'scientific_comparison_still_required':True,'original_full_controller_exit_zero_claimed':False,
        'closure_basis':'NEW_TAIL_ACTUAL_ZERO_EXITS_PLUS_EXPLICIT_FRESH_PREFIX_AND_GAO_R_WORKER_CERTIFICATE_ADOPTION',
        'source_models_seeds_and_diagnostic_gates_changed':False,'finished_utc':now()})

def retire(a):
    run=a.run_root.resolve();op=run/'previous800_relay';plan=read(op/'relay_plan.json')
    barrier(a.controller_pid,plan['original_controller_command']);gate(run)
    if gao_terminal(run,a.gao_operation_root.resolve())is None:raise RuntimeError('Gao is not terminal')
    for stage in ('previous800',*SIDE_STAGES):
        if read(run/f'legacy_{stage}_execution.json').get('status')!='ALL_COMMANDS_EXITED_ZERO':raise RuntimeError('Side-stage writer is not finished')
    children=subprocess.check_output(['pgrep','-P',str(a.controller_pid)],text=True).split() if subprocess.run(['pgrep','-P',str(a.controller_pid)],capture_output=True).returncode==0 else []
    if any('Z'not in ps(int(pid)).split(None,3)[2]for pid in children):raise RuntimeError('Original controller still has a living child; do not retire')
    original.save(op/'original_controller_retirement_intent.json',{'classification':'ORIGINAL_STOPPED_CONTROLLER_RETIRED_WITHOUT_FULL_SUCCESS_CLAIM',
        'pid':a.controller_pid,'original_command':plan['original_controller_command'],'before':ps(a.controller_pid),
        'side_stage_writers_terminal':True,'original_full_controller_completed':False,'signal':'SIGKILL_ONLY_NAMED_TOP_CONTROLLER','utc':now()})
    os.kill(a.controller_pid,signal.SIGKILL)
    original.save(op/'original_controller_retirement.json',{'status':'ORIGINAL_CONTROLLER_INTERRUPTED_PRESERVED',
        'pid':a.controller_pid,'original_command':plan['original_controller_command'],'signal':'SIGKILL_ONLY_NAMED_TOP_CONTROLLER',
        'original_full_controller_completed':False,'original_full_controller_exit_zero_claimed':False,'utc':now()})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--controller-pid',type=int,required=True)
    p.add_argument('--gao-operation-root',type=Path,required=True)
    p.add_argument('--action',choices=('prepare','run','retire-original-controller'),required=True)
    a=p.parse_args()
    if a.run_root.resolve().is_relative_to(HERE.parent):raise ValueError('All operational runtime files must stay outside public repository')
    if a.action=='prepare':run,op,plan=prepare(a);print(json.dumps({'status':'PREPARED_NO_FITS','tail_commands':14,'Hogg_workers':4,'source_snapshot_files':len(NAMES)}))
    elif a.action=='run':execute(a)
    else:retire(a)

if __name__=='__main__':main()
