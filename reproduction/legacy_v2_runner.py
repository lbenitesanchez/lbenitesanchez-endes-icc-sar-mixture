#!/usr/bin/env python3
"""Separate legacy rerun with an explicit DGP orientation correction.

The new workspace retains frozen original sources. A public geometric rule
corrects the unspecified eigenvector signs during input generation. Every
observed sample and direct input must match the historical model_inputs before
any fit can start; old population arrays and fitted outputs are never copied.
"""
from pathlib import Path
import argparse
import datetime as dt
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import legacy_runner as original

HERE=Path(__file__).resolve().parent
HELPERS=('legacy_v2_runner.py','legacy_v2_worker.py','legacy_orientation_adapter.py',
         'legacy_orientation_rule.json','legacy_input_audit.py','legacy_runner.py','legacy_worker.py',
         'requirements.legacy.lock.txt')
IDENTITY_STATUS='PASS_800_OBSERVED_INPUTS_EXACT_FIT_GEOMETRY_ROUNDOFF'


def hashes():
    return {name:original.sha(HERE/name)for name in HELPERS}


def verify_versions():
    expected={}
    for line in (HERE/'requirements.legacy.lock.txt').read_text().splitlines():
        if line.strip()and not line.startswith('#'):
            name,version=line.split('==');expected[name]=version
    actual={name:importlib.metadata.version(name)for name in expected}
    if actual!=expected:raise RuntimeError('Use the separately pinned legacy environment: '+json.dumps(actual))
    return actual


def reuse_fresh_compilation(previous,study,run_root):
    if (run_root/'legacy_v2_compilation_transfer.json').exists():return
    certificate=previous/'legacy_initial_setup_completion.json'
    if not certificate.is_file()or not json.loads(certificate.read_text()).get('complete'):
        raise RuntimeError('Previous workspace has no fresh successful compilation/setup certificate')
    old=previous/'study/scripts/competitors';new=study/'scripts/competitors'
    original_source_hashes={str(p.relative_to(old)):original.sha(p)for p in old.rglob('*.stan')}
    new_source_hashes={str(p.relative_to(new)):original.sha(p)for p in new.rglob('*.stan')}
    if original_source_hashes!=new_source_hashes:raise RuntimeError('Stan sources differ; do not transfer executables')
    transferred={}
    for name in ('hogg_stage1','hogg_stage2','hogg_stage2_design_extension','hogg_stage2_bym2','beta_power_hybrid'):
        source=old/name
        if not source.is_file():raise FileNotFoundError(source)
        stamp=old/(name+'_compiled_source.json')
        if stamp.exists():
            stamp_record=json.loads(stamp.read_text())
            if stamp_record['stan_sha256']!=original.sha(old/(name+'.stan'))or stamp_record['executable_sha256']!=original.sha(source):
                raise RuntimeError('Fresh compiled source stamp differs: '+name)
            shutil.copyfile(stamp,new/stamp.name)
        shutil.copyfile(source,new/name);(new/name).chmod(source.stat().st_mode)
        transferred[name]=original.sha(source)
    original.save(run_root/'legacy_v2_compilation_transfer.json',{
        'classification':'REUSE_EXECUTABLES_COMPILED_FROM_ZERO_IN_REJECTED_INPUT_RUN',
        'previous_private_workspace':str(previous),'previous_setup_certificate_sha256':original.sha(certificate),
        'frozen_Stan_sources_byte_identical':True,'Stan_source_sha256':new_source_hashes,
        'fresh_executable_sha256':transferred,'old_fitted_predictions_draws_or_response_inputs_copied':False})


def plan(args,study):
    commands=original.plan(args,study)
    worker=HERE/'legacy_v2_worker.py'
    base=[sys.executable,str(worker),'--study-root',str(study),'--orientation-rule',str(args.orientation_rule),
          '--workers',str(args.workers)]
    commands['previous800'][1]=base+['prepare800']
    commands['previous800'].insert(2,base+['--reference-root',str(args.reference_root),
        'verify-input-identity','--output',str(args.run_root/'legacy_v2_input_identity.json')])
    if args.input_catalogue:
        commands['previous800'][2]+=['--input-catalogue',str(args.input_catalogue)]
    return commands


def require_identity(args):
    path=args.run_root/'legacy_v2_input_identity.json'
    record=json.loads(path.read_text())
    if record['status']!=IDENTITY_STATUS or record['worlds']!=800 or record['failures']:
        raise RuntimeError('All 800 observed inputs must match before fitting')
    proof=json.loads((args.run_root/'legacy_v2_generation_contract.json').read_text())
    if proof['operational_helper_sha256']!=hashes():
        raise RuntimeError('V2 helper/rule bytes changed after generation; preserve and investigate')
    if record['orientation_rule_sha256']!=original.sha(args.orientation_rule):
        raise RuntimeError('Identity certificate belongs to a different orientation rule')
    return path


def execute(args,commands,inputs_only=False):
    env=os.environ.copy()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):env[name]='1'
    env['ENDES_STUDY_ROOT']=str(args.run_root/'study');env['ENDES_EXPORT_PRIVATE_INPUTS']='1'
    if (args.run_root/'study/.Rlib').exists():env['ENDES_R_LIB']=str((args.run_root/'study/.Rlib').resolve())
    stages=('previous800',)if inputs_only else original.STAGES if args.stage=='all'else(args.stage,)
    for stage in stages:
        selected=commands[stage][:3]if inputs_only else commands[stage]
        if not inputs_only and stage!='previous800':require_identity(args)
        for index,command in enumerate(selected):
            if not inputs_only and stage=='previous800'and index>=3:require_identity(args)
            receipt=args.run_root/'receipts'/stage/f'{index:03d}.json'
            if receipt.exists():
                old=json.loads(receipt.read_text())
                if old['command']!=command or old['exit_code']!=0 or old['operational_helper_sha256']!=hashes():
                    raise RuntimeError('Investigate incomplete or changed command: '+str(receipt))
                continue
            log=args.run_root/'logs'/stage/f'{index:03d}.log';log.parent.mkdir(parents=True,exist_ok=True)
            print(json.dumps({'stage':stage,'command_index':index,'log':str(log),'started_utc':dt.datetime.now(dt.timezone.utc).isoformat()}),flush=True)
            with log.open('x')as handle:
                result=subprocess.run(command,cwd=args.run_root/'study',env=env,stdout=handle,stderr=subprocess.STDOUT)
            original.save(receipt,{'command':command,'exit_code':result.returncode,'log_sha256':original.sha(log),
                'operational_helper_sha256':hashes(),'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
            if result.returncode:raise RuntimeError('Command failed; preserve outputs and inspect '+str(log))
        if not inputs_only:
            original.save(args.run_root/f'legacy_{stage}_execution.json',{
                'status':'ALL_COMMANDS_EXITED_ZERO','stage':stage,'commands':len(selected),
                'input_identity_certificate_sha256':original.sha(require_identity(args)),
                'scientific_comparison_still_required':True,'input_generation_correction':'explicit public DGP eigenvector orientation'})
    if inputs_only:
        certificate=require_identity(args)
        original.save(args.run_root/'legacy_initial_setup_completion.json',{
            'complete':True,'800_observed_input_identity_confirmed':True,
            'input_identity_certificate_sha256':original.sha(certificate),'frozen_sources_checked':len(original.check_sources(args.run_root/'study')),
            'all_fits_pending':True,'input_generation_correction_explicit':True,'operational_helper_sha256':hashes()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-root','source-root','run-root'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('cmdstan','r-lib','shared-private-root','reuse-fresh-compilation'):p.add_argument('--'+name,type=Path)
    p.add_argument('--orientation-rule',type=Path,default=HERE/'legacy_orientation_rule.json')
    p.add_argument('--input-catalogue',type=Path,help='Public observed-input checksum catalogue; no private historical model_inputs needed')
    p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--action',choices=('prepare','preflight','plan','inputs-only','run'),default='prepare')
    p.add_argument('--stage',choices=(*original.STAGES,'all'),default='all')
    args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.expanduser().resolve())
    if args.run_root.is_relative_to(HERE.parent):raise ValueError('Private outputs must stay outside the public repository')
    if args.shared_private_root is None:args.shared_private_root=args.run_root.parent/'real_private_study'
    if args.orientation_rule!=HERE/'legacy_orientation_rule.json':raise ValueError('Production uses the reviewed immutable public orientation rule')
    versions=verify_versions();study=original.prepare(args)
    if args.reuse_fresh_compilation:reuse_fresh_compilation(args.reuse_fresh_compilation,study,args.run_root)
    contract=args.run_root/'legacy_v2_generation_contract.json'
    record={'classification':'INPUT_GENERATION_CORRECTION','operational_helper_sha256':hashes(),'pinned_Python_packages':versions,
        'original_sources_retained':True,'old_population_cache_directories_forbidden':True,
        'archive_generating_geometry_truth_response_or_fit_outputs_loaded_by_generation':False,
        'all_800_observed_inputs_required_before_fits':True,'first_failed_attempt_preserved':str(args.reuse_fresh_compilation)if args.reuse_fresh_compilation else None}
    if contract.exists()and json.loads(contract.read_text())!=record:raise RuntimeError('Generation contract changed; preserve workspace')
    original.save(contract,record);commands=plan(args,study);original.save(args.run_root/'legacy_commands.json',commands)
    if args.action=='preflight':print(json.dumps(original.preflight(args,study),indent=2))
    elif args.action in ('inputs-only','run'):execute(args,commands,args.action=='inputs-only')
    else:print(json.dumps({'status':'PREPARED_V2_NO_FITS','plan':str(args.run_root/'legacy_commands.json'),'workers':args.workers},indent=2))


if __name__=='__main__':main()
