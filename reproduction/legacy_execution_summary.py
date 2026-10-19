#!/usr/bin/env python3
"""Export scalar execution provenance without private paths or process IDs.

This is an operational completion summary, never a numerical-reproduction
certificate. In particular, a mixed relay closure is not relabelled as twenty
observed returns from the interrupted original Python controller.
"""
from pathlib import Path
import argparse,hashlib,json

STAGES=('previous800','cut40','application2025','external2024','cut16')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())if path.exists()else None

def build(run):
    relay=read(run/'previous800_relay/relay_plan.json')
    if not relay or relay.get('classification')!='OPERATIONAL_CONTROLLER_HANDOFF_UNCHANGED_MODELS_SEEDS_AND_GATES':
        raise RuntimeError('This summary requires the separately recorded operational relay plan')
    phases=[]
    for stage in STAGES:
        path=run/f'legacy_{stage}_execution.json';record=read(path)
        phases.append({'stage':stage,'status':record.get('status')if record else'INCOMPLETE_NO_TERMINAL_CERTIFICATE',
            'planned_logical_commands':record.get('commands')if record else None,
            'actually_executed_new_tail_commands':record.get('actually_executed_new_tail_commands')if record else None,
            'adopted_fresh_prefix_commands':record.get('adopted_fresh_prefix_commands')if record else None,
            'adopted_fresh_Gao_terminal_worker_certificate':record.get('adopted_fresh_Gao_terminal_worker_certificate')if record else None,
            'closure_basis':record.get('closure_basis')if record else None,
            'original_full_controller_exit_zero_claimed':record.get('original_full_controller_exit_zero_claimed',False)if record else False,
            'certificate_sha256':sha(path)if record else None})
    complete=all(row['status']=='ALL_COMMANDS_EXITED_ZERO'for row in phases)
    retirement=read(run/'previous800_relay/original_controller_retirement.json')
    identity=read(run/'legacy_v2_input_identity.json')
    gao=read(run/'previous800_relay/adopted_gao_terminal.json')
    return {'format':'iaos-legacy-operational-execution-summary-v1',
        'status':'OPERATIONAL_PHASES_COMPLETE_NUMERICAL_COMPARISON_SEPARATE'if complete else'INCOMPLETE',
        'all_five_phases_terminal':complete,'numerical_reproduction_confirmed':False,
        'numerical_comparison_is_not_inferred_from_successful_execution':True,
        'original_controller':{'status':retirement.get('status')if retirement else'PAUSED_OR_RETIREMENT_NOT_CERTIFIED',
            'full_controller_exit_zero_claimed':False,'full_controller_completed':False},
        'input_identity':{'status':identity.get('status')if identity else'INCOMPLETE',
            'worlds':identity.get('worlds')if identity else None,'csv_files_compared':identity.get('csv_hashes_checked')if identity else None,
            'failed_worlds':len(identity.get('failures',[]))if identity else None},
        'Gao_adoption':{'classification':gao.get('classification')if gao else None,
            'R_worker_exit_codes':gao.get('R_worker_exit_codes')if gao else None,
            'complete_valid_prediction_pairs':gao.get('complete_valid_prediction_pairs')if gao else None,
            'original_controller_command_005_return_observed':False},
        'Hogg_relay_worker_count':relay['Hogg_worker_count_actually_requested'],'stages':phases,
        'private_paths_argv_process_ids_individual_records_draws_exported':False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();record=build(a.run_root.resolve())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':record['status'],'all_five_phases_terminal':record['all_five_phases_terminal']}))

if __name__=='__main__':main()
