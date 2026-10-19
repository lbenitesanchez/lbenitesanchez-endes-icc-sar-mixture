"""Operational continuation of an existing one-worker Gao lane.

Waits for the preceding immutable batch to account for every requested task,
then dispatches its disjoint preplanned shard. No seed/model/score is changed.
"""
from pathlib import Path
import argparse,json,subprocess,sys,time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'resultados/resultados_comparadores'

def main():
 p=argparse.ArgumentParser();prior=p.add_mutually_exclusive_group(required=True)
 prior.add_argument('--after-batch');prior.add_argument('--after-application',action='store_true')
 p.add_argument('--tasks',type=Path,required=True);p.add_argument('--batch-label',required=True)
 p.add_argument('--expected-prior-tasks',type=int);a=p.parse_args()
 source=(OUT/'application_final_candidate_v2/queue_completion.json'if a.after_application
         else OUT/'dispatch_logs'/a.after_batch/'execution.json')
 record=OUT/'dispatch_logs'/('continuation_'+a.batch_label+'.json')
 while not source.exists():time.sleep(15)
 previous=json.loads(source.read_text())
 if a.after_application:
  if not previous.get('completed')or previous.get('years')!=2 or not previous.get('all_final_endpoints_passed'):
   raise RuntimeError('The two application refits have not both passed; preserve the borrowed lane for inspection')
 else:
  accounted=previous['completed_prediction_artifacts']+previous['explicit_failure_artifacts']
  if not previous['completed_execution']or previous['missing_artifacts']or previous['requested_fits']!=a.expected_prior_tasks or accounted!=a.expected_prior_tasks:
   raise RuntimeError('Preceding lane has unaccounted tasks; preserve it and inspect before continuation')
 state={'state':'dispatching','utc':datetime.now(timezone.utc).isoformat(),
   'after_batch':a.after_batch,'after_application':a.after_application,'batch':a.batch_label,'workers':1,
   'statistical_configuration_changed':False,'seed_replacement':False,
   'performance_inspection':False,'other_project_processes_touched':False}
 record.write_text(json.dumps(state,indent=2)+'\n')
 result=subprocess.run([sys.executable,str(Path(__file__).with_name('gao_dispatch_final.py')),
    '--tasks',str(a.tasks.resolve()),'--batch-label',a.batch_label,'--workers','1'],cwd=ROOT)
 state.update(state='finished',returncode=result.returncode,finished_utc=datetime.now(timezone.utc).isoformat())
 record.write_text(json.dumps(state,indent=2)+'\n')
 if result.returncode:raise RuntimeError('Continuation failed; preserve batch artifacts')
if __name__=='__main__':main()
