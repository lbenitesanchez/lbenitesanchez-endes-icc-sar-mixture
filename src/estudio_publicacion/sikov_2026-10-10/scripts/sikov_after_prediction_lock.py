"""Operational queue for additive references, B300 and independent audit.

No outcomes are accessed before the complete joint prediction lock. This scheduler
does not modify frozen models, selected weights, scores or existing outputs.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import hashlib,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1]
CONTEXTS=('SAR_probability_srs','SAR_logit_srs','SAR_logit_psu_sparse','IID_logit_psu_sparse')
LOCK=ROOT/'validacion/fresh_prediction_lock.json'
ENV=os.environ.copy()
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    ENV[key]='1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def event(stage,**kwargs):print(json.dumps({'stage':stage,**kwargs}),flush=True)

def one(context,rep):
    from sikov_bootstrap_sensitivity import prediction_gate
    lock_sha,_=prediction_gate(LOCK,context,rep)
    stem=f'sikov_bootstrap_evaluation_{context}_rep{rep:05d}_B300'
    summary=ROOT/'validacion'/f'{stem}.json';arrays=summary.with_suffix('.npz')
    if summary.exists() or arrays.exists():
        if not summary.is_file() or not arrays.is_file():raise RuntimeError('Partial existing B300 output; preserve and investigate '+stem)
        proof=json.loads(summary.read_text())
        if proof['B']!=300 or proof['lock_sha256']!=lock_sha or proof['protocol_sha256']!=sha(ROOT/'protocol_freeze.json') or proof['outputs_sha256']!=sha(arrays) or proof['truth_was_read'] is not False or proof['uncertainty_selection_performed'] is not False:
            raise RuntimeError('Existing B300 output does not match the fixed protocol '+stem)
        if not all(item['converged'] for item in proof['metadata']):raise RuntimeError('Existing B300 fit failed '+stem)
        return {'context':context,'rep':rep,'reused_verified_output':True}
    log=ROOT/'validacion'/f'{stem}.log'
    with log.open('x') as stream:
        subprocess.run([sys.executable,str(ROOT/'scripts/sikov_bootstrap_sensitivity.py'),'--phase','evaluation','--context',context,'--rep',str(rep),'--B','300','--lock',str(LOCK)],cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
    proof=json.loads(summary.read_text())
    return {'context':context,'rep':rep,'elapsed_seconds':proof['elapsed_seconds'],'point_max_difference':proof['closed_point_max_absolute_difference'],'converged':all(item['converged'] for item in proof['metadata'])}

def run_script(script,args,logname):
    log=ROOT/'validacion'/logname
    with log.open('x') as stream:
        subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)

def real_reference(scope):
    event('waiting_for_original_real_fit',scope=scope)
    while not (ROOT/'evaluacion'/('real_'+scope)/'completion.json').is_file():time.sleep(30)
    if not (ROOT/'evaluacion/published_sar'/('real_'+scope)/'completion.json').is_file():
        event('starting_additional_real_reference',scope=scope)
        run_script('sikov_published_fit.py',['--scope',scope],f'sikov_published_real_{scope}.log')
    event('additional_real_reference_complete',scope=scope)

def fresh_references():
    required=[ROOT/'evaluacion/fits'/c/f'rep_{r:05d}/completion.json' for c in CONTEXTS for r in range(30000,30064)]
    event('waiting_for_original_fresh_cohort',worlds=256)
    while not all(path.is_file() for path in required):time.sleep(30)
    event('starting_additional_fresh_references',worlds=256,workers=4)
    run_script('sikov_published_fit.py',['--workers','4','--resume'],'sikov_published_fresh_dispatch.log')
    event('waiting_for_original_and_additional_fresh_prediction_locks')
    # The root joint scheduler alone emits the locks, avoiding concurrent writes.
    while not LOCK.is_file() or not (ROOT/'validacion/sikov_published_fresh_prediction_lock.json').is_file():time.sleep(30)
    protocol=json.loads((ROOT/'protocol_freeze.json').read_text());locked=json.loads(LOCK.read_text())
    if not locked.get('approved_for_scoring') or locked.get('worlds')!=protocol['fresh_worlds'] or locked.get('fresh_truth_read_before_lock') is not False or locked.get('protocol_sha256')!=sha(ROOT/'protocol_freeze.json'):
        raise RuntimeError('Complete, valid prediction lock required')
    event('dispatching_fixed_B300',jobs=8,workers=4,prediction_lock_sha256=sha(LOCK))
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(one,context,rep) for context in CONTEXTS for rep in (30000,30001)]
        results=[job.result() for job in as_completed(jobs)]
        for item in results:event('B300_complete',**item)
    event('additional_fresh_references_and_B300_complete')

def main():
    from sikov_published_fit import gate
    gate()
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks=[pool.submit(real_reference,scope) for scope in ('temporal','territorial')]
        tasks.append(pool.submit(fresh_references))
        for task in as_completed(tasks):task.result()
    event('waiting_for_jointly_closed_fresh_and_external_evaluation')
    required=[ROOT/'validacion/joint_prediction_lock.json',ROOT/'validacion/fresh_evaluation_status.json',ROOT/'validacion/sikov_published_evaluation_status.json',ROOT/'resultados/external_comparison/external_final_summary.json',ROOT/'resultados/external_comparison_Holm12/external_final_summary.json',ROOT/'validacion/align_external_extended_independent_audit.json']
    while not all(path.is_file() for path in required):time.sleep(30)
    event('running_complete_independent_audit')
    log=ROOT/'validacion/sikov_final_independent_audit.log'
    with log.open('x') as stream:
        subprocess.run([sys.executable,str(ROOT/'scripts/sikov_final_audit.py'),'--include-results','--external-dir',str(ROOT/'resultados/external_comparison')],cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
    result=json.loads((ROOT/'validacion/sikov_final_independent_audit.json').read_text())
    if result['status']!='PASS_COMPLETE_COMPARISON_INDEPENDENT_AUDIT':raise RuntimeError('Full audit did not pass')
    event('PASS_COMPLETE_COMPARISON_INDEPENDENT_AUDIT',audit_sha256=sha(ROOT/'validacion/sikov_final_independent_audit.json'))

if __name__=='__main__':main()
