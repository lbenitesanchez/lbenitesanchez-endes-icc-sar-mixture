"""Operational joint seal before any fresh truth or held-out 2024 scoring.

The original and additional scientific sources stay frozen. This validates
all 256 original/additional worlds and both original/additional actual fits;
it neither fits models nor opens responses. Its CLI also runs the two closed
synthetic score collectors after writing the joint seal.
"""
from pathlib import Path
import argparse,datetime as dt,hashlib,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1]
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def command(*args):
    subprocess.run([sys.executable,str(ROOT/'scripts'/args[0]),*args[1:]],
                   cwd=ROOT,check=True)

def ready():
    from comparison_simulation import CONTEXTS,EVALUATION_IDS
    expected=[ROOT/'evaluacion'/family/c/f'rep_{r:05d}/completion.json'
              for family in ('fits','published_sar')
              for c in CONTEXTS for r in EVALUATION_IDS]
    expected += [ROOT/'evaluacion'/prefix/('real_'+scope)/'completion.json'
                 for prefix in ('published_sar',)for scope in ('temporal','territorial')]
    expected += [ROOT/'evaluacion'/('real_'+scope)/'completion.json'
                 for scope in ('temporal','territorial')]
    return all(p.is_file()for p in expected)

def close():
    from sikov_published_fit import gate
    addendum,addsha=gate()
    protocolsha=SHA(ROOT/'protocol_freeze.json')
    command('comparison_collect.py','--lock-only')
    command('sikov_published_collect.py','--lock-only')
    first=ROOT/'validacion/fresh_prediction_lock.json'
    second=ROOT/'validacion/sikov_published_fresh_prediction_lock.json'
    a=json.loads(first.read_text());b=json.loads(second.read_text())
    if not(a['worlds']==b['worlds']==256)or a['protocol_sha256']!=protocolsha:
        raise RuntimeError('Joint lock requires both complete cohorts')
    if b['protocol_sha256']!=protocolsha or b['addendum_sha256']!=addsha:
        raise RuntimeError('Additional cohort has a different freeze')
    expected={(r['context'],r['replication'])for r in a['prediction_items']}
    if len(expected)!=256 or expected!={(r['context'],r['replication'])for r in b['prediction_items']}:
        raise RuntimeError('Different paired population cohorts')
    if a['fresh_truth_read_before_lock']or b['fresh_truth_read_before_lock']:
        raise RuntimeError('Fresh truth used before sealing predictions')
    paths=[ROOT/'protocol_freeze.json',ROOT/'validacion/sikov_published_addendum_freeze.json',first,second]
    realflags={}
    for scope in ('temporal','territorial'):
        original=ROOT/'evaluacion'/('real_'+scope)
        published=ROOT/'evaluacion/published_sar'/('real_'+scope)
        old=json.loads((original/'completion.json').read_text())
        new=json.loads((published/'completion.json').read_text())
        for folder,record in ((original,old),(published,new)):
            if record['protocol_sha256']!=protocolsha or record['heldout_outcomes_read_by_fits']:
                raise RuntimeError('Wrong actual fit role')
            for filename,digest in record['output_sha256'].items():
                if SHA(folder/filename)!=digest:raise RuntimeError('Changed actual fit output')
            if SHA(folder/'predictions.csv')!=record['predictions_sha256']:
                raise RuntimeError('Actual predictions differ from completion')
            paths += [folder/'completion.json',folder/'predictions.csv']
        if new['addendum_sha256']!=addsha or new['original_completion_sha256']!=SHA(original/'completion.json'):
            raise RuntimeError('Additional actual fit does not preserve its original')
        if new['original_predictions_sha256']!=SHA(original/'predictions.csv'):
            raise RuntimeError('Changed original actual prediction seal')
        required=('Selected','Sikov_FH','Sikov_SAR')
        if not all(old['convergence'].get(k,False)for k in required):
            raise RuntimeError('Actual primary original methods did not converge')
        if not new['convergence'].get('Sikov_SAR_Published',False):
            raise RuntimeError('Actual published SAR did not converge')
        realflags[scope]={'primary_models_converged':True,
                         'old_all_methods_converged':old['all_converged'],
                         'additional_all_methods_converged':new['all_converged']}
        for filename in ('real_prediction_lock_'+scope+'.json',
                         'sikov_published_prediction_lock_'+scope+'.json'):
            paths.append(ROOT/'validacion'/filename)
    gate()
    result={'approved_for_scoring':True,'locked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
      'protocol_sha256':protocolsha,'addendum_sha256':addsha,'fresh_cohort_size':256,
      'all_expected_worlds':True,'fresh_original_prediction_lock_sha256':SHA(first),
      'fresh_published_prediction_lock_sha256':SHA(second),
      'fresh_truth_read_before_joint':False,'heldout2024responses_used_before_joint':False,
      'heldout_outcomes_used':False,'selected_recipe_unchanged':True,
      'fresh_convergence_failures_retained':{'original':a['convergence_failures_retained'],
                                           'additional':b['convergence_failures_retained']},
      'actual_numerical_gates':realflags,'operational_seal_source_sha256':SHA(__file__),
      'files':[{'path':str(p.resolve()),'sha256':SHA(p)}for p in paths]}
    dest=ROOT/'validacion/joint_prediction_lock.json'
    if dest.exists():
        previous=json.loads(dest.read_text())
        if previous['files']!=result['files']or previous['protocol_sha256']!=protocolsha:
            raise RuntimeError('Existing joint seal differs')
        return previous
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS_JOINT_PREDICTION_SEAL_BEFORE_ANY_SCORING',flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true')
    p.add_argument('--score-fresh',action='store_true');args=p.parse_args()
    if args.wait:
        while not ready():time.sleep(15)
    elif not ready():raise RuntimeError('Complete joint prediction cohort not yet available')
    close()
    if args.score_fresh:
        command('comparison_collect.py')
        command('sikov_published_collect.py')
        command('comparison_figures.py')
        print('PASS_COMPLETE_ORIGINAL_AND_PUBLISHED_FRESH_SCORING',flush=True)

if __name__=='__main__':main()
