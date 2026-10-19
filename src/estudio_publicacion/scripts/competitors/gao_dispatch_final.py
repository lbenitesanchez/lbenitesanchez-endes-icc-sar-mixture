"""Dispatch frozen Gao fits after independently validated synthetic preparation.

This orchestration helper adds no statistical choices. It writes only derived
blinded observations, task shards, execution logs and completion counters.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(key,'1')
import pandas as pd
from gao_prepare_extension import prepare_synthetic

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'resultados/resultados_comparadores'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check_frozen():
    protocol=json.loads((OUT/'protocol_freeze.json').read_text())
    if not protocol['approved_for_independent_evaluation']:raise RuntimeError('Approved freeze required')
    for name,wanted in protocol['source_sha256'].items():
        path=ROOT/'scripts/models.py' if name=='models.py' else Path(__file__).with_name(name)
        if sha(path)!=wanted:raise RuntimeError('Frozen source differs: '+name)
    manifest=ROOT/'resultados/competidores_gao/gao_frozen_code_manifest.json'
    if sha(manifest)!=protocol['competitor_freeze_manifest_sha256']['gao']:
        raise RuntimeError('Gao frozen manifest differs')
    for item in json.loads(manifest.read_text())['files']:
        if sha(ROOT/item['file'])!=item['sha256']:raise RuntimeError('Gao source differs: '+item['file'])
    return protocol

def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--batch-label',required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    if not 1<=args.workers<=6:raise ValueError('Use1..6 single-thread INLA processes within the approved CPU budget')
    protocol=check_frozen();tasks=pd.read_csv(args.tasks).fillna('')
    if tasks.duplicated(['scenario','replication','method']).any():raise ValueError('Duplicate requested fit')
    planned=protocol['independent_seed_ids']
    observations=[]
    for row in tasks.to_dict('records'):
        if int(row['replication']) not in planned[row['scenario']]:raise ValueError('Unplanned population')
        indir=Path(row['model_inputs']);folder=indir.parent
        if indir.name!='model_inputs' or not indir.resolve().is_relative_to(OUT.resolve()):raise ValueError('Only blinded approved input folders permitted')
        meta=json.loads((folder/'manifest.json').read_text())
        for name,wanted in meta['input_sha256'].items():
            if sha(indir/name)!=wanted:raise RuntimeError('Input hash differs: '+str(indir/name))
        if row['stage2_mode']=='common_continuity':
            target=Path(row['extension_file'])
            if target.resolve().is_relative_to(indir.resolve()):raise ValueError('Derived extension must stay outside frozen input folder')
            if target.exists():
                recorded=json.loads(target.with_suffix('.json').read_text())
                if recorded['input_sha256']!=sha(indir/'direct.npz') or recorded['primary_models_source_sha256']!=protocol['source_sha256']['models.py']:
                    raise RuntimeError('Existing extension provenance differs')
            else:prepare_synthetic(indir,target)
            observations.append(dict(scenario=row['scenario'],replication=int(row['replication']),file=str(target.relative_to(ROOT)),sha256=sha(target)))
    directory=OUT/'dispatch_logs'/args.batch_label;directory.mkdir(parents=True,exist_ok=True)
    provenance=dict(requested_fits=len(tasks),blinded_extension_files=len(observations),extensions=observations,
        task_sha256=sha(args.tasks),protocol_sha256=sha(OUT/'protocol_freeze.json'),dispatcher_sha256=sha(__file__),statistical_sources_unchanged=True,truth_access=False)
    (directory/'preparation.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps({k:v for k,v in provenance.items() if k!='extensions'}),flush=True)
    if args.prepare_only:return
    ready=OUT/'input_validation_final800.json'
    if not ready.exists() or not json.loads(ready.read_text()).get('passed'):
        raise RuntimeError('Final800 validated marker required before dispatch; respect shared CPU budget externally')
    if (directory/'execution.json').exists():raise FileExistsError('Use a fresh execution batch label')
    processes=[];logs=[];started=time.monotonic()
    try:
        for worker in range(args.workers):
            shard=directory/f'tasks_worker{worker+1}.csv';tasks.iloc[worker::args.workers].to_csv(shard,index=False)
            log=(directory/f'worker{worker+1}.log').open('w');logs.append(log)
            process=subprocess.Popen(['Rscript',str(Path(__file__).with_name('gao_worker.R')),str(shard)],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
            processes.append(process)
        returns=[process.wait() for process in processes]
    finally:
        for log in logs:log.close()
    completed=failed=0
    for row in tasks.to_dict('records'):
        folder=Path(row['output_dir'])
        if (folder/'predictions.csv').exists() and (folder/'diagnostics.json').exists():completed+=1
        elif (folder/'failure.json').exists():failed+=1
    result=dict(completed_execution=all(code==0 for code in returns),worker_exit_codes=returns,
        requested_fits=len(tasks),completed_prediction_artifacts=completed,explicit_failure_artifacts=failed,
        missing_artifacts=len(tasks)-completed-failed,seconds=time.monotonic()-started,
        protocol_sha256=provenance['protocol_sha256'],source_sha256=protocol['source_sha256'])
    (directory/'execution.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
