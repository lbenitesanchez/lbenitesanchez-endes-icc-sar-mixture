"""S2-only BYM2 sensitivity on the 100 fixed structural BYM2 worlds.

Run only after the separate spatial NUTS pilot and freeze. Reads saved modular
S1 summaries from the frozen nonspatial Hogg fits; does not resample units or
fit stage one. Each requested population remains a reported denominator.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import traceback
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
import pandas as pd
import hogg_tsln as h
import hogg_spatial as sp


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--start',type=int,default=20000)
    p.add_argument('--stop',type=int,default=20100,help='Exclusive fixed population ID boundary')
    p.add_argument('--tag',required=True)
    args=p.parse_args()
    if not 20000<=args.start<args.stop<=20100:raise ValueError('Spatial sensitivity restricted to the 100 fixed structural BYM2 worlds')
    frozen_path=h.ROOT/'resultados/competidores_hogg/hogg_spatial_freeze.json'
    freeze=json.loads(frozen_path.read_text())
    if not freeze.get('approved_after_numerical_pilot'):raise ValueError('Separate spatial pilot and freeze required')
    for name,expected in freeze['source_sha256'].items():
        if sha(h.HERE/name)!=expected:raise ValueError('Spatial frozen source hash changed: '+name)
    source_plan=h.ROOT/'resultados/resultados_comparadores/task_plan/hogg_collector1100.csv'
    plan=pd.read_csv(source_plan)
    # The collector specifies the canonical nonspatial source output. It is
    # selected only by arm/scenario/population ID, never posterior quality or
    # performance. Stage-one numerical failures are recorded as unavailable.
    idcolumn='replication'if 'replication'in plan else'rep_ID'
    if not {'scenario',idcolumn,'output_dir'}.issubset(plan):
        raise ValueError('Canonical Hogg collector requires scenario, replication and output_dir columns')
    selected=plan.loc[(plan.scenario=='sparse_psu_bym2')&
        (plan[idcolumn]>=args.start)&(plan[idcolumn]<args.stop)].copy()
    if len(selected)!=args.stop-args.start or selected[idcolumn].duplicated().any():raise ValueError('Requested spatial source set is incomplete or duplicated')
    for directory in selected.output_dir:
        source=Path(directory)
        if not any((source/name).exists()for name in ['task_execution.json','worker_failure.json','execution_failure.json']):
            raise FileNotFoundError('Source S1 task is still pending; do not classify scheduling as a statistical failure: '+str(source))
    output=h.ROOT/'resultados/competidores_hogg'/args.tag
    if output.exists():raise FileExistsError('Use a new immutable spatial dispatch tag')
    output.mkdir(parents=True)
    selected=selected.sort_values(idcolumn)
    records=[]
    model=sp.compile_spatial()
    for row in selected.to_dict('records'):
        replication=int(row[idcolumn]);source=Path(row['output_dir'])
        run=output/'structural_bym2'/f'rep_{replication:05d}'
        run.mkdir(parents=True)
        record={'scenario':'sparse_psu_bym2','replication':replication,
            'arm':'HoggS2BYM2','output':str(run),'source_S1':str(source),
            'spatial_freeze_sha256':sha(frozen_path),'S1_refit':False}
        start=time.monotonic()
        try:
            source_status=json.loads((source/'status.json').read_text())
            for endpoint in ['anemia','stunting']:
                if not source_status['outcomes'][endpoint]['stage1']['success']:
                    raise ValueError('Source stage-one posterior failed fixed numerical diagnostics: '+endpoint)
            inputs=h.ROOT/f'resultados/resultados_comparadores/inputs/sparse_psu_bym2/rep_{replication:05d}/model_inputs'
            sources={k:source/f'stage2_modular_summary_{k}.npz'for k in ['anemia','stunting']}
            cfg=h.MCMCConfig(parallel_chains=1,phase='spatial_structural_sensitivity',seed=20261005+replication)
            result=sp.fit_spatial_pair(sources,inputs/'adjacency_edges.csv',output_dir=run,
                config=cfg,model=model,retry_on_diagnostics=True,retain_sampler_csv=False)
            record.update(completed=True,success=result['status']['success'],
                source_status_sha256=sha(source/'status.json'))
        except Exception as error:
            record.update(completed=False,success=False,error_type=type(error).__name__,
                error=str(error),traceback=traceback.format_exc())
            h.write_json(run/'execution_failure.json',record)
        record['seconds']=time.monotonic()-start
        h.write_json(run/'task_execution.json',record)
        records.append(record)
        pd.DataFrame([{k:v for k,v in r.items()if k!='traceback'}for r in records]).to_csv(output/'execution_status.csv',index=False)
        print(json.dumps({k:v for k,v in record.items()if k!='traceback'}),flush=True)
    pd.DataFrame([{k:v for k,v in r.items()if k not in ['traceback','error']}for r in records]).to_csv(output/'task_output_contract.csv',index=False)
    h.write_json(output/'batch_manifest.json',{'requested_tasks':args.stop-args.start,
        'population_IDs':list(range(args.start,args.stop)),'accounted_for':len(records),
        'completed_fits':sum(bool(r['completed'])for r in records),
        'converged':sum(bool(r['success'])for r in records),'source_plan_sha256':sha(source_plan),
        'spatial_freeze_sha256':sha(frozen_path),'primary_Holm_family':'Unchanged; this is a separate 100-world sensitivity.'})


if __name__=='__main__':main()
