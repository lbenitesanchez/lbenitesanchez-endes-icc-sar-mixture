"""Reserved2024 training-only CUT8 runner with separate state/collect actions."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
from pathlib import Path
import argparse
import datetime as dt
import json
import time
import numpy as np
import pandas as pd
from validacion_fit_models import ROOT, gate, sha, unit_dataset, assert_training_only, assert_prepared_training
from diseno_fit_cut import conditional_fit, pool_equal_weight, freeze_gate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--aux',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--state',choices=[str(i)for i in range(1,9)]+['all','collect'],required=True)
    args=p.parse_args()
    protocol_sha=gate(2024)
    freeze_gate()
    proof=json.loads((args.prepared/'preparation_manifest.json').read_text())
    if proof['year']!=2024 or proof['scope']!='territorial_training' or proof.get('reserved_test_outcomes_read',True):
        raise RuntimeError('PreparedCUT inputs differ from reserved training scope')
    areas=pd.read_csv(args.aux,dtype={'ubigeo':str})
    areas['ubigeo']=areas.ubigeo.str.zfill(6)
    units,_=unit_dataset(ROOT/'datos/private_external/children_2024_training.csv',areas)
    assert_training_only(units,2024,'children_2024_training.csv')
    inputs=dict(np.load(args.prepared/'fit_inputs.npz'))
    assert_prepared_training(inputs,units,areas,2024)
    ep=[np.load(args.prepared/f'{name}_calibration.npz')for name in('anemia','stunting')]
    nstates=[np.column_stack([v['mixture_effective_n'][k]for v in ep])for k in range(8)]
    args.output.mkdir(parents=True,exist_ok=True)
    wanted=range(8)if args.state=='all'else[]if args.state=='collect'else[int(args.state)-1]
    for k in wanted:
        conditional_fit(inputs,nstates[k],args.output/f'state_{k+1:02d}',
                        20261007+2024*100+10000+k*100003,
                        f'2024_territorial_training_cut8_state{k+1:02d}')
        print(json.dumps({'completed_state':k+1,'success':True}),flush=True)
    if args.state not in('all','collect'):
        return
    if(args.output/'completion.json').exists():
        raise FileExistsError('CompletedCUT prediction output must be preserved')
    for k in range(8):
        finished=args.output/f'state_{k+1:02d}/completion.json'
        if not finished.exists() or not json.loads(finished.read_text()).get('success',False):
            raise RuntimeError('Collect requires all eight completed successful states; it never dispatches a missing state')
    states=[];statuses=[]
    for k in range(8):
        # conditional_fit checks completed-target hash and posterior SHA and
        # never reruns a completed state, even after separate worker jobs.
        value,status=conditional_fit(inputs,nstates[k],args.output/f'state_{k+1:02d}',
                        20261007+2024*100+10000+k*100003,
                        f'2024_territorial_training_cut8_state{k+1:02d}')
        states.append(value);statuses.append(status)
    pooled=pool_equal_weight(states)
    pred=areas[['ubigeo']].copy()
    for j,endpoint in enumerate(('anemia','stunting')):
        for key in('p_mean','lower95','upper95','p_variance','within_state_variance','between_state_variance'):
            pred[key+'_'+endpoint]=pooled[key][:,j]
    pred.to_csv(args.output/'predictions.csv',index=False)
    report={'status':'PASS_TRAINING_ONLY_FIXED_WEIGHT_CUT8','success':True,'year':2024,
            'scope':'territorial_training','completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'protocol_sha256':protocol_sha,'source_sha256':sha(Path(__file__)),
            'preparation_sha256':sha(args.prepared/'preparation_manifest.json'),
            'prediction_sha256':sha(args.output/'predictions.csv'),
            'state_weights':[.125]*8,'conditional_states':8,'conditional_status':statuses,
            'reserved_test_outcomes_read':False,'normalizers_not_compared':True,
            'interpretation':'Variance-informed approximate modularposterior; fixedweight normalizedconditionalposteriors; no exactdesigncoverage claim',
            'balanced_pooled_draws_per_state':pooled['draws_per_state']}
    (args.output/'completion.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':report['status'],'prediction_sha256':report['prediction_sha256']}),flush=True)


if __name__=='__main__':
    main()
