#!/usr/bin/env python3
"""Verify retained sampler evidence without reading or exporting truth values."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numerical_cohorts_portable as run

def audit():
    protocol=run.gate();run.modules();rows=[];pending=[]
    from beta_power import working_icc,cluster_effective_n,EndpointPosterior
    from improve_candidates import stratified_design
    for cohort in ('development_nuts','sparse57'):
        for _,context,rep in run.jobs_for(cohort):
            folder=run.PRIVATE/cohort/'fits'/context/f'rep_{rep:05d}'
            for method in (('ICC','ICC_poverty') if cohort=='development_nuts' else ('ICC',)):
                state=folder/(method+'_private_stan_state')
                if not (folder/(method+'_completion.json')).exists():
                    if state.exists():pending.append(dict(cohort=cohort,context=context,replication=rep,method=method,status='IN_PROGRESS_OR_FAILED_NO_COMPLETED_METHOD_CLAIM'))
                    continue
                session=run.read(state/'capture_session.json');status=run.read(folder/(method+'_status.json'))
                expected_samples=sum(len(x['sampler_trials']) for x in status['endpoints']);assert len(session['samples'])==expected_samples
                assert len(session['MAP_states'])==2
                chain_count=0;native_files=0
                inp=(run.PRIVATE/'sparse57/evaluacion/inputs' if cohort=='sparse57' else run.OLD/'sikov/desarrollo/inputs')/context/f'rep_{rep:05d}'
                data=dict(np.load(inp/'model_inputs/inputs.npz'));units=pd.read_csv(inp/'model_inputs/sample.csv')
                mf=run.read(inp/'manifest.json');assert run.sha(inp/'model_inputs/inputs.npz')==mf['fit_inputs_sha256'] and run.sha(inp/'model_inputs/sample.csv')==mf['sample_sha256']
                expected=[]
                for j,name in enumerate(('anemia','stunting')):
                    info=working_icc(units,name)[0];effective,_=cluster_effective_n(units,name,data['sample_idx'],data['n_eff'][:,j],info)
                    X=data['X'] if method=='ICC' else stratified_design(data['X'],data['strata'])
                    expected.append(EndpointPosterior(X,data['y'][:,j],effective,data['sample_idx'],data['basis']))
                for j,m in enumerate(session['MAP_states']):
                    p=state/m['directory'];assert run.sha(p/'MAP_capture.json')==m['receipt_sha256'];q=run.read(p/'MAP_capture.json');assert run.sha(p/'complete_MAP_state.npz')==q['state_sha256']
                    captured=dict(np.load(p/'complete_MAP_state.npz'))
                    for key in ('X','B','idx','y','n','k'):
                        assert np.array_equal(captured[key],getattr(expected[j],key)),(cohort,context,rep,method,j,key)
                for sample in session['samples']:
                    p=state/sample['directory'];after=run.read(p/'sample_after.json');before=run.read(p/'sample_before.json');assert sample['receipt_sha256']==run.sha(p/'sample_after.json')
                    assert after['status']=='NATIVE_SAMPLE_RETURNED' and len(after['CSV_sha256'])==4
                    args=before['requested_arguments'];assert args['chains']==4 and args['parallel_chains']==1 and args['metric']=='diag_e'
                    assert before['executable_sha256']==protocol['runtime_executable_sha256']
                    assert args['iter_warmup'] in (1000,2000) and args['iter_sampling']==args['iter_warmup'] and args['adapt_delta'] in (.95,.99)
                    assert len(after['captured_native_commands'])==4
                    for command in after['captured_native_commands']:
                        q=p/('process_'+str(command['process_index']));native=run.read(q/'native_command.json');assert run.sha(q/'native_command.json')==command['receipt_sha256']
                        for record in native['serialized_files'].values():assert run.sha(q/record['copy'])==record['sha256'];native_files+=1
                    output=Path(args['output_dir'])
                    for record in after['CSV_sha256']:assert run.sha(output/record['name'])==record['sha256'];chain_count+=1
                rows.append(dict(cohort=cohort,context=context,replication=rep,method=method,status='PASS_RETAINED_MAP_NATIVE_INPUTS_AND_CHAINS',endpoint_MAP_states=2,sampler_attempts=expected_samples,chains_per_attempt=4,retained_chain_files=chain_count,serialized_native_files=native_files,truth_read=False,private_values_exported=False,actual_post_inputs_equal_registered_arrays=True,model_optimization_repeated_for_audit=False))
    result=dict(format='iaos-new-run-retained-sampler-capture-audit-v1',audited_utc=run.utc(),completed_method_evidence_checks=len(rows),planned_method_fits=32*2+256,complete_all_planned_methods=len(rows)==320,retained_chain_files_verified=sum(x['retained_chain_files'] for x in rows),truth_read=False,private_values_exported=False,checked=rows,pending_state_folders=pending,protocol_sha256=run.sha(run.PROTOCOL))
    run.save(run.PUBLIC/'retained_sampler_capture_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('checked','pending_state_folders')}))
if __name__=='__main__':audit()
