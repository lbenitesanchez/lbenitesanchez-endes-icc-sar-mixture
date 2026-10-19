"""Operational two-worker dispatcher for the already frozen forty worlds."""
from pathlib import Path
import argparse,datetime as dt,json,os,traceback
import sim_design_diagnostics as sim

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,choices=[0,1],required=True)
    args=p.parse_args();freeze=sim.gate()
    trace=sim.PHASE/f'validacion/root_simulation_worker_{args.worker}.jsonl'
    def event(value):
        value.update(time_utc=dt.datetime.now(dt.timezone.utc).isoformat(),worker=args.worker,
                     pid=os.getpid(),protocol_sha256=freeze)
        with trace.open('a') as f: f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    jobs=[(context,rep)for context in sim.CONTEXTS for rep in range(sim.POPULATIONS_PER_CONTEXT)]
    event({'event':'worker_start','assigned_worlds':20,'algorithm_source_sha256':sim.sha(sim.__file__)})
    for index,(context,rep)in enumerate(jobs):
        if index%2!=args.worker:continue
        event({'event':'world_start','context':context,'rep':rep})
        try:
            sim.generate_world(context,rep)
            sim.fit_world(context,rep)
        except Exception as exc:
            out=sim.PHASE/f'resultados/design_simulation/{context}/rep_{rep:03d}'
            out.mkdir(parents=True,exist_ok=True)
            error={'context':context,'rep':rep,'status':'UNAVAILABLE_WORLD_RETAINED_NO_SEED_REPLACEMENT',
                   'error_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
                   'protocol_sha256':freeze,'reserved2024response_access':False}
            failure=out/'operational_failure.json'
            if failure.exists():raise FileExistsError('Existing failure must not be overwritten')
            failure.write_text(json.dumps(error,indent=2)+'\n')
            event({'event':'world_unavailable','context':context,'rep':rep,'error':str(exc)})
            continue
        event({'event':'world_complete','context':context,'rep':rep})
    event({'event':'worker_finished','no_seed_replacement':True})

if __name__=='__main__':main()
