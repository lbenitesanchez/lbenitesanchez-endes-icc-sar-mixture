#!/usr/bin/env python3
"""Future-run private Stan state recorder. It never changes sampler arguments.

Use around ONE fit-worker entry point, not a dispatcher which starts unwrapped
children. Tested contract: CmdStanPy 1.3.0 and the archived EndpointPosterior API.
Serialized inputs are copied immediately before the existing native command.
No output from an earlier execution is retroactively declared captured here.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import runpy
import sys
import threading
import uuid
import numpy as np

FORMAT='iaos-private-stan-state-capture-v1'
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def encode(value):
    if isinstance(value,np.ndarray):return dict(array=True,shape=list(value.shape),dtype=value.dtype.str,
        sha256_C_bytes=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):encode(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [encode(v) for v in value]
    if isinstance(value,Path):return str(value)
    if isinstance(value,(str,int,float,bool))or value is None:return value
    return dict(type=type(value).__qualname__)
def private_dir(path):
    path=Path(path).resolve();path.mkdir(parents=True,exist_ok=False,mode=0o700);os.chmod(path,0o700);return path
def save(path,value):
    data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb')as stream:stream.write(data)
def copy_exact(source,target):
    source=Path(source)
    if not source.is_file():raise FileNotFoundError('Serialized native input must exist before dispatch: '+str(source))
    blob=source.read_bytes();fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb')as stream:stream.write(blob)
    if sha(source)!=sha(target):raise RuntimeError('Native serialized input changed while being captured')
    return dict(source=str(source),copy=target.name,sha256=sha(target),bytes=len(blob))
def native_inputs(command):
    inputs={};section=None
    for token in command[1:]:
        if '=' not in token:section=token;continue
        key,value=token.split('=',1)
        if key=='file'and section=='data':inputs['data']=value
        elif key=='init':
            if Path(value).is_file():inputs['initial']=value
            else:
                try:float(value)
                except ValueError:raise FileNotFoundError('Explicit initialization file is absent')
        elif key=='metric_file':inputs['metric']=value
    if 'data'not in inputs:raise RuntimeError('This recorder requires explicit serialized Stan data')
    return inputs

class Recorder:
    def __init__(self,root,model_class,posterior_class,version):
        if version!='1.3.0':raise RuntimeError('Unreviewed CmdStanPy API version: '+version)
        self.root=private_dir(root);self.model_class=model_class;self.posterior_class=posterior_class
        self.lock=threading.Lock();self.active={};self.latest_map=threading.local();self.samples=[];self.maps=[]
        self.original_sample=model_class.sample;self.original_run=model_class._run_cmdstan;self.original_optimize=posterior_class.optimize

    def install(self):
        recorder=self
        def optimize(post,*args,**kwargs):
            result=recorder.original_optimize(post,*args,**kwargs)
            folder=private_dir(recorder.root/('MAP_'+uuid.uuid4().hex))
            arrays={key:np.asarray(getattr(post,key)) for key in ('X','B','idx','y','n','k','mode','cov')}
            path=folder/'complete_MAP_state.npz'
            with path.open('xb')as stream:np.savez_compressed(stream,**arrays)
            os.chmod(path,0o600)
            record=dict(format=FORMAT,role='NEW_FUTURE_RUN_MAP_STATE',created_utc=utc(),state_sha256=sha(path),
                numerical_status=encode(post.status),config=encode(post.config),array_schema=encode(arrays),
                source_optimize_sha256=hashlib.sha256(inspect.getsource(recorder.original_optimize).encode()).hexdigest(),
                source_module_sha256=sha(Path(inspect.getfile(recorder.original_optimize))))
            save(folder/'MAP_capture.json',record)
            recorder.latest_map.record=(folder,arrays['mode'].copy())
            with recorder.lock:recorder.maps.append(dict(directory=folder.name,receipt_sha256=sha(folder/'MAP_capture.json')))
            return result
        def sample(model,*args,**kwargs):
            bound=inspect.signature(recorder.original_sample).bind(model,*args,**kwargs)
            invocation={k:encode(v) for k,v in bound.arguments.items()if k!='self'}
            folder=private_dir(recorder.root/('sample_'+uuid.uuid4().hex));map_binding=None
            pending=getattr(recorder.latest_map,'record',None)
            initial=bound.arguments.get('inits')
            if pending is not None and isinstance(initial,dict)and set(initial)>={'beta','gamma','log_kappa'}:
                vector=np.r_[np.asarray(initial['beta']),np.asarray(initial['gamma']),initial['log_kappa']]
                if not np.array_equal(vector,pending[1]):raise RuntimeError('Sample init does not match the captured MAP mode')
                map_binding=dict(directory=pending[0].name,receipt_sha256=sha(pending[0]/'MAP_capture.json'))
            context=dict(directory=folder,commands=[],MAP_binding=map_binding)
            with recorder.lock:
                if id(model)in recorder.active:raise RuntimeError('Concurrent sample calls on one model object are outside the reviewed capture contract')
                recorder.active[id(model)]=context
            executable=Path(model.exe_file)
            outcome=None
            try:
                save(folder/'sample_before.json',dict(format=FORMAT,created_utc=utc(),role='NEW_FUTURE_RUN_NATIVE_INVOCATION',
                    requested_arguments=invocation,MAP_binding=map_binding,executable_sha256=sha(executable),
                    existing_sampler_arguments_changed=False,source_adapter_sha256=sha(Path(__file__)),
                    cmdstanpy_version='1.3.0',numpy_version=np.__version__,python_version=sys.version,platform=platform.platform(),
                    thread_environment={k:os.environ.get(k)for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','STAN_NUM_THREADS']}))
                result=recorder.original_sample(model,*args,**kwargs)
                if len(context['commands'])!=result.runset.num_procs:
                    raise RuntimeError('Not every native command was captured before dispatch')
                files=list(result.runset.csv_files)
                outcome=dict(status='NATIVE_SAMPLE_RETURNED',finished_utc=utc(),CSV_sha256=[dict(name=Path(p).name,sha256=sha(p))for p in files],
                    captured_native_commands=context['commands'],MAP_binding=map_binding,
                    convergence_or_article_agreement_certified_by_recorder=False)
                return result
            except BaseException as exc:
                outcome=dict(status='NATIVE_SAMPLE_FAILED_OR_CAPTURE_FAILED',finished_utc=utc(),exception_type=type(exc).__name__,
                    captured_native_commands=context['commands'],MAP_binding=map_binding,convergence_or_article_agreement_certified_by_recorder=False)
                raise
            finally:
                with recorder.lock:recorder.active.pop(id(model),None)
                if outcome is not None:
                    save(folder/'sample_after.json',outcome)
                    with recorder.lock:recorder.samples.append(dict(directory=folder.name,status=outcome['status'],receipt_sha256=sha(folder/'sample_after.json')))
        def run(model,*args,**kwargs):
            bound=inspect.signature(recorder.original_run).bind(model,*args,**kwargs)
            runset=bound.arguments['runset'];idx=bound.arguments['idx']
            with recorder.lock:context=recorder.active.get(id(model))
            if context is None:raise RuntimeError('Native process has no bound captured sample invocation')
            command=runset.cmd(idx);folder=private_dir(context['directory']/('process_'+str(idx)))
            copies={role:copy_exact(path,folder/(role+Path(path).suffix))for role,path in native_inputs(command).items()}
            record=dict(format=FORMAT,created_before_native_dispatch_utc=utc(),process_index=idx,command=command,
                serialized_files=copies,command_arguments_changed=False,executable_sha256=sha(Path(command[0])))
            save(folder/'native_command.json',record)
            with recorder.lock:context['commands'].append(dict(process_index=idx,receipt_sha256=sha(folder/'native_command.json')))
            return recorder.original_run(model,*args,**kwargs)
        self.model_class.sample=sample;self.model_class._run_cmdstan=run;self.posterior_class.optimize=optimize
        return self

    def uninstall(self):
        self.model_class.sample=self.original_sample;self.model_class._run_cmdstan=self.original_run;self.posterior_class.optimize=self.original_optimize
    def finish(self,status):
        save(self.root/'capture_session.json',dict(format=FORMAT,status=status,finished_utc=utc(),samples=self.samples,MAP_states=self.maps,
            future_run_only=True,statistical_sources_changed=False,sampler_arguments_changed=False,
            recorder_source_sha256=sha(Path(__file__)),privacy='PRIVATE: NEVER EXPORT INPUT JSON, MAP STATE, CHAINS OR RAW TRAINING VALUES'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-root',type=Path,required=True);parser.add_argument('--public-repository',type=Path,required=True)
    parser.add_argument('--competitors-root',type=Path,required=True);parser.add_argument('--script',type=Path,required=True)
    parser.add_argument('script_arguments',nargs=argparse.REMAINDER);a=parser.parse_args()
    root=a.private_root.resolve();public=a.public_repository.resolve()
    if root==public or root.is_relative_to(public):raise ValueError('Sampler state must be outside the public repository')
    sys.path.insert(0,str(a.competitors_root.resolve()));sys.path.insert(0,str(a.script.resolve().parent))
    import cmdstanpy
    from beta_power import EndpointPosterior
    recorder=Recorder(root,cmdstanpy.CmdStanModel,EndpointPosterior,cmdstanpy.__version__).install()
    arguments=a.script_arguments[1:]if a.script_arguments[:1]==['--']else a.script_arguments
    previous=sys.argv;sys.argv=[str(a.script.resolve()),*arguments];status='WORKER_RETURNED_NO_STATISTICAL_CERTIFICATION'
    try:
        try:runpy.run_path(str(a.script.resolve()),run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None,0):raise
        if not recorder.samples:raise RuntimeError('No sample invocation observed; dispatchers or unwrapped child processes are not supported')
        if any(r['status']!='NATIVE_SAMPLE_RETURNED'for r in recorder.samples):raise RuntimeError('An observed sample failed; partial state is retained')
    except BaseException:
        status='WORKER_FAILED_OR_NO_SAMPLE_OBSERVED';raise
    finally:
        sys.argv=previous;recorder.uninstall();recorder.finish(status)

if __name__=='__main__':main()
