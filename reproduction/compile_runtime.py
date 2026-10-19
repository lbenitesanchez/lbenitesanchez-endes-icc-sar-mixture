#!/usr/bin/env python3
"""Compile unchanged Stan sources into an external runtime directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--cmdstan',type=Path,required=True)
    parser.add_argument('--runtime-root',type=Path,required=True)
    parser.add_argument('--sdkroot',type=Path)
    parser.add_argument('--deployment-target')
    args=parser.parse_args()
    runtime=args.runtime_root.resolve()
    source=args.source_root.resolve()/'scripts/competitors'
    if runtime.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError('Compiled files must be outside the public repository')
    runtime.mkdir(parents=True,exist_ok=True)
    identities={}
    for name in ('beta_power_hybrid.stan','beta_power_stable_functions.stan'):
        original=source/name
        shutil.copyfile(original,runtime/name)
        identities[name]=sha(original)
    options={}
    if args.sdkroot:
        os.environ['SDKROOT']=str(args.sdkroot.resolve())
    if args.deployment_target:
        os.environ['MACOSX_DEPLOYMENT_TARGET']=args.deployment_target
        options['CXXFLAGS']='-mmacosx-version-min='+args.deployment_target
    from cmdstanpy import CmdStanModel,set_cmdstan_path
    set_cmdstan_path(str(args.cmdstan.resolve()))
    started=time.time()
    model=CmdStanModel(stan_file=str(runtime/'beta_power_hybrid.stan'),force_compile=True,cpp_options=options)
    exe=Path(model.exe_file)
    receipt=dict(compiled_fresh=True,executable=str(exe),executable_sha256=sha(exe),
                 statistical_sources_sha256=identities,cmdstan=str(args.cmdstan.resolve()),
                 platform=platform.platform(),sdkroot=os.environ.get('SDKROOT'),
                 deployment_target=os.environ.get('MACOSX_DEPLOYMENT_TARGET'),cpp_options=options,
                 elapsed_seconds=time.time()-started,
                 compiler=subprocess.check_output(['c++','--version'],text=True),
                 operational_adapter_sha256=sha(Path(__file__)))
    (runtime/'compile_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    main()
