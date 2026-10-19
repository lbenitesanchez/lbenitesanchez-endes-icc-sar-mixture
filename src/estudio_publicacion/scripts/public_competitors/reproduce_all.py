"""Prepare a new whole-manuscript reproduction; execution remains explicit.

Both phase plans run in the same fresh workspace. No old documents, final
predictions, private units, traces or fitted posteriors are copied. Documents
remain drafts for author confirmation and require separate visual inspection.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys
import reproduce_application

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();target=args.directory.resolve()
    old=sys.argv;sys.argv=['reproduce_application.py','--directory',str(target)]
    try:reproduce_application.main()
    finally:sys.argv=old
    commands=json.loads((target/'reproduction_commands.json').read_text())
    commands+=json.loads((target/'application_reproduction_commands.json').read_text())
    python=target/'.competitor_env/bin/python'
    commands += [[str(python),str(target/'scripts'/name)]
                 for name in ['13_competitor_figures.py','12_competitor_documents.py']]
    (target/'complete_reproduction_commands.json').write_text(json.dumps(commands,indent=2))
    print('Complete fresh-workspace plan saved; final tables, figures and five documents require every declared gate.')
    if args.execute:
        env=os.environ.copy();env['ENDES_EXPORT_PRIVATE_INPUTS']='1'
        for command in commands:subprocess.run(command,cwd=target,env=env,check=True)

if __name__=='__main__':main()
