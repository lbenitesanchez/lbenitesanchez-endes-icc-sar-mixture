"""Verify source identities and external input files without generating outputs."""
from pathlib import Path
import argparse
import json
import os
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from support.runtime import verify_inputs
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=os.environ.get('IAOS_SIMPLE_DATA_ROOT'),
                        help='External input bundle containing datos/ and referencias/; or IAOS_SIMPLE_DATA_ROOT')
    args=parser.parse_args()
    if args.data_root is None:parser.error('Provide --data-root or IAOS_SIMPLE_DATA_ROOT')
    try:result=verify_inputs(args.data_root)
    except (ValueError,RuntimeError,OSError)as error:parser.exit(1,'Error: '+str(error)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
