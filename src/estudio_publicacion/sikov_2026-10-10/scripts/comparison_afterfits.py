"""Compatibility entry point for the complete additive joint-seal queue.

The historical original-only scheduler is superseded by the frozen joint
workflow. Even an accidental invocation must await all original and Published
fresh and actual predictions before any truth or held-out response is scored.
"""
from pathlib import Path
import os,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'

def main():
    subprocess.run([sys.executable,str(ROOT/'scripts/comparison_joint_close.py'),
                    '--wait','--score-fresh'],cwd=ROOT,check=True)
    print('PASS_COMPLETE_JOINT_LOCK_SCORING_AND_FIGURES',flush=True)

if __name__=='__main__':main()
