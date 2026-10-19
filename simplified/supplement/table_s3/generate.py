"""Sparse57 sensitivity by district support: regenerate this artifact using external inputs.

See simplified/README.md for --data-root and --output-root.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from support.runtime import main
if __name__=='__main__':main(artifact_id='Supplement_Table_S3')
