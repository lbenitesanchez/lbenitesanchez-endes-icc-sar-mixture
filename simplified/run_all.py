"""Regenerate all current tables and figures from external certified inputs."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from support.runtime import main
if __name__=='__main__':main()
