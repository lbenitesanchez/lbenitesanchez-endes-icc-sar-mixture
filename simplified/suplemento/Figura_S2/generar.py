"""Mapa descriptivo distrital: entrada individual con insumos y salidas externos.

Ver simplified/README.md para las opciones --data-root y --output-root.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from soporte.runtime import main
if __name__=='__main__':main(object_id='Suplemento_Figura_S2')
