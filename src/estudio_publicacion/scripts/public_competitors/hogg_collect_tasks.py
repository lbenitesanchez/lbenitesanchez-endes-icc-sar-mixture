"""Translate a completed Hogg batch contract for the frozen evaluator."""
from pathlib import Path
import argparse
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('input_csv',type=Path);p.add_argument('output_csv',type=Path)
    a=p.parse_args();d=pd.read_csv(a.input_csv)
    aliases={'primary':'HoggTSLN','finite_plusH':'HoggTSLNFullH','design_PSU':'HoggDesignPsiPSU',
             'HoggS2BYM2':'HoggSpatial'}
    d['method']=d.arm.map(aliases)
    if d.method.isna().any():raise ValueError('Unknown frozen Hogg arm')
    d['protocol']=d.arm.map({'primary':'frozen_independent_v1','finite_plusH':'extra_population_auxiliary_H_v1','design_PSU':'experimental_design_PSU_v1',
                           'HoggS2BYM2':'author_D2_spatial_sensitivity_v1'})
    d['output_dir']=d.output;d['diagnostic_style']='hogg'
    d[['scenario','replication','method','protocol','output_dir','diagnostic_style']].to_csv(a.output_csv,index=False)

if __name__=='__main__':main()
