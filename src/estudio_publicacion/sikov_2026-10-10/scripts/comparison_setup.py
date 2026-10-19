"""Set up shared census-only simulation inputs and preserve earlier releases."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from comparison_simulation import ROOT,STUDY,NAMES,prepare_template,sha,save,CONTEXTS,DEVELOPMENT_IDS,EVALUATION_IDS

def main():
    if (ROOT/'protocol_freeze.json').exists() or (ROOT/'simulation_design_plan.json').exists():raise FileExistsError('Setup is immutable; use the existing template and plan')
    preserved=[]
    paths=list((STUDY/'scripts/competitors').glob('*.py'))+list((STUDY/'scripts/competitors').glob('*.stan'))+list((STUDY/'fase2_2026-10-07/scripts').glob('*.py'))+list((STUDY/'fase2_2026-10-07/manuscrito').glob('*.docx'))
    paths += [STUDY/'resultados/resultados_comparadores/protocol_freeze.json',STUDY/'fase2_2026-10-07/protocol_freeze.json',STUDY/'ENDES_mejoras_1_2_3_2026-10-07_revision_autores.zip']
    for p in paths:preserved.append({'path':str(p.relative_to(STUDY)),'sha256':sha(p)})
    save(ROOT/'validacion/previous_sources_and_release_baseline.json',preserved)
    f=ROOT/'datos/aux_sikov14_official.csv';d=pd.read_csv(f,dtype={'ubigeo':str});d.insert(0,'intercept',1.)
    out=ROOT/'datos/census_only_full_template.npz'
    np.savez_compressed(out,X_raw=d[NAMES].to_numpy(float),strata=d.poverty_stratum.to_numpy(int),coords=d[['longitude','latitude']].to_numpy(float),covariate_names=np.asarray(NAMES))
    prepare_template(out,f)
    save(ROOT/'simulation_design_plan.json',{'phase':'design_before_development','contexts':CONTEXTS,'development_ids':DEVELOPMENT_IDS,'evaluation_ids':EVALUATION_IDS,'new_independent_worlds':256,'prior_800_worlds_reused_for_tuning':False,'prior_phase2_40_worlds_reused_for_tuning':False,'point_criteria':['RMSE','MAE','bias'],'interval_criteria':['IS95','coverage95','width'],'primary':'Balanced interval score95, equal context, outcome and district weights','multiplicity':'Holm two contrasts selected candidate vs Sikov FH and Sikov SAR; paired world differences, all complete worlds required','MC_intervals':'paired world Monte Carlo 95% normal intervals, not population CIs','conditional_gaussian_intervals':'Primary untruncated Gaussian intervals from naive parametric REML bootstrap MSE B100; BC same draws and B300 fixed subset descriptive sensitivities','model_gaps':['Unknown original W and original poverty vector','Outcome definition and direct point estimator ambiguities','Published variable selection procedure and CI method absent'],'external_2024':'Previously evaluated data reused as secondary assessment only; never newly untouched validation','validation_integrity':'Fresh 30000..30063 streams per context, selected recipe + kernels hashed before final fits; outcome scoring after predictions sealed','same_information':'14 census2017 variables, observed children, PSU, original weights, direct y, geography and stabilized sampling variances available to every estimator','native_scale_sensitivity':'Original Gaussian point predictions and normal intervals evaluated without clipping; physical bound violations reported'})
    print('Template prepared from census only; no survey outcomes used.')
if __name__=='__main__':main()
