"""Common phase2 auxiliary matrix and training-only design preparation."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
from validacion_fit_models import ROOT, STUDY, gate, sha, unit_dataset, assert_training_only
from diseno_calibration import CONFIG, frame_from_households, stratified_psu_counts, calibrate, write_summary
sys.path.insert(0, str(STUDY / 'scripts'))
from models import spatial_basis


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--year', type=int, choices=[2019,2024], required=True)
    p.add_argument('--aux', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pre-freeze-development-prepare', action='store_true')
    args = p.parse_args()
    if args.pre_freeze_development_prepare and args.year == 2024:
        raise RuntimeError('Reserved2024 preparation forbidden before response unlock')
    protocol_sha = None if args.pre_freeze_development_prepare else gate(args.year)
    household_path = (STUDY / 'datos/private_competitors/households_2019.csv' if args.year == 2019
                      else ROOT / 'datos/private_external/households_2024.csv')
    child_path = (STUDY / 'datos/private_competitors/children_2019.csv' if args.year == 2019
                  else ROOT / 'datos/private_external/children_2024_training.csv')
    if args.output.exists():
        raise FileExistsError(args.output)
    areas = pd.read_csv(args.aux, dtype={'ubigeo':str, 'prov_code':str, 'dept_code':str})
    areas['ubigeo'] = areas.ubigeo.str.zfill(6)
    areas['prov_code'], areas['dept_code'] = areas.ubigeo.str[:4], areas.ubigeo.str[:2]
    features = ['altitude','poverty','illiteracy','electricity','spanish','biomass']
    if areas.ubigeo.duplicated().any() or not np.isfinite(areas[features+['latitude','longitude']].to_numpy(float)).all():
        raise ValueError('Common auxiliary frame must have distinct codes and complete finite frozenX')
    units, excluded = unit_dataset(child_path, areas)
    assert_training_only(units, args.year, child_path)
    household = pd.read_csv(household_path, dtype={'ubigeo':str, 'psu_key':str})
    frame = frame_from_households(household)
    valid = (np.isfinite(units.y_anemia) | np.isfinite(units.y_stunting)) & units.weight.gt(0)
    idx = np.sort(units.loc[valid,'area'].unique())
    raw = areas[features].to_numpy(float)
    center, scale = raw.mean(axis=0), raw.std(axis=0)
    X = np.column_stack([np.ones(len(raw)), (raw-center)/scale])
    basis = spatial_basis(areas[['latitude','longitude']].to_numpy(float), rank=40)
    counts, replicate_checks = stratified_psu_counts(frame)
    result = [calibrate(frame, units, idx, endpoint, counts=counts) for endpoint in ('anemia','stunting')]
    y = np.column_stack([v['direct']['y'] for v in result])
    kish = np.column_stack([v['direct']['n_kish'] for v in result])
    D = np.zeros((len(idx),2,2))
    for j in range(2):
        D[:,j,j] = result[j]['direct']['variance']
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'fit_inputs.npz', X=X,basis=basis,sample_idx=idx,y=y,kish_n=kish,D=D,
                        test_idx=np.array([],dtype=int))
    areas[['ubigeo','prov_code','dept_code']].to_csv(args.output/'area_ids.csv',index=False)
    for endpoint, value in zip(('anemia','stunting'),result):
        value['status'].update(year=args.year,scope='full' if args.year==2019 else'territorial_training',
                               heldout_training_overlap=0,replicate_checks=replicate_checks)
        write_summary(value,args.output,endpoint)
    proof = {'status':'PASS_COMMON_AUX_TRAINING_ONLY_DESIGN_PREPARATION',
             'year':args.year,'scope':'full' if args.year==2019 else'territorial_training',
             'config':CONFIG,'protocol_sha256':protocol_sha,'source_sha256':sha(Path(__file__)),
             'input_sha256':{'households':sha(household_path),'children':sha(child_path),'aux':sha(args.aux)},
             'full_household_design_used':True,'full_households':len(household),'fullPSUs':len(frame),
             'training_domains':len(idx),'auxiliary_domains':len(areas),
             'reserved_2024_read':args.year==2024,'reserved_test_outcomes_read':False,
             'training_test_overlap':0,'X_center':center.tolist(),'X_scale':scale.tolist(),
             'response_dependent_covariate_screening':False,
             'excluded_no_auxiliary_records_by_domain':excluded,
             'D_crosscovariance_scope':'Only endpointmarginaldiagonal stored, no jointoutcome model fitted',
             'model_fits_dispatched':False,'outcome_scores_computed':False,
             'output_sha256':{p.name:sha(p)for p in args.output.iterdir()if p.is_file()}}
    (args.output/'preparation_manifest.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps({k:proof[k]for k in ['status','year','training_domains','auxiliary_domains','reserved_test_outcomes_read']}),flush=True)


if __name__ == '__main__':
    main()
