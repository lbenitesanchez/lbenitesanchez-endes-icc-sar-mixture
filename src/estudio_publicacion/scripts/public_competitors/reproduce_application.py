"""Optional private ENDES application in a new sibling workspace.

Initial FH comparison measurements/folds are reused as frozen references. The
Both years are refitted completely with the approved v2 evaluator, using the
declared application-only diagonal metric. The plan includes all manuscript
application sensitivities; every command is explicit before execution.
"""
from pathlib import Path
import argparse,json,shutil,subprocess,sys
import reproduce

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--include-age-sex',action='store_true',
        help='Retained compatibility option; the complete manuscript recipe always includes age/sex and spatial sensitivities')
    p.add_argument('--extra-predictions-json',type=Path)
    a=p.parse_args();target=a.directory.resolve()
    saved=sys.argv;sys.argv=['reproduce.py','--directory',str(target)]
    try:reproduce.main()
    finally:sys.argv=saved
    for year in [2019,2025]:
        folder=ROOT/f'resultados/aplicacion_{year}_who2024'
        for name in ['validation_folds.csv','locked_test_area_scores.csv']:
            dst=target/f'resultados/aplicacion_{year}_who2024'/name
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(folder/name,dst)
    py=target/'.competitor_env/bin/python';base=target/'scripts/competitors'
    historical=Path('resultados/resultados_comparadores/application_sensitivity_v1/declared_plan.json')
    (target/historical).parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/historical,target/historical)
    budget=target/'resultados/resultados_comparadores/application_sensitivity_v2/application_sensitivity_budget.json'
    budget.parent.mkdir(parents=True,exist_ok=True);budget.write_text(json.dumps({'active_workers':1}))
    commands=[
        [str(py),str(target/'codigo/download_official_data.py'),'--extract'],
        ['Rscript',str(target/'codigo/01_prepare_data.R')],
        ['Rscript',str(target/'scripts/11_export_unit_inputs.R')],
        [str(py),str(base/'beta_power_application_v2.py'),'--year','2019','--tag','application_final_candidate_v2'],
        [str(py),str(base/'beta_power_application_v2.py'),'--year','2025','--tag','application_final_candidate_v2'],
        [str(py),str(base/'gao_prepare_endes.py')],
    ]
    for year in [2019,2025]:
      for split in ['full','locked_train']:
       for mode in ['author_strict','common_continuity']:
        commands.append(['Rscript',str(base/'gao_run_endes.R'),
            str(target/f'resultados/competidores_gao/inputs/endes_{year}_{split}'),
            str(target/f'resultados/competidores_gao/endes_{year}_{mode}_{split}'),mode])
    commands.append([str(py),str(base/'hogg_application.py'),'--parallel-chains','1'])
    commands.append([str(py),str(base/'hogg_strong_application.py'),'--parallel-chains','1'])
    commands.append([str(py),*reproduce.spatial_driver(target,base,'application'),'--modes','full','locked_train'])
    commands.append([str(py),str(base/'beta_power_application_sensitivities.py')])
    commands.append([str(py),str(base/'beta_power_sensitivity_scores.py')])
    extra=target/'extra_application_predictions.json'
    extra.write_text(json.dumps([{'year':2025,'method':'HoggAgeSexSpatial',
        'path':'resultados/competidores_hogg/endes_2025_locked_train_age_sex_s2_bym2/predictions.csv',
        'area_key':'ubigeo','mean_prefix':'p_mean_',
        'information_arm':'survey_age_sex_extra_spatial_S2'}],indent=2))
    comparison=[str(py),str(target/'scripts/11_application_comparison.py'),
                '--candidate-tag','application_final_candidate_v2','--include-age-sex',
                '--extra-predictions-json',str(extra)]
    if a.extra_predictions_json:
        supplied=json.loads(a.extra_predictions_json.read_text())
        extra.write_text(json.dumps(json.loads(extra.read_text())+supplied,indent=2))
    commands.append(comparison)
    (target/'application_reproduction_commands.json').write_text(json.dumps(commands,indent=2))
    print('Private application plan prepared; no survey units copied from the public package.')
    if a.execute:
        import os
        env=os.environ.copy();env['ENDES_EXPORT_PRIVATE_INPUTS']='1'
        for command in commands:subprocess.run(command,cwd=target,env=env,check=True)

if __name__=='__main__':main()
