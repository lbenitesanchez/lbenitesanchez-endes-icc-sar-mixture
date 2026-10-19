"""Seal a fully completed development choice before any fresh fit or score."""
from pathlib import Path
import datetime as dt,json
from comparison_simulation import ROOT,STUDY,CONTEXTS,EVALUATION_IDS,DEVELOPMENT_IDS,sha,save
from comparison_evaluate import METHODS,SAMPLER

def main():
    destination=ROOT/'protocol_freeze.json'
    if destination.exists():raise FileExistsError('The selection freeze is immutable')
    if any((ROOT/'evaluacion').rglob('completion.json')):raise RuntimeError('Fresh fits already exist')
    development=ROOT/'desarrollo/improve_development_v4_native_rho999'
    selection=development/'selected_recipe.json'
    if not selection.is_file():raise RuntimeError('Complete native development selection required')
    chosen=json.loads(selection.read_text())
    complete=json.loads((development/'completion.json').read_text())
    if not complete.get('complete') or complete.get('completed_worlds')!=32:raise RuntimeError('Incomplete development')
    for name in ['sikov_model_validation.json','sikov_bootstrap_validation.json','sikov_boundary_validation.json']:
        check=json.loads((ROOT/'validacion'/name).read_text())
        if not check.get('passed',check.get('pass_check',False)):raise RuntimeError('Native independent QA failed '+name)
    audits=[ROOT/'validacion'/f'improve_improve_development_v4_native_rho999{suffix}.json' for suffix in ['_audit','_native_mse_audit']]
    for path in audits:
        if json.loads(path.read_text())['status']!='PASS':raise RuntimeError('Native development audit failed')
    own=['comparison_simulation.py','comparison_evaluate.py','comparison_collect.py','comparison_real.py',
         'comparison_setup.py','comparison_freeze.py','improve_candidates.py','improve_select.py',
         'improve_development.py','improve_native_development.py','sikov_model.py','sikov_bootstrap.py',
         'sikov_bootstrap_sensitivity.py','align_real_training.py','align_external_score.py']
    sources=[ROOT/'scripts'/n for n in own]
    sources += [STUDY/'scripts/models.py',STUDY/'fase2_2026-10-07/scripts/diseno_calibration.py']
    sources += [STUDY/'scripts/competitors'/n for n in ['beta_power.py','beta_power_stable.py','beta_power_hmc.py',
        'beta_power_hybrid.stan','beta_power_stable_functions.stan','benchmark_metrics.py']]
    # Discovery/selection records enter the input seal, not the later test data.
    inputs=[ROOT/'simulation_design_plan.json',ROOT/'scripts/improve_candidates_protocol.json',
        ROOT/'datos/aux_sikov14_official.csv',ROOT/'datos/aux_sikov14_common_inputs.npz',
        ROOT/'datos/simulation_template/auxiliary.npz',ROOT/'datos/real_2019_full/fit_inputs.npz',
        ROOT/'datos/real_2019_full/private/units.csv',ROOT/'datos/real_2019_full/preparation_manifest.json',
        selection,development/'replications.csv',development/'development_plan.json',development/'completion.json',
        ROOT/'validacion/sikov_reconstruction_specification.json',
        ROOT/'validacion/sikov_model_validation.json',ROOT/'validacion/sikov_bootstrap_validation.json',ROOT/'validacion/sikov_boundary_validation.json',
        STUDY/'fase2_2026-10-07/datos/private_external/children_2024_training.csv',
        STUDY/'fase2_2026-10-07/datos/private_external/households_2024.csv']
    inputs += audits
    missing=[str(p) for p in sources+inputs if not p.is_file()]
    if missing:raise FileNotFoundError(missing)
    recipe={'path_anchor':str(STUDY.resolve()),'version':'matched_sikov2023_native_bootstrap_rho999_fixed_candidate_v1',
      'frozen_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
      'approved_for_fresh_fits':True,'approved_for_real_training':True,
      'selected_method':chosen['selected_method'],'selected_recipe':chosen['selected_recipe'],
      'selection_integration':'Laplace diagnostic development; unchanged posterior NUTS in fresh evaluation',
      'development_worlds':32,'development_ids':DEVELOPMENT_IDS,
      'contexts':CONTEXTS,'evaluation_ids':EVALUATION_IDS,'fresh_worlds':256,'methods':METHODS,
      'primary':'equal context, endpoint and area balanced IS95; selected versus FH/SAR, Holm2',
      'secondary':'RMSE, MAE, bias, coverage, width, all contexts, sampled/non-sampled and numerical flags',
      'bootstrap':{'primary_B':100,'primary_MSE':'naive parametric REML bootstrap',
        'BC':'same bootstrap draws; negative raw MSE retained, zero floor only for sqrt interval',
        'B300_subset':{'contexts':CONTEXTS,'replications':[30000,30001],'role':'fixed descriptive SAR sensitivity'},
        'minimum_bootstrap_success_fraction':.95},
      'shared_physical_projection':'secondary, clip mean and both95% interval ends to[0,1] for every method, no refit or selection',
      'rho_bound':.999,'rho_bound_reason':'numerical safeguard aligned with CRANsae; source paper gives invertibility, original optimizer unknown',
      'graph':'Euclidean longitude/latitude kNN6 within poverty strata; k4/k8/haversine fixed sensitivity',
      'sampler':SAMPLER,'gates':{'chains':4,'warmup':1000,'samples':1000,'retry_each':2000,
        'max_Rhat':1.01,'min_ESS_bulk_tail':400,'min_BFMI':.3,'divergences':0,'max_treedepth_hits':0,'beta_shape_guard_cells':0},
      'matched_information':'same 14 census variables, observed weighted children, PSU, direct y, stabilized D, geography and poverty',
      'endpoint_missingness':'endpoint-specific finite observed y,D and positive working D; full common 1874-area prediction frame',
      'external_2024':'previously assessed data reused as SECONDARY temporal and territorial validation; province-cluster Holm8 complete family',
      'no_guarantee_of_superiority':True,'exact_original_numerical_replication':False,
      'remaining_original_gaps':json.loads((ROOT/'validacion/sikov_reconstruction_specification.json').read_text())['nonidentifiable_original_details'],
      'fresh_truth_used_for_selection':False,'heldout_2024_responses_used_for_selection':False,
      'runtime_binary_sha256':{str((STUDY/'scripts/competitors/beta_power_hybrid').resolve()):sha(STUDY/'scripts/competitors/beta_power_hybrid')},
      'runtime_environment':{'python':'3.12','numpy':'2.5.4','scipy':'1.18.1','pandas':'3.0.6','cmdstan':'2.40.0','cmdstanpy':'1.3.0','max_parallel_fresh_workers':8},
      'statistical_sources_sha256':{str(p.resolve()):sha(p)for p in sources},
      'frozen_inputs_sha256':{str(p.resolve()):sha(p)for p in inputs}}
    save(destination,recipe)
    print(json.dumps({'frozen':str(destination),'sha256':sha(destination),'selected_method':recipe['selected_method'],'recipe':recipe['selected_recipe']},indent=2))
if __name__=='__main__':main()
