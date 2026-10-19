"""Build an allow-listed author-review archive after complete scientific QA."""
from pathlib import Path
import argparse,datetime as dt,hashlib,json,shutil,zipfile

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT.parent
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SAFE_SUFFIXES={'.py','.r','.stan','.csv','.json','.npz','.png','.pdf','.docx','.md','.txt'}
PRIVATE_SUFFIXES={'.sav','.rds','.dta','.xlsx','.xls','.sqlite','.db','.exe','.so','.dylib','.dll','.pyc'}
PRIVATE_NAMES=('units','children','households','municipal','directory','credential','secret','token',
               'conditional_draw','beta_draw','mcmc','sampler_input','chain')

# These are names, not glob patterns. New artifacts require an explicit review
# and an edit here before they can enter the public archive.
PUBLIC_HELPERS=(
    'align_aux_sikov.py','align_build_inputs.py','align_census_queries.py',
    'align_real_training.py','align_survey_2019.R','align_validate.py',
    'comparison_setup.py','comparison_freeze.py','comparison_afterfits.py',
    'comparison_joint_close.py','comparison_figures.py','comparison_application_figures.py',
    'improve_native_development.py','sikov_after_prediction_lock.py',
    'sikov_validate.py','sikov_bootstrap_validate.py','sikov_bootstrap_sensitivity.py',
    'sikov_boundary_qa.py','sikov_full_graph_scope_audit.py',
    'sikov_published_freeze.py','sikov_published_validate.py','sikov_published_contract_qa.py',
    'sikov_published_collect_qa.py','align_external_score_qa.py','align_external_extended_score_qa.py',
    'align_training_independent_audit.py','align_external_independent_audit.py',
    'public_package.py','public_verify.py','public_replay.py','report_documents.py',
    'sikov_poverty_stratum_descriptive.py',
    'sikov_poverty_stratum_descriptive_v1_invalid_labels.py',
    'sikov_published_b300_sensitivity.py',
    'sensitivity_sampled_graph_generate.py','sensitivity_sampled_graph_fit.py',
    'sensitivity_sampled_graph_audit.py','sensitivity_sampled_graph_score.py','sensitivity_sampled_graph_qa.py',
    'improve_candidates_protocol.json')
PUBLIC_CERTIFICATES=(
    'align_2024_inventory_and_existing_locks.json','align_aux_sikov_manifest.json',
    'align_external_extended_synthetic_qa.json','align_external_independent_auditor_synthetic_qa.json',
    'align_external_score_synthetic_qa.json','align_final_validation.json','align_inputs_manifest.json',
    'align_original_directRDS_criterion_counts.json','align_original_input_inventory.json',
    'align_poverty2009_official_audit.json','align_poverty2013_official_audit.json',
    'align_poverty_vintage_source_provenance.json','align_real2019_preparation_validation.json',
    'align_real2019_training_independent_audit.json','align_real2024_training_independent_audit.json',
    'align_source_reproduction_conclusion.json','align_survey2019_R_sessionInfo.txt',
    'align_survey2019_independent_survey_checks.csv','comparison_application_figures_preparation_QA.json',
    'improve_development_source_correction.json','improve_native_rho_correction.json',
    'improve_numerical_qa.json','improve_improve_development_v4_native_rho999_audit.json',
    'improve_improve_development_v4_native_rho999_native_mse_audit.json',
    'previous_sources_and_release_baseline.json','report_helpers_preparation_status.json',
    'report_preparation_status.json','sikov_baseline_preservation_QA.json',
    'sikov_bootstrap_validation.json','sikov_boundary_validation.json',
    'sikov_development_input_covariance_audit.json','sikov_development_optimizer_diagnostic.json',
    'sikov_final_pilot_prediction_QA.json','sikov_full_graph_scope_audit.json',
    'sikov_independent_dense_development_fixture.json','sikov_independent_external_oracle_QA.json',
    'sikov_model_validation.json','sikov_pre_freeze_paired_QA.json',
    'sikov_published_collect_validation.json','sikov_published_contract_validation.json',
    'sikov_published_reference_validation.json','sikov_reconstruction_specification.json')
BASE_RESULTS=('fresh_replications.csv','fresh_summary.csv','fresh_balanced_paired_contrasts.csv',
    'fresh_context_endpoint_paired_contrasts.csv','fresh_physical_projection_replications.csv',
    'fresh_physical_projection_summary.csv','fresh_physical_projection_contrasts.csv')
ADDED_RESULTS=('per_replication_scores.csv','summary.csv','paired_balanced_contrasts_Holm3.csv',
    'physical_projection_scores.csv','physical_projection_summary.csv','physical_projection_contrasts.csv')
EXTERNAL_SCOPE_RESULTS=('district_scores.csv','endpoint_summary.csv',
    'paired_balanced_contrasts_unadjusted.csv','excluded_domains.csv','summary.json')

def public_logical_path(path,study):
    """Preserve an allowed logical identity and reject symlinks below its root."""
    base=Path(study).absolute();path=Path(path).absolute()
    if '..'in path.parts:raise ValueError('Unnormalized public path')
    relative=path.relative_to(base)
    cursor=base
    if cursor.is_symlink():raise ValueError('Public root is a symbolic link')
    for part in relative.parts:
        cursor=cursor/part
        if cursor.is_symlink():raise ValueError('Public path contains a symbolic link: '+str(cursor))
    return path

def expected_public_paths(proof,addendum,root=None,study=None):
    """Exact public inventory; enumerate the fixed cohort without directory scans.

    Frozen metadata contributes only explicitly sealed source/certificate names.
    Optional entries are individually named historical preparation certificates.
    """
    root=Path(root or ROOT).absolute();study=Path(study or STUDY).absolute()
    required=set();optional=set()
    def add(name,optional_entry=False):
        path=(root/name).absolute()
        if not path.is_relative_to(study):raise ValueError('Public path escapes study')
        (optional if optional_entry else required).add(path)
    for document,fields in ((proof,('statistical_sources_sha256',)),
                            (addendum,('additional_sources_sha256','frozen_inputs_sha256'))):
        for field in fields:
            for name in document[field]:
                relative=Path(name).relative_to(Path(proof['path_anchor']))
                path=(study/relative).absolute()
                if not path.is_relative_to(study):raise ValueError('Frozen source escapes study')
                required.add(path)
    for name in PUBLIC_HELPERS:add('scripts/'+name)
    for name in PUBLIC_CERTIFICATES:add('validacion/'+name,True)
    for name in ('protocol_freeze.json','simulation_design_plan.json'):
        add(name,optional_entry=name=='simulation_design_plan.json')
    for name in ('literature_update_2026-10-10.json','author_source_archive_inventory_2026-10-10.json'):
        add('fuentes/'+name)
    for name in ('journal_editorial_check_2026-10-10.json','mixture_method_reference_2026-10-10.json'):
        add('fuentes/'+name)
    add('fuentes/historical_endpoint_and_survey_reference_2026-10-10.json')
    add('fuentes/solicitud_insumos_originales_Sikov.txt')
    for name in ('sikov_article_source_metadata.json','sikov_code_search.json'):
        add('fuentes/'+name,True)
    for name in ('sikov_published_addendum_freeze.json','joint_prediction_lock.json',
                 'fresh_prediction_lock.json','sikov_published_fresh_prediction_lock.json',
                 'fresh_evaluation_status.json','sikov_published_evaluation_status.json',
                 'sikov_final_independent_audit.json','align_external_extended_independent_audit.json',
                 'report_visual_QA.json','report_build_status.json','comparison_figures.json','comparison_application_figures.json',
                 'sikov_B300_descriptive_scores.csv','sikov_B300_descriptive_balanced.csv',
                 'sikov_B300_descriptive_differences.csv','sikov_B300_descriptive_status.json',
                 'report_template/artifact.md','align_poverty_vintage_source_provenance.json',
                 'comparison_application_figures_preparation_QA.json','report_bibliography_preparation_QA.json'):
        add('validacion/'+name)
    add('validacion/public_helpers_generated_fixture_QA.json')
    add('validacion/sensitivity_sampled_graph_operational_restart.json')
    for name in ('sikov_poverty_stratum_descriptive_plan.json','sikov_poverty_stratum_descriptive_qa.json',
                 'sikov_poverty_stratum_descriptive_status.json',
                 'sikov_poverty_stratum_descriptive_plan_v1_invalid_labels.json',
                 'sikov_poverty_stratum_descriptive_qa_v1_invalid_labels.json',
                 'sikov_poverty_stratum_label_correction.json','sikov_poverty_stratum_boundary_scope.json'):
        add('validacion/'+name)
    for name in ('sikov_published_b300_plan.json','sikov_published_b300_qa.json',
                 'sikov_published_b300_native_status.json','sikov_published_b300_status.json'):
        add('validacion/'+name)
    for name in ('per_population_scores.csv','summary.csv','selected_reference_differences.csv'):
        add('resultados/published_b300_descriptive/'+name)
    for context in proof['contexts']:
        for rep in (30000,30001):
            for name in ('predictions.npz','status.json','completion.json'):
                add('evaluacion/published_b300/'+context+f'/rep_{rep:05d}/'+name)
    for name in ('per_population_stratum_scores.csv','stratum_descriptive_summary.csv',
                 'selected_reference_descriptive_differences.csv'):
        add('resultados/poverty_stratum_descriptive/'+name)
    for name in ('sensitivity_sampled_graph_freeze.json','sensitivity_sampled_graph_prefit_QA.json',
                 'sensitivity_sampled_graph_prediction_lock.json','sensitivity_sampled_graph_prediction_audit.json',
                 'sensitivity_sampled_graph_evaluation_status.json','sensitivity_sampled_graph_result_audit.json'):
        add('validacion/'+name)
    for name in ('descriptive_replications.csv','descriptive_summary.csv','descriptive_paired_contrasts.csv',
                 'physical_projection_replications.csv','physical_projection_summary.csv','physical_projection_paired_contrasts.csv'):
        add('sensibilidad_sampled_graph/resultados/'+name)
    # A separate 64-world descriptive sensitivity, never part of fresh256.
    for rep in range(30000,30064):
        source='sensibilidad_sampled_graph/inputs/'+f'rep_{rep:05d}/'
        for name in ('manifest.json','model_inputs/inputs.npz','model_inputs/sample.csv','evaluation_only/truth.npz'):
            add(source+name)
        fit='sensibilidad_sampled_graph/fits/'+f'rep_{rep:05d}/'
        add(fit+'completion.json')
        for prefix,methods in (('original/',proof['methods']),('published/',addendum['new_methods'])):
            for method in methods:
                add(fit+prefix+method+'.npz');add(fit+prefix+method+'_status.json')
    for name in ('Manuscript_ENDES_IAOS_Sikov_2026-10-10.docx',
                 'Reporte_comparacion_Sikov_2026-10-10.docx'):
        add('manuscrito/'+name)
    for name in BASE_RESULTS:add('resultados/'+name)
    for name in ADDED_RESULTS:add('resultados/published_sar_comparison/'+name)
    for scope in ('temporal','territorial'):
        for prefix in ('external_','external_extended_'):
            for name in EXTERNAL_SCOPE_RESULTS:add('resultados/'+prefix+scope+'/'+name)
        for name in ('real_prediction_lock_'+scope+'.json','sikov_published_prediction_lock_'+scope+'.json'):
            add('validacion/'+name)
    for family in (8,12):
        folder='external_comparison'+('_Holm12'if family==12 else'')
        for name in ('external_final_summary.json','endpoint_summary_both_scopes.csv',
                     'paired_balanced_contrasts_Holm'+str(family)+'.csv'):
            add('resultados/'+folder+'/'+name)
    for stem in ('matched_rmse_coverage','matched_interval_score'):
        for ext in ('.png','.pdf'):add('resultados/figuras/'+stem+ext)
    for name in ('matched_application2019_districts.csv','matched_application2019_summary.csv',
                 'matched_application2019_pointmaps.png','matched_application2019_pointmaps.pdf'):
        add('resultados/aplicacion2019/'+name)
    for name in ('auxiliary.npz','area_ids.csv','manifest.json'):add('datos/simulation_template/'+name)
    for name in ('aux_sikov14_official.csv','aux_sikov14_common_inputs.npz'):add('datos/'+name)
    for scope in ('real_2019_full','real_2024_training'):
        for name in ('preparation_manifest.json','fit_inputs.npz','area_frame.csv','area_calibration.csv'):
            add('datos/'+scope+'/'+name)
    for context in proof['contexts']:
        for rep in proof['evaluation_ids']:
            source='evaluacion/inputs/'+context+f'/rep_{rep:05d}/'
            for name in ('manifest.json','model_inputs/inputs.npz','model_inputs/sample.csv','evaluation_only/truth.npz'):
                add(source+name)
            for prefix,methods in (('fits',proof['methods']),('published_sar',addendum['new_methods'])):
                folder='evaluacion/'+prefix+'/'+context+f'/rep_{rep:05d}/'
                for method in methods:
                    add(folder+method+'.npz');add(folder+method+'_status.json')
                add(folder+'completion.json')
        for rep in (30000,30001):
            stem='validacion/sikov_bootstrap_evaluation_'+context+f'_rep{rep:05d}_B300'
            add(stem+'.json');add(stem+'.npz')
    for scope in ('temporal','territorial'):
        for prefix,methods in (('',proof['methods']),('published_sar/',addendum['new_methods'])):
            folder='evaluacion/'+prefix+'real_'+scope+'/'
            for method in methods:
                add(folder+method+'.npz');add(folder+method+'_status.json')
            add(folder+'completion.json');add(folder+'predictions.csv')
    development='desarrollo/improve_development_v4_native_rho999/'
    for name in ('LOPO_selection_rule_metrics.csv','candidate_ranking.csv','completion.json',
                 'development_plan.json','replications.csv','selected_recipe.json','summary.csv'):
        add(development+name)
    for name in ('improve_select.py','comparison_simulation.py','beta_power_stable.py','beta_power.py',
                 'sikov_model.py','improve_candidates.py','improve_candidates_protocol.json',
                 'sikov_bootstrap.py','improve_native_development.py','snapshot_status.json'):
        add(development+'source_snapshots/'+name)
    required.add(study/'fase2_2026-10-07/manuscrito/Manuscript_ENDES_IAOS_fase2.docx')
    return required,optional-required

def select_public_files(proof,addendum,root=None,study=None):
    """Selection depends solely on exact expected names, never file discovery."""
    required,optional=expected_public_paths(proof,addendum,root,study)
    base=Path(study or STUDY).absolute()
    for path in required|optional:
        public_logical_path(path,base)
    missing=sorted(str(p)for p in required if not p.is_file())
    if missing:raise FileNotFoundError('Required public artifacts missing: '+', '.join(missing))
    return required|{p for p in optional if p.is_file()}

def public_path_allowed(path):
    """Positive formats plus path/name exclusions, including private_external."""
    path=Path(path);relative=path.resolve().relative_to(STUDY.resolve())
    parts=[p.lower()for p in relative.parts]
    if any(p.startswith('.')or p=='__pycache__'or p=='private'or p.startswith('private_')for p in parts):return False
    if path.suffix.lower()in PRIVATE_SUFFIXES or path.suffix.lower()not in SAFE_SUFFIXES:return False
    if any(word in part for part in parts for word in PRIVATE_NAMES):return False
    if any(p in {'node_modules','venv','env','cmdstan'}or p.startswith('cmdstan-')for p in parts):return False
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--zip',type=Path,required=True);a=p.parse_args()
    from public_verify import prediction_preflight,completed_result_gates,poverty_supplement_gate,sampled_graph_supplement_gate,published_b300_gate
    meta=prediction_preflight();gates=completed_result_gates(meta);audit=gates['audit']
    poverty_supplement_gate(meta);sampled_graph_supplement_gate(meta)
    published_b300_gate(meta)
    visual=json.loads((ROOT/'validacion/report_visual_QA.json').read_text())
    if not visual.get('passed')or not visual.get('all_pages_reviewed'):raise RuntimeError('Word visual review incomplete')
    proof=meta['proof'];addendum=meta['addendum']
    if (audit.get('protocol_sha256')!=SHA(ROOT/'protocol_freeze.json')
            or audit.get('prediction_lock_sha256')!=SHA(ROOT/'validacion/fresh_prediction_lock.json')):
        raise RuntimeError('Scientific audit does not match the frozen prediction lock')
    expected_docs={'manuscrito/Manuscript_ENDES_IAOS_Sikov_2026-10-10.docx',
                   'manuscrito/Reporte_comparacion_Sikov_2026-10-10.docx'}
    if set(visual.get('documents_sha256',{}))!=expected_docs:raise RuntimeError('Both final Word hashes required in visual QA')
    for name,digest in visual['documents_sha256'].items():
        if SHA(ROOT/name)!=digest:raise RuntimeError('Final Word changed after visual QA '+name)
    if (proof['fresh_worlds']!=256 or len(proof['contexts'])!=4 or len(proof['evaluation_ids'])!=64
            or len(set(proof['methods']))!=9):raise RuntimeError('Unexpected complete frozen cohort')
    fresh=json.loads((ROOT/'validacion/fresh_evaluation_status.json').read_text())
    if (fresh['worlds']!=256 or fresh['protocol_sha256']!=SHA(ROOT/'protocol_freeze.json')
            or not fresh['all_method_world_pairs_represented']or not fresh['all_models_converged']):
        raise RuntimeError('Complete closed and audited fresh comparison required')
    for name,digest in fresh['outputs_sha256'].items():
        if Path(name).name!=name or SHA(ROOT/'resultados'/name)!=digest:
            raise RuntimeError('Scored artifact changed after closed comparison '+name)
    selected=select_public_files(proof,addendum)
    for path in selected:
        if not public_path_allowed(path):raise ValueError('Private records are excluded '+str(path))
    for name,value in proof['statistical_sources_sha256'].items():
        if SHA(name)!=value:raise RuntimeError('Frozen statistical source changed')
    for field in ('additional_sources_sha256','frozen_inputs_sha256'):
        for name,value in addendum[field].items():
            if SHA(name)!=value:raise RuntimeError('Frozen addendum source/certificate changed')
    if a.destination.exists()or a.zip.exists():raise FileExistsError('Use a new author-review destination')
    if a.destination.resolve().is_relative_to(ROOT)or a.zip.resolve().is_relative_to(ROOT):
        raise ValueError('Place release outside the immutable comparison directory')
    if a.zip.resolve().is_relative_to(a.destination.resolve()):raise ValueError('Archive cannot be inside its staging directory')
    a.destination.mkdir(parents=True)
    for f in sorted(selected):
        target=a.destination/'estudio_publicacion'/f.relative_to(STUDY)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
    environment=proof['runtime_environment']
    requirements=''.join(name+'=='+environment[name]+'\n'for name in ('numpy','scipy','pandas','cmdstanpy'))+'lxml\nmatplotlib\n'
    (a.destination/'requirements.txt').write_text(requirements)
    instructions='''ENDES comparison with reconstructed FH and two distinct SAR baselines

This local author-review package preserves the frozen 2026-10-10 comparison.
Read the Spanish comparison report and updated English manuscript first.
The FH equations, poverty strata and covariate lists were reconstructed.
The stored Sikov_SAR baseline uses a likelihood over the full target graph.
Sikov_SAR_Published instead fits the article's observed-district SAR/REML
family with a declared kNN6 graph and a directed extension to non-sampled
districts based on option 3 of the authors' 2024 study. These comparator
labels remain separate; original weights and cohort decisions are unresolved.
The original poverty vector, 765-area selection and spatial matrix are not
identified; this is not an exact numerical replication of the article.
The article introduction (p. 173) mentions stunting below -2 standard
deviations and tenth-percentile risk; its variable description (p. 188)
instead states the 2.5th percentile and prints hemoglobin below 11 g/l.
Those formulations and the printed unit do not identify the original implemented
endpoint or coding. This benchmark explicitly uses HC70 < -200 and HC56 < 110 g/L
for every strategy; it does not assign a tenth-percentile indicator to the authors.

PROTOCOL NAVIGATION
simulation_design_plan.json retains the historical base Holm2 multiplicity field.
The final prespecified families are defined in
validacion/sikov_published_addendum_freeze.json: primary balanced IS95 compares
Selected with FH, full-graph SAR and observed-domain SAR through three two-sided
tests adjusted by Holm3. Secondary MSE/NLL compares the same references in two
reused 2024 scopes through twelve tests adjusted by Holm12. The original Holm2
and Holm8 families remain archived separately; the frozen base plan is unchanged.

The primary comparison uses 256 fresh finite populations and parametric REML
bootstrap B100. The 75% ICC and 25% full-graph SAR weights were selected in
32 development worlds using a Laplace approximation for ICC integration. The
fresh evaluation uses four NUTS chains for the same ICC model with fixed weights.
Leave-one-population-out diagnostics audit approximate selection rather than
NUTS predictive performance. The additive comparator never
reselected these weights. The probability-scale SAR data-generating context
uses the frozen full graph, rather than the new comparator's observed graph.
Eight full-graph B300 worlds and physical projection are descriptive
sensitivities. A separate prospectively frozen B300 sensitivity repeats the
observed-domain SAR on the same eight cases, retaining naive and BC uncertainty,
unchanged points and the Selected_B100 strategy with its full-graph component.
It adds descriptive MCSE only; two populations per context give limited precision.
No mixture using the observed-domain SAR was selected or introduced.
A separate prospectively frozen sensitivity contains 64 worlds
whose population mechanism conditions on the sampled districts defining W11.
Clipping, binomial populations and GVF remain approximate Gaussian working
models. These worlds are not a fifth primary context or 320 primary worlds.
A poverty supplement reuses all 256 principal worlds and 11 strategies across
three stored strata and all/observed-endpoint/outside-common-sample masks.
It adds descriptive means and MCSE, without p values or confidence intervals;
its RMSE averages population roots. The sampled-graph supplement instead uses
the square root of mean MSE and unadjusted descriptive Monte Carlo intervals.
Neither sensitivity reselects weights or expands Holm3/Holm12.
ENDES 2024 temporal and territorial responses were previously
evaluated and are a disclosed secondary assessment, not a new untouched test.
All strategies receive the same observed sample, endpoints, masks and 14
available auxiliaries; FH/SAR use their published subset regressions, while
ICC uses the full auxiliary frame with regularization. Interpret strategy
differences, without attributing the entire effect only to ICC or mixing.
Do not reselect the recipe from these final results. This comparison does not
reproduce every later Sikov model. The updated bibliography states its scope.

VERIFY THE RELEASE AND RECOMPUTE EVERY SYNTHETIC SCORE
Run these commands from the extracted package root. Use Python 3.12 or a
compatible version, an isolated environment, and requirements.txt:
  python -m pip install -r requirements.txt
  python estudio_publicacion/sikov_2026-10-10/scripts/public_verify.py
The verifier validates all 256 principal and 64 separate closed prediction sets
and their independent result certificates before opening any
synthetic truth, then recomputes raw and physical-projection replication
scores, summaries and paired contrasts for all 11 methods, including the
three primary Holm tests, the original two-test family retained separately,
and descriptive context-by-endpoint contrasts. It also recomputes the Holm12
numeric external audit from district-level aggregates. It separately recomputes
all poverty-stratum, sampled-graph and observed-SAR B300 descriptive tables,
including the latter's independent native Gaussian algebra. It does not require
CmdStan compilation, private ENDES records or a new stochastic model fit.

REPLAY ONE SYNTHETIC MODEL FIT
Install CmdStan 2.40.0 through CmdStanPy and a C++ toolchain. For example:
  python -c "import cmdstanpy; cmdstanpy.install_cmdstan(version='2.40.0')"
  python estudio_publicacion/sikov_2026-10-10/scripts/public_replay.py --context SAR_probability_srs --rep 30000 --output ../replay_world30000
For the separately frozen sampled-graph sensitivity, use its declared context:
  python estudio_publicacion/sikov_2026-10-10/scripts/public_replay.py --context SAR_probability_sampled_graph_srs --rep 30000 --output ../replay_sampled_graph30000
The runner fits all 11 methods using frozen formulas and seeds. It copies the
Stan source and compiles in an output folder outside this release; archived
files stay unchanged, and scoring truth is never read.
Both replay choices first validate all 256 principal and 64 supplemental seals.
The supplementary fit streams are 973 (original methods) and 974 (Published).
Compiler/platform changes may cause floating-point or MCMC differences.
The diagnostic gates still apply. Full HMC conditional draws and executables
are omitted for size and platform portability; their original hash records and
independent numerical audit are retained. Original-host gates include absolute
paths and private input hashes; use the portable entry points in this package.

ACTUAL DATA REPRODUCIBILITY
Official census auxiliaries, district predictions and aggregated scores are
included. The supplied synthetic sample.csv files are generated simulation
records. Individual ENDES children/households, raw SAV files, environments,
credentials and the private municipal directory are excluded. Full actual-data
refits require authorized acquisition/preparation of the official 2019/2024
microdata; the preparation and scoring sources are provided. Stored source
records contain original-host paths/hashes for provenance, not private records.
The supplementary Spanish request for eight original reproduction inputs is a
local unsent draft; no author contact or change to the fixed protocol is implied.

This archive is verified for author review. Authorship, declarations and journal
submission remain separate. Publication or universal superiority is not
guaranteed by the package or by the journal quartile.
'''
    (a.destination/'README.txt').write_text(instructions)
    expected_staging={'estudio_publicacion/'+str(f.relative_to(STUDY))for f in selected}|{'requirements.txt','README.txt'}
    staging={str(f.relative_to(a.destination))for f in a.destination.rglob('*')if f.is_file()}
    if staging!=expected_staging:raise RuntimeError('Unexpected file entered release staging')
    files={name:SHA(a.destination/name)for name in sorted(expected_staging)}
    manifest={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'role':'verified author-review archive',
      'protocol_sha256':SHA(ROOT/'protocol_freeze.json'),'files_sha256':files,
      'addendum_sha256':meta['addendum_sha256'],'joint_prediction_lock_sha256':meta['joint_sha256'],
      'private_individual_records_exported':False,'stochastic_conditional_draws_exported':False,
      'synthetic_sample_records_exported':True,'synthetic_truths_exported':True,
      'real_units_children_households_exported':False,'executables_or_environments_exported':False,
      'expected_fresh_worlds':256,'expected_methods':meta['methods'],
      'expected_separate_sampled_graph_worlds':64,'poverty_supplement_reuses_primary_worlds':256,
      'supplemental_worlds_added_to_primary_family':False,
      'sampled_graph_plan_sha256':SHA(ROOT/'validacion/sensitivity_sampled_graph_freeze.json'),
      'poverty_plan_sha256':SHA(ROOT/'validacion/sikov_poverty_stratum_descriptive_plan.json'),
      'published_B300_plan_sha256':SHA(ROOT/'validacion/sikov_published_b300_plan.json'),
      'published_B300_reuses_primary_worlds':8,
      'primary_family':'prespecified additive balanced IS95 Holm3; original Holm2 preserved',
      'secondary_family':'reused 2024 MSE/NLL Holm12; original Holm8 preserved',
      'exact_original_numerical_replication':False,'submitted':False,'accepted':False}
    (a.destination/'public_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    a.zip.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6)as z:
        for f in sorted(a.destination.rglob('*')):
            if f.is_file():z.write(f,str(f.relative_to(a.destination)))
    with zipfile.ZipFile(a.zip)as z:
        bad=z.testzip()
        if bad:raise RuntimeError('CRC failure '+bad)
        if len(z.namelist())!=len(files)+1:raise RuntimeError('Unexpected archive members')
    print(json.dumps({'zip':str(a.zip.resolve()),'sha256':SHA(a.zip),'files':len(files)+1,
                      'bytes':a.zip.stat().st_size,'CRC':'PASS','public_verification_pending':True},indent=2))

if __name__=='__main__':main()
