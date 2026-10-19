"""Seal the additive benchmark before any new accuracy or held-out responses."""
from pathlib import Path
import datetime as dt,hashlib,json

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    dest=ROOT/'validacion/sikov_published_addendum_freeze.json'
    if dest.exists():raise FileExistsError('Preserve existing additive freeze')
    base=ROOT/'protocol_freeze.json';protocol=json.loads(base.read_text())
    for section in ('statistical_sources_sha256','frozen_inputs_sha256','runtime_binary_sha256'):
        for path,digest in protocol.get(section,{}).items():
            if sha(path)!=digest:raise RuntimeError('Original frozen file changed '+path)
    blocked=[ROOT/'validacion/fresh_evaluation_status.json',ROOT/'validacion/sikov_published_evaluation_status.json',ROOT/'validacion/joint_prediction_lock.json',ROOT/'resultados/fresh_replications.csv',ROOT/'resultados/external_comparison/external_final_summary.json',ROOT/'resultados/external_comparison_Holm12/external_final_summary.json']
    if any(path.exists() for path in blocked):raise RuntimeError('The additional protocol must precede scoring/joint seal')
    qa_names=('sikov_published_reference_validation.json','sikov_published_contract_validation.json','sikov_published_collect_validation.json','sikov_full_graph_scope_audit.json','align_external_extended_synthetic_qa.json')
    qas={}
    for filename in qa_names:
        path=ROOT/'validacion'/filename;qa=json.loads(path.read_text())
        if not qa.get('status','').startswith('PASS'):raise RuntimeError('Unpassed additional QA '+filename)
        qas[str(path)]=sha(path)
    names=('sikov_published_reference.py','sikov_published_fit.py','sikov_published_collect.py','sikov_published_audit.py','sikov_published_validate.py','sikov_published_contract_qa.py','sikov_published_collect_qa.py','sikov_published_freeze.py','sikov_final_audit.py','align_external_extended_score.py','align_external_extended_audit.py','align_external_extended_score_qa.py','align_external_independent_audit.py','comparison_joint_close.py')
    sources={str(ROOT/'scripts'/name):sha(ROOT/'scripts'/name) for name in names}
    result={
        'version':'additive_sampled_SAR2023_family_directed2024_option3_before_accuracy_v1',
        'frozen_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'approved_for_additional_reference_fits':True,
        'protocol_sha256':sha(base),'base_protocol_sha256':sha(base),
        'accuracy_or_heldout_outcomes_read_before_addendum':False,
        'base_protocol_and_selected_recipe_unchanged':True,
        'original_Holm2_and_external_Holm8_preserved':True,
        'new_methods':['Sikov_SAR_Published','Sikov_SAR_Published_BC'],
        'new_reference_scope':'Published Gaussian district-level SAR family on the sampled D-dimensional graph for the stated k6 rule; directed non-sampled extrapolation explicitly sourced to the 2024 author preprint. Exact original2023 W and non-sampled numerical implementation remain unavailable.',
        'sampled_graph':'kNN6 Euclidean longitude/latitude, within poverty stratum, row normalized, sampled endpoint areas only',
        'non_sampled_graph':'W11=sampled-only graph, W10=0; non-sampled rows kNN6 among all within-stratum domains excluding self, option3 of 2024 author preprint section2 eq7-10',
        'non_sampled_predictor':'X0 beta +(I-rho W00)^(-1) rho W01 uhat_s',
        'bootstrap':{'B':100,'primary':'naive parametric REML bootstrap','BC':'same bootstrap draws, raw negative MSE retained; zero floor only for interval sqrt','minimum_success_fraction':.95},
        'bootstrap_streams':{'fresh':'SeedSequence([20261010,941,context_index,rep]); endpoint offset1009*j','real':'SeedSequence([20261010,942,scope_index]); endpoint offset1009*j'},
        'fresh_worlds':256,'contexts':protocol['contexts'],'evaluation_ids':protocol['evaluation_ids'],
        'selected_recipe':protocol['selected_recipe'],'selection_performed_again':False,
        'expanded_primary_family':'Three equally context/endpoint/area balanced IS95 contrasts: frozen Selected versus original FH, full-graph SAR and added sampled-only SAR; Holm3, specified before truth access. Original Holm2 remains separately reported.',
        'expanded_external_family':'12 secondary contrasts, scopes temporal/territorial x references FH/full-graph SAR/sampled-only SAR x MSE/NLL, province-cluster inference, Holm12. Original Holm8 remains separately reported.',
        'graph_full_N_role':'Previously frozen full-graph SAR is retained as an adapted full latent graph reference and as the unchanged25percent component of Selected.',
        'B300_role':'Original eight prespecified full-graph SAR sensitivities retained; same fixed75/25 mixture descriptive postprocessing; no new selection or confirmatory tests.',
        'joint_scoring_gate':'All256 original and added fresh fits plus both original and added actual fits must be closed and jointly hashed before any new truth/held-out response scoring.',
        'joint_required_fields':['approved_for_scoring','protocol_sha256','addendum_sha256','files','fresh_cohort_size','all_expected_worlds','fresh_original_prediction_lock_sha256','fresh_published_prediction_lock_sha256','fresh_truth_read_before_joint','heldout2024responses_used_before_joint'],
        'no_world_or_finite_domain_excluded_after_fit':True,
        'actual2024_prior_phase_reuse_disclosed':True,
        'statistical_sources_sha256':sources,'additional_sources_sha256':sources,
        'frozen_inputs_sha256':qas,'runtime_binary_sha256':{},
        'base_kernel_sources_sha256':{str(ROOT/'scripts'/name):sha(ROOT/'scripts'/name) for name in ('sikov_model.py','sikov_bootstrap.py','sikov_bootstrap_sensitivity.py')},
    }
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS_ADDITIVE_REFERENCE_FREEZE_BEFORE_OUTCOMES','addendum_sha256':sha(dest),'base_protocol_sha256':sha(base),'additional_source_count':len(sources),'QA_count':len(qas)},indent=2))

if __name__=='__main__':main()
