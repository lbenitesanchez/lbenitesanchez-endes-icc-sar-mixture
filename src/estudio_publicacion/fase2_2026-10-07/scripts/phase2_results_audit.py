"""Read-only release gate combining actual independent scientific audits."""
from pathlib import Path
import csv,datetime as dt,hashlib,json,zipfile
PHASE=Path(__file__).resolve().parents[1];STUDY=PHASE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    protocol=read(PHASE/'protocol_freeze.json');checks=[]
    def check(name,passed,details=None):
        checks.append({'name':name,'passed':bool(passed),'details':details})
    for name,digest in protocol['source_sha256'].items():
        source=PHASE/name
        if not source.exists():source=STUDY/name
        check('frozen source '+name,source.exists()and sha(source)==digest)
    for name,digest in protocol['input_sha256'].items():
        source=STUDY/name.removeprefix('previous_study/')if name.startswith('previous_study/')else PHASE/name
        check('frozen input '+name,source.exists()and sha(source)==digest)
    audits={}
    for name in ['phase2_simulation_independent_audit.json','phase2_external_independent_audit.json',
                 'externa_final_fit_scope_chain_audit.json','diseno_cut8_2019_application_validation.json',
                 'diseno_cut8_2025_application_validation.json','diseno_cut16_predeclared_state_identity.json',
                 'geografia_independent_phase2_freeze_final_check.json','geografia_hogg_recovery_independent_review.json']:
        source=PHASE/'validacion'/name;data=read(source)
        audits[name]={'sha256':sha(source),'status':data['status']}
        check('independent completed proof '+name,data['status'].startswith('PASS'))
    geography=read(PHASE/'validacion/geografia_completion_manifest.json')
    check('auxiliary reconstruction completed',geography['status']=='COMPLETE_GEOGRAPHY_IMPROVEMENT_3'and geography['all_aux_and_age_sex_audits_pass'])
    chronology=[protocol['frozen_utc']]
    for scope in ('temporal','territorial'):
        lock=read(PHASE/f'validacion/externa_prediction_lock_{scope}.json')
        check(scope+' binds immutable protocol',lock['protocol_sha256']==sha(PHASE/'protocol_freeze.json'))
        check(scope+' complete three methods',len(lock['predictions'])==3)
        for item in lock['predictions']:
            check(scope+' prediction '+item['method'],sha(PHASE/item['path'])==item['sha256'])
            check(scope+' diagnostics '+item['method'],sha(PHASE/item['status_path'])==item['status_sha256'])
        if scope=='temporal':
            chronology.extend([lock['locked_utc'],read(PHASE/'validacion/externa_first_reserved_open.json')['time_utc']])
        else:chronology.append(lock['locked_utc'])
    completion=read(PHASE/'resultados/externa_final/completion.json')
    chronology.append(completion['completed_utc'])
    times=[dt.datetime.fromisoformat(t)for t in chronology]
    check('freeze temporal lock first access territorial lock scoring chronology',all(a<b for a,b in zip(times,times[1:])),chronology)
    check('external final completion',completion['status']=='COMPLETE_RESERVED_EXTERNAL_VALIDATION')
    with (PHASE/'resultados/externa_final/contrasts_Holm8.csv').open()as f:contrasts=list(csv.DictReader(f))
    truth=lambda x:x.lower()=='true'
    wins=sum(truth(r['candidate_significantly_better'])for r in contrasts)
    losses=sum(truth(r['candidate_significantly_worse'])for r in contrasts)
    check('exact eight contrast family',len(contrasts)==8)
    check('complete adverse results',wins==0 and losses==4 and all(r['scope']=='temporal'for r in contrasts if truth(r['candidate_significantly_worse'])))
    sim=read(PHASE/'resultados/design_simulation/final/completion.json')
    check('all forty fixed population worlds completed',sim['unique_populations']==40 and sim['all40_completed'])
    oldzip=STUDY/'ENDES_comparadores_publicacion_2026-10-06_entrega.zip'
    check('previous delivery archive unchanged',sha(oldzip)==protocol['previous_delivery_sha256'])
    with zipfile.ZipFile(oldzip)as z:
        candidates=[p for p in z.namelist()if p.endswith('.docx')]
        for name in candidates:
            matches=[p for p in (STUDY/'manuscrito').glob('*.docx')if p.name==Path(name).name]
            if matches:check('previous Word unchanged '+matches[0].name,hashlib.sha256(z.read(name)).hexdigest()==sha(matches[0]))
    report={'status':'PASS_COMBINED_INDEPENDENT_RESULTS_RELEASE_GATE'if all(c['passed']for c in checks)else'FAIL_COMBINED_RESULTS_RELEASE_GATE',
            'checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'checks':checks,'independent_proofs':audits,
            'primary_external_findings':{'improved':wins,'worsened':losses,'not_detected':len(contrasts)-wins-losses},
            'editorial_decision':'ICC reference retained for application; CUT8 extension reported as sensitivity, without superiority claim.',
            'limitations':['Internal hash freeze is not public preregistration.','One operational QA metadata reexecution is documented rather than declared byte identical.',
                           'CUT8 temporary hyperparameter CSVs are not retained; posterior arrays and diagnostics remain audited.',
                           'Technical readiness is separate from author approval, submission and acceptance.']}
    target=PHASE/'validacion/phase2_results_independent_audit.json'
    if target.exists():raise FileExistsError('Preserve prior release gate')
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'checks':len(checks),'failed':[c for c in checks if not c['passed']],'proof_sha256':sha(target)}))
    if not all(c['passed']for c in checks):raise SystemExit(1)
if __name__=='__main__':main()
