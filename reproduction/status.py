#!/usr/bin/env python3
"""Export aggregate-only progress without private records, draws or file paths."""
import argparse
import datetime as dt
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--corrected-legacy-root',type=Path)
    a=p.parse_args(); base=a.run_base.resolve(); run=base/'sikov'
    controller=json.loads((base/'progress.json').read_text())
    completions=list((run/'evaluacion/fits').glob('*/rep_*/completion.json'))
    published=list((run/'evaluacion/published_sar').glob('*/rep_*/completion.json'))
    gate_passed=sum(all(json.loads(x.read_text())['convergence'].values()) for x in completions)
    actual={scope:{family:(run/'evaluacion'/family/('real_'+scope)/'completion.json').is_file()
                   for family in ('','published_sar')} for scope in ('temporal','territorial')}
    old=base/'legacy/study/resultados/resultados_comparadores'
    legacy_models=len(list(old.glob('final_*_v2/fit_results/*/*/rep_*/status.json')))
    legacy_rejected=(base/'legacy/STOPPED_INPUT_MISMATCH.json').is_file()
    corrected_models=0
    if a.corrected_legacy_root:
        corrected=a.corrected_legacy_root/'study/resultados/resultados_comparadores'
        corrected_models=len(list(corrected.glob('final_*_v2/fit_results/*/*/rep_*/status.json')))
    prefit=base/'sampled_prefit/progress.json'
    sampled_dispatch=json.loads(prefit.read_text()) if prefit.is_file() else {}
    result=dict(updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                status=controller['status'],article_reproduction_confirmed=False,
                development_worlds_closed=32,selected_recipe_75_25_reproduced=True,
                primary_original_worlds_closed=len(completions),primary_original_diagnostics_passed=gate_passed,
                primary_published_worlds_closed=len(published),primary_worlds_expected=256,
                sampled_graph_worlds_closed=len(list((run/'sensibilidad_sampled_graph/fits').glob('rep_*/completion.json'))),
                sampled_graph_worlds_expected=64,
                b300_worlds_closed=len(list((run/'evaluacion/b300').glob('*/rep_*/completion.json'))),
                b300_worlds_expected=8,actual_data_fits_closed=actual,
                legacy_initial_model_outputs_closed=legacy_models,
                legacy_initial_run_rejected_for_input_mismatch=legacy_rejected,
                legacy_initial_outputs_valid_for_reproduction=not legacy_rejected,
                legacy_corrected_model_outputs_closed=corrected_models,
                sampled_prefit_status=sampled_dispatch.get('status'),
                running_controller_jobs=len(controller.get('running',{})),
                failed_controller_jobs=sum(x['exit_code']!=0 for x in controller['commands']),
                private_records_in_this_report=False,
                notes=['Progress counts are not proof of numerical reproduction.',
                       'All actual-data source records and model chains remain outside this repository.',
                       'A disclosed spectral-basis difference is retained for final comparison.',
                       'The initial legacy run is rejected after a generated-input mismatch; its outputs are preserved separately.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
