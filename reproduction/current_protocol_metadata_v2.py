#!/usr/bin/env python3
"""Correct two audited post-fit collector tags on new aggregate copies only."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

PREFIX='resultados/resultados_comparadores/final_evaluation/'
FILES=tuple(PREFIX+n+'.csv'for n in ('combined_replications','combined_summary','additional_arms_summary','converged_only_summary'))
RULES={'HoggTSLNFullH':('extra_population_auxiliary_H_v1','extra_survey_unit_H_v1'),
       'HoggDesignPsiPSU':('experimental_design_PSU_v1','experimental_ultimate_cluster_psi_v1')}

def sha(data):return hashlib.sha256(data).hexdigest()
def correct(data):
    source=csv.DictReader(io.StringIO(data.decode('utf-8-sig')));columns=source.fieldnames
    if not columns or len(columns)!=len(set(columns))or not {'method','protocol'}.issubset(columns):
        raise ValueError('Required unique method/protocol schema missing')
    if any(c.lower()in {'child_id','caseid','household_id','psu_key','hvidx','person_id'}for c in columns):
        raise ValueError('Unit-level record fields cannot be normalized as aggregates')
    old=list(source);new=[];changed={k:0 for k in RULES}
    for row in old:
        if None in row or any(v is None for v in row.values()):raise ValueError('Malformed aggregate row')
        copy=dict(row)
        if row['method']in RULES:
            before,after=RULES[row['method']]
            if row['protocol']not in (before,after):raise ValueError('Unrecognized protocol for audited method')
            if row['protocol']==before:copy['protocol']=after;changed[row['method']]+=1
        new.append(copy)
    if not old:raise ValueError('Empty aggregate')
    text=io.StringIO(newline='');writer=csv.DictWriter(text,fieldnames=columns,lineterminator='\n');writer.writeheader();writer.writerows(new)
    output=text.getvalue().encode()
    reloaded=list(csv.DictReader(io.StringIO(output.decode())))
    if len(old)!=len(reloaded)or any(a[k]!=b[k]for a,b in zip(old,reloaded)for k in columns if k!='protocol'):
        raise ValueError('A nonprotocol aggregate field changed')
    return output,{'rows':len(old),'changed_protocol_rows':changed,'headers_preserved':True,
        'all_nonprotocol_cells_retained_as_identical_strings':True,'numeric_values_converted_or_changed':False,
        'source_byte_sha256':sha(data),'current_byte_sha256':sha(output)}

def run(source,output,cause):
    if output.exists():raise FileExistsError('Preserve previous metadata revisions')
    report=json.loads(cause.read_text())
    if report.get('status')!='METADATA_CAUSE_CONFIRMED_NUMERICAL_DIFFERENCES_RETAINED':
        raise ValueError('Completed independent metadata-cause audit is required')
    original_export=json.loads((source/'export_manifest.json').read_text())
    if original_export.get('status')!='FRESH_AGGREGATES_EXPORTED_FAIL_SUPPORT_RETAINED':
        raise ValueError('The closed, current historical four-file export is required')
    export_seals={row['path']:row['observed_fresh_artifact_sha256']for row in original_export['files']}
    cause_seals={row['file']:row['fresh_sha256']for row in report['25_aggregate_file_bindings']}
    for relative in FILES:
        digest=sha((source/relative).read_bytes())
        cause_digest=cause_seals.get(relative,cause_seals.get(Path(relative).name))
        if digest!=export_seals.get(relative) or digest!=cause_digest:
            raise ValueError('Historical source CSV differs from its closed export or independent causal seal')
    planned={rel:correct((source/rel).read_bytes())for rel in FILES}
    output.mkdir(parents=True)
    for rel,(data,_)in planned.items():
        p=output/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    receipt={'format':'iaos-current-protocol-metadata-correction-v1',
        'status':'PASS_CURRENT_METADATA_CORRECTION_NUMERICAL_VALUES_UNCHANGED',
        'files':{rel:proof for rel,(_,proof)in planned.items()},'rules':RULES,
        'original_independent_cause_report_sha256':sha(cause.read_bytes()),
        'original_closed_four_file_export_manifest_sha256':sha((source/'export_manifest.json').read_bytes()),'helper_sha256':sha(Path(__file__).read_bytes()),
        'original_fit_dispatch_inputs_kernels_seeds_scores_or_comparison_flags_changed':False,
        'old_reference_and_historical_fresh_CSVs_modified':False,'new_model_fits_or_scores_executed':False,
        'new_tags_describe_existing_actual_stage1_targets_not_a_new_model':True}
    (output/'current_metadata_correction.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source-root','output-root','cause-report'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();v=run(a.source_root,a.output_root,a.cause_report);print(json.dumps({'status':v['status'],'files':len(v['files'])}))

if __name__=='__main__':main()
