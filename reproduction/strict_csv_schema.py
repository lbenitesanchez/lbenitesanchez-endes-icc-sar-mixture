#!/usr/bin/env python3
"""Strict schema, cohort and value comparison for every matching public CSV.

Fresh result reads require the complete Sikov score and real-score receipts.
Inventory mode never opens CSV, NumPy, chain, private or source payloads.
Selfcheck is for comparison mechanics only and never certifies reproduction.
Unmatched required score files and schema/key omissions yield INCOMPLETE/FAIL.
All matching eligible CSVs are compared, including those beyond required scores.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

TIMING_COLUMNS = {
    'seconds':'Measured runtime is machine-dependent.',
    'fit_seconds':'Measured model runtime is machine-dependent.',
    'mean_fit_seconds':'Mean measured runtime is machine-dependent.',
    'elapsed_seconds':'Measured elapsed runtime is machine-dependent.',
    'wall_seconds':'Measured wall runtime is machine-dependent.',
}
PROVENANCE_COLUMNS = {
    'phase':'The fresh scorer labels fresh_evaluation / fresh_evaluation_addendum; this identifies execution provenance, not the estimator, population, score or method protocol.',
}
IGNORED_VALUES = {**TIMING_COLUMNS, **PROVENANCE_COLUMNS}
KEY_ORDER = ('scenario','context','scope','replication','rep','method','candidate','reference',
             'protocol','outcome','endpoint','poverty_stratum','group','metric',
             'ubigeo','area','psu','psu_key','child_id','individual_id','unit_id')
NUMERIC_MISSING = {'','nan','na','n/a','null','none'}
SOURCE_SUFFIXES = {'.py','.r','.stan','.sh','.js','.ts'}
REQUIRED_PREFIXES = (
    'resultados/published_sar_comparison/', 'resultados/published_b300_descriptive/',
    'resultados/poverty_stratum_descriptive/', 'sensibilidad_sampled_graph/resultados/',
    'resultados/external_comparison/', 'resultados/external_comparison_Holm12/',
    'resultados/external_temporal/', 'resultados/external_territorial/',
    'resultados/external_extended_temporal/', 'resultados/external_extended_territorial/',
    'resultados/aplicacion2019/',
)
REQUIRED_FILES = {
    'resultados/fresh_balanced_paired_contrasts.csv',
    'validacion/sikov_B300_descriptive_scores.csv',
    'validacion/sikov_B300_descriptive_balanced.csv',
    'validacion/sikov_B300_descriptive_differences.csv',
}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def file_role(relative):
    """Classify by full path only; never inspect a excluded payload."""
    p=Path(relative);parts=tuple(part.lower()for part in p.parts);name=p.name.lower()
    if any(part.startswith('private')for part in parts) or name=='units.csv' or p.suffix.lower()in('.sav','.rds'):
        return 'excluded_private_or_raw_record_payload'
    if any(part in ('hmc_chains','icc_sampler','sampler','chains','chain_outputs')for part in parts) or \
            (p.suffix.lower()=='.csv' and re.search(r'(?:beta_power|hogg_s[12]|model)-\d{8,}',name)):
        return 'excluded_sampler_chain_payload'
    if p.suffix.lower()in('.npz','.npy'):
        if any(part in ('evaluation_only','evaluacion_only')for part in parts) or name.startswith('truth'):
            return 'excluded_numpy_synthetic_truth_payload'
        if any(token in name for token in ('draws','posterior','pooled')):
            return 'excluded_numpy_posterior_draw_payload'
        return 'excluded_numpy_input_or_prediction_payload'
    if p.suffix.lower()in SOURCE_SUFFIXES or '__pycache__'in parts:
        return 'excluded_source_or_runtime_payload'
    if p.suffix.lower()!='.csv':return 'excluded_non_CSV_artifact'
    if relative in REQUIRED_FILES or relative.startswith(REQUIRED_PREFIXES):return 'required_score_or_figure_data_CSV'
    if 'model_inputs'in parts and name=='sample.csv':return 'synthetic_sample_input_CSV'
    if 'datos'in parts or 'validacion'in parts:return 'aggregate_input_or_audit_CSV'
    if 'desarrollo'in parts:return 'development_CSV'
    if relative.startswith('evaluacion/')and'real_'in relative:return 'real_district_prediction_CSV'
    return 'other_aggregate_CSV'

def inventory(root):
    eligible={};excluded=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(root).as_posix();role=file_role(relative)
        if role.startswith('excluded_'):
            excluded.append({'relative_path':relative,'role':role,'payload_read':False})
        else:eligible[relative]=role
    return eligible,excluded

def fresh_gate(root):
    needed=(root/'validacion/sikov_fresh_prediction_lock.json',
            root/'validacion/sikov_score_completion.json',root/'real_scoring_receipt.json')
    missing=[p.relative_to(root).as_posix()for p in needed if not p.is_file()]
    if missing:return {'approved':False,'missing':missing},{}
    lock,score,real=(json.loads(path.read_text())for path in needed)
    approved=(lock.get('approved_for_scoring')is True and lock.get('principal_worlds')==256 and
              lock.get('sampled_graph_worlds')==64 and lock.get('B300_worlds')==8 and
              score.get('joint_fresh_prediction_lock_sha256')==sha(needed[0]) and
              real.get('complete_fresh_prediction_lock_sha256')==sha(needed[0]))
    bindings={**score.get('outputs_sha256',{}),**real.get('results_sha256',{})}
    for relative,digest in bindings.items():
        path=root/relative
        if not path.is_file()or sha(path)!=digest:
            raise RuntimeError('Sealed fresh result changed or missing: '+relative)
    return {'approved':bool(approved),'missing':[],
        'receipt_sha256':{path.relative_to(root).as_posix():sha(path)for path in needed},
        'heldout_response_or_truth_payload_opened_by_this_checker':False},bindings

def read_csv(path):
    with path.open(newline='',encoding='utf-8-sig')as stream:
        header=next(csv.reader(stream),[])
    if not header and path.name=='excluded_domains.csv' and not path.read_text(encoding='utf-8-sig').strip():
        # The frozen external scorer serializes a zero-row exclusion list as
        # a headerless newline. This exception applies only to that sentinel;
        # an omitted field in any nonempty/headered CSV cannot pass here.
        return pd.DataFrame()
    if not header or len(header)!=len(set(header)):
        raise ValueError('Empty or duplicate CSV header')
    # Strings preserve six-digit UBIGEO and categorical integer labels. Numeric
    # values, missingness and Boolean flags are compared explicitly below.
    frame=pd.read_csv(path,dtype=str,keep_default_na=False,na_filter=False)
    if list(frame.columns)!=header:raise ValueError('Parser changed the literal CSV header')
    return frame

def key_columns(frame):
    candidates=[key for key in KEY_ORDER if key in frame.columns and key not in IGNORED_VALUES]
    if not candidates:return []
    return candidates

def numeric_values(series):
    stripped=series.str.strip();missing=stripped.str.lower().isin(NUMERIC_MISSING)
    numeric=pd.to_numeric(stripped.mask(missing),errors='coerce')
    invalid=numeric.isna()&~missing
    if invalid.any():return None
    return numeric.to_numpy(float),missing.to_numpy(bool)

def is_count_column(column):
    return (column in ('replication','rep','replications','worlds','paired_worlds','areas','districts','provinces','B')
            or column.endswith(('_count','_counts','_worlds','_districts','_areas','_replications'))
            or column.startswith(('n_','number_')))

def compare_file(fresh,reference,atol,rtol):
    a,b=read_csv(fresh),read_csv(reference)
    ca,cb=set(a.columns),set(b.columns)
    missing=sorted(cb-ca);extra=sorted(ca-cb)
    # Timing and execution-phase metadata can be added/omitted explicitly;
    # statistical protocol, group, outcome, method and diagnostic flags cannot.
    undocumented=[name for name in missing+extra if name not in IGNORED_VALUES]
    result={'fresh_sha256':sha(fresh),'reference_sha256':sha(reference),
        'fresh_rows':len(a),'reference_rows':len(b),'fresh_columns':list(a.columns),
        'reference_columns':list(b.columns),'same_column_order':list(a.columns)==list(b.columns),
        'missing_columns':missing,'additional_columns':extra,
        'documented_optional_metadata_columns':[name for name in missing+extra if name in IGNORED_VALUES]}
    if not ca and not cb:
        result['documented_headerless_empty_no_exclusions_sentinel']=True
    if undocumented:
        return {**result,'status':'FAIL_SCHEMA','undocumented_column_difference':undocumented}
    if [column for column in a if column not in IGNORED_VALUES]!=[column for column in b if column not in IGNORED_VALUES]:
        return {**result,'status':'FAIL_SCHEMA_COLUMN_ORDER',
                'rule':'Statistical column order must match after documented metadata exclusions.'}
    if len(a)!=len(b):return {**result,'status':'FAIL_COHORT_ROW_COUNT'}
    keys=key_columns(a)
    if keys!=key_columns(b):return {**result,'status':'FAIL_COMPARISON_KEYS','fresh_keys':keys,'reference_keys':key_columns(b)}
    if keys:
        if a.duplicated(keys).any()or b.duplicated(keys).any():
            # Input tables can lack an individual key. In this case preserving
            # exact row order checks the complete cohort; it never drops rows.
            if fresh.name=='sample.csv' and fresh.parent.name=='model_inputs':
                keys=[]
            else:
                return {**result,'status':'FAIL_DUPLICATE_COMPARISON_KEY','comparison_keys':keys,
                        'fresh_duplicate_rows':int(a.duplicated(keys).sum()),'reference_duplicate_rows':int(b.duplicated(keys).sum())}
        else:
            a=a.set_index(keys).sort_index();b=b.set_index(keys).sort_index()
            if not a.index.equals(b.index):return {**result,'status':'FAIL_COHORT_KEY_UNIVERSE','comparison_keys':keys}
    result['comparison_keys']=keys;result['alignment']='complete_key_universe'if keys else'exact_row_order_no_rows_dropped'
    checks={};failures=[]
    for column in sorted((ca&cb)-set(keys)):
        if column in IGNORED_VALUES:
            checks[column]={'status':'DOCUMENTED_RUNTIME_OR_PROVENANCE_VALUE_EXCLUSION','reason':IGNORED_VALUES[column]}
            continue
        left,right=a[column],b[column]
        values=pd.concat([left,right],ignore_index=True).str.strip().str.lower()
        if set(values).issubset({'true','false',''}):
            ok=left.str.strip().str.lower().to_numpy()==right.str.strip().str.lower().to_numpy()
            check={'comparison':'boolean_flags_exact','different_rows':int((~ok).sum())}
        else:
            x,y=numeric_values(left),numeric_values(right)
            if x is not None and y is not None:
                xv,xmissing=x;yv,ymissing=y
                same_missing=np.array_equal(xmissing,ymissing)
                finite=np.isfinite(xv)&np.isfinite(yv)
                exact=is_count_column(column)
                ok=(xv==yv)|(np.isnan(xv)&np.isnan(yv))if exact else np.isclose(xv,yv,atol=atol,rtol=rtol,equal_nan=True)
                check={'comparison':'numeric_count_exact'if exact else'numeric_value',
                    'missingness_identical':bool(same_missing),'different_rows':int((~ok).sum()),
                    'maximum_absolute_difference':float(np.max(np.abs(xv[finite]-yv[finite])))if finite.any()else None,
                    'atol':0 if exact else atol,'rtol':0 if exact else rtol}
                if not same_missing:ok=np.zeros(len(xv),dtype=bool)
            else:
                ok=left.to_numpy()==right.to_numpy()
                check={'comparison':'categorical_or_text_exact','different_rows':int((~ok).sum())}
        check['status']='PASS'if ok.all()else'FAIL_VALUES';checks[column]=check
        if not ok.all():failures.append(column)
    return {**result,'status':'PASS'if not failures else'FAIL_VALUES','columns':checks,
            'failed_columns':failures,'row_values_or_individual_keys_exported':False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fresh-root','reference-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--inventory-only',action='store_true')
    p.add_argument('--self-check',action='store_true',help='Comparison mechanics only; never certify fresh results')
    p.add_argument('--require-all-reference-csv',action='store_true',help='Treat every unmatched eligible reference CSV as incomplete')
    p.add_argument('--atol',type=float,default=1e-10);p.add_argument('--rtol',type=float,default=0.)
    a=p.parse_args();fresh=a.fresh_root.resolve();reference=a.reference_root.resolve()
    ff,excluded_fresh=inventory(fresh);rr,excluded_reference=inventory(reference)
    matching=sorted(set(ff)&set(rr));fresh_only=sorted(set(ff)-set(rr));reference_only=sorted(set(rr)-set(ff))
    required_missing=[name for name in reference_only if rr[name]=='required_score_or_figure_data_CSV']
    required_unmatched=[name for name in fresh_only if ff[name]=='required_score_or_figure_data_CSV']
    result={'status':'INVENTORY_ONLY_NOT_REPRODUCED','helper_sha256':sha(__file__),
        'inventory_only':a.inventory_only,'self_check_not_reproduced':a.self_check,
        'article_reproduction_confirmed':False,'all_matching_eligible_CSVs_requested':True,
        'matching_relative_paths':matching,
        'fresh_only':[{'relative_path':name,'role':ff[name]}for name in fresh_only],
        'reference_only':[{'relative_path':name,'role':rr[name]}for name in reference_only],
        'required_reference_files_missing_fresh':required_missing,
        'required_fresh_files_missing_reference':required_unmatched,
        'excluded_files':{'fresh':excluded_fresh,'reference':excluded_reference},
        'documented_optional_schema_and_value_metadata':IGNORED_VALUES,
        'source_NumPy_private_chain_payloads_opened':False,
        'role_counts':{'fresh':dict(Counter(ff.values())),'reference':dict(Counter(rr.values()))},
        'atol':a.atol,'rtol':a.rtol,'checks':{}}
    gate={'approved':False,'not_executed':'Inventory only'}
    if not a.inventory_only:
        gate=({'approved':True,'mode':'SELF_CHECK_NOT_REPRODUCED'}if a.self_check else fresh_gate(fresh)[0])
        if not gate['approved']:
            result['status']='INCOMPLETE_SCORE_GATE_NOT_CLOSED_NO_CSV_PAYLOAD_READ'
        else:
            for name in matching:
                try:result['checks'][name]=compare_file(fresh/name,reference/name,a.atol,a.rtol)
                except Exception as error:result['checks'][name]={'status':'FAIL_PARSE_OR_SCHEMA','error':repr(error)}
            failed=[name for name,check in result['checks'].items()if check['status']!='PASS']
            incomplete=(required_missing or required_unmatched or (a.require_all_reference_csv and reference_only))
            result['failed_checks']=failed
            result['status']=('FAIL_SCHEMA_COHORT_OR_VALUES_RETAINED'if failed else
                'INCOMPLETE_UNMATCHED_REQUIRED_CSVS'if incomplete else
                'SELF_CHECK_NOT_REPRODUCED'if a.self_check else'PASS_ALL_MATCHING_CSV_SCHEMAS_COHORTS_AND_VALUES')
            result['all_matching_CSV_schemas_cohorts_and_values_match']=not failed
            result['all_required_score_CSVs_present_on_both_sides']=not bool(required_missing or required_unmatched)
            result['unmatched_nonrequired_files_are_not_silently_certified']=True
            result['cells_checked']=sum(check['fresh_rows']*sum(c.get('status')!='DOCUMENTED_RUNTIME_OR_PROVENANCE_VALUE_EXCLUSION'
                for c in check.get('columns',{}).values())for check in result['checks'].values()if 'fresh_rows'in check)
    result['fresh_score_gate']=gate
    for item in excluded_fresh:
        if item['relative_path']in gate.get('receipt_sha256',{}):
            item['payload_read']=True
            item['role']='fresh_completion_metadata_read_for_CSV_gate_only'
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'matching_CSVs':len(matching),'CSV_files_checked':len(result['checks']),
        'failed_checks':len(result.get('failed_checks',[])),'reference_only':len(reference_only),'fresh_only':len(fresh_only),
        'article_reproduction_confirmed':False,'excluded_payloads_opened':False},indent=2))
    if not a.inventory_only and result['status']not in ('SELF_CHECK_NOT_REPRODUCED','PASS_ALL_MATCHING_CSV_SCHEMAS_COHORTS_AND_VALUES'):
        raise SystemExit(2)

if __name__=='__main__':main()
