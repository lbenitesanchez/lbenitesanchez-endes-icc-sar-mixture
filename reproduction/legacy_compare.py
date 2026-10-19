#!/usr/bin/env python3
"""Compare fresh legacy outputs with every retained aggregate numerical target.

Elapsed times are excluded, but convergence/availability/counts are retained.
Missing results yield INCOMPLETE. No fit, threshold relaxation, seed replacement
or outcome-dependent selection occurs here. Numeric comparison tolerances are
explicit and the three Appendix A tables are checked at published precision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

PHASE='fase2_2026-10-07'
FILES={
 'resultados/resultados_comparadores/final_evaluation/combined_replications.csv': ['scenario','replication','method','protocol','group','outcome'],
 'resultados/resultados_comparadores/final_evaluation/combined_summary.csv': ['scenario','method','protocol','group','outcome'],
 'resultados/resultados_comparadores/final_evaluation/primary_publication_table.csv': ['scenario','method','protocol','group','outcome'],
 'resultados/resultados_comparadores/final_evaluation/primary_balanced_interval_score.csv': ['reference'],
 'resultados/resultados_comparadores/final_evaluation/structural_summary.csv': ['scenario','method','protocol','group','outcome'],
 'resultados/resultados_comparadores/final_evaluation/additional_arms_summary.csv': ['scenario','method','protocol','group','outcome'],
 'resultados/resultados_comparadores/final_evaluation/converged_only_summary.csv': ['scenario','method','protocol','group','outcome'],
 'resultados/resultados_comparadores/final_evaluation/paired_contrasts.csv': ['scenario','contrast','group','outcome'],
 'resultados/resultados_comparadores/spatial_evaluation/summary.csv': ['scenario','method','protocol','group','outcome'],
 'resultados/resultados_comparadores/spatial_evaluation/replications.csv': ['scenario','replication','method','protocol','group','outcome'],
 'resultados/resultados_comparadores/spatial_evaluation/paired_contrasts.csv': ['scenario','contrast','group','outcome'],
 PHASE+'/resultados/design_simulation/final/population_metrics.csv': ['context','rep','method','outcome'],
 PHASE+'/resultados/design_simulation/final/summary.csv': ['context','method','outcome'],
 PHASE+'/resultados/design_simulation/final/paired_diagnostics.csv': ['context','outcome','reference','metric'],
 PHASE+'/resultados/externa_final/summary.csv': ['scope','outcome','method'],
 PHASE+'/resultados/externa_final/contrasts_Holm8.csv': ['scope','outcome','reference'],
 PHASE+'/resultados/application2025_summary.csv': ['method','outcome'],
 PHASE+'/resultados/application2025_baseline/predictions.csv': ['ubigeo'],
 PHASE+'/resultados/application2025_designcut8/district_predictions.csv': ['ubigeo'],
 PHASE+'/resultados/externa_temporal_2019_baseline/predictions.csv': ['ubigeo'],
 PHASE+'/resultados/externa_temporal_2019_designcut8/district_predictions.csv': ['ubigeo'],
 PHASE+'/resultados/externa_temporal_2019_hogg/predictions.csv': ['ubigeo'],
 PHASE+'/resultados/externa_territorial_2024_baseline/predictions.csv': ['ubigeo'],
 PHASE+'/resultados/externa_territorial_2024_designcut8/predictions.csv': ['ubigeo'],
 PHASE+'/resultados/externa_territorial_2024_hogg/predictions.csv': ['ubigeo'],
}
LABEL={'BetaDesignUncertainty':'Variance CUT8','BetaPowerExisting':'ICC beta',
       'HoggAgeSexSpatial':'Hogg age/sex BYM2','baseline':'ICC beta','point':'Variance point','cut8':'Variance CUT8'}


def csv_compare(actual, expected, keys, atol, rtol):
    a=pd.read_csv(actual,dtype={'ubigeo':str});b=pd.read_csv(expected,dtype={'ubigeo':str})
    keys=[k for k in keys if k in a and k in b]
    if not keys or a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError('Missing or duplicate comparison keys')
    a=a.set_index(keys).sort_index();b=b.set_index(keys).sort_index()
    if set(a.index)!=set(b.index):
        return {'status':'FAIL_SUPPORT','fresh_rows':len(a),'reference_rows':len(b)}
    b=b.reindex(a.index)
    ignored={c for c in set(a)|set(b) if 'seconds' in c or c in {'run_tag'}}
    columns=(set(a)|set(b))-ignored
    if not columns.issubset(a) or not columns.issubset(b):
        return {'status':'FAIL_COLUMNS','different_columns':sorted(set(a)^set(b))}
    differences=[];maxima={};checked=0
    for column in sorted(columns):
        av,bv=a[column],b[column]
        if pd.api.types.is_numeric_dtype(av) and pd.api.types.is_numeric_dtype(bv):
            x,y=av.to_numpy(float),bv.to_numpy(float)
            ok=np.isclose(x,y,atol=atol,rtol=rtol,equal_nan=True)
            finite=np.isfinite(x)&np.isfinite(y)
            maxima[column]=float(np.max(np.abs(x[finite]-y[finite]))) if finite.any() else None
        else:
            x,y=av.fillna('<NA>').astype(str).to_numpy(),bv.fillna('<NA>').astype(str).to_numpy()
            ok=x==y
        checked+=len(ok)
        if not ok.all():
            differences.append({'column':column,'different_rows':int((~ok).sum()),
                                'examples':[str(a.index[i]) for i in np.flatnonzero(~ok)[:3]]})
    return {'status':'PASS' if not differences else 'FAIL_NUMERICAL_TARGET','rows':len(a),'cells':checked,
            'maximum_absolute_errors':maxima,'differences':differences,
            'fresh_sha256':hashlib.sha256(actual.read_bytes()).hexdigest(),
            'reference_sha256':hashlib.sha256(expected.read_bytes()).hexdigest()}


def appendix_tables(root):
    sim=pd.read_csv(root/PHASE/'resultados/design_simulation/final/summary.csv')
    external=pd.read_csv(root/PHASE/'resultados/externa_final/contrasts_Holm8.csv')
    app=pd.read_csv(root/PHASE/'resultados/application2025_summary.csv')
    pp=lambda value,n=2:f'{100*float(value):.{n}f}'
    return [
       [['Context','Estimator','Outcome','RMSE','Coverage','Width','IS95','MCSE']]+
       [[r.context,LABEL[r.method],r.outcome,pp(r.rmse),pp(r.coverage95,1),pp(r.mean_width),
         f'{r.interval_score95:.3f}',f'{r.interval_score95_MCSE:.3f}']for r in sim.itertuples()],
       [['Scope','Outcome','Comparator','Difference','95% interval','Holm p']]+
       [[r.scope,r.outcome,LABEL[r.reference],f'{r.difference:.4f}',
         f'[{r.unadjusted_t_ci_low:.4f}, {r.unadjusted_t_ci_high:.4f}]',f'{r.p_Holm8:.3g}']for r in external.itertuples()],
       [['Estimator','Outcome','Districts','Mean','Mean width','Between-state share']]+
       [[LABEL[r.method],r.outcome,str(r.districts),pp(r.mean_prevalence),pp(r.mean_width),pp(r.mean_between_share,1)]for r in app.itertuples()]]


def compare_tables(actual, expected):
    cells=0;different=[]
    if len(actual)!=len(expected):
        return {'status':'FAIL_TABLE_COUNT'}
    for t,(aa,bb) in enumerate(zip(actual,expected),1):
        if len(aa)!=len(bb) or any(len(x)!=len(y)for x,y in zip(aa,bb)):
            return {'status':'FAIL_TABLE_DIMENSIONS','table':t}
        for r,(ar,br)in enumerate(zip(aa,bb),1):
            for c,(av,bv)in enumerate(zip(ar,br),1):
                cells+=1
                if av!=bv:
                    different.append({'table':t,'row':r,'column':c,'fresh':av,'reference':bv})
    return {'status':'PASS'if not different else'FAIL_PUBLISHED_PRECISION','tables':len(actual),
            'cells':cells,'different_cells':len(different),'examples':different[:12]}


def docx_appendix(path):
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with zipfile.ZipFile(path)as z:
        root=ET.fromstring(z.read('word/document.xml'))
    tables=[[[ ''.join(n.text or ''for n in cell.findall('.//w:t',ns))
                for cell in row.findall('w:tc',ns)]for row in table.findall('w:tr',ns)]
              for table in root.findall('.//w:tbl',ns)]
    if len(tables)!=11:
        raise ValueError('Expected the current 11-table IAOS manuscript')
    return tables[-3:]


def cut16_compare(actual,expected,atol,rtol):
    a=json.loads(actual.read_text())['comparisons'];b=json.loads(expected.read_text())['comparisons']
    def flatten(rows):
        records=[]
        for row in rows:
            z={'group':row['group'],'endpoint':row['endpoint']}
            for key,value in row.items():
                if isinstance(value,(int,float,bool)):
                    z[key]=value
                elif isinstance(value,dict):
                    z.update({key+'__'+k:v for k,v in value.items()})
            records.append(z)
        return pd.DataFrame(records).set_index(['group','endpoint']).sort_index()
    aa,bb=flatten(a),flatten(b)
    if set(aa.index)!=set(bb.index):
        return {'status':'FAIL_SUPPORT'}
    # The new helper retains every field required by Tables S4/S4a. Additional
    # old audit-only details are not presented as newly verified certificates.
    fields=list(aa.columns)
    x,y=aa.to_numpy(float),bb.reindex(aa.index)[fields].to_numpy(float)
    ok=np.isclose(x,y,atol=atol,rtol=rtol,equal_nan=True)
    return {'status':'PASS'if ok.all()else'FAIL_NUMERICAL_TARGET','rows':len(aa),'cells':int(ok.size),
            'different_cells':int((~ok).sum()),'maximum_absolute_error':float(np.nanmax(np.abs(x-y))),
            'old_independent_draw_reconstruction_not_substituted':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root',type=Path,required=True)
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--article',type=Path)
    p.add_argument('--reference-selfcheck',action='store_true',help='Test comparison/formatting mechanics; never certify a fresh rerun')
    p.add_argument('--atol',type=float,default=2e-10)
    p.add_argument('--rtol',type=float,default=1e-8)
    a=p.parse_args();checks={};pending=[]
    for rel,keys in FILES.items():
        actual,expected=a.study_root/rel,a.reference_root/rel
        if not expected.exists():
            pending.append('Missing reference: '+rel)
            continue
        if not actual.exists():
            pending.append('Missing fresh result: '+rel)
            continue
        try:checks[rel]=csv_compare(actual,expected,keys,a.atol,a.rtol)
        except Exception as error:checks[rel]={'status':'FAIL','error':repr(error)}
    try:
        fresh=appendix_tables(a.study_root);old=appendix_tables(a.reference_root)
        checks['Appendix_A_tables_at_published_precision']=compare_tables(fresh,old)
        if a.article:
            checks['Appendix_A_tables_in_current_article']=compare_tables(fresh,docx_appendix(a.article))
    except FileNotFoundError:
        pending.append('Three Appendix A fresh table outputs incomplete')
    except Exception as error:
        checks['Appendix_A_tables_at_published_precision']={'status':'FAIL','error':repr(error)}
    rel=PHASE+'/resultados/diseno/2019_full_official/cut16/cut8_comparison.json'
    if (a.study_root/rel).exists() and (a.reference_root/rel).exists():
        checks['CUT16_Supplement_S4_S4a']=cut16_compare(a.study_root/rel,a.reference_root/rel,a.atol,a.rtol)
    else:pending.append('Fresh CUT16/S4/S4a comparison incomplete')
    failures=[key for key,value in checks.items()if value['status']!='PASS']
    stages=('previous800','cut40','application2025','external2024','cut16')
    ledger=a.study_root.parent
    evidence=all((ledger/f'legacy_{stage}_execution.json').is_file() and
                 json.loads((ledger/f'legacy_{stage}_execution.json').read_text()).get('status')=='ALL_COMMANDS_EXITED_ZERO'
                 for stage in stages)
    evidence=evidence and (ledger/'legacy_initial_setup_completion.json').is_file()
    if not evidence and not a.reference_selfcheck:
        pending.append('Fresh execution ledger incomplete: require every stage receipt and initial source/input preparation certificate')
    result={'status':'INCOMPLETE'if pending else'FAIL_DIFFERENCES_RETAINED'if failures else
             'PASS_REFERENCE_COMPARISON_MECHANICS'if a.reference_selfcheck else'PASS_ALL_LEGACY_NUMERICAL_TARGETS',
            'checks':checks,'pending':pending,'failed_checks':failures,'atol':a.atol,'rtol':a.rtol,
            'elapsed_times_excluded':True,'convergence_and_availability_counts_retained':True,
            'models_refitted_by_this_tool':False,'fresh_execution_evidence_present':bool(evidence),
            'article_reproduction_confirmed':bool(not pending and not failures and evidence and not a.reference_selfcheck),
            'public_data_access_not_same_as_actual_SAV_refitting':True}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k!='checks'},indent=2))
    raise SystemExit(0 if result['article_reproduction_confirmed'] or
                     (a.reference_selfcheck and not pending and not failures) else 2)


if __name__=='__main__':main()
