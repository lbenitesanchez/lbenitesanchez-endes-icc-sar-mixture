"""Numerical table construction from aggregate matrices only.

Code-only publication adapter. Original table functions are retained unchanged;
Word/XML reading, manuscript text, document comparisons and editorial CLI are
excluded. Input data and generated table matrices are supplied externally.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import pandas as pd
from legacy_compare import appendix_tables

PHASE = 'fase2_2026-10-07'

LEGACY_LABELS = {'baseline':'ICC beta','point':'Variance point','cut8':'Variance CUT8',
                 'BetaDesignUncertainty':'Variance CUT8','BetaPowerExisting':'ICC beta',
                 'HoggAgeSexSpatial':'Hogg age/sex BYM2'}

SUPPLEMENT_IDS = ('S1','S2','S3','S3_contrasts','S4','S4a','S4b','S5','S6','S7','S7b')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_report(source):
    path = source / 'sikov_2026-10-10/scripts/report_documents.py'
    spec = importlib.util.spec_from_file_location('iaos_coverage_report', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path

def main_inputs(root):
    directory = root / 'resultados/published_sar_comparison'
    paired = pd.read_csv(directory / 'paired_balanced_contrasts_Holm3.csv')
    physical = pd.read_csv(directory / 'physical_projection_contrasts.csv')
    paths = {
        'summary':directory / 'summary.csv',
        'external_paired':root / 'resultados/external_comparison_Holm12/paired_balanced_contrasts_Holm12.csv',
        'B300_summary':root / 'validacion/sikov_B300_descriptive_balanced.csv',
        'published_B300_summary':root / 'resultados/published_b300_descriptive/summary.csv',
        'published_B300_differences':root / 'resultados/published_b300_descriptive/selected_reference_differences.csv',
        'poverty_differences':root / 'resultados/poverty_stratum_descriptive/selected_reference_descriptive_differences.csv',
        'sampled_graph_summary':root / 'sensibilidad_sampled_graph/resultados/descriptive_summary.csv',
        'sampled_graph_contrasts':root / 'sensibilidad_sampled_graph/resultados/descriptive_paired_contrasts.csv'}
    return {'primary':paired[paired.metric.eq('interval_score')],
            'physical':physical[physical.metric.eq('interval_score')],
            **{key:pd.read_csv(path) for key,path in paths.items()}}

def main_tables(report, data):
    external = [[p.scope,report.LABELS.get(p.reference,p.reference),p.metric,
                 report.f(p.difference,5),'['+report.f(p.unadjusted_t_ci_low,5)+', '+
                 report.f(p.unadjusted_t_ci_high,5)+']',report.sci(p.p_Holm12)]
                for p in data['external_paired'].itertuples()]
    physical = [[report.LABELS[p.reference],report.f(p.difference,4),
                 '['+report.f(p.MC_CI95_low,4)+', '+report.f(p.MC_CI95_high,4)+']']
                for p in data['physical'].itertuples()]
    definitions = (
        (['Estimator','Outcome','RMSE','Coverage','Width','IS95'],report.aggregate_rows(data)),
        (['Reference','Difference','95% MC interval','Holm p'],report.primary_rows(data)),
        (['Scope','Reference','Metric','Difference','95% cluster interval','Holm p'],external),
        (['Reference','Difference','95% MC interval'],physical),
        (['Estimator','Outcome','IS95','Coverage %','Width pp'],report.BC_rows(data)),
        (['Estimator','IS95','Coverage %','Width pp'],report.B300_rows(data)),
        (['Stratum','Reference','All districts','Observed endpoint','Outside sample'],report.poverty_rows(data)),
        (['Strategy','RMSE pp','Coverage %','Width pp','IS95'],report.sampled_graph_rows(data)))
    return [[headers,*rows] for headers,rows in definitions]

def supplement_tables(root):
    phase = root / PHASE
    csv = lambda p:pd.read_csv(p,dtype={'ubigeo':str})
    percent = lambda value,n=2:f'{100*float(value):.{n}f}'
    finite_number = lambda value,n:f'{float(value):.{n}f}' if pd.notna(value) else 'NA'
    tables = appendix_tables(root)
    simulation,external,application = tables
    paired = csv(phase/'resultados/design_simulation/final/paired_diagnostics.csv')
    summary = csv(phase/'resultados/externa_final/summary.csv')
    comparison = json.loads((phase/'resultados/diseno/2019_full_official/cut16/cut8_comparison.json').read_text())['comparisons']
    earlier = csv(root/'resultados/resultados_comparadores/final_evaluation/primary_balanced_interval_score.csv')
    additional = csv(root/'resultados/resultados_comparadores/final_evaluation/additional_arms_summary.csv')
    additional = additional[additional.group.eq('all') & additional.method.isin(
        ['HoggAgeSexSpatial','HoggSpatial','GaoAuthorStrictFullH','HoggTSLNFullH'])]
    structural = csv(root/'resultados/resultados_comparadores/final_evaluation/structural_summary.csv')
    structural = structural[structural.group.eq('all')]
    spatial = csv(root/'resultados/resultados_comparadores/spatial_evaluation/summary.csv')
    spatial = spatial[spatial.group.eq('all')]
    standard = lambda frame:[['Context','Estimator','Outcome','Worlds','RMSE (pp)','Coverage (%)','Width (pp)','IS95']]+[
        [r.scenario,r.method,r.outcome,str(r.replications),percent(r.rmse),percent(r.coverage,1),
         percent(r.width),f'{r.interval_score:.3f}'] for r in frame.itertuples()]
    return [simulation,
        [['Context','Outcome','Reference','Metric','Difference','MCSE']]+[
            [r.context,r.outcome,LEGACY_LABELS[r.reference],r.metric,f'{r.cut8_minus_reference:.5f}',
             f'{r.paired_MCSE:.5f}'] for r in paired.itertuples()],
        [['Scope','Outcome','Estimator','Districts','Provinces','NLL','Noisy RMSE','Brier']]+[
            [r.scope,r.outcome,LEGACY_LABELS[r.method],str(r.districts),str(r.provinces),
             f'{r.binary_NLL:.4f}',f'{r.RMSE_noisy_direct_pp:.2f}',f'{r.Brier_binary_expectation:.4f}']
            for r in summary.itertuples()],external,
        [['Support','Outcome','Districts','Median |Δmean|','Max |Δmean|','Median Δwidth','Spearman']]+[
            [r['group'],r['endpoint'],str(r['districts']),finite_number(r['median_absolute_mean_difference_pp'],4),
             finite_number(r['maximum_absolute_mean_difference_pp'],3),
             finite_number(r['median_width_difference_sensitivity_minus_primary_pp'],3),
             finite_number(r['spearman_posterior_mean'],6)] for r in comparison],
        [['Support','Outcome','Variance 8','Variance 16','Between share 8 (%)','Between share 16 (%)']]+[
            [r['group'],r['endpoint'],finite_number(r['primary_absolute_variance_probability_squared']['0.5'],5),
             finite_number(r['sensitivity_absolute_variance_probability_squared']['0.5'],5),
             percent(r['primary_between_state_variance_share']['0.5'],3),
             percent(r['sensitivity_between_state_variance_share']['0.5'],3)] for r in comparison],
        [['Estimator','Outcome','Districts','Mean (%)','Width (pp)','Between (%)'],*application[1:]],
        [['Reference','Score change (%)','Difference','Paired MCSE','Holm3 p']]+[
            [r.reference,f'{r.relative_score_change_pct:.2f}',f'{r.interval_score_change:.4f}',
             f'{r.paired_mcse:.4f}','<1e−300' if r.p_Holm3==0 else f'{r.p_Holm3:.3g}']
            for r in earlier.itertuples()],standard(additional),standard(structural),standard(spatial)]
