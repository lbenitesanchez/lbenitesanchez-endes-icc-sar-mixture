"""Build current diagnostic prose only from closed corrected fit evidence."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return list(csv.DictReader(path.open()))


def build(repo, layer):
    repo, layer = Path(repo).resolve(), Path(layer).resolve()
    out = layer/'results/diagnostics'
    source = out/'diagnostics_summary.csv'
    newer = layer/'results/numerical_cohorts/observed_numerical_diagnostic_summary.csv'
    old_index = {(x['cohort'],x['method']):x for x in rows(source)}
    new_index = {(x['cohort'],x['method']):x for x in rows(newer)}
    def number(row, key, fmt='g'):
        raw = row.get(key)
        if raw in ('',None) or not math.isfinite(float(raw)):
            return 'NA'
        return format(float(raw),fmt)
    def integer(row,key):
        return number(row,key,'.0f')
    def pair(row, x, y, fmtx, fmty):
        return number(row,x,fmtx)+' / '+number(row,y,fmty)
    table = [['Cohort / strategy','Fits','Max Rhat / min E-BFMI',
              'Min ESS bulk / tail','Div. / depth','Retry / NUTS fail',
              'REML fail','Bootstrap fail']]
    selected = [
        ('development_32_Laplace','ICC','Development32 / ICC Laplace'),
        ('principal_256','ICC','Principal256 / ICC'),
        ('observed_graph_64','ICC','Graph64 / ICC'),
        ('real_temporal','ICC','Real2019 / ICC'),
        ('real_territorial','ICC','Real2024 / ICC'),
        ('development_32_Laplace','Sikov_FH','Development32 / FH'),
        ('development_32_Laplace','Sikov_SAR','Development32 / full SAR'),
        ('principal_256','Sikov_FH','Principal256 / FH'),
        ('principal_256','Sikov_SAR','Principal256 / full SAR'),
        ('principal_256','Sikov_SAR_Published','Principal256 / observed SAR'),
        ('observed_graph_64','Sikov_FH','Graph64 / FH'),
        ('observed_graph_64','Sikov_SAR','Graph64 / full SAR'),
        ('observed_graph_64','Sikov_SAR_Published','Graph64 / observed SAR')]
    for cohort,method,label in selected:
        x = old_index[cohort,method]
        table.append([label,integer(x,'fits_or_stages_observed'),
            pair(x,'max_observed_Rhat','min_observed_E_BFMI','.5f','.3f'),
            pair(x,'min_observed_ESS_bulk','min_observed_ESS_tail','.1f','.1f'),
            pair(x,'final_divergences','final_depth_hits','.0f','.0f'),
            pair(x,'retries','final_NUTS_failed_stages','.0f','.0f'),
            integer(x,'final_REML_failed_fits'),integer(x,'bootstrap_failed')])
    for cohort,label in [('real_temporal','Real2019 / FH + both SAR'),
                         ('real_territorial','Real2024 / FH + both SAR')]:
        local = [old_index[cohort,m] for m in ('Sikov_FH','Sikov_SAR','Sikov_SAR_Published')]
        total = lambda key:str(sum(int(float(x[key])) for x in local))
        table.append([label,total('fits_or_stages_observed'),'NA','NA','NA','NA',
                      total('final_REML_failed_fits'),total('bootstrap_failed')])
    selected_new = [
        ('development_nuts','ICC','Development32 NUTS / ICC',64),
        ('development_nuts','ICC_poverty','Development32 NUTS / poverty ICC',64),
        ('sparse57','ICC','NewSparse256 / ICC NUTS',512),
        ('sparse57','Sikov_FH','NewSparse256 / FH',1536),
        ('sparse57','Sikov_SAR','NewSparse256 / full SAR',1536),
        ('sparse57','Sikov_SAR_Published','NewSparse256 / observed SAR',1536)]
    for cohort,method,label,expected in selected_new:
        x = new_index[cohort,method]
        field = 'observed_NUTS_endpoints' if method in ('ICC','ICC_poverty') else 'REML_stratum_fits_observed'
        assert int(float(x[field]))==expected,(cohort,method)
        table.append([label,integer(x,field),
            pair(x,'max_Rhat_final','min_E_BFMI_final','.5f','.3f'),
            pair(x,'min_ESS_bulk_final','min_ESS_tail_final','.1f','.1f'),
            pair(x,'divergences_final','treedepth_hits_final','.0f','.0f'),
            pair(x,'NUTS_retried_endpoints','NUTS_failed_final_endpoints','.0f','.0f'),
            integer(x,'REML_final_failures'),integer(x,'REML_bootstrap_failed')])
    audit_root = repo/'revision_20261015/results/reml_correction'
    verification_path = audit_root/'corrected_verification.json'
    verification = json.loads(verification_path.read_text())
    assert verification['status']=='PASS_ALL_CORRECTED_METHODS_VERIFIED'
    assert verification['verified_jobs']==verification['planned_jobs']==2780
    assert verification['central_fits_verified']==16680
    assert verification['bootstrap_requested']==1107600
    for name,digest in verification['outputs_sha256'].items():
        assert sha(audit_root/name)==digest,name
    for name in ('corrected_verification.json','corrected_method_diagnostic_summary.csv',
                 'corrected_central_REML_stratum_fits.csv','corrected_bootstrap_stratum_summary.csv'):
        shutil.copy2(audit_root/name,out/name)
    method_rows = rows(out/'corrected_method_diagnostic_summary.csv')
    aggregate = defaultdict(lambda:defaultdict(int))
    keys = ('central_fits','central_rescued','bootstrap_requested','bootstrap_rescued','bootstrap_failed')
    for row in method_rows:
        for key in keys:aggregate[row['cohort']][key]+=int(float(row.get(key) or 0))
    labels = {'B300_subset_8':'B300 subset of 8','development_32':'Development32',
              'observed_graph_64':'Graph64','principal_256':'Principal256',
              'real_temporal':'Real temporal','real_territorial':'Real territorial','sparse57':'NewSparse256'}
    correction_table = [['Cohort','Central fits','Central rescued','Bootstrap draws','Bootstrap rescued','Bootstrap rejected']]
    for cohort in ('development_32','principal_256','sparse57','observed_graph_64',
                   'B300_subset_8','real_temporal','real_territorial'):
        correction_table.append([labels[cohort]]+[str(aggregate[cohort][key]) for key in keys])
    central = rows(out/'corrected_central_REML_stratum_fits.csv')
    bad_central = sum(x['final_certificate_passed']!='True' or x['final_stable_certificate_passed']!='True' for x in central)
    assert bad_central==0
    total_rescued = sum(x['central_rescued'] for x in aggregate.values())
    total_boot_rescued = sum(x['bootstrap_rescued'] for x in aggregate.values())
    rejected = verification['bootstrap_failed']
    main = (f'The numerical correction verified {len(central):,} central FH/SAR stratum fits and '
        f'{verification["bootstrap_requested"]:,} fixed bootstrap refits across the reconstructed model cohorts and sensitivities. '
        f'It recovered {total_rescued:,} central and {total_boot_rescued:,} bootstrap solutions; '
        f'{bad_central} central fits remained noncertified and {rejected} bootstrap attempts were rejected. '
        'Every retained bootstrap fit passes both equivalent projected-score calculations at tolerance 0.00001. '
        'These are numerical fit counts, not additional independent populations. Original ICC draws and NUTS diagnostics were reused without resampling.')
    p = old_index['principal_256','ICC']
    nuts = (f'The principal ICC final maximum Rhat is {number(p,"max_observed_Rhat",".5f")}, '
        f'minimum bulk/tail ESS {number(p,"min_observed_ESS_bulk",".1f")}/{number(p,"min_observed_ESS_tail",".1f")}, '
        f'and minimum E-BFMI {number(p,"min_observed_E_BFMI",".3f")}. '
        f'The saved NUTS records contain {integer(p,"retries")} retries and '
        f'{integer(p,"final_NUTS_failed_stages")} final failed stages. '
        'These archived sampler summaries are distinct from the newly repeated Gaussian numerical work.')
    detail = ('The same-model correction preserves all covariates, input hashes, graphs, parameter bounds, '
        'bootstrap sizes, seeds and generation order. Historical fitter flags include the explicit zero-variance '
        'boundary rule and are reported independently of final score certificates. The deterministic profile '
        'uses residual contrasts and an additional local score-root recovery for remaining nonstationary fits. '
        'The score uses the original variance coordinate A/s, with s = max{var(y), mean(D), 0.00001}, '
        '0 <= A/s <= 100 and -0.999 <= rho <= 0.999; active-bound score components are projected as in the original certificate. '
        'At A = 0, rho is unidentified and represented by zero. '
        'Rejected draws are never rerolled. The original minimum bootstrap acceptance fraction remains 95%. '
        'Naive and bias-corrected MSE use the same accepted draws; negative bias-corrected estimates remain '
        'visible and are floored only when taking interval square roots.')
    legacy = ('The earlier 800-population multivariate Gaussian models use a separate covariance '
        'parameterization and retain their archived convergence records. They are historical supporting '
        'evidence and are not relabeled as passing the new univariate FH/SAR certificates. The earlier '
        'CUT8 experiment uses its own posterior computation. The 32 development populations are reused '
        'for the NUTS integration sensitivity, and the eight B300 populations are a subset; neither '
        'reuse adds independent simulation populations.')
    references = [source,newer,out/'corrected_verification.json',out/'corrected_method_diagnostic_summary.csv',
                  out/'corrected_central_REML_stratum_fits.csv',out/'corrected_bootstrap_stratum_summary.csv']
    obj = {'status':'COMPLETE_CORRECTED_NUMERICAL_DIAGNOSTICS',
        'main_paragraphs':[nuts,main],'supplement_paragraphs':[nuts,main,detail,legacy],
        'table':table,'table_note':'NA means unavailable or inapplicable. NUTS rows are endpoint fits; Gaussian rows are endpoint-by-stratum fits. Rhat/ESS/E-BFMI use final sampler attempts. REML failures use strict final score certificates; bootstrap failures count rejected refits. Reused development references are counted once. Historical flags and all attempts remain separately available. Thresholds are Rhat <= 1.01, bulk/tail ESS >= 400, E-BFMI >= 0.30, and zero divergences/depth hits.',
        'numerical_correction_table':correction_table,
        'numerical_correction_table_note':'Each central fit is an outcome-by-poverty-stratum fit. Bootstrap draws are refits, not independent populations. Counts include fixed graph variants and the B300 subset. Every retained draw passes original and residual-contrast projected-score certificates with tolerance 0.00001; the two calculations represent the same REML likelihood. Original model definitions, B, seeds and the 95% minimum acceptance fraction are unchanged.',
        'inputs':[str(x.relative_to(repo)) for x in references],
        'inputs_sha256':{str(x.relative_to(repo)):sha(x) for x in references},
        'source_sha256':sha(Path(__file__)),
        'corrected_central_noncertified':bad_central,'corrected_bootstrap_rejected':rejected}
    (out/'editorial_contract.json').write_text(json.dumps(obj,indent=2)+'\n')
    return obj


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--layer',type=Path,required=True)
    args=ap.parse_args();obj=build(args.repo,args.layer)
    print(json.dumps({'status':obj['status'],'correction_table_rows':len(obj['numerical_correction_table'])}))
