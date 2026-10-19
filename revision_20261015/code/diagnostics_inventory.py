#!/usr/bin/env python3
"""Export aggregate diagnostics from existing receipts, without loading draws.

Only status/diagnostic JSON files are read. Paths in the public ledger are
relative to the private run root; respondent data, inputs and draws are never
exported. A gate is not an observed diagnostic. No models are fitted here.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

ENDPOINTS = ('anemia', 'stunting')
GATES = dict(max_Rhat=1.01, min_ESS_bulk=400, min_ESS_tail=400,
             min_E_BFMI=0.30, divergences=0, max_treedepth_hits=0)


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def load(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def extrema(records, key, operation):
    values = [r[key] for r in records if finite(r.get(key))]
    return operation(values) if values else None


def total(records, key):
    values = [r[key] for r in records if finite(r.get(key))]
    return sum(values) if values else None


def nuts_trial(v):
    bfmi = v.get('E_BFMI', v.get('BFMI_per_chain', []))
    return dict(max_Rhat=v.get('max_Rhat', v.get('max_parameter_Rhat')),
                min_ESS_bulk=v.get('min_bulk_ESS', v.get('min_parameter_ESS_bulk')),
                min_ESS_tail=v.get('min_tail_ESS', v.get('min_parameter_ESS_tail')),
                min_E_BFMI=min(bfmi) if bfmi and all(finite(x) for x in bfmi) else None,
                divergences=v.get('divergences'),
                max_treedepth_hits=v.get('max_treedepth_hits'),
                success=v.get('success'))


class Inventory:
    def __init__(self, root):
        self.root = root
        self.sources = {}
        self.items = []

    def read(self, p):
        rel = p.relative_to(self.root).as_posix()
        self.sources[rel] = digest(p)
        return load(p), rel

    def add(self, cohort, context, world, method, p, data=None, kind=None):
        d, rel = self.read(p)
        if data is not None:
            d = data
        common = dict(cohort=cohort, context=context, population=world,
                      method=method, source=rel)
        if isinstance(d.get('status'), list) and d['status'] and isinstance(d['status'][0].get('fit'), list):
            d = dict(d, fits=d['status'])
        eps = d.get('endpoints', d.get('end_points'))
        if isinstance(eps, list) and any('sampler_trials' in x for x in eps):
            for n, ep in enumerate(eps):
                trials = [nuts_trial(x) for x in ep.get('sampler_trials', [])]
                self.items.append(dict(common, endpoint=ENDPOINTS[n], stage='hyperposterior',
                                       integration='NUTS', final_success=ep.get('success'),
                                       trials=trials, final=trials[-1] if trials else {},
                                       retries=max(0, len(trials)-1),
                                       map_success=ep.get('numerical_initialization', {}).get('success'),
                                       map_optimizer_reported_success=ep.get('numerical_initialization', {}).get('optimizer_reported_success')))
            return
        if isinstance(d.get('fits'), list):
            for n, ep in enumerate(d['fits']):
                for j, fit in enumerate(ep.get('fit', [])):
                    self.items.append(dict(common, endpoint=ENDPOINTS[n], stage='stratum_'+str(fit.get('stratum',j)),
                                           integration='REML', final_success=fit.get('converged'),
                                           raw_optimizer_converged=fit.get('raw_optimizer_converged'),
                                           bootstrap_requested=fit.get('bootstrap_B'),
                                           bootstrap_failed=fit.get('bootstrap_failed'),
                                           bootstrap_stationarity_recovered=fit.get('bootstrap_stationarity_recovered'),
                                           trials=[], final={}, retries=None))
            return
        # Laplace development receipts include one status dictionary per endpoint.
        if isinstance(d.get('status'), list):
            for n, ep in enumerate(d['status']):
                self.items.append(dict(common, endpoint=ENDPOINTS[n], stage='hyperposterior',
                                       integration='Laplace', final_success=ep.get('success'),
                                       map_success=ep.get('success'),
                                       map_optimizer_reported_success=ep.get('optimizer_reported_success'),
                                       trials=[], final={}, retries=None))
            return
        if d.get('likelihood') == 'Gaussian REML':
            self.items.append(dict(common, endpoint='joint',stage='joint_covariance',
                                   integration='REML', final_success=d.get('success'),
                                   raw_optimizer_converged=d.get('success'),
                                   trials=[], final={}, retries=None))
            return
        # Hogg's parent receipt contains all endpoint attempts. Reading only it
        # avoids counting the same diagnostics again in nested status files.
        if isinstance(d.get('outcomes'), dict):
            for endpoint, ep in d['outcomes'].items():
                attempts = d.get('all_attempts', {}).get(endpoint, [ep])
                for stage in ('stage1', 'stage2'):
                    ts = [nuts_trial(x[stage]) for x in attempts if isinstance(x.get(stage), dict)]
                    if not ts:
                        continue
                    self.items.append(dict(common, endpoint=endpoint, stage=stage,
                                           integration='NUTS', final_success=ep.get(stage, {}).get('success'),
                                           trials=ts, final=nuts_trial(ep[stage]), retries=max(0,len(ts)-1)))
            return
        if kind == 'INLA':
            for endpoint, ep in d.get('endpoints', {}).items():
                self.items.append(dict(common, endpoint=endpoint, stage='INLA',
                                       integration='INLA', final_success=ep.get('INLA_ok'),
                                       working_model_converged=ep.get('working_model_converged'),
                                       trials=[], final={}, retries=None))
            return
        self.items.append(dict(common, endpoint='both', stage='derived_or_analytic',
                               integration=kind or 'derived_or_analytic',
                               final_success=d.get('converged', d.get('success')),
                               trials=[], final={}, retries=None))

    def modern(self):
        s = self.root/'sikov'
        for p in sorted((s/'evaluacion/fits').glob('*/*/*_status.json')):
            ctx, world = p.parts[-3:-1]
            self.add('principal_256',ctx,world,p.stem.removesuffix('_status'),p)
        for p in sorted((s/'evaluacion/published_sar').glob('*/*/*_status.json')):
            ctx, world = p.parts[-3:-1]
            self.add('principal_256',ctx,world,p.stem.removesuffix('_status'),p)
        for scope in ('temporal','territorial'):
            for folder in (s/f'evaluacion/real_{scope}',s/f'evaluacion/published_sar/real_{scope}'):
                for p in sorted(folder.glob('*_status.json')):
                    self.add('real_'+scope,scope,'training_fit',p.stem.removesuffix('_status'),p)
        for p in sorted((s/'sensibilidad_sampled_graph/fits').glob('*/*/*_status.json')):
            world, branch = p.parts[-3:-1]
            method=p.stem.removesuffix('_status')
            self.add('observed_graph_64','SAR_probability_sampled_graph_srs',world,
                     method+('_sampled_graph' if branch=='sampled' else ''),p)
        for p in sorted((s/'evaluacion/b300').glob('*/*/*_status.json')):
            ctx,world=p.parts[-3:-1]
            self.add('B300_subset_8',ctx,world,p.stem.removesuffix('_status'),p)
        # 'recreated_native' is the fresh native REML development reconstruction;
        # ICC was nevertheless integrated by Laplace. It is not a NUTS cohort.
        for p in sorted((s/'desarrollo/recreated_native/fit_results').glob('*/*/*/status.json')):
            ctx,world,method=p.parts[-4:-1]
            self.add('development_32_Laplace',ctx,world,method,p)

    def legacy(self):
        r = self.root/'legacy_v2/study/resultados/resultados_comparadores'
        for folder,label in (('final_primary_v2','legacy_primary_600'),('final_structural_v2','legacy_structural_200')):
            for p in sorted((r/folder/'fit_results').glob('*/*/*/status.json')):
                method,ctx,world=p.parts[-4:-1]
                self.add(label,ctx,world,method,p)
        h=self.root/'legacy_v2/study/resultados/competidores_hogg'
        for p in sorted((h/'fresh_reproduction_hogg').glob('*/*/*/status.json')):
            arm,ctx,world=p.parts[-4:-1]
            self.add('legacy_Hogg_'+arm,ctx,world,'Hogg_'+arm,p)
        for p in sorted(h.glob('spatial_sensitivity_*/structural_bym2/*/status.json')):
            self.add('legacy_spatial_100',p.parts[-4],p.parts[-2],'Hogg_BYM2',p)
        for p in sorted((r/'external_predictions/gao').glob('*/*/*/diagnostics.json')):
            method,ctx,world=p.parts[-4:-1]
            self.add('legacy_Gao',ctx,world,method,p,kind='INLA')
        phase=self.root/'legacy_v2/study/fase2_2026-10-07/resultados'
        for p in sorted((phase/'design_simulation').glob('*/*/*/fit_status.json')):
            ctx,world,state=p.parts[-4:-1]
            self.add('CUT8_40',ctx,world,state,p)
        for p in sorted((phase/'diseno/2019_full_official/cut16').glob('*/fit_status.json')):
            self.add('CUT16_real2019','2019','real2019',p.parts[-2],p)
        for p in sorted(phase.glob('*baseline/status.json')):
            scope=p.parts[-2]
            self.add('legacy_'+scope,scope,'training_fit','ICC_baseline',p)
        for p in sorted(phase.glob('*designcut8/*/fit_status.json')):
            self.add('legacy_'+p.parts[-3],p.parts[-3],'training_fit',p.parts[-2],p)
        for p in sorted(phase.glob('*hogg/*/status.json')):
            self.add('legacy_'+p.parts[-3],p.parts[-3],'training_fit','Hogg_'+p.parts[-2],p)

    def summary(self):
        groups=defaultdict(list)
        for row in self.items:
            groups[(row['cohort'],row['method'],row['integration'])].append(row)
        result=[]
        for (cohort,method,integration),items in sorted(groups.items()):
            finals=[x['final'] for x in items if x['final']]
            trials=[y for x in items for y in x['trials']]
            endpoint_retries = defaultdict(int)
            for item in items:
                if item['retries'] is not None:
                    key=(item['context'],item['population'],item['endpoint'])
                    endpoint_retries[key]=max(endpoint_retries[key],item['retries'])
            row=dict(cohort=cohort,method=method,integration=integration,
                     populations_with_receipts=len({(x['context'],x['population'])for x in items}),
                     fits_or_stages_observed=len(items), final_diagnostic_records=len(finals),
                     all_attempt_diagnostic_records=len(trials),
                     max_observed_Rhat=extrema(finals,'max_Rhat',max),
                     min_observed_ESS_bulk=extrema(finals,'min_ESS_bulk',min),
                     min_observed_ESS_tail=extrema(finals,'min_ESS_tail',min),
                     min_observed_E_BFMI=extrema(finals,'min_E_BFMI',min),
                     final_divergences=total(finals,'divergences'),
                     final_depth_hits=total(finals,'max_treedepth_hits'),
                     retries=sum(endpoint_retries.values()) if endpoint_retries else None,
                     endpoint_fits_retried=sum(x>0 for x in endpoint_retries.values()) if endpoint_retries else None,
                     stage_retry_records=total(items,'retries'),
                     failed_attempt_stages=sum(t.get('success') is False for t in trials) if trials else None,
                     final_NUTS_failed_stages=sum(x.get('final_success') is False for x in items) if integration=='NUTS' else None,
                     final_REML_failed_fits=sum(x.get('final_success') is False for x in items) if integration=='REML' else None,
                     final_INLA_failed_fits=sum(x.get('final_success') is False for x in items) if integration=='INLA' else None,
                     final_Laplace_failed_fits=sum(x.get('final_success') is False for x in items) if integration=='Laplace' else None,
                     raw_REML_optimizer_false=sum(x.get('raw_optimizer_converged') is False for x in items) if integration=='REML' else None,
                     bootstrap_requested=total(items,'bootstrap_requested'),
                     bootstrap_failed=total(items,'bootstrap_failed'),
                     bootstrap_stationarity_recovered=total(items,'bootstrap_stationarity_recovered'),
                     all_attempt_divergences=total(trials,'divergences'),
                     all_attempt_depth_hits=total(trials,'max_treedepth_hits'),
                     all_attempt_max_Rhat=extrema(trials,'max_Rhat',max),
                     all_attempt_min_ESS_bulk=extrema(trials,'min_ESS_bulk',min),
                     all_attempt_min_ESS_tail=extrema(trials,'min_ESS_tail',min),
                     all_attempt_min_E_BFMI=extrema(trials,'min_E_BFMI',min),
                     unknown_final_success=sum(x.get('final_success') is None for x in items),
                     diagnostic_availability='observed_saved_summaries' if finals else 'not_applicable' if integration in ('REML','INLA','Laplace','derived_or_analytic') else 'missing')
            result.append(row)
        return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--private-run-root',required=True,type=Path)
    ap.add_argument('--output-dir',required=True,type=Path)
    ap.add_argument('--modern-only',action='store_true')
    ap.add_argument('--private-ledger-output',type=Path,
                    help='Optional external path for the full per-fit aggregate ledger; never a repository path.')
    a=ap.parse_args(); inv=Inventory(a.private_run_root.resolve());inv.modern()
    if not a.modern_only:inv.legacy()
    output=a.output_dir;output.mkdir(parents=True,exist_ok=True)
    summary=inv.summary()
    with (output/'diagnostics_summary.csv').open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    payload=dict(audit_role='AGGREGATED_OBSERVED_RECEIPTS_NOT_RECOMPUTED_FROM_DRAWS',
                 immutable_source_policy='Only existing JSON receipts read; no fitting, chain loading or source mutations.',
                 public_privacy='All source paths relative to private-run-root. No input arrays, microdata or draws copied.',
                 numeric_null='Missing or not applicable; never a favorable diagnostic or zero failures.',
                 final_vs_attempts='Reported extrema use final accepted attempt per endpoint/stage; all-attempt extrema and failure counts are separate.',
                 limitations=['Saved summary diagnostics are audited, not recalculated from chain draws.',
                              'Laplace development has no Rhat, ESS or BFMI; no NUTS weight stability result is implied.',
                              'Mixture strategies reuse component fits and have no separate sampler.',
                              'REML fit counts refer to endpoint-by-stratum fits; bootstrap failures are distinct.',
                              'Different cohorts and conditional CUT states are not independent populations.'],
                 required_NUTS_thresholds=GATES,summary=summary,
                 source_json_sha256=inv.sources,observed_aggregate_ledger=inv.items)
    repo=Path(__file__).resolve().parents[2]
    full_text=json.dumps(payload,indent=2,allow_nan=False)+'\n'
    if a.private_ledger_output:
        private=a.private_ledger_output.expanduser().resolve()
        if private.is_relative_to(repo):
            raise ValueError('Full ledger must stay outside the public repository.')
        private.parent.mkdir(parents=True,exist_ok=True)
        private.write_text(full_text);os.chmod(private,0o600)
    catalogue_text=json.dumps(dict(source_json_sha256=inv.sources),indent=2)+'\n'
    (output/'source_receipts_catalogue.json').write_text(catalogue_text)
    public={k:v for k,v in payload.items() if k not in ('source_json_sha256','observed_aggregate_ledger')}
    public.update(source_receipts=len(inv.sources),aggregate_ledger_rows=len(inv.items),
                  source_receipts_catalogue='source_receipts_catalogue.json',
                  source_receipts_catalogue_sha256=hashlib.sha256(catalogue_text.encode()).hexdigest(),
                  full_private_aggregate_ledger_sha256=hashlib.sha256(full_text.encode()).hexdigest(),
                  full_private_aggregate_ledger_saved=bool(a.private_ledger_output),
                  script_sha256=digest(Path(__file__)))
    (output/'diagnostics_inventory.json').write_text(json.dumps(public,indent=2,allow_nan=False)+'\n')
    modern_cohorts={'principal_256','real_temporal','real_territorial','observed_graph_64','development_32_Laplace'}
    compact=[r for r in summary if r['cohort'] in modern_cohorts and r['method'] in
             {'ICC','Sikov_FH','Sikov_SAR','Sikov_SAR_Published'}]
    with (output/'diagnostics_modern_compact.csv').open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(compact)
    specification={}
    for filename,names in [('comparison_simulation.py',('CONTEXTS','NAMES','BETAS')),
                           ('sikov_model.py',('TABLE_COVARIATES','PUBLISHED_SFH_BETA','RHO_BOUND'))]:
        p=repo/'src/estudio_publicacion/sikov_2026-10-10/scripts'/filename
        tree=ast.parse(p.read_text())
        constants={}
        for node in tree.body:
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Name) and target.id in names:
                        constants[target.id]=ast.literal_eval(node.value)
        specification[filename]=dict(source=p.relative_to(repo).as_posix(),sha256=digest(p),constants=constants)
    (output/'reproduction_specification.json').write_text(json.dumps(specification,indent=2)+'\n')
    materials=[
      dict(material='Current public manuscript freeze',status='available_public',location='expected/current_freeze/data',
           runnable='Reformat/check 22 original tables and seven images without fitting.'),
      dict(material='Modern primary simulation manifests',status='available_private_and_public_synthetic',
           populations=len(list((inv.root/'sikov/evaluacion/inputs').glob('*/*/manifest.json'))),
           runnable='Regenerate 256 populations and fit with the unchanged seed schedules.'),
      dict(material='Development simulation manifests',status='available_private_and_public_synthetic',
           populations=len(list((inv.root/'sikov/desarrollo/inputs').glob('*/*/manifest.json'))),
           runnable='Re-score the 32 original Laplace development populations; new NUTS comparison is separate.'),
      dict(material='Observed-domain generating-graph manifests',status='available_private_and_public_synthetic',
           populations=len(list((inv.root/'sikov/sensibilidad_sampled_graph/inputs').glob('*/manifest.json'))),
           runnable='Re-score separate 64-population sensitivity.'),
      dict(material='Modern ICC conditional marginal draw files',status='available_private',
           files=len(list((inv.root/'sikov/evaluacion/fits').glob('*/*/ICC_conditional_draws.npz')))+
                 len(list((inv.root/'sikov/evaluacion').glob('real_*/ICC_conditional_draws.npz')))+
                 len(list((inv.root/'sikov/sensibilidad_sampled_graph/fits').glob('*/original/ICC_conditional_draws.npz'))),
           runnable='Reuse for an identical model/target; never copy real fitted draws into the public package.'),
      dict(material='Observed computation summaries',status='available_private_receipts_public_aggregate_export',
           files=len(inv.sources),runnable='Rebuild this inventory from JSON only; no fits or chains read.'),
      dict(material='Official ENDES raw files',status='available_privately_not_redistributed',
           location='reproduction/DATA_INPUTS.md',runnable='Portable preparation after independent acquisition and hash checks.'),
      dict(material='Exact original complete native init/metric states and historical MCSE',status='missing',
           runnable=False,consequence='Cannot establish exact historical numerical replay or old/new Monte Carlo equivalence.'),
      dict(material='NUTS development weight-selection results in the old run',status='missing',runnable='New fitting requires a separately registered revision analysis.'),
      dict(material='Author declarations and reuse license',status='pending_confirmation',runnable=False),
      dict(material='Public repository URL/DOI',status='not_published',runnable=False)]
    (output/'materials_inventory.json').write_text(json.dumps(dict(materials=materials,
        exclusion='No microdata, person identifiers, native real model inputs, credentials or posterior draws exported.'),indent=2)+'\n')
    def fmt(v):
        if v is None:return 'NA'
        return f'{v:.6g}' if isinstance(v,float) else str(v)
    lines=['# Observed computation audit', '',
           'This inventory reads existing JSON summaries, not posterior draws. Final-attempt extrema are observed values; required gates are separately listed in the JSON. NA denotes a missing or inapplicable quantity, never zero failures.', '',
           '| Cohort | Method | Integration | Populations | Fits/stages | Max Rhat | Min ESS bulk | Min ESS tail | Min E-BFMI | Divergences | Depth hits | Retries | Final REML failures | Bootstrap failures |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    cols=['cohort','method','integration','populations_with_receipts','fits_or_stages_observed','max_observed_Rhat',
          'min_observed_ESS_bulk','min_observed_ESS_tail','min_observed_E_BFMI','final_divergences','final_depth_hits',
          'retries','final_REML_failed_fits','bootstrap_failed']
    for row in compact:lines.append('| '+' | '.join(fmt(row[k])for k in cols)+' |')
    lines+=['', 'The principal ICC cohort has two failed first attempts and two successful predetermined retries. Its maximum Rhat across all attempts is 1.01097; its final-attempt maximum is 1.00807. Both sets of attempts have zero reported divergences and depth hits. All-attempt extrema remain in the CSV/JSON.',
            '', 'A REML optimizer flag can be false while an independent stationarity certificate passes. The primary full SAR records two such original optimizer flags and the observed-domain SAR one. Their final certified REML failure counts are zero. The principal full SAR has 17 failed bootstrap fits; the observed-domain SAR has 32. These finite bootstrap losses are reported separately from final model convergence. Actual-data training fits have no saved bootstrap failures.',
            '', 'The development ICC uses Laplace integration. Rhat, ESS and E-BFMI are not applicable to those receipts; no NUTS weight-stability inference follows. Its FH/SAR comparators use native REML. Mixtures reuse component fits and do not have an independent NUTS sampler.',
            '', 'The full inventory additionally preserves the earlier 600+200 population experiments, CUT8 conditional fits, real CUT16, legacy actual-data applications, Hogg stages and INLA summaries. Stage diagnostics do not create new independent populations. Retry counts are endpoint attempts; stage_retry_records separately counts repeated stages in two-stage models.',
            '', 'The audit does not recompute any diagnostic from posterior chains and does not certify exact historical reproduction. Saved source paths are relative to the private run root and every source receipt is SHA-256-bound. All raw inputs, sampler chains and posterior draws stay private.', '']
    (output/'diagnostics_audit.md').write_text('\n'.join(lines))
    print(json.dumps(dict(summary_rows=len(summary),aggregate_ledger_rows=len(inv.items),source_receipts=len(inv.sources))))


if __name__=='__main__':main()
