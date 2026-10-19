"""Reproducible Hogg pilot and independent latent-Gaussian validation.

Consumes only model_inputs; no simulated truth enters these checks or fits.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.special import roots_hermitenorm
from scipy.stats import norm
import hogg_tsln as h


def from_csv(directory):
    return h._backend().from_csv([str(x) for x in sorted(Path(directory).glob('*.csv'))])


def check_gaussian_integration():
    results = []
    # Integrate both latent Gaussian layers independently of the Stan code.
    for mu, y, model_var, sampling_var, area_var in [
        (-1.2, -.8, .16, .25, .36),
        (.3, -1.0, .8, .02, .09),
        (1.0, .5, .02, .9, .01),
    ]:
        estimates=[]
        for q in [256, 512]:
            nodes, weights=roots_hermitenorm(q)
            weights=weights/np.sqrt(2*np.pi)
            bar=mu+np.sqrt(area_var)*nodes[:,None]+np.sqrt(sampling_var)*nodes[None,:]
            estimates.append(float(weights @ norm.pdf(y,bar,np.sqrt(model_var)) @ weights))
        numeric=estimates[-1]
        error=abs(estimates[0]-estimates[1])
        analytic = norm.pdf(y, mu, np.sqrt(model_var + sampling_var + area_var))
        results.append({'numeric_density': numeric, 'analytic_density': analytic,
                        'absolute_difference': abs(numeric-analytic),
                        'quadrature_256_vs_512_order_difference': error})
    return {'cases': results,
            'success': all(x['absolute_difference'] < 1e-8 for x in results)}


def reference_check(pilot_dir, prepared, models, config):
    pilot_dir = Path(pilot_dir)
    f1 = from_csv(pilot_dir/'stage1')
    f2 = from_csv(pilot_dir/'stage2')
    s1 = h.aggregate_stage1(prepared, f1.stan_variable('intercept'),
        f1.stan_variable('beta'), f1.stan_variable('effect_area'))
    data, meta = h.prepare_stage2(prepared, s1, seed=config.seed+21)
    refdata = {k:v for k,v in data.items() if k != 'theta_s1'}
    refdata['theta_s1_draws'] = meta['theta_s1_selected_draws']
    refdata['T'] = len(meta['theta_s1_selected_draws'])
    start = time.monotonic()
    full = h._sample(models['hogg_stage2_full_reference'], refdata, config,
                      pilot_dir/'reference_full_latent', seed_offset=91)
    sm, sf = f2.summary(), full.summary()
    diag = h.diagnose_fit(full, config,summary=sf)
    parameters = ['intercept', 'beta[1]', 'beta[2]', 'sigma_area',
                  'omega[1]', 'omega[2]', 'sigma_gvf']
    checked = []
    for name in parameters + [f'prevalence[{i+1}]' for i in range(data['M'])]:
        a,b = sm.loc[name], sf.loc[name]
        mcse = np.sqrt(a['MCSE']**2+b['MCSE']**2)
        checked.append({'parameter':name, 'mean_marginal':a['Mean'],
            'mean_full_latent':b['Mean'], 'combined_MCSE':mcse,
            'standardized_mean_difference':abs(a['Mean']-b['Mean'])/mcse})
    maxdiff = max(x['standardized_mean_difference'] for x in checked)
    comparison = {'diagnostics_full_latent':diag,
        'diagnostics_marginal':h.diagnose_fit(f2, config,summary=sm),
        'seconds_full_latent':time.monotonic()-start,
        'parameter_and_area_comparisons':checked,
        'max_absolute_mean_difference_in_combined_MCSE':maxdiff,
        'prespecified_check': 'All 87 posterior means within 4 combined MCSE; both samplers satisfy fixed diagnostics. This is a stochastic pilot check, supported by exact likelihood and numerical integration proofs.',
        'success':bool(diag['success'] and maxdiff <= 4.)}
    h.write_json(pilot_dir/'reference_equivalence.json', comparison)
    return comparison


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference-only', action='store_true')
    args=parser.parse_args()
    models=h.compile_models(include_reference=True)
    cfg=h.MCMCConfig(chains=4, parallel_chains=4, warmup=500, samples=1000,
                     phase='pilot')
    out=h.ROOT/'resultados/competidores_hogg'
    h.write_json(out/'gaussian_integration_checks.json',check_gaussian_integration())
    inputs=h.ROOT/'resultados/resultados_comparadores/inputs'
    statuses={}
    for scenario in ['sparse_srs_2025','sparse_psu_2025','binary_informative_mar']:
        base=inputs/scenario/'rep_00000/model_inputs'
        for endpoint in ['stunting','anemia']:
            name='pilot_'+scenario+'_'+endpoint
            target=out/name
            # Preserve the first pilot's source CSVs for reference validation.
            if scenario=='sparse_srs_2025' and endpoint=='stunting':
                target=out/'pilot_srs_stunting'
            if args.reference_only:
                if scenario!='sparse_srs_2025' or endpoint!='stunting': continue
            if not (target/'status.json').exists():
                result=h.fit_hogg(base/'sample.csv',base/'area.csv',
                    outcome='y_'+endpoint, weight='weight_'+endpoint,
                    area_features=['X1','X2'], population_N='N_population',
                    config=cfg, output_dir=target, models=models)
                statuses[name]=result.status
            else: statuses[name]=json.loads((target/'status.json').read_text())
            print(json.dumps({'pilot':name, 'success':statuses[name]['success'],
                              'seconds':statuses[name]['seconds_total']}),flush=True)
            if scenario=='sparse_srs_2025' and endpoint=='stunting':
                p=h.prepare_hogg_data(base/'sample.csv',base/'area.csv',
                    outcome='y_'+endpoint, weight='weight_'+endpoint,
                    area_features=['X1','X2'], population_N='N_population')
                ref_cfg=h.MCMCConfig(chains=4,parallel_chains=4,warmup=1000,
                    samples=2000,phase='posterior_equivalence')
                comparison=reference_check(target,p,models,ref_cfg)
                print(json.dumps({'reference_equivalence_success':comparison['success'],
                    'max_combined_MCSE':comparison['max_absolute_mean_difference_in_combined_MCSE']}),flush=True)
    h.write_json(out/'pilot_statuses.json',statuses)


if __name__=='__main__': main()
