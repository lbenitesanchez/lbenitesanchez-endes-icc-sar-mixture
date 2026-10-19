"""Independent ICAR-precision NUTS reference for the Hogg BYM2 S2 arm.

Uses an existing development S1 posterior and its identical modular summary.
No final population, population outcome or performance metric enters this check.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import hogg_tsln as h
import hogg_spatial as sp


def sampled_summary(summary,reference=False):
    names=summary.index.to_series().astype(str)
    latent='spatial_free['if reference else'spatial_z['
    keep=names.str.startswith(('intercept','beta[','sigma_area','rho',latent,'omega[','sigma_gvf'))
    return summary.loc[keep.values]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default='spatial_pilot')
    args=parser.parse_args()
    output=h.ROOT/'resultados/competidores_hogg'/args.tag
    if output.exists():raise FileExistsError('Preserve pilot attempts; use an explicit new validation tag for a repeated check')
    output.mkdir(parents=True)
    inputs=h.ROOT/'resultados/resultados_comparadores/inputs/sparse_psu_2025/rep_00000/model_inputs'
    source=h.ROOT/'resultados/competidores_hogg/pilot_sparse_psu_2025_stunting'
    codes,data,recovery=sp.recover_stage2_from_saved(source,inputs/'sample.csv',inputs/'area.csv',
        outcome='y_stunting',weight='weight_stunting',area_features=['X1','X2'],population_N='N_population')
    np.savez_compressed(output/'source_modular_summary_stunting.npz',area=codes,**data)
    h.write_json(output/'source_recovery.json',recovery)
    graph,metadata,reference=sp.graph_factor(codes,inputs/'adjacency_edges.csv')
    integration=sp.fixed_hyper_gaussian_check(graph['scaled_icar_factor'])
    rng=np.random.default_rng(20261005)
    latent=rng.normal(size=graph['R']);indices=data['sampled_area']-1
    row_error=float(abs((graph['scaled_icar_factor']@latent)[indices]-
        graph['scaled_icar_factor'][indices]@latent).max())
    h.write_json(output/'algebra_preflight.json',{'graph':metadata,'integration':integration,
        'observed_row_extraction_maxerror':row_error,'passed':integration['success']and row_error<1e-12})
    if not integration['success']or row_error>=1e-12:raise FloatingPointError('Exact algebra check failed')
    # Compile once, serially, in this single CPU allocation.
    marginal=sp.compile_spatial();ref=sp.compile_spatial(reference=True)
    cfg=h.MCMCConfig(parallel_chains=1,phase='spatial_development_pilot')
    result=sp.fit_spatial_stage2(codes,data,inputs/'adjacency_edges.csv',
        output_dir=output/'whitened',config=cfg,model=marginal)
    if not result['status']['success']:
        retry=h.MCMCConfig(**{**h.asdict(cfg),'warmup':1000,'samples':2000,'adapt_delta':.99})
        result=sp.fit_spatial_stage2(codes,data,inputs/'adjacency_edges.csv',
            output_dir=output/'whitened_retry',config=retry,model=marginal)
    reference_cfg=h.MCMCConfig(parallel_chains=1,warmup=1000,samples=2000,
        adapt_delta=.99,seed=cfg.seed+91,phase='spatial_precision_reference')
    start=time.monotonic()
    full=h._sample(ref,{**data,**reference},reference_cfg,output/'icar_precision_reference')
    full_summary=full.summary();full_summary.to_csv(output/'reference_parameter_summary.csv')
    full_diagnostics=h.diagnose_fit(full,reference_cfg,summary=sampled_summary(full_summary,True))
    marginal_summary=result['fit'].summary()
    params=['intercept']+[f'beta[{j+1}]'for j in range(data['P'])]+['sigma_area','rho',
        'omega[1]','omega[2]','sigma_gvf']+[f'prevalence[{i+1}]'for i in range(data['M'])]
    checked=[]
    for name in params:
        a,b=marginal_summary.loc[name],full_summary.loc[name]
        mcse=float(np.sqrt(a.MCSE**2+b.MCSE**2))
        checked.append({'parameter':name,'whitened_mean':float(a.Mean),'precision_mean':float(b.Mean),
            'combined_MCSE':mcse,'absolute_difference_in_combined_MCSE':float(abs(a.Mean-b.Mean)/mcse)})
    maxdiff=max(x['absolute_difference_in_combined_MCSE']for x in checked)
    report={'source_recovery':recovery,'diagnostics_whitened':result['status']['stage2'],
        'diagnostics_precision_reference':full_diagnostics,'reference_seconds':time.monotonic()-start,
        'posterior_mean_checks':checked,'max_absolute_mean_difference_in_combined_MCSE':maxdiff,
        'prespecified_gate':'All comparable posterior means within 4 combined MCSE and both samplers satisfy Rhat<=1.01, bulk/tailESS>=400, zero divergences/depth hits, all-chain BFMI>=.3. Supported by exact algebra and an independent dense review.',
        'scientific_scope':'Development numerical check only; not performance selection, full Australian replication or a coverage guarantee.',
        'passed':bool(result['status']['success']and full_diagnostics['success']and maxdiff<=4)}
    h.write_json(output/'posterior_equivalence.json',report)
    print(json.dumps({'passed':report['passed'],'max_combined_MCSE':maxdiff}),flush=True)


if __name__=='__main__':main()
