"""Experimental Hogg design-variance arm with explicit unreliable-domain mask.

The frozen primary generic estimator is unchanged. This arm keeps its S1 priors
and uses the identity psiS1=2D_design when all unit covariates are area-constant.
For the sparse PSU stress, supplied D is the ultimate-cluster approximation,
which omits within-PSU second-stage variance. It is not an oracle or full-design
variance. Invalid/near-zero D and one-PSU domains train neither the variance
summary nor GVF and receive the jointly fitted GVF instead.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib
import time
import numpy as np
import pandas as pd
import hogg_tsln as h
import hogg_diagnostics as hd


def compile_extension():
    source=h.HERE/'hogg_stage2_design_extension.stan'
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    stamp=source.with_name('hogg_stage2_design_extension_compiled_source.json')
    old=__import__('json').loads(stamp.read_text()) if stamp.exists() else {}
    model=h._backend().CmdStanModel(stan_file=str(source),force_compile=old.get('stan_sha256')!=before)
    after=hashlib.sha256(source.read_bytes()).hexdigest()
    if before!=after:raise RuntimeError('Extension source changed during compilation')
    h.write_json(stamp,{'stan_sha256':after,'executable_sha256':hashlib.sha256(Path(model.exe_file).read_bytes()).hexdigest()})
    return model


def replace_sampling_variance(prepared,s1,design_variance,n_psu):
    D=np.asarray(design_variance,float);n_psu=np.asarray(n_psu,int)
    if D.shape!=(len(prepared['n']),):raise ValueError('Design variance domain count mismatch')
    scale=1/np.maximum(prepared['n_eff_kish'],1)
    threshold=np.maximum(1e-15,scale*1e-12)
    positive=np.isfinite(D)&(D>threshold)
    reliable=positive&(n_psu>=2)
    direct_interior=(prepared['direct']>0)&(prepared['direct']<1)
    stable=reliable&direct_interior
    if stable.sum()<3:raise ValueError('Fewer than3 reliable stable design-variance domains; no silently prior-only GVF')
    mu=s1['mu_draws'];gamma=np.full_like(mu,np.nan);psi=np.full_like(mu,np.nan)
    psi[:,reliable]=2*D[reliable]
    gamma[:,reliable]=psi[:,reliable]/(mu[:,reliable]*(1-mu[:,reliable]))**2
    result={**s1,'psi_draws':psi,'gamma_draws':gamma,'stable':stable,
            'direct_variance':np.where(reliable,D,np.nan)}
    return result,{'design_variance_threshold':threshold,'design_variance_finite_positive':positive,
        'at_least_two_measured_psus':n_psu>=2,'reliable_design_variance':reliable,
        'stable_GVF_training':stable,'unreliable_or_boundary_GVF_imputed':~stable,
        'n_psu':n_psu,'unreliable_domains_count':int((~reliable).sum()),
        'one_psu_domains_count':int((n_psu==1).sum()),
        'zero_or_invalid_variance_count':int((~positive).sum())}


def fit_design_endpoint(units,areas,direct_data,*,endpoint_index,outcome,weight,
                        area_features=('X1','X2'),area_col='area',psu_col='psu',
                        population_N='N_population',config=None,output_dir=None,
                        primary_models=None,extension_model=None):
    cfg=h.MCMCConfig(parallel_chains=1,phase='experimental_design_variance') if config is None else config
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    prepared=h.prepare_hogg_data(units,areas,outcome=outcome,weight=weight,
        area_col=area_col,area_features=area_features,population_N=population_N)
    data=np.load(direct_data,allow_pickle=False) if isinstance(direct_data,(str,Path)) else direct_data
    lookup={int(idx):j for j,idx in enumerate(data['sample_idx'])}
    rows=[lookup[int(idx)] for idx in prepared['sampled_idx']]
    D=np.asarray(data['D'])[rows,endpoint_index,endpoint_index]
    # Verify the survey directs against the same measured records, preventing
    # accidental variance/outcome or domain-order mismatch.
    supplied=np.asarray(data['y'])[rows,endpoint_index]
    if not np.allclose(supplied,prepared['direct'],atol=1e-10,rtol=1e-10):
        raise ValueError('Supplied design D does not match endpoint/direct records')
    measured=prepared['units']
    n_psu=measured.groupby('_area_index')[psu_col].nunique().reindex(prepared['sampled_idx']).to_numpy(int)
    models=h.compile_models() if primary_models is None else primary_models
    ext=compile_extension() if extension_model is None else extension_model
    start=time.monotonic()
    f1=h._sample(models['hogg_stage1'],prepared['s1_data'],cfg,out/'stage1')
    sm1=f1.summary();diag1=h.diagnose_fit(f1,cfg,summary=sm1)
    sm1.to_csv(out/'stage1_parameter_summary.csv')
    initial=h.aggregate_stage1(prepared,f1.stan_variable('intercept'),
        f1.stan_variable('beta'),f1.stan_variable('effect_area'))
    s1,mask=replace_sampling_variance(prepared,initial,D,n_psu)
    d2,meta=h.prepare_stage2(prepared,s1,profile='author_code',seed=cfg.seed+21)
    d2['Q_gvf']=3
    d2['L']=np.column_stack([np.ones(len(n_psu)),np.log(prepared['n']),np.log(n_psu)])
    f2=h._sample(ext,d2,cfg,out/'stage2',seed_offset=1)
    sm2=f2.summary();diag2=h.diagnose_fit(f2,cfg,summary=sm2)
    sm2.to_csv(out/'stage2_parameter_summary.csv')
    draws=f2.stan_variable('prevalence')
    if not np.isfinite(draws).all() or ((draws<=0)|(draws>=1)).any():
        raise FloatingPointError('Invalid extension posterior prevalence')
    status={'method':'Hogg_experimental_ultimate_cluster_variance','completed':True,
        'success':diag1['success'] and diag2['success'],'phase':cfg.phase,
        'stage1':diag1,'stage2':diag2,'config':asdict(cfg),'seconds_total':time.monotonic()-start,
        'priors':'Same genericS1 andS2 priors; GVFNormal(0,2) coefficients now3dimensions',
        'sampling_variance':'psiS1=2×supplied ultimate-cluster D in probability scale, exact constant-within-area residual identity; D excludes within-PSU variance in this stress and is not full-design/oracle.',
        'GVF':'Joint logSD regression on intercept,log(number of measured children),log(number of measured PSUs); only direct-interior domains with>=2measuredPSUs and finite positiveD trainGVF.',
        'unreliable_mask_summary':{k:v for k,v in mask.items() if np.ndim(v)==0},
        'primary_unchanged':True,'input_metadata':prepared['metadata'],
        'input_sha256':{'units':h._input_hash(units),'areas':h._input_hash(areas),
            'direct':hashlib.sha256(Path(direct_data).read_bytes()).hexdigest() if isinstance(direct_data,(str,Path)) else None},
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),h.HERE/'hogg_stage2_design_extension.stan']}}
    result=h.HoggResult(prepared['all_codes'],draws,s1,d2,status,out)
    result.predictions().to_csv(out/'area_predictions.csv',index=False)
    np.savez_compressed(out/'posterior_area_draws.npz',area=result.areas,prevalence=draws)
    rows=pd.DataFrame({'area':prepared['sampled_codes'],'direct':prepared['direct'],
        'n_measured':prepared['n'],'n_psu':n_psu,'D_ultimate_cluster':D,
        **{k:v for k,v in mask.items() if np.ndim(v)>0}})
    rows.to_csv(out/'variance_reliability_mask.csv',index=False)
    h.write_json(out/'smoothing_diagnostics_ALC.json',hd.smoothing_diagnostics(prepared,s1))
    h.write_json(out/'status.json',status)
    return result


def fit_design_pair(units,areas,direct_data,*,config=None,output_dir=None,
                    area_features=('X1','X2'),population_N='N_population',
                    primary_models=None,extension_model=None,
                    retry_on_diagnostics=True,retain_sampler_csv=False):
    cfg=h.MCMCConfig(parallel_chains=1,phase='experimental_design_variance') if config is None else config
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    models=h.compile_models() if primary_models is None else primary_models
    ext=compile_extension() if extension_model is None else extension_model
    start=time.monotonic();results={};attempts={}
    for j,label in enumerate(['anemia','stunting']):
        ec=h.MCMCConfig(**{**asdict(cfg),'seed':cfg.seed+100*j})
        r=fit_design_endpoint(units,areas,direct_data,endpoint_index=j,
            outcome='y_'+label,weight='weight_'+label,area_features=area_features,
            population_N=population_N,config=ec,output_dir=output/label/'attempt_0',
            primary_models=models,extension_model=ext)
        attempts[label]=[r.status]
        if retry_on_diagnostics and not r.status['success']:
            ec=h.MCMCConfig(**{**asdict(ec),'warmup':1000,'samples':2000,'adapt_delta':.99})
            r=fit_design_endpoint(units,areas,direct_data,endpoint_index=j,
                outcome='y_'+label,weight='weight_'+label,area_features=area_features,
                population_N=population_N,config=ec,output_dir=output/label/'attempt_1',
                primary_models=models,extension_model=ext)
            attempts[label].append(r.status)
        results[label]=r
    prediction=pd.DataFrame({'area':results['anemia'].areas})
    for label,r in results.items():
        p=r.predictions();prediction['p_mean_'+label]=p['mean']
        prediction['lo95_'+label]=p['lo95_ET'];prediction['hi95_'+label]=p['hi95_ET']
    prediction.to_csv(output/'predictions.csv',index=False)
    np.savez_compressed(output/'posterior_draws.npz',area=results['anemia'].areas,
        anemia=results['anemia'].draws,stunting=results['stunting'].draws)
    status={'method':'Hogg_experimental_ultimate_cluster_variance','completed':True,
        'success':all(r.status['success'] for r in results.values()),
        'seconds_total':time.monotonic()-start,'outcomes':{k:r.status for k,r in results.items()},
        'all_attempts':attempts,'extension_label':'Experimental design proxy, not original generic Hogg2024; not a full multistage variance estimator.',
        'independence':'Two separately fitted endpoints; no paired-rule change posterior'}
    if not retain_sampler_csv:
        deleted={}
        for label in results:
          for attempt in range(len(attempts[label])):
            for stage in ['stage1','stage2']:
              for p in (output/label/f'attempt_{attempt}'/stage).glob('*.csv'):
                deleted[str(p.relative_to(output))]=hashlib.sha256(p.read_bytes()).hexdigest();p.unlink()
        status['removed_redundant_sampler_csv_sha256']=deleted
    h.write_json(output/'status.json',status)
    return {'predictions':prediction,'results':results,'status':status,
            'marginal_draws':{k:r.draws for k,r in results.items()}}
