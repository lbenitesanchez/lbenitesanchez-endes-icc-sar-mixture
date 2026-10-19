"""NUTS evaluation of the defined beta-power posterior.

The survey-likelihood and working ICC remain approximations to survey
uncertainty. NUTS removes the Laplace integration approximation only. Each
endpoint is fitted independently; no within-child joint prevalence is inferred.
"""
import os,json,tempfile,time,hashlib
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import beta as beta_distribution
from beta_power import CONFIG,EndpointPosterior,working_icc,cluster_effective_n

HMC_CONFIG=dict(chains=4,warmup=1000,samples=1000,adapt_delta=.99,max_treedepth=12,
                min_bulk_ESS=400,min_tail_ESS=400,min_E_BFMI=.3,max_Rhat=1.01,retry_warmup=2000,retry_samples=2000,
                retry_adapt_delta=.995,parallel_chains=1,metric='dense_e',
                initial_metric='mode covariance if unrepaired positive Hessian, otherwise identity')
_MODEL=None


def get_model():
    global _MODEL
    if _MODEL is None:
        import cmdstanpy
        project=Path(os.environ.get('ENDES_STUDY_ROOT',Path(__file__).resolve().parents[2]))
        local=project/'.competitor_env/cmdstan/cmdstan-2.40.0'
        if local.exists():cmdstanpy.set_cmdstan_path(str(local))
        sdk=Path('/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk')
        if sdk.exists():os.environ.setdefault('SDKROOT',str(sdk))
        _MODEL=cmdstanpy.CmdStanModel(stan_file=str(Path(__file__).with_name('beta_power_hybrid.stan')))
    return _MODEL


def fit(X,y,n_eff,sample_idx,units=None,basis=None,seed=20261005,config=None,
        sampler_config=None,output_dir=None):
    config={**CONFIG,**(config or {}),"implementation":"collapsed beta-power hyperposterior NUTS v1"}
    hconfig={**HMC_CONFIG,**(sampler_config or {})}
    y=np.asarray(y,float);ne=np.asarray(n_eff,float)
    if y.ndim==1:y=y[:,None];ne=ne[:,None]
    model=get_model();outputs=[];statuses=[];effective=[]
    with tempfile.TemporaryDirectory(prefix='beta_power_hmc_') as temporary:
        destination=Path(output_dir) if output_dir is not None else Path(temporary)
        destination.mkdir(parents=True,exist_ok=True)
        for j,name in enumerate(('anemia','stunting')[:y.shape[1]]):
            icc,info=working_icc(units,name) if config['cluster_adjustment'] else (0.,{'available':False,'reason':'disabled'})
            n,_=cluster_effective_n(units,name,sample_idx,ne[:,j],icc)
            post=EndpointPosterior(X,y[:,j],n,sample_idx,basis,config).optimize()
            data=dict(A=len(X),P=post.X.shape[1],Q=post.B.shape[1],X=post.X,B=post.B,S=len(post.idx),
                      idx=post.idx+1,successes=post.k,failures=post.n-post.k,
                      **{k:config[k] for k in ['beta_sd','intercept_sd','spatial_sd','log_kappa_mean','log_kappa_sd']})
            trials=[];tick=time.perf_counter()
            for attempt in range(2):
                run=destination/name/f'attempt_{attempt+1}';run.mkdir(parents=True,exist_ok=True)
                sampled=model.sample(data=data,seed=seed+j*1009+attempt*104729,
                    chains=hconfig['chains'],parallel_chains=hconfig['parallel_chains'],
                    iter_warmup=hconfig['warmup' if attempt==0 else 'retry_warmup'],
                    iter_sampling=hconfig['samples' if attempt==0 else 'retry_samples'],
                    adapt_delta=hconfig['adapt_delta' if attempt==0 else 'retry_adapt_delta'],
                    max_treedepth=hconfig['max_treedepth'],
                    metric=hconfig['metric'],
                    inv_metric=post.cov if post.status['success'] and post.status['hessian_repairs']==0 else np.eye(len(post.mode)),
                    inits=dict(beta=post.mode[:post.X.shape[1]],gamma=post.mode[post.X.shape[1]:-1],log_kappa=post.mode[-1]),
                    output_dir=str(run),show_progress=False)
                summary=sampled.summary();variables=sampled.method_variables()
                # No warmup draws are saved in production. Slice explicitly
                # nevertheless: CmdStanPy method_variables includes warmup if
                # it was saved, whereas stan_variable normally excludes it.
                variables={k:v[-sampled.num_draws_sampling:] for k,v in variables.items()}
                energy=variables['energy__']
                bfmi=np.mean(np.diff(energy,axis=0)**2,axis=0)/np.var(energy,axis=0)
                finite_diagnostics=bool(np.all(np.isfinite(summary[['R_hat','ESS_bulk','ESS_tail']].to_numpy())) and np.all(np.isfinite(bfmi)))
                diagnosis=dict(max_Rhat=float(summary.R_hat.max()),min_bulk_ESS=float(summary.ESS_bulk.min()),
                    min_tail_ESS=float(summary.ESS_tail.min()),E_BFMI=bfmi.tolist(),finite_diagnostics=finite_diagnostics,
                    divergences=int(variables['divergent__'].sum()),
                    max_treedepth_hits=int((variables['treedepth__']>=hconfig['max_treedepth']).sum()),attempt=attempt+1)
                diagnosis['success']=bool(finite_diagnostics and diagnosis['max_Rhat']<=hconfig['max_Rhat'] and
                    diagnosis['min_bulk_ESS']>=hconfig['min_bulk_ESS'] and diagnosis['min_tail_ESS']>=hconfig['min_tail_ESS'] and
                    np.min(bfmi)>=hconfig['min_E_BFMI'] and diagnosis['divergences']==0 and diagnosis['max_treedepth_hits']==0)
                trials.append(diagnosis)
                if diagnosis['success']:break
            hbeta=sampled.stan_variable('beta')
            hgamma=sampled.stan_variable('gamma') if post.B.shape[1]>0 else np.zeros((len(hbeta),0))
            eta=hbeta@post.X.T+hgamma@post.B.T
            lk=sampled.stan_variable('log_kappa');kap=np.exp(lk)[:,None]
            a=kap*expit(eta);b=kap*expit(-eta)
            guard=int(np.sum((a<1e-10)|(b<1e-10)))
            a=np.maximum(a,1e-10);b=np.maximum(b,1e-10)
            a[:,post.idx]+=post.k;b[:,post.idx]+=post.n-post.k
            conditional=a/(a+b)
            rng=np.random.default_rng(seed+j*2003+92717)
            pdraws=beta_distribution.ppf(rng.uniform(1e-12,1-1e-12,a.shape),a,b)
            sd=dict(success=diagnosis['success'],sampler_trials=trials,sampler_config=hconfig,
                numerical_initialization=post.status,config=config,ICC_working_estimate=info,
                posterior_computation='NUTS of collapsed hyperposterior; exact conditional beta draws',
                working_ICC_uncertainty_propagated=False,beta_shape_guard_cells=guard,
                seconds=time.perf_counter()-tick,
                mean_kappa=float(np.mean(np.exp(lk))),median_kappa=float(np.median(np.exp(lk))))
            sd['posterior_draws']=len(lk)
            outputs.append(dict(p_mean=conditional.mean(axis=0),lower95=np.quantile(pdraws,.025,axis=0),
                upper95=np.quantile(pdraws,.975,axis=0),p_variance=np.mean(a*b/((a+b)**2*(a+b+1)),axis=0)+np.var(conditional,axis=0),p_draws=pdraws))
            statuses.append(sd);effective.append(n)
    count=min(len(v['p_draws']) for v in outputs)
    joint=[]
    for j,o in enumerate(outputs):
        # Marginal draws are fitted independently. A reproducible permutation
        # samples all chains when endpoint retry sizes differ. Marginal
        # summaries above retain every draw; this array is only a convenience.
        rng=np.random.default_rng(seed+50021+j*3011)
        choose=rng.permutation(len(o['p_draws']))[:count]
        joint.append(o['p_draws'][choose])
    return dict(p_mean=np.column_stack([o['p_mean'] for o in outputs]),
                lower95=np.column_stack([o['lower95'] for o in outputs]),
                upper95=np.column_stack([o['upper95'] for o in outputs]),
                p_variance=np.column_stack([o['p_variance'] for o in outputs]),
                p_draws=np.stack(joint,axis=-1),marginal_draws=[o['p_draws'] for o in outputs],
                effective_n=np.column_stack(effective),status=statuses)
