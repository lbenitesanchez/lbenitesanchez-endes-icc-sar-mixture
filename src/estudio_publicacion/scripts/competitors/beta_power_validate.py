"""Check analytic gradients and hyperposterior Laplace against Stan NUTS.

Uses only explicitly labelled development replications. Stan samples exactly
the same collapsed power posterior; numerical convergence is checked separately
from the scientific interval coverage benchmark.
"""
import argparse, json, os
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit
from scipy.stats import beta as beta_distribution
import cmdstanpy
from beta_power import EndpointPosterior, working_icc, cluster_effective_n, CONFIG

ROOT=Path(__file__).resolve().parents[2]


def main():
    arg=argparse.ArgumentParser();arg.add_argument('--scenario',required=True);arg.add_argument('--rep',type=int,default=0)
    arg.add_argument('--warmup',type=int,default=1000);arg.add_argument('--samples',type=int,default=2000)
    args=arg.parse_args()
    cmdstanpy.set_cmdstan_path(str(ROOT/'.competitor_env/cmdstan/cmdstan-2.40.0'))
    model=cmdstanpy.CmdStanModel(stan_file=str(Path(__file__).with_name('beta_power_reference.stan')))
    src=ROOT/f'resultados/resultados_comparadores/inputs/{args.scenario}/rep_{args.rep:05d}/model_inputs'
    dest=ROOT/f'resultados/resultados_comparadores/validation/beta_power/{args.scenario}/rep_{args.rep:05d}'
    dest.mkdir(parents=True,exist_ok=True)
    d=np.load(src/'direct.npz');area=pd.read_csv(src/'area.csv');X=np.c_[np.ones(len(area)),area[['X1','X2']]]
    B=np.load(src/'spatial_basis.npy');units=pd.read_csv(src/'sample.csv')
    summaries=[]
    for j,name in enumerate(['anemia','stunting']):
        icc,info=working_icc(units,name);n,_=cluster_effective_n(units,name,d['sample_idx'],d['n_eff'][:,j],icc)
        post=EndpointPosterior(X,d['y'][:,j],n,d['sample_idx'],B).optimize()
        lap=post.draws(27182+j)
        gradient=[]
        for h in [post.prior_mean,post.mode,post.mode+.2]:
            numeric=approx_derivative(lambda z:post.objective(z)[0],h,method='3-point').ravel()
            gradient.append(float(np.max(abs(numeric-post.objective(h)[1]))))
        data=dict(A=len(X),P=X.shape[1],Q=B.shape[1],X=X,B=B,S=len(post.idx),idx=post.idx+1,
                  successes=post.k,failures=post.n-post.k,
                  **{k:CONFIG[k] for k in ['beta_sd','intercept_sd','spatial_sd','log_kappa_mean','log_kappa_sd']})
        fit=model.sample(data=data,seed=20261005+j,chains=4,parallel_chains=4,iter_warmup=args.warmup,
                         iter_sampling=args.samples,adapt_delta=.95,max_treedepth=12,
                         inits=dict(beta=post.mode[:X.shape[1]],gamma=post.mode[X.shape[1]:-1],log_kappa=post.mode[-1]),
                         output_dir=str(dest/name),show_progress=False)
        eta=fit.stan_variable('eta');kap=np.exp(fit.stan_variable('log_kappa'))[:,None]
        a=kap*expit(eta);b=kap*expit(-eta)
        a[:,post.idx]+=post.k;b[:,post.idx]+=post.n-post.k
        reference_mean=np.mean(a/(a+b),axis=0)
        rng=np.random.default_rng(731+j)
        reference_draws=beta_distribution.ppf(rng.uniform(1e-12,1-1e-12,size=a.shape),a,b)
        reference_lo=np.quantile(reference_draws,.025,axis=0);reference_hi=np.quantile(reference_draws,.975,axis=0)
        summary=fit.summary()
        divergences=int(fit.method_variables()['divergent__'].sum())
        treehits=int((fit.method_variables()['treedepth__']>=12).sum())
        all_hyper=summary.loc[~summary.index.str.startswith('eta[')]
        report=dict(scenario=args.scenario,rep=args.rep,outcome=name,
            maximum_gradient_absolute_error=max(gradient),
            maximum_hyper_Rhat=float(all_hyper.R_hat.max()),minimum_hyper_bulk_ESS=float(all_hyper.ESS_bulk.min()),
            divergences=divergences,max_treedepth_hits=treehits,
            mean_absolute_posterior_mean_difference_pp=float(np.mean(abs(reference_mean-lap['p_mean']))*100),
            maximum_posterior_mean_difference_pp=float(np.max(abs(reference_mean-lap['p_mean']))*100),
            mean_absolute_lower95_difference_pp=float(np.mean(abs(reference_lo-lap['lower95']))*100),
            mean_absolute_upper95_difference_pp=float(np.mean(abs(reference_hi-lap['upper95']))*100),
            laplace_status=lap['status'],ICC=info,
            validation_scope='development only; NUTS checks numerical approximation, not survey calibration')
        summaries.append(report)
        pd.DataFrame(dict(area=area.area,laplace_mean=lap['p_mean'],reference_mean=reference_mean,
                          laplace_lo=lap['lower95'],reference_lo=reference_lo,
                          laplace_hi=lap['upper95'],reference_hi=reference_hi)).to_csv(dest/f'{name}_comparison.csv',index=False)
        (dest/'validation.json').write_text(json.dumps(summaries,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k not in ['laplace_status','ICC']}),flush=True)


if __name__=='__main__':main()
